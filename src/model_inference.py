"""
Model inference for FollowBench evaluation
Supports multiple models: GPT-4, DeepSeek, Qwen
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .config import MODEL_CONFIGS, get_api_key, RESULTS_DIR
from .followbench_generator import TestCase


@dataclass
class ModelResponse:
    """Model response to a test case"""
    test_case_id: str
    model_name: str
    prompt: str
    response: str
    latency: float  # seconds
    error: Optional[str] = None


class ModelInference:
    """Run model inference on FollowBench test cases"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name)
        
        if not self.config:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.api_key = get_api_key(model_name)
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize API client for the model"""
        if not OpenAI:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )
        
        base_url = self.config.get("base_url")
        
        if base_url:
            return OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            return OpenAI(api_key=self.api_key)
    
    def infer(self, test_case: TestCase) -> ModelResponse:
        """Run inference on a single test case"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[
                    {"role": "system", "content": test_case.base_instruction},
                    {"role": "user", "content": test_case.full_instruction}
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            
            latency = time.time() - start_time
            
            return ModelResponse(
                test_case_id=test_case.id,
                model_name=self.model_name,
                prompt=test_case.full_instruction,
                response=response.choices[0].message.content,
                latency=latency
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return ModelResponse(
                test_case_id=test_case.id,
                model_name=self.model_name,
                prompt=test_case.full_instruction,
                response="",
                latency=latency,
                error=str(e)
            )
    
    def infer_batch(
        self,
        test_cases: List[TestCase],
        save_path: Optional[Path] = None
    ) -> List[ModelResponse]:
        """Run inference on a batch of test cases"""
        responses = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Processing {i}/{len(test_cases)}: {test_case.id}")
            
            response = self.infer(test_case)
            responses.append(response)
            
            if response.error:
                print(f"  Error: {response.error}")
            else:
                print(f"  Success (latency: {response.latency:.2f}s)")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save responses if path provided
        if save_path:
            self.save_responses(responses, save_path)
        
        return responses
    
    def save_responses(self, responses: List[ModelResponse], output_file: Path):
        """Save model responses to JSON file"""
        data = []
        for response in responses:
            data.append({
                "test_case_id": response.test_case_id,
                "model_name": response.model_name,
                "prompt": response.prompt,
                "response": response.response,
                "latency": response.latency,
                "error": response.error
            })
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(responses)} responses to {output_file}")
    
    @staticmethod
    def load_responses(input_file: Path) -> List[ModelResponse]:
        """Load model responses from JSON file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        responses = []
        for item in data:
            responses.append(ModelResponse(
                test_case_id=item["test_case_id"],
                model_name=item["model_name"],
                prompt=item["prompt"],
                response=item["response"],
                latency=item["latency"],
                error=item.get("error")
            ))
        
        return responses


def run_inference_pipeline(
    model_name: str,
    test_cases_file: Path,
    output_file: Optional[Path] = None
):
    """Run complete inference pipeline for a model"""
    print(f"\n{'='*60}")
    print(f"Running inference: {model_name}")
    print(f"Test cases: {test_cases_file}")
    print(f"{'='*60}\n")
    
    # Load test cases
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases_data = json.load(f)
    
    # Convert to TestCase objects
    test_cases = []
    for tc_data in test_cases_data:
        test_case = TestCase(
            id=tc_data["id"],
            level=tc_data["level"],
            base_instruction=tc_data["base_instruction"],
            constraints=[],  # Will be loaded if needed
            full_instruction=tc_data["full_instruction"],
            source_prompt=tc_data["source_prompt"],
            transformation_stage=tc_data["transformation_stage"]
        )
        test_cases.append(test_case)
    
    # Set output file if not provided
    if not output_file:
        stage = test_cases[0].transformation_stage if test_cases else "unknown"
        output_file = RESULTS_DIR / f"responses_{model_name}_{stage}.json"
    
    # Run inference
    inference = ModelInference(model_name)
    responses = inference.infer_batch(test_cases, output_file)
    
    # Print summary
    successful = sum(1 for r in responses if not r.error)
    failed = len(responses) - successful
    avg_latency = sum(r.latency for r in responses) / len(responses) if responses else 0
    
    print(f"\n{'='*60}")
    print(f"Inference Summary")
    print(f"{'='*60}")
    print(f"Total: {len(responses)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"{'='*60}\n")
    
    return responses


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.model_inference <model_name> <test_cases_file>")
        print("Example: python -m src.model_inference gpt-4 followbench_data/test_cases_original.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    test_cases_file = Path(sys.argv[2])
    
    run_inference_pipeline(model_name, test_cases_file)
