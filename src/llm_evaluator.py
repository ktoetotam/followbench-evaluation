"""
LLM-based evaluation of model responses against constraints
Uses GPT-4 to evaluate whether responses satisfy constraints
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .config import get_api_key, RESULTS_DIR
from .model_inference import ModelResponse


@dataclass
class ConstraintEvaluation:
    """Evaluation result for a single constraint"""
    constraint_type: str
    constraint_description: str
    satisfied: bool
    confidence: float  # 0.0 to 1.0
    explanation: str


@dataclass
class TestCaseEvaluation:
    """Evaluation result for a complete test case"""
    test_case_id: str
    model_name: str
    level: int
    constraint_evaluations: List[ConstraintEvaluation]
    hsr: float  # Hard Satisfaction Rate (all constraints satisfied)
    ssr: float  # Soft Satisfaction Rate (average satisfaction)


class LLMEvaluator:
    """Evaluate model responses using GPT-4"""
    
    @staticmethod
    def load_followbench_dataset(split: str = "train") -> List[Dict]:
        """Load FollowBench dataset from HuggingFace"""
        if not load_dataset:
            raise ImportError("datasets package required. Run: pip install datasets")
        
        print(f"Loading FollowBench dataset from HuggingFace...")
        dataset = load_dataset("YuxinJiang/FollowBench", split=split)
        
        # Convert to list of dicts
        test_cases = []
        for item in dataset:
            test_case = {
                "id": f"followbench_{item['id']}_{item['level']}",
                "level": item["level"],
                "first_prompt": item["first_prompt"],
                "prompt": item["prompt"],
                "constraint": item["constraint"],
                "answer": item.get("answer", "")
            }
            test_cases.append(test_case)
        
        print(f"Loaded {len(test_cases)} test cases from FollowBench")
        return test_cases
    
    EVALUATION_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI assistant's response satisfies specific constraints.

**Task**: Evaluate if the response satisfies each constraint listed below.

**Base Instruction**:
{base_instruction}

**Full Instruction with Constraints**:
{full_instruction}

**Assistant's Response**:
{response}

**Constraints to Evaluate**:
{constraints_list}

For each constraint, evaluate:
1. Whether the response satisfies the constraint (Yes/No)
2. Confidence level (0.0 to 1.0)
3. Brief explanation

Respond in JSON format:
{{
  "evaluations": [
    {{
      "constraint_id": 1,
      "constraint_type": "content",
      "satisfied": true,
      "confidence": 0.95,
      "explanation": "The response includes all required technical terms."
    }},
    ...
  ]
}}

Be strict and precise in your evaluation. Only mark a constraint as satisfied if it is clearly and fully met."""
    
    def __init__(self, evaluator_model: str = "gpt-4", use_azure: bool = False):
        self.evaluator_model = evaluator_model
        self.use_azure = use_azure
        
        if not OpenAI:
            raise ImportError("OpenAI package required. Run: pip install openai")
        
        # Initialize Azure OpenAI client if requested
        if use_azure or evaluator_model.startswith("azure-"):
            if not AzureOpenAI:
                raise ImportError("OpenAI package with Azure support required. Run: pip install openai")
            
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            
            if not api_key or not endpoint:
                raise ValueError(
                    "Azure OpenAI requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables. "
                    "Optionally set AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION."
                )
            
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            # Use deployment name if provided, otherwise use model name
            self.deployment_name = deployment if deployment else evaluator_model.replace("azure-", "")
            self.use_azure = True
        else:
            # Regular OpenAI client
            self.api_key = get_api_key(evaluator_model)
            self.client = OpenAI(api_key=self.api_key)
            self.deployment_name = None
    
    def evaluate_followbench_response(
        self,
        test_case: Dict,
        response: str
    ) -> TestCaseEvaluation:
        """Evaluate a model response against FollowBench test case"""
        
        # Extract constraint description from test case
        constraint_desc = test_case.get("constraint", "")
        level = test_case["level"]
        
        # For level 0, there are no constraints
        if level == 0 or not constraint_desc:
            return TestCaseEvaluation(
                test_case_id=test_case["id"],
                model_name="evaluated_model",
                level=level,
                constraint_evaluations=[],
                hsr=1.0,
                ssr=1.0
            )
        
        # Build evaluation prompt
        prompt = f"""You are an expert evaluator assessing whether an AI assistant's response satisfies specific constraints.

**Base Instruction** (Level 0):
{test_case['first_prompt']}

**Full Instruction with Constraints** (Level {level}):
{test_case['prompt']}

**Constraint Description**:
{constraint_desc}

**Assistant's Response**:
{response}

**Task**: Evaluate if the response satisfies the constraint(s) described above.

For each constraint implied in the description, evaluate:
1. Whether the response satisfies the constraint (Yes/No)
2. Confidence level (0.0 to 1.0)
3. Brief explanation

Respond in JSON format:
{{{{
  "evaluations": [
    {{{{
      "constraint_id": 1,
      "constraint_type": "content",
      "satisfied": true,
      "confidence": 0.95,
      "explanation": "The response meets the requirement."
    }}}},
    ...
  ]
}}}}

Be strict and precise in your evaluation. Only mark a constraint as satisfied if it is clearly and fully met."""
        
        # Call GPT-4 for evaluation
        try:
            model_param = self.deployment_name if self.use_azure else "gpt-4"
            response_obj = self.client.chat.completions.create(
                model=model_param,
                messages=[
                    {"role": "system", "content": "You are an expert constraint evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Parse evaluation results
            eval_result = json.loads(response_obj.choices[0].message.content)
            
            # Create ConstraintEvaluation objects
            constraint_evaluations = []
            for eval_item in eval_result["evaluations"]:
                constraint_eval = ConstraintEvaluation(
                    constraint_type=eval_item.get("constraint_type", "content"),
                    constraint_description=constraint_desc,
                    satisfied=eval_item["satisfied"],
                    confidence=eval_item["confidence"],
                    explanation=eval_item["explanation"]
                )
                constraint_evaluations.append(constraint_eval)
            
            # Calculate metrics
            satisfied_count = sum(1 for ce in constraint_evaluations if ce.satisfied)
            total_constraints = len(constraint_evaluations)
            
            hsr = 1.0 if satisfied_count == total_constraints else 0.0
            ssr = satisfied_count / total_constraints if total_constraints > 0 else 0.0
            
            return TestCaseEvaluation(
                test_case_id=test_case["id"],
                model_name="evaluated_model",
                level=level,
                constraint_evaluations=constraint_evaluations,
                hsr=hsr,
                ssr=ssr
            )
        
        except Exception as e:
            print(f"Error evaluating {test_case['id']}: {e}")
            
            # Return empty evaluation on error
            return TestCaseEvaluation(
                test_case_id=test_case["id"],
                model_name="evaluated_model",
                level=level,
                constraint_evaluations=[],
                hsr=0.0,
                ssr=0.0
            )
    
    def evaluate_response(
        self,
        test_case_data: Dict,
        model_response: ModelResponse
    ) -> TestCaseEvaluation:
        """Evaluate a model response against test case constraints"""
        
        # Build constraints list for prompt
        constraints_list = ""
        for i, constraint in enumerate(test_case_data["constraints"], 1):
            constraints_list += f"{i}. [{constraint['type']}] {constraint['description']}\n"
        
        # Create evaluation prompt
        prompt = self.EVALUATION_PROMPT_TEMPLATE.format(
            base_instruction=test_case_data["base_instruction"],
            full_instruction=test_case_data["full_instruction"],
            response=model_response.response,
            constraints_list=constraints_list
        )
        
        # Call GPT-4 for evaluation
        try:
            model_param = self.deployment_name if self.use_azure else "gpt-4"
            response = self.client.chat.completions.create(
                model=model_param,
                messages=[
                    {"role": "system", "content": "You are an expert constraint evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Parse evaluation results
            eval_result = json.loads(response.choices[0].message.content)
            
            # Create ConstraintEvaluation objects
            constraint_evaluations = []
            for eval_item in eval_result["evaluations"]:
                constraint = test_case_data["constraints"][eval_item["constraint_id"] - 1]
                
                constraint_eval = ConstraintEvaluation(
                    constraint_type=constraint["type"],
                    constraint_description=constraint["description"],
                    satisfied=eval_item["satisfied"],
                    confidence=eval_item["confidence"],
                    explanation=eval_item["explanation"]
                )
                constraint_evaluations.append(constraint_eval)
            
            # Calculate metrics
            satisfied_count = sum(1 for ce in constraint_evaluations if ce.satisfied)
            total_constraints = len(constraint_evaluations)
            
            hsr = 1.0 if satisfied_count == total_constraints else 0.0
            ssr = satisfied_count / total_constraints if total_constraints > 0 else 0.0
            
            return TestCaseEvaluation(
                test_case_id=test_case_data["id"],
                model_name=model_response.model_name,
                level=test_case_data["level"],
                constraint_evaluations=constraint_evaluations,
                hsr=hsr,
                ssr=ssr
            )
        
        except Exception as e:
            print(f"Error evaluating {test_case_data['id']}: {e}")
            
            # Return empty evaluation on error
            return TestCaseEvaluation(
                test_case_id=test_case_data["id"],
                model_name=model_response.model_name,
                level=test_case_data["level"],
                constraint_evaluations=[],
                hsr=0.0,
                ssr=0.0
            )
    
    def evaluate_batch(
        self,
        test_cases_file: Path,
        responses_file: Path,
        output_file: Optional[Path] = None
    ) -> List[TestCaseEvaluation]:
        """Evaluate a batch of model responses"""
        
        # Load test cases
        with open(test_cases_file, 'r', encoding='utf-8') as f:
            test_cases_data = json.load(f)
        
        # Create lookup dict
        test_cases_dict = {tc["id"]: tc for tc in test_cases_data}
        
        # Load responses
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)
        
        evaluations = []
        
        for i, response_data in enumerate(responses_data, 1):
            test_case_id = response_data["test_case_id"]
            print(f"Evaluating {i}/{len(responses_data)}: {test_case_id}")
            
            if test_case_id not in test_cases_dict:
                print(f"  Warning: Test case {test_case_id} not found")
                continue
            
            test_case = test_cases_dict[test_case_id]
            
            # Create ModelResponse object
            model_response = ModelResponse(
                test_case_id=response_data["test_case_id"],
                model_name=response_data["model_name"],
                prompt=response_data["prompt"],
                response=response_data["response"],
                latency=response_data["latency"],
                error=response_data.get("error")
            )
            
            # Skip if response has error
            if model_response.error:
                print(f"  Skipping due to error: {model_response.error}")
                continue
            
            # Evaluate
            evaluation = self.evaluate_response(test_case, model_response)
            evaluations.append(evaluation)
            
            print(f"  HSR: {evaluation.hsr:.2f}, SSR: {evaluation.ssr:.2f}")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save evaluations
        if output_file:
            self.save_evaluations(evaluations, output_file)
        
        return evaluations
    
    def save_evaluations(
        self,
        evaluations: List[TestCaseEvaluation],
        output_file: Path
    ):
        """Save evaluations to JSON file"""
        data = []
        for evaluation in evaluations:
            data.append({
                "test_case_id": evaluation.test_case_id,
                "model_name": evaluation.model_name,
                "level": evaluation.level,
                "hsr": evaluation.hsr,
                "ssr": evaluation.ssr,
                "constraint_evaluations": [
                    {
                        "type": ce.constraint_type,
                        "description": ce.constraint_description,
                        "satisfied": ce.satisfied,
                        "confidence": ce.confidence,
                        "explanation": ce.explanation
                    }
                    for ce in evaluation.constraint_evaluations
                ]
            })
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(evaluations)} evaluations to {output_file}")


def run_followbench_evaluation(
    responses_file: Path,
    output_file: Optional[Path] = None,
    split: str = "train",
    use_azure: bool = False
):
    """Run evaluation on FollowBench dataset"""
    print(f"\n{'='*60}")
    print(f"Running FollowBench Evaluation")
    print(f"Responses: {responses_file}")
    print(f"Using Azure OpenAI: {use_azure}")
    print(f"{'='*60}\n")
    
    # Load FollowBench dataset
    evaluator = LLMEvaluator(use_azure=use_azure)
    test_cases = evaluator.load_followbench_dataset(split=split)
    
    # Load responses
    with open(responses_file, 'r', encoding='utf-8') as f:
        responses_data = json.load(f)
    
    # Create response lookup
    responses_dict = {r["test_case_id"]: r["response"] for r in responses_data}
    
    evaluations = []
    
    for i, test_case in enumerate(test_cases, 1):
        test_case_id = test_case["id"]
        print(f"Evaluating {i}/{len(test_cases)}: {test_case_id}")
        
        if test_case_id not in responses_dict:
            print(f"  Warning: Response not found for {test_case_id}")
            continue
        
        response = responses_dict[test_case_id]
        
        # Evaluate
        evaluation = evaluator.evaluate_followbench_response(test_case, response)
        evaluations.append(evaluation)
        
        print(f"  HSR: {evaluation.hsr:.2f}, SSR: {evaluation.ssr:.2f}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save evaluations
    if output_file:
        evaluator.save_evaluations(evaluations, output_file)
    
    # Print summary
    if evaluations:
        avg_hsr = sum(e.hsr for e in evaluations) / len(evaluations)
        avg_ssr = sum(e.ssr for e in evaluations) / len(evaluations)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total evaluations: {len(evaluations)}")
        print(f"Average HSR: {avg_hsr:.4f}")
        print(f"Average SSR: {avg_ssr:.4f}")
        print(f"{'='*60}\n")
    
    return evaluations


def run_evaluation_pipeline(
    test_cases_file: Path,
    responses_file: Path,
    output_file: Optional[Path] = None
):
    """Run complete LLM evaluation pipeline"""
    print(f"\n{'='*60}")
    print(f"Running LLM Evaluation")
    print(f"Test cases: {test_cases_file}")
    print(f"Responses: {responses_file}")
    print(f"{'='*60}\n")
    
    # Set output file if not provided
    if not output_file:
        model_name = responses_file.stem.replace("responses_", "")
        output_file = RESULTS_DIR / f"evaluations_{model_name}.json"
    
    # Run evaluation
    evaluator = LLMEvaluator()
    evaluations = evaluator.evaluate_batch(
        test_cases_file,
        responses_file,
        output_file
    )
    
    # Print summary
    if evaluations:
        avg_hsr = sum(e.hsr for e in evaluations) / len(evaluations)
        avg_ssr = sum(e.ssr for e in evaluations) / len(evaluations)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total evaluations: {len(evaluations)}")
        print(f"Average HSR: {avg_hsr:.4f}")
        print(f"Average SSR: {avg_ssr:.4f}")
        print(f"{'='*60}\n")
    
    return evaluations


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.llm_evaluator <responses_file> [--followbench] [--azure]")
        print("  --followbench: Use FollowBench dataset from HuggingFace")
        print("  --azure: Use Azure OpenAI instead of OpenAI")
        sys.exit(1)
    
    responses_file = Path(sys.argv[1])
    use_followbench = "--followbench" in sys.argv
    use_azure = "--azure" in sys.argv
    
    if use_followbench:
        run_followbench_evaluation(responses_file, use_azure=use_azure)
    else:
        if len(sys.argv) < 3:
            print("Usage: python -m src.llm_evaluator <test_cases_file> <responses_file> [--azure]")
            sys.exit(1)
        test_cases_file = Path(sys.argv[1])
        responses_file = Path(sys.argv[2])
        run_evaluation_pipeline(test_cases_file, responses_file)
