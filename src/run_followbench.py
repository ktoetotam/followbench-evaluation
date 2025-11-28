#!/usr/bin/env python3
"""
Run FollowBench evaluation from HuggingFace dataset.

This script:
1. Downloads the FollowBench dataset from HuggingFace
2. Runs inference with your chosen model
3. Evaluates responses using gpt-5-nano as judge
4. Generates analysis reports

Usage:
    poetry run python -m src.run_followbench --model gpt-5-nano
    poetry run python -m src.run_followbench --model deepseek-v3 --azure
    poetry run python -m src.run_followbench --model gpt-5-nano --max-samples 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package required. Install with: poetry add datasets")
    sys.exit(1)

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    print("Error: 'openai' package required. Install with: pip install openai")
    sys.exit(1)

# Support both direct execution and module execution
try:
    from src.config import MODEL_CONFIGS, get_api_key, RESULTS_DIR
except ImportError:
    from config import MODEL_CONFIGS, get_api_key, RESULTS_DIR


def filter_test_cases(
    test_cases: List[Dict],
    levels: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Filter test cases by constraint level and/or category, then apply max_samples limit.
    
    Args:
        test_cases: List of test cases to filter
        levels: List of levels to include (e.g., [1, 2, 3]). None = all levels.
        categories: List of categories to include (e.g., ["content", "format"]). None = all categories.
        max_samples: Maximum number of samples to return after filtering. None = all.
    
    Returns:
        Filtered list of test cases
    """
    filtered = test_cases
    
    if levels:
        filtered = [tc for tc in filtered if tc["level"] in levels]
        print(f"Filtered to levels {levels}: {len(filtered)} test cases")
    
    if categories:
        # Case-insensitive matching
        categories_lower = [c.lower() for c in categories]
        filtered = [tc for tc in filtered if tc["category"].lower() in categories_lower]
        print(f"Filtered to categories {categories}: {len(filtered)} test cases")
    
    if max_samples and len(filtered) > max_samples:
        filtered = filtered[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    return filtered


def get_available_categories(test_cases: List[Dict]) -> List[str]:
    """Get list of unique categories in the dataset."""
    return sorted(set(tc["category"] for tc in test_cases if tc["category"]))


def get_available_levels(test_cases: List[Dict]) -> List[int]:
    """Get list of unique levels in the dataset."""
    return sorted(set(tc["level"] for tc in test_cases))


def load_followbench_dataset() -> List[Dict]:
    """Load FollowBench dataset from HuggingFace."""
    print("Loading FollowBench dataset from HuggingFace...")
    print("Dataset: YuxinJiang/FollowBench")
    
    dataset = load_dataset("YuxinJiang/FollowBench", split="train")
    
    test_cases = []
    for item in dataset:
        # Dataset fields: example_id, category, source, instruction, level, target
        test_case = {
            "id": f"followbench_{item['example_id']}_{item['level']}",
            "level": item["level"],
            "first_prompt": item["instruction"],  # Base instruction
            "prompt": item["instruction"],  # Full instruction with constraints
            "constraint": item.get("target", ""),  # Target/constraint description
            "category": item.get("category", ""),
            "source": item.get("source", ""),
        }
        test_cases.append(test_case)
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Print level distribution
    level_counts = {}
    for tc in test_cases:
        level = tc["level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("\nLevel distribution:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} samples")
    
    # Print category distribution
    category_counts = {}
    for tc in test_cases:
        cat = tc["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat}: {category_counts[cat]} samples")
    
    return test_cases


def run_inference(
    test_cases: List[Dict],
    model_name: str,
    output_file: Path
) -> List[Dict]:
    """Run model inference on test cases."""
    print(f"\n{'='*60}")
    print(f"Running inference with: {model_name}")
    print(f"{'='*60}\n")
    
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    api_key = get_api_key(model_name)
    
    # Initialize client
    base_url = config.get("base_url")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    responses = []
    
    for i, tc in enumerate(test_cases, 1):
        print(f"Processing {i}/{len(test_cases)}: {tc['id']}")
        
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "user", "content": tc["prompt"]}
                ]
            )
            
            latency = time.time() - start_time
            
            responses.append({
                "test_case_id": tc["id"],
                "model_name": model_name,
                "prompt": tc["prompt"],
                "response": response.choices[0].message.content,
                "latency": latency,
                "error": None
            })
            
            print(f"  ✓ Success ({latency:.2f}s)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            responses.append({
                "test_case_id": tc["id"],
                "model_name": model_name,
                "prompt": tc["prompt"],
                "response": "",
                "latency": 0,
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save responses
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    print(f"\nResponses saved to: {output_file}")
    
    successful = sum(1 for r in responses if not r["error"])
    print(f"Successful: {successful}/{len(responses)}")
    
    return responses


def run_evaluation(
    test_cases: List[Dict],
    responses: List[Dict],
    output_file: Path,
    use_azure: bool = False
) -> List[Dict]:
    """Evaluate responses using gpt-5-nano as judge."""
    print(f"\n{'='*60}")
    print(f"Running LLM evaluation (gpt-5-nano)")
    print(f"Using Azure: {use_azure}")
    print(f"{'='*60}\n")
    
    # Initialize evaluator client
    if use_azure:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not api_key or not endpoint:
            print("Error: Azure OpenAI requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
            sys.exit(1)
        
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        model_param = deployment
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable required for evaluation")
            sys.exit(1)
        
        client = OpenAI(api_key=api_key)
        model_param = "gpt-5-nano"
    
    # Create response lookup
    response_lookup = {r["test_case_id"]: r for r in responses}
    
    evaluations = []
    
    for i, tc in enumerate(test_cases, 1):
        test_case_id = tc["id"]
        print(f"Evaluating {i}/{len(test_cases)}: {test_case_id}")
        
        if test_case_id not in response_lookup:
            print(f"  ⚠ No response found, skipping")
            continue
        
        response_data = response_lookup[test_case_id]
        
        if response_data.get("error"):
            print(f"  ⚠ Response had error, skipping")
            continue
        
        model_response = response_data["response"]
        level = tc["level"]
        
        # Level 0 has no constraints
        if level == 0:
            evaluations.append({
                "test_case_id": test_case_id,
                "model_name": response_data["model_name"],
                "level": level,
                "hsr": 1.0,
                "ssr": 1.0,
                "constraint_evaluations": []
            })
            print(f"  ✓ Level 0 (no constraints)")
            continue
        
        # Build evaluation prompt
        eval_prompt = f"""You are an expert evaluator assessing whether an AI assistant's response satisfies specific constraints.

**Base Instruction** (Level 0):
{tc['first_prompt']}

**Full Instruction with Constraints** (Level {level}):
{tc['prompt']}

**Constraint Description**:
{tc['constraint']}

**Assistant's Response**:
{model_response}

**Task**: Evaluate if the response satisfies the constraint(s) described above.

For each constraint implied in the description, evaluate:
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
      "explanation": "The response meets the requirement."
    }}
  ]
}}

Be strict and precise in your evaluation. Only mark a constraint as satisfied if it is clearly and fully met."""
        
        try:
            eval_response = client.chat.completions.create(
                model=model_param,
                messages=[
                    {"role": "system", "content": "You are an expert constraint evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_completion_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Check if response content is valid
            response_content = eval_response.choices[0].message.content
            if not response_content:
                print(f"  ⚠ Empty response from evaluator")
                # Check if there was a refusal
                if hasattr(eval_response.choices[0].message, 'refusal') and eval_response.choices[0].message.refusal:
                    print(f"    Refusal: {eval_response.choices[0].message.refusal}")
                evaluations.append({
                    "test_case_id": test_case_id,
                    "model_name": response_data["model_name"],
                    "level": level,
                    "hsr": 0.0,
                    "ssr": 0.0,
                    "constraint_evaluations": [],
                    "error": "Empty response from evaluator"
                })
                continue
            
            eval_result = json.loads(response_content)
            
            constraint_evals = eval_result.get("evaluations", [])
            
            # Calculate metrics
            if constraint_evals:
                satisfied_count = sum(1 for ce in constraint_evals if ce.get("satisfied", False))
                total = len(constraint_evals)
                hsr = 1.0 if satisfied_count == total else 0.0
                ssr = satisfied_count / total
            else:
                hsr = 0.0
                ssr = 0.0
            
            evaluations.append({
                "test_case_id": test_case_id,
                "model_name": response_data["model_name"],
                "level": level,
                "hsr": hsr,
                "ssr": ssr,
                "constraint_evaluations": constraint_evals
            })
            
            print(f"  ✓ HSR: {hsr:.2f}, SSR: {ssr:.2f}")
            
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parsing error: {e}")
            # Try to show what we received
            if 'response_content' in locals() and response_content:
                print(f"    Raw response (first 200 chars): {response_content[:200]}")
            evaluations.append({
                "test_case_id": test_case_id,
                "model_name": response_data["model_name"],
                "level": level,
                "hsr": 0.0,
                "ssr": 0.0,
                "constraint_evaluations": [],
                "error": f"JSON parsing error: {e}"
            })
        except Exception as e:
            print(f"  ✗ Evaluation error: {type(e).__name__}: {e}")
            evaluations.append({
                "test_case_id": test_case_id,
                "model_name": response_data["model_name"],
                "level": level,
                "hsr": 0.0,
                "ssr": 0.0,
                "constraint_evaluations": [],
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save evaluations
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluations saved to: {output_file}")
    
    return evaluations


def calculate_metrics(evaluations: List[Dict]) -> Dict:
    """Calculate HSR, SSR, and CSL metrics."""
    if not evaluations:
        return {"HSR": 0.0, "SSR": 0.0, "CSL": 0, "total_evaluations": 0, "by_level": {}}
    
    # Overall metrics
    total_hsr = sum(e["hsr"] for e in evaluations)
    total_ssr = sum(e["ssr"] for e in evaluations)
    
    hsr = total_hsr / len(evaluations)
    ssr = total_ssr / len(evaluations)
    
    # Calculate CSL
    from collections import defaultdict
    level_satisfaction = defaultdict(list)
    for e in evaluations:
        level_satisfaction[e["level"]].append(e["hsr"])
    
    csl = 0
    for level in sorted(level_satisfaction.keys()):
        if not level_satisfaction[level]:
            continue
        avg_hsr_at_level = sum(level_satisfaction[level]) / len(level_satisfaction[level])
        if avg_hsr_at_level >= 0.5:
            csl = level
        else:
            break
    
    return {
        "HSR": hsr,
        "SSR": ssr,
        "CSL": csl,
        "total_evaluations": len(evaluations),
        "by_level": {
            level: {
                "HSR": sum(hsrs) / len(hsrs),
                "count": len(hsrs)
            }
            for level, hsrs in level_satisfaction.items()
        }
    }


def print_report(metrics: Dict, model_name: str):
    """Print evaluation report."""
    print(f"\n{'='*60}")
    print(f"FollowBench Evaluation Report: {model_name}")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  Hard Satisfaction Rate (HSR): {metrics['HSR']:.4f}")
    print(f"  Soft Satisfaction Rate (SSR): {metrics['SSR']:.4f}")
    print(f"  Consistent Satisfaction Levels (CSL): {metrics['CSL']}")
    print(f"  Total Evaluations: {metrics['total_evaluations']}")
    
    print(f"\nPerformance by Level:")
    for level in sorted(metrics['by_level'].keys()):
        level_data = metrics['by_level'][level]
        print(f"  Level {level}: HSR={level_data['HSR']:.4f} (n={level_data['count']})")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run FollowBench evaluation from HuggingFace dataset"
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-5-nano",
        help="Model to evaluate (default: gpt-5-nano)"
    )
    
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Use Azure OpenAI for gpt-5-nano evaluation"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference step (use existing responses file)"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step (use existing evaluations file)"
    )
    
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Filter by constraint levels (e.g., --levels 1 2 3)"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Filter by categories (e.g., --categories content format style)"
    )
    
    parser.add_argument(
        "--list-filters",
        action="store_true",
        help="List available levels and categories, then exit"
    )
    
    args = parser.parse_args()
    
    # Setup output paths
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load full dataset
    test_cases = load_followbench_dataset()
    
    # Handle --list-filters option
    if args.list_filters:
        print("\n" + "="*60)
        print("Available Filters")
        print("="*60)
        print(f"\nLevels: {get_available_levels(test_cases)}")
        print(f"Categories: {get_available_categories(test_cases)}")
        print("\nExample usage:")
        print("  --levels 1 2 3          # Only levels 1, 2, and 3")
        print("  --categories content    # Only 'content' category")
        print("  --levels 3 --categories format style  # Level 3, format or style")
        return
    
    # Step 2: Filter test cases by level and/or category, then apply max_samples
    test_cases = filter_test_cases(
        test_cases, 
        levels=args.levels, 
        categories=args.categories,
        max_samples=args.max_samples
    )
    if not test_cases:
        print("Error: No test cases match the specified filters")
        sys.exit(1)
    
    # Build output filename with filter info
    filter_suffix = ""
    if args.levels:
        filter_suffix += f"_L{''.join(map(str, args.levels))}"
    if args.categories:
        filter_suffix += f"_{'_'.join(args.categories)}"
    
    responses_file = RESULTS_DIR / f"followbench_responses_{args.model}{filter_suffix}.json"
    evaluations_file = RESULTS_DIR / f"followbench_evaluations_{args.model}{filter_suffix}.json"
    
    # Step 3: Run inference
    if args.skip_inference and responses_file.exists():
        print(f"\nLoading existing responses from: {responses_file}")
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    else:
        responses = run_inference(test_cases, args.model, responses_file)
    
    # Step 4: Run evaluation
    if args.skip_evaluation and evaluations_file.exists():
        print(f"\nLoading existing evaluations from: {evaluations_file}")
        with open(evaluations_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
    else:
        evaluations = run_evaluation(
            test_cases,
            responses,
            evaluations_file,
            use_azure=args.azure
        )
    
    # Step 5: Calculate and print metrics
    metrics = calculate_metrics(evaluations)
    print_report(metrics, args.model)
    
    print("✅ FollowBench evaluation complete!")
    print(f"   Responses: {responses_file}")
    print(f"   Evaluations: {evaluations_file}")


if __name__ == "__main__":
    main()
