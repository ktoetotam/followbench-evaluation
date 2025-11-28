#!/usr/bin/env python3
"""
Evaluate transformed FollowBench prompts.

This script runs the FollowBench evaluation pipeline on prompts that have been
transformed using different optimization strategies.

It compares model performance across:
- Original prompts (baseline)
- Conflict-resolved prompts
- Lost-in-middle optimized prompts
- DeepSeek-optimized prompts
- Qwen-optimized prompts

Usage:
    poetry run python -m src.evaluate_transformed --model gpt-5-nano --transformation deepseek-optimized
    poetry run python -m src.evaluate_transformed --model gpt-5-nano --all
    poetry run python -m src.evaluate_transformed --compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package required")
    sys.exit(1)

try:
    from src.config import MODEL_CONFIGS, get_api_key, RESULTS_DIR
    from src.run_followbench import calculate_metrics, print_report
except ImportError:
    from config import MODEL_CONFIGS, get_api_key, RESULTS_DIR
    from run_followbench import calculate_metrics, print_report


TRANSFORMATIONS = [
    "original",
    "conflict_resolved",
    "lost_in_middle",
    "deepseek_optimized",
    "qwen_optimized"
]


def load_transformed_prompts(transform_dir: Path, transformation: str) -> List[Dict]:
    """Load transformed prompts from a transformation folder."""
    if transformation == "original":
        # For original, we need to load from the full transformed_prompts.json
        # and extract original instructions
        for t in ["conflict_resolved", "deepseek_optimized", "qwen_optimized", "lost_in_middle"]:
            full_path = transform_dir / t / "transformed_prompts.json"
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [
                    {
                        "id": item["original_id"],
                        "instruction": item["original_instruction"],
                        "level": item["level"],
                        "category": item["category"]
                    }
                    for item in data
                ]
        raise FileNotFoundError("No transformed prompts found to extract originals from")
    
    instructions_file = transform_dir / transformation / "instructions.json"
    if not instructions_file.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_file}")
    
    with open(instructions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_inference_on_prompts(
    prompts: List[Dict],
    model_name: str,
    output_file: Path
) -> List[Dict]:
    """Run inference on a list of prompts."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
    
    api_key = get_api_key(model_name)
    base_url = config.get("base_url")
    
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    responses = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"  Inference {i}/{len(prompts)}: ID {prompt['id']}")
        
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt["instruction"]}]
            )
            
            latency = time.time() - start_time
            
            responses.append({
                "test_case_id": f"transformed_{prompt['id']}_{prompt['level']}",
                "model_name": model_name,
                "prompt": prompt["instruction"],
                "response": response.choices[0].message.content,
                "latency": latency,
                "level": prompt["level"],
                "category": prompt["category"],
                "error": None
            })
            
            print(f"    ✓ Success ({latency:.2f}s)")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            responses.append({
                "test_case_id": f"transformed_{prompt['id']}_{prompt['level']}",
                "model_name": model_name,
                "prompt": prompt["instruction"],
                "response": "",
                "latency": 0,
                "level": prompt["level"],
                "category": prompt["category"],
                "error": str(e)
            })
        
        time.sleep(0.5)
    
    # Save responses
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    return responses


def run_evaluation_on_responses(
    prompts: List[Dict],
    responses: List[Dict],
    output_file: Path
) -> List[Dict]:
    """Evaluate responses using GPT as judge."""
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    response_lookup = {r["test_case_id"]: r for r in responses}
    evaluations = []
    
    for i, prompt in enumerate(prompts, 1):
        test_case_id = f"transformed_{prompt['id']}_{prompt['level']}"
        print(f"  Evaluating {i}/{len(prompts)}: ID {prompt['id']}")
        
        if test_case_id not in response_lookup:
            print(f"    ⚠ No response found")
            continue
        
        response_data = response_lookup[test_case_id]
        
        if response_data.get("error"):
            print(f"    ⚠ Response had error")
            evaluations.append({
                "test_case_id": test_case_id,
                "level": prompt["level"],
                "hsr": 0.0,
                "ssr": 0.0,
                "constraint_evaluations": []
            })
            continue
        
        level = prompt["level"]
        
        if level == 0:
            evaluations.append({
                "test_case_id": test_case_id,
                "level": level,
                "hsr": 1.0,
                "ssr": 1.0,
                "constraint_evaluations": []
            })
            print(f"    ✓ Level 0 (no constraints)")
            continue
        
        # Simple evaluation prompt
        eval_prompt = f"""Evaluate if this response follows the instructions properly.

INSTRUCTION:
{prompt['instruction']}

RESPONSE:
{response_data['response']}

Rate constraint satisfaction. Respond in JSON:
{{
  "evaluations": [
    {{"constraint_id": 1, "satisfied": true/false, "confidence": 0.0-1.0, "explanation": "..."}}
  ]
}}"""
        
        try:
            eval_response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are an expert constraint evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_completion_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            content = eval_response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")
            
            eval_result = json.loads(content)
            constraint_evals = eval_result.get("evaluations", [])
            
            if constraint_evals:
                satisfied = sum(1 for ce in constraint_evals if ce.get("satisfied", False))
                total = len(constraint_evals)
                hsr = 1.0 if satisfied == total else 0.0
                ssr = satisfied / total
            else:
                hsr = 0.0
                ssr = 0.0
            
            evaluations.append({
                "test_case_id": test_case_id,
                "level": level,
                "hsr": hsr,
                "ssr": ssr,
                "constraint_evaluations": constraint_evals
            })
            
            print(f"    ✓ HSR: {hsr:.2f}, SSR: {ssr:.2f}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            evaluations.append({
                "test_case_id": test_case_id,
                "level": level,
                "hsr": 0.0,
                "ssr": 0.0,
                "constraint_evaluations": []
            })
        
        time.sleep(0.5)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    
    return evaluations


def compare_results(results_dir: Path, model_name: str):
    """Compare results across all transformations."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {model_name}")
    print(f"{'='*70}\n")
    
    comparison_data = {}
    
    for transformation in TRANSFORMATIONS:
        eval_file = results_dir / f"evaluations_{model_name}_{transformation}.json"
        if not eval_file.exists():
            print(f"  {transformation}: No results found")
            continue
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        metrics = calculate_metrics(evaluations)
        comparison_data[transformation] = metrics
        
        print(f"  {transformation:20s}: HSR={metrics['HSR']:.4f}  SSR={metrics['SSR']:.4f}  CSL={metrics['CSL']}")
    
    # Find best performing transformation
    if comparison_data:
        best_hsr = max(comparison_data.items(), key=lambda x: x[1]['HSR'])
        best_ssr = max(comparison_data.items(), key=lambda x: x[1]['SSR'])
        
        print(f"\n{'='*70}")
        print(f"Best HSR: {best_hsr[0]} ({best_hsr[1]['HSR']:.4f})")
        print(f"Best SSR: {best_ssr[0]} ({best_ssr[1]['SSR']:.4f})")
        
        # Show improvement over original
        if "original" in comparison_data:
            baseline = comparison_data["original"]
            print(f"\nImprovement over original:")
            for t, m in comparison_data.items():
                if t != "original":
                    hsr_diff = m['HSR'] - baseline['HSR']
                    ssr_diff = m['SSR'] - baseline['SSR']
                    print(f"  {t:20s}: HSR {hsr_diff:+.4f}  SSR {ssr_diff:+.4f}")
    
    # Save comparison
    comparison_file = results_dir / f"comparison_{model_name}.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transformed FollowBench prompts"
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-5-nano",
        help="Model to evaluate"
    )
    
    parser.add_argument(
        "--transformation", "-t",
        choices=TRANSFORMATIONS,
        default=None,
        help="Specific transformation to evaluate"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all transformations"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results across transformations"
    )
    
    parser.add_argument(
        "--transform-dir",
        type=Path,
        default=Path("transformed_prompts"),
        help="Directory with transformed prompts"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "transformed",
        help="Directory for results"
    )
    
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, use existing responses"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation, use existing results"
    )
    
    args = parser.parse_args()
    
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare mode
    if args.compare:
        compare_results(args.results_dir, args.model)
        return
    
    # Determine transformations to evaluate
    if args.all:
        transformations = TRANSFORMATIONS
    elif args.transformation:
        transformations = [args.transformation]
    else:
        print("Specify --transformation, --all, or --compare")
        sys.exit(1)
    
    # Evaluate each transformation
    for transformation in transformations:
        print(f"\n{'='*60}")
        print(f"Evaluating: {transformation}")
        print(f"{'='*60}")
        
        try:
            prompts = load_transformed_prompts(args.transform_dir, transformation)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue
        
        print(f"  Loaded {len(prompts)} prompts")
        
        responses_file = args.results_dir / f"responses_{args.model}_{transformation}.json"
        evaluations_file = args.results_dir / f"evaluations_{args.model}_{transformation}.json"
        
        # Inference
        if args.skip_inference and responses_file.exists():
            print(f"  Loading existing responses from {responses_file}")
            with open(responses_file, 'r', encoding='utf-8') as f:
                responses = json.load(f)
        else:
            print(f"  Running inference with {args.model}...")
            responses = run_inference_on_prompts(prompts, args.model, responses_file)
        
        # Evaluation
        if args.skip_evaluation and evaluations_file.exists():
            print(f"  Loading existing evaluations from {evaluations_file}")
            with open(evaluations_file, 'r', encoding='utf-8') as f:
                evaluations = json.load(f)
        else:
            print(f"  Running evaluation...")
            evaluations = run_evaluation_on_responses(prompts, responses, evaluations_file)
        
        # Calculate metrics
        metrics = calculate_metrics(evaluations)
        print(f"\n  Results for {transformation}:")
        print(f"    HSR: {metrics['HSR']:.4f}")
        print(f"    SSR: {metrics['SSR']:.4f}")
        print(f"    CSL: {metrics['CSL']}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results directory: {args.results_dir}")
    print(f"\nRun with --compare to see comparison across transformations")


if __name__ == "__main__":
    main()
