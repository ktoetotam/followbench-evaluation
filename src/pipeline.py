"""
Main pipeline runner for FollowBench evaluation
Orchestrates test generation, model inference, evaluation, and analysis
"""

import argparse
from pathlib import Path
import sys

from .config import (
    FOLLOWBENCH_DATA_DIR,
    RESULTS_DIR,
    MODEL_CONFIGS,
    TRANSFORMATION_STAGES
)
from .followbench_generator import FollowBenchGenerator
from .model_inference import run_inference_pipeline
from .llm_evaluator import run_evaluation_pipeline
from .results_analyzer import run_analysis_pipeline


def generate_test_cases():
    """Step 1: Generate FollowBench test cases from prompts"""
    print("\n" + "="*80)
    print("STEP 1: Generating FollowBench Test Cases")
    print("="*80 + "\n")
    
    generator = FollowBenchGenerator()
    all_test_cases = generator.generate_all_test_cases()
    
    return all_test_cases


def run_model_inference(model_name: str, stages: list = None):
    """Step 2: Run model inference on test cases"""
    print("\n" + "="*80)
    print(f"STEP 2: Running Model Inference ({model_name})")
    print("="*80 + "\n")
    
    if stages is None:
        stages = TRANSFORMATION_STAGES
    
    results = {}
    
    for stage in stages:
        test_cases_file = FOLLOWBENCH_DATA_DIR / f"test_cases_{stage}.json"
        
        if not test_cases_file.exists():
            print(f"Warning: Test cases file not found: {test_cases_file}")
            continue
        
        output_file = RESULTS_DIR / f"responses_{model_name}_{stage}.json"
        
        responses = run_inference_pipeline(
            model_name,
            test_cases_file,
            output_file
        )
        
        results[stage] = responses
    
    return results


def run_llm_evaluation(model_name: str, stages: list = None):
    """Step 3: Run LLM-based evaluation"""
    print("\n" + "="*80)
    print(f"STEP 3: Running LLM Evaluation ({model_name})")
    print("="*80 + "\n")
    
    if stages is None:
        stages = TRANSFORMATION_STAGES
    
    results = {}
    
    for stage in stages:
        test_cases_file = FOLLOWBENCH_DATA_DIR / f"test_cases_{stage}.json"
        responses_file = RESULTS_DIR / f"responses_{model_name}_{stage}.json"
        
        if not test_cases_file.exists() or not responses_file.exists():
            print(f"Warning: Required files not found for stage: {stage}")
            continue
        
        output_file = RESULTS_DIR / f"evaluations_{model_name}_{stage}.json"
        
        evaluations = run_evaluation_pipeline(
            test_cases_file,
            responses_file,
            output_file
        )
        
        results[stage] = evaluations
    
    return results


def run_results_analysis(model_name: str):
    """Step 4: Analyze and aggregate results"""
    print("\n" + "="*80)
    print(f"STEP 4: Analyzing Results ({model_name})")
    print("="*80 + "\n")
    
    # Find all evaluation files for this model
    evaluation_files = {}
    for stage in TRANSFORMATION_STAGES:
        eval_file = RESULTS_DIR / f"evaluations_{model_name}_{stage}.json"
        if eval_file.exists():
            evaluation_files[stage] = eval_file
    
    if not evaluation_files:
        print(f"No evaluation files found for model: {model_name}")
        return None
    
    output_dir = RESULTS_DIR / f"analysis_{model_name}"
    run_analysis_pipeline(evaluation_files, output_dir)
    
    return output_dir


def run_full_pipeline(model_name: str, stages: list = None):
    """Run complete FollowBench evaluation pipeline"""
    print("\n" + "="*80)
    print("FOLLOWBENCH EVALUATION PIPELINE")
    print(f"Model: {model_name}")
    print("="*80 + "\n")
    
    # Step 1: Generate test cases (if not already done)
    test_cases_exist = all(
        (FOLLOWBENCH_DATA_DIR / f"test_cases_{stage}.json").exists()
        for stage in (stages or TRANSFORMATION_STAGES)
    )
    
    if not test_cases_exist:
        generate_test_cases()
    else:
        print("Test cases already exist, skipping generation...")
    
    # Step 2: Model inference
    run_model_inference(model_name, stages)
    
    # Step 3: LLM evaluation
    run_llm_evaluation(model_name, stages)
    
    # Step 4: Results analysis
    output_dir = run_results_analysis(model_name)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the analysis report")
    print("2. Check CSV files for detailed results")
    print("3. Compare across different models if available")
    print("="*80 + "\n")


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="FollowBench Evaluation Pipeline for Prompt Optimization"
    )
    
    parser.add_argument(
        "command",
        choices=["generate", "infer", "evaluate", "analyze", "full"],
        help="Pipeline step to run"
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-4",
        help="Model to evaluate"
    )
    
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=TRANSFORMATION_STAGES,
        help="Transformation stages to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "generate":
            generate_test_cases()
        
        elif args.command == "infer":
            run_model_inference(args.model, args.stages)
        
        elif args.command == "evaluate":
            run_llm_evaluation(args.model, args.stages)
        
        elif args.command == "analyze":
            run_results_analysis(args.model)
        
        elif args.command == "full":
            run_full_pipeline(args.model, args.stages)
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
