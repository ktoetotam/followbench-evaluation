"""
Configuration for FollowBench evaluation pipeline
"""

import os
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CORRECTED_PROMPTS_DIR = PROJECT_ROOT / "corrected_prompts"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
FOLLOWBENCH_DATA_DIR = PROJECT_ROOT / "followbench_data"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)
FOLLOWBENCH_DATA_DIR.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "gpt-4": {
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "azure-gpt-4": {
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "endpoint_env": "AZURE_OPENAI_ENDPOINT",
        "deployment_env": "AZURE_OPENAI_DEPLOYMENT",
        "api_version": "2024-02-15-preview",
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 2048,
        "is_azure": True,
    },
    "deepseek-v3": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "qwen-max": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "model_name": "qwen-max",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
}

# FollowBench constraint types
CONSTRAINT_TYPES = [
    "content",
    "situation", 
    "style",
    "format",
    "example",
]

# Evaluation metrics
METRICS = {
    "HSR": "Hard Satisfaction Rate",  # All constraints satisfied
    "SSR": "Soft Satisfaction Rate",  # Average satisfaction per constraint
    "CSL": "Consistent Satisfaction Levels",  # Consecutive levels satisfied
}

# Prompt transformation stages
TRANSFORMATION_STAGES = [
    "original",
    "resolved_conflicts",
    "lost_in_the_middle",
    "deepseek-optimised",
    "qwen-optimised",
]

# Test configuration
TEST_CONFIG = {
    "constraint_levels": [1, 2, 3, 4, 5],  # Number of constraints to test
    "samples_per_level": 10,  # Number of test samples per level
    "constraint_types_to_test": CONSTRAINT_TYPES,
    "evaluator_model": "gpt-4",  # Model to use for LLM-based evaluation
}


def get_api_key(model_name: str) -> str:
    """Get API key for specified model from environment"""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
    
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"API key not found. Please set {config['api_key_env']} environment variable"
        )
    
    return api_key


def get_prompt_files() -> List[Path]:
    """Get all prompt files from prompts directory"""
    return list(PROMPTS_DIR.glob("*.txt"))


def get_transformed_prompt_files(stage: str) -> List[Path]:
    """Get transformed prompt files for a specific stage"""
    if stage == "original":
        return get_prompt_files()
    
    stage_dir = CORRECTED_PROMPTS_DIR / stage
    if not stage_dir.exists():
        return []
    
    return list(stage_dir.glob("*.txt"))
