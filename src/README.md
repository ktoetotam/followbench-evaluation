# FollowBench Evaluation Pipeline

This directory contains the implementation of a FollowBench-style evaluation pipeline for testing prompt optimization across different transformation stages.

## Overview

The pipeline evaluates prompts using the FollowBench methodology:
- **Multi-level constraint testing**: Tests with 1-5 constraints at different difficulty levels
- **Fine-grained constraint types**: Content, Situation, Style, Format, Example
- **LLM-based evaluation**: Uses GPT-4 to assess constraint satisfaction
- **Comprehensive metrics**: HSR (Hard Satisfaction Rate), SSR (Soft Satisfaction Rate), CSL (Consistent Satisfaction Levels)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys
export OPENAI_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"  # Optional
export DASHSCOPE_API_KEY="your-key-here"  # Optional for Qwen
```

## Pipeline Steps

### 1. Generate Test Cases

Generate FollowBench test cases from your prompt files:

```bash
python -m src.pipeline generate
```

This creates test cases for each transformation stage in `followbench_data/`.

### 2. Run Model Inference

Run inference with a specific model:

```bash
python -m src.pipeline infer --model gpt-4
python -m src.pipeline infer --model deepseek-v3
python -m src.pipeline infer --model qwen-max
```

Results saved to `evaluation_results/responses_*.json`.

### 3. Evaluate Responses

Use GPT-4 to evaluate whether responses satisfy constraints:

```bash
python -m src.pipeline evaluate --model gpt-4
```

Results saved to `evaluation_results/evaluations_*.json`.

### 4. Analyze Results

Generate comprehensive analysis report:

```bash
python -m src.pipeline analyze --model gpt-4
```

Creates:
- Text report with overall metrics
- CSV files with detailed breakdowns
- Comparisons across transformation stages

### 5. Run Full Pipeline

Run all steps in sequence:

```bash
python -m src.pipeline full --model gpt-4
```

Optional: Specify specific transformation stages:

```bash
python -m src.pipeline full --model gpt-4 --stages original deepseek-optimised
```

## Directory Structure

```
src/
├── __init__.py              # Package initialization
├── config.py                # Configuration and constants
├── followbench_generator.py # Test case generation
├── model_inference.py       # Model API calls
├── llm_evaluator.py         # LLM-based evaluation
├── results_analyzer.py      # Results analysis and reporting
└── pipeline.py              # Main pipeline orchestration

followbench_data/           # Generated test cases
├── test_cases_original.json
├── test_cases_resolved_conflicts.json
├── test_cases_lost_in_the_middle.json
├── test_cases_deepseek-optimised.json
└── test_cases_qwen-optimised.json

evaluation_results/         # Inference and evaluation results
├── responses_*.json        # Model responses
├── evaluations_*.json      # Evaluation results
└── analysis_*/             # Analysis reports
    ├── followbench_report.txt
    └── csv/               # Detailed CSV files
```

## Metrics

### Hard Satisfaction Rate (HSR)
The proportion of test cases where ALL constraints are satisfied. Strict metric.

### Soft Satisfaction Rate (SSR)
The average satisfaction rate across all individual constraints. More lenient.

### Consistent Satisfaction Levels (CSL)
The highest consecutive constraint level (1-5) where the model maintains HSR ≥ 0.5.

## Configuration

Edit `src/config.py` to customize:

- **Model configurations**: Add/modify model API settings
- **Constraint types**: Adjust types of constraints to test
- **Test configuration**: Change number of samples, levels, etc.
- **Transformation stages**: Define which optimization stages to compare

## Example Workflow

```bash
# 1. Generate test cases for all prompts
python -m src.pipeline generate

# 2. Test with GPT-4
python -m src.pipeline full --model gpt-4

# 3. Test with DeepSeek (for comparison)
python -m src.pipeline full --model deepseek-v3

# 4. Compare results
python -m src.results_analyzer
```

## Advanced Usage

### Custom Test Generation

```python
from src.followbench_generator import FollowBenchGenerator

generator = FollowBenchGenerator()

# Generate for specific file
test_cases = generator.generate_test_cases(
    prompt_file=Path("prompts/MyPrompt.txt"),
    transformation_stage="original",
    num_samples_per_level=20  # More samples
)

generator.save_test_cases(test_cases, Path("custom_tests.json"))
```

### Batch Model Evaluation

```python
from src.model_inference import ModelInference
from src.config import FOLLOWBENCH_DATA_DIR

inference = ModelInference("gpt-4")

# Load test cases
test_cases_file = FOLLOWBENCH_DATA_DIR / "test_cases_original.json"
with open(test_cases_file) as f:
    test_cases_data = json.load(f)

# Run inference
responses = inference.infer_batch(test_cases, save_path=Path("my_responses.json"))
```

### Custom Analysis

```python
from src.results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer()

# Load and analyze
evaluation_files = {
    "original": Path("evaluation_results/evaluations_gpt-4_original.json"),
    "optimized": Path("evaluation_results/evaluations_gpt-4_deepseek-optimised.json")
}

analyzer.generate_report(evaluation_files, Path("my_report.txt"))
analyzer.export_to_csv(evaluation_files, Path("my_csv/"))
```

## Troubleshooting

**API Key Errors:**
```bash
# Verify keys are set
echo $OPENAI_API_KEY
echo $DEEPSEEK_API_KEY
```

**Rate Limiting:**
- Pipeline includes 0.5s delays between API calls
- Adjust in `model_inference.py` and `llm_evaluator.py` if needed

**Missing Test Cases:**
- Run `python -m src.pipeline generate` first
- Check `followbench_data/` directory exists

**Evaluation Failures:**
- Check GPT-4 API key is valid
- Verify test case JSON files are well-formed
- Review error messages in console output

## Citation

This implementation is inspired by:

```bibtex
@inproceedings{jiang-etal-2024-followbench,
    title = "{F}ollow{B}ench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models",
    author = "Jiang, Yuxin and others",
    booktitle = "Proceedings of ACL 2024",
    year = "2024"
}
```

## Contributing

To add new constraint types or evaluation methods:

1. Edit `followbench_generator.py` to add constraint templates
2. Update `config.py` with new constraint type names
3. Modify `llm_evaluator.py` evaluation prompt if needed
4. Run tests to verify functionality

## License

See main project LICENSE file.
