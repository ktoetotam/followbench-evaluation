# FollowBench Evaluation Pipeline

This codebase implements a multi-stage LLM evaluation pipeline using the FollowBench methodology for instruction-following capability testing.

## Architecture Overview

The pipeline follows a **4-step sequential workflow**:

1. **Generate** (`followbench_generator.py`) - Creates test cases with 1-5 constraints from prompt files
2. **Infer** (`model_inference.py`) - Runs target models (GPT-4, DeepSeek, Qwen) on test cases
3. **Evaluate** (`llm_evaluator.py`) - Uses GPT-4 as judge to assess constraint satisfaction
4. **Analyze** (`results_analyzer.py`) - Aggregates metrics (HSR/SSR/CSL) and generates reports

Each step produces JSON artifacts in `followbench_data/` or `evaluation_results/` that feed the next stage.

## Critical Patterns

### Dual-Mode Operation
The pipeline supports TWO distinct evaluation modes:
- **Custom prompts**: Place `.txt` files in `prompts/` or `corrected_prompts/<stage>/` directories
- **FollowBench dataset**: Load official test cases from HuggingFace with `--followbench` flag

When editing code, check if it affects custom prompt mode, dataset mode, or both.

### Constraint System
Constraints are typed (`content`, `situation`, `style`, `format`, `example`) and generated using templates in `followbench_generator._load_constraint_templates()`. Each constraint needs:
- `description`: User-facing requirement
- `checker`: Rule-based or LLM evaluation strategy

### Model Configuration
Add new models in `config.MODEL_CONFIGS`. Required fields depend on API:
- Standard OpenAI: `api_key_env`, `model_name`, `temperature`, `max_tokens`
- Custom endpoint: Add `base_url`
- Azure OpenAI: Add `is_azure: True`, `endpoint_env`, `deployment_env`, `api_version`

Azure support is implemented via flags (`--azure`) and separate client initialization in `llm_evaluator.LLMEvaluator.__init__()`.

### Metrics Calculation
- **HSR** (Hard Satisfaction Rate): 1.0 if ALL constraints satisfied, else 0.0 (strict binary per test case)
- **SSR** (Soft Satisfaction Rate): Average satisfaction across individual constraints (percentage)
- **CSL** (Consistent Satisfaction Levels): Highest consecutive level where average HSR ≥ 0.5

These are calculated in `results_analyzer.calculate_metrics()`. CSL requires iterating levels sequentially until threshold breaks.

## Developer Workflows

### Environment Setup
This project uses **Poetry** for dependency management. Always run Python through Poetry's virtual environment:

```bash
# Install dependencies (creates .venv)
poetry install

# Run any Python script via Poetry
poetry run python -m src.pipeline full --model gpt-4
poetry run python -m src.run_followbench --model gpt-4

# Or activate the shell first
poetry shell
python -m src.pipeline full --model gpt-4
```

**Important**: Never run `python` directly without `poetry run` or activating the Poetry shell first. This ensures the correct virtual environment and dependencies are used.

### Running the Pipeline
```bash
# Full pipeline (all stages)
poetry run python -m src.pipeline full --model gpt-4

# Individual steps
poetry run python -m src.pipeline generate
poetry run python -m src.pipeline infer --model gpt-4 --stages original deepseek-optimised
poetry run python -m src.pipeline evaluate --model gpt-4
poetry run python -m src.pipeline analyze --model gpt-4
```

### Adding a New Model
1. Add entry to `config.MODEL_CONFIGS` with API configuration
2. Set environment variable for API key (specified in `api_key_env`) in `.env` file
3. If using custom API client, modify `model_inference.ModelInference._initialize_client()`
4. Test with: `poetry run python -m src.pipeline infer --model your-model`

### Evaluating on FollowBench Dataset
```bash
# Load official dataset and evaluate responses
poetry run python -m src.llm_evaluator responses.json --followbench

# Use Azure OpenAI for evaluation
poetry run python -m src.llm_evaluator responses.json --followbench --azure
```

Requires `datasets` package and HuggingFace access. Dataset loader is in `llm_evaluator.LLMEvaluator.load_followbench_dataset()`.

## File Conventions

### Directory Structure
- `followbench_data/` - Generated test cases (JSON): `test_cases_{stage}.json`
- `evaluation_results/` - Inference outputs: `responses_{model}_{stage}.json`
- `evaluation_results/` - Evaluation results: `evaluations_{model}_{stage}.json`
- `evaluation_results/analysis_{model}/` - Reports and CSV exports

### Transformation Stages
Defined in `config.TRANSFORMATION_STAGES`:
- `original` - Baseline prompts from `prompts/`
- `resolved_conflicts`, `lost_in_the_middle` - Intermediate optimizations
- `deepseek-optimised`, `qwen-optimised` - Model-specific optimizations

Stages map to directories in `corrected_prompts/`. Pipeline can filter stages with `--stages` flag.

### Data Formats
All intermediate data uses dataclasses with `asdict()` for JSON serialization:
- `TestCase`: Test case with constraints
- `ModelResponse`: Model inference output
- `TestCaseEvaluation`: GPT-4 evaluation with constraint-level results

When modifying these, update both dataclass and JSON serialization/deserialization code.

## Important Implementation Details

### Rate Limiting
Both `model_inference.py` and `llm_evaluator.py` include `time.sleep(0.5)` between API calls. Adjust for different rate limits or faster models.

### Error Handling
- Inference errors store empty response with `error` field populated
- Evaluation errors return zero metrics (HSR=0.0, SSR=0.0)
- Pipeline continues on individual failures but reports them

### JSON Response Format
GPT-4 evaluator uses `response_format={"type": "json_object"}` for structured output. Prompts must explicitly request JSON format and include schema example.

## Testing Strategy

No formal test suite exists. Test changes by:
1. Generate test cases: `poetry run python -m src.pipeline generate` (fast, no API calls)
2. Run inference on small subset by modifying `TEST_CONFIG["samples_per_level"]` in `config.py`
3. Check intermediate JSON files for correctness
4. Verify metrics calculation with known ground truth

When adding constraint types, manually verify evaluation prompt in `llm_evaluator.EVALUATION_PROMPT_TEMPLATE` correctly describes new type.
