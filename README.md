# FollowBench Evaluation Pipeline

A comprehensive toolkit for evaluating Large Language Models (LLMs) on their instruction-following capabilities using the FollowBench methodology.

## Overview

This project implements the [FollowBench](https://github.com/YJiangcm/FollowBench) evaluation framework, which provides multi-level, fine-grained constraint testing for LLMs. The pipeline supports:

- **Multi-level constraint testing** (1-5 constraints per test)
- **Five constraint types**: Content, Situation, Style, Format, Example
- **Three evaluation metrics**: HSR (Hard Satisfaction Rate), SSR (Soft Satisfaction Rate), CSL (Consistent Satisfaction Levels)
- **Multiple model support**: GPT-4, Azure OpenAI, DeepSeek V3, Qwen Max
- **LLM-based evaluation**: Uses GPT-4 as judge for constraint satisfaction

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/followbench-evaluation.git
cd followbench-evaluation

# Install dependencies
pip install -r requirements.txt
```

### Setup API Keys

Set up your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"

# For DeepSeek (optional)
export DEEPSEEK_API_KEY="your-key-here"

# For Qwen (optional)
export DASHSCOPE_API_KEY="your-key-here"
```

### Basic Usage

#### 1. Evaluate on FollowBench Dataset

The easiest way to get started is to evaluate a model on the FollowBench dataset from HuggingFace:

```bash
# First, generate responses from your model (you need to implement this part)
# The responses should be in JSON format: [{"test_case_id": "...", "response": "..."}, ...]

# Then evaluate using GPT-4 (OpenAI)
python -m src.llm_evaluator responses.json --followbench

# Or using Azure OpenAI
python -m src.llm_evaluator responses.json --followbench --azure
```

#### 2. Custom Prompt Evaluation

You can also evaluate your own prompts:

```bash
# Generate test cases from your prompts
python -m src.pipeline generate

# Run model inference
python -m src.pipeline infer --model gpt-4

# Evaluate responses
python -m src.pipeline evaluate --model gpt-4

# Analyze results
python -m src.pipeline analyze --model gpt-4

# Or run the full pipeline
python -m src.pipeline full --model gpt-4
```

## Features

### FollowBench Dataset Integration

The pipeline directly loads the FollowBench dataset from HuggingFace (`YuxinJiang/FollowBench`), which includes:
- 1,852 test cases
- Levels 0-5 (increasing constraint complexity)
- Multiple constraint types and domains

### Supported Models

- **GPT-4** (OpenAI)
- **Azure OpenAI** (GPT-4)
- **DeepSeek V3**
- **Qwen Max**

### Evaluation Metrics

- **HSR (Hard Satisfaction Rate)**: Percentage of test cases where ALL constraints are satisfied
- **SSR (Soft Satisfaction Rate)**: Average satisfaction rate across all constraints
- **CSL (Consistent Satisfaction Levels)**: Highest consecutive level with HSR ≥ 0.5

## Project Structure

```
.
├── src/
│   ├── config.py              # Configuration and model settings
│   ├── llm_evaluator.py       # LLM-based evaluation with GPT-4
│   ├── model_inference.py     # Model API integration
│   ├── results_analyzer.py    # Results aggregation and reporting
│   ├── pipeline.py            # Main pipeline orchestration
│   └── README.md              # Detailed pipeline documentation
├── example_prompts/           # Place your prompts here
├── evaluation_results/        # Evaluation outputs (auto-created)
├── followbench_data/          # Test case data (auto-created)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Advanced Usage

### Azure OpenAI Configuration

To use Azure OpenAI instead of regular OpenAI:

```bash
# Set environment variables
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Run with --azure flag
python -m src.llm_evaluator responses.json --followbench --azure
```

### Custom Model Integration

To add support for additional models, update `src/config.py`:

```python
MODEL_CONFIGS = {
    "your-model": {
        "api_key_env": "YOUR_API_KEY_ENV_VAR",
        "model_name": "your-model-name",
        "base_url": "https://api.yourmodel.com",  # if needed
        "temperature": 0.0,
        "max_tokens": 2048,
    }
}
```

## Documentation

For detailed documentation on the pipeline components, see:
- [Pipeline Documentation](src/README.md) - Detailed guide for each pipeline step
- [FollowBench Paper](https://arxiv.org/abs/2310.20410) - Original research paper

## Citation

If you use this evaluation pipeline in your research, please cite the FollowBench paper:

```bibtex
@misc{jiang2023followbench,
    title={FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models}, 
    author={Yuxin Jiang and Yufei Wang and Xingshan Zeng and Wanjun Zhong and Liangyou Li and Fei Mi and Lifeng Shang and Xin Jiang and Qun Liu and Wei Wang},
    year={2023},
    eprint={2310.20410},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [FollowBench](https://github.com/YJiangcm/FollowBench) for the evaluation methodology
- OpenAI for GPT-4 API
- HuggingFace for dataset hosting
