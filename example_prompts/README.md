# Example Prompts

This directory is for your own prompts that you want to optimize using the FollowBench evaluation pipeline.

## Structure

Place your prompt files here in `.txt` format. Each file should contain one or more prompts that you want to evaluate.

## Example

Create a file like `my_prompts.txt`:

```
You are a helpful assistant. Help the user with their questions.
Be concise but thorough in your responses.
Always cite sources when providing factual information.
```

Then run the evaluation pipeline on it:

```bash
python -m src.pipeline generate --prompt-file example_prompts/my_prompts.txt
python -m src.pipeline full --model gpt-4
```
