#!/usr/bin/env python3
"""
Prompt Transformer for FollowBench Dataset.

This script applies various prompt transformation strategies from .github/prompts/
to FollowBench instructions, creating modified versions for evaluation.

Transformations:
1. conflict-resolved: Detect and resolve conflicting instructions
2. lost-in-middle: Apply "Universal Sandwich" architecture for better attention
3. deepseek-optimized: Optimize prompts for DeepSeek V3.1 architecture
4. qwen-optimized: Optimize prompts for Qwen 3 architecture

Usage:
    poetry run python -m src.prompt_transformer --transformation deepseek
    poetry run python -m src.prompt_transformer --transformation all
    poetry run python -m src.prompt_transformer --list
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package required. Install with: poetry add datasets")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' package required. Install with: pip install openai")
    sys.exit(1)

# Support both direct execution and module execution
try:
    from src.config import get_api_key, RESULTS_DIR
except ImportError:
    from config import get_api_key, RESULTS_DIR


# Define transformation types and their output folders
TRANSFORMATIONS = {
    "conflict-resolved": {
        "description": "Detect and resolve conflicting instructions",
        "folder": "conflict_resolved",
        "prompt_file": "conflict-detector.prompt.md"
    },
    "lost-in-middle": {
        "description": "Apply Universal Sandwich architecture",
        "folder": "lost_in_middle",
        "prompt_file": "lost-in-the-middle.prompt.md"
    },
    "deepseek-optimized": {
        "description": "Optimize for DeepSeek V3.1 architecture",
        "folder": "deepseek_optimized",
        "prompt_file": "refine-for-deepseek-v3.1.prompt.md"
    },
    "qwen-optimized": {
        "description": "Optimize for Qwen 3 architecture",
        "folder": "qwen_optimized",
        "prompt_file": "refine-for-qwen3.prompt.md"
    }
}


@dataclass
class TransformedPrompt:
    """Represents a transformed prompt."""
    original_id: str
    transformation: str
    original_instruction: str
    transformed_instruction: str
    level: int
    category: str
    changes_summary: str


def load_transformation_prompt(prompt_file: str) -> str:
    """Load a transformation prompt from .github/prompts/"""
    prompt_path = Path(__file__).parent.parent / ".github" / "prompts" / prompt_file
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    content = prompt_path.read_text(encoding='utf-8')
    
    # Remove markdown code fences if present
    if content.startswith("```") or content.startswith("````"):
        lines = content.split('\n')
        # Find start and end of code block
        start_idx = 0
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("```") or line.startswith("````"):
                if start_idx == 0:
                    start_idx = i + 1
                else:
                    end_idx = i
                    break
        content = '\n'.join(lines[start_idx:end_idx])
    
    return content.strip()


def load_followbench_samples(
    max_samples: Optional[int] = None,
    levels: Optional[List[int]] = None,
    categories: Optional[List[str]] = None
) -> List[Dict]:
    """Load and filter FollowBench samples."""
    print("Loading FollowBench dataset...")
    dataset = load_dataset("YuxinJiang/FollowBench", split="train")
    
    samples = []
    for item in dataset:
        sample = {
            "id": item['example_id'],
            "instruction": item['instruction'],
            "level": item['level'],
            "category": item.get('category', ''),
            "target": item.get('target', ''),
            "source": item.get('source', '')
        }
        samples.append(sample)
    
    # Apply filters
    if levels:
        samples = [s for s in samples if s['level'] in levels]
    
    if categories:
        categories_lower = [c.lower() for c in categories]
        samples = [s for s in samples if s['category'].lower() in categories_lower]
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Selected {len(samples)} samples for transformation")
    return samples


def transform_prompt_with_llm(
    instruction: str,
    transformation_prompt: str,
    transformation_name: str,
    client: OpenAI
) -> tuple[str, str]:
    """
    Use GPT to transform a prompt according to the transformation template.
    
    Returns: (transformed_instruction, changes_summary)
    """
    # Build the request based on transformation type
    if transformation_name == "conflict-resolved":
        user_message = f"""Analyze the following prompt for conflicts and provide a resolved version.

ORIGINAL PROMPT:
{instruction}

First, output the corrected prompt (with all conflicts resolved), then provide a brief summary of changes.

Your response format:
---TRANSFORMED---
[the transformed prompt here - this should be the complete, usable prompt]
---SUMMARY---
[brief summary of changes made]"""

    elif transformation_name == "lost-in-middle":
        user_message = f"""Restructure the following prompt using the Universal Sandwich architecture to prevent the "Lost in the Middle" phenomenon.

ORIGINAL PROMPT:
{instruction}

Apply the three-zone structure:
- Zone 1 (Primacy): Critical instructions at the beginning
- Zone 2 (Middle): Context data wrapped in <context_data> tags
- Zone 3 (Recency): Final instructions and constraint reminders

Your response format:
---TRANSFORMED---
[the restructured prompt here - this should be the complete, usable prompt]
---SUMMARY---
[brief summary of structural changes]"""

    elif transformation_name == "deepseek-optimized":
        user_message = f"""Optimize the following prompt for DeepSeek V3.1 architecture.

ORIGINAL PROMPT:
{instruction}

Apply DeepSeek V3.1 best practices:
- Use ### headers
- Wrap inputs in XML tags  
- Remove redundancy
- Use positive constraints
- Add appropriate mode (Standard/Thinking)

Your response format:
---TRANSFORMED---
[the optimized prompt here - this should be the complete, usable prompt]
---SUMMARY---
[brief summary of optimizations applied]"""

    elif transformation_name == "qwen-optimized":
        user_message = f"""Optimize the following prompt for Qwen 3 architecture.

ORIGINAL PROMPT:
{instruction}

Apply Qwen 3 best practices:
- Start with "You are [Role]" header
- Use XML tags for constraints (<rules>, <constraints>)
- Add tone suppression block
- Use <protocol> for procedural steps

Your response format:
---TRANSFORMED---
[the optimized prompt here - this should be the complete, usable prompt]
---SUMMARY---
[brief summary of optimizations applied]"""

    else:
        raise ValueError(f"Unknown transformation: {transformation_name}")

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a prompt optimization expert. Transform prompts as requested. Always use the exact format markers: ---TRANSFORMED--- and ---SUMMARY---"},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=4096
        )
        
        result = response.choices[0].message.content
        
        # Debug: print first 500 chars of response
        # print(f"    DEBUG Response (first 500): {result[:500] if result else 'EMPTY'}")
        
        # Parse the response using the markers
        transformed = ""
        summary = ""
        
        if "---TRANSFORMED---" in result and "---SUMMARY---" in result:
            # Standard format
            parts = result.split("---TRANSFORMED---")
            if len(parts) > 1:
                rest = parts[1]
                if "---SUMMARY---" in rest:
                    prompt_part, summary_part = rest.split("---SUMMARY---", 1)
                    transformed = prompt_part.strip()
                    summary = summary_part.strip()
                else:
                    transformed = rest.strip()
                    summary = "Transformation applied"
        elif "TRANSFORMED_PROMPT:" in result:
            # Fallback format
            parts = result.split("TRANSFORMED_PROMPT:")
            if len(parts) > 1:
                rest = parts[1]
                if "CHANGES_SUMMARY:" in rest:
                    prompt_part, summary_part = rest.split("CHANGES_SUMMARY:", 1)
                    transformed = prompt_part.strip()
                    summary = summary_part.strip()
                else:
                    transformed = rest.strip()
                    summary = "Transformation applied"
        else:
            # Last resort: The model likely just returned the transformed prompt directly
            # Use it as-is if it looks different from the original
            transformed = result.strip()
            # Remove any markdown code blocks if wrapping the whole thing
            if transformed.startswith("```"):
                lines = transformed.split('\n')
                # Find end of code block
                end_idx = len(lines)
                for i in range(1, len(lines)):
                    if lines[i].startswith("```"):
                        end_idx = i
                        break
                transformed = '\n'.join(lines[1:end_idx])
            summary = "Transformation applied (direct output)"
        
        # Ensure we have something meaningful
        if not transformed or len(transformed) < 10:
            transformed = instruction  # Fall back to original
            summary = "Error: Could not parse transformation, using original"
        
        return transformed, summary
        
    except Exception as e:
        print(f"    Error during transformation: {e}")
        return instruction, f"Error: {str(e)}"


def run_transformation(
    transformation_name: str,
    samples: List[Dict],
    output_dir: Path,
    client: OpenAI
) -> List[TransformedPrompt]:
    """Run a specific transformation on all samples."""
    
    config = TRANSFORMATIONS[transformation_name]
    print(f"\n{'='*60}")
    print(f"Transformation: {transformation_name}")
    print(f"Description: {config['description']}")
    print(f"Output folder: {output_dir / config['folder']}")
    print(f"{'='*60}\n")
    
    # Load transformation prompt
    try:
        transformation_prompt = load_transformation_prompt(config['prompt_file'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    
    # Create output directory
    transform_dir = output_dir / config['folder']
    transform_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        print(f"Processing {i}/{len(samples)}: ID {sample['id']} (Level {sample['level']})")
        
        transformed_instruction, changes_summary = transform_prompt_with_llm(
            sample['instruction'],
            transformation_prompt,
            transformation_name,
            client
        )
        
        result = TransformedPrompt(
            original_id=str(sample['id']),
            transformation=transformation_name,
            original_instruction=sample['instruction'],
            transformed_instruction=transformed_instruction,
            level=sample['level'],
            category=sample['category'],
            changes_summary=changes_summary
        )
        results.append(result)
        
        print(f"  ✓ Transformed ({len(changes_summary)} char summary)")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save results
    output_file = transform_dir / "transformed_prompts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save just the transformed instructions for easy use
    instructions_file = transform_dir / "instructions.json"
    instructions_data = [
        {
            "id": r.original_id,
            "level": r.level,
            "category": r.category,
            "instruction": r.transformed_instruction
        }
        for r in results
    ]
    with open(instructions_file, 'w', encoding='utf-8') as f:
        json.dump(instructions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Instructions saved to: {instructions_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transform FollowBench prompts using various optimization strategies"
    )
    
    parser.add_argument(
        "--transformation", "-t",
        choices=list(TRANSFORMATIONS.keys()) + ["all"],
        default="all",
        help="Transformation to apply (default: all)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples to transform (default: 10)"
    )
    
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Filter by constraint levels"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Filter by categories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transformed_prompts"),
        help="Output directory (default: transformed_prompts)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available transformations and exit"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nAvailable Transformations:")
        print("="*60)
        for name, config in TRANSFORMATIONS.items():
            print(f"\n  {name}")
            print(f"    Description: {config['description']}")
            print(f"    Output folder: {config['folder']}")
            print(f"    Prompt file: {config['prompt_file']}")
        print("\nUsage examples:")
        print("  poetry run python -m src.prompt_transformer -t deepseek-optimized --max-samples 5")
        print("  poetry run python -m src.prompt_transformer -t all --levels 3 4 5")
        return
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable required")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load samples
    samples = load_followbench_samples(
        max_samples=args.max_samples,
        levels=args.levels,
        categories=args.categories
    )
    
    if not samples:
        print("No samples to transform")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which transformations to run
    if args.transformation == "all":
        transformations_to_run = list(TRANSFORMATIONS.keys())
    else:
        transformations_to_run = [args.transformation]
    
    # Run transformations
    all_results = {}
    for transformation in transformations_to_run:
        results = run_transformation(
            transformation,
            samples,
            args.output_dir,
            client
        )
        all_results[transformation] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRANSFORMATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(samples)}")
    print(f"Transformations applied: {len(transformations_to_run)}")
    for transformation, results in all_results.items():
        successful = sum(1 for r in results if "Error" not in r.changes_summary)
        print(f"  {transformation}: {successful}/{len(results)} successful")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
