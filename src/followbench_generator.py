"""
Generate FollowBench test cases from prompt files
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from .config import CONSTRAINT_TYPES, TEST_CONFIG, FOLLOWBENCH_DATA_DIR


@dataclass
class Constraint:
    """Represents a single constraint"""
    type: str  # content, situation, style, format, example
    description: str
    checker: str  # Rule-based checker description or LLM evaluation prompt


@dataclass
class TestCase:
    """Represents a single FollowBench test case"""
    id: str
    level: int  # Number of constraints (1-5)
    base_instruction: str
    constraints: List[Constraint]
    full_instruction: str  # Base instruction + all constraints
    source_prompt: str  # Which prompt file this came from
    transformation_stage: str  # original, resolved_conflicts, etc.


class FollowBenchGenerator:
    """Generate FollowBench-style test cases from prompt files"""
    
    def __init__(self):
        self.constraint_templates = self._load_constraint_templates()
    
    def _load_constraint_templates(self) -> Dict[str, List[Dict]]:
        """Load constraint templates for different types"""
        return {
            "content": [
                {
                    "description": "Include specific technical terms: {terms}",
                    "checker": "Check if response contains all specified terms",
                },
                {
                    "description": "Mention at least {n} examples",
                    "checker": "Count number of examples provided",
                },
                {
                    "description": "Focus on {aspect} aspect only",
                    "checker": "Verify response focuses on specified aspect",
                },
            ],
            "situation": [
                {
                    "description": "Assume the user is a {role}",
                    "checker": "Check if language/detail level matches role",
                },
                {
                    "description": "Context: This is for {context}",
                    "checker": "Verify response is appropriate for context",
                },
            ],
            "style": [
                {
                    "description": "Use a {tone} tone",
                    "checker": "Evaluate tone of response",
                },
                {
                    "description": "Be {length}",
                    "checker": "Check response length",
                },
                {
                    "description": "Write in {style} style",
                    "checker": "Verify writing style matches requirement",
                },
            ],
            "format": [
                {
                    "description": "Provide response in JSON format",
                    "checker": "Validate JSON structure",
                },
                {
                    "description": "Use bullet points for main items",
                    "checker": "Check if bullet points are used",
                },
                {
                    "description": "Structure as: {structure}",
                    "checker": "Verify structural requirements",
                },
                {
                    "description": "Limit response to {n} sentences",
                    "checker": "Count sentences in response",
                },
            ],
            "example": [
                {
                    "description": "Follow this example format: {example}",
                    "checker": "Compare response format to example",
                },
                {
                    "description": "Similar to this: {example}",
                    "checker": "Check similarity to provided example",
                },
            ],
        }
    
    def extract_base_instructions(self, prompt_content: str, prompt_name: str) -> List[str]:
        """Extract base instructions from prompt file"""
        # Simple extraction - can be made more sophisticated
        instructions = []
        
        # Look for common instruction patterns
        lines = prompt_content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('#') or line.startswith('==='):
                continue
            
            # Look for instruction-like patterns
            if any(keyword in line.lower() for keyword in [
                'you are', 'your task', 'you must', 'you should',
                'objective', 'goal', 'when answering', 'provide'
            ]):
                # Extract meaningful instruction
                if len(line) > 20 and len(line) < 300:
                    instructions.append(line)
        
        return instructions[:10]  # Limit to 10 base instructions per file
    
    def generate_constraints(
        self, 
        level: int, 
        base_instruction: str
    ) -> List[Constraint]:
        """Generate N constraints for a given level"""
        constraints = []
        
        # Select constraint types (diversify)
        available_types = CONSTRAINT_TYPES.copy()
        random.shuffle(available_types)
        
        for i in range(level):
            constraint_type = available_types[i % len(available_types)]
            templates = self.constraint_templates[constraint_type]
            template = random.choice(templates)
            
            # Instantiate template with specific values
            description = self._instantiate_template(
                template["description"],
                constraint_type,
                base_instruction
            )
            
            constraint = Constraint(
                type=constraint_type,
                description=description,
                checker=template["checker"]
            )
            constraints.append(constraint)
        
        return constraints
    
    def _instantiate_template(
        self,
        template: str,
        constraint_type: str,
        base_instruction: str
    ) -> str:
        """Replace placeholders in constraint template"""
        # Define possible values for placeholders
        replacements = {
            "{terms}": "API, configuration, initialization",
            "{n}": str(random.randint(2, 5)),
            "{aspect}": random.choice(["technical", "practical", "theoretical"]),
            "{role}": random.choice(["beginner", "expert", "technical manager"]),
            "{context}": random.choice(["documentation", "tutorial", "troubleshooting guide"]),
            "{tone}": random.choice(["formal", "casual", "technical", "friendly"]),
            "{length}": random.choice(["brief", "detailed", "comprehensive"]),
            "{style}": random.choice(["narrative", "instructional", "conversational"]),
            "{structure}": "Introduction, Main Points, Conclusion",
            "{example}": "Step 1: ..., Step 2: ..., Step 3: ...",
        }
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result
    
    def generate_test_cases(
        self,
        prompt_file: Path,
        transformation_stage: str,
        num_samples_per_level: int = 10
    ) -> List[TestCase]:
        """Generate test cases from a prompt file"""
        # Read prompt content
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        prompt_name = prompt_file.stem
        base_instructions = self.extract_base_instructions(content, prompt_name)
        
        if not base_instructions:
            return []
        
        test_cases = []
        test_id = 0
        
        # Generate test cases for each level
        for level in TEST_CONFIG["constraint_levels"]:
            for _ in range(num_samples_per_level):
                # Select a base instruction
                base_instruction = random.choice(base_instructions)
                
                # Generate constraints for this level
                constraints = self.generate_constraints(level, base_instruction)
                
                # Create full instruction
                full_instruction = base_instruction + "\n\nConstraints:\n"
                for i, constraint in enumerate(constraints, 1):
                    full_instruction += f"{i}. {constraint.description}\n"
                
                # Create test case
                test_case = TestCase(
                    id=f"{prompt_name}_{transformation_stage}_L{level}_{test_id}",
                    level=level,
                    base_instruction=base_instruction,
                    constraints=constraints,
                    full_instruction=full_instruction,
                    source_prompt=prompt_name,
                    transformation_stage=transformation_stage
                )
                
                test_cases.append(test_case)
                test_id += 1
        
        return test_cases
    
    def save_test_cases(
        self,
        test_cases: List[TestCase],
        output_file: Path
    ):
        """Save test cases to JSON file"""
        data = [asdict(tc) for tc in test_cases]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(test_cases)} test cases to {output_file}")
    
    def generate_all_test_cases(self) -> Dict[str, List[TestCase]]:
        """Generate test cases for all prompts and transformation stages"""
        from .config import get_prompt_files, get_transformed_prompt_files, TRANSFORMATION_STAGES
        
        all_test_cases = {}
        
        for stage in TRANSFORMATION_STAGES:
            print(f"\nGenerating test cases for stage: {stage}")
            
            prompt_files = get_transformed_prompt_files(stage)
            if not prompt_files:
                print(f"  No files found for stage: {stage}")
                continue
            
            stage_test_cases = []
            for prompt_file in prompt_files:
                print(f"  Processing: {prompt_file.name}")
                test_cases = self.generate_test_cases(
                    prompt_file,
                    stage,
                    TEST_CONFIG["samples_per_level"]
                )
                stage_test_cases.extend(test_cases)
            
            all_test_cases[stage] = stage_test_cases
            
            # Save stage test cases
            output_file = FOLLOWBENCH_DATA_DIR / f"test_cases_{stage}.json"
            self.save_test_cases(stage_test_cases, output_file)
        
        return all_test_cases


if __name__ == "__main__":
    generator = FollowBenchGenerator()
    all_test_cases = generator.generate_all_test_cases()
    
    # Print summary
    print("\n" + "="*60)
    print("Test Case Generation Summary")
    print("="*60)
    for stage, test_cases in all_test_cases.items():
        print(f"{stage}: {len(test_cases)} test cases")
