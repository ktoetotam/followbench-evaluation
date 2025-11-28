"""
Aggregate and analyze FollowBench evaluation results
Calculate metrics and generate comparison reports
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import pandas as pd

from .config import RESULTS_DIR, TRANSFORMATION_STAGES, METRICS


class ResultsAnalyzer:
    """Analyze and compare FollowBench evaluation results"""
    
    def __init__(self):
        self.results = {}
    
    def load_evaluations(self, evaluation_file: Path) -> List[Dict]:
        """Load evaluation results from JSON file"""
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_metrics(self, evaluations: List[Dict]) -> Dict:
        """Calculate HSR, SSR, and CSL metrics"""
        if not evaluations:
            return {
                "HSR": 0.0,
                "SSR": 0.0,
                "CSL": 0,
                "total_cases": 0,
                "by_level": {},
                "by_constraint_type": {}
            }
        
        # Overall metrics
        total_hsr = sum(e["hsr"] for e in evaluations)
        total_ssr = sum(e["ssr"] for e in evaluations)
        
        hsr = total_hsr / len(evaluations)
        ssr = total_ssr / len(evaluations)
        
        # Calculate CSL (Consistent Satisfaction Levels)
        level_satisfaction = defaultdict(list)
        for e in evaluations:
            level_satisfaction[e["level"]].append(e["hsr"])
        
        # CSL is the highest level where average HSR >= 0.5
        csl = 0
        for level in sorted(level_satisfaction.keys()):
            avg_hsr_at_level = sum(level_satisfaction[level]) / len(level_satisfaction[level])
            if avg_hsr_at_level >= 0.5:
                csl = level
            else:
                break
        
        # Metrics by level
        by_level = {}
        for level, hsrs in level_satisfaction.items():
            by_level[level] = {
                "HSR": sum(hsrs) / len(hsrs),
                "count": len(hsrs)
            }
        
        # Metrics by constraint type
        by_constraint_type = defaultdict(lambda: {"satisfied": 0, "total": 0})
        for e in evaluations:
            for ce in e["constraint_evaluations"]:
                by_constraint_type[ce["type"]]["total"] += 1
                if ce["satisfied"]:
                    by_constraint_type[ce["type"]]["satisfied"] += 1
        
        # Calculate satisfaction rates per constraint type
        constraint_metrics = {}
        for ctype, counts in by_constraint_type.items():
            constraint_metrics[ctype] = {
                "satisfaction_rate": counts["satisfied"] / counts["total"] if counts["total"] > 0 else 0.0,
                "count": counts["total"]
            }
        
        return {
            "HSR": hsr,
            "SSR": ssr,
            "CSL": csl,
            "total_cases": len(evaluations),
            "by_level": by_level,
            "by_constraint_type": constraint_metrics
        }
    
    def compare_stages(
        self,
        evaluation_files: Dict[str, Path]
    ) -> pd.DataFrame:
        """Compare results across transformation stages"""
        
        results_by_stage = {}
        
        for stage, eval_file in evaluation_files.items():
            if not eval_file.exists():
                continue
            
            evaluations = self.load_evaluations(eval_file)
            metrics = self.calculate_metrics(evaluations)
            results_by_stage[stage] = metrics
        
        # Create comparison dataframe
        comparison_data = []
        for stage, metrics in results_by_stage.items():
            row = {
                "Stage": stage,
                "HSR": f"{metrics['HSR']:.4f}",
                "SSR": f"{metrics['SSR']:.4f}",
                "CSL": metrics["CSL"],
                "Total Cases": metrics["total_cases"]
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_report(
        self,
        evaluation_files: Dict[str, Path],
        output_file: Path
    ):
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("="*80)
        report.append("FollowBench Evaluation Report")
        report.append("="*80)
        report.append("")
        
        # Overall comparison
        comparison_df = self.compare_stages(evaluation_files)
        report.append("## Overall Comparison Across Stages")
        report.append("")
        report.append(comparison_df.to_string(index=False))
        report.append("")
        report.append("")
        
        # Detailed analysis per stage
        for stage, eval_file in evaluation_files.items():
            if not eval_file.exists():
                continue
            
            report.append(f"## Detailed Analysis: {stage}")
            report.append("-"*80)
            
            evaluations = self.load_evaluations(eval_file)
            metrics = self.calculate_metrics(evaluations)
            
            report.append(f"Total Test Cases: {metrics['total_cases']}")
            report.append(f"Hard Satisfaction Rate (HSR): {metrics['HSR']:.4f}")
            report.append(f"Soft Satisfaction Rate (SSR): {metrics['SSR']:.4f}")
            report.append(f"Consistent Satisfaction Levels (CSL): {metrics['CSL']}")
            report.append("")
            
            # By level
            report.append("### Performance by Constraint Level")
            for level in sorted(metrics['by_level'].keys()):
                level_data = metrics['by_level'][level]
                report.append(f"  Level {level}: HSR={level_data['HSR']:.4f} (n={level_data['count']})")
            report.append("")
            
            # By constraint type
            report.append("### Performance by Constraint Type")
            for ctype in sorted(metrics['by_constraint_type'].keys()):
                type_data = metrics['by_constraint_type'][ctype]
                report.append(f"  {ctype}: {type_data['satisfaction_rate']:.4f} (n={type_data['count']})")
            report.append("")
            report.append("")
        
        # Insights
        report.append("## Key Insights")
        report.append("-"*80)
        
        # Find best performing stage
        best_stage = None
        best_hsr = 0.0
        for stage, eval_file in evaluation_files.items():
            if not eval_file.exists():
                continue
            evaluations = self.load_evaluations(eval_file)
            metrics = self.calculate_metrics(evaluations)
            if metrics["HSR"] > best_hsr:
                best_hsr = metrics["HSR"]
                best_stage = stage
        
        if best_stage:
            report.append(f"• Best performing stage: {best_stage} (HSR: {best_hsr:.4f})")
        
        report.append("")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {output_file}")
        print("\n" + report_text)
        
        return report_text
    
    def export_to_csv(
        self,
        evaluation_files: Dict[str, Path],
        output_dir: Path
    ):
        """Export results to CSV files for further analysis"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Overall metrics
        comparison_df = self.compare_stages(evaluation_files)
        comparison_df.to_csv(output_dir / "overall_comparison.csv", index=False)
        
        # Detailed results per stage
        for stage, eval_file in evaluation_files.items():
            if not eval_file.exists():
                continue
            
            evaluations = self.load_evaluations(eval_file)
            
            # Create detailed dataframe
            rows = []
            for e in evaluations:
                for ce in e["constraint_evaluations"]:
                    rows.append({
                        "test_case_id": e["test_case_id"],
                        "model": e["model_name"],
                        "stage": stage,
                        "level": e["level"],
                        "constraint_type": ce["type"],
                        "constraint_description": ce["description"],
                        "satisfied": ce["satisfied"],
                        "confidence": ce["confidence"]
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_dir / f"detailed_{stage}.csv", index=False)
        
        print(f"\nCSV files exported to: {output_dir}")


def run_analysis_pipeline(
    evaluation_files: Dict[str, Path],
    output_dir: Path
):
    """Run complete analysis pipeline"""
    print(f"\n{'='*60}")
    print(f"Running Analysis Pipeline")
    print(f"{'='*60}\n")
    
    analyzer = ResultsAnalyzer()
    
    # Generate report
    report_file = output_dir / "followbench_report.txt"
    analyzer.generate_report(evaluation_files, report_file)
    
    # Export to CSV
    analyzer.export_to_csv(evaluation_files, output_dir / "csv")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from .config import RESULTS_DIR
    
    # Find all evaluation files
    evaluation_files = {}
    for eval_file in RESULTS_DIR.glob("evaluations_*.json"):
        # Extract stage from filename
        stage = eval_file.stem.replace("evaluations_", "").replace("gpt-4_", "")
        evaluation_files[stage] = eval_file
    
    if not evaluation_files:
        print("No evaluation files found in", RESULTS_DIR)
    else:
        run_analysis_pipeline(evaluation_files, RESULTS_DIR)
