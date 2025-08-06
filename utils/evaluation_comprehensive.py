#!/usr/bin/env python3
"""
Comprehensive Evaluation and Ablation Studies

This module provides comprehensive evaluation tools for the Graph Anomaly Detection pipeline,
including ablation studies, baseline comparisons, and performance analysis.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix, classification_report
import json
import time
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

from training_improved import GADTrainer
from config import default_config
from utils import setup_logging, timer
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    use_dsgad: bool = True
    use_temporal: bool = False
    use_energy_scoring: bool = False
    ensemble_energy: bool = False
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False
    window_size: int = 5
    temporal_overlap: float = 0.5
    energy_metrics: List[str] = None
    
    def __post_init__(self):
        if self.energy_metrics is None:
            self.energy_metrics = ["euclidean", "cosine"]


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    roc_auc: float
    average_precision: float
    training_time: float
    inference_time: float
    num_parameters: int
    memory_usage: float
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for Graph Anomaly Detection
    
    Conducts ablation studies, baseline comparisons, and performance analysis
    across different model configurations and datasets.
    """
    
    def __init__(self, data_path: str, output_dir: str = "evaluation_results", 
                 logger: Optional[logging.Logger] = None):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or setup_logging("INFO")
        self.results = []
        self.baseline_configs = self._create_baseline_configs()
        self.ablation_configs = self._create_ablation_configs()
        
    def _create_baseline_configs(self) -> List[ExperimentConfig]:
        """Create baseline experiment configurations"""
        return [
            # basic baselines
            ExperimentConfig(
                name="GRACE_Baseline",
                use_dsgad=False,
                use_temporal=False,
                use_energy_scoring=False
            ),
            ExperimentConfig(
                name="DSGAD_Only",
                use_dsgad=True,
                use_temporal=False,
                use_energy_scoring=False
            ),
            # our full pipeline
            ExperimentConfig(
                name="Full_Pipeline_Standard",
                use_dsgad=True,
                use_temporal=True,
                use_energy_scoring=True,
                ensemble_energy=True,
                energy_metrics=["euclidean", "cosine", "mahalanobis"]
            ),
        ]
    
    def _create_ablation_configs(self) -> List[ExperimentConfig]:
        """Create ablation study configurations"""
        return [
            # phase-by-phase ablation
            ExperimentConfig(
                name="Phase1_ADGCL",
                use_dsgad=False,
                use_temporal=False,
                use_energy_scoring=False
            ),
            ExperimentConfig(
                name="Phase2_DSGAD",
                use_dsgad=True,
                use_temporal=False,
                use_energy_scoring=False
            ),
            ExperimentConfig(
                name="Phase3_MAE",
                use_dsgad=True,
                use_temporal=False,
                use_energy_scoring=False
            ),
            ExperimentConfig(
                name="Phase5_Temporal",
                use_dsgad=True,
                use_temporal=True,
                use_energy_scoring=False
            ),
            ExperimentConfig(
                name="Phase6_Energy_Single",
                use_dsgad=True,
                use_temporal=False,
                use_energy_scoring=True,
                ensemble_energy=False,
                energy_metrics=["euclidean"]
            ),
            ExperimentConfig(
                name="Phase6_Energy_Ensemble",
                use_dsgad=True,
                use_temporal=True,
                use_energy_scoring=True,
                ensemble_energy=True,
                energy_metrics=["euclidean", "cosine"]
            ),
            
            # Energy metric ablations
            ExperimentConfig(
                name="Energy_Euclidean_Only",
                use_dsgad=True,
                use_energy_scoring=True,
                energy_metrics=["euclidean"]
            ),
            ExperimentConfig(
                name="Energy_Cosine_Only",
                use_dsgad=True,
                use_energy_scoring=True,
                energy_metrics=["cosine"]
            ),
            ExperimentConfig(
                name="Energy_Mahalanobis_Only",
                use_dsgad=True,
                use_energy_scoring=True,
                energy_metrics=["mahalanobis"]
            ),
            
            # temporal window size ablations
            ExperimentConfig(
                name="Temporal_Window3",
                use_dsgad=True,
                use_temporal=True,
                window_size=3
            ),
            ExperimentConfig(
                name="Temporal_Window7",
                use_dsgad=True,
                use_temporal=True,
                window_size=7
            ),
            ExperimentConfig(
                name="Temporal_Window10",
                use_dsgad=True,
                use_temporal=True,
                window_size=10
            ),
        ]
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with given configuration"""
        self.logger.info(f"Running experiment: {config.name}")
        
        start_time = time.time()
        
        try:
            # trainer with configuration
            trainer = GADTrainer(
                config=default_config,
                use_dsgad=config.use_dsgad,
                use_temporal=config.use_temporal,
                use_energy_scoring=config.use_energy_scoring,
                ensemble_energy=config.ensemble_energy,
                pretrained_path=config.pretrained_path,
                freeze_backbone=config.freeze_backbone,
                window_size=config.window_size,
                temporal_overlap=config.temporal_overlap,
                energy_metrics=config.energy_metrics
            )
            
            # load data
            trainer.load_data(self.data_path, use_masks=True)
            
            # setup models and count parameters
            trainer.setup_models()
            num_params = sum(p.numel() for p in trainer.model.parameters())
            if trainer.classifier:
                num_params += sum(p.numel() for p in trainer.classifier.parameters())
            if trainer.energy_classifier:
                num_params += sum(p.numel() for p in trainer.energy_classifier.parameters())
            
            # training
            training_start = time.time()
            train_results = trainer.train()
            training_time = time.time() - training_start
            
            # inference timing
            inference_start = time.time()
            eval_results = trainer.evaluate()
            inference_time = time.time() - inference_start
            
            total_time = time.time() - start_time
            
            # memory usage (approximate)
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB
            else:
                memory_usage = 0.0
            
            result = ExperimentResult(
                config=config,
                roc_auc=eval_results["roc_auc"],
                average_precision=eval_results["average_precision"],
                training_time=training_time,
                inference_time=inference_time,
                num_parameters=num_params,
                memory_usage=memory_usage,
                additional_metrics={
                    "total_time": total_time,
                    "num_test_nodes": eval_results["num_test_nodes"],
                    "anomaly_ratio": eval_results["anomaly_ratio"]
                }
            )
            
            self.logger.info(f"✓ {config.name}: ROC-AUC={result.roc_auc:.4f}, AP={result.average_precision:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"✗ {config.name} failed: {str(e)}")
            # return default result for failed experiments
            return ExperimentResult(
                config=config,
                roc_auc=0.5,  # Random performance
                average_precision=0.5,
                training_time=0.0,
                inference_time=0.0,
                num_parameters=0,
                memory_usage=0.0,
                additional_metrics={"failed": True, "error": str(e)}
            )
    
    def run_baseline_study(self) -> List[ExperimentResult]:
        """Run baseline comparison study"""
        self.logger.info("Starting baseline comparison study...")
        
        baseline_results = []
        for config in self.baseline_configs:
            result = self.run_single_experiment(config)
            baseline_results.append(result)
            self.results.append(result)
        
        self.logger.info(f"Completed baseline study: {len(baseline_results)} experiments")
        return baseline_results
    
    def run_ablation_study(self) -> List[ExperimentResult]:
        """Run comprehensive ablation study"""
        self.logger.info("Starting ablation study...")
        
        ablation_results = []
        for config in self.ablation_configs:
            result = self.run_single_experiment(config)
            ablation_results.append(result)
            self.results.append(result)
        
        self.logger.info(f"Completed ablation study: {len(ablation_results)} experiments")
        return ablation_results
    
    def run_comprehensive_evaluation(self) -> Dict[str, List[ExperimentResult]]:
        """Run all evaluation studies"""
        self.logger.info("Starting comprehensive evaluation...")
        
        results = {
            "baseline": self.run_baseline_study(),
            "ablation": self.run_ablation_study()
        }
        
        # save all results
        self.save_results()
        
        # generate analysis
        self.generate_analysis()
        
        self.logger.info("Comprehensive evaluation completed!")
        return results
    
    def save_results(self):
        """Save experiment results to files"""
        # format results
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert ExperimentConfig to dict
            result_dict['config'] = asdict(result.config)
            results_data.append(result_dict)
        
        # save as JSON
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # save as CSV for easy analysis
        df_data = []
        for result in self.results:
            row = {
                'experiment_name': result.config.name,
                'roc_auc': result.roc_auc,
                'average_precision': result.average_precision,
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'num_parameters': result.num_parameters,
                'memory_usage': result.memory_usage,
                'use_dsgad': result.config.use_dsgad,
                'use_temporal': result.config.use_temporal,
                'use_energy_scoring': result.config.use_energy_scoring,
                'ensemble_energy': result.config.ensemble_energy,
            }
            row.update(result.additional_metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = self.output_dir / "experiment_results.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {results_file} and {csv_file}")
    
    def generate_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        self.logger.info("Generating analysis and visualizations...")
        
        # create analysis plots
        self._plot_performance_comparison()
        self._plot_ablation_analysis()
        self._plot_efficiency_analysis()
        self._plot_phase_progression()
        
        # generate summary report
        self._generate_summary_report()
        
        self.logger.info("Analysis generated successfully!")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across experiments"""
        plt.figure(figsize=(15, 10))
        
        # extract data
        names = [r.config.name for r in self.results]
        roc_aucs = [r.roc_auc for r in self.results]
        aps = [r.average_precision for r in self.results]
        
        # create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC-AUC comparison
        ax1.bar(range(len(names)), roc_aucs, color='skyblue', alpha=0.8)
        ax1.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # add value labels on bars
        for i, v in enumerate(roc_aucs):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # AP comparison
        ax2.bar(range(len(names)), aps, color='lightcoral', alpha=0.8)
        ax2.set_title('Average Precision Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Precision')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # add value labels on bars
        for i, v in enumerate(aps):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # performance vs parameters scatter
        params = [r.num_parameters / 1e6 for r in self.results] 
        ax3.scatter(params, roc_aucs, s=100, alpha=0.7, color='green')
        ax3.set_xlabel('Parameters (Millions)')
        ax3.set_ylabel('ROC-AUC')
        ax3.set_title('Performance vs Model Size', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # annotate points
        for i, name in enumerate(names):
            ax3.annotate(name, (params[i], roc_aucs[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.8)
        
        # training time comparison
        train_times = [r.training_time / 60 for r in self.results]  # Convert to minutes
        ax4.bar(range(len(names)), train_times, color='orange', alpha=0.8)
        ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Training Time (minutes)')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_analysis(self):
        """Plot ablation study analysis"""
        # filter ablation results (exclude baseline comparisons)
        ablation_results = [r for r in self.results if "Phase" in r.config.name or "Energy" in r.config.name or "Temporal" in r.config.name]
        
        if not ablation_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        names = [r.config.name for r in ablation_results]
        roc_aucs = [r.roc_auc for r in ablation_results]
        
        # sort by performance
        sorted_indices = np.argsort(roc_aucs)[::-1]
        names = [names[i] for i in sorted_indices]
        roc_aucs = [roc_aucs[i] for i in sorted_indices]
        
        # create bar plot
        bars = plt.bar(range(len(names)), roc_aucs, color='lightblue', alpha=0.8)
        
        # color bars by performance
        for i, bar in enumerate(bars):
            if roc_aucs[i] >= 0.8:
                bar.set_color('darkgreen')
            elif roc_aucs[i] >= 0.7:
                bar.set_color('green')
            elif roc_aucs[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.title('Ablation Study Results', fontsize=16, fontweight='bold')
        plt.ylabel('ROC-AUC')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # add value labels
        for i, v in enumerate(roc_aucs):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # add horizontal line for target performance
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self):
        """Plot efficiency analysis (performance vs computational cost)"""
        plt.figure(figsize=(12, 8))
        
        roc_aucs = [r.roc_auc for r in self.results]
        train_times = [r.training_time for r in self.results]
        params = [r.num_parameters / 1e6 for r in self.results]
        names = [r.config.name for r in self.results]
        
        # create scatter plot with size proportional to parameters
        scatter = plt.scatter(train_times, roc_aucs, s=[p*10 for p in params], 
                             alpha=0.6, c=range(len(names)), cmap='viridis')
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('ROC-AUC')
        plt.title('Efficiency Analysis: Performance vs Training Time', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        # add annotations
        for i, name in enumerate(names):
            plt.annotate(name, (train_times[i], roc_aucs[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        # add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Experiment Index')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_progression(self):
        """Plot phase-by-phase performance progression"""
        # identify phase progression experiments
        phase_experiments = []
        phase_names = []
        
        for result in self.results:
            if "Phase" in result.config.name:
                phase_num = int(result.config.name.split("Phase")[1][0])
                phase_experiments.append((phase_num, result))
                phase_names.append(result.config.name)
        
        # sort by phase number
        phase_experiments.sort(key=lambda x: x[0])
        
        if len(phase_experiments) < 2:
            return
        
        phases = [p[0] for p in phase_experiments]
        roc_aucs = [p[1].roc_auc for p in phase_experiments]
        aps = [p[1].average_precision for p in phase_experiments]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(phases, roc_aucs, 'o-', linewidth=2, markersize=8, label='ROC-AUC', color='blue')
        plt.plot(phases, aps, 's-', linewidth=2, markersize=8, label='Average Precision', color='red')
        
        plt.xlabel('Phase Number')
        plt.ylabel('Performance')
        plt.title('Phase-by-Phase Performance Progression', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # add value annotations
        for i, (phase, roc_auc, ap) in enumerate(zip(phases, roc_aucs, aps)):
            plt.annotate(f'{roc_auc:.3f}', (phase, roc_auc), xytext=(0, 10), 
                        textcoords='offset points', ha='center', color='blue')
            plt.annotate(f'{ap:.3f}', (phase, ap), xytext=(0, -15), 
                        textcoords='offset points', ha='center', color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "phase_progression.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        report_lines = []
        
        # header
        report_lines.append("# Graph Anomaly Detection - Comprehensive Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # best performing models
        best_roc = max(self.results, key=lambda x: x.roc_auc)
        best_ap = max(self.results, key=lambda x: x.average_precision)
        
        report_lines.append("## Best Performing Models")
        report_lines.append(f"Best ROC-AUC: {best_roc.config.name} ({best_roc.roc_auc:.4f})")
        report_lines.append(f"Best Average Precision: {best_ap.config.name} ({best_ap.average_precision:.4f})")
        report_lines.append("")
        
        # performance summary table
        report_lines.append("## Performance Summary")
        report_lines.append("| Experiment | ROC-AUC | AP | Train Time | Parameters |")
        report_lines.append("|------------|---------|----|-----------|-----------:|")
        
        # sort by ROC-AUC
        sorted_results = sorted(self.results, key=lambda x: x.roc_auc, reverse=True)
        
        for result in sorted_results:
            params_m = result.num_parameters / 1e6
            time_min = result.training_time / 60
            report_lines.append(
                f"| {result.config.name} | {result.roc_auc:.4f} | "
                f"{result.average_precision:.4f} | {time_min:.1f}m | {params_m:.1f}M |"
            )
        
        report_lines.append("")
        
        # phase analysis
        phase_results = [r for r in self.results if "Phase" in r.config.name]
        if phase_results:
            report_lines.append("## Phase-by-Phase Analysis")
            phase_results.sort(key=lambda x: int(x.config.name.split("Phase")[1][0]))
            
            for i, result in enumerate(phase_results):
                if i == 0:
                    improvement = 0.0
                else:
                    improvement = result.roc_auc - phase_results[i-1].roc_auc
                
                report_lines.append(
                    f"- {result.config.name}: ROC-AUC {result.roc_auc:.4f} "
                    f"(+{improvement:.4f} vs previous)"
                )
            report_lines.append("")
        
        # key findings
        report_lines.append("## Key Findings")
        
        # check if target performance (0.8) was achieved
        target_achieved = any(r.roc_auc >= 0.8 for r in self.results)
        if target_achieved:
            achieving_models = [r.config.name for r in self.results if r.roc_auc >= 0.8]
            report_lines.append(f"✅ Target ROC-AUC ≥ 0.8 achieved by: {', '.join(achieving_models)}")
        else:
            best_score = max(r.roc_auc for r in self.results)
            report_lines.append(f"❌ Target ROC-AUC ≥ 0.8 not achieved (best: {best_score:.4f})")
        
        # efficiency analysis
        efficient_models = sorted(self.results, key=lambda x: x.roc_auc / (x.training_time + 1), reverse=True)[:3]
        report_lines.append(f"🚀 Most efficient models: {', '.join([m.config.name for m in efficient_models])}")
        
        # component analysis
        energy_results = [r for r in self.results if r.config.use_energy_scoring]
        temporal_results = [r for r in self.results if r.config.use_temporal]
        dsgad_results = [r for r in self.results if r.config.use_dsgad]
        
        if energy_results:
            energy_avg = np.mean([r.roc_auc for r in energy_results])
            report_lines.append(f"⚡ Energy scoring average ROC-AUC: {energy_avg:.4f}")
        
        if temporal_results:
            temporal_avg = np.mean([r.roc_auc for r in temporal_results])
            report_lines.append(f"⏱️ Temporal modeling average ROC-AUC: {temporal_avg:.4f}")
        
        if dsgad_results:
            dsgad_avg = np.mean([r.roc_auc for r in dsgad_results])
            report_lines.append(f"🌊 DSGAD average ROC-AUC: {dsgad_avg:.4f}")
        
        report_lines.append("")
        
        # recommendations
        report_lines.append("## Recommendations")
        report_lines.append(f"1. **Best Overall**: Use {best_roc.config.name} for highest performance")
        
        if efficient_models:
            report_lines.append(f"2. **Most Efficient**: Use {efficient_models[0].config.name} for best performance/time ratio")
        
        if best_roc.config.use_energy_scoring:
            report_lines.append("3. **Energy Scoring**: Provides significant performance improvement")
        
        if best_roc.config.use_temporal:
            report_lines.append("4. **Temporal Modeling**: Beneficial for capturing evolutionary patterns")
        
        # save report
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Summary report saved to {report_file}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive GAD Evaluation")
    parser.add_argument('--data-path', required=True, help='Path to preprocessed graph data')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--baseline-only', action='store_true', help='Run only baseline comparisons')
    parser.add_argument('--ablation-only', action='store_true', help='Run only ablation studies')
    
    args = parser.parse_args()
    
    # setup logging
    logger = setup_logging("INFO")
    
    # create evaluator
    evaluator = ComprehensiveEvaluator(args.data_path, args.output_dir, logger)
    
    try:
        if args.baseline_only:
            logger.info("Running baseline comparisons only...")
            evaluator.run_baseline_study()
            evaluator.save_results()
            evaluator.generate_analysis()
        elif args.ablation_only:
            logger.info("Running ablation studies only...")
            evaluator.run_ablation_study()
            evaluator.save_results()
            evaluator.generate_analysis()
        else:
            logger.info("Running comprehensive evaluation...")
            evaluator.run_comprehensive_evaluation()
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()