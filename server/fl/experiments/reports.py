"""Report Generation for FL Experiments.

This module provides functions to generate:
- Per-model experiment reports
- Comparative reports across experiments
- Statistical summaries with confidence intervals
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np

from .runner import ExperimentResult, RunResult

logger = logging.getLogger(__name__)


def generate_experiment_report(result: ExperimentResult) -> Dict[str, Any]:
    """Generate a detailed report for a single experiment.
    
    Args:
        result: ExperimentResult from experiment runner
    
    Returns:
        Dictionary containing detailed experiment report
    """
    config = result.config
    
    report = {
        "experiment_id": result.experiment_id,
        "name": config.name,
        "timestamp": datetime.now().isoformat(),
        
        # Configuration summary
        "configuration": {
            "algorithm": config.algorithm.value,
            "model": config.model.value,
            "dataset": config.data.dataset.value,
            "num_partitions": config.data.num_partitions,
            "partition_strategy": config.data.partition_strategy.value,
            "num_rounds": config.server.num_rounds,
            "local_epochs": config.client.local_epochs,
            "learning_rate": config.client.learning_rate,
            "batch_size": config.client.local_batch_size,
        },
        
        # Run summary
        "runs": {
            "total": len(result.runs),
            "successful": len([r for r in result.runs if r.status == "completed"]),
            "failed": len([r for r in result.runs if r.status == "failed"]),
        },
        
        # Accuracy statistics
        "accuracy": {
            "mean": result.mean_accuracy,
            "std": result.std_accuracy,
            "min": result.min_accuracy,
            "max": result.max_accuracy,
            "confidence_interval_95": _compute_confidence_interval(
                [r.final_accuracy for r in result.runs if r.status == "completed"],
                confidence=0.95
            ),
        },
        
        # Best accuracy statistics
        "best_accuracy": {
            "mean": result.mean_best_accuracy,
            "std": result.std_best_accuracy,
        },
        
        # Timing
        "timing": {
            "total_seconds": result.total_time_seconds,
            "avg_run_seconds": result.total_time_seconds / len(result.runs) if result.runs else 0,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        },
        
        # Per-run details
        "run_details": [
            {
                "run_id": run.run_id,
                "seed": run.seed,
                "final_accuracy": run.final_accuracy,
                "best_accuracy": run.best_accuracy,
                "best_round": run.best_round,
                "training_time": run.training_time_seconds,
                "status": run.status,
                "error": run.error_message,
            }
            for run in result.runs
        ],
        
        # Convergence analysis
        "convergence": _analyze_convergence(result),
    }
    
    return report


def generate_comparative_report(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Generate a comparative report across multiple experiments.
    
    Args:
        results: List of ExperimentResult from multiple experiments
    
    Returns:
        Dictionary containing comparative analysis
    """
    if not results:
        return {"error": "No results to compare"}
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        
        # Summary table
        "summary": [
            {
                "name": r.config.name,
                "algorithm": r.config.algorithm.value,
                "model": r.config.model.value,
                "mean_accuracy": r.mean_accuracy,
                "std_accuracy": r.std_accuracy,
                "best_accuracy": r.mean_best_accuracy,
                "num_runs": len(r.runs),
            }
            for r in results
        ],
        
        # Rankings
        "rankings": {
            "by_mean_accuracy": _rank_by_metric(results, "mean_accuracy"),
            "by_best_accuracy": _rank_by_metric(results, "mean_best_accuracy"),
            "by_consistency": _rank_by_consistency(results),
        },
        
        # Statistical comparisons
        "statistical_tests": _compute_statistical_tests(results),
        
        # Best performer
        "best_performer": _get_best_performer(results),
        
        # Recommendations
        "recommendations": _generate_recommendations(results),
    }
    
    return report


def generate_statistical_summary(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Generate statistical summary with detailed analysis.
    
    Args:
        results: List of ExperimentResult
    
    Returns:
        Dictionary with statistical analysis
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
    }
    
    for result in results:
        if not result.runs:
            continue
        
        accuracies = [r.final_accuracy for r in result.runs if r.status == "completed"]
        best_accs = [r.best_accuracy for r in result.runs if r.status == "completed"]
        
        if not accuracies:
            continue
        
        exp_stats = {
            "name": result.config.name,
            "algorithm": result.config.algorithm.value,
            "model": result.config.model.value,
            
            # Descriptive statistics
            "final_accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "median": float(np.median(accuracies)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "range": float(np.max(accuracies) - np.min(accuracies)),
                "iqr": float(np.percentile(accuracies, 75) - np.percentile(accuracies, 25)) if len(accuracies) >= 4 else 0,
                "ci_95": _compute_confidence_interval(accuracies, 0.95),
                "ci_99": _compute_confidence_interval(accuracies, 0.99),
            },
            
            "best_accuracy": {
                "mean": float(np.mean(best_accs)),
                "std": float(np.std(best_accs)),
                "median": float(np.median(best_accs)),
            },
            
            # Sample size
            "n_runs": len(accuracies),
            
            # Coefficient of variation (consistency measure)
            "cv": float(np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0,
        }
        
        summary["experiments"].append(exp_stats)
    
    # Cross-experiment analysis
    if len(summary["experiments"]) > 1:
        all_means = [e["final_accuracy"]["mean"] for e in summary["experiments"]]
        summary["cross_experiment"] = {
            "mean_of_means": float(np.mean(all_means)),
            "std_of_means": float(np.std(all_means)),
            "best_experiment": summary["experiments"][np.argmax(all_means)]["name"],
            "worst_experiment": summary["experiments"][np.argmin(all_means)]["name"],
        }
    
    return summary


def _compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """Compute confidence interval for a list of values."""
    if not values or len(values) < 2:
        return {"lower": 0, "upper": 0, "margin": 0}
    
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # t-value for confidence level (approximation for small samples)
    if confidence == 0.95:
        t_value = 2.0 if n < 30 else 1.96
    elif confidence == 0.99:
        t_value = 2.75 if n < 30 else 2.58
    else:
        t_value = 1.96
    
    margin = t_value * (std / np.sqrt(n))
    
    return {
        "lower": float(mean - margin),
        "upper": float(mean + margin),
        "margin": float(margin),
    }


def _analyze_convergence(result: ExperimentResult) -> Dict[str, Any]:
    """Analyze convergence behavior across runs."""
    if not result.runs:
        return {}
    
    # Get accuracy curves from all runs
    curves = []
    for run in result.runs:
        if run.status == "completed" and run.round_metrics:
            curve = [m.accuracy for m in run.round_metrics]
            curves.append(curve)
    
    if not curves:
        return {}
    
    # Find minimum length
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    
    # Compute mean and std at each round
    curves_array = np.array(curves)
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    
    # Find convergence point (where improvement < threshold)
    threshold = 0.001
    convergence_round = min_len
    for i in range(1, min_len):
        if abs(mean_curve[i] - mean_curve[i-1]) < threshold:
            convergence_round = i
            break
    
    return {
        "mean_curve": mean_curve.tolist(),
        "std_curve": std_curve.tolist(),
        "convergence_round": convergence_round,
        "final_improvement_rate": float(mean_curve[-1] - mean_curve[-2]) if min_len > 1 else 0,
        "total_improvement": float(mean_curve[-1] - mean_curve[0]) if min_len > 0 else 0,
    }


def _rank_by_metric(results: List[ExperimentResult], metric: str) -> List[Dict[str, Any]]:
    """Rank experiments by a specific metric."""
    ranked = sorted(
        results,
        key=lambda r: getattr(r, metric, 0),
        reverse=True
    )
    
    return [
        {
            "rank": i + 1,
            "name": r.config.name,
            "value": getattr(r, metric, 0),
        }
        for i, r in enumerate(ranked)
    ]


def _rank_by_consistency(results: List[ExperimentResult]) -> List[Dict[str, Any]]:
    """Rank experiments by consistency (lower std = more consistent)."""
    ranked = sorted(
        results,
        key=lambda r: r.std_accuracy if r.std_accuracy > 0 else float('inf')
    )
    
    return [
        {
            "rank": i + 1,
            "name": r.config.name,
            "std": r.std_accuracy,
        }
        for i, r in enumerate(ranked)
    ]


def _compute_statistical_tests(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Compute statistical significance tests between experiments."""
    if len(results) < 2:
        return {"message": "Need at least 2 experiments for comparison"}
    
    tests = {
        "pairwise_comparisons": [],
    }
    
    # Pairwise t-tests (simplified)
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i >= j:
                continue
            
            acc1 = [r.final_accuracy for r in r1.runs if r.status == "completed"]
            acc2 = [r.final_accuracy for r in r2.runs if r.status == "completed"]
            
            if len(acc1) < 2 or len(acc2) < 2:
                continue
            
            # Simple t-test approximation
            mean_diff = np.mean(acc1) - np.mean(acc2)
            pooled_std = np.sqrt((np.var(acc1) + np.var(acc2)) / 2)
            
            if pooled_std > 0:
                t_stat = mean_diff / (pooled_std * np.sqrt(2 / len(acc1)))
                significant = abs(t_stat) > 2.0  # Rough threshold
            else:
                t_stat = 0
                significant = False
            
            tests["pairwise_comparisons"].append({
                "experiment_1": r1.config.name,
                "experiment_2": r2.config.name,
                "mean_difference": float(mean_diff),
                "t_statistic": float(t_stat),
                "significant_at_95": significant,
                "winner": r1.config.name if mean_diff > 0 else r2.config.name,
            })
    
    return tests


def _get_best_performer(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Identify the best performing experiment."""
    if not results:
        return {}
    
    best = max(results, key=lambda r: r.mean_accuracy)
    
    return {
        "name": best.config.name,
        "algorithm": best.config.algorithm.value,
        "model": best.config.model.value,
        "mean_accuracy": best.mean_accuracy,
        "std_accuracy": best.std_accuracy,
        "configuration": best.config.to_dict(),
    }


def _generate_recommendations(results: List[ExperimentResult]) -> List[str]:
    """Generate recommendations based on experiment results."""
    recommendations = []
    
    if not results:
        return ["No experiments to analyze"]
    
    # Find best performer
    best = max(results, key=lambda r: r.mean_accuracy)
    recommendations.append(
        f"Best performing configuration: {best.config.name} "
        f"({best.config.algorithm.value} + {best.config.model.value}) "
        f"with {best.mean_accuracy:.4f} accuracy"
    )
    
    # Check for high variance
    high_variance = [r for r in results if r.std_accuracy > 0.02]
    if high_variance:
        names = ", ".join(r.config.name for r in high_variance)
        recommendations.append(
            f"High variance detected in: {names}. "
            "Consider more runs or hyperparameter tuning."
        )
    
    # Check for non-IID impact
    noniid_results = [r for r in results 
                      if "dirichlet" in r.config.data.partition_strategy.value.lower()
                      or "pathological" in r.config.data.partition_strategy.value.lower()]
    iid_results = [r for r in results 
                   if r.config.data.partition_strategy.value.lower() == "iid"]
    
    if noniid_results and iid_results:
        noniid_mean = np.mean([r.mean_accuracy for r in noniid_results])
        iid_mean = np.mean([r.mean_accuracy for r in iid_results])
        gap = iid_mean - noniid_mean
        
        if gap > 0.05:
            recommendations.append(
                f"Significant IID vs Non-IID gap ({gap:.4f}). "
                "Consider FedProx or other heterogeneity-robust algorithms."
            )
    
    # Check for algorithm comparison
    algorithms = set(r.config.algorithm.value for r in results)
    if len(algorithms) > 1:
        algo_perf = {}
        for r in results:
            algo = r.config.algorithm.value
            if algo not in algo_perf:
                algo_perf[algo] = []
            algo_perf[algo].append(r.mean_accuracy)
        
        best_algo = max(algo_perf.items(), key=lambda x: np.mean(x[1]))
        recommendations.append(
            f"Best algorithm overall: {best_algo[0]} "
            f"(avg accuracy: {np.mean(best_algo[1]):.4f})"
        )
    
    return recommendations


def format_report_as_text(report: Dict[str, Any]) -> str:
    """Format a report dictionary as readable text."""
    lines = []
    
    lines.append("=" * 60)
    lines.append(f"FL EXPERIMENT REPORT")
    lines.append("=" * 60)
    
    if "name" in report:
        lines.append(f"\nExperiment: {report['name']}")
    
    if "configuration" in report:
        lines.append("\n--- Configuration ---")
        for key, value in report["configuration"].items():
            lines.append(f"  {key}: {value}")
    
    if "accuracy" in report:
        lines.append("\n--- Accuracy ---")
        acc = report["accuracy"]
        lines.append(f"  Mean: {acc['mean']:.4f} ± {acc['std']:.4f}")
        lines.append(f"  Range: [{acc['min']:.4f}, {acc['max']:.4f}]")
        if "confidence_interval_95" in acc:
            ci = acc["confidence_interval_95"]
            lines.append(f"  95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
    if "runs" in report:
        lines.append("\n--- Runs ---")
        runs = report["runs"]
        lines.append(f"  Total: {runs['total']}, Successful: {runs['successful']}, Failed: {runs['failed']}")
    
    if "timing" in report:
        lines.append("\n--- Timing ---")
        lines.append(f"  Total time: {report['timing']['total_seconds']:.1f}s")
    
    if "recommendations" in report:
        lines.append("\n--- Recommendations ---")
        for rec in report["recommendations"]:
            lines.append(f"  • {rec}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
