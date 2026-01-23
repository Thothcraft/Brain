"""Statistical Analysis Utilities for FL Experiments.

This module provides statistical analysis functions for comparing
FL experiment results across multiple runs.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_statistics_report(
    results: List[Dict[str, Any]],
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Generate comprehensive statistical report for experiments.
    
    Args:
        results: List of experiment results with 'name', 'runs' (list of accuracy values)
        confidence_level: Confidence level for intervals (0.95 or 0.99)
    
    Returns:
        Dictionary with statistical analysis
    """
    report = {
        "confidence_level": confidence_level,
        "experiments": [],
        "comparisons": [],
    }
    
    for result in results:
        name = result.get("name", "Unknown")
        runs = result.get("runs", [])
        
        if not runs:
            continue
        
        runs = np.array(runs)
        n = len(runs)
        
        # Descriptive statistics
        stats = {
            "name": name,
            "n": n,
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)) if n > 1 else 0,
            "median": float(np.median(runs)),
            "min": float(np.min(runs)),
            "max": float(np.max(runs)),
            "range": float(np.max(runs) - np.min(runs)),
        }
        
        # Confidence interval
        if n > 1:
            ci = _compute_ci(runs, confidence_level)
            stats["ci_lower"] = ci[0]
            stats["ci_upper"] = ci[1]
            stats["margin_of_error"] = ci[2]
        
        # Coefficient of variation
        if stats["mean"] > 0:
            stats["cv"] = stats["std"] / stats["mean"]
        else:
            stats["cv"] = 0
        
        # Quartiles
        if n >= 4:
            stats["q1"] = float(np.percentile(runs, 25))
            stats["q3"] = float(np.percentile(runs, 75))
            stats["iqr"] = stats["q3"] - stats["q1"]
        
        report["experiments"].append(stats)
    
    # Pairwise comparisons
    if len(report["experiments"]) >= 2:
        for i, exp1 in enumerate(report["experiments"]):
            for j, exp2 in enumerate(report["experiments"]):
                if i >= j:
                    continue
                
                comparison = _compare_experiments(
                    results[i].get("runs", []),
                    results[j].get("runs", []),
                    exp1["name"],
                    exp2["name"],
                )
                report["comparisons"].append(comparison)
    
    # Summary
    if report["experiments"]:
        means = [e["mean"] for e in report["experiments"]]
        best_idx = np.argmax(means)
        report["summary"] = {
            "best_experiment": report["experiments"][best_idx]["name"],
            "best_mean": report["experiments"][best_idx]["mean"],
            "overall_mean": float(np.mean(means)),
            "overall_std": float(np.std(means)),
        }
    
    return report


def compute_effect_size(
    group1: List[float],
    group2: List[float],
    method: str = "cohens_d"
) -> Dict[str, float]:
    """Compute effect size between two groups.
    
    Args:
        group1: First group of values
        group2: Second group of values
        method: Effect size method ('cohens_d', 'hedges_g', 'glass_delta')
    
    Returns:
        Dictionary with effect size and interpretation
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    
    mean_diff = mean1 - mean2
    
    if method == "cohens_d":
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
    elif method == "hedges_g":
        # Cohen's d with correction for small samples
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        # Hedges' correction factor
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        effect_size = d * correction
        
    elif method == "glass_delta":
        # Use control group (group2) std
        std2 = np.sqrt(var2)
        effect_size = mean_diff / std2 if std2 > 0 else 0
        
    else:
        effect_size = 0
    
    # Interpretation (Cohen's conventions)
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        interpretation = "negligible"
    elif abs_effect < 0.5:
        interpretation = "small"
    elif abs_effect < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        "effect_size": float(effect_size),
        "method": method,
        "interpretation": interpretation,
        "mean_difference": float(mean_diff),
    }


def compute_convergence_metrics(
    curves: List[List[float]],
    threshold: float = 0.001,
) -> Dict[str, Any]:
    """Compute convergence metrics from training curves.
    
    Args:
        curves: List of accuracy curves (one per run)
        threshold: Improvement threshold for convergence detection
    
    Returns:
        Dictionary with convergence metrics
    """
    if not curves:
        return {}
    
    # Align curves
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    curves_array = np.array(curves)
    
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    
    # Find convergence round (where improvement < threshold)
    convergence_rounds = []
    for curve in curves:
        conv_round = len(curve)
        for i in range(1, len(curve)):
            if abs(curve[i] - curve[i-1]) < threshold:
                conv_round = i
                break
        convergence_rounds.append(conv_round)
    
    # Compute area under curve (normalized)
    auc_values = [np.trapz(c) / len(c) for c in curves]
    
    # Compute final improvement rate
    final_improvements = []
    for curve in curves:
        if len(curve) >= 10:
            # Average improvement in last 10 rounds
            improvements = [curve[i] - curve[i-1] for i in range(-9, 0)]
            final_improvements.append(np.mean(improvements))
    
    return {
        "mean_convergence_round": float(np.mean(convergence_rounds)),
        "std_convergence_round": float(np.std(convergence_rounds)),
        "min_convergence_round": int(np.min(convergence_rounds)),
        "max_convergence_round": int(np.max(convergence_rounds)),
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values)),
        "mean_final_improvement_rate": float(np.mean(final_improvements)) if final_improvements else 0,
        "total_improvement": float(mean_curve[-1] - mean_curve[0]),
        "final_accuracy_mean": float(mean_curve[-1]),
        "final_accuracy_std": float(std_curve[-1]),
    }


def _compute_ci(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute confidence interval.
    
    Returns:
        Tuple of (lower, upper, margin_of_error)
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # t-value approximation
    if confidence == 0.95:
        t_val = 2.0 if n < 30 else 1.96
    elif confidence == 0.99:
        t_val = 2.75 if n < 30 else 2.58
    else:
        t_val = 1.96
    
    margin = t_val * (std / np.sqrt(n))
    
    return (float(mean - margin), float(mean + margin), float(margin))


def _compare_experiments(
    runs1: List[float],
    runs2: List[float],
    name1: str,
    name2: str,
) -> Dict[str, Any]:
    """Compare two experiments statistically."""
    if not runs1 or not runs2:
        return {"error": "Empty data"}
    
    r1 = np.array(runs1)
    r2 = np.array(runs2)
    
    mean1, mean2 = np.mean(r1), np.mean(r2)
    std1, std2 = np.std(r1, ddof=1), np.std(r2, ddof=1)
    
    # Simple t-test
    n1, n2 = len(r1), len(r2)
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    
    if pooled_std > 0:
        t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
    else:
        t_stat = 0
    
    # Effect size
    effect = compute_effect_size(runs1, runs2, "cohens_d")
    
    return {
        "experiment_1": name1,
        "experiment_2": name2,
        "mean_1": float(mean1),
        "mean_2": float(mean2),
        "std_1": float(std1),
        "std_2": float(std2),
        "mean_difference": float(mean1 - mean2),
        "t_statistic": float(t_stat),
        "significant_at_95": abs(t_stat) > 2.0,
        "significant_at_99": abs(t_stat) > 2.75,
        "effect_size": effect["effect_size"],
        "effect_interpretation": effect["interpretation"],
        "winner": name1 if mean1 > mean2 else name2,
    }


def format_statistics_table(report: Dict[str, Any]) -> str:
    """Format statistics report as a readable table.
    
    Args:
        report: Statistics report from generate_statistics_report
    
    Returns:
        Formatted string table
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 80)
    
    # Experiment statistics
    lines.append("\n--- Experiment Statistics ---\n")
    lines.append(f"{'Experiment':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20} {'n':>5}")
    lines.append("-" * 70)
    
    for exp in report.get("experiments", []):
        ci_str = f"[{exp.get('ci_lower', 0):.4f}, {exp.get('ci_upper', 0):.4f}]"
        lines.append(
            f"{exp['name']:<20} {exp['mean']:>10.4f} {exp['std']:>10.4f} {ci_str:>20} {exp['n']:>5}"
        )
    
    # Comparisons
    if report.get("comparisons"):
        lines.append("\n--- Pairwise Comparisons ---\n")
        lines.append(f"{'Comparison':<30} {'Diff':>10} {'t-stat':>10} {'Effect':>10} {'Sig':>5}")
        lines.append("-" * 70)
        
        for comp in report["comparisons"]:
            comparison_name = f"{comp['experiment_1']} vs {comp['experiment_2']}"
            sig = "Yes" if comp["significant_at_95"] else "No"
            lines.append(
                f"{comparison_name:<30} {comp['mean_difference']:>10.4f} "
                f"{comp['t_statistic']:>10.2f} {comp['effect_size']:>10.2f} {sig:>5}"
            )
    
    # Summary
    if report.get("summary"):
        lines.append("\n--- Summary ---")
        lines.append(f"Best Experiment: {report['summary']['best_experiment']}")
        lines.append(f"Best Mean Accuracy: {report['summary']['best_mean']:.4f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)
