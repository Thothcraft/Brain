"""Statistical analysis utilities for multi-trial experiments.

Provides statistical tools for:
- Confidence interval computation
- Significance testing (t-test, Wilcoxon, etc.)
- Effect size calculation (Cohen's d, etc.)
- Bootstrap analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some statistical functions will be limited")


@dataclass
class StatisticalSummary:
    """Summary statistics for a metric."""
    mean: float
    std: float
    median: float
    min: float
    max: float
    n: int
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    se: float  # Standard error


@dataclass
class SignificanceTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    interpretation: str = ""


class StatisticalAnalyzer:
    """Statistical analysis for ML/DL/FL experiments."""
    
    @classmethod
    def compute_summary(cls, values: List[float], 
                       confidence: float = 0.95) -> StatisticalSummary:
        """Compute summary statistics for a list of values.
        
        Args:
            values: List of metric values
            confidence: Confidence level for CI (default 0.95)
        
        Returns:
            StatisticalSummary object
        """
        values = np.array(values)
        n = len(values)
        
        if n == 0:
            return StatisticalSummary(
                mean=0, std=0, median=0, min=0, max=0,
                n=0, ci_lower=0, ci_upper=0, se=0
            )
        
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        
        # Confidence interval
        if SCIPY_AVAILABLE and n > 1:
            ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
            ci_lower, ci_upper = ci
        else:
            # Fallback: use z-score approximation
            z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
            ci_lower = mean - z * se
            ci_upper = mean + z * se
        
        return StatisticalSummary(
            mean=float(mean),
            std=float(std),
            median=float(np.median(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            n=n,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            se=float(se)
        )
    
    @classmethod
    def paired_t_test(cls, values1: List[float], values2: List[float],
                     alpha: float = 0.05) -> SignificanceTestResult:
        """Perform paired t-test between two sets of values.
        
        Args:
            values1: First set of values
            values2: Second set of values
            alpha: Significance level
        
        Returns:
            SignificanceTestResult
        """
        if not SCIPY_AVAILABLE:
            return SignificanceTestResult(
                test_name="Paired t-test",
                statistic=0, p_value=1.0,
                is_significant=False, alpha=alpha,
                interpretation="scipy not available"
            )
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        if len(values1) != len(values2):
            raise ValueError("Paired t-test requires equal length arrays")
        
        statistic, p_value = stats.ttest_rel(values1, values2)
        
        # Cohen's d for paired samples
        diff = values1 - values2
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        is_significant = p_value < alpha
        
        interpretation = cls._interpret_effect_size(effect_size, "Cohen's d")
        if is_significant:
            interpretation = f"Significant difference (p={p_value:.4f}). {interpretation}"
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"
        
        return SignificanceTestResult(
            test_name="Paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            effect_size=float(effect_size),
            effect_size_name="Cohen's d",
            interpretation=interpretation
        )
    
    @classmethod
    def independent_t_test(cls, values1: List[float], values2: List[float],
                          alpha: float = 0.05) -> SignificanceTestResult:
        """Perform independent samples t-test.
        
        Args:
            values1: First group values
            values2: Second group values
            alpha: Significance level
        
        Returns:
            SignificanceTestResult
        """
        if not SCIPY_AVAILABLE:
            return SignificanceTestResult(
                test_name="Independent t-test",
                statistic=0, p_value=1.0,
                is_significant=False, alpha=alpha,
                interpretation="scipy not available"
            )
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        statistic, p_value = stats.ttest_ind(values1, values2)
        
        # Cohen's d for independent samples
        pooled_std = np.sqrt(
            ((len(values1) - 1) * np.var(values1, ddof=1) + 
             (len(values2) - 1) * np.var(values2, ddof=1)) /
            (len(values1) + len(values2) - 2)
        )
        effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
        
        is_significant = p_value < alpha
        
        interpretation = cls._interpret_effect_size(effect_size, "Cohen's d")
        if is_significant:
            interpretation = f"Significant difference (p={p_value:.4f}). {interpretation}"
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"
        
        return SignificanceTestResult(
            test_name="Independent t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            effect_size=float(effect_size),
            effect_size_name="Cohen's d",
            interpretation=interpretation
        )
    
    @classmethod
    def wilcoxon_test(cls, values1: List[float], values2: List[float],
                     alpha: float = 0.05) -> SignificanceTestResult:
        """Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            values1: First set of values
            values2: Second set of values
            alpha: Significance level
        
        Returns:
            SignificanceTestResult
        """
        if not SCIPY_AVAILABLE:
            return SignificanceTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0, p_value=1.0,
                is_significant=False, alpha=alpha,
                interpretation="scipy not available"
            )
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        try:
            statistic, p_value = stats.wilcoxon(values1, values2)
        except ValueError as e:
            return SignificanceTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0, p_value=1.0,
                is_significant=False, alpha=alpha,
                interpretation=f"Test failed: {e}"
            )
        
        # Effect size: r = Z / sqrt(N)
        n = len(values1)
        z = stats.norm.ppf(1 - p_value / 2)
        effect_size = z / np.sqrt(n) if n > 0 else 0
        
        is_significant = p_value < alpha
        
        interpretation = cls._interpret_effect_size(effect_size, "r")
        if is_significant:
            interpretation = f"Significant difference (p={p_value:.4f}). {interpretation}"
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"
        
        return SignificanceTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            effect_size=float(effect_size),
            effect_size_name="r",
            interpretation=interpretation
        )
    
    @classmethod
    def mann_whitney_test(cls, values1: List[float], values2: List[float],
                         alpha: float = 0.05) -> SignificanceTestResult:
        """Perform Mann-Whitney U test (non-parametric independent test).
        
        Args:
            values1: First group values
            values2: Second group values
            alpha: Significance level
        
        Returns:
            SignificanceTestResult
        """
        if not SCIPY_AVAILABLE:
            return SignificanceTestResult(
                test_name="Mann-Whitney U test",
                statistic=0, p_value=1.0,
                is_significant=False, alpha=alpha,
                interpretation="scipy not available"
            )
        
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Effect size: r = Z / sqrt(N)
        n = len(values1) + len(values2)
        z = stats.norm.ppf(1 - p_value / 2)
        effect_size = z / np.sqrt(n) if n > 0 else 0
        
        is_significant = p_value < alpha
        
        interpretation = cls._interpret_effect_size(effect_size, "r")
        if is_significant:
            interpretation = f"Significant difference (p={p_value:.4f}). {interpretation}"
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). {interpretation}"
        
        return SignificanceTestResult(
            test_name="Mann-Whitney U test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            effect_size=float(effect_size),
            effect_size_name="r",
            interpretation=interpretation
        )
    
    @classmethod
    def bootstrap_ci(cls, values: List[float], n_bootstrap: int = 10000,
                    confidence: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.
        
        Args:
            values: Sample values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        
        Returns:
            Tuple of (lower, upper) CI bounds
        """
        values = np.array(values)
        n = len(values)
        
        if n == 0:
            return (0.0, 0.0)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Compute percentile CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return (float(lower), float(upper))
    
    @classmethod
    def _interpret_effect_size(cls, effect_size: float, name: str) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if name == "Cohen's d":
            if abs_effect < 0.2:
                magnitude = "negligible"
            elif abs_effect < 0.5:
                magnitude = "small"
            elif abs_effect < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif name == "r":
            if abs_effect < 0.1:
                magnitude = "negligible"
            elif abs_effect < 0.3:
                magnitude = "small"
            elif abs_effect < 0.5:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"
        
        return f"Effect size ({name}={effect_size:.3f}): {magnitude}"
    
    @classmethod
    def compare_models(cls, model_results: Dict[str, List[float]],
                      alpha: float = 0.05) -> Dict[str, Any]:
        """Compare multiple models using pairwise tests.
        
        Args:
            model_results: Dict mapping model names to lists of metric values
            alpha: Significance level
        
        Returns:
            Dictionary with comparison results
        """
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Compute summaries
        summaries = {
            name: cls.compute_summary(values)
            for name, values in model_results.items()
        }
        
        # Pairwise comparisons
        comparisons = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                name1, name2 = model_names[i], model_names[j]
                values1, values2 = model_results[name1], model_results[name2]
                
                # Use appropriate test
                if len(values1) == len(values2):
                    result = cls.paired_t_test(values1, values2, alpha)
                else:
                    result = cls.independent_t_test(values1, values2, alpha)
                
                comparisons.append({
                    'model1': name1,
                    'model2': name2,
                    'test_result': result,
                    'mean_diff': summaries[name1].mean - summaries[name2].mean,
                })
        
        # Rank models by mean
        ranked = sorted(summaries.items(), key=lambda x: x[1].mean, reverse=True)
        
        return {
            'summaries': summaries,
            'comparisons': comparisons,
            'ranking': [(name, summary.mean) for name, summary in ranked],
            'best_model': ranked[0][0] if ranked else None,
        }
    
    @classmethod
    def format_summary_table(cls, summaries: Dict[str, StatisticalSummary],
                            metric_name: str = "Accuracy") -> str:
        """Format summaries as a text table.
        
        Args:
            summaries: Dict mapping names to StatisticalSummary
            metric_name: Name of the metric
        
        Returns:
            Formatted table string
        """
        lines = []
        lines.append(f"{'Model':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20} {'N':>5}")
        lines.append("-" * 70)
        
        for name, summary in summaries.items():
            ci = f"[{summary.ci_lower:.4f}, {summary.ci_upper:.4f}]"
            lines.append(
                f"{name:<20} {summary.mean:>10.4f} {summary.std:>10.4f} {ci:>20} {summary.n:>5}"
            )
        
        return "\n".join(lines)
    
    @classmethod
    def format_latex_table(cls, summaries: Dict[str, StatisticalSummary],
                          metric_name: str = "Accuracy",
                          caption: str = "",
                          label: str = "") -> str:
        """Format summaries as a LaTeX table.
        
        Args:
            summaries: Dict mapping names to StatisticalSummary
            metric_name: Name of the metric
            caption: Table caption
            label: Table label
        
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{tab:{label}}}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append(f"Model & Mean & Std & 95\\% CI & N \\\\")
        lines.append("\\midrule")
        
        # Find best model
        best_name = max(summaries.items(), key=lambda x: x[1].mean)[0]
        
        for name, summary in summaries.items():
            ci = f"[{summary.ci_lower:.3f}, {summary.ci_upper:.3f}]"
            mean_str = f"{summary.mean:.4f}"
            
            # Bold best result
            if name == best_name:
                mean_str = f"\\textbf{{{mean_str}}}"
            
            lines.append(
                f"{name} & {mean_str} & {summary.std:.4f} & {ci} & {summary.n} \\\\"
            )
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
