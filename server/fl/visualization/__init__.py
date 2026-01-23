"""Visualization Module for Federated Learning.

This module provides plotting utilities for FL experiments:
- Accuracy curves with confidence bands
- Sample distribution across clients
- Comparative bar charts
- Shadow plots for multiple runs
- Statistical analysis visualizations
"""

from .plots import (
    plot_accuracy_curves,
    plot_sample_distribution,
    plot_comparative_results,
    plot_shadow_curves,
    plot_convergence_comparison,
    plot_client_performance,
)
from .statistics import (
    generate_statistics_report,
    compute_effect_size,
    compute_convergence_metrics,
)

__all__ = [
    "plot_accuracy_curves",
    "plot_sample_distribution",
    "plot_comparative_results",
    "plot_shadow_curves",
    "plot_convergence_comparison",
    "plot_client_performance",
    "generate_statistics_report",
    "compute_effect_size",
    "compute_convergence_metrics",
]
