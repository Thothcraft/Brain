"""Visualization Module for ML/DL Training.

This module provides visualization utilities for:
- Training curves (loss, accuracy)
- Confusion matrices
- ROC and PR curves
- Feature importance
- Publication-ready figure export

Re-exports from figure_export.py and metrics/visualizer.py.
"""

from ..figure_export import (
    setup_ieee_style,
    IEEE_SINGLE_COLUMN,
    IEEE_DOUBLE_COLUMN,
    IEEE_COLORS,
)

from ..metrics.visualizer import (
    MetricsVisualizer,
)

__all__ = [
    # IEEE styling
    "setup_ieee_style",
    "IEEE_SINGLE_COLUMN",
    "IEEE_DOUBLE_COLUMN",
    "IEEE_COLORS",
    # Visualizer
    "MetricsVisualizer",
]
