"""Reporting Module for Training Results.

This module provides comprehensive reporting functionality for ML/DL training:
- Training metrics collection and tracking
- Report generation (JSON, HTML, PDF)
- Shareable read-only view links
- Export utilities

Re-exports from training_report.py for backward compatibility.
"""

from ..training_report import (
    EpochMetrics,
    ClassMetrics,
    ConfusionMatrixData,
    ROCCurveData,
    PRCurveData,
    TrainingReport,
    ReportManager,
)

__all__ = [
    "EpochMetrics",
    "ClassMetrics",
    "ConfusionMatrixData",
    "ROCCurveData",
    "PRCurveData",
    "TrainingReport",
    "ReportManager",
]
