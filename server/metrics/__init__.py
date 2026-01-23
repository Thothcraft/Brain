"""Standardized Training Metrics Module.

This module provides unified metrics tracking and reporting across
all training modes (central ML, DL, and FL).

Features:
- Standardized metric definitions
- Real-time metric tracking
- Professional visualization
- Publication-quality exports
- Metric aggregation and comparison

Usage:
    from server.metrics import MetricsTracker, MetricsVisualizer
    
    # Create tracker
    tracker = MetricsTracker(job_id="train_001")
    
    # Log metrics
    tracker.log_epoch(epoch=1, train_loss=0.5, train_acc=0.8, val_loss=0.6, val_acc=0.75)
    
    # Generate visualizations
    visualizer = MetricsVisualizer(tracker)
    plots = visualizer.generate_all_plots()
"""

from .tracker import (
    MetricsTracker,
    EpochMetrics,
    BatchMetrics,
    ClassMetrics,
    FLRoundMetrics,
)
from .visualizer import MetricsVisualizer
from .exporter import MetricsExporter

__all__ = [
    "MetricsTracker",
    "MetricsVisualizer",
    "MetricsExporter",
    "EpochMetrics",
    "BatchMetrics",
    "ClassMetrics",
    "FLRoundMetrics",
]
