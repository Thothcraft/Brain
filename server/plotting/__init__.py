"""Thoth Plotting Library - Professional ML/DL/FL Visualization.

A modular, extensible plotting library for machine learning research that provides:
- Live plotting with real-time updates during training
- Publication-quality exports (PDF, EPS, SVG, PNG, TikZ/PGFPlots)
- Specialized plotters for ML, DL, and FL paradigms
- Theme system for different publication venues (NeurIPS, ICML, CVPR, JMLR, IEEE)
- Statistical visualization with confidence intervals and significance tests
- Multi-trial aggregation and comparison

Architecture:
    BasePlotter (abstract)
    ├── MLPlotter (classical ML models)
    ├── DLPlotter (deep learning with epochs)
    ├── FLPlotter (federated learning)
    └── ComparisonPlotter (cross-model comparison)

Usage:
    from server.plotting import create_plotter, ThemeManager
    
    # Create appropriate plotter based on model type
    plotter = create_plotter('dl', theme='neurips')
    
    # Live plotting during training
    plotter.start_live_session()
    for epoch in range(epochs):
        # ... training ...
        plotter.update(epoch, train_loss, val_loss, train_acc, val_acc)
    plotter.end_live_session()
    
    # Export for publication
    plotter.export('figures/training_curves', formats=['pdf', 'eps', 'tikz'])
"""

from .base import BasePlotter, PlotConfig, ExportConfig
from .themes import ThemeManager, PublicationTheme
from .ml_plotter import MLPlotter
from .dl_plotter import DLPlotter
from .fl_plotter import FLPlotter
from .comparison import ComparisonPlotter
from .live import LivePlotManager
from .export import ExportManager, TikZExporter, HTMLExporter, LaTeXHelper
from .statistics import StatisticalAnalyzer
from .factory import create_plotter, PlotterFactory
from .advanced_plots import AdvancedPlotter, AdvancedPlotType
from .experiment_tracker import ExperimentTracker, create_tracker

__all__ = [
    # Core
    'BasePlotter',
    'PlotConfig',
    'ExportConfig',
    # Specialized plotters
    'MLPlotter',
    'DLPlotter', 
    'FLPlotter',
    'ComparisonPlotter',
    'AdvancedPlotter',
    # Plot types
    'AdvancedPlotType',
    # Themes
    'ThemeManager',
    'PublicationTheme',
    # Live plotting
    'LivePlotManager',
    # Export
    'ExportManager',
    'TikZExporter',
    'HTMLExporter',
    'LaTeXHelper',
    # Statistics
    'StatisticalAnalyzer',
    # Factory
    'create_plotter',
    'PlotterFactory',
    # Experiment tracking
    'ExperimentTracker',
    'create_tracker',
]

__version__ = '1.0.0'
