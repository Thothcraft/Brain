"""Base classes and interfaces for the plotting library.

This module defines the core abstractions that all plotters inherit from,
ensuring consistency and extensibility across ML, DL, and FL visualizations.
"""

import os
import io
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class PlotType(str, Enum):
    """Types of plots supported by the library."""
    # Common
    CONFUSION_MATRIX = "confusion_matrix"
    ROC_CURVE = "roc_curve"
    PR_CURVE = "pr_curve"
    CLASS_METRICS = "class_metrics"
    
    # DL-specific
    LEARNING_CURVE = "learning_curve"
    LOSS_CURVE = "loss_curve"
    ACCURACY_CURVE = "accuracy_curve"
    LR_SCHEDULE = "lr_schedule"
    GRADIENT_FLOW = "gradient_flow"
    
    # ML-specific
    METRICS_BAR = "metrics_bar"
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_BOUNDARY = "decision_boundary"
    
    # FL-specific
    CONVERGENCE = "convergence"
    CLIENT_DRIFT = "client_drift"
    AGGREGATION_WEIGHTS = "aggregation_weights"
    ROUND_METRICS = "round_metrics"
    
    # Comparison
    MODEL_COMPARISON = "model_comparison"
    TRIAL_COMPARISON = "trial_comparison"
    STATISTICAL_TEST = "statistical_test"


class ExportFormat(str, Enum):
    """Supported export formats."""
    PNG = "png"
    PDF = "pdf"
    EPS = "eps"
    SVG = "svg"
    TIKZ = "tikz"
    PGFPLOTS = "pgfplots"
    HTML = "html"
    JSON = "json"


@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    plot_type: PlotType
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    grid: bool = True
    legend: bool = True
    legend_loc: str = "best"
    tight_layout: bool = True
    
    # Style options
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    
    # Statistical options
    show_ci: bool = True
    ci_level: float = 0.95
    show_std: bool = True
    
    # Annotation options
    annotate_best: bool = True
    annotate_values: bool = False
    
    # Custom options
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """Configuration for exporting plots."""
    output_dir: str = "./figures"
    formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.PNG, ExportFormat.PDF])
    dpi: int = 300
    transparent: bool = False
    bbox_inches: str = "tight"
    pad_inches: float = 0.05
    
    # PDF/EPS specific
    embed_fonts: bool = True
    
    # TikZ specific
    tikz_standalone: bool = False
    tikz_width: str = "\\textwidth"
    
    # Naming
    prefix: str = ""
    suffix: str = ""
    timestamp: bool = False


@dataclass
class PlotData:
    """Container for plot data with metadata."""
    name: str
    x: Optional[np.ndarray] = None
    y: np.ndarray = None
    y_err: Optional[np.ndarray] = None  # Error bars / std
    y_ci_lower: Optional[np.ndarray] = None  # Confidence interval
    y_ci_upper: Optional[np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    marker: Optional[str] = None
    linestyle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePlotter(ABC):
    """Abstract base class for all plotters.
    
    Provides common functionality and defines the interface that all
    specialized plotters must implement.
    """
    
    def __init__(self, theme: str = "default"):
        """Initialize the plotter.
        
        Args:
            theme: Theme name ('default', 'neurips', 'icml', 'cvpr', etc.)
        """
        self.theme_name = theme
        self._figures: Dict[str, Figure] = {}
        self._data: Dict[str, List[PlotData]] = {}
        self._configs: Dict[str, PlotConfig] = {}
        self._live_mode = False
        self._callbacks: List[Callable] = []
        
        if MATPLOTLIB_AVAILABLE:
            self._apply_theme()
    
    def _apply_theme(self):
        """Apply the selected theme to matplotlib."""
        from .themes import ThemeManager
        ThemeManager.apply_theme(self.theme_name)
    
    @property
    @abstractmethod
    def supported_plots(self) -> List[PlotType]:
        """Return list of plot types supported by this plotter."""
        pass
    
    @abstractmethod
    def create_plot(self, plot_type: PlotType, data: Dict[str, Any], 
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        """Create a specific type of plot.
        
        Args:
            plot_type: Type of plot to create
            data: Data dictionary for the plot
            config: Optional plot configuration
        
        Returns:
            Tuple of (matplotlib Figure, base64 encoded string)
        """
        pass
    
    def add_data(self, name: str, data: PlotData):
        """Add data series for plotting.
        
        Args:
            name: Unique name for this data series
            data: PlotData object containing the data
        """
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(data)
    
    def clear_data(self, name: Optional[str] = None):
        """Clear stored data.
        
        Args:
            name: Specific data series to clear, or None to clear all
        """
        if name is None:
            self._data.clear()
        elif name in self._data:
            del self._data[name]
    
    def set_config(self, plot_type: PlotType, config: PlotConfig):
        """Set configuration for a plot type.
        
        Args:
            plot_type: Type of plot
            config: Configuration to apply
        """
        self._configs[plot_type.value] = config
    
    def get_config(self, plot_type: PlotType) -> PlotConfig:
        """Get configuration for a plot type.
        
        Args:
            plot_type: Type of plot
        
        Returns:
            PlotConfig for the specified plot type
        """
        if plot_type.value in self._configs:
            return self._configs[plot_type.value]
        return PlotConfig(plot_type=plot_type)
    
    def _create_figure(self, config: PlotConfig) -> Tuple[Figure, Axes]:
        """Create a new figure with the specified configuration.
        
        Args:
            config: Plot configuration
        
        Returns:
            Tuple of (Figure, Axes)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib is required for plotting")
        
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        if config.grid:
            ax.grid(True, alpha=0.3, linewidth=0.5)
        
        if config.title:
            ax.set_title(config.title)
        if config.xlabel:
            ax.set_xlabel(config.xlabel)
        if config.ylabel:
            ax.set_ylabel(config.ylabel)
        
        return fig, ax
    
    def _finalize_figure(self, fig: Figure, ax: Axes, config: PlotConfig) -> str:
        """Finalize figure and convert to base64.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
            config: Plot configuration
        
        Returns:
            Base64 encoded PNG string
        """
        if config.legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc=config.legend_loc)
        
        if config.tight_layout:
            fig.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=config.dpi, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return b64
    
    def export(self, name: str, export_config: Optional[ExportConfig] = None) -> Dict[str, str]:
        """Export a plot to files.
        
        Args:
            name: Name of the plot to export
            export_config: Export configuration
        
        Returns:
            Dictionary mapping format to file path
        """
        if name not in self._figures:
            raise ValueError(f"No figure named '{name}' found")
        
        if export_config is None:
            export_config = ExportConfig()
        
        from .export import ExportManager
        return ExportManager.export_figure(
            self._figures[name], name, export_config
        )
    
    def export_all(self, export_config: Optional[ExportConfig] = None) -> Dict[str, Dict[str, str]]:
        """Export all plots to files.
        
        Args:
            export_config: Export configuration
        
        Returns:
            Dictionary mapping plot name to {format: filepath}
        """
        results = {}
        for name in self._figures:
            results[name] = self.export(name, export_config)
        return results
    
    def to_base64(self, name: str) -> str:
        """Get base64 encoded PNG of a plot.
        
        Args:
            name: Name of the plot
        
        Returns:
            Base64 encoded string
        """
        if name not in self._figures:
            raise ValueError(f"No figure named '{name}' found")
        
        fig = self._figures[name]
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def get_all_base64(self) -> Dict[str, str]:
        """Get base64 encoded PNGs of all plots.
        
        Returns:
            Dictionary mapping plot name to base64 string
        """
        return {name: self.to_base64(name) for name in self._figures}
    
    def close(self, name: Optional[str] = None):
        """Close figure(s) to free memory.
        
        Args:
            name: Specific figure to close, or None to close all
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if name is None:
            for fig in self._figures.values():
                plt.close(fig)
            self._figures.clear()
        elif name in self._figures:
            plt.close(self._figures[name])
            del self._figures[name]
    
    # Live plotting interface
    def start_live_session(self):
        """Start a live plotting session."""
        self._live_mode = True
        from .live import LivePlotManager
        self._live_manager = LivePlotManager(self)
        self._live_manager.start()
    
    def update_live(self, **kwargs):
        """Update live plots with new data.
        
        Args:
            **kwargs: Data to update (implementation-specific)
        """
        if self._live_mode and hasattr(self, '_live_manager'):
            self._live_manager.update(**kwargs)
    
    def end_live_session(self):
        """End the live plotting session."""
        self._live_mode = False
        if hasattr(self, '_live_manager'):
            self._live_manager.stop()
    
    def register_callback(self, callback: Callable):
        """Register a callback for plot updates.
        
        Args:
            callback: Function to call when plots are updated
        """
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, data: Any = None):
        """Notify all registered callbacks.
        
        Args:
            event: Event name
            data: Event data
        """
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error: {e}")


# Utility functions
def check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    return MATPLOTLIB_AVAILABLE


def check_seaborn() -> bool:
    """Check if seaborn is available."""
    return SEABORN_AVAILABLE
