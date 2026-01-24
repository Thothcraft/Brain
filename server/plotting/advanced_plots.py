"""Advanced Plot Types for Comprehensive ML Experiment Analysis.

Extends the base plotting library with specialized visualizations for:
- Resource utilization (CPU, GPU, memory, disk I/O)
- Timing analysis (per-epoch, per-batch, breakdown)
- Model complexity (parameters, FLOPs, memory footprint)
- Hyperparameter analysis (importance, interactions)
- Data analysis (class distribution, feature correlation)
- Gradient analysis (norms, flow, vanishing/exploding detection)
- Calibration plots (reliability diagrams)
- Error analysis (misclassification patterns)
- Comparison plots (Pareto front, ablation studies)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .base import BasePlotter, PlotType, PlotConfig, MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE
from .themes import ThemeManager

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

if SEABORN_AVAILABLE:
    import seaborn as sns

logger = logging.getLogger(__name__)


class AdvancedPlotType:
    """Extended plot types."""
    # Resource
    RESOURCE_TIMELINE = "resource_timeline"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    MEMORY_USAGE = "memory_usage"
    # Timing
    TIMING_BREAKDOWN = "timing_breakdown"
    EPOCH_TIMING = "epoch_timing"
    THROUGHPUT = "throughput"
    # Model
    PARAMETER_DISTRIBUTION = "parameter_distribution"
    LAYER_COMPLEXITY = "layer_complexity"
    MEMORY_FOOTPRINT = "memory_footprint"
    # Hyperparameter
    HP_IMPORTANCE = "hp_importance"
    HP_OPTIMIZATION_HISTORY = "hp_optimization_history"
    # Data
    CLASS_DISTRIBUTION = "class_distribution"
    FEATURE_CORRELATION = "feature_correlation"
    # Gradient
    GRADIENT_NORM = "gradient_norm"
    GRADIENT_FLOW = "gradient_flow"
    # Calibration
    RELIABILITY_DIAGRAM = "reliability_diagram"
    # Error
    ERROR_DISTRIBUTION = "error_distribution"
    # Comparison
    PARETO_FRONT = "pareto_front"
    ABLATION_STUDY = "ablation_study"
    SCALING_CURVE = "scaling_curve"
    # Embedding
    EMBEDDING_2D = "embedding_2d"


class AdvancedPlotter(BasePlotter):
    """Advanced plotter with comprehensive visualization capabilities."""
    
    @property
    def supported_plots(self) -> List[str]:
        return [v for k, v in vars(AdvancedPlotType).items() if not k.startswith('_')]
    
    def create_plot(self, plot_type: str, data: Dict[str, Any],
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        if config is None:
            config = PlotConfig(plot_type=PlotType.MODEL_COMPARISON)
        
        method_map = {
            AdvancedPlotType.RESOURCE_TIMELINE: self._plot_resource_timeline,
            AdvancedPlotType.TIMING_BREAKDOWN: self._plot_timing_breakdown,
            AdvancedPlotType.EPOCH_TIMING: self._plot_epoch_timing,
            AdvancedPlotType.PARAMETER_DISTRIBUTION: self._plot_parameter_distribution,
            AdvancedPlotType.CLASS_DISTRIBUTION: self._plot_class_distribution,
            AdvancedPlotType.GRADIENT_NORM: self._plot_gradient_norm,
            AdvancedPlotType.RELIABILITY_DIAGRAM: self._plot_reliability_diagram,
            AdvancedPlotType.PARETO_FRONT: self._plot_pareto_front,
            AdvancedPlotType.ABLATION_STUDY: self._plot_ablation_study,
            AdvancedPlotType.EMBEDDING_2D: self._plot_embedding_2d,
        }
        
        if plot_type in method_map:
            return method_map[plot_type](data, config)
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_resource_timeline(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot resource usage over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        colors = ThemeManager.get_colors(4)
        
        ts = data.get('timestamps', list(range(len(data.get('cpu_percent', [])))))
        
        for ax, key, label, color in [
            (axes[0], 'cpu_percent', 'CPU (%)', colors[0]),
            (axes[1], 'gpu_percent', 'GPU (%)', colors[1]),
            (axes[2], 'memory_percent', 'Memory (%)', colors[2]),
        ]:
            vals = data.get(key, [])
            if vals:
                ax.fill_between(ts[:len(vals)], vals, alpha=0.3, color=color)
                ax.plot(ts[:len(vals)], vals, color=color, linewidth=1.5)
                ax.set_ylabel(label)
                ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time (s)')
        fig.suptitle(config.title or 'Resource Usage Timeline', fontsize=14)
        fig.tight_layout()
        self._figures['resource_timeline'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_timing_breakdown(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot timing breakdown."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ThemeManager.get_colors(8)
        
        components = data.get('components', {})
        if components:
            labels, times = zip(*sorted(components.items(), key=lambda x: x[1], reverse=True))
            axes[0].pie(times, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
            axes[0].set_title('Time Distribution')
            
            axes[1].barh(range(len(labels)), times, color=colors[0], alpha=0.8)
            axes[1].set_yticks(range(len(labels)))
            axes[1].set_yticklabels(labels)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_title('Time by Component')
        
        fig.tight_layout()
        self._figures['timing_breakdown'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_epoch_timing(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot per-epoch timing."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = ThemeManager.get_colors(3)
        
        epochs = data.get('epochs', list(range(1, len(data.get('epoch_times', [])) + 1)))
        times = data.get('epoch_times', [])
        
        if times:
            axes[0,0].bar(epochs, times, color=colors[0], alpha=0.7)
            axes[0,0].set_title('Epoch Duration')
            
            axes[0,1].plot(epochs, np.cumsum(times), 'o-', color=colors[0])
            axes[0,1].set_title('Cumulative Time')
            
            axes[1,0].hist(times, bins=20, color=colors[0], alpha=0.7)
            axes[1,0].set_title('Time Distribution')
            
            if len(times) > 5:
                ma = np.convolve(times, np.ones(5)/5, mode='valid')
                axes[1,1].plot(epochs, times, 'o', alpha=0.5)
                axes[1,1].plot(epochs[4:], ma, '-', linewidth=2)
            axes[1,1].set_title('Time Trend')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figures['epoch_timing'] = fig
        return fig, self._finalize_figure(fig, axes[0,0], config)
    
    def _plot_parameter_distribution(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot parameter distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = ThemeManager.get_colors(10)
        
        layers = data.get('layers', [])
        if layers:
            names = [l.get('name', f'L{i}')[:20] for i, l in enumerate(layers)]
            params = [l.get('parameters', 0) for l in layers]
            
            axes[0].barh(range(len(names)), params, color=colors[0])
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, fontsize=8)
            axes[0].set_xlabel('Parameters')
            axes[0].set_title('Parameters by Layer')
            
            type_params = {}
            for l in layers:
                t = l.get('type', 'other')
                type_params[t] = type_params.get(t, 0) + l.get('parameters', 0)
            axes[1].pie(list(type_params.values()), labels=list(type_params.keys()), 
                       autopct='%1.1f%%', colors=colors)
            axes[1].set_title('By Layer Type')
        
        fig.tight_layout()
        self._figures['parameter_distribution'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_class_distribution(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot class distribution."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(3)
        
        names = data.get('class_names', [])
        train = data.get('train_counts', [])
        val = data.get('val_counts', [])
        
        if names and train:
            x = np.arange(len(names))
            width = 0.35
            ax.bar(x - width/2, train, width, label='Train', color=colors[0])
            if val:
                ax.bar(x + width/2, val, width, label='Val', color=colors[1])
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.legend()
        
        ax.set_title(config.title or 'Class Distribution')
        self._figures['class_distribution'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_gradient_norm(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot gradient norms."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ThemeManager.get_colors(2)
        
        norms = data.get('grad_norms', [])
        if norms:
            axes[0].semilogy(norms, color=colors[0], alpha=0.7)
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Gradient Norm')
            axes[0].set_title('Gradient Norm Over Training')
            
            axes[1].hist(np.log10(np.array(norms) + 1e-10), bins=50, color=colors[0])
            axes[1].set_xlabel('log10(Norm)')
            axes[1].set_title('Distribution')
        
        fig.tight_layout()
        self._figures['gradient_norm'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_reliability_diagram(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot reliability diagram."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ThemeManager.get_colors(2)
        
        confs = data.get('confidences', [])
        accs = data.get('accuracies', [])
        counts = data.get('counts', [])
        
        if confs and accs:
            width = 1.0 / len(confs)
            axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect')
            axes[0].bar(confs, accs, width=width*0.8, color=colors[0], alpha=0.7)
            axes[0].set_xlabel('Confidence')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].set_title('Reliability Diagram')
            
            if counts:
                axes[1].bar(confs, counts, width=width*0.8, color=colors[1])
                axes[1].set_xlabel('Confidence')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Confidence Distribution')
        
        fig.tight_layout()
        self._figures['reliability_diagram'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_pareto_front(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot Pareto front for multi-objective optimization."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        points = data.get('points', [])
        x_label = data.get('x_label', 'Objective 1')
        y_label = data.get('y_label', 'Objective 2')
        
        if points:
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            pareto = data.get('pareto_indices', [])
            
            ax.scatter(x, y, c=colors[0], alpha=0.5, label='All')
            if pareto:
                px = [x[i] for i in pareto]
                py = [y[i] for i in pareto]
                ax.scatter(px, py, c=colors[1], s=100, marker='*', label='Pareto')
                sorted_pareto = sorted(zip(px, py))
                ax.plot([p[0] for p in sorted_pareto], [p[1] for p in sorted_pareto], 
                       'r--', alpha=0.5)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
        
        ax.set_title(config.title or 'Pareto Front')
        self._figures['pareto_front'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_ablation_study(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot ablation study results."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        components = data.get('components', [])
        baseline = data.get('baseline', 0)
        
        if components:
            names = [c.get('name', f'C{i}') for i, c in enumerate(components)]
            values = [c.get('value', 0) for c in components]
            diffs = [v - baseline for v in values]
            
            colors_bar = [colors[0] if d >= 0 else colors[1] for d in diffs]
            ax.barh(names, diffs, color=colors_bar, alpha=0.8)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Change from Baseline')
            
            ax.text(0.95, 0.05, f'Baseline: {baseline:.4f}', transform=ax.transAxes,
                   ha='right', fontsize=10)
        
        ax.set_title(config.title or 'Ablation Study')
        self._figures['ablation_study'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_embedding_2d(self, data: Dict, config: PlotConfig) -> Tuple[Figure, str]:
        """Plot 2D embeddings."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(20)
        
        emb = np.array(data.get('embeddings', []))
        labels = data.get('labels', [])
        method = data.get('method', 'PCA')
        
        if len(emb) > 0:
            if labels:
                for i, label in enumerate(sorted(set(labels))):
                    mask = np.array(labels) == label
                    ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[i % len(colors)],
                              label=f'Class {label}', alpha=0.6, s=20)
                ax.legend(fontsize=8)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], alpha=0.6, s=20)
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
        
        ax.set_title(config.title or f'{method} Embedding')
        self._figures['embedding_2d'] = fig
        return fig, self._finalize_figure(fig, ax, config)
