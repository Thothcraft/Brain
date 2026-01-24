"""Comparison Plotter - Cross-model and cross-experiment visualization.

Provides visualizations for comparing:
- Multiple models on the same dataset
- Same model across different configurations
- Algorithm comparisons (ML vs DL vs FL)
- Statistical significance visualization
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .base import (
    BasePlotter, PlotType, PlotConfig, PlotData,
    MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE
)
from .themes import ThemeManager
from .statistics import StatisticalAnalyzer, StatisticalSummary

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch

if SEABORN_AVAILABLE:
    import seaborn as sns

logger = logging.getLogger(__name__)


class ComparisonPlotter(BasePlotter):
    """Plotter for comparing multiple models or experiments.
    
    Supports:
    - Bar charts with error bars
    - Box plots
    - Violin plots
    - Critical difference diagrams
    - Radar/spider charts
    """
    
    @property
    def supported_plots(self) -> List[PlotType]:
        return [
            PlotType.MODEL_COMPARISON,
            PlotType.TRIAL_COMPARISON,
            PlotType.STATISTICAL_TEST,
        ]
    
    def create_plot(self, plot_type: PlotType, data: Dict[str, Any],
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        """Create a comparison plot.
        
        Args:
            plot_type: Type of plot to create
            data: Data dictionary
            config: Optional plot configuration
        
        Returns:
            Tuple of (Figure, base64 string)
        """
        if config is None:
            config = self.get_config(plot_type)
        
        if plot_type == PlotType.MODEL_COMPARISON:
            return self._plot_model_comparison(data, config)
        elif plot_type == PlotType.TRIAL_COMPARISON:
            return self._plot_trial_comparison(data, config)
        elif plot_type == PlotType.STATISTICAL_TEST:
            return self._plot_statistical_test(data, config)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_model_comparison(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot comparison across multiple models."""
        plot_style = data.get('style', 'bar')  # 'bar', 'box', 'violin', 'radar'
        
        if plot_style == 'bar':
            return self._plot_bar_comparison(data, config)
        elif plot_style == 'box':
            return self._plot_box_comparison(data, config)
        elif plot_style == 'violin':
            return self._plot_violin_comparison(data, config)
        elif plot_style == 'radar':
            return self._plot_radar_comparison(data, config)
        else:
            return self._plot_bar_comparison(data, config)
    
    def _plot_bar_comparison(self, data: Dict[str, Any],
                            config: PlotConfig) -> Tuple[Figure, str]:
        """Plot bar chart comparison with error bars."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('models', [])))
        
        models = data.get('models', [])
        metrics = data.get('metrics', ['accuracy'])
        
        if isinstance(metrics, str):
            metrics = [metrics]
        
        n_models = len(models)
        n_metrics = len(metrics)
        
        x = np.arange(n_models)
        width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            values = []
            errors = []
            
            for model in models:
                if 'trials' in model:
                    # Multi-trial: compute mean and std
                    trial_values = [t.get(metric, 0) for t in model['trials']]
                    values.append(np.mean(trial_values))
                    errors.append(np.std(trial_values, ddof=1) if len(trial_values) > 1 else 0)
                else:
                    values.append(model.get(metric, 0))
                    errors.append(model.get(f'{metric}_std', 0))
            
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         yerr=errors if any(e > 0 for e in errors) else None,
                         label=metric.replace('_', ' ').title(),
                         color=colors[i % len(colors)], alpha=0.8,
                         capsize=3, edgecolor='black', linewidth=0.5)
            
            # Value labels
            if config.annotate_values:
                for bar, val, err in zip(bars, values, errors):
                    label = f'{val:.3f}'
                    if err > 0:
                        label += f'±{err:.3f}'
                    ax.annotate(label,
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=7, rotation=45)
        
        model_names = [m.get('name', f'Model {i+1}') for i, m in enumerate(models)]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(config.title or 'Model Comparison')
        
        if n_metrics > 1:
            ax.legend(loc='upper right')
        
        # Highlight best
        if config.annotate_best and len(metrics) == 1:
            best_idx = np.argmax(values)
            ax.get_children()[best_idx].set_edgecolor('gold')
            ax.get_children()[best_idx].set_linewidth(2)
        
        self._figures['model_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_box_comparison(self, data: Dict[str, Any],
                            config: PlotConfig) -> Tuple[Figure, str]:
        """Plot box plot comparison."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('models', [])))
        
        models = data.get('models', [])
        metric = data.get('metric', 'accuracy')
        
        # Collect trial data
        box_data = []
        labels = []
        
        for model in models:
            name = model.get('name', 'Model')
            if 'trials' in model:
                values = [t.get(metric, 0) for t in model['trials']]
            else:
                values = [model.get(metric, 0)]
            
            box_data.append(values)
            labels.append(name)
        
        if SEABORN_AVAILABLE:
            # Prepare data for seaborn
            plot_data = []
            for i, (values, label) in enumerate(zip(box_data, labels)):
                for v in values:
                    plot_data.append({'Model': label, metric: v})
            
            import pandas as pd
            df = pd.DataFrame(plot_data)
            sns.boxplot(x='Model', y=metric, data=df, ax=ax, palette=colors)
            sns.stripplot(x='Model', y=metric, data=df, ax=ax, 
                         color='black', alpha=0.5, size=4)
        else:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(config.title or f'Model Comparison - {metric.title()}')
        plt.xticks(rotation=45, ha='right')
        
        self._figures['box_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_violin_comparison(self, data: Dict[str, Any],
                               config: PlotConfig) -> Tuple[Figure, str]:
        """Plot violin plot comparison."""
        if not SEABORN_AVAILABLE:
            logger.warning("Seaborn required for violin plots, falling back to box plot")
            return self._plot_box_comparison(data, config)
        
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('models', [])))
        
        models = data.get('models', [])
        metric = data.get('metric', 'accuracy')
        
        # Prepare data
        plot_data = []
        for model in models:
            name = model.get('name', 'Model')
            if 'trials' in model:
                values = [t.get(metric, 0) for t in model['trials']]
            else:
                values = [model.get(metric, 0)]
            
            for v in values:
                plot_data.append({'Model': name, metric: v})
        
        import pandas as pd
        df = pd.DataFrame(plot_data)
        
        sns.violinplot(x='Model', y=metric, data=df, ax=ax, palette=colors, inner='box')
        sns.stripplot(x='Model', y=metric, data=df, ax=ax, color='black', alpha=0.5, size=3)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(config.title or f'Model Comparison - {metric.title()}')
        plt.xticks(rotation=45, ha='right')
        
        self._figures['violin_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_radar_comparison(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot radar/spider chart comparison."""
        fig, ax = plt.subplots(figsize=config.figsize, subplot_kw=dict(polar=True))
        colors = ThemeManager.get_colors(len(data.get('models', [])))
        
        models = data.get('models', [])
        metrics = data.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        for i, model in enumerate(models):
            name = model.get('name', f'Model {i+1}')
            values = [model.get(m, 0) for m in metrics]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=name)
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title(config.title or 'Model Comparison')
        
        self._figures['radar_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_trial_comparison(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot comparison across trials."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        trials = data.get('trials', [])
        metric = data.get('metric', 'accuracy')
        
        values = [t.get(metric, 0) for t in trials]
        trial_names = [f"Trial {i+1}" for i in range(len(trials))]
        x = np.arange(len(trial_names))
        
        bars = ax.bar(x, values, color=colors[0], alpha=0.7, edgecolor='black')
        
        # Statistics
        mean_val = np.mean(values)
        ax.axhline(mean_val, color=colors[1], linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        
        if len(values) > 1:
            std_val = np.std(values, ddof=1)
            ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2,
                      color=colors[1], label=f'±1 Std: {std_val:.4f}')
            
            # 95% CI
            summary = StatisticalAnalyzer.compute_summary(values)
            ax.axhspan(summary.ci_lower, summary.ci_upper, alpha=0.1,
                      color='green', label=f'95% CI')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Trial')
        ax.set_xticks(x)
        ax.set_xticklabels(trial_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_title(config.title or f'Trial Comparison')
        
        self._figures['trial_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_statistical_test(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot statistical significance matrix."""
        fig, ax = self._create_figure(config)
        
        model_results = data.get('model_results', {})
        alpha = data.get('alpha', 0.05)
        
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Compute pairwise p-values
        p_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                values1 = model_results[model_names[i]]
                values2 = model_results[model_names[j]]
                
                if len(values1) == len(values2):
                    result = StatisticalAnalyzer.paired_t_test(values1, values2, alpha)
                else:
                    result = StatisticalAnalyzer.independent_t_test(values1, values2, alpha)
                
                p_matrix[i, j] = result.p_value
                p_matrix[j, i] = result.p_value
        
        # Plot heatmap
        if SEABORN_AVAILABLE:
            mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
            sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                       xticklabels=model_names, yticklabels=model_names,
                       ax=ax, mask=mask, vmin=0, vmax=0.1,
                       cbar_kws={'label': 'p-value'})
        else:
            im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_yticklabels(model_names)
            
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        color = 'white' if p_matrix[i, j] < 0.05 else 'black'
                        ax.text(j, i, f'{p_matrix[i, j]:.3f}', ha='center', va='center', color=color)
            
            plt.colorbar(im, ax=ax, label='p-value')
        
        ax.set_title(config.title or f'Statistical Significance (α={alpha})')
        
        self._figures['statistical_test'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    # Convenience methods
    def compare_models(self, models: List[Dict[str, Any]], 
                      metrics: List[str] = None,
                      style: str = 'bar',
                      title: str = None) -> Dict[str, str]:
        """Compare multiple models.
        
        Args:
            models: List of model result dictionaries
            metrics: Metrics to compare
            style: Plot style ('bar', 'box', 'violin', 'radar')
            title: Plot title
        
        Returns:
            Dictionary mapping plot names to base64 strings
        """
        if metrics is None:
            metrics = ['accuracy']
        
        data = {
            'models': models,
            'metrics': metrics,
            'style': style,
        }
        
        config = PlotConfig(
            plot_type=PlotType.MODEL_COMPARISON,
            title=title,
            annotate_values=True,
            annotate_best=True,
        )
        
        self.create_plot(PlotType.MODEL_COMPARISON, data, config)
        
        return self.get_all_base64()
    
    def compare_with_statistics(self, model_results: Dict[str, List[float]],
                               metric_name: str = "Accuracy",
                               alpha: float = 0.05) -> Dict[str, Any]:
        """Compare models with full statistical analysis.
        
        Args:
            model_results: Dict mapping model names to lists of metric values
            metric_name: Name of the metric
            alpha: Significance level
        
        Returns:
            Dictionary with plots and statistical results
        """
        # Statistical analysis
        analysis = StatisticalAnalyzer.compare_models(model_results, alpha)
        
        # Create comparison plots
        models = [
            {
                'name': name,
                'trials': [{'accuracy': v} for v in values],
                'accuracy': analysis['summaries'][name].mean,
                'accuracy_std': analysis['summaries'][name].std,
            }
            for name, values in model_results.items()
        ]
        
        # Bar comparison
        bar_data = {'models': models, 'metrics': ['accuracy'], 'style': 'bar'}
        bar_config = PlotConfig(plot_type=PlotType.MODEL_COMPARISON, 
                               title=f'{metric_name} Comparison',
                               annotate_values=True)
        self.create_plot(PlotType.MODEL_COMPARISON, bar_data, bar_config)
        
        # Box plot
        box_data = {'models': models, 'metric': 'accuracy', 'style': 'box'}
        box_config = PlotConfig(plot_type=PlotType.MODEL_COMPARISON,
                               title=f'{metric_name} Distribution')
        self._plot_box_comparison(box_data, box_config)
        
        # Statistical significance
        sig_data = {'model_results': model_results, 'alpha': alpha}
        sig_config = PlotConfig(plot_type=PlotType.STATISTICAL_TEST)
        self.create_plot(PlotType.STATISTICAL_TEST, sig_data, sig_config)
        
        return {
            'plots': self.get_all_base64(),
            'analysis': analysis,
            'summary_table': StatisticalAnalyzer.format_summary_table(
                analysis['summaries'], metric_name
            ),
            'latex_table': StatisticalAnalyzer.format_latex_table(
                analysis['summaries'], metric_name,
                caption=f'{metric_name} comparison across models',
                label='model-comparison'
            ),
        }
