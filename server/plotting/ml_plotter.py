"""Machine Learning Plotter - Specialized visualization for classical ML models.

Provides visualizations for non-iterative ML models (SVM, RF, KNN, etc.):
- Metrics bar charts (no epoch curves)
- Feature importance
- Decision boundaries (2D)
- Cross-validation results
- Hyperparameter analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .base import (
    BasePlotter, PlotType, PlotConfig, PlotData,
    MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE
)
from .themes import ThemeManager

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch

if SEABORN_AVAILABLE:
    import seaborn as sns

logger = logging.getLogger(__name__)


class MLPlotter(BasePlotter):
    """Plotter specialized for classical machine learning models.
    
    Unlike DL models, classical ML models don't have epoch-based training,
    so this plotter focuses on:
    - Final metrics comparison
    - Feature importance
    - Cross-validation analysis
    - Multi-trial statistical comparison
    """
    
    @property
    def supported_plots(self) -> List[PlotType]:
        return [
            PlotType.METRICS_BAR,
            PlotType.FEATURE_IMPORTANCE,
            PlotType.CONFUSION_MATRIX,
            PlotType.ROC_CURVE,
            PlotType.PR_CURVE,
            PlotType.CLASS_METRICS,
            PlotType.TRIAL_COMPARISON,
            PlotType.MODEL_COMPARISON,
        ]
    
    def create_plot(self, plot_type: PlotType, data: Dict[str, Any],
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        """Create an ML-specific plot.
        
        Args:
            plot_type: Type of plot to create
            data: Data dictionary containing metrics and results
            config: Optional plot configuration
        
        Returns:
            Tuple of (Figure, base64 string)
        """
        if config is None:
            config = self.get_config(plot_type)
        
        if plot_type == PlotType.METRICS_BAR:
            return self._plot_metrics_bar(data, config)
        elif plot_type == PlotType.FEATURE_IMPORTANCE:
            return self._plot_feature_importance(data, config)
        elif plot_type == PlotType.CONFUSION_MATRIX:
            return self._plot_confusion_matrix(data, config)
        elif plot_type == PlotType.ROC_CURVE:
            return self._plot_roc_curves(data, config)
        elif plot_type == PlotType.PR_CURVE:
            return self._plot_pr_curves(data, config)
        elif plot_type == PlotType.CLASS_METRICS:
            return self._plot_class_metrics(data, config)
        elif plot_type == PlotType.TRIAL_COMPARISON:
            return self._plot_trial_comparison(data, config)
        elif plot_type == PlotType.MODEL_COMPARISON:
            return self._plot_model_comparison(data, config)
        else:
            raise ValueError(f"Unsupported plot type for MLPlotter: {plot_type}")
    
    def _plot_metrics_bar(self, data: Dict[str, Any],
                         config: PlotConfig) -> Tuple[Figure, str]:
        """Plot metrics as bar chart with error bars for multi-trial."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(4)
        
        # Extract metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Check if multi-trial
        if 'trials' in data:
            means, stds, individual = self._aggregate_trial_metrics(data['trials'])
            show_individual = data.get('show_individual', True)
        else:
            means = [
                data.get('accuracy', 0),
                data.get('precision', 0),
                data.get('recall', 0),
                data.get('f1_score', data.get('f1', 0)),
            ]
            stds = [0, 0, 0, 0]
            individual = None
            show_individual = False
        
        x = np.arange(len(metrics))
        width = 0.6
        
        # Bar chart with error bars
        bars = ax.bar(x, means, width, yerr=stds if any(s > 0 for s in stds) else None,
                     capsize=5, color=colors[0], alpha=0.7, edgecolor='black',
                     error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        # Show individual trial points
        if show_individual and individual is not None:
            for i, metric in enumerate(metrics):
                values = individual.get(metric, [])
                if values:
                    jitter = np.random.uniform(-0.15, 0.15, len(values))
                    ax.scatter(x[i] + jitter, values, color=colors[1],
                              s=30, alpha=0.6, zorder=3, edgecolor='white', linewidth=0.5)
        
        # Value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            label = f'{mean:.3f}'
            if std > 0:
                label += f'±{std:.3f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Score')
        ax.set_xlabel('Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.15])
        ax.set_title(config.title or f"{data.get('model_type', 'ML')} Performance")
        
        # Legend for individual points
        if show_individual and individual is not None:
            legend_elements = [
                Patch(facecolor=colors[0], alpha=0.7, edgecolor='black', label='Mean ± Std'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1],
                          markersize=8, label='Individual Trials')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        self._figures['metrics_bar'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _aggregate_trial_metrics(self, trials: List[Dict]) -> Tuple[List[float], List[float], Dict]:
        """Aggregate metrics from multiple trials."""
        metrics = {
            'Accuracy': [t.get('accuracy', 0) for t in trials],
            'Precision': [t.get('precision', 0) for t in trials],
            'Recall': [t.get('recall', 0) for t in trials],
            'F1-Score': [t.get('f1_score', t.get('f1', 0)) for t in trials],
        }
        
        means = [np.mean(metrics[m]) for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        stds = [np.std(metrics[m], ddof=1) if len(metrics[m]) > 1 else 0 
                for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        
        return means, stds, metrics
    
    def _plot_feature_importance(self, data: Dict[str, Any],
                                config: PlotConfig) -> Tuple[Figure, str]:
        """Plot feature importance bar chart."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(1)
        
        features = data.get('feature_names', [])
        importance = data.get('importance', [])
        
        # Sort by importance
        if data.get('sort', True):
            sorted_idx = np.argsort(importance)[::-1]
            features = [features[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
        
        # Limit to top N
        top_n = data.get('top_n', 20)
        if len(features) > top_n:
            features = features[:top_n]
            importance = importance[:top_n]
        
        y = np.arange(len(features))
        
        ax.barh(y, importance, color=colors[0], alpha=0.8, edgecolor='black')
        ax.set_yticks(y)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(config.title or 'Feature Importance')
        
        self._figures['feature_importance'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_confusion_matrix(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot confusion matrix heatmap."""
        fig, ax = self._create_figure(config)
        
        cm = np.array(data['confusion_matrix'])
        class_names = data.get('class_names', [f'Class {i}' for i in range(len(cm))])
        normalize = data.get('normalize', False)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
        else:
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
            
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = f'{cm[i, j]:{fmt}}'
                    color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color)
            
            plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(config.title or 'Confusion Matrix')
        
        self._figures['confusion_matrix'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_roc_curves(self, data: Dict[str, Any],
                        config: PlotConfig) -> Tuple[Figure, str]:
        """Plot ROC curves."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('roc_data', [])))
        
        for i, roc in enumerate(data.get('roc_data', [])):
            ax.plot(roc['fpr'], roc['tpr'], color=colors[i], linewidth=2,
                   label=f"{roc['class_name']} (AUC = {roc['auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(config.title or 'ROC Curves')
        ax.legend(loc='lower right')
        
        self._figures['roc_curves'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_pr_curves(self, data: Dict[str, Any],
                       config: PlotConfig) -> Tuple[Figure, str]:
        """Plot Precision-Recall curves."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('pr_data', [])))
        
        for i, pr in enumerate(data.get('pr_data', [])):
            ax.plot(pr['recall'], pr['precision'], color=colors[i], linewidth=2,
                   label=f"{pr['class_name']} (AP = {pr['ap']:.3f})")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(config.title or 'Precision-Recall Curves')
        ax.legend(loc='lower left')
        
        self._figures['pr_curves'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_class_metrics(self, data: Dict[str, Any],
                           config: PlotConfig) -> Tuple[Figure, str]:
        """Plot per-class metrics."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(3)
        
        class_names = data.get('class_names', [])
        precision = data.get('precision', [])
        recall = data.get('recall', [])
        f1 = data.get('f1', [])
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', color=colors[0])
        ax.bar(x, recall, width, label='Recall', color=colors[1])
        ax.bar(x + width, f1, width, label='F1-Score', color=colors[2])
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(config.title or 'Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        self._figures['class_metrics'] = fig
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
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Trial')
        ax.set_xticks(x)
        ax.set_xticklabels(trial_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_title(config.title or f'Trial Comparison - {metric.title()}')
        
        self._figures['trial_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_model_comparison(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot comparison across different models."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('models', [])))
        
        models = data.get('models', [])
        metric = data.get('metric', 'accuracy')
        
        model_names = [m.get('name', f'Model {i+1}') for i, m in enumerate(models)]
        values = [m.get(metric, 0) for m in models]
        errors = [m.get(f'{metric}_std', 0) for m in models]
        
        x = np.arange(len(model_names))
        
        bars = ax.bar(x, values, yerr=errors if any(e > 0 for e in errors) else None,
                     capsize=5, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        
        # Value labels
        for bar, val, err in zip(bars, values, errors):
            label = f'{val:.3f}'
            if err > 0:
                label += f'±{err:.3f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title(config.title or f'Model Comparison - {metric.title()}')
        
        self._figures['model_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    # Convenience methods
    def plot_results(self, results: Dict[str, Any], model_type: str = "ML",
                    title: str = None) -> Dict[str, str]:
        """Plot complete ML results.
        
        Args:
            results: Dictionary with accuracy, precision, recall, f1, etc.
            model_type: Name of the model
            title: Optional title
        
        Returns:
            Dictionary mapping plot names to base64 strings
        """
        data = {**results, 'model_type': model_type}
        
        config = PlotConfig(
            plot_type=PlotType.METRICS_BAR,
            title=title or f'{model_type} Performance'
        )
        
        self.create_plot(PlotType.METRICS_BAR, data, config)
        
        # Add confusion matrix if available
        if 'confusion_matrix' in results:
            cm_config = PlotConfig(plot_type=PlotType.CONFUSION_MATRIX)
            self.create_plot(PlotType.CONFUSION_MATRIX, results, cm_config)
        
        # Add feature importance if available
        if 'feature_names' in results and 'importance' in results:
            fi_config = PlotConfig(plot_type=PlotType.FEATURE_IMPORTANCE)
            self.create_plot(PlotType.FEATURE_IMPORTANCE, results, fi_config)
        
        return self.get_all_base64()
