"""Deep Learning Plotter - Specialized visualization for DL training.

Provides epoch-based training visualizations including:
- Learning curves (loss/accuracy over epochs)
- Learning rate schedules
- Gradient flow analysis
- Multi-trial aggregation with confidence intervals
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
    from matplotlib.ticker import MaxNLocator

if SEABORN_AVAILABLE:
    import seaborn as sns

logger = logging.getLogger(__name__)


class DLPlotter(BasePlotter):
    """Plotter specialized for deep learning training visualization.
    
    Handles epoch-based training with support for:
    - Training/validation loss and accuracy curves
    - Learning rate schedules
    - Multi-trial aggregation with statistical analysis
    - Live updating during training
    """
    
    @property
    def supported_plots(self) -> List[PlotType]:
        return [
            PlotType.LEARNING_CURVE,
            PlotType.LOSS_CURVE,
            PlotType.ACCURACY_CURVE,
            PlotType.LR_SCHEDULE,
            PlotType.GRADIENT_FLOW,
            PlotType.CONFUSION_MATRIX,
            PlotType.ROC_CURVE,
            PlotType.PR_CURVE,
            PlotType.CLASS_METRICS,
            PlotType.TRIAL_COMPARISON,
        ]
    
    def create_plot(self, plot_type: PlotType, data: Dict[str, Any],
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        """Create a DL-specific plot.
        
        Args:
            plot_type: Type of plot to create
            data: Data dictionary containing:
                - epochs: List of epoch numbers
                - train_loss/val_loss: Loss values
                - train_acc/val_acc: Accuracy values
                - learning_rates: LR values (for LR schedule)
                - trials: List of trial data (for multi-trial)
            config: Optional plot configuration
        
        Returns:
            Tuple of (Figure, base64 string)
        """
        if config is None:
            config = self.get_config(plot_type)
        
        if plot_type == PlotType.LEARNING_CURVE:
            return self._plot_learning_curves(data, config)
        elif plot_type == PlotType.LOSS_CURVE:
            return self._plot_loss_curve(data, config)
        elif plot_type == PlotType.ACCURACY_CURVE:
            return self._plot_accuracy_curve(data, config)
        elif plot_type == PlotType.LR_SCHEDULE:
            return self._plot_lr_schedule(data, config)
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
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_learning_curves(self, data: Dict[str, Any], 
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot combined loss and accuracy learning curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ThemeManager.get_colors(4)
        
        epochs = data.get('epochs', [])
        if not epochs and 'train_loss' in data:
            epochs = list(range(1, len(data['train_loss']) + 1))
        
        # Loss subplot
        ax1 = axes[0]
        self._plot_metric_curve(
            ax1, epochs, data, 'train_loss', 'val_loss',
            'Train Loss', 'Val Loss', colors[0], colors[1],
            ylabel='Loss', title='Training Loss'
        )
        
        # Accuracy subplot
        ax2 = axes[1]
        self._plot_metric_curve(
            ax2, epochs, data, 'train_acc', 'val_acc',
            'Train Accuracy', 'Val Accuracy', colors[0], colors[1],
            ylabel='Accuracy', title='Training Accuracy'
        )
        
        # Mark best epoch
        if 'val_acc' in data and data['val_acc']:
            val_acc = data['val_acc']
            best_idx = np.argmax(val_acc)
            best_epoch = epochs[best_idx] if epochs else best_idx + 1
            ax2.axvline(x=best_epoch, color=colors[2], linestyle='--', 
                       alpha=0.7, label=f'Best (Epoch {best_epoch})')
            ax2.scatter([best_epoch], [val_acc[best_idx]], s=100, 
                       c=colors[2], marker='*', zorder=5)
            ax2.legend()
        
        if config.title:
            fig.suptitle(config.title, fontsize=12, y=1.02)
        
        fig.tight_layout()
        
        # Store figure
        self._figures['learning_curves'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_metric_curve(self, ax: Axes, epochs: List[int], data: Dict,
                          train_key: str, val_key: str,
                          train_label: str, val_label: str,
                          train_color: str, val_color: str,
                          ylabel: str, title: str):
        """Helper to plot a single metric curve with optional CI."""
        # Check for multi-trial data
        if 'trials' in data:
            self._plot_multi_trial_curve(
                ax, epochs, data['trials'], train_key, val_key,
                train_label, val_label, train_color, val_color
            )
        else:
            # Single trial
            if train_key in data and data[train_key]:
                ax.plot(epochs, data[train_key], '-', color=train_color,
                       linewidth=2, label=train_label)
            
            if val_key in data and data[val_key]:
                ax.plot(epochs[:len(data[val_key])], data[val_key], '--',
                       color=val_color, linewidth=2, label=val_label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_multi_trial_curve(self, ax: Axes, epochs: List[int],
                                trials: List[Dict], train_key: str, val_key: str,
                                train_label: str, val_label: str,
                                train_color: str, val_color: str):
        """Plot curves aggregated from multiple trials with CI."""
        # Collect data from trials
        train_data = [t.get(train_key, []) for t in trials if t.get(train_key)]
        val_data = [t.get(val_key, []) for t in trials if t.get(val_key)]
        
        if train_data:
            min_len = min(len(d) for d in train_data)
            train_array = np.array([d[:min_len] for d in train_data])
            train_mean = np.mean(train_array, axis=0)
            train_std = np.std(train_array, axis=0, ddof=1) if len(train_data) > 1 else np.zeros_like(train_mean)
            
            epochs_plot = epochs[:min_len] if epochs else list(range(1, min_len + 1))
            
            ax.plot(epochs_plot, train_mean, '-', color=train_color,
                   linewidth=2, label=f'{train_label} (n={len(train_data)})')
            ax.fill_between(epochs_plot, train_mean - train_std, train_mean + train_std,
                           color=train_color, alpha=0.2)
        
        if val_data:
            min_len = min(len(d) for d in val_data)
            val_array = np.array([d[:min_len] for d in val_data])
            val_mean = np.mean(val_array, axis=0)
            val_std = np.std(val_array, axis=0, ddof=1) if len(val_data) > 1 else np.zeros_like(val_mean)
            
            epochs_plot = epochs[:min_len] if epochs else list(range(1, min_len + 1))
            
            ax.plot(epochs_plot, val_mean, '--', color=val_color,
                   linewidth=2, label=f'{val_label} (n={len(val_data)})')
            ax.fill_between(epochs_plot, val_mean - val_std, val_mean + val_std,
                           color=val_color, alpha=0.2)
    
    def _plot_loss_curve(self, data: Dict[str, Any],
                        config: PlotConfig) -> Tuple[Figure, str]:
        """Plot loss curve only."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        epochs = data.get('epochs', [])
        if not epochs and 'train_loss' in data:
            epochs = list(range(1, len(data['train_loss']) + 1))
        
        self._plot_metric_curve(
            ax, epochs, data, 'train_loss', 'val_loss',
            'Train Loss', 'Val Loss', colors[0], colors[1],
            ylabel='Loss', title=config.title or 'Training Loss'
        )
        
        self._figures['loss_curve'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_accuracy_curve(self, data: Dict[str, Any],
                            config: PlotConfig) -> Tuple[Figure, str]:
        """Plot accuracy curve only."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(3)
        
        epochs = data.get('epochs', [])
        if not epochs and 'train_acc' in data:
            epochs = list(range(1, len(data['train_acc']) + 1))
        
        self._plot_metric_curve(
            ax, epochs, data, 'train_acc', 'val_acc',
            'Train Accuracy', 'Val Accuracy', colors[0], colors[1],
            ylabel='Accuracy', title=config.title or 'Training Accuracy'
        )
        
        # Mark best
        if 'val_acc' in data and data['val_acc']:
            val_acc = data['val_acc']
            best_idx = np.argmax(val_acc)
            best_epoch = epochs[best_idx] if epochs else best_idx + 1
            ax.axvline(x=best_epoch, color=colors[2], linestyle='--',
                      alpha=0.7, label=f'Best (Epoch {best_epoch})')
            ax.scatter([best_epoch], [val_acc[best_idx]], s=100,
                      c=colors[2], marker='*', zorder=5)
            ax.legend()
        
        self._figures['accuracy_curve'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_lr_schedule(self, data: Dict[str, Any],
                         config: PlotConfig) -> Tuple[Figure, str]:
        """Plot learning rate schedule."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(1)
        
        epochs = data.get('epochs', [])
        lrs = data.get('learning_rates', data.get('lr', []))
        
        if not epochs:
            epochs = list(range(1, len(lrs) + 1))
        
        ax.plot(epochs, lrs, 'o-', color=colors[0], linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(config.title or 'Learning Rate Schedule')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        self._figures['lr_schedule'] = fig
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
                       xticklabels=class_names, yticklabels=class_names, ax=ax,
                       cbar_kws={'label': 'Proportion' if normalize else 'Count'})
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
            
            plt.colorbar(im, ax=ax, label='Proportion' if normalize else 'Count')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(config.title or 'Confusion Matrix')
        
        self._figures['confusion_matrix'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_roc_curves(self, data: Dict[str, Any],
                        config: PlotConfig) -> Tuple[Figure, str]:
        """Plot ROC curves for all classes."""
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
        """Plot per-class precision, recall, F1 bar chart."""
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
        """Plot comparison across multiple trials."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        trials = data.get('trials', [])
        metric = data.get('metric', 'accuracy')
        
        if metric == 'accuracy':
            values = [t.get('final_accuracy', t.get('accuracy', 0)) for t in trials]
            ylabel = 'Accuracy'
        elif metric == 'f1':
            values = [t.get('f1_score', 0) for t in trials]
            ylabel = 'F1-Score'
        else:
            values = [t.get(metric, 0) for t in trials]
            ylabel = metric.replace('_', ' ').title()
        
        trial_names = [f"Trial {i+1}" for i in range(len(trials))]
        x = np.arange(len(trial_names))
        
        bars = ax.bar(x, values, color=colors[0], alpha=0.7, edgecolor='black')
        
        # Mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color=colors[1], linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        
        # Std band
        if len(values) > 1:
            std_val = np.std(values, ddof=1)
            ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2,
                      color=colors[1], label=f'±1 Std: {std_val:.4f}')
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Trial')
        ax.set_xticks(x)
        ax.set_xticklabels(trial_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_title(config.title or f'Trial Comparison - {ylabel}')
        
        self._figures['trial_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    # Convenience methods for common use cases
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = None) -> Dict[str, str]:
        """Plot complete training history.
        
        Args:
            history: Dictionary with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
            title: Optional title
        
        Returns:
            Dictionary mapping plot names to base64 strings
        """
        # Normalize key names
        data = {}
        key_mapping = {
            'loss': 'train_loss',
            'accuracy': 'train_acc',
            'acc': 'train_acc',
            'val_loss': 'val_loss',
            'val_accuracy': 'val_acc',
            'val_acc': 'val_acc',
            'lr': 'learning_rates',
            'learning_rate': 'learning_rates',
        }
        
        for key, value in history.items():
            mapped_key = key_mapping.get(key, key)
            data[mapped_key] = value
        
        config = PlotConfig(
            plot_type=PlotType.LEARNING_CURVE,
            title=title,
            figsize=(12, 5)
        )
        
        self.create_plot(PlotType.LEARNING_CURVE, data, config)
        
        # Also create LR schedule if available
        if 'learning_rates' in data:
            lr_config = PlotConfig(plot_type=PlotType.LR_SCHEDULE)
            self.create_plot(PlotType.LR_SCHEDULE, data, lr_config)
        
        return self.get_all_base64()
