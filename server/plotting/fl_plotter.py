"""Federated Learning Plotter - Specialized visualization for FL training.

Provides visualizations for federated learning experiments:
- Convergence across rounds
- Client drift analysis
- Aggregation weight distribution
- Per-client performance comparison
- Communication efficiency
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


class FLPlotter(BasePlotter):
    """Plotter specialized for federated learning experiments.
    
    Handles round-based FL training with support for:
    - Global model convergence
    - Client heterogeneity visualization
    - Communication and computation analysis
    - Byzantine client detection visualization
    """
    
    @property
    def supported_plots(self) -> List[PlotType]:
        return [
            PlotType.CONVERGENCE,
            PlotType.CLIENT_DRIFT,
            PlotType.AGGREGATION_WEIGHTS,
            PlotType.ROUND_METRICS,
            PlotType.CONFUSION_MATRIX,
            PlotType.ROC_CURVE,
            PlotType.PR_CURVE,
            PlotType.CLASS_METRICS,
            PlotType.TRIAL_COMPARISON,
        ]
    
    def create_plot(self, plot_type: PlotType, data: Dict[str, Any],
                   config: Optional[PlotConfig] = None) -> Tuple[Figure, str]:
        """Create an FL-specific plot.
        
        Args:
            plot_type: Type of plot to create
            data: Data dictionary containing FL metrics
            config: Optional plot configuration
        
        Returns:
            Tuple of (Figure, base64 string)
        """
        if config is None:
            config = self.get_config(plot_type)
        
        if plot_type == PlotType.CONVERGENCE:
            return self._plot_convergence(data, config)
        elif plot_type == PlotType.CLIENT_DRIFT:
            return self._plot_client_drift(data, config)
        elif plot_type == PlotType.AGGREGATION_WEIGHTS:
            return self._plot_aggregation_weights(data, config)
        elif plot_type == PlotType.ROUND_METRICS:
            return self._plot_round_metrics(data, config)
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
            raise ValueError(f"Unsupported plot type for FLPlotter: {plot_type}")
    
    def _plot_convergence(self, data: Dict[str, Any],
                         config: PlotConfig) -> Tuple[Figure, str]:
        """Plot FL convergence across rounds with client variance."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ThemeManager.get_colors(4)
        
        rounds = data.get('rounds', [])
        if not rounds and 'avg_accuracy' in data:
            rounds = list(range(1, len(data['avg_accuracy']) + 1))
        
        # Accuracy convergence with variance band
        ax1 = axes[0]
        avg_acc = data.get('avg_accuracy', [])
        min_acc = data.get('min_accuracy', [])
        max_acc = data.get('max_accuracy', [])
        std_acc = data.get('std_accuracy', [])
        
        if avg_acc:
            ax1.plot(rounds, avg_acc, 'o-', color=colors[0], linewidth=2, 
                    markersize=4, label='Average')
            
            # Variance band
            if min_acc and max_acc:
                ax1.fill_between(rounds, min_acc, max_acc, alpha=0.2, color=colors[0])
                ax1.plot(rounds, min_acc, '--', color=colors[1], alpha=0.5, linewidth=1, label='Min')
                ax1.plot(rounds, max_acc, '--', color=colors[2], alpha=0.5, linewidth=1, label='Max')
            elif std_acc:
                lower = [a - s for a, s in zip(avg_acc, std_acc)]
                upper = [a + s for a, s in zip(avg_acc, std_acc)]
                ax1.fill_between(rounds, lower, upper, alpha=0.2, color=colors[0])
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('FL Convergence')
        ax1.legend(loc='lower right')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, alpha=0.3)
        
        # Client participation
        ax2 = axes[1]
        num_clients = data.get('num_clients', [])
        if num_clients:
            ax2.bar(rounds, num_clients, color=colors[3], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Number of Clients')
            ax2.set_title('Client Participation per Round')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            # If no client data, show loss convergence
            avg_loss = data.get('avg_loss', [])
            if avg_loss:
                ax2.plot(rounds, avg_loss, 'o-', color=colors[0], linewidth=2, markersize=4)
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss Convergence')
                ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if config.title:
            fig.suptitle(config.title, fontsize=12, y=1.02)
        
        fig.tight_layout()
        
        self._figures['convergence'] = fig
        return fig, self._finalize_figure(fig, axes[0], config)
    
    def _plot_client_drift(self, data: Dict[str, Any],
                          config: PlotConfig) -> Tuple[Figure, str]:
        """Plot client model drift from global model."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(len(data.get('clients', [])))
        
        rounds = data.get('rounds', [])
        clients = data.get('clients', [])
        
        for i, client in enumerate(clients):
            client_id = client.get('id', f'Client {i+1}')
            drift = client.get('drift', [])
            if drift:
                ax.plot(rounds[:len(drift)], drift, '-', color=colors[i],
                       linewidth=1.5, alpha=0.7, label=client_id)
        
        # Mean drift
        all_drifts = [c.get('drift', []) for c in clients if c.get('drift')]
        if all_drifts:
            min_len = min(len(d) for d in all_drifts)
            drift_array = np.array([d[:min_len] for d in all_drifts])
            mean_drift = np.mean(drift_array, axis=0)
            ax.plot(rounds[:min_len], mean_drift, 'k-', linewidth=3, 
                   label='Mean Drift', zorder=10)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Model Drift (L2 Distance)')
        ax.set_title(config.title or 'Client Model Drift')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        self._figures['client_drift'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_aggregation_weights(self, data: Dict[str, Any],
                                  config: PlotConfig) -> Tuple[Figure, str]:
        """Plot aggregation weight distribution across clients."""
        fig, ax = self._create_figure(config)
        
        weights = data.get('weights', {})
        client_ids = list(weights.keys())
        weight_values = list(weights.values())
        
        if SEABORN_AVAILABLE:
            colors = ThemeManager.get_colors(len(client_ids))
            ax.pie(weight_values, labels=client_ids, autopct='%1.1f%%',
                  colors=colors, explode=[0.02] * len(client_ids))
        else:
            ax.bar(range(len(client_ids)), weight_values, 
                  color=ThemeManager.get_colors(1)[0], alpha=0.8)
            ax.set_xticks(range(len(client_ids)))
            ax.set_xticklabels(client_ids, rotation=45, ha='right')
            ax.set_ylabel('Weight')
        
        ax.set_title(config.title or 'Aggregation Weights')
        
        self._figures['aggregation_weights'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    def _plot_round_metrics(self, data: Dict[str, Any],
                           config: PlotConfig) -> Tuple[Figure, str]:
        """Plot detailed metrics per round."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colors = ThemeManager.get_colors(4)
        
        rounds = data.get('rounds', [])
        
        # Accuracy
        ax1 = axes[0, 0]
        avg_acc = data.get('avg_accuracy', [])
        if avg_acc:
            ax1.plot(rounds, avg_acc, 'o-', color=colors[0], linewidth=2)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Global Accuracy')
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Loss
        ax2 = axes[0, 1]
        avg_loss = data.get('avg_loss', [])
        if avg_loss:
            ax2.plot(rounds, avg_loss, 'o-', color=colors[1], linewidth=2)
            ax2.set_ylabel('Loss')
            ax2.set_title('Global Loss')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Communication
        ax3 = axes[1, 0]
        comm_bytes = data.get('communication_bytes', [])
        if comm_bytes:
            comm_mb = [b / (1024 * 1024) for b in comm_bytes]
            ax3.bar(rounds, comm_mb, color=colors[2], alpha=0.7)
            ax3.set_ylabel('Communication (MB)')
            ax3.set_title('Communication per Round')
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Aggregation time
        ax4 = axes[1, 1]
        agg_time = data.get('aggregation_time_ms', [])
        if agg_time:
            ax4.bar(rounds, agg_time, color=colors[3], alpha=0.7)
            ax4.set_ylabel('Time (ms)')
            ax4.set_title('Aggregation Time per Round')
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for ax in axes.flat:
            ax.set_xlabel('Round')
            ax.grid(True, alpha=0.3)
        
        if config.title:
            fig.suptitle(config.title, fontsize=14, y=1.02)
        
        fig.tight_layout()
        
        self._figures['round_metrics'] = fig
        return fig, self._finalize_figure(fig, axes[0, 0], config)
    
    def _plot_confusion_matrix(self, data: Dict[str, Any],
                              config: PlotConfig) -> Tuple[Figure, str]:
        """Plot confusion matrix."""
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
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
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
        """Plot PR curves."""
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
        """Plot comparison across FL experiment trials."""
        fig, ax = self._create_figure(config)
        colors = ThemeManager.get_colors(2)
        
        trials = data.get('trials', [])
        metric = data.get('metric', 'final_accuracy')
        
        values = [t.get(metric, t.get('accuracy', 0)) for t in trials]
        trial_names = [f"Trial {i+1}" for i in range(len(trials))]
        x = np.arange(len(trial_names))
        
        bars = ax.bar(x, values, color=colors[0], alpha=0.7, edgecolor='black')
        
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
        ax.set_title(config.title or f'FL Trial Comparison')
        
        self._figures['trial_comparison'] = fig
        return fig, self._finalize_figure(fig, ax, config)
    
    # Convenience methods
    def plot_fl_experiment(self, rounds_data: List[Dict[str, Any]],
                          algorithm: str = "FedAvg",
                          title: str = None) -> Dict[str, str]:
        """Plot complete FL experiment results.
        
        Args:
            rounds_data: List of per-round metrics dictionaries
            algorithm: FL algorithm name
            title: Optional title
        
        Returns:
            Dictionary mapping plot names to base64 strings
        """
        # Aggregate round data
        data = {
            'rounds': [r.get('round_num', i+1) for i, r in enumerate(rounds_data)],
            'avg_accuracy': [r.get('avg_accuracy', 0) for r in rounds_data],
            'avg_loss': [r.get('avg_loss', 0) for r in rounds_data],
            'min_accuracy': [r.get('min_accuracy', 0) for r in rounds_data],
            'max_accuracy': [r.get('max_accuracy', 0) for r in rounds_data],
            'std_accuracy': [r.get('std_accuracy', 0) for r in rounds_data],
            'num_clients': [r.get('num_clients', 0) for r in rounds_data],
            'communication_bytes': [r.get('communication_bytes', 0) for r in rounds_data],
            'aggregation_time_ms': [r.get('aggregation_time_ms', 0) for r in rounds_data],
        }
        
        # Convergence plot
        conv_config = PlotConfig(
            plot_type=PlotType.CONVERGENCE,
            title=title or f'{algorithm} Convergence'
        )
        self.create_plot(PlotType.CONVERGENCE, data, conv_config)
        
        # Round metrics
        metrics_config = PlotConfig(
            plot_type=PlotType.ROUND_METRICS,
            title=f'{algorithm} Round Metrics'
        )
        self.create_plot(PlotType.ROUND_METRICS, data, metrics_config)
        
        return self.get_all_base64()
