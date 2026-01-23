"""Metrics Visualizer - Professional plotting for training metrics.

Generates publication-quality visualizations for training reports.
"""

import io
import base64
import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


class MetricsVisualizer:
    """Generate professional visualizations for training metrics.
    
    Features:
    - Training curves (loss, accuracy)
    - Confusion matrix heatmap
    - ROC and PR curves
    - Per-class performance bars
    - FL client comparison
    - Learning rate schedule
    """
    
    # Publication-quality settings
    FIGURE_DPI = 150
    EXPORT_DPI = 300
    FONT_SIZE = 11
    TITLE_SIZE = 14
    
    # Color schemes
    COLORS = {
        "train": "#2E86AB",
        "val": "#E94F37",
        "test": "#44AF69",
        "primary": "#2E86AB",
        "secondary": "#E94F37",
        "accent": "#44AF69",
    }
    
    def __init__(self, tracker: 'MetricsTracker'):
        """Initialize visualizer with a metrics tracker.
        
        Args:
            tracker: MetricsTracker instance with logged metrics
        """
        self.tracker = tracker
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style for professional plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': self.FONT_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.labelsize': self.FONT_SIZE,
            'xtick.labelsize': self.FONT_SIZE - 1,
            'ytick.labelsize': self.FONT_SIZE - 1,
            'legend.fontsize': self.FONT_SIZE - 1,
            'figure.titlesize': self.TITLE_SIZE + 2,
        })
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.FIGURE_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> str:
        """Plot training and validation loss/accuracy curves.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.epoch_metrics:
            return ""
        
        epochs = [m.epoch for m in self.tracker.epoch_metrics]
        train_loss = [m.train_loss for m in self.tracker.epoch_metrics]
        train_acc = [m.train_accuracy for m in self.tracker.epoch_metrics]
        val_loss = [m.val_loss for m in self.tracker.epoch_metrics if m.val_loss is not None]
        val_acc = [m.val_accuracy for m in self.tracker.epoch_metrics if m.val_accuracy is not None]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(epochs, train_loss, color=self.COLORS["train"], linewidth=2, label='Train Loss', marker='o', markersize=4)
        if val_loss:
            val_epochs = epochs[:len(val_loss)]
            ax1.plot(val_epochs, val_loss, color=self.COLORS["val"], linewidth=2, label='Val Loss', marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = axes[1]
        ax2.plot(epochs, train_acc, color=self.COLORS["train"], linewidth=2, label='Train Accuracy', marker='o', markersize=4)
        if val_acc:
            val_epochs = epochs[:len(val_acc)]
            ax2.plot(val_epochs, val_acc, color=self.COLORS["val"], linewidth=2, label='Val Accuracy', marker='s', markersize=4)
            
            # Mark best validation accuracy
            best_idx = np.argmax(val_acc)
            ax2.axvline(x=val_epochs[best_idx], color='green', linestyle='--', alpha=0.7, label=f'Best: {val_acc[best_idx]:.4f}')
            ax2.scatter([val_epochs[best_idx]], [val_acc[best_idx]], color='green', s=100, zorder=5, marker='*')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> str:
        """Plot confusion matrix heatmap.
        
        Returns:
            Base64 encoded PNG image
        """
        if self.tracker.confusion_matrix is None:
            return ""
        
        cm = self.tracker.confusion_matrix
        class_names = self.tracker.class_names
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = "white" if cm_normalized[i, j] > thresh else "black"
                text = f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_roc_curves(self, save_path: Optional[str] = None) -> str:
        """Plot ROC curves for all classes.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.roc_curves:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.tracker.roc_curves)))
        
        for i, (class_name, data) in enumerate(self.tracker.roc_curves.items()):
            ax.plot(data["fpr"], data["tpr"], color=colors[i], linewidth=2,
                   label=f'{class_name} (AUC = {data["auc"]:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_pr_curves(self, save_path: Optional[str] = None) -> str:
        """Plot Precision-Recall curves for all classes.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.pr_curves:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.tracker.pr_curves)))
        
        for i, (class_name, data) in enumerate(self.tracker.pr_curves.items()):
            ax.plot(data["recall"], data["precision"], color=colors[i], linewidth=2,
                   label=f'{class_name} (AP = {data["ap"]:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_class_performance(self, save_path: Optional[str] = None) -> str:
        """Plot per-class performance metrics.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.class_metrics:
            return ""
        
        class_names = [c.class_name for c in self.tracker.class_metrics]
        precision = [c.precision for c in self.tracker.class_metrics]
        recall = [c.recall for c in self.tracker.class_metrics]
        f1 = [c.f1_score for c in self.tracker.class_metrics]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color=self.COLORS["primary"])
        bars2 = ax.bar(x, recall, width, label='Recall', color=self.COLORS["secondary"])
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=self.COLORS["accent"])
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_fl_rounds(self, save_path: Optional[str] = None) -> str:
        """Plot federated learning round metrics.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.fl_round_metrics:
            return ""
        
        rounds = [m.round_num for m in self.tracker.fl_round_metrics]
        global_acc = [m.global_accuracy for m in self.tracker.fl_round_metrics]
        global_loss = [m.global_loss for m in self.tracker.fl_round_metrics]
        avg_client_acc = [m.avg_client_accuracy for m in self.tracker.fl_round_metrics]
        min_client_acc = [m.min_client_accuracy for m in self.tracker.fl_round_metrics]
        max_client_acc = [m.max_client_accuracy for m in self.tracker.fl_round_metrics]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot with client variance
        ax1 = axes[0]
        ax1.plot(rounds, global_acc, color=self.COLORS["primary"], linewidth=2, label='Global Accuracy', marker='o', markersize=4)
        ax1.fill_between(rounds, min_client_acc, max_client_acc, alpha=0.2, color=self.COLORS["primary"], label='Client Range')
        ax1.plot(rounds, avg_client_acc, color=self.COLORS["secondary"], linewidth=1.5, linestyle='--', label='Avg Client Accuracy')
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('FL Global & Client Accuracy', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Loss plot
        ax2 = axes[1]
        ax2.plot(rounds, global_loss, color=self.COLORS["primary"], linewidth=2, label='Global Loss', marker='o', markersize=4)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.set_title('FL Global Loss', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def plot_learning_rate(self, save_path: Optional[str] = None) -> str:
        """Plot learning rate schedule.
        
        Returns:
            Base64 encoded PNG image
        """
        if not self.tracker.epoch_metrics:
            return ""
        
        epochs = [m.epoch for m in self.tracker.epoch_metrics]
        lrs = [m.learning_rate for m in self.tracker.epoch_metrics]
        
        if all(lr == 0 for lr in lrs):
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(epochs, lrs, color=self.COLORS["primary"], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
    
    def generate_all_plots(self) -> Dict[str, str]:
        """Generate all available plots.
        
        Returns:
            Dictionary mapping plot names to base64 encoded images
        """
        plots = {}
        
        # Training curves
        training_curves = self.plot_training_curves()
        if training_curves:
            plots["training_curves"] = training_curves
        
        # Confusion matrix
        confusion_matrix = self.plot_confusion_matrix()
        if confusion_matrix:
            plots["confusion_matrix"] = confusion_matrix
        
        # ROC curves
        roc_curves = self.plot_roc_curves()
        if roc_curves:
            plots["roc_curves"] = roc_curves
        
        # PR curves
        pr_curves = self.plot_pr_curves()
        if pr_curves:
            plots["pr_curves"] = pr_curves
        
        # Class performance
        class_performance = self.plot_class_performance()
        if class_performance:
            plots["class_performance"] = class_performance
        
        # FL rounds
        fl_rounds = self.plot_fl_rounds()
        if fl_rounds:
            plots["fl_rounds"] = fl_rounds
        
        # Learning rate
        lr_schedule = self.plot_learning_rate()
        if lr_schedule:
            plots["learning_rate"] = lr_schedule
        
        return plots
    
    def generate_summary_figure(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive summary figure with multiple subplots.
        
        Returns:
            Base64 encoded PNG image
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Training curves (top left, spans 2 columns)
        if self.tracker.epoch_metrics:
            ax1 = fig.add_subplot(gs[0, :2])
            epochs = [m.epoch for m in self.tracker.epoch_metrics]
            train_acc = [m.train_accuracy for m in self.tracker.epoch_metrics]
            val_acc = [m.val_accuracy for m in self.tracker.epoch_metrics if m.val_accuracy is not None]
            
            ax1.plot(epochs, train_acc, color=self.COLORS["train"], linewidth=2, label='Train')
            if val_acc:
                ax1.plot(epochs[:len(val_acc)], val_acc, color=self.COLORS["val"], linewidth=2, label='Val')
            ax1.set_title('Training Progress', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Confusion matrix (top right)
        if self.tracker.confusion_matrix is not None:
            ax2 = fig.add_subplot(gs[0, 2])
            cm = self.tracker.confusion_matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            im = ax2.imshow(cm_norm, cmap='Blues')
            ax2.set_title('Confusion Matrix', fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
        
        # Class performance (bottom left)
        if self.tracker.class_metrics:
            ax3 = fig.add_subplot(gs[1, 0])
            class_names = [c.class_name[:10] for c in self.tracker.class_metrics]
            f1_scores = [c.f1_score for c in self.tracker.class_metrics]
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
            ax3.barh(class_names, f1_scores, color=colors)
            ax3.set_title('F1-Score by Class', fontweight='bold')
            ax3.set_xlim([0, 1])
            ax3.grid(True, alpha=0.3, axis='x')
        
        # ROC curves (bottom middle)
        if self.tracker.roc_curves:
            ax4 = fig.add_subplot(gs[1, 1])
            for class_name, data in self.tracker.roc_curves.items():
                ax4.plot(data["fpr"], data["tpr"], linewidth=1.5, label=f'{class_name[:8]}')
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax4.set_title('ROC Curves', fontweight='bold')
            ax4.set_xlabel('FPR')
            ax4.set_ylabel('TPR')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        # Summary stats (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        summary = self.tracker.get_summary()
        
        stats_text = f"""
Training Summary
{'='*30}

Best Val Accuracy: {summary['best_val_accuracy']:.4f}
Best Epoch: {summary['best_val_epoch']}

Final Train Acc: {summary['final_train_accuracy']:.4f if summary['final_train_accuracy'] else 'N/A'}
Final Val Acc: {summary['final_val_accuracy']:.4f if summary['final_val_accuracy'] else 'N/A'}

Total Epochs: {summary['total_epochs']}
Training Time: {summary['total_training_time_seconds']:.1f}s

Device: {summary['device']}
Classes: {summary['num_classes']}
"""
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Training Report: {self.tracker.job_id}', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=self.EXPORT_DPI, bbox_inches='tight')
        
        return self._fig_to_base64(fig)
