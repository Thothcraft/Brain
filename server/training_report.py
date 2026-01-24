"""Training Report and Visualization System.

This module provides comprehensive performance tracking and visualization:
- Detailed metrics collection during training
- Publication-quality plots with matplotlib
- Exportable reports in multiple formats
- Shareable read-only view links

Inspired by Flower FL visualization: https://flower.ai/docs/framework/how-to-visualize-results.html
"""

import os
import io
import json
import uuid
import base64
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, plotting disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ============================================================================
# METRICS DATA STRUCTURES
# ============================================================================

@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.001
    batch_time_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ClassMetrics:
    """Per-class metrics."""
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of samples


@dataclass
class ConfusionMatrixData:
    """Confusion matrix data."""
    matrix: List[List[int]]
    class_names: List[str]
    normalized: bool = False


@dataclass
class ROCCurveData:
    """ROC curve data for each class."""
    class_name: str
    fpr: List[float]  # False positive rates
    tpr: List[float]  # True positive rates
    auc: float


@dataclass
class PRCurveData:
    """Precision-Recall curve data."""
    class_name: str
    precision: List[float]
    recall: List[float]
    ap: float  # Average precision


@dataclass
class FLRoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    num_clients: int
    avg_loss: float
    avg_accuracy: float
    min_accuracy: float
    max_accuracy: float
    std_accuracy: float
    aggregation_time_ms: float
    communication_bytes: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TrainingReport:
    """Complete training report with all metrics and visualizations."""
    # Identification
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    user_id: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Training configuration
    model_type: str = ""
    training_mode: str = "central"  # central, federated
    dataset_name: str = ""
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    
    # Training parameters
    epochs: int = 0
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_time_seconds: float = 0.0
    preprocessing_time_seconds: float = 0.0
    training_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    
    # Epoch-by-epoch metrics
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # Final evaluation metrics
    final_train_loss: float = 0.0
    final_train_accuracy: float = 0.0
    final_val_loss: Optional[float] = None
    final_val_accuracy: Optional[float] = None
    final_test_loss: Optional[float] = None
    final_test_accuracy: Optional[float] = None
    
    # Best metrics
    best_val_accuracy: float = 0.0
    best_val_epoch: int = 0
    
    # Per-class metrics
    class_metrics: List[ClassMetrics] = field(default_factory=list)
    
    # Confusion matrix
    confusion_matrix: Optional[ConfusionMatrixData] = None
    
    # ROC curves (one per class for multi-class)
    roc_curves: List[ROCCurveData] = field(default_factory=list)
    
    # PR curves
    pr_curves: List[PRCurveData] = field(default_factory=list)
    
    # FL-specific metrics
    fl_rounds: List[FLRoundMetrics] = field(default_factory=list)
    fl_algorithm: Optional[str] = None
    fl_num_clients: int = 0
    
    # Hardware info
    device: str = "cpu"
    gpu_name: Optional[str] = None
    peak_memory_mb: float = 0.0
    
    # Data statistics
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    input_shape: Optional[List[int]] = None
    
    # Shareable link
    share_token: Optional[str] = None
    is_public: bool = False
    
    # Generated plots (base64 encoded)
    plots: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert nested dataclasses
        result["epoch_metrics"] = [asdict(m) for m in self.epoch_metrics]
        result["class_metrics"] = [asdict(m) for m in self.class_metrics]
        result["fl_rounds"] = [asdict(r) for r in self.fl_rounds]
        result["roc_curves"] = [asdict(r) for r in self.roc_curves]
        result["pr_curves"] = [asdict(p) for p in self.pr_curves]
        if self.confusion_matrix:
            result["confusion_matrix"] = asdict(self.confusion_matrix)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingReport':
        """Create from dictionary."""
        # Convert nested structures
        if "epoch_metrics" in data:
            data["epoch_metrics"] = [EpochMetrics(**m) for m in data["epoch_metrics"]]
        if "class_metrics" in data:
            data["class_metrics"] = [ClassMetrics(**m) for m in data["class_metrics"]]
        if "fl_rounds" in data:
            data["fl_rounds"] = [FLRoundMetrics(**r) for r in data["fl_rounds"]]
        if "roc_curves" in data:
            data["roc_curves"] = [ROCCurveData(**r) for r in data["roc_curves"]]
        if "pr_curves" in data:
            data["pr_curves"] = [PRCurveData(**p) for p in data["pr_curves"]]
        if "confusion_matrix" in data and data["confusion_matrix"]:
            data["confusion_matrix"] = ConfusionMatrixData(**data["confusion_matrix"])
        return cls(**data)


# ============================================================================
# PLOT GENERATION
# ============================================================================

class ReportPlotter:
    """Generate publication-quality plots for training reports."""
    
    # Publication-quality settings
    FIGURE_DPI = 300
    FONT_SIZE = 12
    TITLE_SIZE = 14
    LABEL_SIZE = 11
    TICK_SIZE = 10
    LINE_WIDTH = 2
    MARKER_SIZE = 6
    
    # Color palette (colorblind-friendly)
    COLORS = [
        '#0072B2',  # Blue
        '#D55E00',  # Orange
        '#009E73',  # Green
        '#CC79A7',  # Pink
        '#F0E442',  # Yellow
        '#56B4E9',  # Light blue
        '#E69F00',  # Dark orange
    ]
    
    def __init__(self):
        if MATPLOTLIB_AVAILABLE:
            self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for publication quality."""
        plt.rcParams.update({
            'font.size': self.FONT_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.labelsize': self.LABEL_SIZE,
            'xtick.labelsize': self.TICK_SIZE,
            'ytick.labelsize': self.TICK_SIZE,
            'legend.fontsize': self.TICK_SIZE,
            'figure.dpi': self.FIGURE_DPI,
            'savefig.dpi': self.FIGURE_DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': self.LINE_WIDTH,
            'lines.markersize': self.MARKER_SIZE,
        })
        
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette(self.COLORS)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.FIGURE_DPI, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    def plot_training_curves(self, report: TrainingReport) -> str:
        """Plot training and validation loss/accuracy curves."""
        if not MATPLOTLIB_AVAILABLE or not report.epoch_metrics:
            return ""
        
        epochs = [m.epoch for m in report.epoch_metrics]
        train_loss = [m.train_loss for m in report.epoch_metrics]
        train_acc = [m.train_accuracy for m in report.epoch_metrics]
        val_loss = [m.val_loss for m in report.epoch_metrics if m.val_loss is not None]
        val_acc = [m.val_accuracy for m in report.epoch_metrics if m.val_accuracy is not None]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1 = axes[0]
        ax1.plot(epochs, train_loss, 'o-', color=self.COLORS[0], label='Train Loss')
        if val_loss:
            ax1.plot(epochs[:len(val_loss)], val_loss, 's-', color=self.COLORS[1], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Accuracy plot
        ax2 = axes[1]
        ax2.plot(epochs, train_acc, 'o-', color=self.COLORS[0], label='Train Accuracy')
        if val_acc:
            ax2.plot(epochs[:len(val_acc)], val_acc, 's-', color=self.COLORS[1], label='Val Accuracy')
            # Mark best epoch
            best_idx = np.argmax(val_acc)
            ax2.axvline(x=epochs[best_idx], color=self.COLORS[2], linestyle='--', alpha=0.7, label=f'Best (Epoch {epochs[best_idx]})')
            ax2.scatter([epochs[best_idx]], [val_acc[best_idx]], s=100, c=self.COLORS[2], zorder=5, marker='*')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_confusion_matrix(self, report: TrainingReport) -> str:
        """Plot confusion matrix heatmap."""
        if not MATPLOTLIB_AVAILABLE or not report.confusion_matrix:
            return ""
        
        cm = np.array(report.confusion_matrix.matrix)
        class_names = report.confusion_matrix.class_names
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
        else:
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
            
            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_roc_curves(self, report: TrainingReport) -> str:
        """Plot ROC curves for all classes."""
        if not MATPLOTLIB_AVAILABLE or not report.roc_curves:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, roc in enumerate(report.roc_curves):
            color = self.COLORS[i % len(self.COLORS)]
            ax.plot(roc.fpr, roc.tpr, color=color, lw=2,
                   label=f'{roc.class_name} (AUC = {roc.auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_pr_curves(self, report: TrainingReport) -> str:
        """Plot Precision-Recall curves."""
        if not MATPLOTLIB_AVAILABLE or not report.pr_curves:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, pr in enumerate(report.pr_curves):
            color = self.COLORS[i % len(self.COLORS)]
            ax.plot(pr.recall, pr.precision, color=color, lw=2,
                   label=f'{pr.class_name} (AP = {pr.ap:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_class_metrics(self, report: TrainingReport) -> str:
        """Plot per-class precision, recall, F1 bar chart."""
        if not MATPLOTLIB_AVAILABLE or not report.class_metrics:
            return ""
        
        class_names = [m.class_name for m in report.class_metrics]
        precision = [m.precision for m in report.class_metrics]
        recall = [m.recall for m in report.class_metrics]
        f1 = [m.f1_score for m in report.class_metrics]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color=self.COLORS[0])
        bars2 = ax.bar(x, recall, width, label='Recall', color=self.COLORS[1])
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=self.COLORS[2])
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_fl_convergence(self, report: TrainingReport) -> str:
        """Plot FL convergence across rounds."""
        if not MATPLOTLIB_AVAILABLE or not report.fl_rounds:
            return ""
        
        rounds = [r.round_num for r in report.fl_rounds]
        avg_acc = [r.avg_accuracy for r in report.fl_rounds]
        min_acc = [r.min_accuracy for r in report.fl_rounds]
        max_acc = [r.max_accuracy for r in report.fl_rounds]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy with confidence band
        ax1 = axes[0]
        ax1.plot(rounds, avg_acc, 'o-', color=self.COLORS[0], label='Average')
        ax1.fill_between(rounds, min_acc, max_acc, alpha=0.2, color=self.COLORS[0])
        ax1.plot(rounds, min_acc, '--', color=self.COLORS[1], alpha=0.5, label='Min')
        ax1.plot(rounds, max_acc, '--', color=self.COLORS[2], alpha=0.5, label='Max')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('FL Convergence')
        ax1.legend()
        
        # Client participation
        ax2 = axes[1]
        num_clients = [r.num_clients for r in report.fl_rounds]
        ax2.bar(rounds, num_clients, color=self.COLORS[3], alpha=0.7)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Number of Clients')
        ax2.set_title('Client Participation per Round')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_learning_rate_schedule(self, report: TrainingReport) -> str:
        """Plot learning rate over epochs."""
        if not MATPLOTLIB_AVAILABLE or not report.epoch_metrics:
            return ""
        
        epochs = [m.epoch for m in report.epoch_metrics]
        lrs = [m.learning_rate for m in report.epoch_metrics]
        
        # Check if LR actually changes
        if len(set(lrs)) == 1:
            return ""  # Constant LR, no need to plot
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, lrs, 'o-', color=self.COLORS[0])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_timing_breakdown(self, report: TrainingReport) -> str:
        """Plot timing breakdown pie chart."""
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        times = {
            'Preprocessing': report.preprocessing_time_seconds,
            'Training': report.training_time_seconds,
            'Evaluation': report.evaluation_time_seconds,
        }
        
        # Filter out zero times
        times = {k: v for k, v in times.items() if v > 0}
        
        if not times:
            return ""
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        colors = [self.COLORS[i] for i in range(len(times))]
        wedges, texts, autotexts = ax.pie(
            times.values(),
            labels=times.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(times),
        )
        
        ax.set_title('Time Breakdown')
        
        # Add legend with actual times
        legend_labels = [f'{k}: {v:.1f}s' for k, v in times.items()]
        ax.legend(wedges, legend_labels, loc='lower right')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def generate_all_plots(self, report: TrainingReport) -> Dict[str, str]:
        """Generate all applicable plots for a report.
        
        Dynamically adapts based on model type:
        - DL models (CNN, LSTM, etc.): Show epoch-based training curves
        - ML models (SVM, RF, etc.): Show metrics comparison (no epoch curves)
        - FL models: Show convergence across rounds
        """
        plots = {}
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping plot generation")
            return plots
        
        # Detect model category for appropriate plotting
        model_category = self._detect_model_category(report.model_type)
        
        # Training curves - ONLY for DL models with epochs
        if model_category == "dl_iterative" and report.epoch_metrics:
            plot = self.plot_training_curves(report)
            if plot:
                plots["training_curves"] = plot
        elif model_category == "ml_single_fit":
            # For ML models, generate metrics summary instead of epoch curves
            plot = self.plot_ml_metrics_summary(report)
            if plot:
                plots["metrics_summary"] = plot
        
        # Confusion matrix - applicable to all classification models
        plot = self.plot_confusion_matrix(report)
        if plot:
            plots["confusion_matrix"] = plot
        
        # ROC curves - applicable to all classification models
        plot = self.plot_roc_curves(report)
        if plot:
            plots["roc_curves"] = plot
        
        # PR curves - applicable to all classification models
        plot = self.plot_pr_curves(report)
        if plot:
            plots["pr_curves"] = plot
        
        # Class metrics - applicable to all classification models
        plot = self.plot_class_metrics(report)
        if plot:
            plots["class_metrics"] = plot
        
        # FL convergence - only for federated learning
        if report.training_mode == "federated":
            plot = self.plot_fl_convergence(report)
            if plot:
                plots["fl_convergence"] = plot
        
        # Learning rate - only for DL models with epochs
        if model_category == "dl_iterative":
            plot = self.plot_learning_rate_schedule(report)
            if plot:
                plots["learning_rate"] = plot
        
        # Timing breakdown - applicable to all models
        plot = self.plot_timing_breakdown(report)
        if plot:
            plots["timing_breakdown"] = plot
        
        return plots
    
    def _detect_model_category(self, model_type: str) -> str:
        """Detect model category for appropriate plotting.
        
        Returns:
            'dl_iterative': DL models with epoch-based training
            'ml_single_fit': ML models without epochs
            'fl_rounds': Federated learning models
            'clustering': Unsupervised clustering models
        """
        model_lower = model_type.lower().replace('-', '_').replace(' ', '_')
        
        # ML models that don't have epoch-based training
        ml_single_fit = {
            'svm', 'svc', 'random_forest', 'rf', 'knn', 'k_nearest_neighbors',
            'logistic_regression', 'lr', 'decision_tree', 'dt', 'gradient_boosting',
            'gb', 'adaboost', 'naive_bayes', 'nb', 'linear_svm', 'rbf_svm',
            'xgboost', 'lightgbm', 'catboost'
        }
        
        # Clustering models
        clustering = {
            'kmeans', 'k_means', 'dbscan', 'hierarchical', 'agglomerative',
            'spectral_clustering', 'mean_shift', 'optics', 'birch'
        }
        
        # Check for FL
        if 'fed' in model_lower or 'fl_' in model_lower:
            return "fl_rounds"
        
        # Check for clustering
        for cluster_model in clustering:
            if cluster_model in model_lower:
                return "clustering"
        
        # Check for ML single-fit
        for ml_model in ml_single_fit:
            if ml_model in model_lower:
                return "ml_single_fit"
        
        # Default to DL iterative
        return "dl_iterative"
    
    def plot_ml_metrics_summary(self, report: TrainingReport) -> str:
        """Plot metrics summary for ML models (no epoch curves).
        
        Creates a bar chart showing final metrics instead of epoch-based curves.
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Collect metrics
        metrics = {
            'Accuracy': report.final_train_accuracy if report.final_train_accuracy else 0,
        }
        
        # Add per-class metrics if available
        if report.class_metrics:
            avg_precision = np.mean([m.precision for m in report.class_metrics])
            avg_recall = np.mean([m.recall for m in report.class_metrics])
            avg_f1 = np.mean([m.f1_score for m in report.class_metrics])
            metrics['Precision'] = avg_precision
            metrics['Recall'] = avg_recall
            metrics['F1-Score'] = avg_f1
        
        if report.final_val_accuracy:
            metrics['Val Accuracy'] = report.final_val_accuracy
        
        x = np.arange(len(metrics))
        values = list(metrics.values())
        labels = list(metrics.keys())
        
        bars = ax.bar(x, values, color=self.COLORS[:len(values)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score')
        ax.set_xlabel('Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim([0, 1.15])
        ax.set_title(f'{report.model_type} - Performance Metrics')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate comprehensive training reports."""
    
    def __init__(self):
        self.plotter = ReportPlotter()
    
    def create_report(
        self,
        job_id: str,
        user_id: int,
        model_type: str,
        training_mode: str = "central",
        dataset_name: str = "",
        class_names: Optional[List[str]] = None,
    ) -> TrainingReport:
        """Create a new training report."""
        report = TrainingReport(
            job_id=job_id,
            user_id=user_id,
            model_type=model_type,
            training_mode=training_mode,
            dataset_name=dataset_name,
            class_names=class_names or [],
            num_classes=len(class_names) if class_names else 0,
            started_at=datetime.utcnow().isoformat(),
        )
        
        # Generate share token
        report.share_token = self._generate_share_token(report.report_id)
        
        return report
    
    def _generate_share_token(self, report_id: str) -> str:
        """Generate a unique share token for the report."""
        data = f"{report_id}:{datetime.utcnow().isoformat()}:{uuid.uuid4()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def add_epoch_metrics(
        self,
        report: TrainingReport,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: float = 0.001,
    ):
        """Add metrics for a training epoch."""
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
        )
        report.epoch_metrics.append(metrics)
        
        # Update best metrics
        if val_accuracy is not None and val_accuracy > report.best_val_accuracy:
            report.best_val_accuracy = val_accuracy
            report.best_val_epoch = epoch
    
    def add_fl_round_metrics(
        self,
        report: TrainingReport,
        round_num: int,
        num_clients: int,
        avg_loss: float,
        avg_accuracy: float,
        min_accuracy: float,
        max_accuracy: float,
        std_accuracy: float,
        aggregation_time_ms: float = 0,
        communication_bytes: int = 0,
    ):
        """Add metrics for an FL round."""
        metrics = FLRoundMetrics(
            round_num=round_num,
            num_clients=num_clients,
            avg_loss=avg_loss,
            avg_accuracy=avg_accuracy,
            min_accuracy=min_accuracy,
            max_accuracy=max_accuracy,
            std_accuracy=std_accuracy,
            aggregation_time_ms=aggregation_time_ms,
            communication_bytes=communication_bytes,
        )
        report.fl_rounds.append(metrics)
    
    def set_evaluation_results(
        self,
        report: TrainingReport,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ):
        """Set evaluation results including confusion matrix, ROC, PR curves."""
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
            roc_curve,
            auc,
            precision_recall_curve,
            average_precision_score,
        )
        from sklearn.preprocessing import label_binarize
        
        class_names = report.class_names
        num_classes = len(class_names)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report.confusion_matrix = ConfusionMatrixData(
            matrix=cm.tolist(),
            class_names=class_names,
        )
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        report.class_metrics = []
        for i, name in enumerate(class_names):
            report.class_metrics.append(ClassMetrics(
                class_name=name,
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1_score=float(f1[i]),
                support=int(support[i]),
            ))
        
        # ROC and PR curves (if probabilities available)
        if y_proba is not None and num_classes > 1:
            # Binarize labels for multi-class
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            
            if num_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
            report.roc_curves = []
            report.pr_curves = []
            
            for i, name in enumerate(class_names):
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                report.roc_curves.append(ROCCurveData(
                    class_name=name,
                    fpr=fpr.tolist(),
                    tpr=tpr.tolist(),
                    auc=float(roc_auc),
                ))
                
                # PR curve
                prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
                report.pr_curves.append(PRCurveData(
                    class_name=name,
                    precision=prec.tolist(),
                    recall=rec.tolist(),
                    ap=float(ap),
                ))
    
    def finalize_report(self, report: TrainingReport) -> TrainingReport:
        """Finalize report with plots and timing."""
        report.completed_at = datetime.utcnow().isoformat()
        
        # Calculate total time
        if report.started_at:
            start = datetime.fromisoformat(report.started_at)
            end = datetime.fromisoformat(report.completed_at)
            report.total_time_seconds = (end - start).total_seconds()
        
        # Set final metrics from last epoch
        if report.epoch_metrics:
            last = report.epoch_metrics[-1]
            report.final_train_loss = last.train_loss
            report.final_train_accuracy = last.train_accuracy
            report.final_val_loss = last.val_loss
            report.final_val_accuracy = last.val_accuracy
        
        # Generate all plots
        report.plots = self.plotter.generate_all_plots(report)
        
        return report
    
    def export_to_json(self, report: TrainingReport) -> str:
        """Export report to JSON string."""
        return json.dumps(report.to_dict(), indent=2)
    
    def export_to_html(self, report: TrainingReport) -> str:
        """Export report to standalone HTML with embedded plots."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {report.job_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #0072B2; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #0072B2; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Report</h1>
        <p><strong>Job ID:</strong> {report.job_id}</p>
        <p><strong>Model:</strong> {report.model_type} | <strong>Mode:</strong> {report.training_mode}</p>
        <p><strong>Dataset:</strong> {report.dataset_name} ({report.num_classes} classes)</p>
        <p><strong>Created:</strong> {report.created_at}</p>
        
        <h2>Summary Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{report.best_val_accuracy:.2f}%</div>
                <div class="metric-label">Best Validation Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.final_train_accuracy:.2f}%</div>
                <div class="metric-label">Final Train Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.epochs}</div>
                <div class="metric-label">Epochs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.total_time_seconds:.1f}s</div>
                <div class="metric-label">Total Time</div>
            </div>
        </div>
"""
        
        # Add plots
        for plot_name, plot_data in report.plots.items():
            title = plot_name.replace('_', ' ').title()
            html += f"""
        <h2>{title}</h2>
        <div class="plot">
            <img src="data:image/png;base64,{plot_data}" alt="{title}">
        </div>
"""
        
        # Add class metrics table
        if report.class_metrics:
            html += """
        <h2>Per-Class Metrics</h2>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
"""
            for m in report.class_metrics:
                html += f"            <tr><td>{m.class_name}</td><td>{m.precision:.3f}</td><td>{m.recall:.3f}</td><td>{m.f1_score:.3f}</td><td>{m.support}</td></tr>\n"
            html += "        </table>\n"
        
        html += f"""
        <div class="footer">
            <p>Report ID: {report.report_id}</p>
            <p>Generated by Thoth ML Platform</p>
        </div>
    </div>
</body>
</html>
"""
        return html


# ============================================================================
# SHAREABLE REPORT SYSTEM
# ============================================================================

class ShareableReportManager:
    """Manage shareable read-only report links."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "/tmp/thoth_reports"
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_report(self, report: TrainingReport) -> str:
        """Save report and return share URL path."""
        # Save JSON
        json_path = os.path.join(self.storage_path, f"{report.report_id}.json")
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f)
        
        # Save HTML
        generator = ReportGenerator()
        html_content = generator.export_to_html(report)
        html_path = os.path.join(self.storage_path, f"{report.report_id}.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return f"/reports/view/{report.share_token}"
    
    def get_report_by_token(self, share_token: str) -> Optional[TrainingReport]:
        """Get report by share token."""
        # Search for report with matching token
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if data.get('share_token') == share_token:
                        return TrainingReport.from_dict(data)
                except Exception as e:
                    logger.error(f"Error loading report {filename}: {e}")
        return None
    
    def get_report_by_id(self, report_id: str) -> Optional[TrainingReport]:
        """Get report by ID."""
        filepath = os.path.join(self.storage_path, f"{report_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return TrainingReport.from_dict(data)
        return None
    
    def list_user_reports(self, user_id: int) -> List[Dict[str, Any]]:
        """List all reports for a user."""
        reports = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if data.get('user_id') == user_id:
                        reports.append({
                            "report_id": data.get('report_id'),
                            "job_id": data.get('job_id'),
                            "model_type": data.get('model_type'),
                            "training_mode": data.get('training_mode'),
                            "best_accuracy": data.get('best_val_accuracy'),
                            "created_at": data.get('created_at'),
                            "share_token": data.get('share_token'),
                        })
                except Exception as e:
                    logger.error(f"Error loading report {filename}: {e}")
        
        return sorted(reports, key=lambda x: x.get('created_at', ''), reverse=True)
