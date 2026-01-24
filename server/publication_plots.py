"""Publication-Quality Plotting System for ML/DL Training.

This module provides professional plotting capabilities that:
1. Dynamically adapt to model type (ML vs DL)
2. Support multiple trials with statistical analysis
3. Export in publication-quality formats (PDF, EPS, SVG, PNG)
4. Follow best practices for top-venue publications (NeurIPS, ICML, CVPR, etc.)

Inspired by:
- Matplotlib best practices: https://matplotlib.org/stable/tutorials/introductory/customizing.html
- Seaborn statistical visualization: https://seaborn.pydata.org/
- Publication guidelines from top ML venues
"""

import os
import io
import base64
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# MODEL TYPE DETECTION
# ============================================================================

class ModelCategory:
    """Model categories for appropriate plot selection."""
    DL_ITERATIVE = "dl_iterative"  # DL models with epochs (CNN, LSTM, etc.)
    ML_SINGLE_FIT = "ml_single_fit"  # ML models without epochs (SVM, RF, etc.)
    FL_ROUNDS = "fl_rounds"  # Federated learning with rounds
    CLUSTERING = "clustering"  # Unsupervised clustering (K-Means, DBSCAN)


# ML models that don't have epoch-based training
ML_SINGLE_FIT_MODELS = {
    'svm', 'svc', 'random_forest', 'rf', 'knn', 'k_nearest_neighbors',
    'logistic_regression', 'lr', 'decision_tree', 'dt', 'gradient_boosting',
    'gb', 'adaboost', 'naive_bayes', 'nb', 'linear_svm', 'rbf_svm',
    'xgboost', 'lightgbm', 'catboost'
}

# Clustering models (unsupervised)
CLUSTERING_MODELS = {
    'kmeans', 'k_means', 'dbscan', 'hierarchical', 'agglomerative',
    'spectral_clustering', 'mean_shift', 'optics', 'birch'
}

# DL models with iterative training
DL_ITERATIVE_MODELS = {
    'cnn', 'lstm', 'gru', 'transformer', 'mlp', 'resnet', 'vgg',
    'cnn1d', 'cnn2d', 'cnn3d', 'bilstm', 'bigru', 'attention'
}


def detect_model_category(model_type: str) -> str:
    """Detect the category of a model for appropriate plotting."""
    model_lower = model_type.lower().replace('-', '_').replace(' ', '_')
    
    # Check for FL
    if 'fed' in model_lower or 'fl_' in model_lower:
        return ModelCategory.FL_ROUNDS
    
    # Check for clustering
    for cluster_model in CLUSTERING_MODELS:
        if cluster_model in model_lower:
            return ModelCategory.CLUSTERING
    
    # Check for ML single-fit
    for ml_model in ML_SINGLE_FIT_MODELS:
        if ml_model in model_lower:
            return ModelCategory.ML_SINGLE_FIT
    
    # Default to DL iterative
    return ModelCategory.DL_ITERATIVE


# ============================================================================
# MULTI-TRIAL DATA STRUCTURES
# ============================================================================

@dataclass
class TrialResult:
    """Results from a single training trial."""
    trial_id: int
    model_type: str
    
    # Final metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Optional: epoch-wise metrics for DL
    epoch_accuracies: List[float] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    # Training time
    training_time_seconds: float = 0.0
    
    # Hyperparameters used
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Per-class metrics
    class_accuracies: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    # ROC/PR data
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


@dataclass
class MultiTrialResults:
    """Aggregated results from multiple training trials."""
    model_type: str
    num_trials: int
    class_names: List[str]
    
    # Individual trial results
    trials: List[TrialResult] = field(default_factory=list)
    
    # Aggregated statistics
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    
    mean_f1: float = 0.0
    std_f1: float = 0.0
    
    mean_training_time: float = 0.0
    std_training_time: float = 0.0
    
    # Confidence intervals (95%)
    accuracy_ci_lower: float = 0.0
    accuracy_ci_upper: float = 0.0
    
    def compute_statistics(self):
        """Compute aggregate statistics from trials."""
        if not self.trials:
            return
        
        accuracies = [t.accuracy for t in self.trials]
        f1_scores = [t.f1_score for t in self.trials]
        times = [t.training_time_seconds for t in self.trials]
        
        self.num_trials = len(self.trials)
        
        # Basic statistics
        self.mean_accuracy = np.mean(accuracies)
        self.std_accuracy = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0
        self.min_accuracy = np.min(accuracies)
        self.max_accuracy = np.max(accuracies)
        
        self.mean_f1 = np.mean(f1_scores)
        self.std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0
        
        self.mean_training_time = np.mean(times)
        self.std_training_time = np.std(times, ddof=1) if len(times) > 1 else 0
        
        # 95% Confidence interval
        if SCIPY_AVAILABLE and len(accuracies) > 1:
            ci = stats.t.interval(
                0.95,
                len(accuracies) - 1,
                loc=self.mean_accuracy,
                scale=stats.sem(accuracies)
            )
            self.accuracy_ci_lower = ci[0]
            self.accuracy_ci_upper = ci[1]
        else:
            # Fallback: use mean ± 1.96 * std / sqrt(n)
            se = self.std_accuracy / np.sqrt(len(accuracies)) if len(accuracies) > 0 else 0
            self.accuracy_ci_lower = self.mean_accuracy - 1.96 * se
            self.accuracy_ci_upper = self.mean_accuracy + 1.96 * se


# ============================================================================
# PUBLICATION-QUALITY PLOTTER
# ============================================================================

class PublicationPlotter:
    """Generate publication-quality plots for ML/DL training results.
    
    Supports:
    - Dynamic adaptation to model type (ML vs DL)
    - Multiple trials with statistical analysis
    - Export in multiple formats (PDF, EPS, SVG, PNG)
    - Customizable styles for different venues
    """
    
    # Publication-quality settings (NeurIPS/ICML style)
    STYLES = {
        'neurips': {
            'figure.figsize': (5.5, 4.0),
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
        },
        'icml': {
            'figure.figsize': (6.0, 4.0),
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
        },
        'cvpr': {
            'figure.figsize': (6.5, 4.5),
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
        },
        'default': {
            'figure.figsize': (8, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
        }
    }
    
    # Colorblind-friendly palette (Wong, 2011)
    COLORS = [
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#009E73',  # Bluish green
        '#CC79A7',  # Reddish purple
        '#F0E442',  # Yellow
        '#56B4E9',  # Sky blue
        '#E69F00',  # Orange
        '#000000',  # Black
    ]
    
    # Marker styles for distinguishing lines
    MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    
    # Line styles
    LINE_STYLES = ['-', '--', '-.', ':']
    
    def __init__(self, style: str = 'default', use_latex: bool = False):
        """Initialize plotter with style settings.
        
        Args:
            style: Publication style ('neurips', 'icml', 'cvpr', 'default')
            use_latex: Whether to use LaTeX for text rendering
        """
        self.style = style
        self.use_latex = use_latex
        
        if MATPLOTLIB_AVAILABLE:
            self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for publication quality."""
        style_params = self.STYLES.get(self.style, self.STYLES['default'])
        
        plt.rcParams.update({
            **style_params,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        if self.use_latex:
            plt.rcParams.update({
                'text.usetex': True,
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman'],
            })
        
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette(self.COLORS)
    
    def _get_color(self, idx: int) -> str:
        """Get color from palette."""
        return self.COLORS[idx % len(self.COLORS)]
    
    def _get_marker(self, idx: int) -> str:
        """Get marker style."""
        return self.MARKERS[idx % len(self.MARKERS)]
    
    def _save_figure(
        self,
        fig,
        filename: str,
        formats: List[str] = None,
        output_dir: str = None
    ) -> Dict[str, str]:
        """Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            formats: List of formats ('png', 'pdf', 'eps', 'svg')
            output_dir: Output directory
        
        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = ['png', 'pdf']
        
        if output_dir is None:
            output_dir = '/tmp/thoth_plots'
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            saved_files[fmt] = filepath
        
        return saved_files
    
    def _fig_to_base64(self, fig) -> str:
        """Convert figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    # ========================================================================
    # ML-SPECIFIC PLOTS (No epochs)
    # ========================================================================
    
    def plot_ml_metrics_comparison(
        self,
        results: MultiTrialResults,
        title: str = None,
        show_individual: bool = True
    ) -> Tuple[plt.Figure, str]:
        """Plot metrics comparison for ML models (no epoch curves).
        
        Creates a bar chart with error bars showing mean ± std across trials.
        
        Args:
            results: Multi-trial results
            title: Plot title
            show_individual: Show individual trial points
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Metrics to plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Compute means and stds
        means = []
        stds = []
        individual_values = {m: [] for m in metrics}
        
        for metric in metrics:
            if metric == 'Accuracy':
                values = [t.accuracy for t in results.trials]
            elif metric == 'Precision':
                values = [t.precision for t in results.trials]
            elif metric == 'Recall':
                values = [t.recall for t in results.trials]
            else:  # F1-Score
                values = [t.f1_score for t in results.trials]
            
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1) if len(values) > 1 else 0)
            individual_values[metric] = values
        
        x = np.arange(len(metrics))
        width = 0.6
        
        # Bar chart with error bars
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                     color=self.COLORS[0], alpha=0.7, edgecolor='black',
                     error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        # Show individual trial points
        if show_individual and len(results.trials) > 1:
            for i, metric in enumerate(metrics):
                values = individual_values[metric]
                # Add jitter for visibility
                jitter = np.random.uniform(-0.15, 0.15, len(values))
                ax.scatter(x[i] + jitter, values, color=self.COLORS[1],
                          s=30, alpha=0.6, zorder=3, edgecolor='white', linewidth=0.5)
        
        # Labels and formatting
        ax.set_ylabel('Score')
        ax.set_xlabel('Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}±{std:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{results.model_type} Performance ({results.num_trials} trials)')
        
        # Add legend for individual points
        if show_individual and len(results.trials) > 1:
            legend_elements = [
                Patch(facecolor=self.COLORS[0], alpha=0.7, edgecolor='black', label='Mean ± Std'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS[1],
                          markersize=8, label='Individual Trials')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    def plot_ml_training_time(
        self,
        results: MultiTrialResults,
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot training time distribution for ML models.
        
        Args:
            results: Multi-trial results
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        times = [t.training_time_seconds for t in results.trials]
        
        if len(times) > 1:
            # Histogram with KDE
            if SEABORN_AVAILABLE:
                sns.histplot(times, kde=True, ax=ax, color=self.COLORS[0], alpha=0.7)
            else:
                ax.hist(times, bins='auto', color=self.COLORS[0], alpha=0.7, edgecolor='black')
            
            # Add mean line
            mean_time = np.mean(times)
            ax.axvline(mean_time, color=self.COLORS[1], linestyle='--', linewidth=2,
                      label=f'Mean: {mean_time:.2f}s')
            ax.legend()
        else:
            ax.bar(['Training Time'], times, color=self.COLORS[0], alpha=0.7)
            ax.annotate(f'{times[0]:.2f}s', xy=(0, times[0]), xytext=(0, 5),
                       textcoords="offset points", ha='center', fontsize=10)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Frequency' if len(times) > 1 else 'Time (s)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{results.model_type} Training Time Distribution')
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    # ========================================================================
    # DL-SPECIFIC PLOTS (With epochs)
    # ========================================================================
    
    def plot_dl_learning_curves(
        self,
        results: MultiTrialResults,
        metric: str = 'accuracy',
        show_ci: bool = True,
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot learning curves for DL models with confidence intervals.
        
        Args:
            results: Multi-trial results
            metric: 'accuracy' or 'loss'
            show_ci: Show 95% confidence interval
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Collect epoch-wise data from all trials
        if metric == 'accuracy':
            train_data = [t.epoch_accuracies for t in results.trials if t.epoch_accuracies]
            val_data = [t.val_accuracies for t in results.trials if t.val_accuracies]
            ylabel = 'Accuracy'
        else:
            train_data = [t.epoch_losses for t in results.trials if t.epoch_losses]
            val_data = [t.val_losses for t in results.trials if t.val_losses]
            ylabel = 'Loss'
        
        if not train_data:
            plt.close(fig)
            return None, ""
        
        # Find minimum length (in case trials have different epochs)
        min_epochs = min(len(d) for d in train_data)
        train_data = [d[:min_epochs] for d in train_data]
        
        epochs = np.arange(1, min_epochs + 1)
        train_array = np.array(train_data)
        
        # Compute mean and std
        train_mean = np.mean(train_array, axis=0)
        train_std = np.std(train_array, axis=0, ddof=1) if len(train_data) > 1 else np.zeros_like(train_mean)
        
        # Plot training curve with confidence band
        ax.plot(epochs, train_mean, '-', color=self.COLORS[0], linewidth=2,
               label=f'Train ({results.num_trials} trials)')
        
        if show_ci and len(train_data) > 1:
            ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                           color=self.COLORS[0], alpha=0.2)
        
        # Plot validation if available
        if val_data:
            min_val_epochs = min(len(d) for d in val_data)
            val_data = [d[:min_val_epochs] for d in val_data]
            val_array = np.array(val_data)
            val_mean = np.mean(val_array, axis=0)
            val_std = np.std(val_array, axis=0, ddof=1) if len(val_data) > 1 else np.zeros_like(val_mean)
            
            val_epochs = np.arange(1, min_val_epochs + 1)
            ax.plot(val_epochs, val_mean, '--', color=self.COLORS[1], linewidth=2,
                   label='Validation')
            
            if show_ci and len(val_data) > 1:
                ax.fill_between(val_epochs, val_mean - val_std, val_mean + val_std,
                               color=self.COLORS[1], alpha=0.2)
            
            # Mark best validation point
            if metric == 'accuracy':
                best_idx = np.argmax(val_mean)
            else:
                best_idx = np.argmin(val_mean)
            
            ax.scatter([val_epochs[best_idx]], [val_mean[best_idx]], s=100,
                      color=self.COLORS[2], marker='*', zorder=5,
                      label=f'Best (Epoch {val_epochs[best_idx]})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{results.model_type} Learning Curves')
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    def plot_dl_loss_accuracy_combined(
        self,
        results: MultiTrialResults,
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot combined loss and accuracy curves for DL models.
        
        Args:
            results: Multi-trial results
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1 = axes[0]
        loss_data = [t.epoch_losses for t in results.trials if t.epoch_losses]
        val_loss_data = [t.val_losses for t in results.trials if t.val_losses]
        
        if loss_data:
            min_epochs = min(len(d) for d in loss_data)
            loss_data = [d[:min_epochs] for d in loss_data]
            epochs = np.arange(1, min_epochs + 1)
            loss_array = np.array(loss_data)
            loss_mean = np.mean(loss_array, axis=0)
            loss_std = np.std(loss_array, axis=0, ddof=1) if len(loss_data) > 1 else np.zeros_like(loss_mean)
            
            ax1.plot(epochs, loss_mean, '-', color=self.COLORS[0], linewidth=2, label='Train')
            if len(loss_data) > 1:
                ax1.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                               color=self.COLORS[0], alpha=0.2)
            
            if val_loss_data:
                min_val_epochs = min(len(d) for d in val_loss_data)
                val_loss_data = [d[:min_val_epochs] for d in val_loss_data]
                val_loss_array = np.array(val_loss_data)
                val_loss_mean = np.mean(val_loss_array, axis=0)
                val_loss_std = np.std(val_loss_array, axis=0, ddof=1) if len(val_loss_data) > 1 else np.zeros_like(val_loss_mean)
                
                val_epochs = np.arange(1, min_val_epochs + 1)
                ax1.plot(val_epochs, val_loss_mean, '--', color=self.COLORS[1], linewidth=2, label='Validation')
                if len(val_loss_data) > 1:
                    ax1.fill_between(val_epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std,
                                   color=self.COLORS[1], alpha=0.2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Accuracy plot
        ax2 = axes[1]
        acc_data = [t.epoch_accuracies for t in results.trials if t.epoch_accuracies]
        val_acc_data = [t.val_accuracies for t in results.trials if t.val_accuracies]
        
        if acc_data:
            min_epochs = min(len(d) for d in acc_data)
            acc_data = [d[:min_epochs] for d in acc_data]
            epochs = np.arange(1, min_epochs + 1)
            acc_array = np.array(acc_data)
            acc_mean = np.mean(acc_array, axis=0)
            acc_std = np.std(acc_array, axis=0, ddof=1) if len(acc_data) > 1 else np.zeros_like(acc_mean)
            
            ax2.plot(epochs, acc_mean, '-', color=self.COLORS[0], linewidth=2, label='Train')
            if len(acc_data) > 1:
                ax2.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                               color=self.COLORS[0], alpha=0.2)
            
            if val_acc_data:
                min_val_epochs = min(len(d) for d in val_acc_data)
                val_acc_data = [d[:min_val_epochs] for d in val_acc_data]
                val_acc_array = np.array(val_acc_data)
                val_acc_mean = np.mean(val_acc_array, axis=0)
                val_acc_std = np.std(val_acc_array, axis=0, ddof=1) if len(val_acc_data) > 1 else np.zeros_like(val_acc_mean)
                
                val_epochs = np.arange(1, min_val_epochs + 1)
                ax2.plot(val_epochs, val_acc_mean, '--', color=self.COLORS[1], linewidth=2, label='Validation')
                if len(val_acc_data) > 1:
                    ax2.fill_between(val_epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                                   color=self.COLORS[1], alpha=0.2)
                
                # Mark best
                best_idx = np.argmax(val_acc_mean)
                ax2.scatter([val_epochs[best_idx]], [val_acc_mean[best_idx]], s=100,
                          color=self.COLORS[2], marker='*', zorder=5,
                          label=f'Best: {val_acc_mean[best_idx]:.3f}')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    # ========================================================================
    # COMMON PLOTS (Both ML and DL)
    # ========================================================================
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Normalize by row (true labels)
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        cm = confusion_matrix.copy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
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
                    ax.text(j, i, text, ha='center', va='center',
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')
            
            plt.colorbar(im, ax=ax, label='Proportion' if normalize else 'Count')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    def plot_roc_curves(
        self,
        roc_data: List[Dict[str, Any]],
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot ROC curves for all classes.
        
        Args:
            roc_data: List of dicts with 'class_name', 'fpr', 'tpr', 'auc'
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        for i, roc in enumerate(roc_data):
            color = self._get_color(i)
            ax.plot(roc['fpr'], roc['tpr'], color=color, linewidth=2,
                   label=f"{roc['class_name']} (AUC = {roc['auc']:.3f})")
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=9)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('ROC Curves')
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    def plot_trial_comparison(
        self,
        results: MultiTrialResults,
        metric: str = 'accuracy',
        title: str = None
    ) -> Tuple[plt.Figure, str]:
        """Plot comparison of individual trials.
        
        Args:
            results: Multi-trial results
            metric: Metric to compare
            title: Plot title
        
        Returns:
            Tuple of (figure, base64_string)
        """
        if not MATPLOTLIB_AVAILABLE:
            return None, ""
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if metric == 'accuracy':
            values = [t.accuracy for t in results.trials]
            ylabel = 'Accuracy'
        elif metric == 'f1':
            values = [t.f1_score for t in results.trials]
            ylabel = 'F1-Score'
        else:
            values = [t.accuracy for t in results.trials]
            ylabel = 'Accuracy'
        
        trial_ids = [f'Trial {t.trial_id}' for t in results.trials]
        x = np.arange(len(trial_ids))
        
        # Bar chart
        bars = ax.bar(x, values, color=self.COLORS[0], alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color=self.COLORS[1], linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        
        # Add std band
        if len(values) > 1:
            std_val = np.std(values, ddof=1)
            ax.axhspan(mean_val - std_val, mean_val + std_val, alpha=0.2,
                      color=self.COLORS[1], label=f'±1 Std: {std_val:.4f}')
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Trial')
        ax.set_xticks(x)
        ax.set_xticklabels(trial_ids, rotation=45, ha='right')
        ax.legend(loc='lower right')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{results.model_type} - Trial Comparison')
        
        plt.tight_layout()
        return fig, self._fig_to_base64(fig)
    
    # ========================================================================
    # ADAPTIVE PLOT GENERATION
    # ========================================================================
    
    def generate_appropriate_plots(
        self,
        results: MultiTrialResults,
        output_dir: str = None,
        formats: List[str] = None
    ) -> Dict[str, Any]:
        """Generate appropriate plots based on model type.
        
        Automatically detects model category and generates relevant plots.
        
        Args:
            results: Multi-trial results
            output_dir: Directory to save plots
            formats: Export formats
        
        Returns:
            Dictionary with plot names and base64 strings
        """
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "matplotlib not available"}
        
        plots = {}
        saved_files = {}
        
        # Detect model category
        category = detect_model_category(results.model_type)
        
        # Compute statistics
        results.compute_statistics()
        
        # Generate plots based on category
        if category == ModelCategory.ML_SINGLE_FIT:
            # ML models: No epoch curves, show metrics comparison
            fig, b64 = self.plot_ml_metrics_comparison(results)
            if fig:
                plots['metrics_comparison'] = b64
                if output_dir:
                    saved_files['metrics_comparison'] = self._save_figure(
                        fig, 'metrics_comparison', formats, output_dir
                    )
                plt.close(fig)
            
            fig, b64 = self.plot_ml_training_time(results)
            if fig:
                plots['training_time'] = b64
                if output_dir:
                    saved_files['training_time'] = self._save_figure(
                        fig, 'training_time', formats, output_dir
                    )
                plt.close(fig)
        
        elif category == ModelCategory.DL_ITERATIVE:
            # DL models: Show learning curves with epochs
            fig, b64 = self.plot_dl_loss_accuracy_combined(results)
            if fig:
                plots['learning_curves'] = b64
                if output_dir:
                    saved_files['learning_curves'] = self._save_figure(
                        fig, 'learning_curves', formats, output_dir
                    )
                plt.close(fig)
        
        # Common plots for all categories
        if results.num_trials > 1:
            fig, b64 = self.plot_trial_comparison(results)
            if fig:
                plots['trial_comparison'] = b64
                if output_dir:
                    saved_files['trial_comparison'] = self._save_figure(
                        fig, 'trial_comparison', formats, output_dir
                    )
                plt.close(fig)
        
        # Confusion matrix (if available from first trial)
        if results.trials and results.trials[0].confusion_matrix is not None:
            fig, b64 = self.plot_confusion_matrix(
                results.trials[0].confusion_matrix,
                results.class_names
            )
            if fig:
                plots['confusion_matrix'] = b64
                if output_dir:
                    saved_files['confusion_matrix'] = self._save_figure(
                        fig, 'confusion_matrix', formats, output_dir
                    )
                plt.close(fig)
        
        return {
            'plots': plots,
            'saved_files': saved_files,
            'model_category': category,
            'statistics': {
                'mean_accuracy': results.mean_accuracy,
                'std_accuracy': results.std_accuracy,
                'ci_95': (results.accuracy_ci_lower, results.accuracy_ci_upper),
                'num_trials': results.num_trials,
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_publication_report(
    trials: List[TrialResult],
    model_type: str,
    class_names: List[str],
    output_dir: str = None,
    style: str = 'neurips',
    formats: List[str] = None
) -> Dict[str, Any]:
    """Create a complete publication-ready report from trial results.
    
    Args:
        trials: List of trial results
        model_type: Model type string
        class_names: List of class names
        output_dir: Directory to save plots
        style: Publication style ('neurips', 'icml', 'cvpr', 'default')
        formats: Export formats (default: ['png', 'pdf'])
    
    Returns:
        Dictionary with plots, statistics, and file paths
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    # Create multi-trial results
    results = MultiTrialResults(
        model_type=model_type,
        num_trials=len(trials),
        class_names=class_names,
        trials=trials
    )
    
    # Create plotter and generate plots
    plotter = PublicationPlotter(style=style)
    return plotter.generate_appropriate_plots(results, output_dir, formats)


def export_for_latex(
    output_dir: str,
    figure_names: List[str] = None
) -> str:
    """Generate LaTeX code to include exported figures.
    
    Args:
        output_dir: Directory containing exported figures
        figure_names: List of figure names (without extension)
    
    Returns:
        LaTeX code string
    """
    if figure_names is None:
        # Find all PDF files in output_dir
        figure_names = [
            f[:-4] for f in os.listdir(output_dir)
            if f.endswith('.pdf')
        ]
    
    latex = "% Auto-generated LaTeX code for figures\n"
    latex += "% Add to your document preamble: \\usepackage{graphicx}\n\n"
    
    for name in figure_names:
        latex += f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{name}}}
    \\caption{{Caption for {name.replace('_', ' ').title()}}}
    \\label{{fig:{name}}}
\\end{{figure}}

"""
    
    return latex
