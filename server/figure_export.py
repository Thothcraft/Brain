"""Publication-Ready Figure Export System.

Generates high-quality figures for academic publications with:
- IEEE-styled formatting
- High-DPI PNG and PDF export
- Native LaTeX/TikZ export for vector graphics
- Consistent styling across all figure types

Supported figure types:
- Training curves (loss, accuracy)
- Confusion matrix
- ROC curves (per-class and micro/macro averaged)
- Precision-Recall curves
- Class distribution bar charts
- Feature importance plots
- Cross-validation box plots
"""

import io
import base64
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# IEEE column widths (in inches)
IEEE_SINGLE_COLUMN = 3.5  # Single column width
IEEE_DOUBLE_COLUMN = 7.16  # Double column width
IEEE_FONT_SIZE = 8  # Base font size for IEEE papers

# Color palette - colorblind-friendly, print-safe
IEEE_COLORS = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion/Orange
    '#009E73',  # Bluish green
    '#CC79A7',  # Reddish purple
    '#F0E442',  # Yellow
    '#56B4E9',  # Sky blue
    '#E69F00',  # Orange
    '#000000',  # Black
]

# Line styles for distinguishing curves in B&W printing
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']


def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # IEEE-compliant settings
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': IEEE_FONT_SIZE,
        'axes.labelsize': IEEE_FONT_SIZE,
        'axes.titlesize': IEEE_FONT_SIZE + 1,
        'xtick.labelsize': IEEE_FONT_SIZE - 1,
        'ytick.labelsize': IEEE_FONT_SIZE - 1,
        'legend.fontsize': IEEE_FONT_SIZE - 1,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Line settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        
        # Axes settings
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        
        # Tick settings
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Use Type 1 fonts for PDF compatibility
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def get_figure_size(width: str = 'single', aspect_ratio: float = 0.75) -> Tuple[float, float]:
    """Get figure dimensions for IEEE format.
    
    Args:
        width: 'single' for single column, 'double' for double column
        aspect_ratio: height/width ratio (default 0.75 = 4:3)
    
    Returns:
        (width, height) in inches
    """
    w = IEEE_SINGLE_COLUMN if width == 'single' else IEEE_DOUBLE_COLUMN
    h = w * aspect_ratio
    return (w, h)


def export_figure(fig, format: str = 'pdf', dpi: int = 300) -> bytes:
    """Export matplotlib figure to bytes.
    
    Args:
        fig: matplotlib figure
        format: 'pdf', 'png', 'svg', or 'eps'
        dpi: resolution for raster formats
    
    Returns:
        Figure as bytes
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    buffer.seek(0)
    return buffer.getvalue()


def figure_to_base64(fig, format: str = 'png', dpi: int = 300) -> str:
    """Convert matplotlib figure to base64 string for web display."""
    data = export_figure(fig, format=format, dpi=dpi)
    return base64.b64encode(data).decode('utf-8')


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    width: str = 'single',
    show_best_epoch: bool = True,
    best_epoch: Optional[int] = None
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready training curves.
    
    Creates a two-panel figure showing loss and accuracy over epochs.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Training accuracy per epoch
        val_accs: Validation accuracy per epoch
        width: 'single' or 'double' column
        show_best_epoch: Whether to mark the best epoch
        best_epoch: Epoch with best validation accuracy (auto-detected if None)
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    epochs = list(range(1, len(train_losses) + 1))
    
    if best_epoch is None and show_best_epoch:
        best_epoch = int(np.argmax(val_accs)) + 1
    
    fig_size = get_figure_size(width, aspect_ratio=0.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_size[0], fig_size[1]))
    
    # Loss plot
    ax1.plot(epochs, train_losses, color=IEEE_COLORS[0], linestyle='-', 
             label='Training', linewidth=1.0)
    ax1.plot(epochs, val_losses, color=IEEE_COLORS[1], linestyle='--', 
             label='Validation', linewidth=1.0)
    
    if show_best_epoch and best_epoch:
        ax1.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, len(epochs))
    
    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in train_accs], color=IEEE_COLORS[0], 
             linestyle='-', label='Training', linewidth=1.0)
    ax2.plot(epochs, [a * 100 for a in val_accs], color=IEEE_COLORS[1], 
             linestyle='--', label='Validation', linewidth=1.0)
    
    if show_best_epoch and best_epoch:
        ax2.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        best_val_acc = val_accs[best_epoch - 1] * 100
        ax2.scatter([best_epoch], [best_val_acc], color=IEEE_COLORS[1], 
                   marker='*', s=50, zorder=5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='lower right')
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    normalize: bool = True,
    width: str = 'single',
    cmap: str = 'Blues',
    show_values: bool = True,
    show_colorbar: bool = True
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready confusion matrix.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        class_names: List of class names
        normalize: Whether to show percentages (True) or counts (False)
        width: 'single' or 'double' column
        cmap: Colormap name
        show_values: Whether to show values in cells
        show_colorbar: Whether to show colorbar
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    cm = np.array(cm)
    n_classes = len(class_names)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)  # Handle division by zero
        fmt = '.1%'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()
    
    # Adjust figure size based on number of classes
    aspect = 1.0 if n_classes <= 5 else 0.8
    fig_size = get_figure_size(width, aspect_ratio=aspect)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va='bottom')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels if needed
    if max(len(name) for name in class_names) > 6 or n_classes > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Add text annotations
    if show_values:
        thresh = (vmax + vmin) / 2
        for i in range(n_classes):
            for j in range(n_classes):
                value = cm_display[i, j]
                if normalize:
                    text = f'{value:.1%}'
                else:
                    text = f'{value:d}'
                color = 'white' if value > thresh else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=IEEE_FONT_SIZE - 1)
    
    plt.tight_layout()
    return fig


# =============================================================================
# ROC CURVES
# =============================================================================

def plot_roc_curves(
    roc_data: Dict[str, Dict[str, Any]],
    width: str = 'single',
    show_auc_in_legend: bool = True,
    show_diagonal: bool = True
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready ROC curves.
    
    Args:
        roc_data: Dict mapping class_name -> {'points': [{'fpr': x, 'tpr': y}, ...], 'auc': float}
        width: 'single' or 'double' column
        show_auc_in_legend: Whether to show AUC values in legend
        show_diagonal: Whether to show random classifier diagonal
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    fig_size = get_figure_size(width, aspect_ratio=0.9)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot diagonal (random classifier)
    if show_diagonal:
        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=0.8, label='Random')
    
    # Plot each class
    for idx, (class_name, data) in enumerate(roc_data.items()):
        points = data['points']
        auc_val = data.get('auc', 0)
        
        fpr = [p['fpr'] for p in points]
        tpr = [p['tpr'] for p in points]
        
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        if show_auc_in_legend:
            label = f'{class_name} (AUC={auc_val:.3f})'
        else:
            label = class_name
        
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=1.0, label=label)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=IEEE_FONT_SIZE - 2)
    
    plt.tight_layout()
    return fig


# =============================================================================
# PRECISION-RECALL CURVES
# =============================================================================

def plot_pr_curves(
    pr_data: Dict[str, Dict[str, Any]],
    width: str = 'single',
    show_baseline: bool = True,
    class_distribution: Optional[Dict[str, float]] = None
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready Precision-Recall curves.
    
    Args:
        pr_data: Dict mapping class_name -> {'points': [{'precision': x, 'recall': y}, ...]}
        width: 'single' or 'double' column
        show_baseline: Whether to show random classifier baseline
        class_distribution: Optional dict of class proportions for baseline
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    fig_size = get_figure_size(width, aspect_ratio=0.9)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot each class
    for idx, (class_name, data) in enumerate(pr_data.items()):
        points = data['points']
        
        precision = [p['precision'] for p in points]
        recall = [p['recall'] for p in points]
        
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        
        ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=1.0, label=class_name)
        
        # Show baseline for this class if distribution provided
        if show_baseline and class_distribution and class_name in class_distribution:
            baseline = class_distribution[class_name]
            ax.axhline(y=baseline, color=color, linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower left', fontsize=IEEE_FONT_SIZE - 2)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def plot_class_distribution(
    class_counts: Dict[str, int],
    width: str = 'single',
    show_percentages: bool = True,
    horizontal: bool = False
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready class distribution bar chart.
    
    Args:
        class_counts: Dict mapping class_name -> count
        width: 'single' or 'double' column
        show_percentages: Whether to show percentage labels
        horizontal: Whether to use horizontal bars
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)
    
    n_classes = len(classes)
    aspect = 0.5 if horizontal else 0.6
    fig_size = get_figure_size(width, aspect_ratio=aspect)
    fig, ax = plt.subplots(figsize=fig_size)
    
    colors = [IEEE_COLORS[i % len(IEEE_COLORS)] for i in range(n_classes)]
    
    if horizontal:
        bars = ax.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Class')
        
        if show_percentages:
            for bar, count in zip(bars, counts):
                pct = count / total * 100
                ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', va='center', fontsize=IEEE_FONT_SIZE - 1)
    else:
        bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        
        # Rotate labels if needed
        if max(len(name) for name in classes) > 6 or n_classes > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        if show_percentages:
            for bar, count in zip(bars, counts):
                pct = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total * 0.01,
                       f'{pct:.1f}%', ha='center', fontsize=IEEE_FONT_SIZE - 1)
    
    plt.tight_layout()
    return fig


# =============================================================================
# CROSS-VALIDATION BOX PLOTS
# =============================================================================

def plot_cv_boxplot(
    cv_results: Dict[str, List[float]],
    metric_name: str = 'Accuracy',
    width: str = 'single',
    show_points: bool = True,
    show_mean: bool = True
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready cross-validation box plot.
    
    Args:
        cv_results: Dict mapping model_name -> list of fold scores
        metric_name: Name of the metric (for y-axis label)
        width: 'single' or 'double' column
        show_points: Whether to show individual fold points
        show_mean: Whether to show mean marker
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    models = list(cv_results.keys())
    data = [cv_results[m] for m in models]
    n_models = len(models)
    
    fig_size = get_figure_size(width, aspect_ratio=0.6)
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create box plot
    bp = ax.boxplot(data, labels=models, patch_artist=True, widths=0.6)
    
    # Color boxes
    for idx, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        box.set_facecolor(color)
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Show individual points
    if show_points:
        for idx, (model_data, x_pos) in enumerate(zip(data, range(1, n_models + 1))):
            jitter = np.random.uniform(-0.1, 0.1, len(model_data))
            ax.scatter([x_pos + j for j in jitter], model_data, 
                      color='black', s=15, alpha=0.6, zorder=3)
    
    # Show mean
    if show_mean:
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, n_models + 1), means, marker='D', 
                  color='red', s=30, zorder=4, label='Mean')
    
    ax.set_ylabel(f'{metric_name} (%)')
    ax.set_ylim(0, 100)
    
    # Rotate labels if needed
    if max(len(name) for name in models) > 8 or n_models > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    if show_mean:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    width: str = 'double',
    show_error_bars: bool = True
) -> 'matplotlib.figure.Figure':
    """Generate publication-ready model comparison bar chart.
    
    Args:
        results: Dict mapping model_name -> {metric_name: value, metric_name_std: std_value}
        metrics: List of metrics to compare
        width: 'single' or 'double' column
        show_error_bars: Whether to show standard deviation error bars
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    setup_ieee_style()
    
    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(metrics)
    
    fig_size = get_figure_size(width, aspect_ratio=0.5)
    fig, ax = plt.subplots(figsize=fig_size)
    
    x = np.arange(n_models)
    bar_width = 0.8 / n_metrics
    
    for idx, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) * 100 for m in models]
        stds = [results[m].get(f'{metric}_std', 0) * 100 for m in models] if show_error_bars else None
        
        offset = (idx - n_metrics/2 + 0.5) * bar_width
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        
        bars = ax.bar(x + offset, values, bar_width, label=metric.capitalize(),
                     color=color, edgecolor='black', linewidth=0.5,
                     yerr=stds if show_error_bars else None, capsize=2)
    
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', ncol=min(4, n_metrics))
    
    # Rotate labels if needed
    if max(len(name) for name in models) > 8 or n_models > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    return fig


# =============================================================================
# LATEX/TIKZ EXPORT
# =============================================================================

def generate_latex_table(
    data: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    caption: str = 'Model Performance Comparison',
    label: str = 'tab:results',
    highlight_best: bool = True,
    show_std: bool = True
) -> str:
    """Generate LaTeX table for model comparison.
    
    Args:
        data: Dict mapping model_name -> {metric: value, metric_std: std}
        metrics: List of metrics to include
        caption: Table caption
        label: LaTeX label for referencing
        highlight_best: Whether to bold the best value per metric
        show_std: Whether to show standard deviation
    
    Returns:
        LaTeX table string
    """
    models = list(data.keys())
    n_metrics = len(metrics)
    
    # Find best values for highlighting
    best_values = {}
    if highlight_best:
        for metric in metrics:
            values = [data[m].get(metric, 0) for m in models]
            best_values[metric] = max(values)
    
    # Build table
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{l' + 'c' * n_metrics + '}',
        r'\toprule',
    ]
    
    # Header
    header = 'Model & ' + ' & '.join([m.capitalize() for m in metrics]) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    # Data rows
    for model in models:
        row_parts = [model]
        for metric in metrics:
            value = data[model].get(metric, 0) * 100
            std = data[model].get(f'{metric}_std', 0) * 100
            
            if show_std and std > 0:
                cell = f'{value:.1f} $\\pm$ {std:.1f}'
            else:
                cell = f'{value:.1f}'
            
            # Highlight best
            if highlight_best and abs(data[model].get(metric, 0) - best_values.get(metric, 0)) < 0.001:
                cell = r'\textbf{' + cell + '}'
            
            row_parts.append(cell)
        
        lines.append(' & '.join(row_parts) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    return '\n'.join(lines)


def generate_confusion_matrix_latex(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    caption: str = 'Confusion Matrix',
    label: str = 'tab:confusion'
) -> str:
    """Generate LaTeX table for confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        caption: Table caption
        label: LaTeX label
    
    Returns:
        LaTeX table string
    """
    cm = np.array(cm)
    n_classes = len(class_names)
    
    # Normalize for display
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{l|' + 'c' * n_classes + '}',
        r'\toprule',
    ]
    
    # Header
    header = r' & ' + ' & '.join(class_names) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    
    # Data rows
    for i, class_name in enumerate(class_names):
        row_parts = [class_name]
        for j in range(n_classes):
            pct = cm_norm[i, j] * 100
            if i == j:  # Diagonal - highlight
                cell = r'\textbf{' + f'{pct:.1f}' + r'\%}'
            else:
                cell = f'{pct:.1f}\\%'
            row_parts.append(cell)
        lines.append(' & '.join(row_parts) + r' \\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    return '\n'.join(lines)


# =============================================================================
# EXPORT ALL FIGURES
# =============================================================================

def export_all_training_figures(
    training_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    formats: List[str] = ['pdf', 'png'],
    width: str = 'single'
) -> Dict[str, Dict[str, Union[bytes, str]]]:
    """Export all training figures in multiple formats.
    
    Args:
        training_results: Training results dictionary from training pipeline
        output_dir: Optional directory to save files (if None, returns bytes)
        formats: List of formats to export ('pdf', 'png', 'svg', 'eps')
        width: 'single' or 'double' column
    
    Returns:
        Dict mapping figure_name -> {format: bytes_or_path}
    """
    import matplotlib.pyplot as plt
    
    results = {}
    
    # Training curves
    if all(k in training_results for k in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']):
        fig = plot_training_curves(
            training_results['train_losses'],
            training_results['val_losses'],
            training_results['train_accuracies'],
            training_results['val_accuracies'],
            width=width,
            best_epoch=training_results.get('best_epoch')
        )
        results['training_curves'] = _export_fig_formats(fig, formats, output_dir, 'training_curves')
        plt.close(fig)
    
    # Confusion matrix
    if 'confusion_matrix' in training_results and 'class_names' in training_results:
        fig = plot_confusion_matrix(
            training_results['confusion_matrix'],
            training_results['class_names'],
            width=width
        )
        results['confusion_matrix'] = _export_fig_formats(fig, formats, output_dir, 'confusion_matrix')
        plt.close(fig)
        
        # Also generate LaTeX
        latex = generate_confusion_matrix_latex(
            training_results['confusion_matrix'],
            training_results['class_names']
        )
        results['confusion_matrix']['latex'] = latex
    
    # ROC curves
    if 'roc_curves' in training_results:
        fig = plot_roc_curves(training_results['roc_curves'], width=width)
        results['roc_curves'] = _export_fig_formats(fig, formats, output_dir, 'roc_curves')
        plt.close(fig)
    
    # PR curves
    if 'pr_curves' in training_results:
        fig = plot_pr_curves(training_results['pr_curves'], width=width)
        results['pr_curves'] = _export_fig_formats(fig, formats, output_dir, 'pr_curves')
        plt.close(fig)
    
    return results


def _export_fig_formats(fig, formats: List[str], output_dir: Optional[str], name: str) -> Dict[str, Union[bytes, str]]:
    """Helper to export figure in multiple formats."""
    from pathlib import Path
    
    result = {}
    for fmt in formats:
        data = export_figure(fig, format=fmt, dpi=300)
        
        if output_dir:
            path = Path(output_dir) / f'{name}.{fmt}'
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
            result[fmt] = str(path)
        else:
            result[fmt] = data
    
    return result
