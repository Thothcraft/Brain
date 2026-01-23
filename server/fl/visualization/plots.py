"""Plotting Utilities for FL Experiments.

This module provides matplotlib-based visualizations for FL experiments,
inspired by Flower's plotting utilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, plotting functions will return None")


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed. Install with: pip install matplotlib")
        return False
    return True


def plot_accuracy_curves(
    results: List[Dict[str, Any]],
    title: str = "FL Training Accuracy",
    xlabel: str = "Round",
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_std: bool = True,
) -> Optional["Figure"]:
    """Plot accuracy curves for multiple experiments.
    
    Args:
        results: List of experiment results, each with 'name' and 'curves' keys
                 curves should be list of accuracy values per round
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        figsize: Figure size
        show_std: Whether to show standard deviation bands
    
    Returns:
        matplotlib Figure or None if matplotlib not available
    """
    if not _check_matplotlib():
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, result in enumerate(results):
        name = result.get("name", f"Experiment {idx + 1}")
        curves = result.get("curves", [])
        
        if not curves:
            continue
        
        # If multiple runs, compute mean and std
        if isinstance(curves[0], list):
            # Multiple runs
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            curves_array = np.array(curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            
            rounds = np.arange(1, len(mean_curve) + 1)
            ax.plot(rounds, mean_curve, label=name, color=colors[idx], linewidth=2)
            
            if show_std:
                ax.fill_between(
                    rounds,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.2,
                    color=colors[idx]
                )
        else:
            # Single run
            rounds = np.arange(1, len(curves) + 1)
            ax.plot(rounds, curves, label=name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved accuracy curves to {save_path}")
    
    return fig


def plot_sample_distribution(
    distributions: Dict[int, Dict[int, int]],
    num_classes: int = 10,
    title: str = "Sample Distribution Across Clients",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    class_names: Optional[List[str]] = None,
) -> Optional["Figure"]:
    """Plot sample distribution across clients (heatmap style).
    
    Args:
        distributions: Dict mapping client_id -> {label -> count}
        num_classes: Number of classes
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        class_names: Optional list of class names
    
    Returns:
        matplotlib Figure or None
    """
    if not _check_matplotlib():
        return None
    
    num_clients = len(distributions)
    
    # Build distribution matrix
    matrix = np.zeros((num_clients, num_classes))
    for client_id, label_counts in distributions.items():
        for label, count in label_counts.items():
            if label < num_classes:
                matrix[client_id, label] = count
    
    # Normalize by row (per client)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / row_sums
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Heatmap of raw counts
    im1 = ax1.imshow(matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Client', fontsize=12)
    ax1.set_title('Sample Counts', fontsize=12)
    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_clients))
    if class_names:
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # Heatmap of normalized distribution
    im2 = ax2.imshow(matrix_normalized, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Client', fontsize=12)
    ax2.set_title('Normalized Distribution', fontsize=12)
    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_clients))
    if class_names:
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
    plt.colorbar(im2, ax=ax2, label='Proportion')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sample distribution to {save_path}")
    
    return fig


def plot_comparative_results(
    results: List[Dict[str, Any]],
    metric: str = "accuracy",
    title: str = "Comparative Results",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    sort_by_value: bool = True,
) -> Optional["Figure"]:
    """Plot comparative bar chart with error bars.
    
    Args:
        results: List of dicts with 'name', 'mean', 'std' keys
        metric: Metric name for labeling
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        sort_by_value: Whether to sort bars by value
    
    Returns:
        matplotlib Figure or None
    """
    if not _check_matplotlib():
        return None
    
    if sort_by_value:
        results = sorted(results, key=lambda x: x.get('mean', 0), reverse=True)
    
    names = [r.get('name', f'Exp {i}') for i, r in enumerate(results)]
    means = [r.get('mean', 0) for r in results]
    stds = [r.get('std', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results)))
    
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.3f}±{std:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.2 if means else 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparative results to {save_path}")
    
    return fig


def plot_shadow_curves(
    runs: List[List[float]],
    name: str = "Experiment",
    title: str = "Training Curves with Shadow",
    xlabel: str = "Round",
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "blue",
    show_individual: bool = True,
) -> Optional["Figure"]:
    """Plot shadow curves showing all runs with mean and std band.
    
    Args:
        runs: List of accuracy curves (one per run)
        name: Experiment name for legend
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure
        figsize: Figure size
        color: Color for the plot
        show_individual: Whether to show individual run curves
    
    Returns:
        matplotlib Figure or None
    """
    if not _check_matplotlib():
        return None
    
    if not runs:
        return None
    
    # Align curves to minimum length
    min_len = min(len(r) for r in runs)
    runs = [r[:min_len] for r in runs]
    runs_array = np.array(runs)
    
    mean_curve = np.mean(runs_array, axis=0)
    std_curve = np.std(runs_array, axis=0)
    min_curve = np.min(runs_array, axis=0)
    max_curve = np.max(runs_array, axis=0)
    
    rounds = np.arange(1, min_len + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual runs (faded)
    if show_individual:
        for i, run in enumerate(runs):
            ax.plot(rounds, run, alpha=0.2, color=color, linewidth=0.5)
    
    # Plot min-max range
    ax.fill_between(
        rounds, min_curve, max_curve,
        alpha=0.1, color=color, label='Min-Max Range'
    )
    
    # Plot std band
    ax.fill_between(
        rounds,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3, color=color, label='±1 Std Dev'
    )
    
    # Plot mean curve
    ax.plot(rounds, mean_curve, color=color, linewidth=2, label=f'{name} (mean)')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\n({len(runs)} runs)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add statistics annotation
    stats_text = (
        f"Final: {mean_curve[-1]:.4f} ± {std_curve[-1]:.4f}\n"
        f"Best: {np.max(mean_curve):.4f}\n"
        f"Runs: {len(runs)}"
    )
    ax.annotate(
        stats_text,
        xy=(0.02, 0.98), xycoords='axes fraction',
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved shadow curves to {save_path}")
    
    return fig


def plot_convergence_comparison(
    results: List[Dict[str, Any]],
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional["Figure"]:
    """Plot convergence comparison across experiments.
    
    Args:
        results: List of dicts with 'name', 'curves' (list of runs), 'convergence_round'
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure or None
    """
    if not _check_matplotlib():
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Left: Accuracy curves
    for idx, result in enumerate(results):
        name = result.get('name', f'Exp {idx}')
        curves = result.get('curves', [])
        
        if not curves:
            continue
        
        if isinstance(curves[0], list):
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
        else:
            mean_curve = np.array(curves)
            std_curve = np.zeros_like(mean_curve)
        
        rounds = np.arange(1, len(mean_curve) + 1)
        ax1.plot(rounds, mean_curve, label=name, color=colors[idx], linewidth=2)
        ax1.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.2, color=colors[idx])
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Curves', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Convergence rounds bar chart
    names = [r.get('name', f'Exp {i}') for i, r in enumerate(results)]
    conv_rounds = [r.get('convergence_round', 0) for r in results]
    
    bars = ax2.barh(names, conv_rounds, color=colors, edgecolor='black')
    ax2.set_xlabel('Convergence Round', fontsize=12)
    ax2.set_title('Convergence Speed', fontsize=12)
    
    for bar, val in zip(bars, conv_rounds):
        ax2.annotate(
            f'{val}',
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va='center', fontsize=9
        )
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved convergence comparison to {save_path}")
    
    return fig


def plot_client_performance(
    client_metrics: Dict[int, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Per-Client Performance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional["Figure"]:
    """Plot per-client performance metrics.
    
    Args:
        client_metrics: Dict mapping client_id -> {metric_name -> value}
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure or None
    """
    if not _check_matplotlib():
        return None
    
    client_ids = sorted(client_metrics.keys())
    values = [client_metrics[cid].get(metric, 0) for cid in client_ids]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.array(values) / max(values) if max(values) > 0 else np.ones(len(values)))
    
    bars = ax.bar([f'Client {cid}' for cid in client_ids], values, color=colors, edgecolor='black')
    
    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    
    ax.set_xlabel('Client', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved client performance to {save_path}")
    
    return fig
