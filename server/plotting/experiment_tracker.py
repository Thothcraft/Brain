"""Comprehensive Experiment Tracker for ML/DL/FL Experiments.

Tracks all components of ML experiments:
- Time: epoch/batch/total timing, ETA estimation
- Performance: accuracy, loss, metrics over time
- Resources: CPU, GPU, memory, disk I/O
- Model: parameters, FLOPs, memory footprint
- Data: samples processed, augmentation stats
- Gradients: norms, flow, vanishing/exploding detection
- Hyperparameters: current values, search history

Usage:
    tracker = ExperimentTracker(experiment_name="my_experiment")
    tracker.start()
    
    for epoch in range(epochs):
        tracker.start_epoch(epoch)
        for batch in dataloader:
            tracker.start_batch()
            # ... training ...
            tracker.end_batch(loss=loss, metrics={'acc': acc})
        tracker.end_epoch(val_loss=val_loss, val_acc=val_acc)
    
    tracker.end()
    report = tracker.generate_report()
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class TimingMetrics:
    """Timing metrics for an experiment."""
    total_time_seconds: float = 0.0
    preprocessing_time: float = 0.0
    training_time: float = 0.0
    validation_time: float = 0.0
    evaluation_time: float = 0.0
    
    epoch_times: List[float] = field(default_factory=list)
    batch_times: List[float] = field(default_factory=list)
    
    avg_epoch_time: float = 0.0
    avg_batch_time: float = 0.0
    
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    # CPU
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    cpu_percent_timeline: List[float] = field(default_factory=list)
    
    # Memory
    memory_percent_avg: float = 0.0
    memory_percent_max: float = 0.0
    memory_gb_max: float = 0.0
    memory_timeline: List[float] = field(default_factory=list)
    
    # GPU
    gpu_percent_avg: float = 0.0
    gpu_percent_max: float = 0.0
    gpu_memory_percent_avg: float = 0.0
    gpu_memory_percent_max: float = 0.0
    gpu_memory_gb_max: float = 0.0
    gpu_timeline: List[float] = field(default_factory=list)
    gpu_memory_timeline: List[float] = field(default_factory=list)
    
    # Disk
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


@dataclass
class ModelMetrics:
    """Model complexity metrics."""
    total_parameters: int = 0
    trainable_parameters: int = 0
    non_trainable_parameters: int = 0
    
    model_size_mb: float = 0.0
    
    estimated_flops: int = 0
    estimated_memory_mb: float = 0.0
    
    layer_info: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Training performance metrics."""
    # Per-epoch
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Final
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    
    # Per-class
    class_accuracies: Dict[str, float] = field(default_factory=dict)
    class_f1_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class GradientMetrics:
    """Gradient analysis metrics."""
    grad_norms: List[float] = field(default_factory=list)
    grad_norm_avg: float = 0.0
    grad_norm_max: float = 0.0
    grad_norm_min: float = 0.0
    
    vanishing_detected: bool = False
    exploding_detected: bool = False
    
    layer_grad_norms: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class DataMetrics:
    """Data processing metrics."""
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    
    samples_processed: int = 0
    batches_processed: int = 0
    
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    augmentation_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentMetadata:
    """Experiment metadata."""
    experiment_name: str = ""
    experiment_id: str = ""
    
    model_type: str = ""
    model_name: str = ""
    
    dataset_name: str = ""
    
    started_at: str = ""
    completed_at: str = ""
    
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    tags: List[str] = field(default_factory=list)
    notes: str = ""


class ResourceMonitor:
    """Background thread for monitoring system resources."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._running = False
        self._thread = None
        
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.gpu_memory_samples = []
        self.timestamps = []
    
    def start(self):
        """Start monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        start_time = time.time()
        
        while self._running:
            try:
                self.timestamps.append(time.time() - start_time)
                
                # CPU
                if PSUTIL_AVAILABLE:
                    self.cpu_samples.append(psutil.cpu_percent())
                    self.memory_samples.append(psutil.virtual_memory().percent)
                
                # GPU
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.gpu_samples.append(gpus[0].load * 100)
                        self.gpu_memory_samples.append(gpus[0].memoryUtil * 100)
                
            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get aggregated resource metrics."""
        metrics = ResourceMetrics()
        
        if self.cpu_samples:
            metrics.cpu_percent_avg = np.mean(self.cpu_samples)
            metrics.cpu_percent_max = np.max(self.cpu_samples)
            metrics.cpu_percent_timeline = self.cpu_samples.copy()
        
        if self.memory_samples:
            metrics.memory_percent_avg = np.mean(self.memory_samples)
            metrics.memory_percent_max = np.max(self.memory_samples)
            metrics.memory_timeline = self.memory_samples.copy()
            if PSUTIL_AVAILABLE:
                metrics.memory_gb_max = psutil.virtual_memory().total / (1024**3) * metrics.memory_percent_max / 100
        
        if self.gpu_samples:
            metrics.gpu_percent_avg = np.mean(self.gpu_samples)
            metrics.gpu_percent_max = np.max(self.gpu_samples)
            metrics.gpu_timeline = self.gpu_samples.copy()
        
        if self.gpu_memory_samples:
            metrics.gpu_memory_percent_avg = np.mean(self.gpu_memory_samples)
            metrics.gpu_memory_percent_max = np.max(self.gpu_memory_samples)
            metrics.gpu_memory_timeline = self.gpu_memory_samples.copy()
        
        return metrics


class ExperimentTracker:
    """Comprehensive experiment tracker for ML/DL/FL experiments."""
    
    def __init__(self, experiment_name: str, output_dir: str = None,
                 monitor_resources: bool = True, monitor_interval: float = 1.0):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save tracking data
            monitor_resources: Whether to monitor system resources
            monitor_interval: Resource monitoring interval in seconds
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir or f"./experiments/{experiment_name}"
        self.monitor_resources = monitor_resources
        
        # Initialize metrics
        self.metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            experiment_id=f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.timing = TimingMetrics()
        self.resources = ResourceMetrics()
        self.model = ModelMetrics()
        self.performance = PerformanceMetrics()
        self.gradients = GradientMetrics()
        self.data = DataMetrics()
        
        # Internal state
        self._start_time = None
        self._epoch_start_time = None
        self._batch_start_time = None
        self._current_epoch = 0
        self._batch_count = 0
        self._samples_this_epoch = 0
        
        # Resource monitor
        self._resource_monitor = ResourceMonitor(interval=monitor_interval) if monitor_resources else None
        
        # Callbacks
        self._callbacks: List[Callable] = []
    
    def start(self):
        """Start experiment tracking."""
        self._start_time = time.time()
        self.metadata.started_at = datetime.now().isoformat()
        
        if self._resource_monitor:
            self._resource_monitor.start()
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Started tracking experiment: {self.experiment_name}")
        self._notify_callbacks('experiment_start', {})
    
    def end(self):
        """End experiment tracking."""
        if self._start_time:
            self.timing.total_time_seconds = time.time() - self._start_time
        
        self.metadata.completed_at = datetime.now().isoformat()
        
        if self._resource_monitor:
            self._resource_monitor.stop()
            self.resources = self._resource_monitor.get_metrics()
        
        # Compute final statistics
        self._compute_final_stats()
        
        # Save tracking data
        self.save()
        
        logger.info(f"Ended tracking experiment: {self.experiment_name}")
        self._notify_callbacks('experiment_end', self.get_summary())
    
    def start_epoch(self, epoch: int):
        """Start tracking an epoch."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()
        self._samples_this_epoch = 0
        self._notify_callbacks('epoch_start', {'epoch': epoch})
    
    def end_epoch(self, train_loss: float = None, train_acc: float = None,
                 val_loss: float = None, val_acc: float = None,
                 learning_rate: float = None, **kwargs):
        """End tracking an epoch."""
        if self._epoch_start_time:
            epoch_time = time.time() - self._epoch_start_time
            self.timing.epoch_times.append(epoch_time)
        
        # Record metrics
        if train_loss is not None:
            self.performance.train_loss.append(train_loss)
        if train_acc is not None:
            self.performance.train_accuracy.append(train_acc)
        if val_loss is not None:
            self.performance.val_loss.append(val_loss)
        if val_acc is not None:
            self.performance.val_accuracy.append(val_acc)
            if val_acc > self.performance.best_val_accuracy:
                self.performance.best_val_accuracy = val_acc
                self.performance.best_epoch = self._current_epoch
        if val_loss is not None and val_loss < self.performance.best_val_loss:
            self.performance.best_val_loss = val_loss
        if learning_rate is not None:
            self.performance.learning_rates.append(learning_rate)
        
        self._notify_callbacks('epoch_end', {
            'epoch': self._current_epoch,
            'train_loss': train_loss,
            'val_acc': val_acc,
            **kwargs
        })
    
    def start_batch(self):
        """Start tracking a batch."""
        self._batch_start_time = time.time()
    
    def end_batch(self, batch_size: int = None, loss: float = None, **kwargs):
        """End tracking a batch."""
        if self._batch_start_time:
            batch_time = time.time() - self._batch_start_time
            self.timing.batch_times.append(batch_time)
        
        self._batch_count += 1
        if batch_size:
            self._samples_this_epoch += batch_size
            self.data.samples_processed += batch_size
        self.data.batches_processed += 1
    
    def log_gradient_norm(self, norm: float, layer_norms: Dict[str, float] = None):
        """Log gradient norm."""
        self.gradients.grad_norms.append(norm)
        
        if layer_norms:
            for layer, n in layer_norms.items():
                if layer not in self.gradients.layer_grad_norms:
                    self.gradients.layer_grad_norms[layer] = []
                self.gradients.layer_grad_norms[layer].append(n)
        
        # Detect issues
        if norm < 1e-7:
            self.gradients.vanishing_detected = True
        if norm > 100:
            self.gradients.exploding_detected = True
    
    def log_model_info(self, model=None, total_params: int = None,
                      trainable_params: int = None, **kwargs):
        """Log model information."""
        if total_params is not None:
            self.model.total_parameters = total_params
        if trainable_params is not None:
            self.model.trainable_parameters = trainable_params
            self.model.non_trainable_parameters = (total_params or 0) - trainable_params
        
        # Try to extract from PyTorch model
        if model is not None:
            try:
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.model.total_parameters = total
                self.model.trainable_parameters = trainable
                self.model.non_trainable_parameters = total - trainable
                
                # Estimate size
                self.model.model_size_mb = total * 4 / (1024 * 1024)  # Assuming float32
            except Exception as e:
                logger.debug(f"Could not extract model info: {e}")
        
        for key, value in kwargs.items():
            setattr(self.model, key, value)
    
    def log_data_info(self, train_samples: int = None, val_samples: int = None,
                     test_samples: int = None, class_distribution: Dict = None, **kwargs):
        """Log data information."""
        if train_samples:
            self.data.train_samples = train_samples
        if val_samples:
            self.data.val_samples = val_samples
        if test_samples:
            self.data.test_samples = test_samples
        if class_distribution:
            self.data.class_distribution = class_distribution
        
        self.data.total_samples = (self.data.train_samples + 
                                   self.data.val_samples + 
                                   self.data.test_samples)
    
    def log_hyperparameters(self, **hyperparameters):
        """Log hyperparameters."""
        self.metadata.hyperparameters.update(hyperparameters)
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a custom metric."""
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}
        if name not in self._custom_metrics:
            self._custom_metrics[name] = []
        self._custom_metrics[name].append({'value': value, 'step': step})
    
    def _compute_final_stats(self):
        """Compute final statistics."""
        # Timing
        if self.timing.epoch_times:
            self.timing.avg_epoch_time = np.mean(self.timing.epoch_times)
        if self.timing.batch_times:
            self.timing.avg_batch_time = np.mean(self.timing.batch_times)
        
        if self.timing.total_time_seconds > 0 and self.data.samples_processed > 0:
            self.timing.samples_per_second = self.data.samples_processed / self.timing.total_time_seconds
        if self.timing.total_time_seconds > 0 and self.data.batches_processed > 0:
            self.timing.batches_per_second = self.data.batches_processed / self.timing.total_time_seconds
        
        # Gradients
        if self.gradients.grad_norms:
            self.gradients.grad_norm_avg = np.mean(self.gradients.grad_norms)
            self.gradients.grad_norm_max = np.max(self.gradients.grad_norms)
            self.gradients.grad_norm_min = np.min(self.gradients.grad_norms)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            'metadata': asdict(self.metadata),
            'timing': asdict(self.timing),
            'resources': asdict(self.resources),
            'model': asdict(self.model),
            'performance': asdict(self.performance),
            'gradients': asdict(self.gradients),
            'data': asdict(self.data),
        }
    
    def save(self, filename: str = None):
        """Save tracking data to file."""
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.metadata.experiment_id}.json")
        
        summary = self.get_summary()
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        summary = convert(summary)
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved experiment data to {filename}")
    
    def register_callback(self, callback: Callable):
        """Register a callback for tracking events."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, event: str, data: Dict):
        """Notify callbacks of an event."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def generate_report(self, output_format: str = 'html') -> str:
        """Generate a comprehensive report.
        
        Args:
            output_format: 'html', 'json', or 'markdown'
        
        Returns:
            Report content as string
        """
        from .advanced_plots import AdvancedPlotter, AdvancedPlotType
        
        plotter = AdvancedPlotter(theme='default')
        plots = {}
        
        # Generate plots
        try:
            # Resource timeline
            if self.resources.cpu_percent_timeline:
                _, b64 = plotter.create_plot(AdvancedPlotType.RESOURCE_TIMELINE, {
                    'cpu_percent': self.resources.cpu_percent_timeline,
                    'gpu_percent': self.resources.gpu_timeline,
                    'memory_percent': self.resources.memory_timeline,
                })
                plots['resource_timeline'] = b64
            
            # Timing breakdown
            if self.timing.training_time > 0:
                _, b64 = plotter.create_plot(AdvancedPlotType.TIMING_BREAKDOWN, {
                    'components': {
                        'Training': self.timing.training_time,
                        'Validation': self.timing.validation_time,
                        'Preprocessing': self.timing.preprocessing_time,
                    }
                })
                plots['timing_breakdown'] = b64
            
            # Epoch timing
            if self.timing.epoch_times:
                _, b64 = plotter.create_plot(AdvancedPlotType.EPOCH_TIMING, {
                    'epoch_times': self.timing.epoch_times,
                })
                plots['epoch_timing'] = b64
            
            # Gradient norms
            if self.gradients.grad_norms:
                _, b64 = plotter.create_plot(AdvancedPlotType.GRADIENT_NORM, {
                    'grad_norms': self.gradients.grad_norms,
                })
                plots['gradient_norm'] = b64
            
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
        
        if output_format == 'json':
            return json.dumps(self.get_summary(), indent=2)
        elif output_format == 'markdown':
            return self._generate_markdown_report(plots)
        else:
            return self._generate_html_report(plots)
    
    def _generate_html_report(self, plots: Dict[str, str]) -> str:
        """Generate HTML report."""
        summary = self.get_summary()
        
        html = f"""<!DOCTYPE html>
<html><head><title>Experiment Report: {self.experiment_name}</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
h1 {{ color: #333; border-bottom: 2px solid #0072B2; }}
h2 {{ color: #555; margin-top: 30px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
.metric {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
.metric-value {{ font-size: 24px; font-weight: bold; color: #0072B2; }}
.metric-label {{ font-size: 12px; color: #666; }}
.plot {{ margin: 20px 0; text-align: center; }}
.plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
</style></head><body><div class="container">
<h1>Experiment Report: {self.experiment_name}</h1>
<p><strong>ID:</strong> {self.metadata.experiment_id}</p>
<p><strong>Duration:</strong> {self.timing.total_time_seconds:.1f}s</p>

<h2>Performance Summary</h2>
<div class="metrics">
<div class="metric"><div class="metric-value">{self.performance.best_val_accuracy:.4f}</div><div class="metric-label">Best Val Accuracy</div></div>
<div class="metric"><div class="metric-value">{self.performance.best_epoch}</div><div class="metric-label">Best Epoch</div></div>
<div class="metric"><div class="metric-value">{self.model.total_parameters:,}</div><div class="metric-label">Parameters</div></div>
<div class="metric"><div class="metric-value">{self.timing.samples_per_second:.1f}</div><div class="metric-label">Samples/sec</div></div>
</div>

<h2>Resource Usage</h2>
<div class="metrics">
<div class="metric"><div class="metric-value">{self.resources.cpu_percent_avg:.1f}%</div><div class="metric-label">Avg CPU</div></div>
<div class="metric"><div class="metric-value">{self.resources.memory_percent_max:.1f}%</div><div class="metric-label">Peak Memory</div></div>
<div class="metric"><div class="metric-value">{self.resources.gpu_percent_avg:.1f}%</div><div class="metric-label">Avg GPU</div></div>
<div class="metric"><div class="metric-value">{self.resources.gpu_memory_percent_max:.1f}%</div><div class="metric-label">Peak GPU Mem</div></div>
</div>
"""
        
        for name, b64 in plots.items():
            title = name.replace('_', ' ').title()
            html += f'<h2>{title}</h2><div class="plot"><img src="data:image/png;base64,{b64}"></div>'
        
        html += "</div></body></html>"
        return html
    
    def _generate_markdown_report(self, plots: Dict[str, str]) -> str:
        """Generate Markdown report."""
        md = f"""# Experiment Report: {self.experiment_name}

**ID:** {self.metadata.experiment_id}  
**Duration:** {self.timing.total_time_seconds:.1f}s

## Performance Summary

| Metric | Value |
|--------|-------|
| Best Val Accuracy | {self.performance.best_val_accuracy:.4f} |
| Best Epoch | {self.performance.best_epoch} |
| Parameters | {self.model.total_parameters:,} |
| Samples/sec | {self.timing.samples_per_second:.1f} |

## Resource Usage

| Resource | Avg | Peak |
|----------|-----|------|
| CPU | {self.resources.cpu_percent_avg:.1f}% | {self.resources.cpu_percent_max:.1f}% |
| Memory | {self.resources.memory_percent_avg:.1f}% | {self.resources.memory_percent_max:.1f}% |
| GPU | {self.resources.gpu_percent_avg:.1f}% | {self.resources.gpu_percent_max:.1f}% |

## Hyperparameters

```json
{json.dumps(self.metadata.hyperparameters, indent=2)}
```
"""
        return md


# Convenience function
def create_tracker(name: str, **kwargs) -> ExperimentTracker:
    """Create an experiment tracker."""
    return ExperimentTracker(name, **kwargs)
