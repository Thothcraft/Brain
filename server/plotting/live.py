"""Live plotting system for real-time training visualization.

Provides real-time plot updates during training with:
- Efficient incremental updates
- WebSocket support for browser display
- Callback system for custom integrations
- Memory-efficient data management
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class LiveDataBuffer:
    """Circular buffer for live data with configurable max size."""
    max_size: int = 1000
    
    epochs: deque = field(default_factory=lambda: deque(maxlen=1000))
    train_loss: deque = field(default_factory=lambda: deque(maxlen=1000))
    val_loss: deque = field(default_factory=lambda: deque(maxlen=1000))
    train_acc: deque = field(default_factory=lambda: deque(maxlen=1000))
    val_acc: deque = field(default_factory=lambda: deque(maxlen=1000))
    learning_rate: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # FL-specific
    rounds: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_accuracy: deque = field(default_factory=lambda: deque(maxlen=1000))
    client_accuracies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def __post_init__(self):
        # Update maxlen based on max_size
        for attr in ['epochs', 'train_loss', 'val_loss', 'train_acc', 
                     'val_acc', 'learning_rate', 'rounds', 'avg_accuracy',
                     'client_accuracies']:
            setattr(self, attr, deque(maxlen=self.max_size))
    
    def add_epoch_data(self, epoch: int, train_loss: float = None,
                      val_loss: float = None, train_acc: float = None,
                      val_acc: float = None, lr: float = None):
        """Add data for a training epoch."""
        self.epochs.append(epoch)
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if train_acc is not None:
            self.train_acc.append(train_acc)
        if val_acc is not None:
            self.val_acc.append(val_acc)
        if lr is not None:
            self.learning_rate.append(lr)
    
    def add_round_data(self, round_num: int, avg_acc: float = None,
                      client_accs: List[float] = None):
        """Add data for an FL round."""
        self.rounds.append(round_num)
        if avg_acc is not None:
            self.avg_accuracy.append(avg_acc)
        if client_accs is not None:
            self.client_accuracies.append(client_accs)
    
    def clear(self):
        """Clear all buffers."""
        for attr in ['epochs', 'train_loss', 'val_loss', 'train_acc',
                     'val_loss', 'learning_rate', 'rounds', 'avg_accuracy',
                     'client_accuracies']:
            getattr(self, attr).clear()
    
    def to_dict(self) -> Dict[str, List]:
        """Convert buffers to dictionary."""
        return {
            'epochs': list(self.epochs),
            'train_loss': list(self.train_loss),
            'val_loss': list(self.val_loss),
            'train_acc': list(self.train_acc),
            'val_acc': list(self.val_acc),
            'learning_rate': list(self.learning_rate),
            'rounds': list(self.rounds),
            'avg_accuracy': list(self.avg_accuracy),
        }


class LivePlotManager:
    """Manages live plot updates during training.
    
    Supports both matplotlib interactive mode and callback-based
    updates for web interfaces.
    """
    
    def __init__(self, plotter, update_interval: float = 0.5,
                 max_buffer_size: int = 1000):
        """Initialize live plot manager.
        
        Args:
            plotter: Parent plotter instance
            update_interval: Minimum seconds between plot updates
            max_buffer_size: Maximum data points to keep in buffer
        """
        self.plotter = plotter
        self.update_interval = update_interval
        self.buffer = LiveDataBuffer(max_size=max_buffer_size)
        
        self._running = False
        self._update_thread = None
        self._last_update = 0
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
        # Matplotlib figures for live display
        self._fig = None
        self._axes = None
        self._lines = {}
    
    def start(self):
        """Start live plotting session."""
        self._running = True
        self.buffer.clear()
        
        if MATPLOTLIB_AVAILABLE:
            plt.ion()  # Enable interactive mode
            self._setup_live_figure()
        
        logger.info("Live plotting session started")
    
    def stop(self):
        """Stop live plotting session."""
        self._running = False
        
        if MATPLOTLIB_AVAILABLE:
            plt.ioff()
        
        # Final update
        self._trigger_update(force=True)
        
        logger.info("Live plotting session stopped")
    
    def _setup_live_figure(self):
        """Setup matplotlib figure for live updates."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self._fig, self._axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss subplot
        ax1 = self._axes[0]
        self._lines['train_loss'], = ax1.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self._lines['val_loss'], = ax1.plot([], [], 'r--', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy subplot
        ax2 = self._axes[1]
        self._lines['train_acc'], = ax2.plot([], [], 'b-', label='Train Acc', linewidth=2)
        self._lines['val_acc'], = ax2.plot([], [], 'r--', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self._fig.tight_layout()
        plt.show(block=False)
    
    def update(self, **kwargs):
        """Update live plots with new data.
        
        Args:
            **kwargs: Data to update. Supported keys:
                - epoch, train_loss, val_loss, train_acc, val_acc, lr (DL)
                - round, avg_accuracy, client_accuracies (FL)
        """
        if not self._running:
            return
        
        with self._lock:
            # Add to buffer
            if 'epoch' in kwargs:
                self.buffer.add_epoch_data(
                    epoch=kwargs.get('epoch'),
                    train_loss=kwargs.get('train_loss'),
                    val_loss=kwargs.get('val_loss'),
                    train_acc=kwargs.get('train_acc'),
                    val_acc=kwargs.get('val_acc'),
                    lr=kwargs.get('lr', kwargs.get('learning_rate'))
                )
            elif 'round' in kwargs:
                self.buffer.add_round_data(
                    round_num=kwargs.get('round'),
                    avg_acc=kwargs.get('avg_accuracy'),
                    client_accs=kwargs.get('client_accuracies')
                )
        
        # Throttle updates
        current_time = time.time()
        if current_time - self._last_update >= self.update_interval:
            self._trigger_update()
            self._last_update = current_time
    
    def _trigger_update(self, force: bool = False):
        """Trigger plot update."""
        # Update matplotlib figure
        if MATPLOTLIB_AVAILABLE and self._fig is not None:
            self._update_matplotlib()
        
        # Notify callbacks
        data = self.buffer.to_dict()
        for callback in self._callbacks:
            try:
                callback('update', data)
            except Exception as e:
                logger.warning(f"Live plot callback error: {e}")
    
    def _update_matplotlib(self):
        """Update matplotlib figure with current buffer data."""
        if not self._fig or not self._axes:
            return
        
        epochs = list(self.buffer.epochs)
        
        if not epochs:
            return
        
        # Update loss lines
        if self.buffer.train_loss:
            self._lines['train_loss'].set_data(epochs[:len(self.buffer.train_loss)],
                                               list(self.buffer.train_loss))
        if self.buffer.val_loss:
            self._lines['val_loss'].set_data(epochs[:len(self.buffer.val_loss)],
                                             list(self.buffer.val_loss))
        
        # Update accuracy lines
        if self.buffer.train_acc:
            self._lines['train_acc'].set_data(epochs[:len(self.buffer.train_acc)],
                                              list(self.buffer.train_acc))
        if self.buffer.val_acc:
            self._lines['val_acc'].set_data(epochs[:len(self.buffer.val_acc)],
                                            list(self.buffer.val_acc))
        
        # Rescale axes
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()
        
        # Redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
    
    def register_callback(self, callback: Callable):
        """Register a callback for live updates.
        
        Args:
            callback: Function(event: str, data: dict) to call on updates
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_current_data(self) -> Dict[str, List]:
        """Get current buffer data."""
        with self._lock:
            return self.buffer.to_dict()
    
    def get_base64_snapshot(self) -> str:
        """Get base64 encoded PNG of current plot state."""
        if not MATPLOTLIB_AVAILABLE or not self._fig:
            return ""
        
        import io
        import base64
        
        buf = io.BytesIO()
        self._fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


class WebSocketLivePlotter:
    """Live plotter with WebSocket support for browser display.
    
    Sends plot updates to connected WebSocket clients in real-time.
    """
    
    def __init__(self, plotter, websocket_handler=None):
        """Initialize WebSocket live plotter.
        
        Args:
            plotter: Parent plotter instance
            websocket_handler: WebSocket handler for sending updates
        """
        self.plotter = plotter
        self.websocket_handler = websocket_handler
        self.live_manager = LivePlotManager(plotter)
        
        # Register WebSocket callback
        self.live_manager.register_callback(self._on_update)
    
    def _on_update(self, event: str, data: Dict):
        """Handle live update events."""
        if self.websocket_handler and event == 'update':
            try:
                # Send data to WebSocket clients
                message = {
                    'type': 'plot_update',
                    'data': data,
                    'timestamp': time.time(),
                }
                self.websocket_handler.broadcast(message)
            except Exception as e:
                logger.warning(f"WebSocket broadcast error: {e}")
    
    def start(self):
        """Start live plotting."""
        self.live_manager.start()
    
    def stop(self):
        """Stop live plotting."""
        self.live_manager.stop()
    
    def update(self, **kwargs):
        """Update with new data."""
        self.live_manager.update(**kwargs)


class TrainingProgressTracker:
    """Track and visualize training progress with ETA estimation."""
    
    def __init__(self, total_epochs: int = None, total_rounds: int = None):
        """Initialize progress tracker.
        
        Args:
            total_epochs: Total epochs for DL training
            total_rounds: Total rounds for FL training
        """
        self.total_epochs = total_epochs
        self.total_rounds = total_rounds
        self.total = total_epochs or total_rounds or 0
        
        self.start_time = None
        self.current = 0
        self.history: List[Dict] = []
    
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.current = 0
        self.history.clear()
    
    def update(self, current: int, metrics: Dict[str, float] = None):
        """Update progress.
        
        Args:
            current: Current epoch/round number
            metrics: Optional metrics dictionary
        """
        self.current = current
        
        entry = {
            'step': current,
            'time': time.time() - self.start_time if self.start_time else 0,
            'metrics': metrics or {},
        }
        self.history.append(entry)
    
    def get_eta(self) -> float:
        """Get estimated time remaining in seconds."""
        if not self.history or self.current == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        return remaining / rate if rate > 0 else 0
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        eta = self.get_eta()
        
        return {
            'current': self.current,
            'total': self.total,
            'percent': (self.current / self.total * 100) if self.total > 0 else 0,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta,
            'rate': self.current / elapsed if elapsed > 0 else 0,
        }
    
    def format_progress(self) -> str:
        """Get formatted progress string."""
        p = self.get_progress()
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"
        
        return (f"[{p['current']}/{p['total']}] "
                f"{p['percent']:.1f}% | "
                f"Elapsed: {format_time(p['elapsed_seconds'])} | "
                f"ETA: {format_time(p['eta_seconds'])}")
