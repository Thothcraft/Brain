"""Metrics Tracker - Standardized metric tracking for all training modes.

Provides unified metric tracking across central ML, DL, and FL training.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    batch_idx: int
    loss: float
    accuracy: float = 0.0
    learning_rate: float = 0.0
    batch_size: int = 0
    time_ms: float = 0.0


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    epoch_time_seconds: float = 0.0
    batches: List[BatchMetrics] = field(default_factory=list)
    
    # Additional metrics
    train_precision: Optional[float] = None
    train_recall: Optional[float] = None
    train_f1: Optional[float] = None
    val_precision: Optional[float] = None
    val_recall: Optional[float] = None
    val_f1: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
            "epoch_time_seconds": self.epoch_time_seconds,
            "train_precision": self.train_precision,
            "train_recall": self.train_recall,
            "train_f1": self.train_f1,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
        }


@dataclass
class ClassMetrics:
    """Per-class metrics."""
    class_name: str
    class_idx: int
    precision: float
    recall: float
    f1_score: float
    support: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class FLRoundMetrics:
    """Metrics for a federated learning round."""
    round_num: int
    global_loss: float
    global_accuracy: float
    num_clients: int
    aggregation_time_ms: float = 0.0
    client_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Per-client statistics
    avg_client_loss: float = 0.0
    avg_client_accuracy: float = 0.0
    min_client_accuracy: float = 0.0
    max_client_accuracy: float = 0.0
    client_variance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "global_loss": self.global_loss,
            "global_accuracy": self.global_accuracy,
            "num_clients": self.num_clients,
            "aggregation_time_ms": self.aggregation_time_ms,
            "avg_client_loss": self.avg_client_loss,
            "avg_client_accuracy": self.avg_client_accuracy,
            "min_client_accuracy": self.min_client_accuracy,
            "max_client_accuracy": self.max_client_accuracy,
            "client_variance": self.client_variance,
        }


class MetricsTracker:
    """Unified metrics tracker for all training modes.
    
    Tracks:
    - Epoch-level metrics (loss, accuracy, precision, recall, F1)
    - Batch-level metrics (for detailed analysis)
    - Class-level metrics (per-class performance)
    - FL round metrics (for federated learning)
    - Timing information
    - Hardware utilization
    """
    
    def __init__(
        self,
        job_id: str,
        training_mode: str = "central",  # central, federated
        num_classes: int = 2,
        class_names: List[str] = None,
    ):
        self.job_id = job_id
        self.training_mode = training_mode
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Metric storage
        self.epoch_metrics: List[EpochMetrics] = []
        self.fl_round_metrics: List[FLRoundMetrics] = []
        self.class_metrics: List[ClassMetrics] = []
        
        # Confusion matrix
        self.confusion_matrix: Optional[np.ndarray] = None
        
        # ROC/PR curves
        self.roc_curves: Dict[str, Dict[str, List[float]]] = {}
        self.pr_curves: Dict[str, Dict[str, List[float]]] = {}
        
        # Best metrics
        self.best_val_accuracy = 0.0
        self.best_val_epoch = 0
        self.best_val_loss = float('inf')
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_training_time = 0.0
        
        # Hardware
        self.device = "cpu"
        self.gpu_name: Optional[str] = None
        self.peak_memory_mb = 0.0
    
    def start_training(self):
        """Mark training start."""
        self.start_time = datetime.now()
    
    def end_training(self):
        """Mark training end."""
        self.end_time = datetime.now()
        if self.start_time:
            self.total_training_time = (self.end_time - self.start_time).total_seconds()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: float = 0.0,
        epoch_time: float = 0.0,
        **kwargs,
    ):
        """Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
            **kwargs: Additional metrics (precision, recall, f1, etc.)
        """
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            epoch_time_seconds=epoch_time,
            train_precision=kwargs.get("train_precision"),
            train_recall=kwargs.get("train_recall"),
            train_f1=kwargs.get("train_f1"),
            val_precision=kwargs.get("val_precision"),
            val_recall=kwargs.get("val_recall"),
            val_f1=kwargs.get("val_f1"),
        )
        
        self.epoch_metrics.append(metrics)
        
        # Update best metrics
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_val_epoch = epoch
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        logger.debug(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}")
    
    def log_fl_round(
        self,
        round_num: int,
        global_loss: float,
        global_accuracy: float,
        num_clients: int,
        client_metrics: List[Dict[str, Any]] = None,
        aggregation_time: float = 0.0,
    ):
        """Log metrics for an FL round.
        
        Args:
            round_num: Round number
            global_loss: Global model loss
            global_accuracy: Global model accuracy
            num_clients: Number of participating clients
            client_metrics: Per-client metrics
            aggregation_time: Time for aggregation
        """
        client_metrics = client_metrics or []
        
        # Calculate client statistics
        client_accs = [c.get("accuracy", 0) for c in client_metrics]
        client_losses = [c.get("loss", 0) for c in client_metrics]
        
        metrics = FLRoundMetrics(
            round_num=round_num,
            global_loss=global_loss,
            global_accuracy=global_accuracy,
            num_clients=num_clients,
            aggregation_time_ms=aggregation_time * 1000,
            client_metrics=client_metrics,
            avg_client_loss=np.mean(client_losses) if client_losses else 0,
            avg_client_accuracy=np.mean(client_accs) if client_accs else 0,
            min_client_accuracy=min(client_accs) if client_accs else 0,
            max_client_accuracy=max(client_accs) if client_accs else 0,
            client_variance=np.var(client_accs) if client_accs else 0,
        )
        
        self.fl_round_metrics.append(metrics)
        
        # Update best metrics
        if global_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = global_accuracy
            self.best_val_epoch = round_num
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Log confusion matrix from predictions."""
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    def log_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate and log per-class metrics."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        self.class_metrics = []
        for i in range(len(precision)):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            self.class_metrics.append(ClassMetrics(
                class_name=class_name,
                class_idx=i,
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1_score=float(f1[i]),
                support=int(support[i]),
            ))
    
    def log_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Calculate and log ROC curves for each class."""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels for multi-class
        if self.num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        else:
            y_true_bin = y_true.reshape(-1, 1)
            y_proba = y_proba[:, 1:2] if y_proba.ndim > 1 else y_proba.reshape(-1, 1)
        
        self.roc_curves = {}
        for i in range(y_true_bin.shape[1]):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i] if y_proba.ndim > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            
            self.roc_curves[class_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc),
            }
    
    def log_pr_curves(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Calculate and log Precision-Recall curves."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        
        if self.num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        else:
            y_true_bin = y_true.reshape(-1, 1)
            y_proba = y_proba[:, 1:2] if y_proba.ndim > 1 else y_proba.reshape(-1, 1)
        
        self.pr_curves = {}
        for i in range(y_true_bin.shape[1]):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i] if y_proba.ndim > 1 else y_proba)
            ap = average_precision_score(y_true_bin[:, i], y_proba[:, i] if y_proba.ndim > 1 else y_proba)
            
            self.pr_curves[class_name] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "ap": float(ap),
            }
    
    def log_hardware_info(self, device: str, gpu_name: str = None, peak_memory_mb: float = 0):
        """Log hardware information."""
        self.device = device
        self.gpu_name = gpu_name
        self.peak_memory_mb = peak_memory_mb
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        return {
            "job_id": self.job_id,
            "training_mode": self.training_mode,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            
            # Training progress
            "total_epochs": len(self.epoch_metrics),
            "total_fl_rounds": len(self.fl_round_metrics),
            
            # Best metrics
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_epoch": self.best_val_epoch,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float('inf') else None,
            
            # Final metrics
            "final_train_loss": self.epoch_metrics[-1].train_loss if self.epoch_metrics else None,
            "final_train_accuracy": self.epoch_metrics[-1].train_accuracy if self.epoch_metrics else None,
            "final_val_loss": self.epoch_metrics[-1].val_loss if self.epoch_metrics else None,
            "final_val_accuracy": self.epoch_metrics[-1].val_accuracy if self.epoch_metrics else None,
            
            # Timing
            "total_training_time_seconds": self.total_training_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            
            # Hardware
            "device": self.device,
            "gpu_name": self.gpu_name,
            "peak_memory_mb": self.peak_memory_mb,
        }
    
    def get_epoch_history(self) -> List[Dict[str, Any]]:
        """Get epoch metrics history."""
        return [m.to_dict() for m in self.epoch_metrics]
    
    def get_fl_round_history(self) -> List[Dict[str, Any]]:
        """Get FL round metrics history."""
        return [m.to_dict() for m in self.fl_round_metrics]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics to dictionary."""
        return {
            "summary": self.get_summary(),
            "epoch_metrics": self.get_epoch_history(),
            "fl_round_metrics": self.get_fl_round_history(),
            "class_metrics": [
                {
                    "class_name": c.class_name,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1_score": c.f1_score,
                    "support": c.support,
                }
                for c in self.class_metrics
            ],
            "confusion_matrix": self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            "roc_curves": self.roc_curves,
            "pr_curves": self.pr_curves,
        }
