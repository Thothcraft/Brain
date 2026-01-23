"""Training Utilities for ML/DL Models.

This module provides training utilities that work with both
classical ML and deep learning models.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: torch.device = None,
    callback: Optional[Callable] = None,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    """Train a PyTorch model with validation.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        callback: Optional callback(epoch, total, train_loss, train_acc, val_loss, val_acc)
        early_stopping_patience: Epochs to wait before early stopping
    
    Returns:
        Dictionary with training history and best metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_loss = running_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_running_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Track best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Callback
        if callback:
            callback(epoch, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy)
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'final_epoch': epoch,
    }


def compute_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: torch.device = None,
) -> Dict[str, Any]:
    """Compute detailed metrics for a trained model.
    
    Args:
        model: Trained PyTorch model
        data_loader: Data loader for evaluation
        class_names: List of class names
        device: Device to use
    
    Returns:
        Dictionary with per-class metrics, confusion matrix, ROC/PR curves
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    per_class_metrics = []
    for i, name in enumerate(class_names):
        per_class_metrics.append({
            'class_name': name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
        })
    
    # ROC curves
    num_classes = len(class_names)
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    
    roc_curves = []
    for i, name in enumerate(class_names):
        if labels_bin.shape[1] > i:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_curves.append({
                'class_name': name,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc),
            })
    
    # PR curves
    pr_curves = []
    for i, name in enumerate(class_names):
        if labels_bin.shape[1] > i:
            prec, rec, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
            ap = auc(rec, prec)
            pr_curves.append({
                'class_name': name,
                'precision': prec.tolist(),
                'recall': rec.tolist(),
                'ap': float(ap),
            })
    
    return {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves,
        'accuracy': float(accuracy_score(all_labels, all_preds)),
    }


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders from numpy arrays.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of data loader workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def save_model_to_bytes(model: nn.Module, config: dict) -> bytes:
    """Save a PyTorch model to bytes.
    
    Args:
        model: Model to save
        config: Configuration to save with model
    
    Returns:
        Bytes containing the saved model
    """
    import io
    import pickle
    
    buffer = io.BytesIO()
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    torch.save(save_dict, buffer)
    return buffer.getvalue()


def load_model_from_bytes(model_class, model_bytes: bytes) -> Tuple[nn.Module, dict]:
    """Load a PyTorch model from bytes.
    
    Args:
        model_class: Model class to instantiate
        model_bytes: Bytes containing the saved model
    
    Returns:
        Tuple of (model, config)
    """
    import io
    
    buffer = io.BytesIO(model_bytes)
    save_dict = torch.load(buffer, map_location='cpu')
    
    config = save_dict['config']
    model = model_class(**config)
    model.load_state_dict(save_dict['model_state_dict'])
    
    return model, config
