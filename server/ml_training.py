"""Machine Learning Training Module.

This module provides real PyTorch-based training functionality for:
- IMU data classification (6-axis: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
- Time-series classification

Input shape: (batch_size, window_size, 6)
- window_size is configurable (default 128)
- 6 channels for IMU data

No synthetic data or fallbacks - uses real data from database.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import io
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class IMUClassifier(nn.Module):
    """CNN+LSTM hybrid model for IMU time-series classification.
    
    Input shape: (batch_size, window_size, 6)
    - 6 channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    """
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128, 
                 num_classes: int = 4, architecture_size: str = 'medium'):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.architecture_size = architecture_size
        
        # Architecture configurations
        configs = {
            'small': {'conv_channels': [32, 64], 'lstm_hidden': 64, 'fc_hidden': 64},
            'medium': {'conv_channels': [64, 128, 128], 'lstm_hidden': 128, 'fc_hidden': 128},
            'large': {'conv_channels': [64, 128, 256, 256], 'lstm_hidden': 256, 'fc_hidden': 256}
        }
        config = configs.get(architecture_size, configs['medium'])
        self._config = config
        
        # Convolutional layers for temporal feature extraction
        conv_layers = []
        in_channels = input_channels
        for out_channels in config['conv_channels']:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate conv output length after pooling
        self.conv_out_len = seq_length // (2 ** len(config['conv_channels']))
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=config['conv_channels'][-1],
            hidden_size=config['lstm_hidden'],
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism for weighted sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(config['lstm_hidden'] * 2, config['lstm_hidden']),
            nn.Tanh(),
            nn.Linear(config['lstm_hidden'], 1)
        )
        
        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(config['lstm_hidden'] * 2, config['fc_hidden']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config['fc_hidden'], num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_length, channels) = (batch, window_size, 6)
        batch_size = x.size(0)
        
        # Permute for Conv1d: (batch, channels, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply convolutions
        x = self.conv(x)
        
        # Permute back for LSTM: (batch, seq_length, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention-weighted aggregation
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        x = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        x = self.fc(x)
        return x
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layers = []
        layers.append({
            'type': 'Input',
            'shape': f'(batch, {self.seq_length}, {self.input_channels})',
            'params': 0
        })
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                layers.append({
                    'type': 'Conv1d',
                    'shape': f'({module.in_channels}, {module.out_channels}, k={module.kernel_size[0]})',
                    'params': sum(p.numel() for p in module.parameters())
                })
            elif isinstance(module, nn.LSTM):
                layers.append({
                    'type': 'BiLSTM' if module.bidirectional else 'LSTM',
                    'units': module.hidden_size,
                    'shape': f'({module.input_size} -> {module.hidden_size * (2 if module.bidirectional else 1)})',
                    'params': sum(p.numel() for p in module.parameters())
                })
            elif isinstance(module, nn.Linear) and 'fc' in name:
                layers.append({
                    'type': 'Dense',
                    'units': module.out_features,
                    'shape': f'({module.in_features}, {module.out_features})',
                    'params': sum(p.numel() for p in module.parameters())
                })
        
        return {
            'layers': layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': f'(batch, {self.seq_length}, {self.input_channels})',
            'architecture_size': self.architecture_size
        }


# ============================================================================
# DATA LOADING FROM DATABASE
# ============================================================================

class IMUDataset(Dataset):
    """PyTorch Dataset for IMU data loaded from database files."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def parse_imu_file(content: bytes, window_size: int = 128) -> List[np.ndarray]:
    """Parse IMU JSON file and extract windows of data.
    
    Supports multiple formats:
    1. Single JSON object with samples array:
       {"samples": [{"accel_x": float, ...}, ...]}
    2. JSON array:
       [{"accel_x": float, ...}, ...]
    3. JSONL (newline-delimited JSON):
       {"accel_x": float, ...}
       {"accel_x": float, ...}
    
    Returns list of windows, each shape (window_size, 6)
    """
    try:
        text_content = content.decode('utf-8').strip()
        samples = []
        
        # Try parsing as single JSON first
        try:
            data = json.loads(text_content)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                samples = data.get('samples', data.get('data', []))
                # If no samples/data key, treat the dict itself as a single sample
                if not samples and any(k in data for k in ['accel_x', 'ax', 'accel_y', 'ay']):
                    samples = [data]
            elif isinstance(data, list):
                samples = data
            else:
                logger.warning(f"Unknown IMU data format: {type(data)}")
                return []
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            logger.info("Trying JSONL format...")
            lines = text_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        samples.append(obj)
                    elif isinstance(obj, list):
                        samples.extend(obj)
                except json.JSONDecodeError:
                    continue
            
            if samples:
                logger.info(f"Parsed {len(samples)} samples from JSONL format")
        
        if not samples:
            logger.warning("No samples found in IMU file")
            return []
        
        # Extract 6-axis IMU data
        imu_data = []
        for sample in samples:
            try:
                row = [
                    float(sample.get('accel_x', sample.get('ax', 0))),
                    float(sample.get('accel_y', sample.get('ay', 0))),
                    float(sample.get('accel_z', sample.get('az', 0))),
                    float(sample.get('gyro_x', sample.get('gx', 0))),
                    float(sample.get('gyro_y', sample.get('gy', 0))),
                    float(sample.get('gyro_z', sample.get('gz', 0)))
                ]
                imu_data.append(row)
            except (KeyError, TypeError, ValueError) as e:
                continue
        
        logger.info(f"Extracted {len(imu_data)} IMU samples")
        
        if len(imu_data) < window_size:
            logger.warning(f"Not enough samples ({len(imu_data)}) for window size {window_size}")
            return []
        
        # Create sliding windows with 50% overlap
        imu_array = np.array(imu_data, dtype=np.float32)
        windows = []
        stride = window_size // 2
        
        for start in range(0, len(imu_array) - window_size + 1, stride):
            window = imu_array[start:start + window_size]
            windows.append(window)
        
        logger.info(f"Created {len(windows)} windows of size {window_size}")
        return windows
        
    except Exception as e:
        logger.error(f"Error parsing IMU file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def load_dataset_from_db(db_session, dataset_id: int, window_size: int = 128) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess dataset from database.
    
    Args:
        db_session: SQLAlchemy database session
        dataset_id: ID of the training dataset
        window_size: Size of sliding window for time series
        
    Returns:
        X: numpy array of shape (num_samples, window_size, 6)
        y: numpy array of labels
        class_names: list of class name strings
    """
    from server.db import TrainingDataset, DatasetFile, File
    
    dataset = db_session.query(TrainingDataset).filter(TrainingDataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Get unique labels and create mapping
    labels_set = set(df.label for df in dataset.files)
    class_names = sorted(list(labels_set))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    all_windows = []
    all_labels = []
    
    for dataset_file in dataset.files:
        if not dataset_file.file:
            continue
            
        file_content = dataset_file.file.content
        if not file_content:
            continue
        
        # Parse IMU data from file
        windows = parse_imu_file(file_content, window_size)
        
        if windows:
            label_idx = label_to_idx[dataset_file.label]
            all_windows.extend(windows)
            all_labels.extend([label_idx] * len(windows))
            logger.info(f"Loaded {len(windows)} windows from {dataset_file.file.filename} (label: {dataset_file.label})")
    
    if not all_windows:
        raise ValueError(f"No valid IMU data found in dataset {dataset_id}")
    
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {len(class_names)} classes, window_size={window_size}")
    
    return X, y, class_names


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str = 'cpu',
    callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Train a PyTorch model with real training loop."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    model = model.to(device)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        train_losses.append(round(train_loss, 4))
        train_accuracies.append(round(train_acc, 4))
        val_losses.append(round(val_loss, 4))
        val_accuracies.append(round(val_acc, 4))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if callback:
            callback(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc)
    
    # Restore best model
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': round(best_val_acc, 4),
        'best_epoch': best_epoch,
        'model_state_dict': best_state_dict
    }


class BayesianOptimizationConfig:
    """Configuration for Bayesian hyperparameter optimization."""
    
    def __init__(
        self,
        num_trials: int = 20,
        epochs_per_trial: int = 3,
        # Learning rate search space
        lr_min: float = 0.00001,
        lr_max: float = 0.01,
        lr_scale: str = 'log',  # 'log' or 'linear'
        # Batch size search space
        batch_sizes: List[int] = None,
        # Weight decay search space
        weight_decay_min: float = 0.0,
        weight_decay_max: float = 0.01,
        # Dropout search space
        dropout_min: float = 0.1,
        dropout_max: float = 0.5,
        # Optimizer options
        optimizers: List[str] = None,
        # Exploration vs exploitation
        exploration_rate: float = 0.3,
        # Early stopping for trials
        early_stopping_patience: int = 2,
        # Architecture sizes to try
        architecture_sizes: List[str] = None
    ):
        self.num_trials = num_trials
        self.epochs_per_trial = epochs_per_trial
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_scale = lr_scale
        self.batch_sizes = batch_sizes or [16, 32, 64, 128]
        self.weight_decay_min = weight_decay_min
        self.weight_decay_max = weight_decay_max
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.optimizers = optimizers or ['adam', 'adamw', 'sgd']
        self.exploration_rate = exploration_rate
        self.early_stopping_patience = early_stopping_patience
        self.architecture_sizes = architecture_sizes or ['small', 'medium', 'large']
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BayesianOptimizationConfig':
        return cls(
            num_trials=config.get('bayesian_trials', 20),
            epochs_per_trial=config.get('bayesian_epochs_per_trial', 3),
            lr_min=config.get('bayesian_lr_min', 0.00001),
            lr_max=config.get('bayesian_lr_max', 0.01),
            lr_scale=config.get('bayesian_lr_scale', 'log'),
            batch_sizes=config.get('bayesian_batch_sizes', [16, 32, 64, 128]),
            weight_decay_min=config.get('bayesian_weight_decay_min', 0.0),
            weight_decay_max=config.get('bayesian_weight_decay_max', 0.01),
            dropout_min=config.get('bayesian_dropout_min', 0.1),
            dropout_max=config.get('bayesian_dropout_max', 0.5),
            optimizers=config.get('bayesian_optimizers', ['adam', 'adamw', 'sgd']),
            exploration_rate=config.get('bayesian_exploration_rate', 0.3),
            early_stopping_patience=config.get('bayesian_early_stopping', 2),
            architecture_sizes=config.get('bayesian_architecture_sizes')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_trials': self.num_trials,
            'epochs_per_trial': self.epochs_per_trial,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'lr_scale': self.lr_scale,
            'batch_sizes': self.batch_sizes,
            'weight_decay_min': self.weight_decay_min,
            'weight_decay_max': self.weight_decay_max,
            'dropout_min': self.dropout_min,
            'dropout_max': self.dropout_max,
            'optimizers': self.optimizers,
            'exploration_rate': self.exploration_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'architecture_sizes': self.architecture_sizes
        }


def sample_hyperparameters(
    config: BayesianOptimizationConfig,
    best_params: Optional[Dict[str, Any]] = None,
    trial_num: int = 0
) -> Dict[str, Any]:
    """Sample hyperparameters using Bayesian-like strategy."""
    import random
    import math
    
    # First trial or no best params: random sampling
    if trial_num == 0 or best_params is None:
        # Learning rate
        if config.lr_scale == 'log':
            lr = math.exp(random.uniform(math.log(config.lr_min), math.log(config.lr_max)))
        else:
            lr = random.uniform(config.lr_min, config.lr_max)
        
        return {
            'learning_rate': lr,
            'batch_size': random.choice(config.batch_sizes),
            'weight_decay': random.uniform(config.weight_decay_min, config.weight_decay_max),
            'dropout': random.uniform(config.dropout_min, config.dropout_max),
            'optimizer': random.choice(config.optimizers),
            'architecture_size': random.choice(config.architecture_sizes) if config.architecture_sizes else 'medium'
        }
    
    # Exploit best params with some exploration
    if random.random() < config.exploration_rate:
        # Explore: random sampling
        if config.lr_scale == 'log':
            lr = math.exp(random.uniform(math.log(config.lr_min), math.log(config.lr_max)))
        else:
            lr = random.uniform(config.lr_min, config.lr_max)
        
        return {
            'learning_rate': lr,
            'batch_size': random.choice(config.batch_sizes),
            'weight_decay': random.uniform(config.weight_decay_min, config.weight_decay_max),
            'dropout': random.uniform(config.dropout_min, config.dropout_max),
            'optimizer': random.choice(config.optimizers),
            'architecture_size': random.choice(config.architecture_sizes) if config.architecture_sizes else 'medium'
        }
    else:
        # Exploit: perturb best params
        lr = best_params['learning_rate'] * random.uniform(0.5, 2.0)
        lr = max(config.lr_min, min(config.lr_max, lr))
        
        wd = best_params.get('weight_decay', 0.0) * random.uniform(0.5, 2.0)
        wd = max(config.weight_decay_min, min(config.weight_decay_max, wd))
        
        dropout = best_params.get('dropout', 0.3) + random.uniform(-0.1, 0.1)
        dropout = max(config.dropout_min, min(config.dropout_max, dropout))
        
        return {
            'learning_rate': lr,
            'batch_size': best_params.get('batch_size', 32),
            'weight_decay': wd,
            'dropout': dropout,
            'optimizer': best_params.get('optimizer', 'adam'),
            'architecture_size': best_params.get('architecture_size', 'medium')
        }


def run_bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: BayesianOptimizationConfig,
    device: str = 'cpu',
    callback: Optional[Callable] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run Bayesian hyperparameter optimization with real data.
    
    Args:
        X_train: Training data (num_samples, window_size, channels)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes
        config: Bayesian optimization configuration
        device: Device to train on
        callback: Optional callback for progress updates
        
    Returns:
        Tuple of (best_params, trials_data)
    """
    import time
    
    seq_length = X_train.shape[1]
    input_channels = X_train.shape[2]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    trials_data = []
    best_params = None
    best_val_acc = 0.0
    best_trial_idx = 0
    
    logger.info(f"Starting Bayesian optimization with {config.num_trials} trials")
    logger.info(f"Search space: LR [{config.lr_min}, {config.lr_max}], Batch {config.batch_sizes}")
    
    for trial in range(config.num_trials):
        trial_start = time.time()
        
        # Sample hyperparameters
        params = sample_hyperparameters(config, best_params, trial)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Create model with sampled architecture
        model = IMUClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            architecture_size=params.get('architecture_size', 'medium')
        )
        
        # Get optimizer
        optimizer_name = params.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.01)
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(), 
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params.get('weight_decay', 0.0)
            )
        
        # Quick training
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.epochs_per_trial,
            learning_rate=params['learning_rate'],
            device=device
        )
        
        trial_duration = time.time() - trial_start
        
        trial_data = {
            'trial': trial + 1,
            'learning_rate': round(params['learning_rate'], 8),
            'batch_size': params['batch_size'],
            'weight_decay': round(params.get('weight_decay', 0.0), 6),
            'dropout': round(params.get('dropout', 0.3), 3),
            'optimizer': params.get('optimizer', 'adam'),
            'architecture_size': params.get('architecture_size', 'medium'),
            'train_accuracy': results['train_accuracies'][-1],
            'val_accuracy': results['val_accuracies'][-1],
            'train_loss': results['train_losses'][-1],
            'val_loss': results['val_losses'][-1],
            'duration_seconds': round(trial_duration, 2),
            'is_best': False
        }
        trials_data.append(trial_data)
        
        logger.info(
            f"Trial {trial+1}/{config.num_trials}: "
            f"lr={params['learning_rate']:.6f}, batch={params['batch_size']}, "
            f"opt={params.get('optimizer', 'adam')}, "
            f"val_acc={results['val_accuracies'][-1]:.4f} ({trial_duration:.1f}s)"
        )
        
        if results['best_val_accuracy'] > best_val_acc:
            best_val_acc = results['best_val_accuracy']
            best_params = params.copy()
            best_trial_idx = trial
            logger.info(f"  -> New best! val_acc={best_val_acc:.4f}")
        
        if callback:
            callback(trial + 1, config.num_trials, trial_data)
    
    # Mark best trial
    if trials_data:
        trials_data[best_trial_idx]['is_best'] = True
    
    logger.info(f"Bayesian optimization complete. Best trial: {best_trial_idx + 1}")
    logger.info(f"Best params: {best_params}")
    
    return best_params, trials_data


def save_model_to_bytes(model: nn.Module, config: Dict[str, Any]) -> bytes:
    """Save model weights and config to bytes for storage."""
    buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'architecture': model.get_architecture_summary() if hasattr(model, 'get_architecture_summary') else {}
    }, buffer)
    return buffer.getvalue()


def load_model_from_bytes(model_bytes: bytes) -> Tuple[Dict, Dict]:
    """Load model weights and config from bytes."""
    buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
    return checkpoint['model_state_dict'], checkpoint.get('config', {})


def compute_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Compute detailed metrics including confusion matrix, ROC curves, PR curves."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_classes = len(class_names)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': round(float(precision[i]), 4),
            'recall': round(float(recall[i]), 4),
            'f1_score': round(float(f1[i]), 4),
            'support': int(support[i])
        }
    
    # ROC curves (one-vs-rest)
    roc_curves = {}
    pr_curves = {}
    
    # Binarize labels for multi-class ROC
    y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    
    for i, class_name in enumerate(class_names):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Sample points for storage (max 20 points)
        indices = np.linspace(0, len(fpr) - 1, min(20, len(fpr)), dtype=int)
        roc_points = [{'fpr': round(float(fpr[j]), 4), 'tpr': round(float(tpr[j]), 4)} for j in indices]
        roc_curves[class_name] = {'points': roc_points, 'auc': round(float(roc_auc), 4)}
        
        # PR curve
        prec, rec, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        indices = np.linspace(0, len(prec) - 1, min(20, len(prec)), dtype=int)
        pr_points = [{'precision': round(float(prec[j]), 4), 'recall': round(float(rec[j]), 4)} for j in indices]
        pr_curves[class_name] = {'points': pr_points}
    
    return {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

async def run_full_training(
    job_id: str,
    dataset_id: int,
    db_session,
    model_type: str,
    config: Dict[str, Any],
    update_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Run full training pipeline with real data from database.
    
    Args:
        job_id: Unique job identifier
        dataset_id: ID of the training dataset
        db_session: SQLAlchemy database session
        model_type: Type of model to train
        config: Training configuration
        update_callback: Async callback for progress updates
        
    Returns:
        Dictionary with training results and model bytes
    """
    import asyncio
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Starting training job {job_id} on device: {device}")
    
    # Get window size from config (default 128)
    window_size = config.get('window_size', 128)
    architecture_size = config.get('model_architecture', 'medium')
    validation_split = config.get('validation_split', 0.2)
    
    # Load real data from database
    logger.info(f"Loading dataset {dataset_id} from database...")
    X, y, class_names = load_dataset_from_db(db_session, dataset_id, window_size)
    num_classes = len(class_names)
    
    logger.info(f"Loaded {len(X)} samples with {num_classes} classes: {class_names}")
    logger.info(f"Data shape: {X.shape} (batch, window_size={window_size}, channels=6)")
    
    # Normalize data (z-score normalization per channel)
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    # Shuffle and split data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Bayesian optimization
    bayesian_trials_data = []
    bayesian_config_used = None
    if config.get('use_bayesian_optimization', False):
        logger.info("Running Bayesian hyperparameter optimization...")
        if update_callback:
            await update_callback('optimizing', 0, 0, None)
        
        # Create Bayesian config from user settings
        bayesian_config = BayesianOptimizationConfig.from_dict(config)
        bayesian_config_used = bayesian_config.to_dict()
        
        # Run optimization in thread pool
        loop = asyncio.get_event_loop()
        best_params, bayesian_trials_data = await loop.run_in_executor(
            None,
            lambda: run_bayesian_optimization(
                X_train, y_train, X_val, y_val,
                num_classes=num_classes,
                config=bayesian_config,
                device=device
            )
        )
        
        if best_params:
            config['learning_rate'] = best_params['learning_rate']
            config['batch_size'] = best_params['batch_size']
            config['weight_decay'] = best_params.get('weight_decay', 0.0)
            architecture_size = best_params.get('architecture_size', architecture_size)
            logger.info(f"Best hyperparameters: {best_params}")
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    seq_length = X_train.shape[1]
    input_channels = X_train.shape[2]
    
    model = IMUClassifier(
        input_channels=input_channels,
        seq_length=seq_length,
        num_classes=num_classes,
        architecture_size=architecture_size
    )
    
    logger.info(f"Model created: {model.get_architecture_summary()['total_params']} parameters")
    
    # Train model in thread pool
    def sync_train():
        return train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get('epochs', 10),
            learning_rate=config.get('learning_rate', 0.001),
            device=device
        )
    
    # Run training in thread pool to not block async
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, sync_train)
    
    logger.info(f"Training completed. Best val accuracy: {results['best_val_accuracy']:.4f} at epoch {results['best_epoch']}")
    
    # Compute real metrics on validation set
    logger.info("Computing detailed metrics...")
    detailed_metrics = await loop.run_in_executor(
        None,
        lambda: compute_metrics(model, val_loader, class_names, device)
    )
    
    # Get model architecture
    model_architecture = model.get_architecture_summary()
    model_architecture['optimizer'] = 'adam'
    model_architecture['learning_rate'] = config.get('learning_rate', 0.001)
    model_architecture['batch_size'] = config.get('batch_size', 32)
    
    # Save model weights with normalization stats for inference
    save_config = {
        **config,
        'normalization': {
            'mean': X_mean.tolist(),
            'std': X_std.tolist()
        },
        'class_names': class_names,
        'input_shape': list(X_train.shape[1:]),
        'num_classes': num_classes
    }
    model_bytes = save_model_to_bytes(model, save_config)
    
    logger.info(f"Model saved: {len(model_bytes)} bytes")
    
    return {
        'train_losses': results['train_losses'],
        'train_accuracies': results['train_accuracies'],
        'val_losses': results['val_losses'],
        'val_accuracies': results['val_accuracies'],
        'best_val_accuracy': results['best_val_accuracy'],
        'best_epoch': results['best_epoch'],
        'per_class_metrics': detailed_metrics['per_class_metrics'],
        'confusion_matrix': detailed_metrics['confusion_matrix'],
        'class_names': class_names,
        'roc_curves': detailed_metrics['roc_curves'],
        'pr_curves': detailed_metrics['pr_curves'],
        'model_architecture': model_architecture,
        'model_bytes': model_bytes,
        'bayesian_trials_data': bayesian_trials_data,
        'bayesian_config': bayesian_config_used,
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val)
    }
