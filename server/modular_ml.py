"""Modular Machine Learning and Federated Learning Architecture.

This module provides a modular, extensible architecture for:
- ML algorithms (CNN, LSTM, GRU, Transformer, etc.)
- FL algorithms (FedAvg, FedProx, FedAdam, etc.)
- Preprocessing blocks (normalization, windowing, filtering, etc.)

Easy extensibility through registry pattern and base classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# ALGORITHM REGISTRY PATTERN
# ============================================================================

class AlgorithmRegistry:
    """Generic registry for algorithms and components."""
    
    _registries: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, category: str, name: str, component: Any, metadata: Optional[Dict] = None):
        """Register a component in a category."""
        if category not in cls._registries:
            cls._registries[category] = {}
        
        cls._registries[category][name] = {
            "component": component,
            "metadata": metadata or {},
        }
        logger.debug(f"Registered {category}/{name}")
    
    @classmethod
    def get(cls, category: str, name: str) -> Optional[Any]:
        """Get a component by category and name."""
        if category not in cls._registries:
            return None
        return cls._registries[category].get(name, {}).get("component")
    
    @classmethod
    def list_category(cls, category: str) -> List[Dict[str, Any]]:
        """List all components in a category with metadata."""
        if category not in cls._registries:
            return []
        
        return [
            {"name": name, **info["metadata"]}
            for name, info in cls._registries[category].items()
        ]
    
    @classmethod
    def get_metadata(cls, category: str, name: str) -> Optional[Dict]:
        """Get metadata for a component."""
        if category not in cls._registries:
            return None
        return cls._registries[category].get(name, {}).get("metadata")


# ============================================================================
# ML MODEL ARCHITECTURES
# ============================================================================

class ModelType(str, Enum):
    """Supported model architectures."""
    # Deep Learning
    CNN_1D = "cnn_1d"
    CNN_2D = "cnn_2d"
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    TCN = "tcn"
    RESNET = "resnet"
    
    # Classical ML
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"
    ADABOOST = "adaboost"
    DECISION_TREE = "decision_tree"
    GRADIENT_BOOSTING = "gradient_boosting"


@dataclass
class ModelConfig:
    """Configuration for model creation."""
    model_type: ModelType
    input_channels: int = 6
    seq_length: int = 128
    num_classes: int = 2
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    architecture_size: str = "medium"  # small, medium, large
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseModel(nn.Module, ABC):
    """Base class for all ML models."""
    
    model_type: ModelType = None
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type.value if self.model_type else "unknown",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_channels": self.config.input_channels,
            "seq_length": self.config.seq_length,
            "num_classes": self.config.num_classes,
        }


class CNN1DModel(BaseModel):
    """1D CNN for time-series classification."""
    
    model_type = ModelType.CNN_1D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Architecture based on size
        sizes = {
            "small": [32, 64],
            "medium": [64, 128, 128],
            "large": [64, 128, 256, 256],
        }
        channels = sizes.get(config.architecture_size, sizes["medium"])
        
        layers = []
        in_ch = config.input_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(config.dropout),
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if x.dim() == 3 and x.size(2) == self.config.input_channels:
            x = x.permute(0, 2, 1)
        
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LSTMModel(BaseModel):
    """Bidirectional LSTM for sequence classification."""
    
    model_type = ModelType.LSTM
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.lstm = nn.LSTM(
            input_size=config.input_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        out = lstm_out[:, -1, :]
        return self.fc(out)


class GRUModel(BaseModel):
    """Bidirectional GRU for sequence classification."""
    
    model_type = ModelType.GRU
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.gru = nn.GRU(
            input_size=config.input_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.fc(out)


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid for local feature extraction + temporal modeling."""
    
    model_type = ModelType.CNN_LSTM
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # CNN for local features
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


class TransformerModel(BaseModel):
    """Transformer encoder for sequence classification."""
    
    model_type = ModelType.TRANSFORMER
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.embedding = nn.Linear(config.input_channels, config.hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.seq_length, config.hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.fc(x)


class TCNModel(BaseModel):
    """Temporal Convolutional Network."""
    
    model_type = ModelType.TCN
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        channels = [64, 128, 256]
        layers = []
        in_ch = config.input_channels
        
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(config.dropout),
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# Register models
AlgorithmRegistry.register("models", "cnn_1d", CNN1DModel, {
    "name": "1D CNN",
    "description": "1D Convolutional Neural Network for time-series",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})

AlgorithmRegistry.register("models", "lstm", LSTMModel, {
    "name": "Bidirectional LSTM",
    "description": "Long Short-Term Memory network for sequences",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})

AlgorithmRegistry.register("models", "gru", GRUModel, {
    "name": "Bidirectional GRU",
    "description": "Gated Recurrent Unit network for sequences",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})

AlgorithmRegistry.register("models", "cnn_lstm", CNNLSTMModel, {
    "name": "CNN-LSTM Hybrid",
    "description": "CNN for local features + LSTM for temporal modeling",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})

AlgorithmRegistry.register("models", "transformer", TransformerModel, {
    "name": "Transformer Encoder",
    "description": "Transformer with self-attention for sequences",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})

AlgorithmRegistry.register("models", "tcn", TCNModel, {
    "name": "Temporal Convolutional Network",
    "description": "Dilated causal convolutions for sequences",
    "data_types": ["csi", "imu", "time_series"],
    "input_format": "sequence",
})


# ============================================================================
# PREPROCESSING BLOCKS
# ============================================================================

class PreprocessingBlockType(str, Enum):
    """Types of preprocessing blocks."""
    # Data Loading
    CSI_LOADER = "csi_loader"
    IMU_LOADER = "imu_loader"
    CSV_LOADER = "csv_loader"
    
    # Feature Extraction
    AMPLITUDE_EXTRACTOR = "amplitude_extractor"
    PHASE_EXTRACTOR = "phase_extractor"
    FEATURE_CONCAT = "feature_concat"
    
    # Filtering
    SUBCARRIER_FILTER = "subcarrier_filter"
    LOWPASS_FILTER = "lowpass_filter"
    HIGHPASS_FILTER = "highpass_filter"
    BANDPASS_FILTER = "bandpass_filter"
    MOVING_AVERAGE = "moving_average"
    MEDIAN_FILTER = "median_filter"
    
    # Normalization
    ZSCORE_NORMALIZE = "zscore_normalize"
    MINMAX_NORMALIZE = "minmax_normalize"
    ROBUST_NORMALIZE = "robust_normalize"
    
    # Windowing
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    
    # Dimensionality
    PCA = "pca"
    FLATTEN = "flatten"
    RESHAPE = "reshape"
    
    # Augmentation
    NOISE_INJECTION = "noise_injection"
    TIME_WARP = "time_warp"
    MAGNITUDE_WARP = "magnitude_warp"


@dataclass
class PreprocessingBlockConfig:
    """Configuration for a preprocessing block."""
    block_type: PreprocessingBlockType
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Expected input/output shapes
    input_shape: Optional[str] = None  # "2d", "3d", "any"
    output_shape: Optional[str] = None


class BasePreprocessingBlock(ABC):
    """Base class for preprocessing blocks."""
    
    block_type: PreprocessingBlockType = None
    input_shape: str = "any"
    output_shape: str = "any"
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process data and return (output, metadata)."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get block information."""
        return {
            "type": self.block_type.value if self.block_type else "unknown",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "config": self.config,
        }


class ZScoreNormalizeBlock(BasePreprocessingBlock):
    """Z-score normalization: (x - mean) / std."""
    
    block_type = PreprocessingBlockType.ZSCORE_NORMALIZE
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        normalized = (data - mean) / std
        
        return normalized.astype(np.float32), {
            "mean": mean.flatten().tolist()[:10],  # Sample
            "std": std.flatten().tolist()[:10],
        }


class MinMaxNormalizeBlock(BasePreprocessingBlock):
    """Min-max normalization to [0, 1]."""
    
    block_type = PreprocessingBlockType.MINMAX_NORMALIZE
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        min_val = data.min(axis=0, keepdims=True)
        max_val = data.max(axis=0, keepdims=True)
        range_val = max_val - min_val + 1e-8
        normalized = (data - min_val) / range_val
        
        return normalized.astype(np.float32), {
            "min": min_val.flatten().tolist()[:10],
            "max": max_val.flatten().tolist()[:10],
        }


class MovingAverageBlock(BasePreprocessingBlock):
    """Moving average smoothing."""
    
    block_type = PreprocessingBlockType.MOVING_AVERAGE
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        window = self.config.get("window", 5)
        if window < 2:
            return data, {"skipped": True}
        
        kernel = np.ones(window) / window
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'),
            axis=0, arr=data
        )
        
        return smoothed.astype(np.float32), {"window": window}


class SlidingWindowBlock(BasePreprocessingBlock):
    """Create sliding windows from time-series."""
    
    block_type = PreprocessingBlockType.SLIDING_WINDOW
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        window_size = self.config.get("window_size", 128)
        stride = self.config.get("stride", window_size // 2)
        
        if len(data) < window_size:
            return np.array([]), {"error": f"Data length {len(data)} < window_size {window_size}"}
        
        windows = []
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start:start + window_size])
        
        if not windows:
            return np.array([]), {"error": "No windows created"}
        
        result = np.stack(windows, axis=0)
        return result.astype(np.float32), {
            "num_windows": len(windows),
            "window_size": window_size,
            "stride": stride,
        }


class FlattenBlock(BasePreprocessingBlock):
    """Flatten multi-dimensional data."""
    
    block_type = PreprocessingBlockType.FLATTEN
    output_shape = "2d"
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if data.ndim <= 2:
            return data, {"already_flat": True}
        
        # Flatten all but first dimension
        original_shape = data.shape
        flattened = data.reshape(data.shape[0], -1)
        
        return flattened.astype(np.float32), {
            "original_shape": original_shape,
            "new_shape": flattened.shape,
        }


class NoiseInjectionBlock(BasePreprocessingBlock):
    """Add Gaussian noise for data augmentation."""
    
    block_type = PreprocessingBlockType.NOISE_INJECTION
    
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        noise_level = self.config.get("noise_level", 0.01)
        noise = np.random.randn(*data.shape) * noise_level * data.std()
        augmented = data + noise
        
        return augmented.astype(np.float32), {"noise_level": noise_level}


# Register preprocessing blocks
AlgorithmRegistry.register("preprocessing", "zscore_normalize", ZScoreNormalizeBlock, {
    "name": "Z-Score Normalization",
    "description": "Normalize to zero mean and unit variance",
    "input_shape": "any",
    "output_shape": "same",
    "params": [],
})

AlgorithmRegistry.register("preprocessing", "minmax_normalize", MinMaxNormalizeBlock, {
    "name": "Min-Max Normalization",
    "description": "Scale to [0, 1] range",
    "input_shape": "any",
    "output_shape": "same",
    "params": [],
})

AlgorithmRegistry.register("preprocessing", "moving_average", MovingAverageBlock, {
    "name": "Moving Average",
    "description": "Smooth data with moving average filter",
    "input_shape": "any",
    "output_shape": "same",
    "params": [{"name": "window", "type": "int", "default": 5}],
})

AlgorithmRegistry.register("preprocessing", "sliding_window", SlidingWindowBlock, {
    "name": "Sliding Window",
    "description": "Create overlapping windows from time-series",
    "input_shape": "2d",
    "output_shape": "3d",
    "params": [
        {"name": "window_size", "type": "int", "default": 128},
        {"name": "stride", "type": "int", "default": 64},
    ],
})

AlgorithmRegistry.register("preprocessing", "flatten", FlattenBlock, {
    "name": "Flatten",
    "description": "Flatten multi-dimensional data",
    "input_shape": "any",
    "output_shape": "2d",
    "params": [],
})

AlgorithmRegistry.register("preprocessing", "noise_injection", NoiseInjectionBlock, {
    "name": "Noise Injection",
    "description": "Add Gaussian noise for augmentation",
    "input_shape": "any",
    "output_shape": "same",
    "params": [{"name": "noise_level", "type": "float", "default": 0.01}],
})


# ============================================================================
# FL ALGORITHM CONFIGURATIONS
# ============================================================================

class FLAlgorithmType(str, Enum):
    """Federated Learning algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    FEDADAGRAD = "fedadagrad"
    FEDAVGM = "fedavgm"
    FEDMEDIAN = "fedmedian"
    FEDTRIMMEDAVG = "fedtrimmedavg"
    KRUM = "krum"
    BULYAN = "bulyan"
    QFEDAVG = "qfedavg"
    DPFEDAVG_ADAPTIVE = "dpfedavg_adaptive"
    DPFEDAVG_FIXED = "dpfedavg_fixed"


@dataclass
class FLAlgorithmConfig:
    """Configuration for FL algorithm."""
    algorithm: FLAlgorithmType
    
    # Common parameters
    num_rounds: int = 100
    min_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 0.5
    
    # Algorithm-specific parameters
    proximal_mu: float = 0.01  # FedProx
    server_lr: float = 1.0  # FedAdam/FedYogi
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau: float = 1e-3
    
    # Privacy parameters
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Byzantine robustness
    trimmed_mean_beta: float = 0.1
    krum_num_closest: int = 2


# Register FL algorithms with metadata
FL_ALGORITHMS = [
    ("fedavg", {
        "name": "Federated Averaging",
        "description": "Standard FedAvg algorithm",
        "params": [],
        "reference": "McMahan et al., 2017",
    }),
    ("fedprox", {
        "name": "FedProx",
        "description": "FedAvg with proximal term for non-IID data",
        "params": [{"name": "proximal_mu", "type": "float", "default": 0.01}],
        "reference": "Li et al., 2020",
    }),
    ("fedadam", {
        "name": "FedAdam",
        "description": "Adaptive FL with Adam optimizer on server",
        "params": [
            {"name": "server_lr", "type": "float", "default": 1.0},
            {"name": "beta_1", "type": "float", "default": 0.9},
            {"name": "beta_2", "type": "float", "default": 0.99},
        ],
        "reference": "Reddi et al., 2021",
    }),
    ("fedyogi", {
        "name": "FedYogi",
        "description": "Adaptive FL with controlled adaptivity",
        "params": [
            {"name": "server_lr", "type": "float", "default": 1.0},
            {"name": "tau", "type": "float", "default": 1e-3},
        ],
        "reference": "Reddi et al., 2021",
    }),
    ("fedmedian", {
        "name": "FedMedian",
        "description": "Byzantine-robust median aggregation",
        "params": [],
        "reference": "Yin et al., 2018",
    }),
    ("fedtrimmedavg", {
        "name": "FedTrimmedAvg",
        "description": "Byzantine-robust trimmed mean aggregation",
        "params": [{"name": "beta", "type": "float", "default": 0.1}],
        "reference": "Yin et al., 2018",
    }),
    ("krum", {
        "name": "Krum",
        "description": "Byzantine-robust Krum aggregation",
        "params": [{"name": "num_closest", "type": "int", "default": 2}],
        "reference": "Blanchard et al., 2017",
    }),
    ("qfedavg", {
        "name": "q-FedAvg",
        "description": "Fair federated learning with q-fair objective",
        "params": [{"name": "q", "type": "float", "default": 0.2}],
        "reference": "Li et al., 2020",
    }),
    ("dpfedavg_adaptive", {
        "name": "DP-FedAvg (Adaptive)",
        "description": "Differential privacy with adaptive clipping",
        "params": [
            {"name": "noise_multiplier", "type": "float", "default": 1.0},
            {"name": "max_grad_norm", "type": "float", "default": 1.0},
        ],
        "reference": "Andrew et al., 2021",
    }),
]

for algo_name, metadata in FL_ALGORITHMS:
    AlgorithmRegistry.register("fl_algorithms", algo_name, FLAlgorithmConfig, metadata)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(config: ModelConfig) -> BaseModel:
    """Create a model from configuration."""
    model_class = AlgorithmRegistry.get("models", config.model_type.value)
    
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model_class(config)


def list_available_models() -> List[Dict[str, Any]]:
    """List all available model architectures."""
    return AlgorithmRegistry.list_category("models")


def list_available_preprocessing() -> List[Dict[str, Any]]:
    """List all available preprocessing blocks."""
    return AlgorithmRegistry.list_category("preprocessing")


def list_available_fl_algorithms() -> List[Dict[str, Any]]:
    """List all available FL algorithms."""
    return AlgorithmRegistry.list_category("fl_algorithms")


# ============================================================================
# PREPROCESSING PIPELINE EXECUTOR
# ============================================================================

class PreprocessingPipeline:
    """Execute a sequence of preprocessing blocks."""
    
    def __init__(self, blocks: List[Dict[str, Any]]):
        """Initialize with list of block configurations.
        
        Each block: {"type": "block_type", "enabled": True, "params": {...}}
        """
        self.blocks = blocks
        self.execution_log = []
    
    def execute(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute pipeline and return (processed_data, metadata)."""
        self.execution_log = []
        current_data = data
        
        for block_config in self.blocks:
            if not block_config.get("enabled", True):
                continue
            
            block_type = block_config.get("type", "")
            block_class = AlgorithmRegistry.get("preprocessing", block_type)
            
            if block_class is None:
                logger.warning(f"Unknown preprocessing block: {block_type}")
                continue
            
            try:
                block = block_class(block_config.get("params", {}))
                current_data, block_meta = block.process(current_data)
                
                self.execution_log.append({
                    "block": block_type,
                    "input_shape": list(data.shape) if hasattr(data, 'shape') else None,
                    "output_shape": list(current_data.shape) if hasattr(current_data, 'shape') else None,
                    "metadata": block_meta,
                })
                
            except Exception as e:
                logger.error(f"Error in preprocessing block {block_type}: {e}")
                self.execution_log.append({
                    "block": block_type,
                    "error": str(e),
                })
        
        return current_data, {
            "blocks_executed": len(self.execution_log),
            "execution_log": self.execution_log,
            "final_shape": list(current_data.shape) if hasattr(current_data, 'shape') else None,
        }
