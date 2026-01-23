"""Model architectures for Federated Learning.

This module provides PyTorch model architectures that can be used
in federated learning experiments. 

IMPORTANT: This module integrates with dl_models/ for model architectures.
All models available for FL are sourced from the DLModelRegistry to ensure
consistency across ML/DL and FL training pipelines.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from dl_models for unified model access
from ...dl_models import (
    DLModelRegistry,
    BaseDLModel,
    ModelConfig as DLModelConfig,
    ModelSize,
    InputShape,
)

logger = logging.getLogger(__name__)


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    CNN = "cnn"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    MOBILENET_V2 = "mobilenet_v2"
    VGG16 = "vgg16"
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TCN = "tcn"
    MLP = "mlp"
    LOGISTIC = "logistic"
    CUSTOM = "custom"


# ============================================================================
# CNN Models
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10/MNIST-like datasets."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding before final layer (for knowledge distillation)."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        return F.relu(self.fc1(x))


class ResNetBlock(nn.Module):
    """Basic ResNet block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 for image classification."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding before final layer."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class MobileNetV2Block(nn.Module):
    """Inverted residual block for MobileNetV2."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 for efficient inference."""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        
        # Configuration: [expand_ratio, out_channels, num_blocks, stride]
        config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )
        
        layers = []
        for t, c, n, s in config:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MobileNetV2Block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        self.layers = nn.Sequential(*layers)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# MLP Models
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for tabular/flattened data."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.flatten = nn.Flatten()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)


class LogisticRegression(nn.Module):
    """Simple logistic regression."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)


# ============================================================================
# Sequence Models (LSTM, GRU, TCN)
# ============================================================================

class LSTMClassifier(nn.Module):
    """LSTM for time-series classification."""
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128,
                 hidden_size: int = 128, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


class GRUClassifier(nn.Module):
    """GRU for time-series classification."""
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128,
                 hidden_size: int = 128, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM hybrid for time-series with local feature extraction."""
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128,
                 hidden_size: int = 64, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


class TCN(nn.Module):
    """Temporal Convolutional Network for time-series."""
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128,
                 num_channels: List[int] = None, num_classes: int = 10):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256]
        
        layers = []
        in_ch = input_channels
        for out_ch in num_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
            ])
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry for FL models."""
    
    _models: Dict[str, Type[nn.Module]] = {
        "cnn": SimpleCNN,
        "resnet18": ResNet18,
        "mobilenet_v2": MobileNetV2,
        "mlp": MLP,
        "logistic": LogisticRegression,
        "lstm": LSTMClassifier,
        "gru": GRUClassifier,
        "cnn_lstm": CNNLSTMClassifier,
        "tcn": TCN,
    }
    
    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]):
        """Register a new model."""
        cls._models[name.lower()] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[nn.Module]]:
        """Get a model class by name."""
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._models.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[nn.Module]:
        """Create a model instance."""
        model_class = cls.get(name)
        if model_class:
            return model_class(**kwargs)
        return None


def get_model(
    architecture: str,
    num_classes: int = 10,
    in_channels: int = 3,
    input_dim: int = 784,
    seq_length: int = 128,
    **kwargs
) -> nn.Module:
    """Factory function to create models.
    
    Args:
        architecture: Model architecture name (cnn, resnet18, mlp, lstm, etc.)
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        input_dim: Input dimension for MLP/logistic models
        seq_length: Sequence length for time-series models
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model instance
    """
    arch = architecture.lower() if isinstance(architecture, str) else architecture.value.lower()
    
    # Image models
    if arch in ["cnn", "simplecnn"]:
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    elif arch == "resnet18":
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif arch == "mobilenet_v2":
        return MobileNetV2(num_classes=num_classes, in_channels=in_channels)
    
    # MLP models
    elif arch == "mlp":
        return MLP(input_dim=input_dim, num_classes=num_classes)
    elif arch == "logistic":
        return LogisticRegression(input_dim=input_dim, num_classes=num_classes)
    
    # Sequence models
    elif arch == "lstm":
        return LSTMClassifier(
            input_channels=in_channels,
            seq_length=seq_length,
            num_classes=num_classes
        )
    elif arch == "gru":
        return GRUClassifier(
            input_channels=in_channels,
            seq_length=seq_length,
            num_classes=num_classes
        )
    elif arch == "cnn_lstm":
        return CNNLSTMClassifier(
            input_channels=in_channels,
            seq_length=seq_length,
            num_classes=num_classes
        )
    elif arch == "tcn":
        return TCN(
            input_channels=in_channels,
            seq_length=seq_length,
            num_classes=num_classes
        )
    
    # Try FL ModelRegistry
    model = ModelRegistry.create(arch, num_classes=num_classes, in_channels=in_channels, **kwargs)
    if model:
        return model
    
    # Try DLModelRegistry from dl_models/ for unified access
    try:
        dl_model = DLModelRegistry.create(arch, {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "input_channels": in_channels,
            "seq_length": seq_length,
            **kwargs
        })
        if dl_model:
            logger.info(f"Created model '{arch}' from DLModelRegistry")
            return dl_model
    except Exception as e:
        logger.debug(f"Could not create model from DLModelRegistry: {e}")
    
    # Default to CNN
    logger.warning(f"Unknown architecture '{arch}', defaulting to SimpleCNN")
    return SimpleCNN(num_classes=num_classes, in_channels=in_channels)


def get_available_models() -> Dict[str, List[str]]:
    """Get all available models from both FL and DL registries.
    
    Returns:
        Dictionary with 'fl_models' and 'dl_models' lists
    """
    fl_models = ModelRegistry.list_models()
    
    try:
        dl_models = [m["type"] for m in DLModelRegistry.list_models()]
    except Exception:
        dl_models = []
    
    return {
        "fl_models": fl_models,
        "dl_models": dl_models,
        "all": list(set(fl_models + dl_models))
    }


def get_models_by_input_shape(input_shape: str) -> List[Dict[str, Any]]:
    """Get models compatible with a specific input shape.
    
    Args:
        input_shape: One of '1d', '2d', '3d', '4d'
            - 1d: (batch, features) - MLP
            - 2d: (batch, seq_len, features) - LSTM, GRU, CNN1D, Transformer
            - 3d: (batch, channels, height, width) - CNN2D, ResNet
            - 4d: (batch, num_frames, channels, height, width) - CNN3D
    
    Returns:
        List of model info dictionaries
    """
    try:
        return DLModelRegistry.list_by_shape(input_shape)
    except Exception:
        return []
