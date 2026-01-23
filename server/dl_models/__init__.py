"""Deep Learning Models Module.

This module provides PyTorch-based DL models for different data shapes.
Each model has nano, mini, and max variants for different complexity needs.

Naming Convention:
- File: model_{architecture}_{shape}.py (e.g., model_mlp_1d.py, model_lstm_2d.py)
- Class: {Architecture}{Size}Model (e.g., MLPNanoModel, LSTMMiniModel)

Shape Conventions:
- 1D (batch, features): MLP - Flattened feature vectors
- 2D (batch, seq_len, features): LSTM, GRU, CNN1D, Transformer - Sequential/time-series
- 3D (batch, channels, height, width): CNN2D, ResNet - Single images
- 4D (batch, num_frames, channels, height, width): CNN3D - Video/volumetric data

Model Sizes:
- nano: Minimal parameters, fast inference, edge deployment
- mini: Balanced parameters, good accuracy
- max: Maximum parameters, best accuracy, requires more compute

Usage:
    from server.dl_models import DLModelRegistry, create_dl_model
    
    # List models for a specific shape
    models = DLModelRegistry.list_by_shape("2d")  # Sequential models
    
    # Create model
    model = create_dl_model("lstm_mini", {
        "input_channels": 64,
        "seq_length": 128,
        "num_classes": 4,
    })
"""

import torch
import torch.nn as nn

from .base import (
    BaseDLModel,
    DLModelRegistry,
    ModelSize,
    InputShape,
    ModelConfig,
)

# Import all models to register them
from . import model_mlp_1d
from . import model_lstm_3d
from . import model_gru_3d
from . import model_cnn1d_3d
from . import model_transformer_3d
from . import model_cnn2d_4d
from . import model_resnet_4d
from . import model_cnn3d_4d

__all__ = [
    "BaseDLModel",
    "DLModelRegistry",
    "ModelSize",
    "InputShape",
    "ModelConfig",
    "create_dl_model",
]

def create_dl_model(model_type: str, config: dict = None) -> nn.Module:
    """Convenience function to create a DL model."""
    return DLModelRegistry.create(model_type, config)
