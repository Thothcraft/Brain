"""MLP Models for 1D Input (batch, features).

Multi-Layer Perceptron models in nano, mini, and max variants.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any

from .base import (
    BaseDLModel,
    ModelConfig,
    ModelSize,
    InputShape,
    register_model,
)


@register_model
class MLPNanoModel(BaseDLModel):
    """Minimal MLP for 1D features - edge deployment."""
    
    model_type_id = "mlp_nano"
    model_name = "MLP Nano"
    model_description = "Minimal MLP (2 layers, ~5K params) for edge deployment"
    architecture = "mlp"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_1D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.net = nn.Sequential(
            nn.Linear(config.input_features, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model
class MLPMiniModel(BaseDLModel):
    """Balanced MLP for 1D features."""
    
    model_type_id = "mlp_mini"
    model_name = "MLP Mini"
    model_description = "Balanced MLP (3 layers, ~50K params)"
    architecture = "mlp"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_1D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.net = nn.Sequential(
            nn.Linear(config.input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model
class MLPMaxModel(BaseDLModel):
    """Maximum capacity MLP for 1D features."""
    
    model_type_id = "mlp_max"
    model_name = "MLP Max"
    model_description = "Maximum capacity MLP (5 layers, ~500K params)"
    architecture = "mlp"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_1D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.net = nn.Sequential(
            nn.Linear(config.input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
