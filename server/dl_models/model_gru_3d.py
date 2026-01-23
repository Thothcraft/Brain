"""GRU Models for 3D Input (batch, seq_len, features).

Bidirectional GRU models in nano, mini, and max variants.
"""

import torch
import torch.nn as nn

from .base import (
    BaseDLModel,
    ModelConfig,
    ModelSize,
    InputShape,
    register_model,
)


@register_model
class GRUNanoModel(BaseDLModel):
    """Minimal GRU for sequential data."""
    
    model_type_id = "gru_nano"
    model_name = "GRU Nano"
    model_description = "Minimal GRU (1 layer, ~8K params) for edge deployment"
    architecture = "gru"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.gru = nn.GRU(config.input_channels, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


@register_model
class GRUMiniModel(BaseDLModel):
    """Balanced bidirectional GRU."""
    
    model_type_id = "gru_mini"
    model_name = "GRU Mini"
    model_description = "Balanced BiGRU (2 layers, ~80K params)"
    architecture = "gru"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.gru = nn.GRU(config.input_channels, 64, 2, batch_first=True, dropout=config.dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


@register_model
class GRUMaxModel(BaseDLModel):
    """Maximum capacity bidirectional GRU."""
    
    model_type_id = "gru_max"
    model_name = "GRU Max"
    model_description = "Maximum capacity BiGRU (3 layers, ~800K params)"
    architecture = "gru"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.gru = nn.GRU(config.input_channels, 256, 3, batch_first=True, dropout=config.dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
