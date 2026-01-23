"""LSTM Models for 2D Input (batch, seq_len, features).

Bidirectional LSTM models in nano, mini, and max variants.
Input shape: (batch, seq_len, features) - Sequential/time-series data
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
class LSTMNanoModel(BaseDLModel):
    """Minimal LSTM for sequential data - edge deployment."""
    
    model_type_id = "lstm_nano"
    model_name = "LSTM Nano"
    model_description = "Minimal LSTM (1 layer, ~10K params) for edge deployment"
    architecture = "lstm"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.lstm = nn.LSTM(
            input_size=config.input_channels,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(32, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        return self.fc(out)


@register_model
class LSTMMiniModel(BaseDLModel):
    """Balanced bidirectional LSTM for sequential data."""
    
    model_type_id = "lstm_mini"
    model_name = "LSTM Mini"
    model_description = "Balanced BiLSTM (2 layers, ~100K params)"
    architecture = "lstm"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.lstm = nn.LSTM(
            input_size=config.input_channels,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 for bidirectional
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


@register_model
class LSTMMaxModel(BaseDLModel):
    """Maximum capacity bidirectional LSTM."""
    
    model_type_id = "lstm_max"
    model_name = "LSTM Max"
    model_description = "Maximum capacity BiLSTM (3 layers, ~1M params)"
    architecture = "lstm"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.lstm = nn.LSTM(
            input_size=config.input_channels,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Self-attention over sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Global average pooling
        out = attn_out.mean(dim=1)
        return self.fc(out)
