"""Transformer Models for 2D Input (batch, seq_len, features).

Transformer encoder models in nano, mini, and max variants.
Input shape: (batch, seq_len, features) - Sequential/time-series data

Verification References:
- Original Transformer Paper: https://arxiv.org/abs/1706.03762 (Vaswani et al., 2017 - "Attention Is All You Need")
- PyTorch TransformerEncoder: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- PyTorch TransformerEncoderLayer: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- Positional Encoding: Sinusoidal encoding from original paper (Section 3.5)
- Parameters verified: d_model, nhead, dim_feedforward, dropout, num_layers, batch_first
- Note: batch_first=True for PyTorch >= 1.9 compatibility
"""

import torch
import torch.nn as nn
import math

from .base import (
    BaseDLModel,
    ModelConfig,
    ModelSize,
    InputShape,
    register_model,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


@register_model
class TransformerNanoModel(BaseDLModel):
    """Minimal Transformer for sequential data."""
    
    model_type_id = "transformer_nano"
    model_name = "Transformer Nano"
    model_description = "Minimal Transformer (1 layer, 2 heads, ~20K params)"
    architecture = "transformer"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        d_model = 32
        self.embedding = nn.Linear(config.input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, config.seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64, dropout=config.dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc = nn.Linear(d_model, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


@register_model
class TransformerMiniModel(BaseDLModel):
    """Balanced Transformer for sequential data."""
    
    model_type_id = "transformer_mini"
    model_name = "Transformer Mini"
    model_description = "Balanced Transformer (3 layers, 4 heads, ~200K params)"
    architecture = "transformer"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        d_model = 64
        self.embedding = nn.Linear(config.input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, config.seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, dropout=config.dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


@register_model
class TransformerMaxModel(BaseDLModel):
    """Maximum capacity Transformer."""
    
    model_type_id = "transformer_max"
    model_name = "Transformer Max"
    model_description = "Maximum Transformer (6 layers, 8 heads, ~2M params)"
    architecture = "transformer"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        d_model = 256
        self.embedding = nn.Linear(config.input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, config.seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=config.dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
