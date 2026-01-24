"""CNN1D Models for 2D Input (batch, seq_len, features).

1D Convolutional models for time-series in nano, mini, and max variants.
Input shape: (batch, seq_len, features) - Sequential/time-series data

Verification References:
- PyTorch Conv1d: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
- PyTorch MaxPool1d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
- PyTorch BatchNorm1d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- PyTorch AdaptiveAvgPool1d: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html
- Parameters verified: in_channels, out_channels, kernel_size, padding, stride
- Note: Input permuted from (batch, seq_len, features) to (batch, features, seq_len) for Conv1d
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
class CNN1DNanoModel(BaseDLModel):
    """Minimal 1D CNN for sequential data."""
    
    model_type_id = "cnn1d_nano"
    model_name = "CNN1D Nano"
    model_description = "Minimal 1D CNN (2 conv layers, ~15K params)"
    architecture = "cnn1d"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


@register_model
class CNN1DMiniModel(BaseDLModel):
    """Balanced 1D CNN for sequential data."""
    
    model_type_id = "cnn1d_mini"
    model_name = "CNN1D Mini"
    model_description = "Balanced 1D CNN (4 conv layers, ~150K params)"
    architecture = "cnn1d"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


@register_model
class CNN1DMaxModel(BaseDLModel):
    """Maximum capacity 1D CNN with residual connections."""
    
    model_type_id = "cnn1d_max"
    model_name = "CNN1D Max"
    model_description = "Maximum 1D CNN with residuals (~1M params)"
    architecture = "cnn1d"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_2D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.input_conv = nn.Conv1d(config.input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res1 = self._make_res_block(64, 128)
        self.res2 = self._make_res_block(128, 256)
        self.res3 = self._make_res_block(256, 512)
        
        # Projection layers for residual connections (1x1 conv to match channels)
        self.proj1 = nn.Conv1d(64, 128, kernel_size=1)
        self.proj2 = nn.Conv1d(128, 256, kernel_size=1)
        self.proj3 = nn.Conv1d(256, 512, kernel_size=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def _make_res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.input_conv(x)))
        
        # Residual block 1 with projection
        identity1 = nn.functional.max_pool1d(self.proj1(x), 2)
        x = nn.functional.max_pool1d(self.res1(x), 2)
        x = torch.relu(x + identity1)
        
        # Residual block 2 with projection
        identity2 = nn.functional.max_pool1d(self.proj2(x), 2)
        x = nn.functional.max_pool1d(self.res2(x), 2)
        x = torch.relu(x + identity2)
        
        # Residual block 3 with projection
        identity3 = nn.functional.max_pool1d(self.proj3(x), 2)
        x = nn.functional.max_pool1d(self.res3(x), 2)
        x = torch.relu(x + identity3)
        
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
