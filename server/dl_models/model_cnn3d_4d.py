"""CNN3D Models for 4D Input (batch, num_frames, channels, height, width).

3D Convolutional models for video/volumetric data in nano, mini, and max variants.
Input shape: (batch, num_frames, channels, height, width) - Video/volumetric data

Verification References:
- PyTorch Conv3d: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
- PyTorch MaxPool3d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
- PyTorch BatchNorm3d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html
- PyTorch AdaptiveAvgPool3d: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html
- 3D ResNet Paper: https://arxiv.org/abs/1711.11248 (Hara et al., 2017)
- Parameters verified: in_channels, out_channels, kernel_size (3-tuple), stride, padding, bias
- Note: Input permuted from (batch, frames, C, H, W) to (batch, C, frames, H, W) for Conv3d
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
class CNN3DNanoModel(BaseDLModel):
    """Minimal 3D CNN for video - edge deployment."""
    
    model_type_id = "cnn3d_nano"
    model_name = "CNN3D Nano"
    model_description = "Minimal 3D CNN for video (~50K params)"
    architecture = "cnn3d"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv3d(config.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_frames, channels, height, width) -> (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register_model
class CNN3DMiniModel(BaseDLModel):
    """Balanced 3D CNN for video."""
    
    model_type_id = "cnn3d_mini"
    model_name = "CNN3D Mini"
    model_description = "Balanced 3D CNN for video (~500K params)"
    architecture = "cnn3d"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv3d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_frames, channels, height, width) -> (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register_model
class CNN3DMaxModel(BaseDLModel):
    """Maximum capacity 3D CNN for video."""
    
    model_type_id = "cnn3d_max"
    model_name = "CNN3D Max"
    model_description = "Maximum 3D CNN for video (~5M params)"
    architecture = "cnn3d"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers.append(Res3DBlock(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(Res3DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_frames, channels, height, width) -> (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class Res3DBlock(nn.Module):
    """3D Residual block."""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return torch.relu(out + identity)
