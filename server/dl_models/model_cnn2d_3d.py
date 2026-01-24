"""CNN2D Models for 3D Input (batch, channels, height, width).

2D Convolutional models for images in nano, mini, and max variants.
Input shape: (batch, channels, height, width) - Single images

Verification References:
- PyTorch Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch MaxPool2d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- PyTorch BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- PyTorch AdaptiveAvgPool2d: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
- ResNet Architecture: https://arxiv.org/abs/1512.03385 (He et al., 2015)
- Parameters verified: in_channels, out_channels, kernel_size, stride, padding, bias
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
class CNN2DNanoModel(BaseDLModel):
    """Minimal 2D CNN for images - edge deployment."""
    
    model_type_id = "cnn2d_nano"
    model_name = "CNN2D Nano"
    model_description = "Minimal 2D CNN (3 conv layers, ~30K params)"
    architecture = "cnn2d"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register_model
class CNN2DMiniModel(BaseDLModel):
    """Balanced 2D CNN for images."""
    
    model_type_id = "cnn2d_mini"
    model_name = "CNN2D Mini"
    model_description = "Balanced 2D CNN (5 conv layers, ~500K params)"
    architecture = "cnn2d"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@register_model
class CNN2DMaxModel(BaseDLModel):
    """Maximum capacity 2D CNN with residual connections."""
    
    model_type_id = "cnn2d_max"
    model_name = "CNN2D Max"
    model_description = "Maximum 2D CNN with residuals (~5M params)"
    architecture = "cnn2d"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_3D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        
        return ResBlock(in_ch, out_ch, stride, downsample)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResBlock(nn.Module):
    """Residual block for CNN2D Max."""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return torch.relu(out)
