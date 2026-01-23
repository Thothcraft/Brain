"""ResNet Models for 4D Input (batch, channels, height, width).

ResNet-style models in nano, mini, and max variants.
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


class BasicBlock(nn.Module):
    """Basic residual block."""
    expansion = 1
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return torch.relu(out + identity)


@register_model
class ResNetNanoModel(BaseDLModel):
    """Minimal ResNet for images - edge deployment."""
    
    model_type_id = "resnet_nano"
    model_name = "ResNet Nano"
    model_description = "Minimal ResNet (8 layers, ~100K params)"
    architecture = "resnet"
    size = ModelSize.NANO
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 1, stride=2)
        self.layer3 = self._make_layer(64, 128, 1, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, config.num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@register_model
class ResNetMiniModel(BaseDLModel):
    """Balanced ResNet (similar to ResNet-18)."""
    
    model_type_id = "resnet_mini"
    model_name = "ResNet Mini"
    model_description = "Balanced ResNet-18 style (~1M params)"
    architecture = "resnet"
    size = ModelSize.MINI
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, config.num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@register_model
class ResNetMaxModel(BaseDLModel):
    """Maximum capacity ResNet (similar to ResNet-50)."""
    
    model_type_id = "resnet_max"
    model_name = "ResNet Max"
    model_description = "Maximum ResNet-50 style (~25M params)"
    architecture = "resnet"
    size = ModelSize.MAX
    input_shape = InputShape.SHAPE_4D
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
