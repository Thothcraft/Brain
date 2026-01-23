"""Base classes and registry for Deep Learning models.

This module defines the base class and registry pattern for all DL models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Type
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelSize(str, Enum):
    """Model size variants."""
    NANO = "nano"   # Minimal parameters, edge deployment
    MINI = "mini"   # Balanced parameters
    MAX = "max"     # Maximum parameters


class InputShape(str, Enum):
    """Input shape types."""
    SHAPE_1D = "1d"  # (batch, features)
    SHAPE_3D = "3d"  # (batch, seq_len, features)
    SHAPE_4D = "4d"  # (batch, channels, height, width)


@dataclass
class ModelConfig:
    """Configuration for DL model creation."""
    # Common parameters
    num_classes: int = 2
    dropout: float = 0.3
    
    # 1D input parameters
    input_features: int = 128
    
    # 3D input parameters (sequential)
    seq_length: int = 128
    input_channels: int = 64
    
    # 4D input parameters (image/video)
    image_height: int = 224
    image_width: int = 224
    in_channels: int = 3
    num_frames: int = 16  # For video
    
    # Architecture parameters
    hidden_size: int = 128
    num_layers: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "input_features": self.input_features,
            "seq_length": self.seq_length,
            "input_channels": self.input_channels,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "in_channels": self.in_channels,
            "num_frames": self.num_frames,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


@dataclass
class ModelMetadata:
    """Metadata for a DL model."""
    name: str
    description: str
    architecture: str
    size: ModelSize
    input_shape: InputShape
    params: List[Dict[str, Any]] = field(default_factory=list)
    estimated_params: int = 0
    version: str = "1.0.0"


class BaseDLModel(nn.Module, ABC):
    """Base class for all DL models.
    
    All DL models must inherit from this class and implement:
    - forward(): The forward pass
    - get_metadata(): Return model metadata
    
    Naming Convention:
    - File: model_{architecture}_{shape}.py
    - Class: {Architecture}{Size}Model
    """
    
    # Class-level metadata (override in subclasses)
    model_type_id: str = "base"
    model_name: str = "Base Model"
    model_description: str = "Base DL model"
    architecture: str = "base"
    size: ModelSize = ModelSize.MINI
    input_shape: InputShape = InputShape.SHAPE_1D
    version: str = "1.0.0"
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            name=cls.model_name,
            description=cls.model_description,
            architecture=cls.architecture,
            size=cls.size,
            input_shape=cls.input_shape,
            params=cls.get_param_schema(),
            version=cls.version,
        )
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        """Get parameter schema for the model."""
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model instance information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "type": self.model_type_id,
            "name": self.model_name,
            "architecture": self.architecture,
            "size": self.size.value,
            "input_shape": self.input_shape.value,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "config": self.config.to_dict(),
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DLModelRegistry:
    """Registry for DL models."""
    
    _models: Dict[str, Type[BaseDLModel]] = {}
    _metadata: Dict[str, ModelMetadata] = {}
    
    @classmethod
    def register(cls, model_class: Type[BaseDLModel]):
        """Register a DL model."""
        model_type_id = model_class.model_type_id
        cls._models[model_type_id] = model_class
        cls._metadata[model_type_id] = model_class.get_metadata()
        logger.debug(f"Registered DL model: {model_type_id}")
    
    @classmethod
    def get(cls, model_type_id: str) -> Optional[Type[BaseDLModel]]:
        """Get a model class by type ID."""
        return cls._models.get(model_type_id)
    
    @classmethod
    def create(cls, model_type_id: str, config: Optional[Dict] = None) -> Optional[BaseDLModel]:
        """Create a model instance."""
        model_class = cls.get(model_type_id)
        if model_class:
            model_config = ModelConfig(**config) if config else ModelConfig()
            return model_class(model_config)
        return None
    
    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """List all registered models."""
        return [
            {
                "type": model_type_id,
                "name": meta.name,
                "description": meta.description,
                "architecture": meta.architecture,
                "size": meta.size.value,
                "input_shape": meta.input_shape.value,
                "params": meta.params,
            }
            for model_type_id, meta in cls._metadata.items()
        ]
    
    @classmethod
    def list_by_shape(cls, shape: str) -> List[Dict[str, Any]]:
        """List models compatible with a specific input shape."""
        return [
            model for model in cls.list_models()
            if model["input_shape"] == shape
        ]
    
    @classmethod
    def list_by_size(cls, size: str) -> List[Dict[str, Any]]:
        """List models of a specific size."""
        return [
            model for model in cls.list_models()
            if model["size"] == size
        ]
    
    @classmethod
    def get_compatible_models(cls, input_shape: InputShape) -> Dict[str, List[str]]:
        """Get compatible models grouped by size for an input shape."""
        models = cls.list_by_shape(input_shape.value)
        
        result = {"nano": [], "mini": [], "max": []}
        for model in models:
            size = model["size"]
            if size in result:
                result[size].append(model["type"])
        
        return result


def register_model(cls: Type[BaseDLModel]) -> Type[BaseDLModel]:
    """Decorator to register a DL model."""
    DLModelRegistry.register(cls)
    return cls
