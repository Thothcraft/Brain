"""Base classes and registry for ML models.

This module defines the base class and registry pattern for all ML models.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Type, Union
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models."""
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    REGRESSION = "regression"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ModelMetadata:
    """Metadata for an ML model."""
    name: str
    description: str
    model_type: ModelType
    input_shape: str = "1d"  # 1d, 2d, any
    params: List[Dict[str, Any]] = field(default_factory=list)
    supports_proba: bool = False
    supports_feature_importance: bool = False
    category: str = "general"
    version: str = "1.0.0"


@dataclass
class TrainingResult:
    """Result from training an ML model."""
    success: bool
    train_time_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PredictionResult:
    """Result from model prediction."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMLModel(ABC):
    """Base class for all ML models.
    
    All ML models must inherit from this class and implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_metadata(): Return model metadata
    
    Naming Convention:
    - File: model_{model_name}.py
    - Class: {ModelName}Model
    """
    
    # Class-level metadata (override in subclasses)
    model_type_id: str = "base"
    model_name: str = "Base Model"
    model_description: str = "Base ML model"
    model_type: ModelType = ModelType.CLASSIFICATION
    input_shape: str = "1d"
    supports_proba: bool = False
    supports_feature_importance: bool = False
    category: str = "general"
    version: str = "1.0.0"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize model with parameters.
        
        Args:
            params: Dictionary of model-specific parameters
        """
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.classes_ = None
        self.n_features_ = None
        self._validate_params()
    
    def _validate_params(self):
        """Validate parameters against expected schema."""
        pass  # Override in subclasses if needed
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Train the model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - optional for unsupervised
            
        Returns:
            TrainingResult with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions.
        
        Args:
            X: Features to predict (n_samples, n_features)
            
        Returns:
            PredictionResult with predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities (if supported).
        
        Args:
            X: Features to predict
            
        Returns:
            Probability array or None if not supported
        """
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (if supported).
        
        Returns:
            Feature importance array or None if not supported
        """
        return None
    
    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        """Get model metadata for registration and UI display."""
        return ModelMetadata(
            name=cls.model_name,
            description=cls.model_description,
            model_type=cls.model_type,
            input_shape=cls.input_shape,
            params=cls.get_param_schema(),
            supports_proba=cls.supports_proba,
            supports_feature_importance=cls.supports_feature_importance,
            category=cls.category,
            version=cls.version,
        )
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        """Get parameter schema for the model.
        
        Override in subclasses to define parameters.
        
        Returns:
            List of parameter definitions
        """
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get model instance information."""
        return {
            "type": self.model_type_id,
            "name": self.model_name,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "n_features": self.n_features_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }
    
    def save(self, path: str):
        """Save model to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseMLModel':
        """Load model from file."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class MLModelRegistry:
    """Registry for ML models.
    
    Provides centralized registration and lookup of ML models.
    """
    
    _models: Dict[str, Type[BaseMLModel]] = {}
    _metadata: Dict[str, ModelMetadata] = {}
    
    @classmethod
    def register(cls, model_class: Type[BaseMLModel]):
        """Register an ML model.
        
        Args:
            model_class: The model class to register
        """
        model_type_id = model_class.model_type_id
        cls._models[model_type_id] = model_class
        cls._metadata[model_type_id] = model_class.get_metadata()
        logger.debug(f"Registered ML model: {model_type_id}")
    
    @classmethod
    def get(cls, model_type_id: str) -> Optional[Type[BaseMLModel]]:
        """Get a model class by type ID."""
        return cls._models.get(model_type_id)
    
    @classmethod
    def create(cls, model_type_id: str, params: Optional[Dict] = None) -> Optional[BaseMLModel]:
        """Create a model instance."""
        model_class = cls.get(model_type_id)
        if model_class:
            return model_class(params)
        return None
    
    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """List all registered models with metadata."""
        return [
            {
                "type": model_type_id,
                "name": meta.name,
                "description": meta.description,
                "model_type": meta.model_type.value,
                "input_shape": meta.input_shape,
                "params": meta.params,
                "supports_proba": meta.supports_proba,
                "supports_feature_importance": meta.supports_feature_importance,
                "category": meta.category,
            }
            for model_type_id, meta in cls._metadata.items()
        ]
    
    @classmethod
    def list_by_type(cls, model_type: ModelType) -> List[Dict[str, Any]]:
        """List models filtered by type."""
        return [
            model for model in cls.list_models()
            if model["model_type"] == model_type.value
        ]
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all categories."""
        return list(set(meta.category for meta in cls._metadata.values()))


def register_model(cls: Type[BaseMLModel]) -> Type[BaseMLModel]:
    """Decorator to register an ML model."""
    MLModelRegistry.register(cls)
    return cls
