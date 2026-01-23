"""Unified Machine Learning Module.

This module provides a unified interface for:
- Classical ML models (SVM, Random Forest, KNN, etc.)
- Deep Learning models (CNN, LSTM, GRU, Transformer, etc.)
- Training pipelines
- Model selection and recommendation

Submodules:
- ml.models: Classical ML models (re-exports from ml_models/)
- ml.deep: Deep Learning models (re-exports from dl_models/)
- ml.training: Training utilities and pipelines
- ml.selection: Automatic model selection

Usage:
    from server.ml import (
        create_model,
        train_model,
        ModelSelector,
        list_models,
    )
    
    # Create and train a model
    model = create_model("random_forest", {"n_estimators": 100})
    results = train_model(model, X_train, y_train, X_val, y_val)
"""

# Re-export from ml_models
from ..ml_models import (
    BaseMLModel,
    MLModelRegistry,
    ModelType as MLModelType,
    ModelMetadata,
    TrainingResult,
    PredictionResult,
    create_ml_model,
)

# Re-export from dl_models
from ..dl_models import (
    BaseDLModel,
    DLModelRegistry,
    ModelSize,
    InputShape,
    ModelConfig as DLModelConfig,
    create_dl_model,
)

# Re-export from model_selector
from ..model_selector import (
    ModelSelector,
    ModelRecommendation,
    DataType,
    OutputShape,
)

__all__ = [
    # ML Models
    "BaseMLModel",
    "MLModelRegistry",
    "MLModelType",
    "ModelMetadata",
    "TrainingResult",
    "PredictionResult",
    "create_ml_model",
    # DL Models
    "BaseDLModel",
    "DLModelRegistry",
    "ModelSize",
    "InputShape",
    "DLModelConfig",
    "create_dl_model",
    # Model Selection
    "ModelSelector",
    "ModelRecommendation",
    "DataType",
    "OutputShape",
    # Convenience functions
    "create_model",
    "list_all_models",
]


def create_model(model_type: str, params: dict = None, is_deep_learning: bool = None):
    """Create a model by type (auto-detects ML vs DL).
    
    Args:
        model_type: Model type string (e.g., "random_forest", "lstm_mini")
        params: Model parameters
        is_deep_learning: Force DL (True) or ML (False), or auto-detect (None)
    
    Returns:
        Model instance
    """
    params = params or {}
    
    # Auto-detect based on model type
    if is_deep_learning is None:
        dl_keywords = ["lstm", "gru", "cnn", "transformer", "mlp", "resnet"]
        is_deep_learning = any(kw in model_type.lower() for kw in dl_keywords)
    
    if is_deep_learning:
        return create_dl_model(model_type, params)
    else:
        return create_ml_model(model_type, params)


def list_all_models() -> dict:
    """List all available models (both ML and DL).
    
    Returns:
        Dictionary with 'ml' and 'dl' model lists
    """
    return {
        "ml": MLModelRegistry.list_models() if hasattr(MLModelRegistry, 'list_models') else [],
        "dl": DLModelRegistry.list_models() if hasattr(DLModelRegistry, 'list_models') else [],
    }
