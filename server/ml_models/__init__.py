"""Machine Learning Models Module.

This module provides classical ML models for classification and analysis.
Each model is in a separate file with standardized naming conventions.

Naming Convention:
- File: model_{model_name}.py (e.g., model_random_forest.py)
- Class: {ModelName}Model (e.g., RandomForestModel)

Supported Models:
- Classification: SVM, Random Forest, KNN, Logistic Regression, etc.
- Clustering: K-Means, DBSCAN, Hierarchical, etc.
- Dimensionality Reduction: PCA (2D/3D visualization)

Usage:
    from server.ml_models import MLModelRegistry, create_ml_model
    
    # List available models
    models = MLModelRegistry.list_models()
    
    # Create and train model
    model = create_ml_model("random_forest", {"n_estimators": 100})
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

from .base import (
    BaseMLModel,
    MLModelRegistry,
    ModelType,
    ModelMetadata,
    TrainingResult,
    PredictionResult,
)

# Import all models to register them
from . import model_svm
from . import model_random_forest
from . import model_knn
from . import model_logistic_regression
from . import model_gradient_boosting
from . import model_decision_tree
from . import model_kmeans
from . import model_dbscan
from . import model_pca_visualizer

# Import MLModelWrapper from the legacy ml_models.py file
# Since there's both ml_models.py and ml_models/ package, we need importlib
import importlib.util
import os

_legacy_module_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models.py")
_spec = importlib.util.spec_from_file_location("ml_models_legacy", _legacy_module_path)
_legacy_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy_module)

MLModelWrapper = _legacy_module.MLModelWrapper
get_default_ml_config = _legacy_module.get_default_ml_config

__all__ = [
    "BaseMLModel",
    "MLModelRegistry",
    "ModelType",
    "ModelMetadata",
    "TrainingResult",
    "PredictionResult",
    "create_ml_model",
    "MLModelWrapper",
    "get_default_ml_config",
]

def create_ml_model(model_type: str, params: dict = None):
    """Convenience function to create an ML model."""
    return MLModelRegistry.create(model_type, params)
