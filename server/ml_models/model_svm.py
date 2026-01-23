"""Support Vector Machine Model.

SVM classifier with multiple kernel options.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    BaseMLModel,
    ModelType,
    TrainingResult,
    PredictionResult,
    register_model,
)


@register_model
class SVMModel(BaseMLModel):
    """Support Vector Machine classifier.
    
    Supports linear, RBF, polynomial, and sigmoid kernels.
    """
    
    model_type_id = "svm"
    model_name = "Support Vector Machine"
    model_description = "SVM classifier with multiple kernel options"
    model_type = ModelType.CLASSIFICATION
    input_shape = "1d"
    supports_proba = True
    supports_feature_importance = False
    category = "classification"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "kernel",
                "type": "str",
                "default": "rbf",
                "options": ["linear", "rbf", "poly", "sigmoid"],
                "description": "Kernel type",
            },
            {
                "name": "C",
                "type": "float",
                "default": 1.0,
                "description": "Regularization parameter",
            },
            {
                "name": "gamma",
                "type": "str",
                "default": "scale",
                "options": ["scale", "auto"],
                "description": "Kernel coefficient",
            },
            {
                "name": "degree",
                "type": "int",
                "default": 3,
                "description": "Degree for polynomial kernel",
            },
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train SVM model."""
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = SVC(
                kernel=self.params.get("kernel", "rbf"),
                C=self.params.get("C", 1.0),
                gamma=self.params.get("gamma", "scale"),
                degree=self.params.get("degree", 3),
                probability=True,
                random_state=42,
            )
            
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.n_features_ = X.shape[1]
            
            # Calculate training accuracy
            y_pred = self.model.predict(X)
            train_acc = accuracy_score(y, y_pred)
            
            train_time = (time.time() - start_time) * 1000
            
            return TrainingResult(
                success=True,
                train_time_ms=train_time,
                metrics={
                    "train_accuracy": train_acc,
                    "n_support_vectors": sum(self.model.n_support_),
                },
                metadata={
                    "kernel": self.params.get("kernel", "rbf"),
                    "n_classes": len(self.classes_),
                },
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                train_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with SVM."""
        start_time = time.time()
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X) if self.supports_proba else None
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            inference_time_ms=(time.time() - start_time) * 1000,
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
