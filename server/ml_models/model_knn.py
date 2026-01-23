"""K-Nearest Neighbors Model.

Instance-based classifier using distance metrics.
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
class KNNModel(BaseMLModel):
    """K-Nearest Neighbors classifier.
    
    Instance-based learning with configurable distance metrics.
    """
    
    model_type_id = "knn"
    model_name = "K-Nearest Neighbors"
    model_description = "Instance-based classifier using distance metrics"
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
                "name": "n_neighbors",
                "type": "int",
                "default": 5,
                "description": "Number of neighbors to use",
            },
            {
                "name": "weights",
                "type": "str",
                "default": "uniform",
                "options": ["uniform", "distance"],
                "description": "Weight function for prediction",
            },
            {
                "name": "metric",
                "type": "str",
                "default": "minkowski",
                "options": ["euclidean", "manhattan", "minkowski", "cosine"],
                "description": "Distance metric",
            },
            {
                "name": "p",
                "type": "int",
                "default": 2,
                "description": "Power parameter for Minkowski metric",
            },
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train KNN model."""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = KNeighborsClassifier(
                n_neighbors=self.params.get("n_neighbors", 5),
                weights=self.params.get("weights", "uniform"),
                metric=self.params.get("metric", "minkowski"),
                p=self.params.get("p", 2),
                n_jobs=-1,
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
                },
                metadata={
                    "n_neighbors": self.params.get("n_neighbors", 5),
                    "n_classes": len(self.classes_),
                    "n_samples": X.shape[0],
                },
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                train_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with KNN."""
        start_time = time.time()
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            inference_time_ms=(time.time() - start_time) * 1000,
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
