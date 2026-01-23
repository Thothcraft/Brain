"""Gradient Boosting Model.

Ensemble classifier using gradient boosting.
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
class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting classifier.
    
    Sequential ensemble with gradient descent optimization.
    """
    
    model_type_id = "gradient_boosting"
    model_name = "Gradient Boosting"
    model_description = "Sequential ensemble with gradient descent optimization"
    model_type = ModelType.CLASSIFICATION
    input_shape = "1d"
    supports_proba = True
    supports_feature_importance = True
    category = "classification"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "n_estimators", "type": "int", "default": 100, "description": "Number of boosting stages"},
            {"name": "learning_rate", "type": "float", "default": 0.1, "description": "Learning rate"},
            {"name": "max_depth", "type": "int", "default": 3, "description": "Maximum depth of trees"},
            {"name": "subsample", "type": "float", "default": 1.0, "description": "Fraction of samples for fitting"},
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = GradientBoostingClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                learning_rate=self.params.get("learning_rate", 0.1),
                max_depth=self.params.get("max_depth", 3),
                subsample=self.params.get("subsample", 1.0),
                random_state=42,
            )
            
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.n_features_ = X.shape[1]
            
            y_pred = self.model.predict(X)
            train_acc = accuracy_score(y, y_pred)
            
            return TrainingResult(
                success=True,
                train_time_ms=(time.time() - start_time) * 1000,
                metrics={"train_accuracy": train_acc},
                feature_importance=self.model.feature_importances_.tolist(),
            )
        except Exception as e:
            return TrainingResult(success=False, train_time_ms=(time.time() - start_time) * 1000, error=str(e))
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        start_time = time.time()
        return PredictionResult(
            predictions=self.model.predict(X),
            probabilities=self.model.predict_proba(X),
            inference_time_ms=(time.time() - start_time) * 1000,
        )
    
    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_
