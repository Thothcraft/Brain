"""Logistic Regression Model.

Linear classifier with probability estimates.
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
class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier.
    
    Linear model with L1/L2 regularization.
    """
    
    model_type_id = "logistic_regression"
    model_name = "Logistic Regression"
    model_description = "Linear classifier with probability estimates"
    model_type = ModelType.CLASSIFICATION
    input_shape = "1d"
    supports_proba = True
    supports_feature_importance = True
    category = "classification"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "C",
                "type": "float",
                "default": 1.0,
                "description": "Inverse regularization strength",
            },
            {
                "name": "penalty",
                "type": "str",
                "default": "l2",
                "options": ["l1", "l2", "elasticnet", "none"],
                "description": "Regularization type",
            },
            {
                "name": "solver",
                "type": "str",
                "default": "lbfgs",
                "options": ["lbfgs", "liblinear", "saga"],
                "description": "Optimization algorithm",
            },
            {
                "name": "max_iter",
                "type": "int",
                "default": 1000,
                "description": "Maximum iterations",
            },
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train Logistic Regression model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = LogisticRegression(
                C=self.params.get("C", 1.0),
                penalty=self.params.get("penalty", "l2"),
                solver=self.params.get("solver", "lbfgs"),
                max_iter=self.params.get("max_iter", 1000),
                random_state=42,
                n_jobs=-1,
            )
            
            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_
            self.n_features_ = X.shape[1]
            
            y_pred = self.model.predict(X)
            train_acc = accuracy_score(y, y_pred)
            
            train_time = (time.time() - start_time) * 1000
            
            return TrainingResult(
                success=True,
                train_time_ms=train_time,
                metrics={"train_accuracy": train_acc},
                feature_importance=np.abs(self.model.coef_).mean(axis=0).tolist() if len(self.classes_) > 2 else np.abs(self.model.coef_[0]).tolist(),
                metadata={"n_classes": len(self.classes_)},
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                train_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions."""
        start_time = time.time()
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            inference_time_ms=(time.time() - start_time) * 1000,
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        if len(self.classes_) > 2:
            return np.abs(self.model.coef_).mean(axis=0)
        return np.abs(self.model.coef_[0])
