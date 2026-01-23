"""Random Forest Model.

Ensemble classifier using multiple decision trees.
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
class RandomForestModel(BaseMLModel):
    """Random Forest classifier.
    
    Ensemble of decision trees with feature importance.
    """
    
    model_type_id = "random_forest"
    model_name = "Random Forest"
    model_description = "Ensemble of decision trees with feature importance"
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
                "name": "n_estimators",
                "type": "int",
                "default": 100,
                "description": "Number of trees in the forest",
            },
            {
                "name": "max_depth",
                "type": "int",
                "default": None,
                "description": "Maximum depth of trees (None = unlimited)",
            },
            {
                "name": "min_samples_split",
                "type": "int",
                "default": 2,
                "description": "Minimum samples to split a node",
            },
            {
                "name": "min_samples_leaf",
                "type": "int",
                "default": 1,
                "description": "Minimum samples in a leaf node",
            },
            {
                "name": "max_features",
                "type": "str",
                "default": "sqrt",
                "options": ["sqrt", "log2", "auto"],
                "description": "Number of features for best split",
            },
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth"),
                min_samples_split=self.params.get("min_samples_split", 2),
                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                max_features=self.params.get("max_features", "sqrt"),
                random_state=42,
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
                    "n_estimators": self.params.get("n_estimators", 100),
                },
                feature_importance=self.model.feature_importances_.tolist(),
                metadata={
                    "n_classes": len(self.classes_),
                    "n_features": self.n_features_,
                },
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                train_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with Random Forest."""
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
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
