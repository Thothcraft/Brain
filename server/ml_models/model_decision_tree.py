"""Decision Tree Model.

Single decision tree classifier with interpretable rules.
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
class DecisionTreeModel(BaseMLModel):
    """Decision Tree classifier.
    
    Interpretable tree-based classifier.
    """
    
    model_type_id = "decision_tree"
    model_name = "Decision Tree"
    model_description = "Interpretable tree-based classifier"
    model_type = ModelType.CLASSIFICATION
    input_shape = "1d"
    supports_proba = True
    supports_feature_importance = True
    category = "classification"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "max_depth", "type": "int", "default": None, "description": "Maximum depth"},
            {"name": "min_samples_split", "type": "int", "default": 2, "description": "Min samples to split"},
            {"name": "min_samples_leaf", "type": "int", "default": 1, "description": "Min samples in leaf"},
            {"name": "criterion", "type": "str", "default": "gini", "options": ["gini", "entropy"], "description": "Split criterion"},
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        try:
            self.model = DecisionTreeClassifier(
                max_depth=self.params.get("max_depth"),
                min_samples_split=self.params.get("min_samples_split", 2),
                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                criterion=self.params.get("criterion", "gini"),
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
                metrics={"train_accuracy": train_acc, "tree_depth": self.model.get_depth()},
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
