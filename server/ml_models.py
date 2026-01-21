"""Traditional ML Models for Time-Series Classification.

This module provides scikit-learn based ML models for comparison with deep learning:
- AdaBoost
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)

These models work on flattened time-series features.
"""

import numpy as np
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import io

logger = logging.getLogger(__name__)


class MLModelWrapper:
    """Wrapper for scikit-learn models to handle time-series data."""
    
    def __init__(self, model_type: str, config: Dict[str, Any]):
        """Initialize ML model with configuration.
        
        Args:
            model_type: One of 'adaboost', 'knn', 'svc'
            config: Model-specific configuration parameters
        """
        self.model_type = model_type
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
        logger.info(f"Initializing {model_type} model with config: {config}")
        
        try:
            if model_type == 'adaboost':
                self.model = self._create_adaboost(config)
            elif model_type == 'knn':
                self.model = self._create_knn(config)
            elif model_type == 'svc':
                self.model = self._create_svc(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"Successfully created {model_type} model")
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_adaboost(self, config: Dict[str, Any]) -> AdaBoostClassifier:
        """Create AdaBoost classifier."""
        n_estimators = config.get('n_estimators', 50)
        learning_rate = config.get('learning_rate', 1.0)
        algorithm = config.get('algorithm', 'SAMME.R')
        max_depth = config.get('max_depth', 1)
        
        logger.debug(f"AdaBoost params: n_estimators={n_estimators}, lr={learning_rate}, algo={algorithm}, max_depth={max_depth}")
        
        base_tree = DecisionTreeClassifier(max_depth=max_depth)
        # scikit-learn 1.2+ renamed base_estimator to estimator
        return AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=42
        )
    
    def _create_knn(self, config: Dict[str, Any]) -> KNeighborsClassifier:
        """Create KNN classifier."""
        n_neighbors = config.get('n_neighbors', 5)
        weights = config.get('weights', 'uniform')  # 'uniform' or 'distance'
        metric = config.get('metric', 'euclidean')  # 'euclidean', 'manhattan', 'minkowski'
        algorithm = config.get('algorithm', 'auto')  # 'auto', 'ball_tree', 'kd_tree', 'brute'
        p = config.get('p', 2)  # Power parameter for Minkowski metric
        
        logger.debug(f"KNN params: n_neighbors={n_neighbors}, weights={weights}, metric={metric}, algo={algorithm}, p={p}")
        
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            algorithm=algorithm,
            p=p,
            n_jobs=-1  # Use all CPU cores
        )
    
    def _create_svc(self, config: Dict[str, Any]) -> SVC:
        """Create SVC classifier."""
        C = config.get('C', 1.0)
        kernel = config.get('kernel', 'rbf')  # 'linear', 'poly', 'rbf', 'sigmoid'
        gamma = config.get('gamma', 'scale')  # 'scale', 'auto', or float
        degree = config.get('degree', 3)  # For poly kernel
        probability = config.get('probability', True)
        max_iter = config.get('max_iter', -1)
        
        logger.debug(f"SVC params: C={C}, kernel={kernel}, gamma={gamma}, degree={degree}, prob={probability}")
        
        return SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=probability,
            max_iter=max_iter,
            random_state=42,
            cache_size=1000  # MB
        )
    
    def flatten_timeseries(self, X: np.ndarray) -> np.ndarray:
        """Flatten time-series data for ML models.
        
        Args:
            X: Input data of shape (n_samples, window_size, n_features)
            
        Returns:
            Flattened data of shape (n_samples, window_size * n_features)
        """
        logger.debug(f"Flattening input shape {X.shape}")
        
        if len(X.shape) == 3:
            n_samples, window_size, n_features = X.shape
            X_flat = X.reshape(n_samples, window_size * n_features)
            logger.debug(f"Flattened to shape {X_flat.shape}")
            return X_flat
        elif len(X.shape) == 2:
            logger.debug("Input already 2D, no flattening needed")
            return X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the ML model.
        
        Args:
            X_train: Training data (n_samples, window_size, n_features)
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {self.model_type}")
        logger.debug(f"Train shape: {X_train.shape}, labels: {y_train.shape}")
        
        try:
            # Flatten time-series data
            X_train_flat = self.flatten_timeseries(X_train)
            
            # Fit scaler on training data
            logger.debug("Fitting StandardScaler...")
            self.scaler.fit(X_train_flat)
            X_train_scaled = self.scaler.transform(X_train_flat)
            logger.debug(f"Scaled train data: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
            
            # Train model
            logger.info(f"Training {self.model_type} model...")
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True
            logger.info(f"{self.model_type} training completed successfully")
            
            # Compute training accuracy
            train_pred = self.model.predict(X_train_scaled)
            train_acc = accuracy_score(y_train, train_pred)
            logger.info(f"Training accuracy: {train_acc:.4f}")
            
            metrics = {
                'train_accuracy': float(train_acc),
                'model_type': self.model_type
            }
            
            # Compute validation metrics if provided
            if X_val is not None and y_val is not None:
                logger.debug(f"Computing validation metrics, val shape: {X_val.shape}")
                X_val_flat = self.flatten_timeseries(X_val)
                X_val_scaled = self.scaler.transform(X_val_flat)
                
                val_pred = self.model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, val_pred)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                
                metrics['val_accuracy'] = float(val_acc)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for {self.model_type}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        logger.debug(f"Predicting on shape {X.shape}")
        
        try:
            X_flat = self.flatten_timeseries(X)
            X_scaled = self.scaler.transform(X_flat)
            predictions = self.model.predict(X_scaled)
            logger.debug(f"Predictions shape: {predictions.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        logger.debug(f"Computing probabilities for shape {X.shape}")
        
        try:
            X_flat = self.flatten_timeseries(X)
            X_scaled = self.scaler.transform(X_flat)
            
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X_scaled)
                logger.debug(f"Probabilities shape: {probs.shape}")
                return probs
            else:
                logger.warning(f"{self.model_type} does not support predict_proba")
                # Return one-hot encoded predictions as fallback
                preds = self.model.predict(X_scaled)
                n_classes = len(np.unique(preds))
                probs = np.zeros((len(preds), n_classes))
                probs[np.arange(len(preds)), preds] = 1.0
                return probs
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def save_to_bytes(self) -> bytes:
        """Serialize model to bytes."""
        logger.debug(f"Serializing {self.model_type} model")
        
        try:
            buffer = io.BytesIO()
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'config': self.config,
                'is_fitted': self.is_fitted
            }, buffer)
            model_bytes = buffer.getvalue()
            logger.info(f"Model serialized: {len(model_bytes)} bytes")
            return model_bytes
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load_from_bytes(cls, model_bytes: bytes) -> 'MLModelWrapper':
        """Deserialize model from bytes."""
        logger.debug("Deserializing ML model")
        
        try:
            buffer = io.BytesIO(model_bytes)
            data = pickle.load(buffer)
            
            wrapper = cls(data['model_type'], data['config'])
            wrapper.model = data['model']
            wrapper.scaler = data['scaler']
            wrapper.is_fitted = data['is_fitted']
            
            logger.info(f"Model deserialized: {data['model_type']}")
            return wrapper
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_type': self.model_type,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        if self.model_type == 'adaboost':
            info['n_estimators'] = len(self.model.estimators_) if self.is_fitted else self.config.get('n_estimators', 50)
        elif self.model_type == 'knn':
            info['n_neighbors'] = self.config.get('n_neighbors', 5)
        elif self.model_type == 'svc':
            info['kernel'] = self.config.get('kernel', 'rbf')
            info['C'] = self.config.get('C', 1.0)
        
        return info


def get_default_ml_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for ML model types."""
    defaults = {
        'adaboost': {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'max_depth': 1
        },
        'knn': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'euclidean',
            'algorithm': 'auto',
            'p': 2
        },
        'svc': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'degree': 3,
            'probability': True,
            'max_iter': -1
        }
    }
    
    return defaults.get(model_type, {})
