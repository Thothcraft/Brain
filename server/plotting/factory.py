"""Factory for creating appropriate plotters based on model type.

Provides automatic plotter selection based on:
- Model type detection (ML, DL, FL)
- Training paradigm
- Data characteristics
"""

import logging
from typing import Optional, Union, Dict, Any

from .base import BasePlotter
from .ml_plotter import MLPlotter
from .dl_plotter import DLPlotter
from .fl_plotter import FLPlotter
from .comparison import ComparisonPlotter

logger = logging.getLogger(__name__)


# Model type mappings
ML_MODELS = {
    'svm', 'svc', 'random_forest', 'rf', 'knn', 'k_nearest_neighbors',
    'logistic_regression', 'lr', 'decision_tree', 'dt', 'gradient_boosting',
    'gb', 'adaboost', 'naive_bayes', 'nb', 'linear_svm', 'rbf_svm',
    'xgboost', 'lightgbm', 'catboost', 'extra_trees', 'bagging',
    'voting', 'stacking', 'ridge', 'lasso', 'elastic_net',
}

DL_MODELS = {
    'cnn', 'lstm', 'gru', 'transformer', 'mlp', 'resnet', 'vgg',
    'cnn1d', 'cnn2d', 'cnn3d', 'bilstm', 'bigru', 'attention',
    'bert', 'gpt', 'vit', 'efficientnet', 'mobilenet', 'densenet',
    'unet', 'autoencoder', 'vae', 'gan', 'diffusion',
}

FL_ALGORITHMS = {
    'fedavg', 'fedprox', 'fedadam', 'fedyogi', 'fedadagrad',
    'scaffold', 'fednova', 'fedopt', 'fedbn', 'fedper',
    'krum', 'multikrum', 'bulyan', 'trimmedmean', 'median',
    'feddf', 'fedmd', 'fedgen', 'moon', 'feddyn',
}


class PlotterFactory:
    """Factory for creating plotters based on model/training type."""
    
    _plotter_cache: Dict[str, BasePlotter] = {}
    
    @classmethod
    def create(cls, plotter_type: str, theme: str = "default",
              cache: bool = True) -> BasePlotter:
        """Create a plotter instance.
        
        Args:
            plotter_type: Type of plotter ('ml', 'dl', 'fl', 'comparison')
            theme: Theme name
            cache: Whether to cache and reuse plotter instances
        
        Returns:
            BasePlotter instance
        """
        cache_key = f"{plotter_type}_{theme}"
        
        if cache and cache_key in cls._plotter_cache:
            return cls._plotter_cache[cache_key]
        
        plotter_type = plotter_type.lower()
        
        if plotter_type in ('ml', 'machine_learning', 'classical'):
            plotter = MLPlotter(theme=theme)
        elif plotter_type in ('dl', 'deep_learning', 'neural'):
            plotter = DLPlotter(theme=theme)
        elif plotter_type in ('fl', 'federated', 'federated_learning'):
            plotter = FLPlotter(theme=theme)
        elif plotter_type in ('comparison', 'compare'):
            plotter = ComparisonPlotter(theme=theme)
        else:
            logger.warning(f"Unknown plotter type '{plotter_type}', defaulting to DL")
            plotter = DLPlotter(theme=theme)
        
        if cache:
            cls._plotter_cache[cache_key] = plotter
        
        return plotter
    
    @classmethod
    def create_for_model(cls, model_type: str, theme: str = "default") -> BasePlotter:
        """Create appropriate plotter based on model type.
        
        Args:
            model_type: Model type string (e.g., 'cnn', 'random_forest', 'fedavg')
            theme: Theme name
        
        Returns:
            Appropriate BasePlotter instance
        """
        category = cls.detect_model_category(model_type)
        return cls.create(category, theme)
    
    @classmethod
    def detect_model_category(cls, model_type: str) -> str:
        """Detect the category of a model.
        
        Args:
            model_type: Model type string
        
        Returns:
            Category string ('ml', 'dl', 'fl')
        """
        model_lower = model_type.lower().replace('-', '_').replace(' ', '_')
        
        # Check for FL algorithms
        for fl_algo in FL_ALGORITHMS:
            if fl_algo in model_lower:
                return 'fl'
        
        # Check for ML models
        for ml_model in ML_MODELS:
            if ml_model in model_lower:
                return 'ml'
        
        # Check for DL models
        for dl_model in DL_MODELS:
            if dl_model in model_lower:
                return 'dl'
        
        # Default to DL (most common in research)
        return 'dl'
    
    @classmethod
    def clear_cache(cls):
        """Clear the plotter cache."""
        for plotter in cls._plotter_cache.values():
            plotter.close()
        cls._plotter_cache.clear()


def create_plotter(plotter_type: str = None, model_type: str = None,
                  theme: str = "default") -> BasePlotter:
    """Convenience function to create a plotter.
    
    Args:
        plotter_type: Explicit plotter type ('ml', 'dl', 'fl', 'comparison')
        model_type: Model type for automatic detection
        theme: Theme name
    
    Returns:
        BasePlotter instance
    
    Examples:
        # Explicit type
        plotter = create_plotter('dl', theme='neurips')
        
        # Auto-detect from model
        plotter = create_plotter(model_type='random_forest')
        
        # Comparison plotter
        plotter = create_plotter('comparison')
    """
    if plotter_type:
        return PlotterFactory.create(plotter_type, theme)
    elif model_type:
        return PlotterFactory.create_for_model(model_type, theme)
    else:
        # Default to DL plotter
        return PlotterFactory.create('dl', theme)


def get_available_themes() -> list:
    """Get list of available themes."""
    from .themes import ThemeManager
    return ThemeManager.list_themes()


def get_available_plotters() -> list:
    """Get list of available plotter types."""
    return ['ml', 'dl', 'fl', 'comparison']
