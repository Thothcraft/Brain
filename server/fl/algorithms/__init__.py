"""FL Algorithms Module using Flower Strategies.

This module provides a modular, extensible architecture for FL algorithms.
Each algorithm is implemented in its own file for easy extension and maintenance.

Architecture:
- base.py: Base classes (BaseStrategyWrapper, AlgorithmRegistry)
- standard/: FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, FedAvgM, FedOpt
- byzantine/: FedMedian, FedTrimmedAvg, Krum, MultiKrum, Bulyan
- fair/: QFedAvg
- privacy/: DPFedAvgAdaptive, DPFedAvgFixed
- xgboost/: FedXgbBagging, FedXgbCyclic, FedXgbNnAvg
- mobile/: FedAvgAndroid
- fault_tolerant/: FaultTolerantFedAvg
- knowledge_distillation/: FedDF, FedMD, FedGen

Usage:
    from server.fl.algorithms import create_strategy, AlgorithmRegistry
    
    # Create strategy from config
    strategy = create_strategy(config)
    
    # List all algorithms
    algorithms = AlgorithmRegistry.list_algorithms()
    
    # Get algorithms by category
    byzantine_algos = AlgorithmRegistry.list_by_category("byzantine")
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple

from flwr.common import Parameters, Metrics
from flwr.server.strategy import Strategy

from ..core.config import FLAlgorithm, ExperimentConfig
from .base import (
    BaseStrategyWrapper,
    AlgorithmRegistry,
    AlgorithmMetadata,
    register_algorithm,
    build_common_params,
    weighted_average_fit,
    weighted_average_evaluate,
)

# Import all algorithm modules to trigger registration
from .standard import *
from .byzantine import *
from .fair import *
from .privacy import *
from .xgboost import *
from .mobile import *
from .fault_tolerant import *

logger = logging.getLogger(__name__)


def create_strategy(
    config: ExperimentConfig,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable] = None,
    on_evaluate_config_fn: Optional[Callable] = None,
) -> Strategy:
    """Create a Flower strategy based on the experiment configuration.
    
    This function uses the modular algorithm registry to create strategies.
    Each algorithm is implemented in its own file for extensibility.
    
    Args:
        config: Experiment configuration
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        on_fit_config_fn: Function to configure client training
        on_evaluate_config_fn: Function to configure client evaluation
    
    Returns:
        Flower Strategy instance
    """
    algorithm = config.algorithm
    
    # Build common parameters
    common_params = build_common_params(
        config,
        initial_parameters,
        evaluate_fn,
        on_fit_config_fn,
        on_evaluate_config_fn,
    )
    
    # Handle knowledge distillation algorithms separately (custom implementations)
    if algorithm in [FLAlgorithm.FEDDF, FLAlgorithm.FEDMD, FLAlgorithm.FEDGEN]:
        from .knowledge_distillation import create_kd_strategy
        return create_kd_strategy(config, initial_parameters, evaluate_fn)
    
    # Look up algorithm in registry
    wrapper_class = AlgorithmRegistry.get(algorithm)
    
    if wrapper_class is not None:
        return wrapper_class.create_strategy(config, common_params)
    
    # Fallback to FedAvg for unknown algorithms
    logger.warning(f"Unknown algorithm {algorithm}, defaulting to FedAvg")
    from flwr.server.strategy import FedAvg
    return FedAvg(**common_params)


def list_algorithms() -> List[Dict[str, Any]]:
    """List all available FL algorithms with metadata."""
    return AlgorithmRegistry.list_algorithms()


def get_algorithm_info(algorithm) -> Dict[str, Any]:
    """Get detailed information about an FL algorithm.
    
    Args:
        algorithm: FLAlgorithm enum value or string
    
    Returns:
        Dictionary with algorithm details
    """
    if isinstance(algorithm, str):
        algorithm = FLAlgorithm(algorithm.lower())
    
    wrapper_class = AlgorithmRegistry.get(algorithm)
    if wrapper_class is not None:
        meta = wrapper_class.get_metadata()
        return {
            "name": meta.name,
            "description": meta.description,
            "paper": meta.paper,
            "category": meta.category,
            "params": meta.params,
            "pros": meta.pros,
            "cons": meta.cons,
            "available": wrapper_class.is_available(),
        }
    
    return {"name": algorithm.value, "description": "FL algorithm"}


# Legacy ALGORITHM_REGISTRY for backward compatibility
# This is auto-generated from the modular registry
def _build_legacy_registry() -> Dict[FLAlgorithm, Dict[str, Any]]:
    """Build legacy registry from modular wrappers."""
    registry = {}
    for algo_id, wrapper_class in AlgorithmRegistry._algorithms.items():
        meta = wrapper_class.get_metadata()
        registry[algo_id] = {
            "name": meta.name,
            "description": meta.description,
            "paper": meta.paper,
            "supports_heterogeneous_models": meta.supports_heterogeneous_models,
            "flower_class": wrapper_class.flower_class,
        }
    return registry


# Build legacy registry after all imports
ALGORITHM_REGISTRY = _build_legacy_registry()

# Add KD algorithms to legacy registry
ALGORITHM_REGISTRY[FLAlgorithm.FEDDF] = {
    "name": "FedDF",
    "description": "Federated Distillation - supports heterogeneous model architectures",
    "paper": "Lin et al., 2020",
    "supports_heterogeneous_models": True,
    "flower_class": None,
}
ALGORITHM_REGISTRY[FLAlgorithm.FEDMD] = {
    "name": "FedMD",
    "description": "Federated Model Distillation - heterogeneous models via soft labels",
    "paper": "Li & Wang, 2019",
    "supports_heterogeneous_models": True,
    "flower_class": None,
}
ALGORITHM_REGISTRY[FLAlgorithm.FEDGEN] = {
    "name": "FedGen",
    "description": "Federated Generative Learning",
    "paper": "Zhu et al., 2021",
    "supports_heterogeneous_models": True,
    "flower_class": None,
}


__all__ = [
    # Main API
    "create_strategy",
    "list_algorithms",
    "get_algorithm_info",
    # Registry
    "AlgorithmRegistry",
    "ALGORITHM_REGISTRY",
    # Base classes for extension
    "BaseStrategyWrapper",
    "AlgorithmMetadata",
    "register_algorithm",
    # Utilities
    "build_common_params",
    "weighted_average_fit",
    "weighted_average_evaluate",
    # Config
    "FLAlgorithm",
]
