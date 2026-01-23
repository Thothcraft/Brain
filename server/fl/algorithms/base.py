"""Base classes for FL algorithm strategies.

This module provides base classes and utilities for creating FL algorithm wrappers.
All algorithm implementations should inherit from BaseStrategyWrapper.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple, Type

from flwr.common import Parameters, Metrics
from flwr.server.strategy import Strategy

from ..core.config import ExperimentConfig, FLAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmMetadata:
    """Metadata for an FL algorithm."""
    name: str
    description: str
    paper: str = ""
    supports_heterogeneous_models: bool = False
    category: str = "standard"  # standard, byzantine, fair, privacy, xgboost, mobile, fault_tolerant, kd
    params: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


class BaseStrategyWrapper(ABC):
    """Base class for FL algorithm strategy wrappers.
    
    All FL algorithms should inherit from this class and implement:
    - algorithm_id: The FLAlgorithm enum value
    - get_metadata(): Return algorithm metadata
    - create_strategy(): Create and return the Flower strategy
    
    This provides a consistent interface for all algorithms and enables
    easy extension and registration.
    """
    
    # Class-level attributes (override in subclasses)
    algorithm_id: FLAlgorithm = None
    flower_class: Type[Strategy] = None
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        """Get algorithm metadata for UI/documentation."""
        pass
    
    @classmethod
    @abstractmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create and return the Flower strategy.
        
        Args:
            config: Experiment configuration
            common_params: Common strategy parameters (fraction_fit, etc.)
        
        Returns:
            Configured Flower Strategy instance
        """
        pass
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this algorithm is available (dependencies installed)."""
        return cls.flower_class is not None


class AlgorithmRegistry:
    """Registry for FL algorithm wrappers.
    
    Provides centralized registration and lookup of algorithm implementations.
    """
    
    _algorithms: Dict[FLAlgorithm, Type[BaseStrategyWrapper]] = {}
    
    @classmethod
    def register(cls, wrapper_class: Type[BaseStrategyWrapper]):
        """Register an algorithm wrapper.
        
        Args:
            wrapper_class: The wrapper class to register
        """
        if wrapper_class.algorithm_id is not None:
            cls._algorithms[wrapper_class.algorithm_id] = wrapper_class
            logger.debug(f"Registered FL algorithm: {wrapper_class.algorithm_id.value}")
    
    @classmethod
    def get(cls, algorithm_id: FLAlgorithm) -> Optional[Type[BaseStrategyWrapper]]:
        """Get an algorithm wrapper by ID.
        
        Args:
            algorithm_id: The FLAlgorithm enum value
            
        Returns:
            The wrapper class or None if not found
        """
        return cls._algorithms.get(algorithm_id)
    
    @classmethod
    def list_algorithms(cls) -> List[Dict[str, Any]]:
        """List all registered algorithms with metadata.
        
        Returns:
            List of algorithm metadata dictionaries
        """
        result = []
        for algo_id, wrapper_class in cls._algorithms.items():
            meta = wrapper_class.get_metadata()
            result.append({
                "id": algo_id.value,
                "name": meta.name,
                "description": meta.description,
                "paper": meta.paper,
                "category": meta.category,
                "supports_heterogeneous_models": meta.supports_heterogeneous_models,
                "available": wrapper_class.is_available(),
            })
        return result
    
    @classmethod
    def list_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """List algorithms filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of algorithm metadata in the category
        """
        return [
            algo for algo in cls.list_algorithms()
            if algo.get("category") == category
        ]
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all algorithm categories."""
        categories = set()
        for wrapper_class in cls._algorithms.values():
            meta = wrapper_class.get_metadata()
            categories.add(meta.category)
        return sorted(list(categories))


def register_algorithm(cls: Type[BaseStrategyWrapper]) -> Type[BaseStrategyWrapper]:
    """Decorator to register an FL algorithm wrapper.
    
    Usage:
        @register_algorithm
        class MyAlgorithm(BaseStrategyWrapper):
            algorithm_id = FLAlgorithm.MY_ALGO
            ...
    """
    AlgorithmRegistry.register(cls)
    return cls


# Common utility functions for strategies

def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics using weighted average."""
    if not metrics:
        return {}
    train_losses = [num * m.get("train_loss", 0) for num, m in metrics]
    examples = [num for num, _ in metrics]
    total = sum(examples)
    return {"train_loss": sum(train_losses) / total if total > 0 else 0.0}


def weighted_average_evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average."""
    if not metrics:
        return {}
    accuracies = [num * m.get("accuracy", 0) for num, m in metrics]
    examples = [num for num, _ in metrics]
    total = sum(examples)
    return {"accuracy": sum(accuracies) / total if total > 0 else 0.0}


def build_common_params(
    config: ExperimentConfig,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable] = None,
    on_evaluate_config_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Build common strategy parameters from config.
    
    Args:
        config: Experiment configuration
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        on_fit_config_fn: Function to configure client training
        on_evaluate_config_fn: Function to configure client evaluation
    
    Returns:
        Dictionary of common strategy parameters
    """
    server = config.server
    client = config.client
    algorithm = config.algorithm
    algo_params = config.algorithm_params
    
    # Default config functions
    if on_fit_config_fn is None:
        def on_fit_config_fn(server_round: int) -> Dict[str, Any]:
            return {
                "server_round": server_round,
                "local_epochs": client.local_epochs,
                "lr": client.learning_rate,
                "proximal_mu": algo_params.proximal_mu if algorithm == FLAlgorithm.FEDPROX else 0.0,
            }
    
    if on_evaluate_config_fn is None:
        def on_evaluate_config_fn(server_round: int) -> Dict[str, Any]:
            return {"server_round": server_round}
    
    return {
        "fraction_fit": server.fraction_fit,
        "fraction_evaluate": server.fraction_evaluate,
        "min_fit_clients": server.min_fit_clients,
        "min_evaluate_clients": server.min_evaluate_clients,
        "min_available_clients": server.min_available_clients,
        "fit_metrics_aggregation_fn": weighted_average_fit,
        "evaluate_metrics_aggregation_fn": weighted_average_evaluate,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
    }
