"""Base classes and registry for Federated Learning algorithms.

This module defines the base class and registry pattern for all FL algorithms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Type, Callable, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FLConfig:
    """Configuration for FL algorithms."""
    # Client selection
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.5
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    # Training
    num_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Algorithm-specific
    proximal_mu: float = 0.01  # FedProx
    server_lr: float = 1.0  # FedAdam/FedYogi
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau: float = 1e-3
    
    # Privacy
    use_dp: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


@dataclass
class FLMetrics:
    """Metrics from FL training round."""
    round_num: int
    loss: float
    accuracy: float
    num_clients: int
    aggregation_time_ms: float = 0.0
    client_metrics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmMetadata:
    """Metadata for an FL algorithm."""
    name: str
    description: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    handles_heterogeneity: bool = False
    communication_efficient: bool = False
    privacy_preserving: bool = False
    version: str = "1.0.0"


class BaseFLAlgorithm(ABC):
    """Base class for all FL algorithms.
    
    All FL algorithms must inherit from this class and implement:
    - aggregate_fit(): Aggregate client model updates
    - aggregate_evaluate(): Aggregate client evaluation metrics
    - configure_fit(): Configure client training
    - configure_evaluate(): Configure client evaluation
    
    Naming Convention:
    - File: algorithm_{algorithm_name}.py
    - Class: {AlgorithmName}Strategy
    """
    
    # Class-level metadata (override in subclasses)
    algorithm_type: str = "base"
    algorithm_name: str = "Base Algorithm"
    algorithm_description: str = "Base FL algorithm"
    handles_heterogeneity: bool = False
    communication_efficient: bool = False
    privacy_preserving: bool = False
    version: str = "1.0.0"
    
    def __init__(self, config: Optional[FLConfig] = None):
        """Initialize algorithm with configuration."""
        self.config = config or FLConfig()
        self.current_round = 0
        self.global_model_params = None
    
    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate client model updates.
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        pass
    
    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate client evaluation metrics.
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, evaluate_result) tuples
            failures: List of failed clients
            
        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        pass
    
    def configure_fit(
        self,
        server_round: int,
        parameters: List[np.ndarray],
        client_manager: Any,
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure client training for this round.
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Client manager instance
            
        Returns:
            List of (client_proxy, fit_config) tuples
        """
        config = {
            "server_round": server_round,
            "local_epochs": self.config.local_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
        }
        return config
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: List[np.ndarray],
        client_manager: Any,
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure client evaluation for this round."""
        config = {
            "server_round": server_round,
            "batch_size": self.config.batch_size,
        }
        return config
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        """Get algorithm metadata."""
        return AlgorithmMetadata(
            name=cls.algorithm_name,
            description=cls.algorithm_description,
            params=cls.get_param_schema(),
            handles_heterogeneity=cls.handles_heterogeneity,
            communication_efficient=cls.communication_efficient,
            privacy_preserving=cls.privacy_preserving,
            version=cls.version,
        )
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        """Get parameter schema for the algorithm."""
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get algorithm instance information."""
        return {
            "type": self.algorithm_type,
            "name": self.algorithm_name,
            "config": self.config.to_dict(),
            "current_round": self.current_round,
        }


class FLAlgorithmRegistry:
    """Registry for FL algorithms."""
    
    _algorithms: Dict[str, Type[BaseFLAlgorithm]] = {}
    _metadata: Dict[str, AlgorithmMetadata] = {}
    
    @classmethod
    def register(cls, algorithm_class: Type[BaseFLAlgorithm]):
        """Register an FL algorithm."""
        algorithm_type = algorithm_class.algorithm_type
        cls._algorithms[algorithm_type] = algorithm_class
        cls._metadata[algorithm_type] = algorithm_class.get_metadata()
        logger.debug(f"Registered FL algorithm: {algorithm_type}")
    
    @classmethod
    def get(cls, algorithm_type: str) -> Optional[Type[BaseFLAlgorithm]]:
        """Get an algorithm class by type."""
        return cls._algorithms.get(algorithm_type)
    
    @classmethod
    def create(cls, algorithm_type: str, config: Optional[Dict] = None) -> Optional[BaseFLAlgorithm]:
        """Create an algorithm instance."""
        algorithm_class = cls.get(algorithm_type)
        if algorithm_class:
            fl_config = FLConfig(**config) if config else FLConfig()
            return algorithm_class(fl_config)
        return None
    
    @classmethod
    def list_algorithms(cls) -> List[Dict[str, Any]]:
        """List all registered algorithms."""
        return [
            {
                "type": algorithm_type,
                "name": meta.name,
                "description": meta.description,
                "params": meta.params,
                "handles_heterogeneity": meta.handles_heterogeneity,
                "communication_efficient": meta.communication_efficient,
                "privacy_preserving": meta.privacy_preserving,
            }
            for algorithm_type, meta in cls._metadata.items()
        ]


def register_algorithm(cls: Type[BaseFLAlgorithm]) -> Type[BaseFLAlgorithm]:
    """Decorator to register an FL algorithm."""
    FLAlgorithmRegistry.register(cls)
    return cls
