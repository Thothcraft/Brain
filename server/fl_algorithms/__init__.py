"""Federated Learning Algorithms Module.

This module provides modular FL algorithms using the Flower framework.
Each algorithm is in a separate file with standardized naming conventions.

Naming Convention:
- File: algorithm_{algorithm_name}.py (e.g., algorithm_fedavg.py)
- Class: {AlgorithmName}Strategy (e.g., FedAvgStrategy)

Supported Algorithms:
- FedAvg: Federated Averaging (baseline)
- FedProx: Federated Proximal (handles heterogeneity)
- FedAdam: Federated Adam optimizer
- FedYogi: Federated Yogi optimizer
- Scaffold: Stochastic Controlled Averaging
- FedNova: Normalized Averaging

Usage:
    from server.fl_algorithms import FLAlgorithmRegistry, create_fl_strategy
    
    # List available algorithms
    algorithms = FLAlgorithmRegistry.list_algorithms()
    
    # Create strategy
    strategy = create_fl_strategy("fedavg", {
        "fraction_fit": 0.5,
        "min_fit_clients": 2,
    })
"""

from .base import (
    BaseFLAlgorithm,
    FLAlgorithmRegistry,
    FLConfig,
    FLMetrics,
)

# Import all algorithms to register them
from . import algorithm_fedavg
from . import algorithm_fedprox
from . import algorithm_fedadam
from . import algorithm_fedyogi
from . import algorithm_scaffold
from . import algorithm_fednova

__all__ = [
    "BaseFLAlgorithm",
    "FLAlgorithmRegistry",
    "FLConfig",
    "FLMetrics",
    "create_fl_strategy",
]

def create_fl_strategy(algorithm_type: str, config: dict = None):
    """Convenience function to create an FL strategy."""
    return FLAlgorithmRegistry.create(algorithm_type, config)
