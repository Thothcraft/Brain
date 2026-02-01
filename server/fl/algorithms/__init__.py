"""
Federated Learning Algorithms Module.

This module provides custom FL strategy implementations based on official
Flower documentation and examples. NO built-in Flower strategies are used -
all aggregation logic is explicitly implemented for clarity and debugging.

=============================================================================
AVAILABLE STRATEGIES
=============================================================================

FedAvg (Federated Averaging):
    The foundational FL algorithm. Computes weighted average of client updates.
    Paper: McMahan et al., 2017 - https://arxiv.org/abs/1602.05629

FedProx:
    FedAvg with a proximal term for heterogeneous (non-IID) data.
    Paper: Li et al., 2020 - https://arxiv.org/abs/1812.06127

FedAvgM:
    FedAvg with server-side momentum for faster convergence.
    Paper: Hsu et al., 2019 - https://arxiv.org/abs/1909.06335

=============================================================================
USAGE
=============================================================================

    from server.fl.algorithms import create_strategy, FedAvgStrategy
    
    # Option 1: Use factory function
    strategy = create_strategy(
        algorithm="fedavg",
        fraction_fit=1.0,
        initial_parameters=params,
    )
    
    # Option 2: Direct instantiation
    strategy = FedAvgStrategy(
        fraction_fit=1.0,
        initial_parameters=params,
        evaluate_fn=my_eval_fn,
    )

=============================================================================
IMPLEMENTATION NOTES
=============================================================================

All strategies are custom implementations that:
1. Inherit from flwr.server.strategy.Strategy
2. Implement explicit aggregation logic (no Flower built-ins)
3. Include detailed docstrings with algorithm explanations
4. Reference original papers and Flower examples

This makes the code:
- Easy to understand and debug
- Easy to extend with new algorithms
- Transparent about what's happening at each step
"""

from .strategies import (
    # Strategy classes
    FedAvgStrategy,
    FedProxStrategy,
    FedAvgMStrategy,
    FedXgbBaggingStrategy,
    # Factory function
    create_strategy,
    list_strategies,
    # Aggregation utilities
    aggregate_weighted_average,
    aggregate_weighted_loss,
    aggregate_metrics,
    # Registry
    STRATEGY_REGISTRY,
)

__all__ = [
    # Strategy classes
    "FedAvgStrategy",
    "FedProxStrategy", 
    "FedAvgMStrategy",
    "FedXgbBaggingStrategy",
    # Factory function
    "create_strategy",
    "list_strategies",
    # Aggregation utilities
    "aggregate_weighted_average",
    "aggregate_weighted_loss",
    "aggregate_metrics",
    # Registry
    "STRATEGY_REGISTRY",
]
