"""FedXgbCyclic - Federated XGBoost with Cyclic Training.

Flower's implementation of federated XGBoost using cyclic training
across clients.
"""

from typing import Dict, Any

from flwr.server.strategy import FedXgbCyclic, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedXgbCyclicWrapper(BaseStrategyWrapper):
    """FedXgbCyclic strategy wrapper.
    
    FedXgbCyclic enables federated learning with XGBoost using
    cyclic training where clients train sequentially.
    """
    
    algorithm_id = FLAlgorithm.FEDXGB_CYCLIC
    flower_class = FedXgbCyclic
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedXgbCyclic",
            description="Federated XGBoost with cyclic training across clients",
            paper="Flower Team",
            category="xgboost",
            params=[],
            pros=["Sequential tree building", "Communication efficient"],
            cons=["Slower convergence", "Order-dependent"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedXgbCyclic strategy."""
        return FedXgbCyclic(**common_params)
