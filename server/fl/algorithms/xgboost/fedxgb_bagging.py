"""FedXgbBagging - Federated XGBoost with Bagging Aggregation.

Flower's implementation of federated XGBoost using bagging for
ensemble aggregation.
"""

from typing import Dict, Any

from flwr.server.strategy import FedXgbBagging, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedXgbBaggingWrapper(BaseStrategyWrapper):
    """FedXgbBagging strategy wrapper.
    
    FedXgbBagging enables federated learning with XGBoost using
    bagging aggregation for tree ensembles.
    """
    
    algorithm_id = FLAlgorithm.FEDXGB_BAGGING
    flower_class = FedXgbBagging
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedXgbBagging",
            description="Federated XGBoost with bagging aggregation for tree ensembles",
            paper="Flower Team",
            category="xgboost",
            params=[],
            pros=["Works with XGBoost", "Ensemble diversity", "Non-neural models"],
            cons=["Requires XGBoost setup", "Tree-specific"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedXgbBagging strategy."""
        return FedXgbBagging(**common_params)
