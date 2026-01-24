"""FedXgbBagging - Federated XGBoost with Bagging Aggregation.

Flower's implementation of federated XGBoost using bagging for
ensemble aggregation.

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedXgbBagging.html
- Flower XGBoost Tutorial: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
- Parameters verified: evaluate_function (server-side evaluation), plus standard strategy params
- Note: Uses evaluate_function instead of evaluate_fn for XGBoost compatibility
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
        """Create FedXgbBagging strategy.
        
        Note: FedXgbBagging uses evaluate_function instead of evaluate_fn.
        """
        # Extract evaluate_fn and rename to evaluate_function for XGBoost strategy
        evaluate_function = common_params.pop("evaluate_fn", None)
        return FedXgbBagging(evaluate_function=evaluate_function, **common_params)
