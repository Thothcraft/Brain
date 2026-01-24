"""FedXgbNnAvg - Federated XGBoost with Neural Network Averaging.

Flower's implementation of federated XGBoost combined with neural
network averaging.

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedXgbNnAvg.html
- Flower XGBoost Tutorial: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
- Parameters verified: evaluate_function (server-side evaluation), plus standard strategy params
- Note: Hybrid approach combining XGBoost trees with NN averaging; uses evaluate_function
"""

from typing import Dict, Any

from flwr.server.strategy import FedXgbNnAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedXgbNnAvgWrapper(BaseStrategyWrapper):
    """FedXgbNnAvg strategy wrapper.
    
    FedXgbNnAvg combines federated XGBoost with neural network
    averaging for hybrid model training.
    """
    
    algorithm_id = FLAlgorithm.FEDXGB_NNAVG
    flower_class = FedXgbNnAvg
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedXgbNnAvg",
            description="Federated XGBoost with neural network averaging",
            paper="Flower Team",
            category="xgboost",
            params=[],
            pros=["Combines XGBoost with NN", "Flexible"],
            cons=["More complex setup"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedXgbNnAvg strategy.
        
        Note: FedXgbNnAvg uses evaluate_function instead of evaluate_fn.
        """
        evaluate_function = common_params.pop("evaluate_fn", None)
        return FedXgbNnAvg(evaluate_function=evaluate_function, **common_params)
