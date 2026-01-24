"""FedAvgM - FedAvg with Server-Side Momentum.

Paper: Hsu et al., 2019 - "Measuring the Effects of Non-Identical Data Distribution
for Federated Visual Classification"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvgM.html
- Original Paper: https://arxiv.org/abs/1909.06335
- Parameters verified: server_momentum (momentum factor for server-side updates, default 0.0)
- Note: Extends FedAvg with momentum to accelerate convergence
"""

from typing import Dict, Any

from flwr.server.strategy import FedAvgM, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedAvgMWrapper(BaseStrategyWrapper):
    """FedAvgM strategy wrapper.
    
    FedAvgM extends FedAvg with server-side momentum to accelerate
    convergence and improve stability.
    """
    
    algorithm_id = FLAlgorithm.FEDAVGM
    flower_class = FedAvgM
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedAvgM",
            description="FedAvg with server-side momentum",
            paper="Hsu et al., 2019",
            category="standard",
            params=["server_momentum"],
            pros=["Faster convergence", "Simple extension"],
            cons=["Extra hyperparameter"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAvgM strategy."""
        algo_params = config.algorithm_params
        return FedAvgM(
            **common_params,
            server_momentum=algo_params.server_momentum,
        )
