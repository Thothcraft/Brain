
"""FedYogi - Adaptive FL with Controlled Adaptivity.

Paper: Reddi et al., 2021 - "Adaptive Federated Optimization"
"""

from typing import Dict, Any

from flwr.server.strategy import FedYogi, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedYogiWrapper(BaseStrategyWrapper):
    """FedYogi strategy wrapper.
    
    FedYogi uses Yogi optimizer on the server side, which provides
    controlled adaptivity that works better for non-convex optimization.
    """
    
    algorithm_id = FLAlgorithm.FEDYOGI
    flower_class = FedYogi
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedYogi",
            description="Adaptive FL with controlled adaptivity",
            paper="Reddi et al., 2021",
            category="standard",
            params=["server_lr", "beta_1", "beta_2", "tau"],
            pros=["Stable adaptivity", "Good for non-convex"],
            cons=["Complex implementation", "Tuning required"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedYogi strategy."""
        algo_params = config.algorithm_params
        client = config.client
        return FedYogi(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
