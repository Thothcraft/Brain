"""FedAdam - Adaptive Federated Optimization with Adam.

Paper: Reddi et al., 2021 - "Adaptive Federated Optimization"
"""

from typing import Dict, Any

from flwr.server.strategy import FedAdam, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedAdamWrapper(BaseStrategyWrapper):
    """FedAdam strategy wrapper.
    
    FedAdam applies Adam optimizer on the server side for adaptive
    learning rate adjustment during federated aggregation.
    """
    
    algorithm_id = FLAlgorithm.FEDADAM
    flower_class = FedAdam
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedAdam",
            description="Adaptive federated optimization with Adam",
            paper="Reddi et al., 2021",
            category="standard",
            params=["server_lr", "beta_1", "beta_2", "tau"],
            pros=["Adaptive learning rate", "Faster convergence"],
            cons=["More hyperparameters", "Higher memory on server"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAdam strategy."""
        algo_params = config.algorithm_params
        client = config.client
        return FedAdam(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
