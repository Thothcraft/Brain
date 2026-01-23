"""FedAdagrad - Adaptive FL with Adagrad Optimizer.

Paper: Reddi et al., 2021 - "Adaptive Federated Optimization"
"""

from typing import Dict, Any

from flwr.server.strategy import FedAdagrad, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedAdagradWrapper(BaseStrategyWrapper):
    """FedAdagrad strategy wrapper.
    
    FedAdagrad uses Adagrad optimizer on the server side, which adapts
    learning rates based on historical gradient information.
    """
    
    algorithm_id = FLAlgorithm.FEDADAGRAD
    flower_class = FedAdagrad
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedAdagrad",
            description="Adaptive FL with Adagrad optimizer",
            paper="Reddi et al., 2021",
            category="standard",
            params=["server_lr", "tau"],
            pros=["Simple adaptivity", "Good for sparse gradients"],
            cons=["Learning rate decay", "May slow down"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAdagrad strategy."""
        algo_params = config.algorithm_params
        client = config.client
        return FedAdagrad(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            tau=algo_params.tau,
        )
