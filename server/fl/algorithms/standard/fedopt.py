"""FedOpt - Generalized Federated Optimization Framework.

Paper: Reddi et al., 2021 - "Adaptive Federated Optimization"
"""

from typing import Dict, Any

from flwr.server.strategy import FedOpt, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedOptWrapper(BaseStrategyWrapper):
    """FedOpt strategy wrapper.
    
    FedOpt is a generalized federated optimization framework that
    supports various server-side optimizers.
    """
    
    algorithm_id = FLAlgorithm.FEDOPT
    flower_class = FedOpt
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedOpt",
            description="Generalized federated optimization framework",
            paper="Reddi et al., 2021",
            category="standard",
            params=["server_lr", "beta_1", "beta_2", "tau"],
            pros=["Flexible", "Supports multiple optimizers"],
            cons=["Many hyperparameters"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedOpt strategy."""
        algo_params = config.algorithm_params
        client = config.client
        return FedOpt(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
