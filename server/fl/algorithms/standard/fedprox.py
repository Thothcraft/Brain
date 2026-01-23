"""FedProx - Federated Optimization with Proximal Term.

Paper: Li et al., 2020 - "Federated Optimization in Heterogeneous Networks"
"""

from typing import Dict, Any

from flwr.server.strategy import FedProx, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedProxWrapper(BaseStrategyWrapper):
    """FedProx strategy wrapper.
    
    FedProx extends FedAvg with a proximal term that limits how far
    local models can deviate from the global model, improving convergence
    on heterogeneous (non-IID) data.
    """
    
    algorithm_id = FLAlgorithm.FEDPROX
    flower_class = FedProx
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedProx",
            description="FedAvg with proximal term for heterogeneous data",
            paper="Li et al., 2020",
            category="standard",
            params=["proximal_mu", "local_epochs", "learning_rate"],
            pros=["Better with non-IID data", "Handles stragglers"],
            cons=["Extra hyperparameter (mu)", "Slightly more computation"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedProx strategy."""
        algo_params = config.algorithm_params
        return FedProx(
            **common_params,
            proximal_mu=algo_params.proximal_mu,
        )
