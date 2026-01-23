"""FedAvg - Federated Averaging Algorithm.

Paper: McMahan et al., 2017 - "Communication-Efficient Learning of Deep Networks
from Decentralized Data"
"""

from typing import Dict, Any

from flwr.server.strategy import FedAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedAvgWrapper(BaseStrategyWrapper):
    """Federated Averaging (FedAvg) strategy wrapper.
    
    FedAvg is the baseline federated learning algorithm that performs
    weighted averaging of client model updates based on the number of
    training examples each client has.
    """
    
    algorithm_id = FLAlgorithm.FEDAVG
    flower_class = FedAvg
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Federated Averaging (FedAvg)",
            description="Standard federated learning with weighted averaging",
            paper="McMahan et al., 2017",
            category="standard",
            params=["local_epochs", "learning_rate", "batch_size"],
            pros=["Simple", "Effective baseline", "Low communication"],
            cons=["Struggles with non-IID data", "No adaptivity"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAvg strategy."""
        return FedAvg(**common_params)
