"""FaultTolerantFedAvg - FedAvg with Fault Tolerance for Client Failures.

Flower's implementation of FedAvg with fault tolerance for handling
client failures and stragglers.
"""

from typing import Dict, Any

from flwr.server.strategy import FaultTolerantFedAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FaultTolerantFedAvgWrapper(BaseStrategyWrapper):
    """FaultTolerantFedAvg strategy wrapper.
    
    FaultTolerantFedAvg extends FedAvg with fault tolerance mechanisms
    to handle client failures and stragglers gracefully.
    """
    
    algorithm_id = FLAlgorithm.FAULT_TOLERANT_FEDAVG
    flower_class = FaultTolerantFedAvg
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Fault-Tolerant FedAvg",
            description="FedAvg with fault tolerance for client failures and stragglers",
            paper="Flower Team",
            category="fault_tolerant",
            params=["min_completion_rate_fit", "min_completion_rate_evaluate"],
            pros=["Handles client failures", "Robust to stragglers"],
            cons=["May have fewer updates per round"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FaultTolerantFedAvg strategy."""
        return FaultTolerantFedAvg(
            **common_params,
            min_completion_rate_fit=0.5,
            min_completion_rate_evaluate=0.5,
        )
