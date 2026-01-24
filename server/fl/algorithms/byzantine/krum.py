"""Krum - Byzantine-Robust Aggregation Selecting Closest Updates.

Paper: Blanchard et al., 2017 - "Machine Learning with Adversaries: Byzantine
Tolerant Gradient Descent"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.Krum.html
- Original Paper: https://arxiv.org/abs/1703.02757
- Parameters verified: num_malicious_clients (expected number of Byzantine clients),
  num_clients_to_keep (number of closest updates to select, 1 for single Krum)
- Note: Requires n >= 2f + 3 where f is num_malicious_clients
"""

from typing import Dict, Any

from flwr.server.strategy import Krum, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class KrumWrapper(BaseStrategyWrapper):
    """Krum strategy wrapper.
    
    Krum selects the update that is closest to its neighbors, providing
    strong Byzantine fault tolerance guarantees.
    """
    
    algorithm_id = FLAlgorithm.KRUM
    flower_class = Krum
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Krum",
            description="Byzantine-robust aggregation selecting closest updates",
            paper="Blanchard et al., 2017",
            category="byzantine",
            params=["num_malicious_clients", "num_clients_to_keep"],
            pros=["Strong Byzantine guarantees", "Theoretical bounds"],
            cons=["Requires knowing number of Byzantine clients"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create Krum strategy."""
        algo_params = config.algorithm_params
        num_malicious = int(algo_params.byzantine_fraction * config.data.num_partitions)
        return Krum(
            **common_params,
            num_malicious_clients=num_malicious,
            num_clients_to_keep=algo_params.krum_num_closest,
        )
