"""Bulyan - Byzantine-Robust Aggregation Combining Krum and Trimmed Mean.

Paper: Mhamdi et al., 2018 - "The Hidden Vulnerability of Distributed Learning
in Byzantium"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.Bulyan.html
- Original Paper: https://arxiv.org/abs/1802.07927
- Parameters verified: num_malicious_clients (expected number of Byzantine clients)
- Note: Requires n >= 4f + 3 where f is num_malicious_clients; combines Krum selection with trimmed mean
"""

from typing import Dict, Any

from flwr.server.strategy import Bulyan, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class BulyanWrapper(BaseStrategyWrapper):
    """Bulyan strategy wrapper.
    
    Bulyan combines Krum selection with trimmed mean aggregation,
    providing stronger Byzantine fault tolerance than either alone.
    """
    
    algorithm_id = FLAlgorithm.BULYAN
    flower_class = Bulyan
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Bulyan",
            description="Byzantine-robust aggregation combining Krum and trimmed mean",
            paper="Mhamdi et al., 2018",
            category="byzantine",
            params=["num_malicious_clients"],
            pros=["Stronger than Krum alone", "Handles more attacks"],
            cons=["Requires many honest clients", "Higher computation"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create Bulyan strategy."""
        algo_params = config.algorithm_params
        num_malicious = int(algo_params.byzantine_fraction * config.data.num_partitions)
        return Bulyan(
            **common_params,
            num_malicious_clients=num_malicious,
        )
