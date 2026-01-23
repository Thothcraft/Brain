"""MultiKrum - Byzantine-Robust Aggregation Selecting Multiple Closest Updates.

Paper: Blanchard et al., 2017 - "Machine Learning with Adversaries: Byzantine
Tolerant Gradient Descent"
"""

import logging
from typing import Dict, Any, Optional, Type

from flwr.server.strategy import Strategy, Krum

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm

logger = logging.getLogger(__name__)

# Try to import MultiKrum (may not be available in all Flower versions)
try:
    from flwr.server.strategy import MultiKrum
    HAS_MULTIKRUM = True
except ImportError:
    MultiKrum = None
    HAS_MULTIKRUM = False


@register_algorithm
class MultiKrumWrapper(BaseStrategyWrapper):
    """MultiKrum strategy wrapper.
    
    MultiKrum extends Krum by selecting multiple closest updates and
    averaging them, providing better accuracy while maintaining
    Byzantine fault tolerance.
    """
    
    algorithm_id = FLAlgorithm.MULTIKRUM
    flower_class = MultiKrum if HAS_MULTIKRUM else None
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Multi-Krum",
            description="Byzantine-robust aggregation selecting multiple closest updates",
            paper="Blanchard et al., 2017",
            category="byzantine",
            params=["num_malicious_clients", "num_clients_to_keep"],
            pros=["More robust than single Krum", "Better accuracy"],
            cons=["Requires knowing number of Byzantine clients"],
        )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if MultiKrum is available."""
        return HAS_MULTIKRUM
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create MultiKrum strategy (falls back to Krum if unavailable)."""
        algo_params = config.algorithm_params
        num_malicious = int(algo_params.byzantine_fraction * config.data.num_partitions)
        
        if not HAS_MULTIKRUM:
            logger.warning("MultiKrum not available, falling back to Krum")
            return Krum(
                **common_params,
                num_malicious_clients=num_malicious,
                num_clients_to_keep=algo_params.krum_num_closest,
            )
        
        return MultiKrum(
            **common_params,
            num_malicious_clients=num_malicious,
            num_clients_to_keep=algo_params.krum_num_closest,
        )
