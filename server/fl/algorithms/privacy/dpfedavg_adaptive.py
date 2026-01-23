"""DPFedAvgAdaptive - Differential Privacy with Adaptive Gradient Clipping.

Paper: Andrew et al., 2021 - "Differentially Private Learning with Adaptive Clipping"
"""

from typing import Dict, Any

from flwr.server.strategy import FedAvg, DPFedAvgAdaptive, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class DPFedAvgAdaptiveWrapper(BaseStrategyWrapper):
    """DPFedAvgAdaptive strategy wrapper.
    
    DPFedAvgAdaptive provides differential privacy guarantees with
    adaptive gradient clipping that automatically adjusts the clip norm.
    """
    
    algorithm_id = FLAlgorithm.DPFEDAVG_ADAPTIVE
    flower_class = DPFedAvgAdaptive
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="DP-FedAvg (Adaptive Clipping)",
            description="Differential privacy with adaptive gradient clipping",
            paper="Andrew et al., 2021",
            category="privacy",
            params=["num_sampled_clients"],
            pros=["Privacy guarantees", "Automatic clip norm tuning"],
            cons=["Accuracy degradation", "Slower convergence"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create DPFedAvgAdaptive strategy."""
        server = config.server
        base_strategy = FedAvg(**common_params)
        return DPFedAvgAdaptive(
            strategy=base_strategy,
            num_sampled_clients=server.min_fit_clients,
        )
