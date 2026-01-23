"""DPFedAvgFixed - Differential Privacy with Fixed Gradient Clipping.

Paper: McMahan et al., 2018 - "Learning Differentially Private Recurrent
Language Models"
"""

from typing import Dict, Any

from flwr.server.strategy import FedAvg, DPFedAvgFixed, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class DPFedAvgFixedWrapper(BaseStrategyWrapper):
    """DPFedAvgFixed strategy wrapper.
    
    DPFedAvgFixed provides differential privacy guarantees with
    fixed gradient clipping and noise addition.
    """
    
    algorithm_id = FLAlgorithm.DPFEDAVG_FIXED
    flower_class = DPFedAvgFixed
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="DP-FedAvg (Fixed Clipping)",
            description="Differential privacy with fixed gradient clipping",
            paper="McMahan et al., 2018",
            category="privacy",
            params=["clip_norm", "noise_multiplier", "num_sampled_clients"],
            pros=["Privacy guarantees", "Predictable privacy budget"],
            cons=["Requires tuning clip norm", "Accuracy degradation"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create DPFedAvgFixed strategy."""
        server = config.server
        privacy = config.privacy
        base_strategy = FedAvg(**common_params)
        return DPFedAvgFixed(
            strategy=base_strategy,
            num_sampled_clients=server.min_fit_clients,
            clip_norm=privacy.max_grad_norm,
            noise_multiplier=privacy.noise_multiplier,
        )
