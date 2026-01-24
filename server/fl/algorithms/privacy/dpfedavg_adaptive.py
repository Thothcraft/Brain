"""DPFedAvgAdaptive - Differential Privacy with Adaptive Gradient Clipping.

Paper: Andrew et al., 2021 - "Differentially Private Learning with Adaptive Clipping"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.DPFedAvgAdaptive.html
- Original Paper: https://arxiv.org/abs/1905.03871
- Parameters verified: strategy (base strategy), num_sampled_clients,
  init_clip_norm (initial clipping threshold), noise_multiplier, target_clipped_quantile
- Note: Deprecated in Flower; consider DifferentialPrivacyServerSideAdaptiveClipping
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
        """Create DPFedAvgAdaptive strategy.
        
        Note: DPFedAvgAdaptive is deprecated in Flower and will be removed.
        Consider using DifferentialPrivacyServerSideAdaptiveClipping instead.
        """
        server = config.server
        privacy = config.privacy
        base_strategy = FedAvg(**common_params)
        return DPFedAvgAdaptive(
            strategy=base_strategy,
            num_sampled_clients=server.min_fit_clients,
            init_clip_norm=privacy.max_grad_norm,
            noise_multiplier=privacy.noise_multiplier,
        )
