"""FedTrimmedAvg - Byzantine-Robust Trimmed Mean Aggregation.

Paper: Yin et al., 2018 - "Byzantine-Robust Distributed Learning: Towards Optimal
Statistical Rates"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedTrimmedAvg.html
- Original Paper: https://arxiv.org/abs/1803.01498
- Parameters verified: beta (fraction of extreme values to trim from each end, default 0.2)
- Note: Trims beta fraction of highest and lowest values before averaging
"""

from typing import Dict, Any

from flwr.server.strategy import FedTrimmedAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedTrimmedAvgWrapper(BaseStrategyWrapper):
    """FedTrimmedAvg strategy wrapper.
    
    FedTrimmedAvg uses trimmed mean for aggregation, discarding extreme
    values to provide Byzantine fault tolerance.
    """
    
    algorithm_id = FLAlgorithm.FEDTRIMMEDAVG
    flower_class = FedTrimmedAvg
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedTrimmedAvg",
            description="Byzantine-robust trimmed mean aggregation",
            paper="Yin et al., 2018",
            category="byzantine",
            params=["beta"],
            pros=["Byzantine-robust", "Configurable trimming"],
            cons=["Requires knowing fraction of Byzantine clients"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedTrimmedAvg strategy."""
        algo_params = config.algorithm_params
        return FedTrimmedAvg(
            **common_params,
            beta=algo_params.trimmed_mean_beta,
        )
