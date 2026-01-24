"""FedMedian - Byzantine-Robust Coordinate-wise Median Aggregation.

Paper: Yin et al., 2018 - "Byzantine-Robust Distributed Learning: Towards Optimal
Statistical Rates"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedMedian.html
- Original Paper: https://arxiv.org/abs/1803.01498
- Parameters verified: Uses standard FedAvg parameters (no additional params needed)
- Note: Computes coordinate-wise median instead of mean for Byzantine robustness
"""

from typing import Dict, Any

from flwr.server.strategy import FedMedian, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class FedMedianWrapper(BaseStrategyWrapper):
    """FedMedian strategy wrapper.
    
    FedMedian uses coordinate-wise median for aggregation, providing
    Byzantine fault tolerance against malicious clients.
    """
    
    algorithm_id = FLAlgorithm.FEDMEDIAN
    flower_class = FedMedian
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedMedian",
            description="Byzantine-robust aggregation using coordinate-wise median",
            paper="Yin et al., 2018",
            category="byzantine",
            params=[],
            pros=["Byzantine-robust", "No extra hyperparameters"],
            cons=["Higher computation", "May be biased"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedMedian strategy."""
        return FedMedian(**common_params)
