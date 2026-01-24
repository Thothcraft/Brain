"""QFedAvg - Fair Federated Learning with q-Fair Aggregation.

Paper: Li et al., 2020 - "Fair Resource Allocation in Federated Learning"

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.QFedAvg.html
- Original Paper: https://arxiv.org/abs/1905.10497
- Parameters verified: q_param (fairness parameter, higher = more fairness),
  qffl_learning_rate (learning rate for q-FFL objective)
- Note: q=0 reduces to standard FedAvg; higher q gives more weight to clients with higher loss
"""

from typing import Dict, Any

from flwr.server.strategy import QFedAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm


@register_algorithm
class QFedAvgWrapper(BaseStrategyWrapper):
    """QFedAvg strategy wrapper.
    
    QFedAvg uses q-fair aggregation to ensure fairness across clients,
    giving more weight to clients with higher loss to reduce variance.
    """
    
    algorithm_id = FLAlgorithm.QFEDAVG
    flower_class = QFedAvg
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="q-FedAvg",
            description="Fair federated learning with q-fair aggregation",
            paper="Li et al., 2020",
            category="fair",
            params=["q_param", "qffl_learning_rate"],
            pros=["Fairness across clients", "Reduces variance"],
            cons=["May sacrifice average accuracy", "Extra computation"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create QFedAvg strategy."""
        algo_params = config.algorithm_params
        return QFedAvg(
            **common_params,
            q_param=algo_params.q_param,
            qffl_learning_rate=algo_params.server_learning_rate,
        )
