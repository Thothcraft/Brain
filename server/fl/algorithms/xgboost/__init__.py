"""XGBoost FL Algorithms.

This module contains XGBoost-based federated learning algorithms:
- FedXgbBagging: Federated XGBoost with bagging aggregation
- FedXgbCyclic: Federated XGBoost with cyclic training
- FedXgbNnAvg: Federated XGBoost with neural network averaging
"""

from .fedxgb_bagging import FedXgbBaggingWrapper
from .fedxgb_cyclic import FedXgbCyclicWrapper
from .fedxgb_nnavg import FedXgbNnAvgWrapper

__all__ = [
    "FedXgbBaggingWrapper",
    "FedXgbCyclicWrapper",
    "FedXgbNnAvgWrapper",
]
