"""Fair FL Algorithms.

This module contains fair federated learning algorithms:
- QFedAvg: q-fair aggregation for fairness across clients
"""

from .qfedavg import QFedAvgWrapper

__all__ = [
    "QFedAvgWrapper",
]
