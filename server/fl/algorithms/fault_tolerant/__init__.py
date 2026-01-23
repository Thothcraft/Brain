"""Fault-Tolerant FL Algorithms.

This module contains fault-tolerant federated learning algorithms:
- FaultTolerantFedAvg: FedAvg with fault tolerance for client failures
"""

from .fault_tolerant_fedavg import FaultTolerantFedAvgWrapper

__all__ = [
    "FaultTolerantFedAvgWrapper",
]
