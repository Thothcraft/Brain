"""Mobile/Edge FL Algorithms.

This module contains mobile and edge-optimized federated learning algorithms:
- FedAvgAndroid: FedAvg optimized for Android devices
"""

from .fedavg_android import FedAvgAndroidWrapper

__all__ = [
    "FedAvgAndroidWrapper",
]
