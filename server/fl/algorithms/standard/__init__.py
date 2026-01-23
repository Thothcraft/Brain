"""Standard FL Algorithms.

This module contains standard federated learning algorithms:
- FedAvg: Federated Averaging
- FedProx: FedAvg with proximal term
- FedAdam: Adaptive federated optimization with Adam
- FedYogi: Adaptive FL with controlled adaptivity
- FedAdagrad: Adaptive FL with Adagrad
- FedAvgM: FedAvg with server-side momentum
- FedOpt: Generalized federated optimization
"""

from .fedavg import FedAvgWrapper
from .fedprox import FedProxWrapper
from .fedadam import FedAdamWrapper
from .fedyogi import FedYogiWrapper
from .fedadagrad import FedAdagradWrapper
from .fedavgm import FedAvgMWrapper
from .fedopt import FedOptWrapper

__all__ = [
    "FedAvgWrapper",
    "FedProxWrapper",
    "FedAdamWrapper",
    "FedYogiWrapper",
    "FedAdagradWrapper",
    "FedAvgMWrapper",
    "FedOptWrapper",
]
