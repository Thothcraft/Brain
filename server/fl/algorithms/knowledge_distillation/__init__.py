"""Knowledge Distillation-based Federated Learning Algorithms.

These algorithms allow clients to have different model architectures by
communicating soft labels (logits) instead of model parameters.

Supported Algorithms:
- FedDF: Federated Distillation using ensemble distillation
- FedMD: Federated Model Distillation using public dataset
- FedGen: Federated Generative Learning (data-free distillation)

Reference Papers:
- FedDF: Lin et al., "Ensemble Distillation for Robust Model Fusion in FL" (2020)
- FedMD: Li & Wang, "FedMD: Heterogeneous FL via Model Distillation" (2019)
- FedGen: Zhu et al., "Data-Free Knowledge Distillation for Heterogeneous FL" (2021)
"""

from .feddf import FedDFStrategy
from .fedmd import FedMDStrategy
from .client import KDClient
from .strategy import create_kd_strategy

__all__ = [
    "FedDFStrategy",
    "FedMDStrategy",
    "KDClient",
    "create_kd_strategy",
]
