"""Factory function for creating Knowledge Distillation FL strategies."""

import logging
from typing import Optional, Callable

from flwr.common import Parameters
from flwr.server.strategy import Strategy

from ...core.config import FLAlgorithm, ExperimentConfig
from .feddf import FedDFStrategy
from .fedmd import FedMDStrategy

logger = logging.getLogger(__name__)


def create_kd_strategy(
    config: ExperimentConfig,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> Strategy:
    """Create a Knowledge Distillation FL strategy.
    
    Args:
        config: Experiment configuration
        initial_parameters: Initial parameters (soft labels for KD)
        evaluate_fn: Server-side evaluation function
    
    Returns:
        KD Strategy instance (FedDF, FedMD, or FedGen)
    """
    algorithm = config.algorithm
    algo_params = config.algorithm_params
    server = config.server
    data = config.data
    
    # Common parameters
    common_params = {
        "fraction_fit": server.fraction_fit,
        "fraction_evaluate": server.fraction_evaluate,
        "min_fit_clients": server.min_fit_clients,
        "min_evaluate_clients": server.min_evaluate_clients,
        "min_available_clients": server.min_available_clients,
        "evaluate_fn": evaluate_fn,
        "temperature": algo_params.temperature,
        "num_classes": 10,  # Will be overridden based on dataset
        "public_dataset_size": algo_params.public_dataset_size,
    }
    
    # Get num_classes from dataset
    from ...datasets import get_dataset_info
    dataset_info = get_dataset_info(data.dataset)
    common_params["num_classes"] = dataset_info.get("num_classes", 10)
    
    if algorithm == FLAlgorithm.FEDDF:
        return FedDFStrategy(
            **common_params,
            distillation_weight=algo_params.distillation_weight,
        )
    
    elif algorithm == FLAlgorithm.FEDMD:
        return FedMDStrategy(
            **common_params,
            digest_epochs=config.client.local_epochs,
            revisit_epochs=max(1, config.client.local_epochs // 2),
        )
    
    elif algorithm == FLAlgorithm.FEDGEN:
        # FedGen uses a generator - for now, fall back to FedDF
        logger.warning("FedGen not fully implemented, using FedDF instead")
        return FedDFStrategy(
            **common_params,
            distillation_weight=algo_params.distillation_weight,
        )
    
    else:
        logger.warning(f"Unknown KD algorithm {algorithm}, defaulting to FedDF")
        return FedDFStrategy(
            **common_params,
            distillation_weight=algo_params.distillation_weight,
        )
