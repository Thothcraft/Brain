"""FL Algorithms Module using Flower Strategies.

This module provides wrappers around Flower's built-in strategies and
custom implementations for knowledge distillation-based FL.

Supported Algorithms:
- Standard: FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, FedAvgM, FedOpt
- Byzantine-robust: FedMedian, FedTrimmedAvg, Krum, Bulyan
- Fair FL: QFedAvg
- Privacy-preserving: DPFedAvgAdaptive, DPFedAvgFixed
- Knowledge Distillation: FedDF, FedMD, FedGen (heterogeneous models)
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple

from flwr.common import Parameters, Metrics
from flwr.server.strategy import (
    Strategy,
    FedAvg,
    FedProx,
    FedAdam,
    FedYogi,
    FedAdagrad,
    FedAvgM,
    FedMedian,
    FedTrimmedAvg,
    FedOpt,
    Krum,
    Bulyan,
    QFedAvg,
    DPFedAvgAdaptive,
    DPFedAvgFixed,
)

from ..core.config import FLAlgorithm, ExperimentConfig

logger = logging.getLogger(__name__)


# Algorithm metadata for UI/documentation
ALGORITHM_REGISTRY = {
    FLAlgorithm.FEDAVG: {
        "name": "FedAvg",
        "description": "Federated Averaging - baseline FL algorithm",
        "paper": "McMahan et al., 2017",
        "supports_heterogeneous_models": False,
        "flower_class": FedAvg,
    },
    FLAlgorithm.FEDPROX: {
        "name": "FedProx",
        "description": "FedAvg with proximal term for heterogeneous data",
        "paper": "Li et al., 2020",
        "supports_heterogeneous_models": False,
        "flower_class": FedProx,
    },
    FLAlgorithm.FEDADAM: {
        "name": "FedAdam",
        "description": "Adaptive federated optimization with Adam",
        "paper": "Reddi et al., 2021",
        "supports_heterogeneous_models": False,
        "flower_class": FedAdam,
    },
    FLAlgorithm.FEDYOGI: {
        "name": "FedYogi",
        "description": "Adaptive FL with controlled adaptivity",
        "paper": "Reddi et al., 2021",
        "supports_heterogeneous_models": False,
        "flower_class": FedYogi,
    },
    FLAlgorithm.FEDADAGRAD: {
        "name": "FedAdagrad",
        "description": "Adaptive FL with Adagrad optimizer",
        "paper": "Reddi et al., 2021",
        "supports_heterogeneous_models": False,
        "flower_class": FedAdagrad,
    },
    FLAlgorithm.FEDAVGM: {
        "name": "FedAvgM",
        "description": "FedAvg with server-side momentum",
        "paper": "Hsu et al., 2019",
        "supports_heterogeneous_models": False,
        "flower_class": FedAvgM,
    },
    FLAlgorithm.FEDMEDIAN: {
        "name": "FedMedian",
        "description": "Byzantine-robust aggregation using coordinate-wise median",
        "paper": "Yin et al., 2018",
        "supports_heterogeneous_models": False,
        "flower_class": FedMedian,
    },
    FLAlgorithm.FEDTRIMMEDAVG: {
        "name": "FedTrimmedAvg",
        "description": "Byzantine-robust trimmed mean aggregation",
        "paper": "Yin et al., 2018",
        "supports_heterogeneous_models": False,
        "flower_class": FedTrimmedAvg,
    },
    FLAlgorithm.KRUM: {
        "name": "Krum",
        "description": "Byzantine-robust aggregation selecting closest updates",
        "paper": "Blanchard et al., 2017",
        "supports_heterogeneous_models": False,
        "flower_class": Krum,
    },
    FLAlgorithm.BULYAN: {
        "name": "Bulyan",
        "description": "Byzantine-robust aggregation combining Krum and trimmed mean",
        "paper": "Mhamdi et al., 2018",
        "supports_heterogeneous_models": False,
        "flower_class": Bulyan,
    },
    FLAlgorithm.QFEDAVG: {
        "name": "q-FedAvg",
        "description": "Fair federated learning with q-fair aggregation",
        "paper": "Li et al., 2020",
        "supports_heterogeneous_models": False,
        "flower_class": QFedAvg,
    },
    FLAlgorithm.DPFEDAVG_ADAPTIVE: {
        "name": "DP-FedAvg (Adaptive)",
        "description": "Differential privacy with adaptive gradient clipping",
        "paper": "Andrew et al., 2021",
        "supports_heterogeneous_models": False,
        "flower_class": DPFedAvgAdaptive,
    },
    FLAlgorithm.DPFEDAVG_FIXED: {
        "name": "DP-FedAvg (Fixed)",
        "description": "Differential privacy with fixed gradient clipping",
        "paper": "McMahan et al., 2018",
        "supports_heterogeneous_models": False,
        "flower_class": DPFedAvgFixed,
    },
    FLAlgorithm.FEDDF: {
        "name": "FedDF",
        "description": "Federated Distillation - supports heterogeneous model architectures",
        "paper": "Lin et al., 2020",
        "supports_heterogeneous_models": True,
        "flower_class": None,  # Custom implementation
    },
    FLAlgorithm.FEDMD: {
        "name": "FedMD",
        "description": "Federated Model Distillation - heterogeneous models via soft labels",
        "paper": "Li & Wang, 2019",
        "supports_heterogeneous_models": True,
        "flower_class": None,  # Custom implementation
    },
}


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics using weighted average."""
    if not metrics:
        return {}
    train_losses = [num * m.get("train_loss", 0) for num, m in metrics]
    examples = [num for num, _ in metrics]
    total = sum(examples)
    return {"train_loss": sum(train_losses) / total if total > 0 else 0.0}


def weighted_average_evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average."""
    if not metrics:
        return {}
    accuracies = [num * m.get("accuracy", 0) for num, m in metrics]
    examples = [num for num, _ in metrics]
    total = sum(examples)
    return {"accuracy": sum(accuracies) / total if total > 0 else 0.0}


def create_strategy(
    config: ExperimentConfig,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable] = None,
    on_evaluate_config_fn: Optional[Callable] = None,
) -> Strategy:
    """Create a Flower strategy based on the experiment configuration.
    
    Args:
        config: Experiment configuration
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        on_fit_config_fn: Function to configure client training
        on_evaluate_config_fn: Function to configure client evaluation
    
    Returns:
        Flower Strategy instance
    """
    algorithm = config.algorithm
    algo_params = config.algorithm_params
    server = config.server
    client = config.client
    
    # Default config functions
    if on_fit_config_fn is None:
        def on_fit_config_fn(server_round: int) -> Dict[str, Any]:
            return {
                "server_round": server_round,
                "local_epochs": client.local_epochs,
                "lr": client.learning_rate,
                "proximal_mu": algo_params.proximal_mu if algorithm == FLAlgorithm.FEDPROX else 0.0,
            }
    
    if on_evaluate_config_fn is None:
        def on_evaluate_config_fn(server_round: int) -> Dict[str, Any]:
            return {"server_round": server_round}
    
    # Common strategy parameters
    common_params = {
        "fraction_fit": server.fraction_fit,
        "fraction_evaluate": server.fraction_evaluate,
        "min_fit_clients": server.min_fit_clients,
        "min_evaluate_clients": server.min_evaluate_clients,
        "min_available_clients": server.min_available_clients,
        "fit_metrics_aggregation_fn": weighted_average_fit,
        "evaluate_metrics_aggregation_fn": weighted_average_evaluate,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
    }
    
    # Create strategy based on algorithm
    if algorithm == FLAlgorithm.FEDAVG:
        return FedAvg(**common_params)
    
    elif algorithm == FLAlgorithm.FEDPROX:
        return FedProx(**common_params, proximal_mu=algo_params.proximal_mu)
    
    elif algorithm == FLAlgorithm.FEDADAM:
        return FedAdam(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
    
    elif algorithm == FLAlgorithm.FEDYOGI:
        return FedYogi(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
    
    elif algorithm == FLAlgorithm.FEDADAGRAD:
        return FedAdagrad(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            tau=algo_params.tau,
        )
    
    elif algorithm == FLAlgorithm.FEDAVGM:
        return FedAvgM(**common_params, server_momentum=algo_params.server_momentum)
    
    elif algorithm == FLAlgorithm.FEDOPT:
        return FedOpt(
            **common_params,
            eta=algo_params.server_learning_rate,
            eta_l=client.learning_rate,
            beta_1=algo_params.beta_1,
            beta_2=algo_params.beta_2,
            tau=algo_params.tau,
        )
    
    elif algorithm == FLAlgorithm.FEDMEDIAN:
        return FedMedian(**common_params)
    
    elif algorithm == FLAlgorithm.FEDTRIMMEDAVG:
        return FedTrimmedAvg(**common_params, beta=algo_params.trimmed_mean_beta)
    
    elif algorithm == FLAlgorithm.KRUM:
        num_malicious = int(algo_params.byzantine_fraction * config.data.num_partitions)
        return Krum(
            **common_params,
            num_malicious_clients=num_malicious,
            num_clients_to_keep=algo_params.krum_num_closest,
        )
    
    elif algorithm == FLAlgorithm.BULYAN:
        num_malicious = int(algo_params.byzantine_fraction * config.data.num_partitions)
        return Bulyan(**common_params, num_malicious_clients=num_malicious)
    
    elif algorithm == FLAlgorithm.QFEDAVG:
        return QFedAvg(
            **common_params,
            q_param=algo_params.q_param,
            qffl_learning_rate=algo_params.server_learning_rate,
        )
    
    elif algorithm == FLAlgorithm.DPFEDAVG_ADAPTIVE:
        base_strategy = FedAvg(**common_params)
        return DPFedAvgAdaptive(
            strategy=base_strategy,
            num_sampled_clients=server.min_fit_clients,
        )
    
    elif algorithm == FLAlgorithm.DPFEDAVG_FIXED:
        base_strategy = FedAvg(**common_params)
        return DPFedAvgFixed(
            strategy=base_strategy,
            num_sampled_clients=server.min_fit_clients,
            clip_norm=config.privacy.max_grad_norm,
            noise_multiplier=config.privacy.noise_multiplier,
        )
    
    elif algorithm in [FLAlgorithm.FEDDF, FLAlgorithm.FEDMD, FLAlgorithm.FEDGEN]:
        # Knowledge distillation strategies - import from submodule
        from .knowledge_distillation import create_kd_strategy
        return create_kd_strategy(config, initial_parameters, evaluate_fn)
    
    else:
        logger.warning(f"Unknown algorithm {algorithm}, defaulting to FedAvg")
        return FedAvg(**common_params)


def list_algorithms() -> List[Dict[str, Any]]:
    """List all available FL algorithms with metadata."""
    return [
        {
            "id": algo.value,
            "name": info["name"],
            "description": info["description"],
            "paper": info.get("paper", ""),
            "supports_heterogeneous_models": info["supports_heterogeneous_models"],
        }
        for algo, info in ALGORITHM_REGISTRY.items()
    ]


def get_algorithm_info(algorithm) -> Dict[str, Any]:
    """Get detailed information about an FL algorithm.
    
    Args:
        algorithm: FLAlgorithm enum value or string
    
    Returns:
        Dictionary with algorithm details
    """
    from ..core.config import FLAlgorithm
    
    if isinstance(algorithm, str):
        algorithm = FLAlgorithm(algorithm.lower())
    
    algorithm_info = {
        FLAlgorithm.FEDAVG: {
            "name": "Federated Averaging (FedAvg)",
            "description": "Standard federated learning with weighted averaging",
            "paper": "McMahan et al., 2017",
            "params": ["local_epochs", "learning_rate", "batch_size"],
            "pros": ["Simple", "Effective baseline", "Low communication"],
            "cons": ["Struggles with non-IID data", "No adaptivity"],
            "flower_class": "flwr.server.strategy.FedAvg"
        },
        FLAlgorithm.FEDPROX: {
            "name": "FedProx",
            "description": "FedAvg with proximal term for heterogeneous data",
            "paper": "Li et al., 2020",
            "params": ["proximal_mu", "local_epochs", "learning_rate"],
            "pros": ["Better with non-IID data", "Handles stragglers"],
            "cons": ["Extra hyperparameter (mu)", "Slightly more computation"],
            "flower_class": "flwr.server.strategy.FedProx"
        },
        FLAlgorithm.FEDADAM: {
            "name": "FedAdam",
            "description": "Adaptive federated optimization with Adam",
            "paper": "Reddi et al., 2021",
            "params": ["server_lr", "beta_1", "beta_2", "tau"],
            "pros": ["Adaptive learning rate", "Faster convergence"],
            "cons": ["More hyperparameters", "Higher memory on server"],
            "flower_class": "flwr.server.strategy.FedAdam"
        },
        FLAlgorithm.FEDYOGI: {
            "name": "FedYogi",
            "description": "Adaptive FL with controlled adaptivity",
            "paper": "Reddi et al., 2021",
            "params": ["server_lr", "beta_1", "beta_2", "tau"],
            "pros": ["Stable adaptivity", "Good for non-convex"],
            "cons": ["Complex implementation", "Tuning required"],
            "flower_class": "flwr.server.strategy.FedYogi"
        },
        FLAlgorithm.FEDADAGRAD: {
            "name": "FedAdagrad",
            "description": "Adaptive FL with Adagrad optimizer",
            "paper": "Reddi et al., 2021",
            "params": ["server_lr", "tau"],
            "pros": ["Simple adaptivity", "Good for sparse gradients"],
            "cons": ["Learning rate decay", "May slow down"],
            "flower_class": "flwr.server.strategy.FedAdagrad"
        },
        FLAlgorithm.FEDAVGM: {
            "name": "FedAvgM",
            "description": "FedAvg with server-side momentum",
            "paper": "Hsu et al., 2019",
            "params": ["server_momentum"],
            "pros": ["Faster convergence", "Simple extension"],
            "cons": ["Extra hyperparameter"],
            "flower_class": "flwr.server.strategy.FedAvgM"
        },
        FLAlgorithm.FEDOPT: {
            "name": "FedOpt",
            "description": "Generalized federated optimization framework",
            "paper": "Reddi et al., 2021",
            "params": ["server_lr", "beta_1", "beta_2", "tau"],
            "pros": ["Flexible", "Supports multiple optimizers"],
            "cons": ["Many hyperparameters"],
            "flower_class": "flwr.server.strategy.FedOpt"
        },
        FLAlgorithm.FEDMEDIAN: {
            "name": "FedMedian",
            "description": "Byzantine-robust aggregation using coordinate-wise median",
            "paper": "Yin et al., 2018",
            "params": [],
            "pros": ["Byzantine-robust", "No extra hyperparameters"],
            "cons": ["Higher computation", "May be biased"],
            "flower_class": "flwr.server.strategy.FedMedian"
        },
        FLAlgorithm.FEDTRIMMEDAVG: {
            "name": "FedTrimmedAvg",
            "description": "Byzantine-robust trimmed mean aggregation",
            "paper": "Yin et al., 2018",
            "params": ["beta"],
            "pros": ["Byzantine-robust", "Configurable trimming"],
            "cons": ["Requires knowing fraction of Byzantine clients"],
            "flower_class": "flwr.server.strategy.FedTrimmedAvg"
        },
        FLAlgorithm.KRUM: {
            "name": "Krum",
            "description": "Byzantine-robust aggregation selecting closest updates",
            "paper": "Blanchard et al., 2017",
            "params": ["num_malicious_clients", "num_clients_to_keep"],
            "pros": ["Strong Byzantine guarantees", "Theoretical bounds"],
            "cons": ["Requires knowing number of Byzantine clients"],
            "flower_class": "flwr.server.strategy.Krum"
        },
        FLAlgorithm.BULYAN: {
            "name": "Bulyan",
            "description": "Byzantine-robust aggregation combining Krum and trimmed mean",
            "paper": "Mhamdi et al., 2018",
            "params": ["num_malicious_clients"],
            "pros": ["Stronger than Krum alone", "Handles more attacks"],
            "cons": ["Requires many honest clients", "Higher computation"],
            "flower_class": "flwr.server.strategy.Bulyan"
        },
        FLAlgorithm.QFEDAVG: {
            "name": "q-FedAvg",
            "description": "Fair federated learning with q-fair aggregation",
            "paper": "Li et al., 2020",
            "params": ["q_param", "qffl_learning_rate"],
            "pros": ["Fairness across clients", "Reduces variance"],
            "cons": ["May sacrifice average accuracy", "Extra computation"],
            "flower_class": "flwr.server.strategy.QFedAvg"
        },
        FLAlgorithm.DPFEDAVG_ADAPTIVE: {
            "name": "DP-FedAvg (Adaptive Clipping)",
            "description": "Differential privacy with adaptive gradient clipping",
            "paper": "Andrew et al., 2021",
            "params": ["num_sampled_clients"],
            "pros": ["Privacy guarantees", "Automatic clip norm tuning"],
            "cons": ["Accuracy degradation", "Slower convergence"],
            "flower_class": "flwr.server.strategy.DPFedAvgAdaptive"
        },
        FLAlgorithm.DPFEDAVG_FIXED: {
            "name": "DP-FedAvg (Fixed Clipping)",
            "description": "Differential privacy with fixed gradient clipping",
            "paper": "McMahan et al., 2018",
            "params": ["clip_norm", "noise_multiplier", "num_sampled_clients"],
            "pros": ["Privacy guarantees", "Predictable privacy budget"],
            "cons": ["Requires tuning clip norm", "Accuracy degradation"],
            "flower_class": "flwr.server.strategy.DPFedAvgFixed"
        },
        FLAlgorithm.FEDDF: {
            "name": "FedDF (Federated Distillation)",
            "description": "Knowledge distillation for heterogeneous model architectures",
            "paper": "Lin et al., 2020",
            "params": ["temperature", "distillation_weight", "public_dataset_size"],
            "pros": ["Supports different model architectures", "Privacy-friendly"],
            "cons": ["Requires public dataset", "More complex"],
            "flower_class": "custom"
        },
        FLAlgorithm.FEDMD: {
            "name": "FedMD (Federated Model Distillation)",
            "description": "Model distillation via consensus on public dataset",
            "paper": "Li & Wang, 2019",
            "params": ["temperature", "digest_epochs", "revisit_epochs"],
            "pros": ["Heterogeneous models", "Communication efficient"],
            "cons": ["Requires public dataset"],
            "flower_class": "custom"
        },
    }
    
    return algorithm_info.get(algorithm, {"name": algorithm.value, "description": "FL algorithm"})


__all__ = [
    "create_strategy",
    "ALGORITHM_REGISTRY",
    "list_algorithms",
    "get_algorithm_info",
    "FLAlgorithm",
]
