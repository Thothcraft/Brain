"""Default FL Pipelines and Experiment Creation.

This module provides pre-configured FL experiment pipelines that are
easy to use and extend. Pipelines are hardcoded defaults that can be
modified programmatically.
"""

import logging
from typing import Dict, Any, List, Optional
from copy import deepcopy

from ..core.config import (
    ExperimentConfig,
    FLAlgorithm,
    ModelArchitecture,
    ServerConfig,
    ClientConfig,
    AlgorithmConfig,
    DataConfig,
    PrivacyConfig,
    MonitoringConfig,
    FLDataset,
    PartitionStrategy,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DEFAULT PIPELINES
# ============================================================================

DEFAULT_PIPELINES = {
    # Quick test pipeline (fast, for debugging)
    "quick_test": {
        "name": "Quick Test",
        "description": "Fast pipeline for testing (5 rounds, 3 clients)",
        "config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "model": ModelArchitecture.CNN,
            "server": {"num_rounds": 5, "min_fit_clients": 2, "fraction_fit": 1.0},
            "client": {"local_epochs": 1, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 3, "partition_strategy": PartitionStrategy.IID},
        },
    },
    
    # Standard CIFAR-10 benchmark
    "cifar10_fedavg": {
        "name": "CIFAR-10 FedAvg Benchmark",
        "description": "Standard FedAvg on CIFAR-10 with 10 clients",
        "config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 100, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
        },
    },
    
    # Non-IID benchmark with FedProx
    "cifar10_noniid_fedprox": {
        "name": "CIFAR-10 Non-IID FedProx",
        "description": "FedProx on non-IID CIFAR-10 (Dirichlet alpha=0.5)",
        "config": {
            "algorithm": FLAlgorithm.FEDPROX,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 100, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {
                "dataset": FLDataset.CIFAR10,
                "num_partitions": 10,
                "partition_strategy": PartitionStrategy.NON_IID_DIRICHLET,
                "dirichlet_alpha": 0.5,
            },
            "algorithm_params": {"proximal_mu": 0.01},
        },
    },
    
    # Adaptive optimization with FedAdam
    "cifar10_fedadam": {
        "name": "CIFAR-10 FedAdam",
        "description": "Adaptive FL with FedAdam optimizer",
        "config": {
            "algorithm": FLAlgorithm.FEDADAM,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 100, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
            "algorithm_params": {"server_learning_rate": 0.1, "beta_1": 0.9, "beta_2": 0.99, "tau": 1e-3},
        },
    },
    
    # MNIST quick benchmark
    "mnist_fedavg": {
        "name": "MNIST FedAvg",
        "description": "FedAvg on MNIST with simple CNN",
        "config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "model": ModelArchitecture.CNN,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 3, "learning_rate": 0.01, "local_batch_size": 64},
            "data": {"dataset": FLDataset.MNIST, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
        },
    },
    
    # Fashion-MNIST non-IID
    "fmnist_noniid": {
        "name": "Fashion-MNIST Non-IID",
        "description": "FedAvg on non-IID Fashion-MNIST",
        "config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "model": ModelArchitecture.CNN,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 3, "learning_rate": 0.01, "local_batch_size": 64},
            "data": {
                "dataset": FLDataset.FASHION_MNIST,
                "num_partitions": 10,
                "partition_strategy": PartitionStrategy.NON_IID_DIRICHLET,
                "dirichlet_alpha": 0.3,
            },
        },
    },
    
    # Byzantine-robust with Krum
    "cifar10_krum": {
        "name": "CIFAR-10 Krum (Byzantine-robust)",
        "description": "Byzantine-robust FL with Krum aggregation",
        "config": {
            "algorithm": FLAlgorithm.KRUM,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 100, "min_fit_clients": 8, "fraction_fit": 0.8},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
            "algorithm_params": {"byzantine_fraction": 0.2, "krum_num_closest": 2},
        },
    },
    
    # Knowledge Distillation with FedDF (heterogeneous models)
    "cifar10_feddf": {
        "name": "CIFAR-10 FedDF (Heterogeneous)",
        "description": "Knowledge distillation FL allowing different model architectures",
        "config": {
            "algorithm": FLAlgorithm.FEDDF,
            "model": ModelArchitecture.CNN,  # Default, but clients can have different models
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 1.0},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
            "algorithm_params": {"temperature": 3.0, "distillation_weight": 0.5, "public_dataset_size": 5000},
        },
    },
    
    # Fair FL with QFedAvg
    "cifar10_qfedavg": {
        "name": "CIFAR-10 q-FedAvg (Fair FL)",
        "description": "Fair federated learning with q-fair aggregation",
        "config": {
            "algorithm": FLAlgorithm.QFEDAVG,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 100, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {
                "dataset": FLDataset.CIFAR10,
                "num_partitions": 10,
                "partition_strategy": PartitionStrategy.NON_IID_DIRICHLET,
                "dirichlet_alpha": 0.3,
            },
            "algorithm_params": {"q_param": 0.2, "server_learning_rate": 0.1},
        },
    },
    
    # Differential Privacy
    "cifar10_dp": {
        "name": "CIFAR-10 DP-FedAvg",
        "description": "Differentially private FL with fixed clipping",
        "config": {
            "algorithm": FLAlgorithm.DPFEDAVG_FIXED,
            "model": ModelArchitecture.CNN,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 3, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
            "privacy": {"differential_privacy": True, "noise_multiplier": 1.0, "max_grad_norm": 1.0},
        },
    },
    
    # Algorithm comparison suite
    "algorithm_comparison": {
        "name": "Algorithm Comparison Suite",
        "description": "Compare FedAvg, FedProx, FedAdam on CIFAR-10",
        "is_suite": True,
        "experiments": [
            {"name": "FedAvg", "algorithm": FLAlgorithm.FEDAVG},
            {"name": "FedProx", "algorithm": FLAlgorithm.FEDPROX, "algorithm_params": {"proximal_mu": 0.01}},
            {"name": "FedAdam", "algorithm": FLAlgorithm.FEDADAM, "algorithm_params": {"server_learning_rate": 0.1}},
            {"name": "FedYogi", "algorithm": FLAlgorithm.FEDYOGI, "algorithm_params": {"server_learning_rate": 0.1}},
        ],
        "base_config": {
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
        },
    },
    
    # Non-IID severity comparison
    "noniid_comparison": {
        "name": "Non-IID Severity Comparison",
        "description": "Compare different levels of data heterogeneity",
        "is_suite": True,
        "experiments": [
            {"name": "IID", "data": {"partition_strategy": PartitionStrategy.IID}},
            {"name": "Dirichlet-1.0", "data": {"partition_strategy": PartitionStrategy.NON_IID_DIRICHLET, "dirichlet_alpha": 1.0}},
            {"name": "Dirichlet-0.5", "data": {"partition_strategy": PartitionStrategy.NON_IID_DIRICHLET, "dirichlet_alpha": 0.5}},
            {"name": "Dirichlet-0.1", "data": {"partition_strategy": PartitionStrategy.NON_IID_DIRICHLET, "dirichlet_alpha": 0.1}},
            {"name": "Pathological", "data": {"partition_strategy": PartitionStrategy.PATHOLOGICAL}},
        ],
        "base_config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "model": ModelArchitecture.RESNET18,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10},
        },
    },
    
    # Model architecture comparison
    "model_comparison": {
        "name": "Model Architecture Comparison",
        "description": "Compare different model architectures",
        "is_suite": True,
        "experiments": [
            {"name": "SimpleCNN", "model": ModelArchitecture.CNN},
            {"name": "ResNet18", "model": ModelArchitecture.RESNET18},
            {"name": "MobileNetV2", "model": ModelArchitecture.MOBILENET_V2},
        ],
        "base_config": {
            "algorithm": FLAlgorithm.FEDAVG,
            "server": {"num_rounds": 50, "min_fit_clients": 5, "fraction_fit": 0.5},
            "client": {"local_epochs": 5, "learning_rate": 0.01, "local_batch_size": 32},
            "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10, "partition_strategy": PartitionStrategy.IID},
        },
    },
}


def create_experiment(
    name: str,
    algorithm: str = "fedavg",
    model: str = "cnn",
    dataset: str = "cifar10",
    num_partitions: int = 10,
    num_rounds: int = 100,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    partition_strategy: str = "iid",
    dirichlet_alpha: float = 0.5,
    num_runs: int = 1,
    seed: int = 42,
    **kwargs
) -> ExperimentConfig:
    """Create an experiment configuration with sensible defaults.
    
    Args:
        name: Experiment name
        algorithm: FL algorithm (fedavg, fedprox, fedadam, etc.)
        model: Model architecture (cnn, resnet18, mobilenet_v2, etc.)
        dataset: Dataset (cifar10, mnist, fashion_mnist, etc.)
        num_partitions: Number of clients
        num_rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        learning_rate: Client learning rate
        batch_size: Client batch size
        partition_strategy: Data partitioning (iid, non_iid_dirichlet, etc.)
        dirichlet_alpha: Alpha for Dirichlet partitioning
        num_runs: Number of times to repeat experiment
        seed: Random seed
        **kwargs: Additional configuration overrides
    
    Returns:
        ExperimentConfig instance
    """
    # Convert string enums
    algo = FLAlgorithm(algorithm.lower()) if isinstance(algorithm, str) else algorithm
    model_arch = ModelArchitecture(model.lower()) if isinstance(model, str) else model
    ds = FLDataset(dataset.lower()) if isinstance(dataset, str) else dataset
    part_strat = PartitionStrategy(partition_strategy.lower()) if isinstance(partition_strategy, str) else partition_strategy
    
    # Build config
    config = ExperimentConfig(
        name=name,
        algorithm=algo,
        model=model_arch,
        server=ServerConfig(
            num_rounds=num_rounds,
            min_fit_clients=max(2, num_partitions // 2),
            min_evaluate_clients=max(2, num_partitions // 2),
            min_available_clients=num_partitions,
            fraction_fit=0.5,
            fraction_evaluate=0.5,
        ),
        client=ClientConfig(
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            local_batch_size=batch_size,
        ),
        data=DataConfig(
            dataset=ds,
            num_partitions=num_partitions,
            partition_strategy=part_strat,
            dirichlet_alpha=dirichlet_alpha,
        ),
        seed=seed,
        num_runs=num_runs,
    )
    
    # Apply algorithm-specific defaults
    if algo == FLAlgorithm.FEDPROX:
        config.algorithm_params.proximal_mu = kwargs.get("proximal_mu", 0.01)
    elif algo in [FLAlgorithm.FEDADAM, FLAlgorithm.FEDYOGI]:
        config.algorithm_params.server_learning_rate = kwargs.get("server_learning_rate", 0.1)
    elif algo in [FLAlgorithm.FEDDF, FLAlgorithm.FEDMD]:
        config.algorithm_params.temperature = kwargs.get("temperature", 3.0)
        config.algorithm_params.distillation_weight = kwargs.get("distillation_weight", 0.5)
    
    return config


def get_pipeline(name: str) -> Optional[Dict[str, Any]]:
    """Get a pipeline configuration by name.
    
    Args:
        name: Pipeline name
    
    Returns:
        Pipeline configuration dictionary or None
    """
    return DEFAULT_PIPELINES.get(name)


def list_pipelines() -> List[Dict[str, Any]]:
    """List all available pipelines.
    
    Returns:
        List of pipeline summaries
    """
    return [
        {
            "id": key,
            "name": pipeline["name"],
            "description": pipeline["description"],
            "is_suite": pipeline.get("is_suite", False),
        }
        for key, pipeline in DEFAULT_PIPELINES.items()
    ]


def pipeline_to_experiments(pipeline_name: str) -> List[ExperimentConfig]:
    """Convert a pipeline to a list of experiment configurations.
    
    Args:
        pipeline_name: Name of the pipeline
    
    Returns:
        List of ExperimentConfig instances
    """
    pipeline = get_pipeline(pipeline_name)
    if not pipeline:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")
    
    if pipeline.get("is_suite", False):
        # Suite: multiple experiments with variations
        base_config = pipeline.get("base_config", {})
        experiments = []
        
        for exp in pipeline["experiments"]:
            # Merge base config with experiment-specific overrides
            config = deepcopy(base_config)
            for key, value in exp.items():
                if key == "name":
                    continue
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            experiments.append(create_experiment(
                name=exp["name"],
                **_flatten_config(config)
            ))
        
        return experiments
    else:
        # Single experiment
        config = pipeline.get("config", {})
        return [create_experiment(
            name=pipeline["name"],
            **_flatten_config(config)
        )]


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested config dict for create_experiment."""
    flat = {}
    
    # Direct mappings
    if "algorithm" in config:
        flat["algorithm"] = config["algorithm"].value if hasattr(config["algorithm"], "value") else config["algorithm"]
    if "model" in config:
        flat["model"] = config["model"].value if hasattr(config["model"], "value") else config["model"]
    
    # Server config
    if "server" in config:
        flat["num_rounds"] = config["server"].get("num_rounds", 100)
    
    # Client config
    if "client" in config:
        flat["local_epochs"] = config["client"].get("local_epochs", 5)
        flat["learning_rate"] = config["client"].get("learning_rate", 0.01)
        flat["batch_size"] = config["client"].get("local_batch_size", 32)
    
    # Data config
    if "data" in config:
        data = config["data"]
        if "dataset" in data:
            flat["dataset"] = data["dataset"].value if hasattr(data["dataset"], "value") else data["dataset"]
        flat["num_partitions"] = data.get("num_partitions", 10)
        if "partition_strategy" in data:
            flat["partition_strategy"] = data["partition_strategy"].value if hasattr(data["partition_strategy"], "value") else data["partition_strategy"]
        flat["dirichlet_alpha"] = data.get("dirichlet_alpha", 0.5)
    
    # Algorithm params
    if "algorithm_params" in config:
        flat.update(config["algorithm_params"])
    
    return flat
