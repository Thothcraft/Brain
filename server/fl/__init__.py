"""Federated Learning Module using Flower Framework.

This module provides a comprehensive FL system built on top of Flower (flwr):
- Multiple FL algorithms (FedAvg, FedProx, FedAdam, FedYogi, etc.)
- Knowledge distillation algorithms for heterogeneous model architectures
- Multi-model experiment queue with statistical analysis
- Visualization utilities (accuracy curves, sample distribution, shadow plots)
- Default pipelines that are easily extensible

Usage:
    from server.fl import (
        FLExperimentRunner,
        create_experiment,
        DEFAULT_PIPELINES,
    )
    
    # Run a single experiment
    experiment = create_experiment(
        name="CIFAR10-FedAvg",
        algorithm="fedavg",
        dataset="cifar10",
        model="resnet18",
    )
    runner = FLExperimentRunner()
    results = await runner.run(experiment)
    
    # Run multiple models with comparative report
    experiments = [
        create_experiment(name="CNN", model="cnn", algorithm="fedavg"),
        create_experiment(name="ResNet", model="resnet18", algorithm="fedavg"),
    ]
    results = await runner.run_queue(experiments, runs_per_experiment=3)
    report = runner.generate_comparative_report(results)

Flower Documentation: https://flower.ai/docs/
"""

# Core config (always available)
from .core.config import (
    FLConfig,
    ClientConfig,
    ServerConfig,
    AlgorithmConfig,
    DataConfig,
    ExperimentConfig,
    PrivacyConfig,
    MonitoringConfig,
    FLAlgorithm,
    FLDataset,
    PartitionStrategy,
    ModelArchitecture,
    AggregationMethod,
    ClientSelectionStrategy,
)

# Alias for backward compatibility with flower_fl.py
FLSessionConfig = ExperimentConfig

# Core models
from .core.models import get_model, ModelRegistry

# Core client
from .core.client import FlowerClient

# Algorithms
from .algorithms import create_strategy, ALGORITHM_REGISTRY, list_algorithms, get_algorithm_info

# Knowledge distillation
from .algorithms.knowledge_distillation import (
    FedDFStrategy,
    FedMDStrategy,
    KDClient,
)

# Datasets
from .datasets import (
    load_partition,
    load_centralized_testset,
    load_public_dataset,
    get_dataset_info,
    get_all_label_distributions,
)

# Experiments
from .experiments import (
    FLExperimentRunner,
    ExperimentResult,
    RunResult,
    create_experiment,
    DEFAULT_PIPELINES,
    list_pipelines,
    pipeline_to_experiments,
)
from .experiments.reports import (
    generate_experiment_report,
    generate_comparative_report,
    generate_statistical_summary,
    format_report_as_text,
)

# Visualization
from .visualization import (
    plot_accuracy_curves,
    plot_sample_distribution,
    plot_comparative_results,
    plot_shadow_curves,
    plot_convergence_comparison,
    generate_statistics_report,
)

# Session manager
from .session import FLSessionManager, FLSession, FLClient, SessionStatus, fl_manager, RoundMetrics, ClientRoundMetrics

# Remote device support for distributed FL
# Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
from .remote_client import (
    RemoteFLDevice,
    RemoteDeviceManager,
    DeviceStatus,
    remote_device_manager,
    generate_client_script,
)

# FL participation request system for Thoth devices
from .participation_request import (
    FLParticipationRequest,
    FLProgressUpdate,
    FLParticipationManager,
    RequestStatus,
    fl_participation_manager,
)

__all__ = [
    # Config
    "FLConfig",
    "FLSessionConfig",  # Backward compatibility alias
    "ClientConfig", 
    "ServerConfig",
    "AlgorithmConfig",
    "DataConfig",
    "ExperimentConfig",
    "PrivacyConfig",
    "MonitoringConfig",
    "FLAlgorithm",
    "FLDataset",
    "PartitionStrategy",
    "ModelArchitecture",
    "AggregationMethod",
    "ClientSelectionStrategy",
    # Models
    "get_model",
    "ModelRegistry",
    # Client
    "FlowerClient",
    # Algorithms
    "create_strategy",
    "ALGORITHM_REGISTRY",
    "list_algorithms",
    "get_algorithm_info",
    "FedDFStrategy",
    "FedMDStrategy",
    "KDClient",
    # Datasets
    "load_partition",
    "load_centralized_testset",
    "load_public_dataset",
    "get_dataset_info",
    "get_all_label_distributions",
    # Experiments
    "FLExperimentRunner",
    "ExperimentResult",
    "RunResult",
    "create_experiment",
    "DEFAULT_PIPELINES",
    "list_pipelines",
    "pipeline_to_experiments",
    "generate_experiment_report",
    "generate_comparative_report",
    "generate_statistical_summary",
    "format_report_as_text",
    # Visualization
    "plot_accuracy_curves",
    "plot_sample_distribution",
    "plot_comparative_results",
    "plot_shadow_curves",
    "plot_convergence_comparison",
    "generate_statistics_report",
    # Session
    "FLSessionManager",
    "FLSession",
    "FLClient",
    "SessionStatus",
    "fl_manager",
    "RoundMetrics",
    "ClientRoundMetrics",
    # Remote devices
    "RemoteFLDevice",
    "RemoteDeviceManager",
    "DeviceStatus",
    "remote_device_manager",
    "generate_client_script",
    # Participation requests
    "FLParticipationRequest",
    "FLProgressUpdate",
    "FLParticipationManager",
    "RequestStatus",
    "fl_participation_manager",
]
