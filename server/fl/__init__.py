"""
Federated Learning Module.

This module provides a complete federated learning system built on Flower (flwr).
All FL algorithms are CUSTOM IMPLEMENTATIONS based on official Flower examples -
no built-in Flower strategies are used.

=============================================================================
IMPLEMENTATION PHILOSOPHY
=============================================================================

1. TRANSPARENCY: All aggregation logic is explicitly implemented so you can
   understand and debug exactly what's happening.

2. DOCUMENTATION: Every function and class includes detailed docstrings with
   algorithm explanations and references to original papers.

3. SIMPLICITY: Only essential algorithms are included (FedAvg, FedProx, FedAvgM).
   This makes the codebase easier to understand and extend.

4. REFERENCES: All implementations cite the original papers and link to
   official Flower examples they're based on.

=============================================================================
AVAILABLE ALGORITHMS
=============================================================================

FedAvg (Federated Averaging):
    Paper: McMahan et al., 2017 - https://arxiv.org/abs/1602.05629
    Flower Example: https://github.com/adap/flower/tree/main/examples/quickstart-pytorch
    
FedProx:
    Paper: Li et al., 2020 - https://arxiv.org/abs/1812.06127
    Flower Example: https://github.com/adap/flower/tree/main/examples/advanced-pytorch

FedAvgM:
    Paper: Hsu et al., 2019 - https://arxiv.org/abs/1909.06335

=============================================================================
QUICK START
=============================================================================

    from server.fl import (
        FedAvgStrategy,
        FlowerClient,
        ExperimentConfig,
        FLAlgorithm,
        get_model,
    )
    
    # 1. Create a model
    model = get_model("cnn", num_classes=10)
    
    # 2. Create a strategy
    strategy = FedAvgStrategy(
        fraction_fit=1.0,
        initial_parameters=model_to_parameters(model),
    )
    
    # 3. Create a client
    client = FlowerClient(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
    )

=============================================================================
MODULE STRUCTURE
=============================================================================

server/fl/
├── __init__.py          # This file - main exports
├── algorithms/          # FL strategies (FedAvg, FedProx, FedAvgM)
│   ├── __init__.py
│   └── strategies.py    # Custom strategy implementations
├── core/                # Core components
│   ├── __init__.py
│   ├── client.py        # FlowerClient implementation
│   ├── config.py        # Configuration dataclasses
│   └── models.py        # Model architectures
├── datasets/            # Data loading and partitioning
├── session.py           # FL session management
└── README.md            # Detailed documentation

=============================================================================
"""

# =============================================================================
# CORE CONFIGURATION
# =============================================================================

from .core.config import (
    # Enums
    FLAlgorithm,
    FLDataset,
    PartitionStrategy,
    ModelArchitecture,
    AggregationMethod,
    ClientSelectionStrategy,
    # Config dataclasses
    ExperimentConfig,
    ClientConfig,
    ServerConfig,
    AlgorithmConfig,
    DataConfig,
    PrivacyConfig,
    MonitoringConfig,
    FLConfig,  # Alias for ExperimentConfig
)

# Alias for backward compatibility with endpoints
FLSessionConfig = ExperimentConfig


def get_algorithm_info(algorithm: FLAlgorithm) -> dict:
    """Get information about an FL algorithm.
    
    Args:
        algorithm: FLAlgorithm enum value
    
    Returns:
        Dictionary with algorithm metadata
    """
    info = {
        FLAlgorithm.FEDAVG: {
            "name": "FedAvg",
            "description": "Federated Averaging - weighted average of client updates",
            "paper": "McMahan et al., 2017",
            "reference": "https://arxiv.org/abs/1602.05629",
            "category": "standard",
        },
        FLAlgorithm.FEDPROX: {
            "name": "FedProx",
            "description": "FedAvg with proximal term for heterogeneous data",
            "paper": "Li et al., 2020",
            "reference": "https://arxiv.org/abs/1812.06127",
            "category": "standard",
        },
        FLAlgorithm.FEDAVGM: {
            "name": "FedAvgM",
            "description": "FedAvg with server-side momentum",
            "paper": "Hsu et al., 2019",
            "reference": "https://arxiv.org/abs/1909.06335",
            "category": "standard",
        },
        FLAlgorithm.FEDXGB_BAGGING: {
            "name": "FedXgbBagging",
            "description": "Federated XGBoost with bagging aggregation",
            "paper": "Flower XGBoost Tutorial",
            "reference": "https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html",
            "category": "xgboost",
        },
    }
    return info.get(algorithm, {
        "name": algorithm.value,
        "description": "Custom FL algorithm",
        "category": "custom",
    })

# =============================================================================
# MODELS
# =============================================================================

from .core.models import get_model, ModelRegistry

# =============================================================================
# CLIENT
# =============================================================================

from .core.client import (
    create_client_app,
    create_xgboost_client_app,
    train_pytorch,
    evaluate_pytorch,
    train_model,
    evaluate_model,
    get_device,
)

from .core.server_app import (
    create_server_app,
    create_simple_fl_app,
    evaluate_fn_factory,
)

# =============================================================================
# STRATEGIES (Custom implementations - NO built-in Flower strategies)
# =============================================================================

from .algorithms import (
    # Strategy classes
    FedAvgStrategy,
    FedProxStrategy,
    FedAvgMStrategy,
    FedXgbBaggingStrategy,
    # Factory function
    create_strategy,
    list_strategies,
    # Aggregation utilities
    aggregate_weighted_average,
    STRATEGY_REGISTRY,
)

# =============================================================================
# DATASETS
# =============================================================================

from .datasets import (
    load_partition,
    load_centralized_testset,
    get_dataset_info,
)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

from .session import (
    FLSessionManager,
    FLSession,
    SessionStatus,
    RoundMetrics,
    ClientRoundMetrics,
    fl_manager,
)

# =============================================================================
# REMOTE DEVICE SUPPORT
# =============================================================================

from .remote_client import (
    remote_device_manager,
    DeviceStatus,
    RemoteFLDevice,
    generate_client_script,
)

# =============================================================================
# PARTICIPATION REQUEST SYSTEM
# =============================================================================

from .participation_request import (
    fl_participation_manager,
    RequestStatus,
    FLParticipationRequest,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "FLAlgorithm",
    "FLDataset",
    "PartitionStrategy",
    "ModelArchitecture",
    "ExperimentConfig",
    "ClientConfig",
    "ServerConfig",
    "AlgorithmConfig",
    "DataConfig",
    "FLConfig",
    # Models
    "get_model",
    "ModelRegistry",
    # Client (ClientApp with decorators)
    "create_client_app",
    "create_xgboost_client_app",
    "train_pytorch",
    "evaluate_pytorch",
    "train_model",
    "evaluate_model",
    "get_device",
    # Strategies
    "FedAvgStrategy",
    "FedProxStrategy",
    "FedAvgMStrategy",
    "FedXgbBaggingStrategy",
    "create_strategy",
    "list_strategies",
    "aggregate_weighted_average",
    "STRATEGY_REGISTRY",
    # Datasets
    "load_partition",
    "load_centralized_testset",
    "get_dataset_info",
    # Session
    "FLSessionManager",
    "FLSession",
    "SessionStatus",
    "RoundMetrics",
    "fl_manager",
    # Server App
    "create_server_app",
    "create_simple_fl_app",
    "evaluate_fn_factory",
    # Remote Device Support
    "remote_device_manager",
    "DeviceStatus",
    "RemoteFLDevice",
    "generate_client_script",
    # Participation Request
    "fl_participation_manager",
    "RequestStatus",
    "FLParticipationRequest",
    # Session Metrics
    "ClientRoundMetrics",
    # Config extras
    "AggregationMethod",
    "ClientSelectionStrategy",
    "PrivacyConfig",
    "MonitoringConfig",
    "FLSessionConfig",
    "get_algorithm_info",
]
