"""
Federated Learning Core Module.

This module provides the core components for federated learning:
- Configuration dataclasses (ExperimentConfig, etc.)
- Model architectures (CNN, ResNet, etc.)
- Client implementation (FlowerClient)

=============================================================================
USAGE
=============================================================================

    from server.fl.core import (
        ExperimentConfig,
        FLAlgorithm,
        FlowerClient,
        get_model,
        train_model,
        evaluate_model,
    )
    
    # Create experiment config
    config = ExperimentConfig(
        name="my_experiment",
        algorithm=FLAlgorithm.FEDAVG,
    )
    
    # Create model
    model = get_model("cnn", num_classes=10)
    
    # Create client
    client = FlowerClient(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
    )
"""

from .config import (
    FLAlgorithm,
    FLDataset,
    PartitionStrategy,
    ModelArchitecture,
    AggregationMethod,
    ClientSelectionStrategy,
    PrivacyConfig,
    ClientConfig,
    ServerConfig,
    AlgorithmConfig,
    DataConfig,
    MonitoringConfig,
    ExperimentConfig,
    FLConfig,
)

from .models import get_model, ModelRegistry

from .client import (
    create_client_app,
    create_xgboost_client_app,
    train_pytorch,
    evaluate_pytorch,
    train_model,
    evaluate_model,
    get_device,
)

from .server_app import (
    create_server_app,
    create_simple_fl_app,
    evaluate_fn_factory,
)

__all__ = [
    # Config enums
    "FLAlgorithm",
    "FLDataset",
    "PartitionStrategy",
    "ModelArchitecture",
    "AggregationMethod",
    "ClientSelectionStrategy",
    # Config dataclasses
    "PrivacyConfig",
    "ClientConfig",
    "ServerConfig",
    "AlgorithmConfig",
    "DataConfig",
    "MonitoringConfig",
    "ExperimentConfig",
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
    # Server
    "create_server_app",
    "create_simple_fl_app",
    "evaluate_fn_factory",
]
