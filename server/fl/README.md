# Federated Learning Module

A comprehensive, modular FL system built on top of **Flower (flwr)** framework.

## Features

- **Multiple FL Algorithms**: FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, FedAvgM, FedMedian, Krum, Bulyan, QFedAvg, DP-FedAvg
- **Knowledge Distillation**: FedDF, FedMD for heterogeneous model architectures (clients can have different models)
- **Multi-Model Experiments**: Queue multiple experiments with different models/parameters
- **Statistical Analysis**: Multiple runs per experiment with std, confidence intervals, shadow plots
- **Visualization**: Accuracy curves, sample distribution heatmaps, comparative bar charts
- **Default Pipelines**: Pre-configured experiments that are easy to extend

## Project Structure

```
fl/
├── __init__.py              # Main exports
├── README.md                # This file
├── examples.py              # Usage examples
├── session.py               # FLSessionManager for session lifecycle
│
├── core/                    # Core components
│   ├── __init__.py
│   ├── config.py            # All configuration dataclasses
│   ├── models.py            # PyTorch models (CNN, ResNet, LSTM, etc.)
│   └── client.py            # Flower NumPyClient implementation
│
├── algorithms/              # FL algorithms using flwr strategies
│   ├── __init__.py          # Strategy factory and registry
│   └── knowledge_distillation/
│       ├── __init__.py
│       ├── client.py        # KD client for heterogeneous models
│       ├── feddf.py         # FedDF strategy
│       ├── fedmd.py         # FedMD strategy
│       └── strategy.py      # KD strategy factory
│
├── datasets/                # Dataset loading with flwr-datasets
│   ├── __init__.py
│   ├── loaders.py           # Partition loading, centralized test set
│   └── partitioners.py      # IID, Dirichlet, Shard, Pathological
│
├── experiments/             # Experiment running and reporting
│   ├── __init__.py
│   ├── runner.py            # FLExperimentRunner for multi-model queue
│   ├── pipelines.py         # Default hardcoded pipelines
│   └── reports.py           # Per-model and comparative reports
│
└── visualization/           # Plotting and statistics
    ├── __init__.py
    ├── plots.py             # Accuracy curves, distribution, shadow plots
    └── statistics.py        # Statistical analysis, effect sizes
```

## Quick Start

### 1. Run a Single Experiment

```python
import asyncio
from server.fl import create_experiment, FLExperimentRunner

async def main():
    # Create experiment configuration
    experiment = create_experiment(
        name="CIFAR10-FedAvg",
        algorithm="fedavg",
        model="resnet18",
        dataset="cifar10",
        num_partitions=10,
        num_rounds=100,
        local_epochs=5,
        num_runs=3,  # Run 3 times for statistical significance
    )
    
    # Run experiment
    runner = FLExperimentRunner()
    result = await runner.run(experiment)
    
    print(f"Mean accuracy: {result.mean_accuracy:.4f} ± {result.std_accuracy:.4f}")

asyncio.run(main())
```

### 2. Compare Multiple Algorithms

```python
from server.fl import create_experiment, FLExperimentRunner, generate_comparative_report

async def compare_algorithms():
    experiments = [
        create_experiment(name="FedAvg", algorithm="fedavg", num_runs=3),
        create_experiment(name="FedProx", algorithm="fedprox", proximal_mu=0.01, num_runs=3),
        create_experiment(name="FedAdam", algorithm="fedadam", server_learning_rate=0.1, num_runs=3),
    ]
    
    runner = FLExperimentRunner()
    results = await runner.run_queue(experiments)
    
    report = generate_comparative_report(results)
    print(f"Best: {report['best_performer']['name']}")
```

### 3. Use Default Pipelines

```python
from server.fl import list_pipelines, pipeline_to_experiments, FLExperimentRunner

# List available pipelines
for p in list_pipelines():
    print(f"{p['id']}: {p['name']}")

# Run a pipeline
experiments = pipeline_to_experiments("cifar10_fedavg")
runner = FLExperimentRunner()
results = await runner.run_queue(experiments)
```

### 4. Knowledge Distillation (Heterogeneous Models)

```python
from server.fl import create_experiment, FLExperimentRunner

# FedDF allows different model architectures per client
experiment = create_experiment(
    name="FedDF-Heterogeneous",
    algorithm="feddf",
    dataset="cifar10",
    num_partitions=10,
    temperature=3.0,
    distillation_weight=0.5,
)

runner = FLExperimentRunner()
result = await runner.run(experiment)
```

### 5. Visualize Results

```python
from server.fl import (
    plot_accuracy_curves,
    plot_sample_distribution,
    plot_shadow_curves,
    get_all_label_distributions,
)

# Plot accuracy curves with std bands
plot_data = [{"name": r.config.name, "curves": [[m.accuracy for m in run.round_metrics] for run in r.runs]} for r in results]
fig = plot_accuracy_curves(plot_data, title="Algorithm Comparison")
fig.savefig("accuracy_curves.png")

# Plot sample distribution for non-IID
distributions = get_all_label_distributions(
    num_partitions=10,
    dataset="cifar10",
    partition_strategy="non_iid_dirichlet",
    dirichlet_alpha=0.5,
)
fig = plot_sample_distribution(distributions, num_classes=10)
fig.savefig("sample_distribution.png")
```

## Available Algorithms

| Algorithm | Description | Heterogeneous Models |
|-----------|-------------|---------------------|
| `fedavg` | Federated Averaging (baseline) | No |
| `fedprox` | FedAvg with proximal term | No |
| `fedadam` | Adaptive FL with Adam | No |
| `fedyogi` | Adaptive FL with Yogi | No |
| `fedadagrad` | Adaptive FL with Adagrad | No |
| `fedavgm` | FedAvg with server momentum | No |
| `fedmedian` | Byzantine-robust median | No |
| `fedtrimmedavg` | Byzantine-robust trimmed mean | No |
| `krum` | Byzantine-robust Krum | No |
| `bulyan` | Byzantine-robust Bulyan | No |
| `qfedavg` | Fair FL with q-fair aggregation | No |
| `dpfedavg_adaptive` | DP with adaptive clipping | No |
| `dpfedavg_fixed` | DP with fixed clipping | No |
| `feddf` | Federated Distillation | **Yes** |
| `fedmd` | Federated Model Distillation | **Yes** |

## Available Models

- **Image**: `cnn`, `resnet18`, `mobilenet_v2`
- **Sequence**: `lstm`, `gru`, `cnn_lstm`, `tcn`
- **Simple**: `mlp`, `logistic`

## Available Datasets

- `cifar10`, `cifar100`
- `mnist`, `fashion_mnist`, `emnist`
- `svhn`

## Partitioning Strategies

| Strategy | Description |
|----------|-------------|
| `iid` | Random uniform distribution |
| `non_iid_dirichlet` | Label skew via Dirichlet(α) |
| `shard` | Each client gets specific label shards |
| `pathological` | Extreme non-IID (few classes per client) |

## Default Pipelines

| Pipeline | Description |
|----------|-------------|
| `quick_test` | Fast test (5 rounds, 3 clients) |
| `cifar10_fedavg` | Standard CIFAR-10 benchmark |
| `cifar10_noniid_fedprox` | Non-IID with FedProx |
| `cifar10_fedadam` | Adaptive optimization |
| `mnist_fedavg` | MNIST quick benchmark |
| `cifar10_krum` | Byzantine-robust |
| `cifar10_feddf` | Knowledge distillation |
| `algorithm_comparison` | Compare FedAvg, FedProx, FedAdam, FedYogi |
| `noniid_comparison` | Compare IID vs various non-IID levels |
| `model_comparison` | Compare CNN, ResNet, MobileNet |

## Dependencies

```
flwr>=1.5.0
flwr-datasets>=0.1.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0  # Optional, for plotting
```

## Extending the System

### Add a New Model

```python
from server.fl.core.models import ModelRegistry
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        # ... define layers
    
    def forward(self, x):
        # ... forward pass
        return x

# Register the model
ModelRegistry.register("my_model", MyCustomModel)

# Use it
experiment = create_experiment(model="my_model", ...)
```

### Add a New Pipeline

```python
from server.fl.experiments.pipelines import DEFAULT_PIPELINES

DEFAULT_PIPELINES["my_pipeline"] = {
    "name": "My Custom Pipeline",
    "description": "Description of what this pipeline does",
    "config": {
        "algorithm": FLAlgorithm.FEDAVG,
        "model": ModelArchitecture.RESNET18,
        "server": {"num_rounds": 100},
        "client": {"local_epochs": 5},
        "data": {"dataset": FLDataset.CIFAR10, "num_partitions": 10},
    },
}
```

## Migration from Old Implementation

The old `flower_fl.py` and `fl_algorithms/` directory are now superseded by this modular system. Key changes:

1. **Configuration**: Use `ExperimentConfig` instead of `FLSessionConfig`
2. **Running experiments**: Use `FLExperimentRunner` for multi-run support
3. **Algorithms**: Use `create_strategy()` which wraps Flower's built-in strategies
4. **Datasets**: Use `load_partition()` which uses `flwr-datasets`

The old files can be kept for backward compatibility but new code should use this module.
