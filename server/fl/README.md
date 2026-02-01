# Federated Learning Module

A simplified, well-documented federated learning system built on [Flower](https://flower.ai/).

## Key Design Principles

1. **ClientApp with Decorators Only** - Uses modern Flower `@app.train` and `@app.evaluate` decorators
2. **No Built-in Flower Strategies** - All FL algorithms are custom implementations with explicit aggregation logic
3. **XGBoost Support** - Includes federated XGBoost based on Flower tutorial
4. **Comprehensive Documentation** - Every function includes detailed docstrings with algorithm explanations
5. **Online References** - All implementations cite original papers and link to Flower examples

---

## Quick Start

```python
from server.fl import (
    create_client_app,
    create_xgboost_client_app,
    FedAvgStrategy,
    FedXgbBaggingStrategy,
    get_model,
    load_partition,
    create_strategy,
)

# 1. Create a model
model = get_model("cnn", num_classes=10)

# 2. Create a strategy (custom implementation, NOT Flower built-in)
strategy = create_strategy(
    algorithm="fedavg",  # or "fedprox", "fedavgm", "fedxgb_bagging"
    fraction_fit=1.0,
)

# 3. Create a client app (uses @app.train decorator)
client_app = create_client_app(
    model_fn=lambda: get_model("cnn", num_classes=10),
    load_data_fn=load_partition,
    default_epochs=5,
    default_lr=0.01,
)
```

---

## Available Algorithms

| Algorithm | Description | Paper | Reference |
|-----------|-------------|-------|-----------|
| **FedAvg** | Federated Averaging - weighted average of client updates | McMahan et al., 2017 | [arXiv](https://arxiv.org/abs/1602.05629) |
| **FedProx** | FedAvg with proximal term for heterogeneous data | Li et al., 2020 | [arXiv](https://arxiv.org/abs/1812.06127) |
| **FedAvgM** | FedAvg with server-side momentum | Hsu et al., 2019 | [arXiv](https://arxiv.org/abs/1909.06335) |
| **FedXgbBagging** | Federated XGBoost with bagging aggregation | Flower Tutorial | [Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html) |

---

## Module Structure

```
server/fl/
├── __init__.py              # Main exports
├── README.md                # This file
├── session.py               # FL session management
│
├── algorithms/              # Custom FL strategies
│   ├── __init__.py
│   └── strategies.py        # FedAvg, FedProx, FedAvgM, FedXgbBagging
│
├── core/                    # Core components
│   ├── __init__.py
│   ├── client.py            # ClientApp with @app.train/@app.evaluate decorators
│   ├── server_app.py        # ServerApp with @app.main decorator
│   ├── config.py            # Configuration dataclasses
│   └── models.py            # Model architectures (CNN, ResNet, MLP, etc.)
│
└── datasets/                # Data loading and partitioning
    ├── __init__.py
    ├── loaders.py           # Dataset loaders (CIFAR10, MNIST, etc.)
    ├── partitioners.py      # IID/non-IID partitioning
    └── preprocessing.py     # Data preprocessing
```

---

## Client Implementation (ClientApp with Decorators)

The client uses the modern Flower ClientApp API with `@app.train` and `@app.evaluate` decorators.

### PyTorch Client

```python
from server.fl import create_client_app, get_model

# Create client app for PyTorch models
client_app = create_client_app(
    model_fn=lambda: get_model("cnn", num_classes=10),
    load_data_fn=load_partition,  # (partition_id, num_partitions, batch_size) -> (train, val)
    default_epochs=5,
    default_lr=0.01,
    proximal_mu=0.0,  # Set > 0 for FedProx
)
```

### XGBoost Client

```python
from server.fl import create_xgboost_client_app

# Create client app for XGBoost
xgb_app = create_xgboost_client_app(
    load_data_fn=load_xgb_partition,  # Returns DMatrix objects
    params={"objective": "binary:logistic", "max_depth": 8},
    default_local_rounds=1,
)
```

**Reference**: [Flower XGBoost Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html)

---

## Strategies

### FedAvg (Federated Averaging)

```python
from server.fl import FedAvgStrategy

strategy = FedAvgStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    initial_parameters=params,
)
```

**Algorithm**:
```
w_global = Σ (n_k / n_total) * w_k  # Weighted average by examples
```

### FedProx

```python
from server.fl import FedProxStrategy

strategy = FedProxStrategy(
    proximal_mu=0.1,  # Regularization strength
    fraction_fit=1.0,
)
```

**Key Difference**: Adds proximal term to client loss:
```
L_total = L_task + (μ/2) * ||w - w_global||²
```

### FedAvgM (Server Momentum)

```python
from server.fl import FedAvgMStrategy

strategy = FedAvgMStrategy(
    server_momentum=0.9,
    fraction_fit=1.0,
)
```

### FedXgbBagging (XGBoost)

```python
from server.fl import FedXgbBaggingStrategy

strategy = FedXgbBaggingStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
)
```

**Reference**: [Flower XGBoost Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html)

---

## Data Selection

Supported datasets with automatic partitioning:

| Dataset | Classes | Shape | Samples |
|---------|---------|-------|---------|
| CIFAR-10 | 10 | 3x32x32 | 60,000 |
| CIFAR-100 | 100 | 3x32x32 | 60,000 |
| MNIST | 10 | 1x28x28 | 70,000 |
| Fashion-MNIST | 10 | 1x28x28 | 70,000 |
| SVHN | 10 | 3x32x32 | 99,289 |
| EMNIST | 62 | 1x28x28 | 814,255 |

```python
from server.fl import load_partition, get_dataset_info, FLDataset

# Get dataset info
info = get_dataset_info(FLDataset.CIFAR10)
print(info["num_classes"])  # 10

# Load a partition
trainloader, valloader = load_partition(
    partition_id=0,
    num_partitions=10,
    dataset=FLDataset.CIFAR10,
    partition_strategy="iid",  # or "non_iid_dirichlet"
    batch_size=32,
)
```

---

## Model Selection

Available models:

| Model | Type | Use Case |
|-------|------|----------|
| `cnn` | SimpleCNN | CIFAR-10, MNIST |
| `resnet18` | ResNet-18 | Image classification |
| `mobilenet_v2` | MobileNetV2 | Mobile/edge devices |
| `mlp` | Multi-layer Perceptron | Tabular data |
| `lstm` | LSTM | Sequences |
| `gru` | GRU | Sequences |
| `tcn` | Temporal CNN | Time series |

```python
from server.fl import get_model

# Image models
model = get_model("cnn", num_classes=10, in_channels=3)
model = get_model("resnet18", num_classes=100)

# Sequence models
model = get_model("lstm", num_classes=10, in_channels=52, seq_length=100)

# Tabular models
model = get_model("mlp", input_dim=784, num_classes=10)
```

---

## Training Functions

```python
from server.fl import train_pytorch, evaluate_pytorch

# Train a model
loss, num_samples = train_pytorch(
    model=model,
    trainloader=train_loader,
    epochs=5,
    lr=0.01,
    device=torch.device("cuda"),
    proximal_mu=0.0,  # FedProx coefficient
)

# Evaluate a model
loss, accuracy, num_samples = evaluate_pytorch(
    model=model,
    testloader=test_loader,
    device=torch.device("cuda"),
)
```

---

## Configuration

```python
from server.fl import ExperimentConfig, FLAlgorithm, FLDataset

config = ExperimentConfig(
    name="my_experiment",
    algorithm=FLAlgorithm.FEDAVG,  # FEDAVG, FEDPROX, FEDAVGM
    server=ServerConfig(
        num_rounds=100,
        fraction_fit=1.0,
        min_fit_clients=2,
    ),
    client=ClientConfig(
        local_epochs=5,
        learning_rate=0.01,
        local_batch_size=32,
    ),
    algorithm_params=AlgorithmConfig(
        proximal_mu=0.01,      # FedProx
        server_momentum=0.9,   # FedAvgM
    ),
    data=DataConfig(
        dataset=FLDataset.CIFAR10,
        num_partitions=10,
        partition_strategy=PartitionStrategy.IID,
    ),
)
```

---

## References

### Papers

| Algorithm | Paper | ArXiv |
|-----------|-------|-------|
| FedAvg | McMahan et al., 2017 | [1602.05629](https://arxiv.org/abs/1602.05629) |
| FedProx | Li et al., 2020 | [1812.06127](https://arxiv.org/abs/1812.06127) |
| FedAvgM | Hsu et al., 2019 | [1909.06335](https://arxiv.org/abs/1909.06335) |

### Flower Documentation

- [Quickstart PyTorch Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Quickstart XGBoost Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html)
- [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [ClientApp API Reference](https://flower.ai/docs/framework/ref-api/flwr.client.ClientApp.html)

### Flower Examples

- [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch)
- [xgboost-quickstart](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart)
- [advanced-pytorch](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)
