# Federated Learning (FL) System Documentation

## Overview

This document describes the FL system architecture, endpoints, and function call sequence.

---

## Supported Algorithms

| Algorithm | ID | Description | Reference |
|-----------|-----|-------------|-----------|
| **FedAvg** | `fedavg` | Federated Averaging - weighted average of client updates | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) |
| **FedProx** | `fedprox` | FedAvg with proximal term for non-IID data | [Li et al., 2020](https://arxiv.org/abs/1812.06127) |
| **FedAvgM** | `fedavgm` | FedAvg with server-side momentum | [Hsu et al., 2019](https://arxiv.org/abs/1909.06335) |
| **FedXgbBagging** | `fedxgb_bagging` | Federated XGBoost with bagging aggregation | [Flower Tutorial](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html) |

---

## API Endpoints

### Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/fl/sessions` | Create a single FL session |
| `POST` | `/fl/sessions/multi` | Create multiple sessions (one per algorithm) |
| `GET` | `/fl/sessions` | List all FL sessions |
| `GET` | `/fl/sessions/{session_id}` | Get session details |
| `POST` | `/fl/sessions/{session_id}/start` | Start a session |
| `POST` | `/fl/sessions/{session_id}/stop` | Stop a running session |
| `DELETE` | `/fl/sessions/{session_id}` | Delete a session |

### Presets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/fl/presets` | List available presets |
| `POST` | `/fl/presets/{preset_name}/apply` | Apply a preset configuration |

---

## Function Call Sequence

### 1. Session Creation

```
Frontend                    Backend (fl_endpoints.py)           FL Module
   |                              |                                |
   |-- POST /fl/sessions -------->|                                |
   |                              |-- convert_to_fl_config() ----->|
   |                              |                                |
   |                              |<-- ExperimentConfig -----------|
   |                              |                                |
   |                              |-- fl_manager.create_session() ->|
   |                              |                                |-- FLSession created
   |                              |                                |-- status = PENDING
   |<-- session_id, status -------|                                |
```

### 2. Session Start

```
Frontend                    Backend (fl_endpoints.py)           FL Module (session.py)
   |                              |                                |
   |-- POST /sessions/{id}/start ->|                               |
   |                              |-- fl_manager.start_session() -->|
   |                              |                                |-- status = RUNNING
   |                              |                                |-- background_tasks.add_task(run_session)
   |<-- 200 OK -------------------|                                |
   |                              |                                |
   |                              |                    (async)     |
   |                              |                                |-- _run_flower_simulation()
```

### 3. FL Simulation (Windows Sequential Mode)

```
_run_flower_simulation()
    |
    |-- 1. Set random seeds (torch, numpy)
    |
    |-- 2. Get dataset info
    |       get_dataset_info(config.data.dataset)
    |
    |-- 3. Create global model
    |       get_model(config.model.value, num_classes, in_channels)
    |
    |-- 4. Get initial parameters
    |       ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict()])
    |
    |-- 5. Load centralized test set
    |       load_centralized_testset(dataset, batch_size)
    |
    |-- 6. Create strategy
    |       create_strategy(
    |           algorithm=config.algorithm.value,
    |           initial_parameters=...,
    |           evaluate_fn=...,
    |           fraction_fit=...,
    |           ...
    |       )
    |
    |-- 7. Define client_fn(context)
    |       |-- partition_id = context.node_config.get("partition-id")
    |       |-- load_partition(partition_id, ...)
    |       |-- get_model(...)
    |       |-- return FlowerClient(model, trainloader, valloader, ...)
    |
    |-- 8. Check platform
    |       if Windows:
    |           _run_sequential_simulation()
    |       else:
    |           run_simulation() (Ray-based)
```

### 4. Sequential Simulation (Windows)

```
_run_sequential_simulation()
    |
    FOR round_num in 1..num_rounds:
        |
        |-- FOR client_id in 0..num_clients:
        |       |-- context = MockContext(client_id)
        |       |-- client = client_fn(context)  --> FlowerClient
        |       |-- params_ndarrays = parameters_to_ndarrays(parameters)
        |       |-- fit_res = client.fit(params_ndarrays, config)
        |       |       |-- client.set_parameters(params)
        |       |       |-- train_pytorch(model, trainloader, epochs, lr, device, ...)
        |       |       |-- return get_parameters(), num_samples, metrics
        |       |-- fit_results.append(...)
        |
        |-- Aggregate (FedAvg weighted average)
        |       total_examples = sum(num_ex for _, num_ex, _ in fit_results)
        |       for client_params, num_examples, _ in fit_results:
        |           weight = num_examples / total_examples
        |           aggregated_params[i] += arr * weight
        |
        |-- Update global model
        |       global_model.load_state_dict(aggregated_params)
        |
        |-- Evaluate on test set
        |       loss, accuracy, _ = evaluate_model(global_model, testloader, device)
        |
        |-- Update session metrics
        |       session.round_metrics[round_num] = RoundMetrics(...)
        |       session.current_round = round_num
        |       if accuracy > session.best_accuracy:
        |           session.best_accuracy = accuracy
```

---

## Key Classes

### `FLSessionManager` (session.py)

Main class managing FL sessions.

```python
class FLSessionManager:
    def create_session(config: ExperimentConfig) -> FLSession
    async def start_session(session_id: str) -> FLSession
    def stop_session(session_id: str) -> bool
    def get_session(session_id: str) -> Optional[FLSession]
    def list_sessions() -> List[FLSession]
    def delete_session(session_id: str) -> bool
```

### `FLSession` (session.py)

Dataclass representing an FL session.

```python
@dataclass
class FLSession:
    session_id: str
    config: ExperimentConfig
    status: SessionStatus  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    current_round: int
    best_accuracy: float
    best_round: int
    round_metrics: Dict[int, RoundMetrics]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
```

### `FlowerClient` (client.py)

NumPy-based client for sequential simulation.

```python
class FlowerClient:
    def __init__(model, trainloader, valloader, local_epochs, learning_rate, device, proximal_mu)
    def get_parameters(config) -> list[np.ndarray]
    def set_parameters(parameters: list[np.ndarray])
    def fit(parameters, config) -> (updated_params, num_samples, metrics)
    def evaluate(parameters, config) -> (loss, num_samples, metrics)
```

### Strategy Classes (strategies.py)

```python
class FedAvgStrategy(Strategy)      # Federated Averaging
class FedProxStrategy(Strategy)     # FedAvg + proximal term
class FedAvgMStrategy(Strategy)     # FedAvg + server momentum
class FedXgbBaggingStrategy(Strategy)  # XGBoost bagging
```

---

## Configuration Classes

### `ExperimentConfig` (config.py)

```python
@dataclass
class ExperimentConfig:
    name: str
    algorithm: FLAlgorithm  # FEDAVG, FEDPROX, FEDAVGM, FEDXGB_BAGGING
    model: ModelArchitecture
    data: DataConfig
    server: ServerConfig
    client: ClientConfig
    algorithm_params: AlgorithmConfig
    seed: int
```

### `AlgorithmConfig` (config.py)

```python
@dataclass
class AlgorithmConfig:
    proximal_mu: float = 0.01        # FedProx
    server_momentum: float = 0.9     # FedAvgM
    server_learning_rate: float = 1.0  # FedAvgM
```

---

## File Structure

```
server/fl/
├── __init__.py              # Top-level exports
├── session.py               # FLSessionManager, FLSession
├── FL_DOCUMENTATION.md      # This file
│
├── core/
│   ├── __init__.py
│   ├── config.py            # ExperimentConfig, enums
│   ├── client.py            # FlowerClient, train_pytorch, evaluate_pytorch
│   ├── models.py            # get_model(), model architectures
│   └── server_app.py        # ServerApp creation
│
├── algorithms/
│   ├── __init__.py
│   └── strategies.py        # FedAvg, FedProx, FedAvgM, FedXgbBagging
│
└── datasets/
    ├── __init__.py
    └── loaders.py           # load_partition, load_centralized_testset
```

---

## Troubleshooting

### Sessions Complete Too Fast

**Symptom**: Sessions show as "completed" almost instantly without training.

**Possible Causes**:
1. **Missing FlowerClient**: The `FlowerClient` class was removed but is needed for Windows sequential simulation.
2. **Import errors**: Check server logs for import errors in `session.py`.
3. **Exception swallowed**: Training exceptions may be caught silently.

**Solution**: Check server logs for errors. Ensure `FlowerClient` is imported in `session.py`.

### Algorithm Not Found

**Symptom**: Error "Unknown algorithm: xxx"

**Solution**: Only use supported algorithms: `fedavg`, `fedprox`, `fedavgm`, `fedxgb_bagging`

### Windows Ray Issues

**Symptom**: Ray-related errors on Windows.

**Solution**: The system automatically falls back to sequential simulation on Windows. No action needed.

---

## References

- [Flower Documentation](https://flower.ai/docs/framework/)
- [Flower Quickstart PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Flower Quickstart XGBoost](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
- [FedAvgM Paper](https://arxiv.org/abs/1909.06335)
