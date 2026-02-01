"""
Federated Learning Client Implementation using ClientApp Decorators.

This module provides the client-side implementation for federated learning
using the modern Flower ClientApp API with @app.train and @app.evaluate decorators.

=============================================================================
IMPLEMENTATION REFERENCES
=============================================================================

Flower Quickstart PyTorch:
    https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    
Flower Quickstart XGBoost:
    https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html

Flower ClientApp API:
    https://flower.ai/docs/framework/ref-api/flwr.client.ClientApp.html

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

This module uses the modern Flower ClientApp pattern with decorators:

    @app.train()   - Called each round to train on local data
    @app.evaluate() - Called to evaluate model on local validation data

The ClientApp receives Messages containing:
    - ArrayRecord with model parameters (key: "arrays")
    - ConfigRecord with training config (key: "config")

=============================================================================
USAGE EXAMPLE
=============================================================================

    from server.fl.core.client import create_client_app
    
    # Create client app for PyTorch models
    client_app = create_client_app(
        model_fn=lambda: Net(),
        load_data_fn=load_partition,
        default_epochs=5,
        default_lr=0.01,
    )
    
    # Create client app for XGBoost
    xgb_client_app = create_xgboost_client_app(
        load_data_fn=load_xgb_partition,
        params={"objective": "binary:logistic", "max_depth": 8},
    )
"""

import logging
from typing import Dict, Any, Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flwr.client import ClientApp
from flwr.common import Context, Message, RecordDict, ArrayRecord, MetricRecord, ConfigRecord

logger = logging.getLogger(__name__)


# =============================================================================
# PYTORCH TRAINING FUNCTIONS
# =============================================================================

def train_pytorch(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_params: Optional[list] = None,
) -> Tuple[float, int]:
    """
    Train a PyTorch model on local data.
    
    ==========================================================================
    ALGORITHM
    ==========================================================================
    
    For each epoch:
        For each batch (x, y) in trainloader:
            1. Forward pass: outputs = model(x)
            2. Compute task loss: L_task = CrossEntropy(outputs, y)
            3. If FedProx (proximal_mu > 0):
               L_prox = (μ/2) * Σ ||w - w_global||²
               L_total = L_task + L_prox
            4. Backward pass and update weights
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    FedAvg: https://arxiv.org/abs/1602.05629
    FedProx: https://arxiv.org/abs/1812.06127
    Flower Tutorial: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    
    Args:
        model: PyTorch model to train
        trainloader: DataLoader for training data
        epochs: Number of local training epochs
        lr: Learning rate
        device: Device to train on
        proximal_mu: FedProx proximal term coefficient (0 for FedAvg)
        global_params: Global model parameters for FedProx
    
    Returns:
        Tuple of (average_loss, num_samples)
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    running_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch in trainloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("image", batch.get("data")))
                labels = batch.get("label", batch.get("labels"))
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add FedProx proximal term if enabled
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    diff = local_param - global_param.to(device)
                    proximal_term += (diff ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    num_samples = len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
    
    return avg_loss, num_samples


def evaluate_pytorch(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Evaluate a PyTorch model on test/validation data.
    
    Args:
        model: PyTorch model to evaluate
        testloader: DataLoader for test data
        device: Device for evaluation
    
    Returns:
        Tuple of (loss, accuracy, num_samples)
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("image", batch.get("data")))
                labels = batch.get("label", batch.get("labels"))
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            batch_loss = criterion(outputs, labels).item()
            total_loss += batch_loss * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, total_samples


# =============================================================================
# PYTORCH CLIENT APP
# =============================================================================

def create_client_app(
    model_fn: Callable[[], nn.Module],
    load_data_fn: Callable,
    default_epochs: int = 5,
    default_lr: float = 0.01,
    proximal_mu: float = 0.0,
    device: Optional[torch.device] = None,
) -> ClientApp:
    """
    Create a Flower ClientApp for PyTorch models.
    
    This follows the modern Flower API pattern from the quickstart tutorial:
    https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        model_fn: Function that returns a new model instance.
            Called once per client to create their local model.
            Example: lambda: SimpleCNN(num_classes=10)
        
        load_data_fn: Function(partition_id, num_partitions, batch_size) -> (train, val)
            Function to load data for a specific partition/client.
        
        default_epochs: Default number of local training epochs.
        
        default_lr: Default learning rate.
        
        proximal_mu: FedProx proximal term coefficient (0 for FedAvg).
        
        device: PyTorch device (auto-detected if None).
    
    Returns:
        Configured ClientApp
    
    ==========================================================================
    EXAMPLE
    ==========================================================================
    
        from server.fl.core.client import create_client_app
        from server.fl.core.models import get_model
        from server.fl.datasets import load_partition
        
        client_app = create_client_app(
            model_fn=lambda: get_model("cnn", num_classes=10),
            load_data_fn=load_partition,
            default_epochs=5,
            default_lr=0.01,
        )
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    app = ClientApp()
    
    @app.train()
    def train(msg: Message, context: Context) -> Message:
        """
        Train the model on local data.
        
        Receives global model parameters, trains locally, returns updated parameters.
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
        """
        # Get partition info from node config
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 10)
        
        # Get training config from message or run config
        config = msg.content.get("config", ConfigRecord())
        epochs = int(config.get("epochs", context.run_config.get("local-epochs", default_epochs)))
        lr = float(config.get("lr", context.run_config.get("learning-rate", default_lr)))
        batch_size = int(config.get("batch_size", context.run_config.get("batch-size", 32)))
        mu = float(config.get("proximal_mu", proximal_mu))
        
        # Load data partition
        trainloader, _ = load_data_fn(partition_id, num_partitions, batch_size)
        
        # Create model and load global weights
        model = model_fn()
        arrays = msg.content.get("arrays", ArrayRecord())
        
        if len(arrays) > 0:
            # Load parameters from ArrayRecord
            state_dict = {}
            for i, (key, _) in enumerate(model.state_dict().items()):
                state_dict[key] = torch.tensor(arrays[str(i)].numpy())
            model.load_state_dict(state_dict)
        
        # Store global params for FedProx
        global_params = None
        if mu > 0:
            global_params = [p.clone().detach() for p in model.parameters()]
        
        # Train locally
        loss, num_samples = train_pytorch(
            model=model,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            device=device,
            proximal_mu=mu,
            global_params=global_params,
        )
        
        # Convert model to ArrayRecord
        model_arrays = ArrayRecord()
        for i, (_, param) in enumerate(model.state_dict().items()):
            model_arrays[str(i)] = param.cpu().numpy()
        
        # Build reply message
        metrics = MetricRecord({"train_loss": loss, "num_examples": num_samples})
        content = RecordDict({"arrays": model_arrays, "metrics": metrics})
        
        return Message(content=content, reply_to=msg)
    
    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        """
        Evaluate the model on local validation data.
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
        """
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 10)
        batch_size = int(context.run_config.get("batch-size", 32))
        
        # Load validation data
        _, valloader = load_data_fn(partition_id, num_partitions, batch_size)
        
        # Create model and load weights
        model = model_fn()
        arrays = msg.content.get("arrays", ArrayRecord())
        
        if len(arrays) > 0:
            state_dict = {}
            for i, (key, _) in enumerate(model.state_dict().items()):
                state_dict[key] = torch.tensor(arrays[str(i)].numpy())
            model.load_state_dict(state_dict)
        
        # Evaluate
        loss, accuracy, num_samples = evaluate_pytorch(model, valloader, device)
        
        # Build reply message
        metrics = MetricRecord({
            "loss": loss,
            "accuracy": accuracy,
            "num_examples": num_samples,
        })
        content = RecordDict({"metrics": metrics})
        
        return Message(content=content, reply_to=msg)
    
    return app


# =============================================================================
# XGBOOST CLIENT APP
# =============================================================================

def create_xgboost_client_app(
    load_data_fn: Callable,
    params: Optional[Dict[str, Any]] = None,
    default_local_rounds: int = 1,
) -> ClientApp:
    """
    Create a Flower ClientApp for XGBoost federated learning.
    
    This follows the Flower XGBoost quickstart tutorial:
    https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
    
    ==========================================================================
    XGBOOST FEDERATED LEARNING
    ==========================================================================
    
    XGBoost FL uses a bagging approach:
    1. First round: Each client trains initial trees from scratch
    2. Subsequent rounds: Clients load global model and add local trees
    3. Server aggregates trees from all clients (bagging)
    
    The model is serialized as raw bytes for transmission.
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        load_data_fn: Function(partition_id, num_partitions) -> (train_dmatrix, valid_dmatrix, num_train, num_val)
            Function to load XGBoost DMatrix data for a partition.
        
        params: XGBoost parameters dictionary. Default:
            {
                "objective": "binary:logistic",
                "eta": 0.1,
                "max_depth": 8,
                "eval_metric": "auc",
                "tree_method": "hist",
            }
        
        default_local_rounds: Number of local boosting rounds per FL round.
    
    Returns:
        Configured ClientApp for XGBoost
    
    ==========================================================================
    EXAMPLE
    ==========================================================================
    
        from server.fl.core.client import create_xgboost_client_app
        
        def load_xgb_data(partition_id, num_partitions):
            # Load and return DMatrix objects
            train_dmatrix = xgb.DMatrix(X_train, label=y_train)
            valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
            return train_dmatrix, valid_dmatrix, len(X_train), len(X_val)
        
        xgb_app = create_xgboost_client_app(
            load_data_fn=load_xgb_data,
            params={"objective": "multi:softmax", "num_class": 10},
        )
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    Flower XGBoost Tutorial: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
    XGBoost Source: https://github.com/adap/flower/tree/main/examples/xgboost-quickstart
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is required for XGBoost FL. Install with: pip install xgboost")
    
    # Default XGBoost parameters
    default_params = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 4,
        "num_parallel_tree": 1,
        "subsample": 1.0,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)
    
    app = ClientApp()
    
    def _local_boost(bst_input, num_local_round, train_dmatrix):
        """
        Update trees based on local training data.
        
        After training, extract the last N=num_local_round trees for server aggregation.
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
        """
        for _ in range(num_local_round):
            bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
        
        # Extract last N trees for aggregation (bagging)
        bst = bst_input[
            bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()
        ]
        return bst
    
    @app.train()
    def train(msg: Message, context: Context) -> Message:
        """
        Train XGBoost model on local data.
        
        First round: Train from scratch
        Subsequent rounds: Load global model and add local trees
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
        """
        # Get partition info
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 10)
        
        # Get config
        num_local_round = int(context.run_config.get("local-epochs", default_local_rounds))
        
        # Load data
        train_dmatrix, _, num_train, _ = load_data_fn(partition_id, num_partitions)
        
        # Get server round from config
        config = msg.content.get("config", ConfigRecord())
        server_round = int(config.get("server-round", 1))
        
        # Get global model from message
        arrays = msg.content.get("arrays", ArrayRecord())
        
        if server_round == 1 or len(arrays) == 0:
            # First round: train from scratch
            bst = xgb.train(
                default_params,
                train_dmatrix,
                num_boost_round=num_local_round,
            )
        else:
            # Load global model and continue training
            bst = xgb.Booster(params=default_params)
            global_model = bytearray(arrays["0"].numpy().tobytes())
            bst.load_model(global_model)
            
            # Local training (add trees)
            bst = _local_boost(bst, num_local_round, train_dmatrix)
        
        # Save model as bytes
        local_model = bst.save_raw("json")
        model_np = np.frombuffer(local_model, dtype=np.uint8)
        
        # Build reply
        model_record = ArrayRecord()
        model_record["0"] = model_np
        
        metrics = MetricRecord({"num_examples": num_train})
        content = RecordDict({"arrays": model_record, "metrics": metrics})
        
        return Message(content=content, reply_to=msg)
    
    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        """
        Evaluate XGBoost model on local validation data.
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
        """
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 10)
        
        # Load validation data
        _, valid_dmatrix, _, num_val = load_data_fn(partition_id, num_partitions)
        
        # Load model
        arrays = msg.content.get("arrays", ArrayRecord())
        if len(arrays) == 0:
            # No model yet
            metrics = MetricRecord({"loss": float("inf"), "num_examples": 0})
            content = RecordDict({"metrics": metrics})
            return Message(content=content, reply_to=msg)
        
        bst = xgb.Booster(params=default_params)
        global_model = bytearray(arrays["0"].numpy().tobytes())
        bst.load_model(global_model)
        
        # Evaluate
        eval_result = bst.eval(valid_dmatrix)
        # Parse eval result string like "eval-auc:0.85"
        try:
            metric_value = float(eval_result.split(":")[-1])
        except:
            metric_value = 0.0
        
        metrics = MetricRecord({
            "eval_metric": metric_value,
            "num_examples": num_val,
        })
        content = RecordDict({"metrics": metrics})
        
        return Message(content=content, reply_to=msg)
    
    return app


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# FLOWER CLIENT CLASS (for sequential simulation on Windows)
# =============================================================================

class FlowerClient:
    """
    NumPy-based Flower client for sequential FL simulation.
    
    This class is used by the sequential simulation fallback on Windows,
    where Ray-based simulation doesn't work well.
    
    Reference: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    """
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        device: Optional[torch.device] = None,
        proximal_mu: float = 0.0,
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device or get_device()
        self.proximal_mu = proximal_mu
        self.global_params = None
    
    def get_parameters(self, config: Dict[str, Any] = None) -> list:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: list) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Store global params for FedProx
        if self.proximal_mu > 0:
            self.global_params = [torch.tensor(p) for p in parameters]
    
    def fit(self, parameters: list, config: Dict[str, Any]) -> Tuple[list, int, Dict[str, float]]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters as list of NumPy arrays
            config: Training configuration
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        loss, num_samples = train_pytorch(
            model=self.model,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            lr=self.learning_rate,
            device=self.device,
            proximal_mu=self.proximal_mu,
            global_params=self.global_params,
        )
        
        return self.get_parameters(), num_samples, {"loss": loss}
    
    def evaluate(self, parameters: list, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local validation data.
        
        Args:
            parameters: Model parameters as list of NumPy arrays
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        loss, accuracy, num_samples = evaluate_pytorch(
            model=self.model,
            testloader=self.valloader,
            device=self.device,
        )
        
        return loss, num_samples, {"accuracy": accuracy}


# Aliases for backward compatibility
train_model = train_pytorch
evaluate_model = evaluate_pytorch
