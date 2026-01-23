"""Flower Client Implementation for Federated Learning.

This module provides the FlowerClient class that implements Flower's NumPyClient
interface for local training and evaluation.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    proximal_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None
) -> Tuple[float, int]:
    """Train the model on the training set.
    
    Args:
        model: PyTorch model to train
        trainloader: Training data loader
        epochs: Number of local epochs
        lr: Learning rate
        device: Device to train on
        optimizer_name: Optimizer to use (sgd, adam, adamw)
        momentum: Momentum for SGD
        weight_decay: Weight decay for regularization
        proximal_mu: FedProx proximal term coefficient
        global_params: Global model parameters for FedProx
    
    Returns:
        Tuple of (average_loss, num_samples)
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    running_loss = 0.0
    num_batches = 0
    
    for _ in range(epochs):
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
            
            # Add proximal term for FedProx
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_p, global_p in zip(model.parameters(), global_params):
                    proximal_term += ((local_p - global_p.to(device)) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    num_samples = len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
    
    return avg_loss, num_samples


def evaluate_model(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, int]:
    """Evaluate the model on the test set.
    
    Args:
        model: PyTorch model to evaluate
        testloader: Test data loader
        device: Device to evaluate on
    
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
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("image", batch.get("data")))
                labels = batch.get("label", batch.get("labels"))
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, total_samples


class FlowerClient(NumPyClient):
    """Flower NumPyClient for federated learning.
    
    This client handles local training and evaluation using PyTorch models.
    It supports FedProx proximal term and various optimizers.
    """
    
    _client_counter = 0
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        device: torch.device = None,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        proximal_mu: float = 0.0,
        client_id: Optional[str] = None,
    ):
        """Initialize the Flower client.
        
        Args:
            model: PyTorch model to train
            trainloader: Training data loader
            valloader: Validation data loader
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on (auto-detected if None)
            optimizer: Optimizer name (sgd, adam, adamw)
            momentum: Momentum for SGD
            weight_decay: Weight decay for regularization
            proximal_mu: FedProx proximal term coefficient
            client_id: Optional client identifier
        """
        FlowerClient._client_counter += 1
        self.client_id = client_id or f"client_{FlowerClient._client_counter}"
        
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.proximal_mu = proximal_mu
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.train_samples = len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
        self.val_samples = len(valloader.dataset) if hasattr(valloader, 'dataset') else 0
        
        logger.debug(f"[{self.client_id}] Initialized with {self.train_samples} train, {self.val_samples} val samples")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model on local data.
        
        Args:
            parameters: Current global model parameters
            config: Configuration from server (may override local settings)
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        # Get config from server or use defaults
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("lr", config.get("learning_rate", self.learning_rate)))
        proximal_mu = float(config.get("proximal_mu", self.proximal_mu))
        server_round = int(config.get("server_round", 0))
        
        logger.debug(f"[{self.client_id}] Fit round {server_round}: epochs={epochs}, lr={lr}")
        
        # Store global parameters for FedProx
        global_params = None
        if proximal_mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]
        
        # Train
        train_loss, num_samples = train_model(
            model=self.model,
            trainloader=self.trainloader,
            epochs=epochs,
            lr=lr,
            device=self.device,
            optimizer_name=self.optimizer_name,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            proximal_mu=proximal_mu,
            global_params=global_params
        )
        
        logger.debug(f"[{self.client_id}] Fit completed: loss={train_loss:.4f}")
        
        return self.get_parameters({}), num_samples, {"train_loss": float(train_loss)}
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data.
        
        Args:
            parameters: Current global model parameters
            config: Configuration from server
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        loss, accuracy, num_samples = evaluate_model(
            model=self.model,
            testloader=self.valloader,
            device=self.device
        )
        
        logger.debug(f"[{self.client_id}] Evaluate: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return float(loss), num_samples, {"accuracy": float(accuracy)}


def create_client_fn(
    model_fn,
    load_partition_fn,
    config: Dict[str, Any],
    device: torch.device = None
):
    """Create a client function for Flower simulation.
    
    Args:
        model_fn: Function that returns a new model instance
        load_partition_fn: Function that takes partition_id and returns (trainloader, valloader)
        config: Client configuration dictionary
        device: Device to train on
    
    Returns:
        Client function for Flower simulation
    """
    def client_fn(context):
        """Create a Flower client for simulation."""
        from flwr.common import Context
        
        # Get partition ID from node config
        partition_id = context.node_config.get("partition-id", 0)
        if isinstance(partition_id, str):
            partition_id = int(partition_id)
        
        # Load data partition
        trainloader, valloader = load_partition_fn(partition_id)
        
        # Create model
        model = model_fn()
        
        return FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            local_epochs=config.get("local_epochs", 5),
            learning_rate=config.get("learning_rate", 0.01),
            device=device,
            optimizer=config.get("optimizer", "sgd"),
            proximal_mu=config.get("proximal_mu", 0.0),
            client_id=f"client_{partition_id}",
        )
    
    return client_fn
