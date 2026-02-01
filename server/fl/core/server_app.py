"""Simplified Flower ServerApp using @app.main decorator.

This module provides a modern Flower server implementation following the
quickstart-pytorch pattern from https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
"""

import logging
from typing import Dict, Any, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

logger = logging.getLogger(__name__)


def evaluate_fn_factory(
    model_fn: Callable[[], nn.Module],
    testloader: DataLoader,
    device: torch.device,
) -> Callable:
    """Create a centralized evaluation function.
    
    Args:
        model_fn: Function that returns a new model instance
        testloader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Evaluation function for the strategy
    """
    def evaluate(server_round: int, parameters, config) -> tuple:
        """Evaluate the global model on centralized test data."""
        model = model_fn()
        
        # Convert parameters to state dict
        params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(parameters))
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
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
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate


def create_server_app(
    model_fn: Callable[[], nn.Module],
    testloader: DataLoader,
    num_rounds: int = 3,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    learning_rate: float = 0.01,
    device: Optional[torch.device] = None,
    on_round_end: Optional[Callable[[int, float, float], None]] = None,
) -> ServerApp:
    """Create a Flower ServerApp with FedAvg strategy.
    
    Args:
        model_fn: Function that returns a new model instance
        testloader: Test data loader for centralized evaluation
        num_rounds: Number of FL rounds
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation
        learning_rate: Learning rate to send to clients
        device: Device for evaluation (auto-detected if None)
        on_round_end: Optional callback(round, loss, accuracy) after each round
    
    Returns:
        Configured ServerApp
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get initial parameters from model
    initial_model = model_fn()
    initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_weights)
    
    # Create evaluation function
    evaluate_fn = evaluate_fn_factory(model_fn, testloader, device)
    
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )
    
    # Create ServerApp
    server_app = ServerApp(
        config={"num-server-rounds": num_rounds},
        strategy=strategy,
    )
    
    return server_app


def create_simple_fl_app(
    model_fn: Callable[[], nn.Module],
    load_data_fn: Callable,
    testloader: DataLoader,
    num_rounds: int = 3,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> tuple:
    """Create both ClientApp and ServerApp for a simple FL setup.
    
    This is a convenience function that creates both apps configured
    to work together.
    
    Args:
        model_fn: Function that returns a new model instance
        load_data_fn: Function(partition_id, num_partitions, batch_size) -> (train, val)
        testloader: Test data loader for centralized evaluation
        num_rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        batch_size: Batch size for training
        device: Device (auto-detected if None)
    
    Returns:
        Tuple of (client_app, server_app)
    """
    from .client_app import create_client_app
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    client_app = create_client_app(
        model_fn=model_fn,
        load_data_fn=load_data_fn,
        default_epochs=local_epochs,
        default_lr=learning_rate,
    )
    
    server_app = create_server_app(
        model_fn=model_fn,
        testloader=testloader,
        num_rounds=num_rounds,
        learning_rate=learning_rate,
        device=device,
    )
    
    return client_app, server_app
