"""Knowledge Distillation Client for Heterogeneous FL.

This client communicates soft labels (logits) instead of model parameters,
allowing different clients to use different model architectures.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)


class KDClient(NumPyClient):
    """Knowledge Distillation Client for heterogeneous FL.
    
    Instead of sharing model parameters, this client:
    1. Receives aggregated soft labels from the server
    2. Trains locally using both hard labels and soft labels (distillation)
    3. Sends back soft labels on a public/shared dataset
    
    This allows each client to have a different model architecture.
    """
    
    _client_counter = 0
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        public_loader: DataLoader,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        device: torch.device = None,
        temperature: float = 3.0,
        distillation_weight: float = 0.5,
        client_id: Optional[str] = None,
    ):
        """Initialize the KD client.
        
        Args:
            model: Client's local model (can be any architecture)
            trainloader: Private training data loader
            valloader: Validation data loader
            public_loader: Public dataset for computing soft labels
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on
            temperature: Temperature for softmax in distillation
            distillation_weight: Weight for distillation loss (alpha)
            client_id: Optional client identifier
        """
        KDClient._client_counter += 1
        self.client_id = client_id or f"kd_client_{KDClient._client_counter}"
        
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.public_loader = public_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.train_samples = len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
        self.public_samples = len(public_loader.dataset) if hasattr(public_loader, 'dataset') else 0
        
        # Store received soft labels from server
        self.server_soft_labels = None
        
        logger.debug(f"[{self.client_id}] Initialized KD client with {self.train_samples} train samples")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return soft labels on public dataset instead of model parameters.
        
        This is the key difference from standard FL - we send logits, not weights.
        """
        self.model.to(self.device)
        self.model.eval()
        
        all_logits = []
        with torch.no_grad():
            for batch in self.public_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("image", batch.get("data")))
                else:
                    images = batch[0]
                
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits.cpu().numpy())
        
        # Concatenate all logits
        soft_labels = np.concatenate(all_logits, axis=0)
        
        # Return as a list with single array (for compatibility with Flower)
        return [soft_labels]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Receive aggregated soft labels from server.
        
        Unlike standard FL, we receive soft labels, not model weights.
        """
        if parameters and len(parameters) > 0:
            self.server_soft_labels = parameters[0]
            logger.debug(f"[{self.client_id}] Received soft labels shape: {self.server_soft_labels.shape}")
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model using both local data and distillation from server soft labels.
        
        Training combines:
        1. Cross-entropy loss on private data (hard labels)
        2. KL divergence loss on public data (soft labels from server)
        """
        self.set_parameters(parameters)
        
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("lr", self.learning_rate))
        temperature = float(config.get("temperature", self.temperature))
        alpha = float(config.get("distillation_weight", self.distillation_weight))
        server_round = int(config.get("server_round", 0))
        
        logger.debug(f"[{self.client_id}] Fit round {server_round}: epochs={epochs}, T={temperature}, alpha={alpha}")
        
        self.model.to(self.device)
        self.model.train()
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        running_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # Train on private data with hard labels
            for batch in self.trainloader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("image", batch.get("data")))
                    labels = batch.get("label", batch.get("labels"))
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion_ce(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            
            # Distillation on public data (if we have server soft labels)
            if self.server_soft_labels is not None and server_round > 0:
                public_iter = iter(self.public_loader)
                batch_idx = 0
                batch_size = self.public_loader.batch_size or 32
                
                for batch in self.public_loader:
                    if isinstance(batch, dict):
                        images = batch.get("img", batch.get("image", batch.get("data")))
                    else:
                        images = batch[0]
                    
                    images = images.to(self.device)
                    
                    # Get corresponding server soft labels
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + images.size(0), len(self.server_soft_labels))
                    
                    if end_idx <= start_idx:
                        break
                    
                    teacher_logits = torch.tensor(
                        self.server_soft_labels[start_idx:end_idx],
                        device=self.device
                    )
                    
                    optimizer.zero_grad()
                    student_logits = self.model(images[:end_idx - start_idx])
                    
                    # KL divergence loss with temperature scaling
                    student_soft = F.log_softmax(student_logits / temperature, dim=1)
                    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
                    
                    distill_loss = criterion_kl(student_soft, teacher_soft) * (temperature ** 2)
                    distill_loss = alpha * distill_loss
                    
                    distill_loss.backward()
                    optimizer.step()
                    
                    running_loss += distill_loss.item()
                    num_batches += 1
                    batch_idx += 1
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        logger.debug(f"[{self.client_id}] Fit completed: loss={avg_loss:.4f}")
        
        # Return soft labels on public dataset
        return self.get_parameters({}), self.train_samples, {"train_loss": float(avg_loss)}
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data."""
        # Note: For KD, we don't set parameters from server (they're soft labels)
        # We just evaluate our local model
        
        self.model.to(self.device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.valloader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("image", batch.get("data")))
                    labels = batch.get("label", batch.get("labels"))
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        logger.debug(f"[{self.client_id}] Evaluate: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        
        return float(avg_loss), total_samples, {"accuracy": float(accuracy)}


def create_kd_client_fn(
    model_configs: List[Dict[str, Any]],
    load_partition_fn,
    public_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device = None
):
    """Create a client function for KD-based FL simulation with heterogeneous models.
    
    Args:
        model_configs: List of model configurations, one per client
                      Each config should have 'architecture' and optional params
        load_partition_fn: Function that takes partition_id and returns (trainloader, valloader)
        public_loader: Shared public dataset loader
        config: Client configuration dictionary
        device: Device to train on
    
    Returns:
        Client function for Flower simulation
    """
    from ...core.models import get_model
    
    def client_fn(context):
        """Create a KD client for simulation."""
        partition_id = context.node_config.get("partition-id", 0)
        if isinstance(partition_id, str):
            partition_id = int(partition_id)
        
        # Load data partition
        trainloader, valloader = load_partition_fn(partition_id)
        
        # Get model config for this client (cycle if fewer configs than clients)
        model_config = model_configs[partition_id % len(model_configs)]
        
        # Create model with client-specific architecture
        model = get_model(
            architecture=model_config.get("architecture", "cnn"),
            num_classes=model_config.get("num_classes", 10),
            in_channels=model_config.get("in_channels", 3),
        )
        
        return KDClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            public_loader=public_loader,
            local_epochs=config.get("local_epochs", 5),
            learning_rate=config.get("learning_rate", 0.01),
            device=device,
            temperature=config.get("temperature", 3.0),
            distillation_weight=config.get("distillation_weight", 0.5),
            client_id=f"kd_client_{partition_id}",
        )
    
    return client_fn
