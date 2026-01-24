"""FedDF: Federated Distillation Strategy.

FedDF (Federated Distillation) enables heterogeneous FL by aggregating
soft labels (logits) instead of model parameters. Each client can have
a different model architecture.

Reference: Lin et al., "Ensemble Distillation for Robust Model Fusion 
in Federated Learning" (NeurIPS 2020)

Verification References:
- Original Paper: https://arxiv.org/abs/2006.07242
- Implementation based on Flower Strategy interface
- Parameters verified: temperature (softmax temperature for distillation),
  distillation_weight (weight for KD loss), num_classes, public_dataset_size
- Note: Custom implementation following FedDF paper; not a built-in Flower strategy

Key Features:
- Clients share soft labels on a public dataset
- Server aggregates soft labels (ensemble averaging)
- Clients distill from aggregated soft labels
- No requirement for same model architecture across clients
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

import numpy as np

from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    FitIns,
    EvaluateIns,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Metrics,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class FedDFStrategy(Strategy):
    """Federated Distillation Strategy for heterogeneous FL.
    
    Instead of aggregating model parameters, FedDF:
    1. Collects soft labels from all clients on a shared public dataset
    2. Aggregates soft labels using ensemble averaging
    3. Distributes aggregated soft labels back to clients
    4. Clients use these soft labels for knowledge distillation
    
    This allows each client to have a completely different model architecture.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        temperature: float = 3.0,
        distillation_weight: float = 0.5,
        num_classes: int = 10,
        public_dataset_size: int = 5000,
    ):
        """Initialize FedDF strategy.
        
        Args:
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            evaluate_fn: Server-side evaluation function
            on_fit_config_fn: Function to configure client training
            on_evaluate_config_fn: Function to configure client evaluation
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate eval metrics
            temperature: Temperature for softmax in distillation
            distillation_weight: Weight for distillation loss
            num_classes: Number of output classes
            public_dataset_size: Size of public dataset for soft labels
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.num_classes = num_classes
        self.public_dataset_size = public_dataset_size
        
        # Aggregated soft labels (ensemble of client predictions)
        self.aggregated_soft_labels: Optional[np.ndarray] = None
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize with empty soft labels (clients will compute their own)."""
        # Initialize with zeros - will be updated after first round
        initial_soft_labels = np.zeros((self.public_dataset_size, self.num_classes))
        return ndarrays_to_parameters([initial_soft_labels])
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training with aggregated soft labels."""
        
        # Sample clients
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Create fit config
        config = {
            "server_round": server_round,
            "temperature": self.temperature,
            "distillation_weight": self.distillation_weight,
        }
        if self.on_fit_config_fn:
            config.update(self.on_fit_config_fn(server_round))
        
        # Send aggregated soft labels to all clients
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate soft labels from clients using ensemble averaging.
        
        Instead of averaging model parameters, we average the soft labels
        (logits) from all clients on the public dataset.
        """
        if not results:
            return None, {}
        
        # Collect soft labels from all clients
        all_soft_labels = []
        weights = []
        
        for client, fit_res in results:
            # Get soft labels from client (stored as parameters)
            client_soft_labels = parameters_to_ndarrays(fit_res.parameters)
            if client_soft_labels and len(client_soft_labels) > 0:
                all_soft_labels.append(client_soft_labels[0])
                weights.append(fit_res.num_examples)
        
        if not all_soft_labels:
            return None, {}
        
        # Ensemble averaging of soft labels (weighted by number of samples)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Stack and compute weighted average
        stacked = np.stack(all_soft_labels, axis=0)
        weights_array = np.array(weights).reshape(-1, 1, 1)
        self.aggregated_soft_labels = np.sum(stacked * weights_array, axis=0)
        
        logger.info(f"[FedDF] Round {server_round}: Aggregated soft labels from {len(results)} clients")
        
        # Aggregate metrics
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        
        metrics["num_clients"] = len(results)
        metrics["num_failures"] = len(failures)
        
        return ndarrays_to_parameters([self.aggregated_soft_labels]), metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )
        
        config = {"server_round": server_round}
        if self.on_evaluate_config_fn:
            config.update(self.on_evaluate_config_fn(server_round))
        
        evaluate_ins = EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}
        
        # Weighted average of losses and accuracies
        total_examples = sum(res.num_examples for _, res in results)
        
        weighted_loss = sum(
            res.loss * res.num_examples for _, res in results
        )
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
        
        # Aggregate metrics
        metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        metrics["num_clients"] = len(results)
        metrics["num_failures"] = len(failures)
        
        # Server-side evaluation if provided
        if self.evaluate_fn and self.aggregated_soft_labels is not None:
            # Note: For FedDF, server-side eval would need a server model
            # This is optional and depends on the use case
            pass
        
        return avg_loss, metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional for FedDF)."""
        if self.evaluate_fn is None:
            return None
        
        # For FedDF, server-side evaluation requires a server model
        # that can be trained on the aggregated soft labels
        return self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for fit."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
