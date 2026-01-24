"""FedMD: Federated Model Distillation Strategy.

FedMD enables heterogeneous FL by using a public dataset for knowledge transfer.
Each client trains on their private data and shares predictions on the public dataset.

Reference: Li & Wang, "FedMD: Heterogeneous Federated Learning via Model Distillation" (2019)

Verification References:
- Original Paper: https://arxiv.org/abs/1910.03581
- Implementation based on Flower Strategy interface
- Parameters verified: temperature (softmax temperature), num_classes, public_dataset_size,
  digest_epochs (private data training), revisit_epochs (distillation training)
- Note: Custom implementation following FedMD paper; not a built-in Flower strategy

Key Features:
- Two-phase training: private data + public data distillation
- Clients can have completely different model architectures
- Uses consensus on public dataset predictions
- Supports both labeled and unlabeled public datasets
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


class FedMDStrategy(Strategy):
    """Federated Model Distillation Strategy.
    
    FedMD works in two phases per round:
    1. Digest Phase: Clients train on their private data
    2. Revisit Phase: Clients compute predictions on public dataset,
       server aggregates, and clients distill from consensus
    
    This enables FL with heterogeneous model architectures.
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
        temperature: float = 1.0,
        num_classes: int = 10,
        public_dataset_size: int = 5000,
        digest_epochs: int = 1,
        revisit_epochs: int = 1,
    ):
        """Initialize FedMD strategy.
        
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
            temperature: Temperature for softmax
            num_classes: Number of output classes
            public_dataset_size: Size of public dataset
            digest_epochs: Epochs for private data training
            revisit_epochs: Epochs for distillation training
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
        self.num_classes = num_classes
        self.public_dataset_size = public_dataset_size
        self.digest_epochs = digest_epochs
        self.revisit_epochs = revisit_epochs
        
        # Consensus predictions (average of all client predictions)
        self.consensus_predictions: Optional[np.ndarray] = None
        
        # Track client contributions for weighted averaging
        self.client_weights: Dict[str, float] = {}
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize with uniform predictions (no prior knowledge)."""
        # Start with uniform distribution over classes
        initial_predictions = np.ones((self.public_dataset_size, self.num_classes)) / self.num_classes
        return ndarrays_to_parameters([initial_predictions])
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for FedMD training."""
        
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Configuration for FedMD
        config = {
            "server_round": server_round,
            "temperature": self.temperature,
            "digest_epochs": self.digest_epochs,
            "revisit_epochs": self.revisit_epochs,
            "phase": "both",  # digest + revisit
        }
        if self.on_fit_config_fn:
            config.update(self.on_fit_config_fn(server_round))
        
        # Send consensus predictions to clients
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client predictions to form consensus.
        
        FedMD aggregates predictions (soft labels) on the public dataset
        to create a consensus that all clients will learn from.
        """
        if not results:
            return None, {}
        
        # Collect predictions from all clients
        all_predictions = []
        weights = []
        
        for client, fit_res in results:
            client_predictions = parameters_to_ndarrays(fit_res.parameters)
            if client_predictions and len(client_predictions) > 0:
                all_predictions.append(client_predictions[0])
                # Weight by number of private training samples
                weights.append(fit_res.num_examples)
        
        if not all_predictions:
            return None, {}
        
        # Compute weighted consensus (average predictions)
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        stacked = np.stack(all_predictions, axis=0)
        weights_array = np.array(weights).reshape(-1, 1, 1)
        self.consensus_predictions = np.sum(stacked * weights_array, axis=0)
        
        # Apply temperature scaling to sharpen/soften consensus
        if self.temperature != 1.0:
            # Convert to logits, scale, convert back to probabilities
            eps = 1e-10
            logits = np.log(self.consensus_predictions + eps)
            scaled_logits = logits / self.temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            self.consensus_predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        logger.info(f"[FedMD] Round {server_round}: Computed consensus from {len(results)} clients")
        
        # Aggregate metrics
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        
        metrics["num_clients"] = len(results)
        metrics["num_failures"] = len(failures)
        
        # Compute consensus entropy (measure of agreement)
        entropy = -np.sum(self.consensus_predictions * np.log(self.consensus_predictions + 1e-10), axis=1)
        metrics["avg_consensus_entropy"] = float(np.mean(entropy))
        
        return ndarrays_to_parameters([self.consensus_predictions]), metrics
    
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
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        total_examples = sum(res.num_examples for _, res in results)
        
        weighted_loss = sum(res.loss * res.num_examples for _, res in results)
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
        
        metrics = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        # Compute accuracy statistics across clients
        accuracies = [res.metrics.get("accuracy", 0) for _, res in results]
        if accuracies:
            metrics["min_accuracy"] = float(min(accuracies))
            metrics["max_accuracy"] = float(max(accuracies))
            metrics["std_accuracy"] = float(np.std(accuracies))
        
        metrics["num_clients"] = len(results)
        metrics["num_failures"] = len(failures)
        
        return avg_loss, metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation."""
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for fit."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
