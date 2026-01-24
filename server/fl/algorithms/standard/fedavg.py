"""FedAvg - Federated Averaging Algorithm (Explicit Implementation).

Paper: McMahan et al., 2017 - "Communication-Efficient Learning of Deep Networks
from Decentralized Data"

This is an EXPLICIT implementation of FedAvg, not using Flower's built-in FedAvg.
The aggregation logic is implemented directly following the original paper.

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html
- Original Paper: https://arxiv.org/abs/1602.05629
- Flower Source: https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedavg.html

For an example of a built-in Flower strategy wrapper, see:
- FedProx: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedProx.html
"""

import logging
from functools import reduce
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm

logger = logging.getLogger(__name__)


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average of model parameters.
    
    This is the core FedAvg aggregation function that performs weighted
    averaging based on the number of examples each client trained on.
    
    Args:
        results: List of (weights, num_examples) tuples from clients
        
    Returns:
        Aggregated model weights as NDArrays
    """
    # Calculate total number of examples used during training
    num_examples_total = sum(num_examples for _, num_examples in results)
    
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results using weighted average.
    
    Args:
        results: List of (num_examples, loss) tuples
        
    Returns:
        Weighted average loss
    """
    num_total_examples = sum(num_examples for num_examples, _ in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_examples


class FedAvgExplicit(Strategy):
    """Explicit Federated Averaging (FedAvg) strategy implementation.
    
    This is a custom implementation of FedAvg that explicitly implements
    the aggregation logic from McMahan et al., 2017, rather than using
    Flower's built-in FedAvg class.
    
    The key method is `aggregate_fit` which performs weighted averaging
    of client model updates based on the number of training examples.
    
    Algorithm:
        1. Server sends global model to selected clients
        2. Each client trains locally for E epochs
        3. Clients send updated weights back to server
        4. Server computes weighted average: w = Σ(n_k/n) * w_k
           where n_k is the number of examples on client k
           and n is the total number of examples
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Initialize FedAvg strategy.
        
        Args:
            fraction_fit: Fraction of clients used during training
            fraction_evaluate: Fraction of clients used during evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum total clients in system
            evaluate_fn: Server-side evaluation function
            on_fit_config_fn: Function to configure client training
            on_evaluate_config_fn: Function to configure client evaluation
            accept_failures: Whether to accept rounds with failures
            initial_parameters: Initial global model parameters
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate eval metrics
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return f"FedAvgExplicit(accept_failures={self.accept_failures})"

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average.
        
        This is the core FedAvg aggregation step. It computes:
            w_global = Σ (n_k / n_total) * w_k
        
        where:
            - w_k are the weights from client k
            - n_k is the number of training examples on client k
            - n_total is the total number of examples across all clients
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed client results
            
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results to (weights, num_examples) format
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Perform weighted averaging (core FedAvg logic)
        aggregated_ndarrays = aggregate(weights_results)
        
        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            logger.warning("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss using weighted average
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            logger.warning("No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


@register_algorithm
class FedAvgWrapper(BaseStrategyWrapper):
    """Federated Averaging (FedAvg) strategy wrapper.
    
    This wrapper uses an EXPLICIT implementation of FedAvg that directly
    implements the weighted averaging aggregation logic, rather than
    relying on Flower's built-in FedAvg class.
    
    FedAvg is the baseline federated learning algorithm that performs
    weighted averaging of client model updates based on the number of
    training examples each client has.
    """
    
    algorithm_id = FLAlgorithm.FEDAVG
    flower_class = FedAvgExplicit  # Our explicit implementation
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Federated Averaging (FedAvg)",
            description="Explicit implementation of FedAvg with weighted averaging aggregation",
            paper="McMahan et al., 2017",
            category="standard",
            params=["local_epochs", "learning_rate", "batch_size"],
            pros=["Simple", "Effective baseline", "Low communication", "Fully customizable"],
            cons=["Struggles with non-IID data", "No adaptivity"],
        )
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAvg strategy using explicit implementation."""
        return FedAvgExplicit(**common_params)
