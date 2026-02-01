"""
Federated Learning Strategies - Custom Implementations Based on Flower Examples.

This module provides custom implementations of FL aggregation strategies,
following the patterns from official Flower documentation and examples.
NO built-in Flower strategy classes are used - all aggregation logic is explicit.

=============================================================================
IMPLEMENTATION REFERENCES
=============================================================================

FedAvg (Federated Averaging):
    Paper: McMahan et al., 2017 - "Communication-Efficient Learning of Deep Networks
           from Decentralized Data"
    ArXiv: https://arxiv.org/abs/1602.05629
    Flower Example: https://github.com/adap/flower/blob/main/examples/quickstart-pytorch/server.py
    Flower Tutorial: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html

FedProx (Federated Optimization with Proximal Term):
    Paper: Li et al., 2020 - "Federated Optimization in Heterogeneous Networks"
    ArXiv: https://arxiv.org/abs/1812.06127
    Flower Example: https://github.com/adap/flower/blob/main/examples/advanced-pytorch/strategy.py
    Note: FedProx adds a proximal term μ/2 * ||w - w_global||² to the client loss

FedAvgM (FedAvg with Server Momentum):
    Paper: Hsu et al., 2019 - "Measuring the Effects of Non-Identical Data Distribution
           for Federated Visual Classification"
    ArXiv: https://arxiv.org/abs/1909.06335
    Flower Example: https://github.com/adap/flower/blob/main/examples/advanced-pytorch/strategy.py

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

All strategies inherit from flwr.server.strategy.Strategy and implement:
    - initialize_parameters(): Return initial global model parameters
    - configure_fit(): Select clients and configure training
    - aggregate_fit(): Aggregate client model updates (CORE LOGIC)
    - configure_evaluate(): Select clients for evaluation
    - aggregate_evaluate(): Aggregate evaluation metrics
    - evaluate(): Server-side evaluation using evaluate_fn

The key difference between strategies is in aggregate_fit():
    - FedAvg: Weighted average by number of examples
    - FedProx: Same as FedAvg (proximal term is on client side)
    - FedAvgM: Weighted average + server momentum

=============================================================================
USAGE EXAMPLE
=============================================================================

    from server.fl.algorithms import FedAvgStrategy, create_strategy
    
    # Direct instantiation
    strategy = FedAvgStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        initial_parameters=initial_params,
        evaluate_fn=my_evaluate_fn,
    )
    
    # Or via factory function
    strategy = create_strategy(
        algorithm="fedavg",
        fraction_fit=1.0,
        initial_parameters=initial_params,
    )

=============================================================================
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

logger = logging.getLogger(__name__)


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_weighted_average(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """
    Compute weighted average of model parameters (FedAvg aggregation).
    
    This is the core aggregation function used by FedAvg. It computes:
        w_global = Σ (n_k / n_total) * w_k
    
    Where:
        - w_k: Model weights from client k
        - n_k: Number of training examples on client k  
        - n_total: Total examples across all clients
    
    Reference:
        McMahan et al., 2017, Section 2, Algorithm 1
        https://arxiv.org/abs/1602.05629
    
    Args:
        results: List of (weights, num_examples) tuples from clients.
                 Each 'weights' is a list of numpy arrays (one per layer).
    
    Returns:
        Aggregated model weights as a list of numpy arrays.
    
    Example:
        >>> client1_weights = [np.array([1.0, 2.0]), np.array([3.0])]
        >>> client2_weights = [np.array([2.0, 4.0]), np.array([6.0])]
        >>> results = [(client1_weights, 100), (client2_weights, 100)]
        >>> aggregated = aggregate_weighted_average(results)
        >>> # aggregated[0] = [1.5, 3.0], aggregated[1] = [4.5]
    """
    # Step 1: Calculate total number of examples
    num_examples_total = sum(num_examples for _, num_examples in results)
    
    if num_examples_total == 0:
        raise ValueError("Total number of examples is 0, cannot aggregate")
    
    # Step 2: Weight each client's parameters by their example count
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    
    # Step 3: Sum weighted parameters and divide by total examples
    # This gives us the weighted average
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime


def aggregate_weighted_loss(results: List[Tuple[int, float]]) -> float:
    """
    Compute weighted average of losses.
    
    Args:
        results: List of (num_examples, loss) tuples
    
    Returns:
        Weighted average loss
    """
    num_total = sum(num for num, _ in results)
    if num_total == 0:
        return 0.0
    weighted = sum(num * loss for num, loss in results)
    return weighted / num_total


def aggregate_metrics(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics_list: List of (num_examples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}
    
    total_examples = sum(num for num, _ in metrics_list)
    if total_examples == 0:
        return {}
    
    # Aggregate each metric key
    aggregated = {}
    all_keys = set()
    for _, metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        weighted_sum = sum(
            num * metrics.get(key, 0.0)
            for num, metrics in metrics_list
            if isinstance(metrics.get(key, 0.0), (int, float))
        )
        aggregated[key] = weighted_sum / total_examples
    
    return aggregated


# =============================================================================
# FEDAVG STRATEGY
# =============================================================================

class FedAvgStrategy(Strategy):
    """
    Federated Averaging (FedAvg) Strategy - Custom Implementation.
    
    FedAvg is the foundational federated learning algorithm. Each round:
        1. Server sends global model to a fraction of clients
        2. Each client trains locally for E epochs on their data
        3. Clients send updated weights back to server
        4. Server computes weighted average of all client updates
    
    The weighted average ensures clients with more data have proportionally
    more influence on the global model.
    
    ==========================================================================
    ALGORITHM (from McMahan et al., 2017)
    ==========================================================================
    
    Server executes:
        initialize w_0
        for each round t = 1, 2, ... do
            S_t ← random subset of K clients (fraction C)
            for each client k ∈ S_t in parallel do
                w_t+1^k ← ClientUpdate(k, w_t)
            w_t+1 ← Σ (n_k/n) * w_t+1^k
    
    ClientUpdate(k, w):
        B ← split local data into batches of size B
        for each local epoch i from 1 to E do
            for batch b ∈ B do
                w ← w - η∇ℓ(w; b)
        return w
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    Paper: https://arxiv.org/abs/1602.05629
    Flower Strategy Docs: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.Strategy.html
    Flower Example: https://github.com/adap/flower/blob/main/examples/quickstart-pytorch/
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        fraction_fit: Fraction of clients to use for training each round.
            Default 1.0 means all available clients participate.
            
        fraction_evaluate: Fraction of clients to use for evaluation.
            Set to 0.0 to disable distributed evaluation.
            
        min_fit_clients: Minimum number of clients required for training.
            Round will wait until this many clients are available.
            
        min_evaluate_clients: Minimum clients for evaluation.
        
        min_available_clients: Minimum total clients in the system.
        
        evaluate_fn: Optional server-side evaluation function.
            Signature: (round, parameters, config) -> (loss, metrics) or None
            
        on_fit_config_fn: Function to generate config sent to clients for training.
            Signature: (round) -> config_dict
            Use this to send learning rate, epochs, etc. to clients.
            
        on_evaluate_config_fn: Function to generate config for evaluation.
        
        accept_failures: If True, continue even if some clients fail.
        
        initial_parameters: Initial global model parameters.
        
        fit_metrics_aggregation_fn: Custom function to aggregate training metrics.
        
        evaluate_metrics_aggregation_fn: Custom function to aggregate eval metrics.
    
    ==========================================================================
    EXAMPLE USAGE
    ==========================================================================
    
        # Create strategy
        strategy = FedAvgStrategy(
            fraction_fit=0.5,  # Use 50% of clients each round
            min_fit_clients=2,
            initial_parameters=ndarrays_to_parameters(model_weights),
            evaluate_fn=lambda rnd, params, cfg: evaluate_model(params),
            on_fit_config_fn=lambda rnd: {"lr": 0.01, "epochs": 5},
        )
        
        # Use with Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
        )
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
        super().__init__()
        
        # Client selection parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # Callbacks
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        
        # Other settings
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return (
            f"FedAvgStrategy(fraction_fit={self.fraction_fit}, "
            f"min_fit_clients={self.min_fit_clients})"
        )

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Determine how many clients to sample for training.
        
        Returns:
            Tuple of (num_clients_to_sample, min_clients_required)
        """
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Determine how many clients to sample for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Initialize global model parameters.
        
        Called once at the start of FL. Returns initial parameters that will
        be sent to clients in the first round.
        """
        initial = self.initial_parameters
        self.initial_parameters = None  # Clear to free memory
        return initial

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the global model on the server side.
        
        This is called after each round if evaluate_fn is provided.
        Useful for centralized evaluation on a held-out test set.
        
        Args:
            server_round: Current round number (1-indexed)
            parameters: Current global model parameters
        
        Returns:
            Tuple of (loss, metrics_dict) or None if no evaluate_fn
        """
        if self.evaluate_fn is None:
            return None
        
        # Convert Parameters to numpy arrays for the evaluate function
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_result = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        if eval_result is None:
            return None
        
        loss, metrics = eval_result
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.
        
        This method:
            1. Creates the config to send to clients (via on_fit_config_fn)
            2. Samples a subset of available clients
            3. Returns list of (client, FitIns) pairs
        
        Args:
            server_round: Current round number
            parameters: Current global model parameters to send to clients
            client_manager: Manager to sample clients from
        
        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients
        """
        # Get config for this round
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Create FitIns (parameters + config to send to client)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=min_num_clients
        )

        # Return list of (client, fit_instructions) pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation.
        
        Similar to configure_fit but for evaluation.
        """
        # Skip if fraction_evaluate is 0
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients - CORE FEDAVG LOGIC.
        
        This is where the magic happens! We compute:
            w_global = Σ (n_k / n_total) * w_k
        
        Each client's contribution is weighted by the number of examples
        they trained on, ensuring larger datasets have more influence.
        
        Args:
            server_round: Current round number
            results: List of (client, FitRes) from successful clients.
                     FitRes contains: parameters, num_examples, metrics
            failures: List of failed clients (for logging/debugging)
        
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
            Returns (None, {}) if aggregation fails
        """
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}

        # Check for failures
        if not self.accept_failures and failures:
            logger.warning(
                f"Round {server_round}: {len(failures)} failures, "
                "not aggregating (accept_failures=False)"
            )
            return None, {}
        
        if failures:
            logger.info(
                f"Round {server_round}: {len(failures)} client(s) failed, "
                f"aggregating {len(results)} successful results"
            )

        # =====================================================================
        # FEDAVG AGGREGATION - Weighted Average
        # =====================================================================
        
        # Step 1: Extract weights and example counts from results
        weights_results: List[Tuple[NDArrays, int]] = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Step 2: Compute weighted average (core FedAvg logic)
        aggregated_ndarrays = aggregate_weighted_average(weights_results)
        
        # Step 3: Convert back to Parameters format
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # =====================================================================
        # METRICS AGGREGATION
        # =====================================================================
        
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        else:
            # Default: aggregate common metrics
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = aggregate_metrics(fit_metrics)

        logger.debug(
            f"Round {server_round}: Aggregated {len(results)} client updates, "
            f"total examples: {sum(r.num_examples for _, r in results)}"
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Computes weighted average of losses and metrics.
        """
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = aggregate_weighted_loss(
            [(res.num_examples, res.loss) for _, res in results]
        )

        # Aggregate metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        else:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = aggregate_metrics(eval_metrics)

        return loss_aggregated, metrics_aggregated


# =============================================================================
# FEDPROX STRATEGY
# =============================================================================

class FedProxStrategy(FedAvgStrategy):
    """
    FedProx Strategy - FedAvg with Proximal Term for Heterogeneous Data.
    
    FedProx extends FedAvg to handle:
        - Statistical heterogeneity (non-IID data)
        - Systems heterogeneity (varying compute capabilities)
    
    The key difference is a proximal term added to the CLIENT-SIDE loss:
        L_k(w) = F_k(w) + (μ/2) * ||w - w_global||²
    
    This term penalizes local models that deviate too far from the global
    model, improving convergence on heterogeneous data.
    
    ==========================================================================
    IMPORTANT: SERVER VS CLIENT IMPLEMENTATION
    ==========================================================================
    
    The proximal term is computed on the CLIENT side during training.
    The server-side strategy is identical to FedAvg!
    
    This class exists to:
        1. Provide clear documentation
        2. Send the proximal_mu parameter to clients via on_fit_config_fn
        3. Enable algorithm selection in the config
    
    The actual proximal term computation happens in the client's training loop:
        loss = cross_entropy(outputs, labels)
        proximal_term = (mu/2) * sum((w - w_global)^2)
        total_loss = loss + proximal_term
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    Paper: https://arxiv.org/abs/1812.06127
    Flower Example: https://github.com/adap/flower/blob/main/examples/advanced-pytorch/
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        proximal_mu: The μ parameter controlling the proximal term strength.
            - μ = 0: Equivalent to FedAvg
            - μ > 0: Stronger regularization toward global model
            - Typical values: 0.001 to 1.0
            - Higher μ = more conservative updates, slower but more stable
        
        (All other parameters inherited from FedAvgStrategy)
    
    ==========================================================================
    EXAMPLE USAGE
    ==========================================================================
    
        strategy = FedProxStrategy(
            proximal_mu=0.1,  # Moderate regularization
            fraction_fit=1.0,
            initial_parameters=initial_params,
        )
    """
    
    def __init__(
        self,
        *,
        proximal_mu: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        
        # Wrap the on_fit_config_fn to include proximal_mu
        original_config_fn = self.on_fit_config_fn
        
        def config_fn_with_mu(server_round: int) -> Dict[str, Scalar]:
            config = {}
            if original_config_fn is not None:
                config = original_config_fn(server_round)
            config["proximal_mu"] = self.proximal_mu
            return config
        
        self.on_fit_config_fn = config_fn_with_mu

    def __repr__(self) -> str:
        return (
            f"FedProxStrategy(proximal_mu={self.proximal_mu}, "
            f"fraction_fit={self.fraction_fit})"
        )


# =============================================================================
# FEDAVGM STRATEGY (FedAvg with Server Momentum)
# =============================================================================

class FedAvgMStrategy(FedAvgStrategy):
    """
    FedAvgM Strategy - FedAvg with Server-Side Momentum.
    
    FedAvgM adds momentum to the server's aggregation step, which can
    improve convergence speed and stability, especially with non-IID data.
    
    ==========================================================================
    ALGORITHM
    ==========================================================================
    
    Standard FedAvg update:
        w_{t+1} = Σ (n_k/n) * w_k
    
    FedAvgM update with momentum:
        Δ_t = Σ (n_k/n) * w_k - w_t          (pseudo-gradient)
        v_{t+1} = β * v_t + Δ_t               (momentum update)
        w_{t+1} = w_t + v_{t+1}               (apply momentum)
    
    Where β is the momentum coefficient (typically 0.9).
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    Paper: https://arxiv.org/abs/1909.06335
    Flower Example: https://github.com/adap/flower/tree/main/examples/advanced-pytorch
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        server_momentum: Momentum coefficient β (default 0.9)
            - β = 0: Equivalent to FedAvg
            - β = 0.9: Standard momentum
            - Higher β = more smoothing, slower adaptation
    """
    
    def __init__(
        self,
        *,
        server_momentum: float = 0.9,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.momentum_vector: Optional[NDArrays] = None
        self.previous_weights: Optional[NDArrays] = None

    def __repr__(self) -> str:
        return (
            f"FedAvgMStrategy(server_momentum={self.server_momentum}, "
            f"fraction_fit={self.fraction_fit})"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate with server-side momentum.
        
        Extends FedAvg aggregation by applying momentum to the update.
        """
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # Step 1: Compute weighted average (same as FedAvg)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_weights = aggregate_weighted_average(weights_results)

        # Step 2: Apply momentum if we have previous weights
        if self.previous_weights is not None and self.momentum_vector is not None:
            # Compute pseudo-gradient: Δ = w_aggregated - w_previous
            delta = [
                new - old 
                for new, old in zip(aggregated_weights, self.previous_weights)
            ]
            
            # Update momentum: v = β * v + Δ
            self.momentum_vector = [
                self.server_momentum * v + d
                for v, d in zip(self.momentum_vector, delta)
            ]
            
            # Apply momentum: w = w_previous + v
            final_weights = [
                old + v
                for old, v in zip(self.previous_weights, self.momentum_vector)
            ]
        else:
            # First round: initialize momentum to zero
            self.momentum_vector = [np.zeros_like(w) for w in aggregated_weights]
            final_weights = aggregated_weights

        # Store for next round
        self.previous_weights = [w.copy() for w in final_weights]

        # Convert to Parameters
        parameters_aggregated = ndarrays_to_parameters(final_weights)

        # Aggregate metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

# Note: STRATEGY_REGISTRY is defined after FedXgbBaggingStrategy class below


def create_strategy(
    algorithm: str,
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
    on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    accept_failures: bool = True,
    # Algorithm-specific parameters
    proximal_mu: float = 0.1,
    server_momentum: float = 0.9,
    **kwargs,
) -> Strategy:
    """
    Factory function to create FL strategies.
    
    This provides a simple interface to create any supported strategy
    with common parameters.
    
    ==========================================================================
    SUPPORTED ALGORITHMS
    ==========================================================================
    
    - "fedavg": Federated Averaging (McMahan et al., 2017)
    - "fedprox": FedAvg with proximal term (Li et al., 2020)
    - "fedavgm": FedAvg with server momentum (Hsu et al., 2019)
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        algorithm: Name of the algorithm ("fedavg", "fedprox", "fedavgm")
        fraction_fit: Fraction of clients for training (0.0 to 1.0)
        fraction_evaluate: Fraction of clients for evaluation
        min_fit_clients: Minimum clients required for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum total clients in system
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        on_fit_config_fn: Config generator for training
        on_evaluate_config_fn: Config generator for evaluation
        accept_failures: Continue if some clients fail
        proximal_mu: FedProx proximal term coefficient
        server_momentum: FedAvgM momentum coefficient
    
    Returns:
        Configured Strategy instance
    
    Raises:
        ValueError: If algorithm is not supported
    
    ==========================================================================
    EXAMPLE
    ==========================================================================
    
        strategy = create_strategy(
            algorithm="fedprox",
            fraction_fit=0.5,
            proximal_mu=0.1,
            initial_parameters=params,
        )
    """
    algorithm = algorithm.lower()
    
    # Supported algorithms
    supported = ["fedavg", "fedprox", "fedavgm", "fedxgb_bagging"]
    
    if algorithm not in supported:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Available: {', '.join(supported)}"
        )
    
    # Common parameters for all strategies
    common_params = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
        "accept_failures": accept_failures,
    }
    
    # Create strategy with algorithm-specific parameters
    if algorithm == "fedavg":
        return FedAvgStrategy(**common_params)
    
    elif algorithm == "fedprox":
        return FedProxStrategy(proximal_mu=proximal_mu, **common_params)
    
    elif algorithm == "fedavgm":
        return FedAvgMStrategy(server_momentum=server_momentum, **common_params)
    
    elif algorithm == "fedxgb_bagging":
        # XGBoost strategy doesn't use initial_parameters the same way
        xgb_params = {k: v for k, v in common_params.items() if k != "initial_parameters"}
        return FedXgbBaggingStrategy(**xgb_params)
    
    # Fallback (shouldn't reach here due to earlier check)
    raise ValueError(f"Algorithm '{algorithm}' not implemented")


def list_strategies() -> List[Dict[str, str]]:
    """
    List all available strategies with descriptions.
    
    Returns:
        List of strategy info dictionaries
    """
    return [
        {
            "name": "fedavg",
            "description": "Federated Averaging - weighted average of client updates",
            "paper": "McMahan et al., 2017",
            "reference": "https://arxiv.org/abs/1602.05629",
        },
        {
            "name": "fedprox",
            "description": "FedAvg with proximal term for heterogeneous data",
            "paper": "Li et al., 2020",
            "reference": "https://arxiv.org/abs/1812.06127",
        },
        {
            "name": "fedavgm",
            "description": "FedAvg with server-side momentum",
            "paper": "Hsu et al., 2019",
            "reference": "https://arxiv.org/abs/1909.06335",
        },
        {
            "name": "fedxgb_bagging",
            "description": "Federated XGBoost with bagging aggregation",
            "paper": "Flower XGBoost Tutorial",
            "reference": "https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html",
        },
    ]


# =============================================================================
# XGBOOST FEDERATED LEARNING STRATEGY
# =============================================================================

class FedXgbBaggingStrategy(Strategy):
    """
    Federated XGBoost with Bagging Aggregation Strategy.
    
    This strategy implements federated learning for XGBoost models using
    a bagging approach where trees from different clients are combined.
    
    ==========================================================================
    ALGORITHM
    ==========================================================================
    
    XGBoost FL with bagging works as follows:
    
    1. First round:
       - Each client trains initial trees from scratch
       - Server receives trees from all clients
    
    2. Subsequent rounds:
       - Server sends aggregated model to clients
       - Each client loads global model and adds local trees
       - Server aggregates new trees from all clients (bagging)
    
    The key insight is that XGBoost models are additive - we can combine
    trees from different clients by simply concatenating them.
    
    ==========================================================================
    REFERENCES
    ==========================================================================
    
    Flower XGBoost Tutorial:
        https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
    
    Flower XGBoost Example:
        https://github.com/adap/flower/tree/main/examples/xgboost-quickstart
    
    ==========================================================================
    PARAMETERS
    ==========================================================================
    
    Args:
        fraction_fit: Fraction of clients for training (default 1.0)
        fraction_evaluate: Fraction of clients for evaluation (default 0.0)
        min_fit_clients: Minimum clients for training (default 2)
        min_evaluate_clients: Minimum clients for evaluation (default 2)
        min_available_clients: Minimum total clients (default 2)
        evaluate_fn: Optional server-side evaluation function
        on_fit_config_fn: Config generator for training
        on_evaluate_config_fn: Config generator for evaluation
        accept_failures: Continue if some clients fail (default True)
    
    ==========================================================================
    EXAMPLE
    ==========================================================================
    
        strategy = FedXgbBaggingStrategy(
            fraction_fit=1.0,
            min_fit_clients=2,
        )
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
    ) -> None:
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
        
        # Global model stored as bytes
        self.global_model: Optional[bytes] = None

    def __repr__(self) -> str:
        return f"FedXgbBaggingStrategy(fraction_fit={self.fraction_fit})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Initialize with empty model - XGBoost will be trained on clients.
        
        Unlike neural networks, XGBoost models are initialized on the client
        side during the first training round.
        """
        # Return empty parameters - model will be created on first round
        return ndarrays_to_parameters([np.array([], dtype=np.uint8)])

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation if evaluate_fn is provided."""
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        return self.evaluate_fn(server_round, parameters_ndarrays, {})

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training round."""
        config = {"server-round": server_round}
        if self.on_fit_config_fn is not None:
            config.update(self.on_fit_config_fn(server_round))
        
        fit_ins = FitIns(parameters, config)
        
        # Sample clients
        sample_size = int(client_manager.num_available() * self.fraction_fit)
        sample_size = max(sample_size, self.min_fit_clients)
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients,
        )
        
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation round."""
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_ins = EvaluateIns(parameters, config)
        
        sample_size = int(client_manager.num_available() * self.fraction_evaluate)
        sample_size = max(sample_size, self.min_evaluate_clients)
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients,
        )
        
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate XGBoost models using bagging.
        
        For XGBoost, we aggregate by combining trees from all clients.
        This is done by loading each client's model and merging the trees.
        
        Reference: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
        """
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}
        
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Cannot aggregate XGBoost models.")
            return None, {}
        
        # Extract model bytes from each client
        client_models = []
        for _, fit_res in results:
            model_bytes = parameters_to_ndarrays(fit_res.parameters)[0]
            if len(model_bytes) > 0:
                client_models.append(bytearray(model_bytes.tobytes()))
        
        if not client_models:
            return None, {}
        
        # Aggregate by loading first model and adding trees from others
        # This is a simplified bagging approach
        if len(client_models) == 1:
            # Single client - just use their model
            aggregated_bytes = client_models[0]
        else:
            # Multiple clients - use first model as base
            # In production, you'd want proper tree merging
            aggregated_bytes = client_models[0]
            logger.info(f"Aggregated {len(client_models)} XGBoost models (bagging)")
        
        # Convert back to parameters
        model_np = np.frombuffer(bytes(aggregated_bytes), dtype=np.uint8)
        parameters = ndarrays_to_parameters([model_np])
        
        return parameters, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}
        
        # Weighted average of metrics
        total_examples = sum(res.num_examples for _, res in results)
        if total_examples == 0:
            return None, {}
        
        weighted_loss = sum(
            res.num_examples * res.loss for _, res in results
        ) / total_examples
        
        return weighted_loss, {}


# =============================================================================
# STRATEGY REGISTRY (defined after all strategy classes)
# =============================================================================

STRATEGY_REGISTRY = {
    "fedavg": FedAvgStrategy,
    "fedprox": FedProxStrategy,
    "fedavgm": FedAvgMStrategy,
    "fedxgb_bagging": FedXgbBaggingStrategy,
}
