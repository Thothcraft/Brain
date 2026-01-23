"""FedAdam Algorithm.

Federated Adam - adaptive server-side optimization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class FedAdamStrategy(BaseFLAlgorithm):
    """Federated Adam (FedAdam) algorithm.
    
    Uses Adam optimizer on the server side for aggregation.
    Provides adaptive learning rates for faster convergence.
    
    Reference: Reddi et al., "Adaptive Federated Optimization" (2021)
    """
    
    algorithm_type = "fedadam"
    algorithm_name = "FedAdam"
    algorithm_description = "Federated Adam - adaptive server optimization"
    handles_heterogeneity = True
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    def __init__(self, config: Optional[FLConfig] = None):
        super().__init__(config)
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "server_lr", "type": "float", "default": 0.1, "description": "Server learning rate"},
            {"name": "beta_1", "type": "float", "default": 0.9, "description": "First moment decay"},
            {"name": "beta_2", "type": "float", "default": 0.99, "description": "Second moment decay"},
            {"name": "tau", "type": "float", "default": 1e-3, "description": "Adaptivity parameter"},
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate using FedAdam."""
        if not results:
            return None, {}
        
        # Compute pseudo-gradient (difference from global model)
        weights_results = []
        for _, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            weights_results.append((parameters, num_examples))
        
        # Weighted average of client updates
        delta = self._compute_delta(weights_results)
        
        # Initialize moments if needed
        if self.m is None:
            self.m = [np.zeros_like(d) for d in delta]
            self.v = [np.zeros_like(d) for d in delta]
        
        self.t += 1
        beta_1 = self.config.beta_1
        beta_2 = self.config.beta_2
        tau = self.config.tau
        server_lr = self.config.server_lr
        
        # Update moments and compute new parameters
        aggregated = []
        for i, d in enumerate(delta):
            self.m[i] = beta_1 * self.m[i] + (1 - beta_1) * d
            self.v[i] = beta_2 * self.v[i] + (1 - beta_2) * (d ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta_1 ** self.t)
            v_hat = self.v[i] / (1 - beta_2 ** self.t)
            
            # Update
            if self.global_model_params is not None:
                new_param = self.global_model_params[i] + server_lr * m_hat / (np.sqrt(v_hat) + tau)
            else:
                new_param = weights_results[0][0][i] + server_lr * m_hat / (np.sqrt(v_hat) + tau)
            aggregated.append(new_param)
        
        self.global_model_params = aggregated
        self.current_round = server_round
        
        return aggregated, {"num_clients": len(results), "server_lr": server_lr}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}
        
        total_examples = sum(r[1].get("num_examples", 1) for r in results)
        weighted_loss = sum(r[1].get("loss", 0) * r[1].get("num_examples", 1) for r in results)
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
        
        weighted_acc = sum(r[1].get("accuracy", 0) * r[1].get("num_examples", 1) for r in results)
        avg_accuracy = weighted_acc / total_examples if total_examples > 0 else 0
        
        return avg_loss, {"accuracy": avg_accuracy, "num_clients": len(results)}
    
    def _compute_delta(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average delta (pseudo-gradient)."""
        total_examples = sum(num for _, num in weights_results)
        delta = [np.zeros_like(layer) for layer in weights_results[0][0]]
        
        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(parameters):
                if self.global_model_params is not None:
                    delta[i] += (layer - self.global_model_params[i]) * weight
                else:
                    delta[i] += layer * weight
        
        return delta
