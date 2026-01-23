"""FedYogi Algorithm.

Federated Yogi - adaptive optimization with controlled adaptivity.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class FedYogiStrategy(BaseFLAlgorithm):
    """Federated Yogi (FedYogi) algorithm.
    
    Similar to FedAdam but with controlled adaptivity that
    prevents the learning rate from becoming too small.
    
    Reference: Reddi et al., "Adaptive Federated Optimization" (2021)
    """
    
    algorithm_type = "fedyogi"
    algorithm_name = "FedYogi"
    algorithm_description = "Federated Yogi - controlled adaptive optimization"
    handles_heterogeneity = True
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    def __init__(self, config: Optional[FLConfig] = None):
        super().__init__(config)
        self.m = None
        self.v = None
        self.t = 0
    
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
        """Aggregate using FedYogi."""
        if not results:
            return None, {}
        
        weights_results = []
        for _, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            weights_results.append((parameters, num_examples))
        
        delta = self._compute_delta(weights_results)
        
        if self.m is None:
            self.m = [np.zeros_like(d) for d in delta]
            self.v = [np.zeros_like(d) for d in delta]
        
        self.t += 1
        beta_1 = self.config.beta_1
        beta_2 = self.config.beta_2
        tau = self.config.tau
        server_lr = self.config.server_lr
        
        aggregated = []
        for i, d in enumerate(delta):
            self.m[i] = beta_1 * self.m[i] + (1 - beta_1) * d
            
            # Yogi update: controlled adaptivity
            self.v[i] = self.v[i] - (1 - beta_2) * np.sign(self.v[i] - d ** 2) * (d ** 2)
            
            if self.global_model_params is not None:
                new_param = self.global_model_params[i] + server_lr * self.m[i] / (np.sqrt(self.v[i]) + tau)
            else:
                new_param = weights_results[0][0][i] + server_lr * self.m[i] / (np.sqrt(np.abs(self.v[i])) + tau)
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
