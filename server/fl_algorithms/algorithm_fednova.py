"""FedNova Algorithm.

Federated Normalized Averaging - handles heterogeneous local steps.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class FedNovaStrategy(BaseFLAlgorithm):
    """Federated Nova (FedNova) algorithm.
    
    Normalizes client updates by the number of local steps to handle
    heterogeneous local computation across clients.
    
    Reference: Wang et al., "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization" (2020)
    """
    
    algorithm_type = "fednova"
    algorithm_name = "FedNova"
    algorithm_description = "Federated Normalized Averaging - handles heterogeneous local steps"
    handles_heterogeneity = True
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "fraction_fit", "type": "float", "default": 0.5, "description": "Fraction of clients for training"},
            {"name": "local_epochs", "type": "int", "default": 1, "description": "Local training epochs per round"},
            {"name": "rho", "type": "float", "default": 0.0, "description": "Proximal regularization (0 = pure FedNova)"},
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate using normalized averaging."""
        if not results:
            return None, {}
        
        # Collect parameters, examples, and local steps
        weights_results = []
        total_tau = 0  # Total normalized steps
        
        for _, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            local_steps = fit_res.get("local_steps", self.config.local_epochs)
            
            # Compute tau_i (normalized coefficient)
            tau_i = local_steps
            total_tau += tau_i * num_examples
            
            weights_results.append((parameters, num_examples, tau_i))
        
        # Normalized aggregation
        aggregated = self._normalized_average(weights_results, total_tau)
        
        self.global_model_params = aggregated
        self.current_round = server_round
        
        return aggregated, {
            "num_clients": len(results),
            "num_failures": len(failures),
            "total_normalized_steps": total_tau,
        }
    
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
    
    def _normalized_average(
        self,
        weights_results: List[Tuple[List[np.ndarray], int, int]],
        total_tau: float,
    ) -> List[np.ndarray]:
        """Compute normalized weighted average."""
        aggregated = [np.zeros_like(layer) for layer in weights_results[0][0]]
        
        for parameters, num_examples, tau_i in weights_results:
            # Weight by normalized local steps
            weight = (tau_i * num_examples) / total_tau if total_tau > 0 else 1.0 / len(weights_results)
            
            for i, layer in enumerate(parameters):
                if self.global_model_params is not None:
                    # Compute normalized gradient and apply
                    delta = (layer - self.global_model_params[i]) / tau_i
                    aggregated[i] += delta * weight * tau_i
                else:
                    aggregated[i] += layer * weight
        
        # If we have global params, add the update to them
        if self.global_model_params is not None:
            for i in range(len(aggregated)):
                aggregated[i] = self.global_model_params[i] + aggregated[i]
        
        return aggregated
