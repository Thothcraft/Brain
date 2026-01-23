"""FedProx Algorithm.

Federated Proximal - handles statistical heterogeneity.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class FedProxStrategy(BaseFLAlgorithm):
    """Federated Proximal (FedProx) algorithm.
    
    Adds a proximal term to handle statistical heterogeneity
    across clients. The proximal term penalizes large deviations
    from the global model.
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
    """
    
    algorithm_type = "fedprox"
    algorithm_name = "FedProx"
    algorithm_description = "Federated Proximal - handles data heterogeneity"
    handles_heterogeneity = True
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "proximal_mu", "type": "float", "default": 0.01, "description": "Proximal term coefficient (μ)"},
            {"name": "fraction_fit", "type": "float", "default": 0.5, "description": "Fraction of clients for training"},
            {"name": "local_epochs", "type": "int", "default": 1, "description": "Local training epochs per round"},
        ]
    
    def configure_fit(self, server_round: int, parameters: List[np.ndarray], client_manager: Any) -> Dict[str, Any]:
        """Configure fit with proximal term."""
        config = super().configure_fit(server_round, parameters, client_manager)
        config["proximal_mu"] = self.config.proximal_mu
        config["global_model"] = parameters  # Send global model for proximal term
        return config
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate using weighted averaging (same as FedAvg)."""
        if not results:
            return None, {}
        
        weights_results = []
        for _, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            weights_results.append((parameters, num_examples))
        
        aggregated = self._weighted_average(weights_results)
        
        total_examples = sum(num for _, num in weights_results)
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "total_examples": total_examples,
            "proximal_mu": self.config.proximal_mu,
        }
        
        self.current_round = server_round
        return aggregated, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}
        
        total_examples = sum(r[1].get("num_examples", 1) for r in results)
        weighted_loss = sum(r[1].get("loss", 0) * r[1].get("num_examples", 1) for r in results)
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
        
        weighted_acc = sum(r[1].get("accuracy", 0) * r[1].get("num_examples", 1) for r in results)
        avg_accuracy = weighted_acc / total_examples if total_examples > 0 else 0
        
        return avg_loss, {"accuracy": avg_accuracy, "num_clients": len(results)}
    
    def _weighted_average(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        total_examples = sum(num for _, num in weights_results)
        aggregated = [np.zeros_like(layer) for layer in weights_results[0][0]]
        
        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(parameters):
                aggregated[i] += layer * weight
        
        return aggregated
