"""SCAFFOLD Algorithm.

Stochastic Controlled Averaging for Federated Learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class ScaffoldStrategy(BaseFLAlgorithm):
    """SCAFFOLD algorithm.
    
    Uses control variates to reduce client drift caused by
    heterogeneous data distributions.
    
    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning" (2020)
    """
    
    algorithm_type = "scaffold"
    algorithm_name = "SCAFFOLD"
    algorithm_description = "Stochastic Controlled Averaging - reduces client drift"
    handles_heterogeneity = True
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    def __init__(self, config: Optional[FLConfig] = None):
        super().__init__(config)
        self.server_control = None  # Server control variate
        self.client_controls = {}   # Client control variates
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "server_lr", "type": "float", "default": 1.0, "description": "Server learning rate"},
            {"name": "fraction_fit", "type": "float", "default": 0.5, "description": "Fraction of clients for training"},
            {"name": "local_epochs", "type": "int", "default": 1, "description": "Local training epochs per round"},
        ]
    
    def configure_fit(self, server_round: int, parameters: List[np.ndarray], client_manager: Any) -> Dict[str, Any]:
        """Configure fit with control variates."""
        config = super().configure_fit(server_round, parameters, client_manager)
        config["server_control"] = self.server_control
        return config
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate using SCAFFOLD."""
        if not results:
            return None, {}
        
        weights_results = []
        control_deltas = []
        
        for client_id, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            control_delta = fit_res.get("control_delta", None)
            
            weights_results.append((parameters, num_examples))
            if control_delta is not None:
                control_deltas.append(control_delta)
                # Update client control
                self.client_controls[client_id] = fit_res.get("new_control", None)
        
        # Aggregate model parameters
        aggregated = self._weighted_average(weights_results)
        
        # Update server control variate
        if control_deltas and self.server_control is not None:
            n_clients = len(control_deltas)
            for i in range(len(self.server_control)):
                delta_sum = sum(cd[i] for cd in control_deltas) / n_clients
                self.server_control[i] = self.server_control[i] + delta_sum
        elif self.server_control is None and aggregated:
            # Initialize server control to zeros
            self.server_control = [np.zeros_like(p) for p in aggregated]
        
        self.global_model_params = aggregated
        self.current_round = server_round
        
        return aggregated, {"num_clients": len(results), "num_failures": len(failures)}
    
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
    
    def _weighted_average(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        total_examples = sum(num for _, num in weights_results)
        aggregated = [np.zeros_like(layer) for layer in weights_results[0][0]]
        
        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(parameters):
                aggregated[i] += layer * weight
        
        return aggregated
