"""FedAvg Algorithm.

Federated Averaging - the baseline FL algorithm.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    BaseFLAlgorithm,
    FLConfig,
    register_algorithm,
)


@register_algorithm
class FedAvgStrategy(BaseFLAlgorithm):
    """Federated Averaging (FedAvg) algorithm.
    
    The baseline FL algorithm that averages client model updates
    weighted by the number of samples.
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (2017)
    """
    
    algorithm_type = "fedavg"
    algorithm_name = "FedAvg"
    algorithm_description = "Federated Averaging - baseline FL algorithm"
    handles_heterogeneity = False
    communication_efficient = False
    privacy_preserving = False
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "fraction_fit", "type": "float", "default": 0.5, "description": "Fraction of clients for training"},
            {"name": "fraction_evaluate", "type": "float", "default": 0.5, "description": "Fraction of clients for evaluation"},
            {"name": "min_fit_clients", "type": "int", "default": 2, "description": "Minimum clients for training"},
            {"name": "local_epochs", "type": "int", "default": 1, "description": "Local training epochs per round"},
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Dict[str, Any]]],
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate using weighted averaging."""
        if not results:
            return None, {}
        
        # Extract weights and parameters
        weights_results = []
        for _, fit_res in results:
            parameters = fit_res.get("parameters", [])
            num_examples = fit_res.get("num_examples", 1)
            weights_results.append((parameters, num_examples))
        
        # Weighted average
        aggregated = self._weighted_average(weights_results)
        
        # Aggregate metrics
        total_examples = sum(num for _, num in weights_results)
        aggregated_metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "total_examples": total_examples,
        }
        
        # Average client metrics
        if results and "metrics" in results[0][1]:
            metric_keys = results[0][1]["metrics"].keys()
            for key in metric_keys:
                values = [r[1]["metrics"].get(key, 0) * r[1].get("num_examples", 1) for r in results]
                aggregated_metrics[f"avg_{key}"] = sum(values) / total_examples if total_examples > 0 else 0
        
        self.current_round = server_round
        return aggregated, aggregated_metrics
    
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
        
        # Weighted average loss
        weighted_loss = sum(
            r[1].get("loss", 0) * r[1].get("num_examples", 1) for r in results
        )
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
        
        # Weighted average accuracy
        weighted_acc = sum(
            r[1].get("accuracy", 0) * r[1].get("num_examples", 1) for r in results
        )
        avg_accuracy = weighted_acc / total_examples if total_examples > 0 else 0
        
        metrics = {
            "num_clients": len(results),
            "num_failures": len(failures),
            "total_examples": total_examples,
            "accuracy": avg_accuracy,
        }
        
        return avg_loss, metrics
    
    def _weighted_average(
        self,
        weights_results: List[Tuple[List[np.ndarray], int]],
    ) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        total_examples = sum(num for _, num in weights_results)
        
        # Initialize with zeros
        aggregated = [np.zeros_like(layer) for layer in weights_results[0][0]]
        
        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, layer in enumerate(parameters):
                aggregated[i] += layer * weight
        
        return aggregated
