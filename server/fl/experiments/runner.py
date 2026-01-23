"""FL Experiment Runner for Multi-Model and Multi-Run Experiments.

This module provides the FLExperimentRunner class that handles:
- Running single or multiple FL experiments
- Queue-based execution for multiple models/configurations
- Multiple runs per experiment for statistical analysis
- Per-model and comparative result reporting
"""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Scalar,
)
from flwr.server import ServerConfig as FlwrServerConfig, ServerApp, ServerAppComponents
from flwr.client import ClientApp
from flwr.simulation import run_simulation

from ..core.config import ExperimentConfig, FLAlgorithm
from ..core.models import get_model
from ..core.client import FlowerClient, evaluate_model
from ..algorithms import create_strategy
from ..datasets import load_partition, load_centralized_testset, get_dataset_info

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    loss: float
    accuracy: float
    train_loss: float = 0.0
    num_clients: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunResult:
    """Result of a single experiment run."""
    run_id: int
    seed: int
    final_accuracy: float
    best_accuracy: float
    best_round: int
    total_rounds: int
    round_metrics: List[RoundMetrics] = field(default_factory=list)
    training_time_seconds: float = 0.0
    status: str = "completed"
    error_message: Optional[str] = None


@dataclass
class ExperimentResult:
    """Result of an experiment (potentially multiple runs)."""
    experiment_id: str
    config: ExperimentConfig
    runs: List[RunResult] = field(default_factory=list)
    
    # Aggregated statistics across runs
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    mean_best_accuracy: float = 0.0
    std_best_accuracy: float = 0.0
    
    # Timing
    total_time_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def compute_statistics(self):
        """Compute aggregate statistics from runs."""
        if not self.runs:
            return
        
        final_accs = [r.final_accuracy for r in self.runs if r.status == "completed"]
        best_accs = [r.best_accuracy for r in self.runs if r.status == "completed"]
        
        if final_accs:
            self.mean_accuracy = float(np.mean(final_accs))
            self.std_accuracy = float(np.std(final_accs))
            self.min_accuracy = float(np.min(final_accs))
            self.max_accuracy = float(np.max(final_accs))
        
        if best_accs:
            self.mean_best_accuracy = float(np.mean(best_accs))
            self.std_best_accuracy = float(np.std(best_accs))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.config.name,
            "algorithm": self.config.algorithm.value,
            "model": self.config.model.value,
            "dataset": self.config.data.dataset.value,
            "num_runs": len(self.runs),
            "successful_runs": len([r for r in self.runs if r.status == "completed"]),
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "min_accuracy": self.min_accuracy,
            "max_accuracy": self.max_accuracy,
            "mean_best_accuracy": self.mean_best_accuracy,
            "std_best_accuracy": self.std_best_accuracy,
            "total_time_seconds": self.total_time_seconds,
        }


class FLExperimentRunner:
    """Runner for FL experiments with multi-model and multi-run support.
    
    Features:
    - Run single experiments or queues of experiments
    - Multiple runs per experiment for statistical significance
    - Per-model result tracking
    - Comparative analysis across experiments
    """
    
    def __init__(
        self,
        device: str = "auto",
        max_parallel_runs: int = 1,
        checkpoint_dir: str = "./checkpoints",
    ):
        """Initialize the experiment runner.
        
        Args:
            device: Device to use (auto, cpu, cuda, mps)
            max_parallel_runs: Maximum parallel runs (1 = sequential)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.device = self._get_device(device)
        self.max_parallel_runs = max_parallel_runs
        self.checkpoint_dir = checkpoint_dir
        
        self.results: Dict[str, ExperimentResult] = {}
        self._running = False
        self._stop_requested = False
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    async def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment (potentially with multiple runs).
        
        Args:
            config: Experiment configuration
        
        Returns:
            ExperimentResult with all run results
        """
        experiment_id = str(uuid.uuid4())[:8]
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            started_at=datetime.now(),
        )
        
        logger.info(f"[Experiment {experiment_id}] Starting: {config.name}")
        logger.info(f"  Algorithm: {config.algorithm.value}, Model: {config.model.value}")
        logger.info(f"  Dataset: {config.data.dataset.value}, Partitions: {config.data.num_partitions}")
        logger.info(f"  Runs: {config.num_runs}, Rounds: {config.server.num_rounds}")
        
        start_time = time.time()
        
        for run_idx in range(config.num_runs):
            if self._stop_requested:
                logger.info(f"[Experiment {experiment_id}] Stop requested, aborting")
                break
            
            # Use different seed for each run
            run_seed = config.seed + run_idx
            
            logger.info(f"[Experiment {experiment_id}] Run {run_idx + 1}/{config.num_runs} (seed={run_seed})")
            
            run_result = await self._run_single(config, run_idx, run_seed)
            result.runs.append(run_result)
            
            logger.info(f"[Experiment {experiment_id}] Run {run_idx + 1} completed: "
                       f"accuracy={run_result.final_accuracy:.4f}, best={run_result.best_accuracy:.4f}")
        
        result.total_time_seconds = time.time() - start_time
        result.completed_at = datetime.now()
        result.compute_statistics()
        
        self.results[experiment_id] = result
        
        logger.info(f"[Experiment {experiment_id}] Completed: "
                   f"mean_acc={result.mean_accuracy:.4f} ± {result.std_accuracy:.4f}")
        
        return result
    
    async def _run_single(
        self,
        config: ExperimentConfig,
        run_idx: int,
        seed: int
    ) -> RunResult:
        """Run a single FL training run."""
        run_result = RunResult(
            run_id=run_idx,
            seed=seed,
            final_accuracy=0.0,
            best_accuracy=0.0,
            best_round=0,
            total_rounds=config.server.num_rounds,
        )
        
        start_time = time.time()
        
        try:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Get dataset info
            dataset_info = get_dataset_info(config.data.dataset)
            num_classes = dataset_info.get("num_classes", 10)
            in_channels = dataset_info.get("input_shape", (3, 32, 32))[0]
            
            # Create global model for initial parameters
            global_model = get_model(
                config.model.value,
                num_classes=num_classes,
                in_channels=in_channels
            )
            
            initial_params = ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in global_model.state_dict().items()]
            )
            
            # Load centralized test set
            testloader = load_centralized_testset(
                config.data.dataset,
                batch_size=config.client.local_batch_size
            )
            
            # Track metrics during training
            round_metrics_list = []
            best_accuracy = 0.0
            best_round = 0
            
            # Server-side evaluation function
            def get_evaluate_fn(model: nn.Module):
                def evaluate(
                    server_round: int,
                    parameters: NDArrays,
                    config_dict: Dict[str, Scalar]
                ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
                    nonlocal best_accuracy, best_round
                    
                    # Set model parameters
                    params_dict = zip(model.state_dict().keys(), parameters)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    model.load_state_dict(state_dict, strict=True)
                    
                    # Evaluate
                    loss, accuracy, _ = evaluate_model(model, testloader, self.device)
                    
                    # Track metrics
                    metrics = RoundMetrics(
                        round_num=server_round,
                        loss=loss,
                        accuracy=accuracy,
                    )
                    round_metrics_list.append(metrics)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_round = server_round
                    
                    return loss, {"accuracy": accuracy}
                
                return evaluate
            
            # Create strategy
            strategy = create_strategy(
                config,
                initial_parameters=initial_params,
                evaluate_fn=get_evaluate_fn(global_model)
            )
            
            # Client function
            def client_fn(context):
                partition_id = context.node_config.get("partition-id", 0)
                if isinstance(partition_id, str):
                    partition_id = int(partition_id)
                
                trainloader, valloader = load_partition(
                    partition_id=partition_id,
                    num_partitions=config.data.num_partitions,
                    dataset=config.data.dataset,
                    partition_strategy=config.data.partition_strategy,
                    batch_size=config.client.local_batch_size,
                    dirichlet_alpha=config.data.dirichlet_alpha,
                    seed=seed,
                )
                
                client_model = get_model(
                    config.model.value,
                    num_classes=num_classes,
                    in_channels=in_channels
                )
                
                proximal_mu = 0.0
                if config.algorithm == FLAlgorithm.FEDPROX:
                    proximal_mu = config.algorithm_params.proximal_mu
                
                return FlowerClient(
                    model=client_model,
                    trainloader=trainloader,
                    valloader=valloader,
                    local_epochs=config.client.local_epochs,
                    learning_rate=config.client.learning_rate,
                    device=self.device,
                    proximal_mu=proximal_mu,
                )
            
            # Server function
            def server_fn(context):
                return ServerAppComponents(
                    strategy=strategy,
                    config=FlwrServerConfig(num_rounds=config.server.num_rounds)
                )
            
            # Run simulation in thread
            def run_fl():
                server_app = ServerApp(server_fn=server_fn)
                client_app = ClientApp(client_fn=client_fn)
                
                run_simulation(
                    server_app=server_app,
                    client_app=client_app,
                    num_supernodes=config.data.num_partitions,
                    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
                )
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, run_fl)
            
            # Update result
            run_result.round_metrics = round_metrics_list
            run_result.best_accuracy = best_accuracy
            run_result.best_round = best_round
            run_result.final_accuracy = round_metrics_list[-1].accuracy if round_metrics_list else 0.0
            run_result.status = "completed"
            
        except Exception as e:
            logger.error(f"Run {run_idx} failed: {e}")
            run_result.status = "failed"
            run_result.error_message = str(e)
        
        run_result.training_time_seconds = time.time() - start_time
        return run_result
    
    async def run_queue(
        self,
        experiments: List[ExperimentConfig],
        runs_per_experiment: int = 1,
    ) -> List[ExperimentResult]:
        """Run a queue of experiments sequentially.
        
        Args:
            experiments: List of experiment configurations
            runs_per_experiment: Override num_runs for all experiments
        
        Returns:
            List of ExperimentResult for each experiment
        """
        results = []
        total = len(experiments)
        
        logger.info(f"Starting experiment queue: {total} experiments")
        
        for idx, config in enumerate(experiments):
            if self._stop_requested:
                logger.info("Stop requested, aborting queue")
                break
            
            # Override num_runs if specified
            if runs_per_experiment > 0:
                config.num_runs = runs_per_experiment
            
            logger.info(f"[Queue {idx + 1}/{total}] Running: {config.name}")
            
            result = await self.run(config)
            results.append(result)
        
        logger.info(f"Queue completed: {len(results)}/{total} experiments")
        return results
    
    def stop(self):
        """Request stop of current experiment/queue."""
        self._stop_requested = True
        logger.info("Stop requested")
    
    def get_results(self) -> Dict[str, ExperimentResult]:
        """Get all experiment results."""
        return self.results
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get result for a specific experiment."""
        return self.results.get(experiment_id)
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
