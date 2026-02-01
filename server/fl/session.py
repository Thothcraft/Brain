"""FL Session Manager for Managing Federated Learning Sessions.

This module provides the FLSessionManager class that integrates all FL components
and manages the lifecycle of FL training sessions.

Flower Documentation References:
- Simulations: https://flower.ai/docs/framework/how-to-run-simulations.html
- Docker deployment: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
- Ray backend: https://docs.ray.io/en/latest/ray-core/configure.html
"""

import asyncio
import logging
import os
import uuid
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

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

from .core.config import ExperimentConfig, FLAlgorithm, FLDataset
from .core.models import get_model
from .core.client import evaluate_model, train_model, create_client_app, FlowerClient
from .core.server_app import create_server_app, create_simple_fl_app
from .algorithms import create_strategy, FedAvgStrategy
from .datasets import load_partition, load_centralized_testset, get_dataset_info

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Status of an FL session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClientRoundMetrics:
    """Metrics for a single client in a single round."""
    client_id: str
    round_num: int
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    num_samples: int = 0
    training_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    model_size_bytes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RoundMetrics:
    """Metrics for a single FL round with detailed per-client tracking."""
    round_num: int
    loss: float
    accuracy: float
    participating_clients: int = 0
    train_loss: float = 0.0
    aggregation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    avg_loss: float = 0.0
    avg_accuracy: float = 0.0
    min_accuracy: float = 0.0
    max_accuracy: float = 0.0
    std_accuracy: float = 0.0
    communication_cost: float = 0.0
    convergence_rate: float = 0.0
    fairness_index: float = 1.0
    round_start_time: Optional[datetime] = None
    round_end_time: Optional[datetime] = None
    round_duration_ms: float = 0.0
    client_metrics: Dict[str, ClientRoundMetrics] = field(default_factory=dict)
    selected_clients: List[str] = field(default_factory=list)
    failed_clients: List[str] = field(default_factory=list)


@dataclass
class FLClient:
    """State of a federated learning client with comprehensive tracking."""
    client_id: str
    device_id: str
    data_samples: int = 0
    compute_capability: float = 1.0
    is_active: bool = True
    is_remote: bool = False  # True if this is a remote Thoth device
    remote_address: Optional[str] = None  # IP:port for remote clients
    rounds_participated: List[int] = field(default_factory=list)
    rounds_failed: List[int] = field(default_factory=list)
    contribution_score: float = 0.0
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    per_round_metrics: Dict[int, ClientRoundMetrics] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    total_training_time_ms: float = 0.0
    total_communication_time_ms: float = 0.0
    avg_accuracy: float = 0.0
    best_accuracy: float = 0.0
    connection_status: str = "connected"  # connected, disconnected, timeout


@dataclass
class FLSession:
    """State of a federated learning session."""
    session_id: str
    config: ExperimentConfig
    status: SessionStatus = SessionStatus.PENDING
    current_round: int = 0
    total_rounds: int = 0
    round_metrics: Dict[int, RoundMetrics] = field(default_factory=dict)
    best_accuracy: float = 0.0
    best_round: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    clients: Dict[str, FLClient] = field(default_factory=dict)
    privacy_budget_spent: float = 0.0
    global_model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "session_name": self.config.name,
            "algorithm": self.config.algorithm.value,
            "model": self.config.model.value,
            "dataset": self.config.data.dataset.value,
            "status": self.status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "best_accuracy": self.best_accuracy,
            "best_round": self.best_round,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


class FLSessionManager:
    """Manages federated learning sessions using the Flower framework.
    
    This class provides:
    - Session creation and lifecycle management
    - Running FL simulations with Flower
    - Progress tracking and metrics collection
    - Session listing and status queries
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the session manager.
        
        Args:
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.sessions: Dict[str, FLSession] = {}
        self._running_threads: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, bool] = {}
        self.device = self._get_device(device)
    
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
    
    def create_session(self, config: ExperimentConfig) -> FLSession:
        """Create a new FL session.
        
        Args:
            config: Experiment configuration
        
        Returns:
            Created FLSession instance
        """
        session_id = str(uuid.uuid4())
        
        session = FLSession(
            session_id=session_id,
            config=config,
            total_rounds=config.server.num_rounds,
        )
        
        self.sessions[session_id] = session
        self._stop_flags[session_id] = False
        
        logger.info(f"Created FL session {session_id}: {config.name}")
        logger.info(f"  Algorithm: {config.algorithm.value}, Model: {config.model.value}")
        logger.info(f"  Dataset: {config.data.dataset.value}, Partitions: {config.data.num_partitions}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[FLSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summary info."""
        return [session.to_dict() for session in self.sessions.values()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session (must be stopped first)."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.status == SessionStatus.RUNNING:
            logger.warning(f"Cannot delete running session {session_id}")
            return False
        
        del self.sessions[session_id]
        self._stop_flags.pop(session_id, None)
        self._running_threads.pop(session_id, None)
        
        logger.info(f"Deleted session {session_id}")
        return True
    
    def add_client(
        self,
        session_id: str,
        device_id: str,
        data_samples: int,
        compute_capability: float = 1.0
    ) -> Optional[FLClient]:
        """Add a client to an FL session.
        
        Args:
            session_id: ID of the session to join
            device_id: Unique device identifier
            data_samples: Number of data samples the client has
            compute_capability: Relative compute capability (1.0 = baseline)
        
        Returns:
            FLClient instance if successful, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        client_id = str(uuid.uuid4())
        client = FLClient(
            client_id=client_id,
            device_id=device_id,
            data_samples=data_samples,
            compute_capability=compute_capability,
        )
        
        session.clients[client_id] = client
        logger.info(f"Client {client_id} ({device_id}) joined session {session_id}")
        
        return client
    
    async def run_session(self, session_id: str) -> FLSession:
        """Run a complete FL session using Flower framework.
        
        Args:
            session_id: ID of the session to run
        
        Returns:
            Updated FLSession with results
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status == SessionStatus.RUNNING:
            raise ValueError(f"Session {session_id} is already running")
        
        session.status = SessionStatus.RUNNING
        session.started_at = datetime.now()
        self._stop_flags[session_id] = False
        
        config = session.config
        logger.info(f"Starting FL session {session_id}")
        
        try:
            await self._run_flower_simulation(session)
            
            if session.status == SessionStatus.RUNNING:
                session.status = SessionStatus.COMPLETED
                session.completed_at = datetime.now()
                logger.info(f"Session {session_id} completed: best_accuracy={session.best_accuracy:.4f}")
        
        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.now()
            logger.error(f"Session {session_id} failed: {e}")
            raise
        
        return session
    
    async def _run_flower_simulation(self, session: FLSession) -> None:
        """Run FL using Flower's simulation capabilities."""
        config = session.config
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Get dataset info
        dataset_info = get_dataset_info(config.data.dataset)
        num_classes = dataset_info.get("num_classes", 10)
        in_channels = dataset_info.get("input_shape", (3, 32, 32))[0]
        
        # Create global model
        global_model = get_model(
            config.model.value,
            num_classes=num_classes,
            in_channels=in_channels
        )
        
        # Get initial parameters
        initial_params = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        )
        
        # Load centralized test set
        testloader = load_centralized_testset(
            config.data.dataset,
            batch_size=config.client.local_batch_size
        )
        
        # Server-side evaluation function
        def get_evaluate_fn(model: nn.Module):
            def evaluate(
                server_round: int,
                parameters: NDArrays,
                config_dict: Dict[str, Scalar]
            ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
                # Set model parameters
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                
                # Evaluate
                loss, accuracy, _ = evaluate_model(model, testloader, self.device)
                
                # Update session metrics
                metrics = RoundMetrics(
                    round_num=server_round,
                    loss=loss,
                    accuracy=accuracy,
                    participating_clients=config.data.num_partitions,
                    avg_loss=loss,
                    avg_accuracy=accuracy,
                    min_accuracy=accuracy,
                    max_accuracy=accuracy,
                )
                session.round_metrics[server_round] = metrics
                session.current_round = server_round
                
                if accuracy > session.best_accuracy:
                    session.best_accuracy = accuracy
                    session.best_round = server_round
                    logger.info(f"[Session {session.session_id[:8]}] Round {server_round}: "
                               f"accuracy={accuracy:.4f} ⭐ NEW BEST")
                else:
                    logger.info(f"[Session {session.session_id[:8]}] Round {server_round}: "
                               f"accuracy={accuracy:.4f}")
                
                return loss, {"accuracy": accuracy}
            
            return evaluate
        
        # Create strategy
        strategy = create_strategy(
            algorithm=config.algorithm.value,  # Pass algorithm string, not config
            initial_parameters=initial_params,
            evaluate_fn=get_evaluate_fn(global_model),
            fraction_fit=config.server.fraction_fit,
            fraction_evaluate=config.server.fraction_evaluate,
            min_fit_clients=config.server.min_fit_clients,
            min_evaluate_clients=config.server.min_evaluate_clients,
            min_available_clients=config.server.min_available_clients,
            proximal_mu=config.algorithm_params.proximal_mu,
            server_momentum=config.algorithm_params.server_momentum,
        )
        
        # Extract ALL values as primitive types to avoid Ray serialization issues
        # Ray's cloudpickle cannot serialize thread locks, enums may also cause issues
        _device_type = str(self.device.type)
        _local_epochs = int(config.client.local_epochs)
        _learning_rate = float(config.client.learning_rate)
        _num_partitions = int(config.data.num_partitions)
        _dataset_value = config.data.dataset.value  # Convert enum to string
        _partition_strategy_value = config.data.partition_strategy.value  # Convert enum to string
        _batch_size = int(config.client.local_batch_size)
        _dirichlet_alpha = float(config.data.dirichlet_alpha)
        _seed = int(config.seed)
        _is_fedprox = config.algorithm == FLAlgorithm.FEDPROX
        _proximal_mu = float(config.algorithm_params.proximal_mu) if _is_fedprox else 0.0
        _model_value = str(config.model.value)
        _num_classes = int(num_classes)
        _in_channels = int(in_channels)
        
        # Client function - only captures serializable primitives (strings, ints, floats, bools)
        def client_fn(context):
            partition_id = context.node_config.get("partition-id", 0)
            if isinstance(partition_id, str):
                partition_id = int(partition_id)
            
            # Import enums inside the function to reconstruct from string values
            from .core.config import FLDataset, PartitionStrategy
            
            trainloader, valloader = load_partition(
                partition_id=partition_id,
                num_partitions=_num_partitions,
                dataset=FLDataset(_dataset_value),
                partition_strategy=PartitionStrategy(_partition_strategy_value),
                batch_size=_batch_size,
                dirichlet_alpha=_dirichlet_alpha,
                seed=_seed,
            )
            
            client_model = get_model(
                _model_value,
                num_classes=_num_classes,
                in_channels=_in_channels
            )
            
            # Recreate device inside the function to avoid serialization issues
            client_device = torch.device(_device_type)
            
            return FlowerClient(
                model=client_model,
                trainloader=trainloader,
                valloader=valloader,
                local_epochs=_local_epochs,
                learning_rate=_learning_rate,
                device=client_device,
                proximal_mu=_proximal_mu,
            )
        
        # Extract server config as primitives
        _num_rounds = int(config.server.num_rounds)
        _session_id_short = session.session_id[:8]
        
        # Server function - strategy is created fresh, not serialized
        def server_fn(context):
            return ServerAppComponents(
                strategy=strategy,
                config=FlwrServerConfig(num_rounds=_num_rounds)
            )
        
        # Configure Ray environment for Docker/container environments
        # Reference: https://docs.ray.io/en/latest/ray-core/configure.html
        def configure_ray_env():
            """Set Ray environment variables to suppress warnings in containers."""
            os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
            os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
            os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
            # Limit object store memory for containers with small /dev/shm
            os.environ.setdefault("RAY_OBJECT_STORE_MEMORY", "100000000")
            # Disable metrics exporter to avoid connection errors
            os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "-1")
        
        # Check if we're on Windows - Ray doesn't work well on Windows
        import platform
        is_windows = platform.system() == "Windows"
        
        # Run simulation in thread
        def run_fl():
            try:
                if is_windows:
                    # Windows: Use sequential simulation without Ray
                    logger.info(f"[Session {_session_id_short}] Starting sequential FL simulation (Windows mode)")
                    logger.info(f"[Session {_session_id_short}] Config: {_num_partitions} clients, {_num_rounds} rounds")
                    
                    # Run sequential simulation
                    self._run_sequential_simulation(
                        session=session,
                        client_fn=client_fn,
                        strategy=strategy,
                        num_clients=_num_partitions,
                        num_rounds=_num_rounds,
                        global_model=global_model,
                        testloader=testloader,
                    )
                else:
                    # Linux/Mac: Use Flower's Ray-based simulation
                    configure_ray_env()
                    logger.info(f"[Session {_session_id_short}] Starting Flower simulation")
                    logger.info(f"[Session {_session_id_short}] Config: {_num_partitions} clients, {_num_rounds} rounds")
                    
                    server_app = ServerApp(server_fn=server_fn)
                    client_app = ClientApp(client_fn=client_fn)
                    
                    # Backend config with Ray initialization args to suppress warnings
                    # Reference: https://flower.ai/docs/framework/how-to-run-simulations.html
                    backend_config = {
                        "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                        "init_args": {
                            "include_dashboard": False,
                            "_metrics_export_port": -1,  # Disable metrics export
                            "configure_logging": False,
                            "logging_level": logging.WARNING,
                        }
                    }
                    
                    run_simulation(
                        server_app=server_app,
                        client_app=client_app,
                        num_supernodes=_num_partitions,
                        backend_config=backend_config,
                    )
                
                logger.info(f"[Session {_session_id_short}] Simulation completed successfully")
                
            except Exception as e:
                logger.error(f"[Session {_session_id_short}] Simulation failed: {e}")
                session.error_message = str(e)
                session.status = SessionStatus.FAILED
        
        # Run in thread
        thread = threading.Thread(target=run_fl, daemon=True)
        self._running_threads[session.session_id] = thread
        thread.start()
        
        # Wait for completion with periodic checks
        while thread.is_alive():
            if self._stop_flags.get(session.session_id, False):
                session.status = SessionStatus.CANCELLED
                break
            await asyncio.sleep(1.0)
        
        thread.join(timeout=10.0)
    
    def stop_session(self, session_id: str) -> bool:
        """Stop a running session.
        
        Args:
            session_id: ID of the session to stop
        
        Returns:
            True if stop was requested, False if session not found/not running
        """
        session = self.get_session(session_id)
        if not session or session.status != SessionStatus.RUNNING:
            return False
        
        self._stop_flags[session_id] = True
        session.status = SessionStatus.CANCELLED
        session.completed_at = datetime.now()
        
        logger.info(f"Stop requested for session {session_id}")
        return True
    
    def _run_sequential_simulation(
        self,
        session: FLSession,
        client_fn,
        strategy,
        num_clients: int,
        num_rounds: int,
        global_model: nn.Module,
        testloader,
    ) -> None:
        """Run FL simulation sequentially without Ray (Windows fallback).
        
        This implements FedAvg-style training by:
        1. For each round, train all clients sequentially
        2. Aggregate updates using the strategy
        3. Evaluate on test set
        """
        from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Status, Code
        
        session_id_short = session.session_id[:8]
        
        # Get initial parameters from global model
        global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        parameters = ndarrays_to_parameters(global_params)
        
        # Create a mock context class for client_fn
        class MockNodeConfig:
            def __init__(self, partition_id):
                self._partition_id = partition_id
            def get(self, key, default=None):
                if key == "partition-id":
                    return self._partition_id
                return default
        
        class MockContext:
            def __init__(self, partition_id):
                self.node_config = MockNodeConfig(partition_id)
        
        for round_num in range(1, num_rounds + 1):
            if self._stop_flags.get(session.session_id, False):
                logger.info(f"[Session {session_id_short}] Stopping at round {round_num}")
                break
            
            logger.info(f"[Session {session_id_short}] Round {round_num}/{num_rounds}")
            round_start = time.time()
            
            # Train each client sequentially
            fit_results = []
            for client_id in range(num_clients):
                try:
                    # Create client
                    context = MockContext(client_id)
                    client = client_fn(context)
                    
                    # Convert Parameters to NDArrays for client.fit()
                    params_ndarrays = parameters_to_ndarrays(parameters)
                    
                    # Train client - fit() expects NDArrays, not Parameters object
                    fit_res = client.fit(params_ndarrays, {"round": round_num, "server_round": round_num})
                    
                    # fit_res is (parameters, num_examples, metrics)
                    client_params, num_examples, metrics = fit_res
                    
                    fit_results.append((
                        ndarrays_to_parameters(client_params),
                        num_examples,
                        metrics
                    ))
                    
                except Exception as e:
                    logger.warning(f"[Session {session_id_short}] Client {client_id} failed: {e}")
            
            if not fit_results:
                logger.error(f"[Session {session_id_short}] No clients completed training in round {round_num}")
                continue
            
            # Aggregate using simple FedAvg (weighted average by num_examples)
            total_examples = sum(num_ex for _, num_ex, _ in fit_results)
            
            # Convert parameters to numpy arrays and aggregate
            aggregated_params = None
            for client_params, num_examples, _ in fit_results:
                weight = num_examples / total_examples
                client_ndarrays = parameters_to_ndarrays(client_params)
                
                if aggregated_params is None:
                    aggregated_params = [arr * weight for arr in client_ndarrays]
                else:
                    for i, arr in enumerate(client_ndarrays):
                        aggregated_params[i] += arr * weight
            
            # Update global parameters
            parameters = ndarrays_to_parameters(aggregated_params)
            
            # Update global model with aggregated parameters
            params_dict = zip(global_model.state_dict().keys(), aggregated_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            global_model.load_state_dict(state_dict, strict=True)
            
            # Evaluate on test set
            loss, accuracy, _ = evaluate_model(global_model, testloader, self.device)
            
            round_duration = (time.time() - round_start) * 1000  # ms
            
            # Update session metrics
            metrics = RoundMetrics(
                round_num=round_num,
                loss=loss,
                accuracy=accuracy,
                participating_clients=len(fit_results),
                avg_loss=loss,
                avg_accuracy=accuracy,
                min_accuracy=accuracy,
                max_accuracy=accuracy,
                round_duration_ms=round_duration,
            )
            session.round_metrics[round_num] = metrics
            session.current_round = round_num
            
            if accuracy > session.best_accuracy:
                session.best_accuracy = accuracy
                session.best_round = round_num
                logger.info(f"[Session {session_id_short}] Round {round_num}: "
                           f"accuracy={accuracy:.4f} ⭐ NEW BEST")
            else:
                logger.info(f"[Session {session_id_short}] Round {round_num}: "
                           f"accuracy={accuracy:.4f}")
        
        logger.info(f"[Session {session_id_short}] Sequential simulation completed")
    
    def _run_sequential_with_train_fn(
        self,
        session: FLSession,
        model_fn,
        load_data_fn,
        testloader,
    ) -> None:
        """Run sequential FL using the simplified train_fn.
        
        This is a Windows-compatible version that uses the simplified
        training function from client_app.py.
        """
        config = session.config
        session_id_short = session.session_id[:8]
        num_clients = config.data.num_partitions
        num_rounds = config.server.num_rounds
        lr = config.client.learning_rate
        local_epochs = config.client.local_epochs
        batch_size = config.client.local_batch_size
        
        # Initialize global model
        global_model = model_fn()
        
        for round_num in range(1, num_rounds + 1):
            if self._stop_flags.get(session.session_id, False):
                logger.info(f"[Session {session_id_short}] Stopping at round {round_num}")
                break
            
            logger.info(f"[Session {session_id_short}] Round {round_num}/{num_rounds}")
            round_start = time.time()
            
            # Collect client updates
            client_updates = []
            total_samples = 0
            
            for client_id in range(num_clients):
                try:
                    # Load client data
                    trainloader, _ = load_data_fn(client_id, num_clients, batch_size)
                    
                    # Create client model and load global weights
                    client_model = model_fn()
                    client_model.load_state_dict(global_model.state_dict())
                    
                    # Train using simplified train_model function
                    train_loss, _ = train_model(
                        model=client_model,
                        trainloader=trainloader,
                        epochs=local_epochs,
                        lr=lr,
                        device=self.device,
                    )
                    
                    num_samples = len(trainloader.dataset)
                    client_updates.append((client_model.state_dict(), num_samples))
                    total_samples += num_samples
                    
                except Exception as e:
                    logger.warning(f"[Session {session_id_short}] Client {client_id} failed: {e}")
            
            if not client_updates:
                logger.error(f"[Session {session_id_short}] No clients completed round {round_num}")
                continue
            
            # FedAvg aggregation
            aggregated_state = {}
            for key in global_model.state_dict().keys():
                aggregated_state[key] = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for client_state, num_samples in client_updates:
                    weight = num_samples / total_samples
                    aggregated_state[key] += client_state[key].float() * weight
                aggregated_state[key] = aggregated_state[key].to(global_model.state_dict()[key].dtype)
            
            global_model.load_state_dict(aggregated_state)
            
            # Evaluate
            loss, accuracy, _ = evaluate_model(global_model, testloader, self.device)
            
            round_duration = (time.time() - round_start) * 1000
            
            # Update session metrics
            metrics = RoundMetrics(
                round_num=round_num,
                loss=loss,
                accuracy=accuracy,
                participating_clients=len(client_updates),
                avg_loss=loss,
                avg_accuracy=accuracy,
                round_duration_ms=round_duration,
            )
            session.round_metrics[round_num] = metrics
            session.current_round = round_num
            
            if accuracy > session.best_accuracy:
                session.best_accuracy = accuracy
                session.best_round = round_num
                logger.info(f"[Session {session_id_short}] Round {round_num}: "
                           f"accuracy={accuracy:.4f} ⭐ NEW BEST")
            else:
                logger.info(f"[Session {session_id_short}] Round {round_num}: "
                           f"accuracy={accuracy:.4f}")
        
        logger.info(f"[Session {session_id_short}] Simplified simulation completed")
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed metrics for a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            Dictionary with session metrics
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        # Extract accuracy curve
        rounds = sorted(session.round_metrics.keys())
        accuracies = [session.round_metrics[r].accuracy for r in rounds]
        losses = [session.round_metrics[r].loss for r in rounds]
        
        return {
            "session_id": session_id,
            "name": session.config.name,
            "status": session.status.value,
            "current_round": session.current_round,
            "total_rounds": session.total_rounds,
            "best_accuracy": session.best_accuracy,
            "best_round": session.best_round,
            "accuracy_curve": accuracies,
            "loss_curve": losses,
            "rounds": rounds,
        }


# Global instance
fl_manager = FLSessionManager()
