"""FL Session Manager for Managing Federated Learning Sessions.

This module provides the FLSessionManager class that integrates all FL components
and manages the lifecycle of FL training sessions.
"""

import asyncio
import logging
import uuid
import threading
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
from .core.client import FlowerClient, evaluate_model
from .algorithms import create_strategy
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
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    loss: float
    accuracy: float
    participating_clients: int = 0
    train_loss: float = 0.0
    aggregation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.config.name,
            "algorithm": self.config.algorithm.value,
            "model": self.config.model.value,
            "dataset": self.config.data.dataset.value,
            "status": self.status.value,
            "progress": f"{self.current_round}/{self.total_rounds}",
            "best_accuracy": self.best_accuracy,
            "best_round": self.best_round,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error_message,
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
                import time
                metrics = RoundMetrics(
                    round_num=server_round,
                    loss=loss,
                    accuracy=accuracy,
                    participating_clients=config.data.num_partitions,
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
                seed=config.seed,
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
            try:
                logger.info(f"[Session {session.session_id[:8]}] Starting Flower simulation")
                
                server_app = ServerApp(server_fn=server_fn)
                client_app = ClientApp(client_fn=client_fn)
                
                run_simulation(
                    server_app=server_app,
                    client_app=client_app,
                    num_supernodes=config.data.num_partitions,
                    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
                )
                
                logger.info(f"[Session {session.session_id[:8]}] Simulation completed")
                
            except Exception as e:
                logger.error(f"[Session {session.session_id[:8]}] Simulation failed: {e}")
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
