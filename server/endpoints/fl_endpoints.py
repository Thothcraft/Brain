"""Federated Learning Endpoints using Flower Framework.

Comprehensive FL API with:
- Multiple FL strategies from Flower (FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, etc.)
- Byzantine-robust aggregation (Krum, Bulyan, FedMedian, FedTrimmedAvg)
- Differential privacy support (DPFedAvgAdaptive, DPFedAvgFixed)
- Fair FL (QFedAvg)
- Built-in datasets via Flower Datasets (CIFAR-10, MNIST, Fashion-MNIST, etc.)
- Dynamic configuration and parameter control
- Real-time monitoring and metrics

Requires: flwr[simulation], flwr-datasets, torch, torchvision
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import asyncio

from .models import StandardResponse
from ..flower_fl import (
    fl_manager,
    FLAlgorithm,
    FLDataset,
    PartitionStrategy,
    AggregationMethod,
    ClientSelectionStrategy,
    ModelArchitecture,
    FLSessionConfig,
    ServerConfig,
    ClientConfig,
    AlgorithmConfig,
    DataConfig,
    PrivacyConfig,
    MonitoringConfig,
    get_dataset_info,
    get_algorithm_info,
)

router = APIRouter(prefix="/fl", tags=["federated-learning"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PrivacyConfigRequest(BaseModel):
    """Privacy configuration for FL session."""
    differential_privacy: bool = False
    noise_multiplier: float = Field(1.0, ge=0.0, le=10.0)
    max_grad_norm: float = Field(1.0, ge=0.1, le=10.0)
    delta: float = Field(1e-5, ge=1e-10, le=1e-3)
    secure_aggregation: bool = False
    encryption_bits: int = Field(256, ge=128, le=512)
    min_clients_for_aggregation: int = Field(3, ge=2, le=100)


class ClientConfigRequest(BaseModel):
    """Client-side training configuration."""
    local_epochs: int = Field(5, ge=1, le=100)
    local_batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(0.01, ge=1e-6, le=10.0)
    momentum: float = Field(0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(1e-4, ge=0.0, le=1.0)
    optimizer: str = Field("sgd", pattern="^(sgd|adam|adamw|rmsprop)$")
    lr_scheduler: Optional[str] = None
    gradient_clipping: Optional[float] = Field(None, ge=0.1, le=100.0)
    data_augmentation: bool = True


class ServerConfigRequest(BaseModel):
    """Server-side FL configuration."""
    num_rounds: int = Field(100, ge=1, le=10000)
    min_fit_clients: int = Field(2, ge=1, le=1000)
    min_evaluate_clients: int = Field(2, ge=1, le=1000)
    min_available_clients: int = Field(2, ge=1, le=1000)
    fraction_fit: float = Field(1.0, ge=0.0, le=1.0)
    fraction_evaluate: float = Field(0.5, ge=0.0, le=1.0)
    accept_failures: bool = True


class AlgorithmConfigRequest(BaseModel):
    """Algorithm-specific hyperparameters."""
    proximal_mu: float = Field(0.01, ge=0.0, le=10.0)
    server_learning_rate: float = Field(1.0, ge=1e-6, le=100.0)
    beta_1: float = Field(0.9, ge=0.0, le=1.0)
    beta_2: float = Field(0.99, ge=0.0, le=1.0)
    tau: float = Field(1e-3, ge=1e-10, le=1.0)
    server_lr: float = Field(1.0, ge=1e-6, le=100.0)
    q_param: float = Field(0.2, ge=0.0, le=10.0)
    byzantine_fraction: float = Field(0.0, ge=0.0, le=0.5)
    trimmed_mean_beta: float = Field(0.1, ge=0.0, le=0.5)
    krum_num_closest: int = Field(2, ge=1, le=100)


class DataConfigRequest(BaseModel):
    """Dataset and partitioning configuration."""
    dataset: str = "cifar10"
    num_partitions: int = Field(10, ge=2, le=1000)
    partition_strategy: str = "iid"
    dirichlet_alpha: float = Field(0.5, ge=0.01, le=100.0)
    min_samples_per_client: int = Field(100, ge=10, le=10000)
    validation_split: float = Field(0.2, ge=0.0, le=0.5)
    test_split: float = Field(0.1, ge=0.0, le=0.5)
    custom_data_path: Optional[str] = None


class MonitoringConfigRequest(BaseModel):
    """Training monitoring configuration."""
    log_interval: int = Field(1, ge=1, le=100)
    checkpoint_interval: int = Field(10, ge=1, le=1000)
    early_stopping_patience: int = Field(20, ge=1, le=1000)
    early_stopping_metric: str = "accuracy"
    early_stopping_mode: str = Field("max", pattern="^(min|max)$")
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: Optional[str] = None


class CreateFLSessionRequest(BaseModel):
    """Request to create a new FL session."""
    session_name: str = Field(..., min_length=1, max_length=100)
    algorithm: str = "fedavg"
    model_architecture: str = "cnn"
    aggregation_method: str = "weighted_average"
    client_selection: str = "random"
    
    server: Optional[ServerConfigRequest] = None
    client: Optional[ClientConfigRequest] = None
    algorithm_params: Optional[AlgorithmConfigRequest] = None
    data: Optional[DataConfigRequest] = None
    privacy: Optional[PrivacyConfigRequest] = None
    monitoring: Optional[MonitoringConfigRequest] = None
    
    seed: int = Field(42, ge=0)
    device: str = Field("auto", pattern="^(auto|cpu|cuda|mps)$")


class JoinSessionRequest(BaseModel):
    """Request to join an FL session as a client."""
    device_id: str = Field(..., min_length=1, max_length=100)
    data_samples: int = Field(..., ge=1)
    compute_capability: float = Field(1.0, ge=0.1, le=10.0)


class UpdateConfigRequest(BaseModel):
    """Request to update session configuration dynamically."""
    client: Optional[ClientConfigRequest] = None
    algorithm_params: Optional[AlgorithmConfigRequest] = None
    monitoring: Optional[MonitoringConfigRequest] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_to_fl_config(request: CreateFLSessionRequest) -> FLSessionConfig:
    """Convert API request to internal FLSessionConfig."""
    # Map string enums to actual enums
    algorithm = FLAlgorithm(request.algorithm.lower())
    model_arch = ModelArchitecture(request.model_architecture.lower())
    agg_method = AggregationMethod(request.aggregation_method.lower())
    client_sel = ClientSelectionStrategy(request.client_selection.lower())
    
    # Build sub-configs
    server_config = ServerConfig(
        **request.server.model_dump() if request.server else {}
    )
    client_config = ClientConfig(
        **request.client.model_dump() if request.client else {}
    )
    algo_config = AlgorithmConfig(
        **request.algorithm_params.model_dump() if request.algorithm_params else {}
    )
    
    # Data config with enum conversion
    data_dict = request.data.model_dump() if request.data else {}
    if "dataset" in data_dict:
        data_dict["dataset"] = FLDataset(data_dict["dataset"].lower())
    if "partition_strategy" in data_dict:
        data_dict["partition_strategy"] = PartitionStrategy(data_dict["partition_strategy"].lower())
    data_config = DataConfig(**data_dict)
    
    privacy_config = PrivacyConfig(
        **request.privacy.model_dump() if request.privacy else {}
    )
    monitoring_config = MonitoringConfig(
        **request.monitoring.model_dump() if request.monitoring else {}
    )
    
    return FLSessionConfig(
        session_name=request.session_name,
        algorithm=algorithm,
        model_architecture=model_arch,
        aggregation_method=agg_method,
        client_selection=client_sel,
        server=server_config,
        client=client_config,
        algorithm_params=algo_config,
        data=data_config,
        privacy=privacy_config,
        monitoring=monitoring_config,
        seed=request.seed,
        device=request.device
    )


# ============================================================================
# ENDPOINTS - SESSION MANAGEMENT
# ============================================================================

@router.post("/sessions", response_model=StandardResponse)
async def create_fl_session(
    request: CreateFLSessionRequest,
    background_tasks: BackgroundTasks
):
    """Create a new Federated Learning session using Flower framework.
    
    Supports multiple FL strategies from Flower:
    - **fedavg**: Standard Federated Averaging
    - **fedprox**: FedAvg with proximal term for non-IID data
    - **fedadam**: Adaptive FL with Adam optimizer
    - **fedyogi**: Adaptive FL with controlled adaptivity
    - **fedadagrad**: Adaptive FL with Adagrad
    - **fedavgm**: FedAvg with server momentum
    - **fedopt**: Generalized federated optimization
    - **fedmedian**: Byzantine-robust median aggregation
    - **fedtrimmedavg**: Byzantine-robust trimmed mean
    - **krum**: Byzantine-robust Krum aggregation
    - **bulyan**: Byzantine-robust Bulyan aggregation
    - **qfedavg**: Fair federated learning
    - **dpfedavg_adaptive**: Differential privacy with adaptive clipping
    - **dpfedavg_fixed**: Differential privacy with fixed clipping
    """
    try:
        config = convert_to_fl_config(request)
        session = fl_manager.create_session(config)
        
        return StandardResponse(
            success=True,
            message=f"FL session '{request.session_name}' created successfully",
            data={
                "session_id": session.session_id,
                "session_name": config.session_name,
                "algorithm": config.algorithm.value,
                "model_architecture": config.model_architecture.value,
                "dataset": config.data.dataset.value,
                "num_rounds": config.server.num_rounds,
                "status": session.status,
                "created_at": session.created_at.isoformat()
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create FL session: {str(e)}")


@router.get("/sessions", response_model=Dict[str, Any])
async def list_fl_sessions(
    status: Optional[str] = Query(None, description="Filter by status"),
    algorithm: Optional[str] = Query(None, description="Filter by algorithm")
):
    """List all FL sessions with optional filtering."""
    try:
        sessions = fl_manager.list_sessions()
        
        # Apply filters
        if status:
            sessions = [s for s in sessions if s["status"] == status]
        if algorithm:
            sessions = [s for s in sessions if s["algorithm"] == algorithm]
        
        return {
            "success": True,
            "total_sessions": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_fl_session(session_id: str):
    """Get detailed information about an FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Build detailed response
        client_summaries = []
        for client in session.clients.values():
            client_summaries.append({
                "client_id": client.client_id,
                "device_id": client.device_id,
                "data_samples": client.data_samples,
                "rounds_participated": len(client.rounds_participated),
                "is_active": client.is_active,
                "last_update": client.last_update.isoformat()
            })
        
        round_metrics = []
        for round_num, metrics in sorted(session.round_metrics.items()):
            round_metrics.append({
                "round": metrics.round_num,
                "participating_clients": metrics.participating_clients,
                "avg_loss": metrics.avg_loss,
                "avg_accuracy": metrics.avg_accuracy,
                "min_accuracy": metrics.min_accuracy,
                "max_accuracy": metrics.max_accuracy,
                "fairness_index": metrics.fairness_index,
                "timestamp": metrics.timestamp.isoformat()
            })
        
        return {
            "success": True,
            "session": {
                "session_id": session.session_id,
                "session_name": session.config.session_name,
                "algorithm": session.config.algorithm.value,
                "model_architecture": session.config.model_architecture.value,
                "dataset": session.config.data.dataset.value,
                "status": session.status,
                "current_round": session.current_round,
                "total_rounds": session.total_rounds,
                "best_accuracy": session.best_accuracy,
                "best_round": session.best_round,
                "privacy_budget_spent": session.privacy_budget_spent,
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "error_message": session.error_message,
                "global_model_path": session.global_model_path
            },
            "config": {
                "server": {
                    "num_rounds": session.config.server.num_rounds,
                    "min_fit_clients": session.config.server.min_fit_clients,
                    "fraction_fit": session.config.server.fraction_fit,
                    "fraction_evaluate": session.config.server.fraction_evaluate
                },
                "client": {
                    "local_epochs": session.config.client.local_epochs,
                    "local_batch_size": session.config.client.local_batch_size,
                    "learning_rate": session.config.client.learning_rate,
                    "optimizer": session.config.client.optimizer
                },
                "privacy": {
                    "differential_privacy": session.config.privacy.differential_privacy,
                    "secure_aggregation": session.config.privacy.secure_aggregation
                }
            },
            "clients": client_summaries,
            "round_metrics": round_metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.post("/sessions/{session_id}/start", response_model=StandardResponse)
async def start_fl_session(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Start an FL session (begin training rounds).
    
    For simulation mode, clients are simulated automatically based on num_partitions.
    No need to register clients manually.
    """
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if session.status != "pending":
            raise HTTPException(status_code=400, detail=f"Session is already {session.status}")
        
        # For Flower simulation mode, we don't need real clients
        # The simulation will create virtual clients based on num_partitions
        # Skip the client check for simulation
        
        # Start session in background
        background_tasks.add_task(fl_manager.run_session, session_id)
        
        return StandardResponse(
            success=True,
            message=f"FL session {session_id} started",
            data={
                "session_id": session_id,
                "status": "running",
                "num_clients": session.config.data.num_partitions,
                "total_rounds": session.total_rounds
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@router.post("/sessions/{session_id}/stop", response_model=StandardResponse)
async def stop_fl_session(session_id: str):
    """Stop a running FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if session.status != "running":
            raise HTTPException(status_code=400, detail=f"Session is not running (status: {session.status})")
        
        session.status = "cancelled"
        session.completed_at = datetime.now()
        
        return StandardResponse(
            success=True,
            message=f"FL session {session_id} stopped",
            data={
                "session_id": session_id,
                "status": session.status,
                "completed_rounds": session.current_round,
                "best_accuracy": session.best_accuracy
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")


@router.patch("/sessions/{session_id}/config", response_model=StandardResponse)
async def update_session_config(
    session_id: str,
    request: UpdateConfigRequest
):
    """Dynamically update session configuration during training."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        updates = []
        
        if request.client:
            for key, value in request.client.model_dump(exclude_unset=True).items():
                setattr(session.config.client, key, value)
                updates.append(f"client.{key}")
        
        if request.algorithm_params:
            for key, value in request.algorithm_params.model_dump(exclude_unset=True).items():
                setattr(session.config.algorithm_params, key, value)
                updates.append(f"algorithm_params.{key}")
        
        if request.monitoring:
            for key, value in request.monitoring.model_dump(exclude_unset=True).items():
                setattr(session.config.monitoring, key, value)
                updates.append(f"monitoring.{key}")
        
        return StandardResponse(
            success=True,
            message=f"Updated {len(updates)} configuration parameters",
            data={
                "session_id": session_id,
                "updated_params": updates
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# ============================================================================
# ENDPOINTS - CLIENT MANAGEMENT
# ============================================================================

@router.post("/sessions/{session_id}/clients", response_model=StandardResponse)
async def join_fl_session(
    session_id: str,
    request: JoinSessionRequest
):
    """Join an FL session as a client."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if session.status not in ["pending", "running"]:
            raise HTTPException(status_code=400, detail=f"Cannot join session with status: {session.status}")
        
        client = fl_manager.add_client(
            session_id=session_id,
            device_id=request.device_id,
            data_samples=request.data_samples,
            compute_capability=request.compute_capability
        )
        
        if not client:
            raise HTTPException(status_code=500, detail="Failed to add client")
        
        return StandardResponse(
            success=True,
            message=f"Device {request.device_id} joined session",
            data={
                "session_id": session_id,
                "client_id": client.client_id,
                "data_samples": client.data_samples,
                "current_round": session.current_round,
                "total_rounds": session.total_rounds,
                "total_clients": len(session.clients)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join session: {str(e)}")


@router.get("/sessions/{session_id}/clients", response_model=Dict[str, Any])
async def list_session_clients(session_id: str):
    """List all clients in an FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        clients = []
        for client in session.clients.values():
            # Get latest metrics if available
            latest_metrics = client.metrics_history[-1] if client.metrics_history else None
            
            clients.append({
                "client_id": client.client_id,
                "device_id": client.device_id,
                "data_samples": client.data_samples,
                "rounds_participated": len(client.rounds_participated),
                "contribution_score": client.contribution_score,
                "compute_capability": client.compute_capability,
                "is_active": client.is_active,
                "last_update": client.last_update.isoformat(),
                "latest_metrics": latest_metrics
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "total_clients": len(clients),
            "clients": clients
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list clients: {str(e)}")


@router.delete("/sessions/{session_id}/clients/{client_id}", response_model=StandardResponse)
async def remove_client(session_id: str, client_id: str):
    """Remove a client from an FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if client_id not in session.clients:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
        
        # Mark as inactive instead of removing (for history)
        session.clients[client_id].is_active = False
        
        return StandardResponse(
            success=True,
            message=f"Client {client_id} removed from session",
            data={
                "session_id": session_id,
                "client_id": client_id,
                "remaining_clients": sum(1 for c in session.clients.values() if c.is_active)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove client: {str(e)}")


# ============================================================================
# ENDPOINTS - METRICS AND MONITORING
# ============================================================================

@router.get("/sessions/{session_id}/metrics", response_model=Dict[str, Any])
async def get_session_metrics(
    session_id: str,
    start_round: Optional[int] = Query(None, ge=1),
    end_round: Optional[int] = Query(None, ge=1)
):
    """Get training metrics for an FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        metrics = []
        for round_num, m in sorted(session.round_metrics.items()):
            if start_round and round_num < start_round:
                continue
            if end_round and round_num > end_round:
                continue
            
            metrics.append({
                "round": m.round_num,
                "participating_clients": m.participating_clients,
                "avg_loss": m.avg_loss,
                "avg_accuracy": m.avg_accuracy,
                "min_accuracy": m.min_accuracy,
                "max_accuracy": m.max_accuracy,
                "std_accuracy": m.std_accuracy,
                "aggregation_time": m.aggregation_time,
                "communication_cost": m.communication_cost,
                "convergence_rate": m.convergence_rate,
                "fairness_index": m.fairness_index,
                "timestamp": m.timestamp.isoformat()
            })
        
        # Compute summary statistics
        if metrics:
            summary = {
                "total_rounds": len(metrics),
                "best_accuracy": max(m["avg_accuracy"] for m in metrics),
                "final_accuracy": metrics[-1]["avg_accuracy"] if metrics else 0,
                "avg_communication_cost": sum(m["communication_cost"] for m in metrics) / len(metrics),
                "total_aggregation_time": sum(m["aggregation_time"] for m in metrics),
                "avg_fairness": sum(m["fairness_index"] for m in metrics) / len(metrics)
            }
        else:
            summary = {}
        
        return {
            "success": True,
            "session_id": session_id,
            "summary": summary,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/sessions/{session_id}/convergence", response_model=Dict[str, Any])
async def get_convergence_analysis(session_id: str):
    """Get convergence analysis for an FL session."""
    try:
        session = fl_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        if not session.round_metrics:
            return {
                "success": True,
                "session_id": session_id,
                "message": "No metrics available yet"
            }
        
        # Extract accuracy progression
        rounds = sorted(session.round_metrics.keys())
        accuracies = [session.round_metrics[r].avg_accuracy for r in rounds]
        losses = [session.round_metrics[r].avg_loss for r in rounds]
        
        # Compute convergence metrics
        import numpy as np
        acc_array = np.array(accuracies)
        
        # Rate of improvement (moving average of differences)
        if len(acc_array) > 1:
            improvements = np.diff(acc_array)
            avg_improvement = float(np.mean(improvements))
            recent_improvement = float(np.mean(improvements[-5:])) if len(improvements) >= 5 else avg_improvement
        else:
            avg_improvement = 0
            recent_improvement = 0
        
        # Estimate rounds to convergence
        if recent_improvement > 0.001:
            target_acc = 0.95
            remaining = target_acc - acc_array[-1]
            estimated_rounds = int(remaining / recent_improvement) if recent_improvement > 0 else float('inf')
        else:
            estimated_rounds = None
        
        return {
            "success": True,
            "session_id": session_id,
            "convergence_analysis": {
                "current_accuracy": float(acc_array[-1]),
                "best_accuracy": float(np.max(acc_array)),
                "avg_improvement_per_round": avg_improvement,
                "recent_improvement_rate": recent_improvement,
                "estimated_rounds_to_95": estimated_rounds,
                "is_converging": recent_improvement > 0.0001,
                "accuracy_progression": accuracies,
                "loss_progression": losses
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze convergence: {str(e)}")


# ============================================================================
# ENDPOINTS - ALGORITHMS AND DATASETS INFO
# ============================================================================

@router.get("/algorithms", response_model=Dict[str, Any])
async def list_fl_algorithms():
    """List all available FL algorithms with descriptions."""
    algorithms = []
    for algo in FLAlgorithm:
        info = get_algorithm_info(algo)
        algorithms.append({
            "id": algo.value,
            **info
        })
    
    return {
        "success": True,
        "total_algorithms": len(algorithms),
        "algorithms": algorithms
    }


@router.get("/algorithms/{algorithm_id}", response_model=Dict[str, Any])
async def get_algorithm_details(algorithm_id: str):
    """Get detailed information about a specific FL algorithm."""
    try:
        algo = FLAlgorithm(algorithm_id.lower())
        info = get_algorithm_info(algo)
        
        return {
            "success": True,
            "algorithm": {
                "id": algo.value,
                **info
            }
        }
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm_id}' not found")


@router.get("/datasets", response_model=Dict[str, Any])
async def list_fl_datasets():
    """List all available built-in datasets."""
    datasets = []
    for ds in FLDataset:
        info = get_dataset_info(ds)
        datasets.append({
            "id": ds.value,
            **info
        })
    
    return {
        "success": True,
        "total_datasets": len(datasets),
        "datasets": datasets
    }


@router.get("/datasets/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset_details(dataset_id: str):
    """Get detailed information about a specific dataset."""
    try:
        ds = FLDataset(dataset_id.lower())
        info = get_dataset_info(ds)
        
        return {
            "success": True,
            "dataset": {
                "id": ds.value,
                **info
            }
        }
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


@router.get("/partition-strategies", response_model=Dict[str, Any])
async def list_partition_strategies():
    """List all available data partitioning strategies."""
    strategies = [
        {
            "id": PartitionStrategy.IID.value,
            "name": "IID (Independent and Identically Distributed)",
            "description": "Data is randomly shuffled and evenly distributed across clients"
        },
        {
            "id": PartitionStrategy.NON_IID_LABEL.value,
            "name": "Non-IID by Label",
            "description": "Each client receives data from only a subset of classes"
        },
        {
            "id": PartitionStrategy.NON_IID_DIRICHLET.value,
            "name": "Non-IID Dirichlet",
            "description": "Label distribution follows Dirichlet distribution (alpha controls heterogeneity)"
        },
        {
            "id": PartitionStrategy.NON_IID_QUANTITY.value,
            "name": "Non-IID by Quantity",
            "description": "Clients have varying amounts of data"
        },
        {
            "id": PartitionStrategy.PATHOLOGICAL.value,
            "name": "Pathological Non-IID",
            "description": "Extreme non-IID: each client has data from only 2 classes"
        },
        {
            "id": PartitionStrategy.PRACTICAL.value,
            "name": "Practical Non-IID",
            "description": "Realistic distribution mimicking real-world scenarios"
        }
    ]
    
    return {
        "success": True,
        "total_strategies": len(strategies),
        "strategies": strategies
    }


@router.get("/aggregation-methods", response_model=Dict[str, Any])
async def list_aggregation_methods():
    """List all available model aggregation methods."""
    methods = [
        {
            "id": AggregationMethod.WEIGHTED_AVERAGE.value,
            "name": "Weighted Average",
            "description": "Standard FedAvg-style weighted averaging by number of samples",
            "robust": False
        },
        {
            "id": AggregationMethod.MEDIAN.value,
            "name": "Coordinate-wise Median",
            "description": "Robust aggregation using median of each parameter",
            "robust": True
        },
        {
            "id": AggregationMethod.TRIMMED_MEAN.value,
            "name": "Trimmed Mean",
            "description": "Remove extreme values before averaging",
            "robust": True
        },
        {
            "id": AggregationMethod.KRUM.value,
            "name": "Krum",
            "description": "Byzantine-resilient aggregation selecting closest updates",
            "robust": True
        },
        {
            "id": AggregationMethod.MULTI_KRUM.value,
            "name": "Multi-Krum",
            "description": "Krum variant selecting multiple updates",
            "robust": True
        },
        {
            "id": AggregationMethod.BULYAN.value,
            "name": "Bulyan",
            "description": "Combines Krum with trimmed mean for stronger robustness",
            "robust": True
        },
        {
            "id": AggregationMethod.GEOMETRIC_MEDIAN.value,
            "name": "Geometric Median",
            "description": "Robust aggregation using geometric median",
            "robust": True
        }
    ]
    
    return {
        "success": True,
        "total_methods": len(methods),
        "methods": methods
    }


@router.get("/model-architectures", response_model=Dict[str, Any])
async def list_model_architectures():
    """List all available model architectures for FL."""
    architectures = [
        {"id": ModelArchitecture.CNN.value, "name": "CNN", "description": "Simple Convolutional Neural Network"},
        {"id": ModelArchitecture.RESNET18.value, "name": "ResNet-18", "description": "18-layer Residual Network"},
        {"id": ModelArchitecture.RESNET50.value, "name": "ResNet-50", "description": "50-layer Residual Network"},
        {"id": ModelArchitecture.MOBILENET.value, "name": "MobileNet", "description": "Efficient mobile architecture"},
        {"id": ModelArchitecture.EFFICIENTNET.value, "name": "EfficientNet", "description": "Scalable efficient architecture"},
        {"id": ModelArchitecture.VGG16.value, "name": "VGG-16", "description": "16-layer VGG network"},
        {"id": ModelArchitecture.LSTM.value, "name": "LSTM", "description": "Long Short-Term Memory network"},
        {"id": ModelArchitecture.TRANSFORMER.value, "name": "Transformer", "description": "Attention-based architecture"},
        {"id": ModelArchitecture.MLP.value, "name": "MLP", "description": "Multi-Layer Perceptron"},
        {"id": ModelArchitecture.CUSTOM.value, "name": "Custom", "description": "User-defined architecture"}
    ]
    
    return {
        "success": True,
        "total_architectures": len(architectures),
        "architectures": architectures
    }


# ============================================================================
# ENDPOINTS - PRESETS AND TEMPLATES
# ============================================================================

@router.get("/presets", response_model=Dict[str, Any])
async def list_fl_presets():
    """List pre-configured FL session templates."""
    presets = [
        {
            "id": "quick_start",
            "name": "Quick Start",
            "description": "Fast training with default settings for testing",
            "config": {
                "algorithm": "fedavg",
                "model_architecture": "cnn",
                "dataset": "mnist",
                "num_rounds": 10,
                "local_epochs": 1,
                "num_partitions": 5
            }
        },
        {
            "id": "production",
            "name": "Production",
            "description": "Optimized settings for production deployment",
            "config": {
                "algorithm": "fedadam",
                "model_architecture": "resnet18",
                "dataset": "cifar10",
                "num_rounds": 100,
                "local_epochs": 5,
                "num_partitions": 10,
                "differential_privacy": True
            }
        },
        {
            "id": "non_iid",
            "name": "Non-IID Robust",
            "description": "Optimized for heterogeneous data distributions",
            "config": {
                "algorithm": "fedprox",
                "model_architecture": "cnn",
                "dataset": "cifar10",
                "partition_strategy": "non_iid_dirichlet",
                "dirichlet_alpha": 0.1,
                "num_rounds": 200,
                "proximal_mu": 0.1
            }
        },
        {
            "id": "privacy_focused",
            "name": "Privacy Focused",
            "description": "Maximum privacy with differential privacy and secure aggregation",
            "config": {
                "algorithm": "fedavg",
                "model_architecture": "cnn",
                "dataset": "mnist",
                "differential_privacy": True,
                "noise_multiplier": 1.5,
                "secure_aggregation": True,
                "num_rounds": 50
            }
        },
        {
            "id": "fair_learning",
            "name": "Fair Learning",
            "description": "Optimized for fairness across clients",
            "config": {
                "algorithm": "qfedavg",
                "model_architecture": "cnn",
                "dataset": "cifar10",
                "q_param": 0.5,
                "num_rounds": 100
            }
        }
    ]
    
    return {
        "success": True,
        "total_presets": len(presets),
        "presets": presets
    }


@router.post("/presets/{preset_id}/apply", response_model=StandardResponse)
async def apply_fl_preset(
    preset_id: str,
    session_name: str = Query(..., min_length=1),
    background_tasks: BackgroundTasks = None
):
    """Create an FL session from a preset template."""
    presets = {
        "quick_start": {
            "algorithm": "fedavg",
            "model_architecture": "cnn",
            "data": {"dataset": "mnist", "num_partitions": 5},
            "server": {"num_rounds": 10},
            "client": {"local_epochs": 1}
        },
        "production": {
            "algorithm": "fedadam",
            "model_architecture": "resnet18",
            "data": {"dataset": "cifar10", "num_partitions": 10},
            "server": {"num_rounds": 100},
            "client": {"local_epochs": 5},
            "privacy": {"differential_privacy": True}
        },
        "non_iid": {
            "algorithm": "fedprox",
            "model_architecture": "cnn",
            "data": {"dataset": "cifar10", "partition_strategy": "non_iid_dirichlet", "dirichlet_alpha": 0.1},
            "server": {"num_rounds": 200},
            "algorithm_params": {"proximal_mu": 0.1}
        },
        "privacy_focused": {
            "algorithm": "fedavg",
            "model_architecture": "cnn",
            "data": {"dataset": "mnist"},
            "server": {"num_rounds": 50},
            "privacy": {"differential_privacy": True, "noise_multiplier": 1.5, "secure_aggregation": True}
        },
        "fair_learning": {
            "algorithm": "qfedavg",
            "model_architecture": "cnn",
            "data": {"dataset": "cifar10"},
            "server": {"num_rounds": 100},
            "algorithm_params": {"q_param": 0.5}
        }
    }
    
    if preset_id not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    
    preset = presets[preset_id]
    
    # Build request from preset
    request = CreateFLSessionRequest(
        session_name=session_name,
        algorithm=preset.get("algorithm", "fedavg"),
        model_architecture=preset.get("model_architecture", "cnn"),
        server=ServerConfigRequest(**preset.get("server", {})) if "server" in preset else None,
        client=ClientConfigRequest(**preset.get("client", {})) if "client" in preset else None,
        algorithm_params=AlgorithmConfigRequest(**preset.get("algorithm_params", {})) if "algorithm_params" in preset else None,
        data=DataConfigRequest(**preset.get("data", {})) if "data" in preset else None,
        privacy=PrivacyConfigRequest(**preset.get("privacy", {})) if "privacy" in preset else None
    )
    
    return await create_fl_session(request, background_tasks)
