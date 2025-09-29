"""Training and Federated Learning Endpoints for Thoth Device.

This module handles:
- On-device model training
- Federated learning sessions
- Training status monitoring
- Model deployment and management
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
import random
from collections import defaultdict

# Import shared models
from .models import StandardResponse

router = APIRouter(prefix="/training", tags=["training"])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def simulate_training_progress(job_id: str):
    """Simulate training progress for demo purposes"""
    if job_id not in training_jobs:
        return
    
    job = training_jobs[job_id]
    epochs = job["config"].get("epochs", 10)
    
    for epoch in range(1, epochs + 1):
        if job["status"] != "running":
            break
            
        await asyncio.sleep(2)  # Simulate training time
        
        # Simulate improving metrics
        base_loss = 1.0 - (epoch / epochs) * 0.8
        base_acc = 0.5 + (epoch / epochs) * 0.4
        
        job["progress"] = int((epoch / epochs) * 100)
        job["metrics"] = {
            "epoch": epoch,
            "loss": base_loss + random.uniform(-0.1, 0.1),
            "accuracy": base_acc + random.uniform(-0.05, 0.05),
            "val_loss": base_loss + random.uniform(-0.05, 0.15),
            "val_accuracy": base_acc + random.uniform(-0.1, 0.1)
        }
        job["logs"].append(f"Epoch {epoch}/{epochs} - Loss: {job['metrics']['loss']:.4f}, Acc: {job['metrics']['accuracy']:.4f}")
    
    if job["status"] == "running":
        job["status"] = "completed"
        job["progress"] = 100
        job["logs"].append("Training completed successfully")

# ============================================================================
# ENUMS
# ============================================================================

class ModelType(str, Enum):
    """Supported model architectures."""
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    LINEAR = "linear"
    CUSTOM = "custom"

class TrainingMode(str, Enum):
    """Training execution modes."""
    ON_DEVICE = "on-device"
    CLOUD = "cloud"
    EDGE = "edge"
    FEDERATED = "federated"

class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class DataSource(str, Enum):
    """Training data sources."""
    SENSORS = "sensors"
    IMAGES = "images"
    AUDIO = "audio"
    TEXT = "text"
    CUSTOM = "custom"

# ============================================================================
# MODELS
# ============================================================================

class TrainingConfig(BaseModel):
    """Configuration for a training job."""
    model: ModelType
    data: DataSource
    mode: TrainingMode
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = ["accuracy"]
    device_id: str = "thoth-001"
    save_model: bool = True
    model_name: Optional[str] = None

class TrainingJob(BaseModel):
    """Training job information."""
    job_id: str
    config: TrainingConfig
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_epoch: int = 0
    total_epochs: int
    metrics: Dict[str, List[float]] = {}
    best_metrics: Dict[str, float] = {}
    error_message: Optional[str] = None
    model_path: Optional[str] = None

class TrainingMetrics(BaseModel):
    """Real-time training metrics."""
    job_id: str
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float
    time_per_epoch: float
    estimated_time_remaining: float
    memory_usage: float  # MB
    gpu_usage: Optional[float] = None  # Percentage

class FederatedConfig(BaseModel):
    """Federated learning configuration."""
    session_name: str
    num_rounds: int = 10
    min_clients: int = 2
    max_clients: int = 10
    client_fraction: float = 1.0
    differential_privacy: bool = False
    noise_multiplier: float = 1.0
    clip_norm: float = 1.0
    secure_aggregation: bool = False
    training_config: TrainingConfig

class FederatedSession(BaseModel):
    """Federated learning session."""
    session_id: str
    config: FederatedConfig
    status: TrainingStatus
    current_round: int = 0
    total_rounds: int
    connected_clients: List[str] = []
    round_metrics: Dict[int, Dict[str, float]] = {}
    global_model_path: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class FederatedClient(BaseModel):
    """Federated learning client information."""
    client_id: str
    device_id: str
    session_id: str
    local_epochs: int = 5
    local_batch_size: int = 32
    data_samples: int
    last_update: datetime
    rounds_participated: List[int] = []
    contribution_score: float = 0.0

# ============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# ============================================================================

# Active training jobs
training_jobs: Dict[str, TrainingJob] = {}

# Federated learning sessions
federated_sessions: Dict[str, FederatedSession] = {}

# Federated clients
federated_clients: Dict[str, FederatedClient] = {}

# Training metrics history
metrics_history: Dict[str, List[TrainingMetrics]] = defaultdict(list)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def simulate_training(job: TrainingJob):
    """Simulate model training (replace with actual training in production)."""
    job.status = TrainingStatus.RUNNING
    job.started_at = datetime.now()
    
    for epoch in range(job.config.epochs):
        if job.status == TrainingStatus.CANCELLED:
            break
            
        # Simulate epoch training
        await asyncio.sleep(2)  # Simulate training time
        
        job.current_epoch = epoch + 1
        
        # Generate mock metrics
        metrics = TrainingMetrics(
            job_id=job.job_id,
            epoch=epoch + 1,
            batch=100,
            loss=0.5 * (0.9 ** epoch) + random.uniform(-0.05, 0.05),
            accuracy=min(0.95, 0.6 + 0.03 * epoch + random.uniform(-0.02, 0.02)),
            val_loss=0.6 * (0.9 ** epoch) + random.uniform(-0.05, 0.05),
            val_accuracy=min(0.93, 0.55 + 0.03 * epoch + random.uniform(-0.02, 0.02)),
            learning_rate=job.config.learning_rate,
            time_per_epoch=2.0,
            estimated_time_remaining=(job.config.epochs - epoch - 1) * 2.0,
            memory_usage=random.uniform(100, 500),
            gpu_usage=random.uniform(40, 90) if job.config.mode != TrainingMode.ON_DEVICE else None
        )
        
        # Update job metrics
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            if key not in job.metrics:
                job.metrics[key] = []
            value = getattr(metrics, key)
            if value is not None:
                job.metrics[key].append(value)
        
        # Track best metrics
        if metrics.val_accuracy:
            if "val_accuracy" not in job.best_metrics or metrics.val_accuracy > job.best_metrics["val_accuracy"]:
                job.best_metrics["val_accuracy"] = metrics.val_accuracy
                job.best_metrics["best_epoch"] = epoch + 1
        
        # Store metrics
        metrics_history[job.job_id].append(metrics)
    
    # Complete training
    if job.status == TrainingStatus.RUNNING:
        job.status = TrainingStatus.COMPLETED
        job.completed_at = datetime.now()
        job.model_path = f"/models/{job.job_id}/model.h5"

async def simulate_federated_round(session: FederatedSession, round_num: int):
    """Simulate a federated learning round."""
    # Select clients for this round
    num_clients = int(len(session.connected_clients) * session.config.client_fraction)
    selected_clients = random.sample(session.connected_clients, min(num_clients, len(session.connected_clients)))
    
    # Simulate client training
    await asyncio.sleep(5)  # Simulate round time
    
    # Generate round metrics
    round_metrics = {
        "round": round_num,
        "clients": len(selected_clients),
        "avg_loss": 0.4 * (0.9 ** round_num) + random.uniform(-0.05, 0.05),
        "avg_accuracy": min(0.92, 0.65 + 0.025 * round_num + random.uniform(-0.02, 0.02)),
        "convergence_rate": min(1.0, 0.1 * round_num)
    }
    
    session.round_metrics[round_num] = round_metrics
    session.current_round = round_num
    
    # Update client participation
    for client_id in selected_clients:
        if client_id in federated_clients:
            federated_clients[client_id].rounds_participated.append(round_num)
            federated_clients[client_id].last_update = datetime.now()

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/training/setup", response_model=StandardResponse)
async def setup_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """Start a new training job with specified configuration.
    
    Supports various model architectures and training modes:
    - On-device: Train directly on Thoth device
    - Cloud: Offload to cloud infrastructure
    - Edge: Distributed edge computing
    - Federated: Privacy-preserving collaborative learning
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            config=config,
            status=TrainingStatus.PENDING,
            created_at=datetime.now(),
            total_epochs=config.epochs,
            model_name=config.model_name or f"{config.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Store job
        training_jobs[job_id] = job
        
        # Start training in background
        background_tasks.add_task(simulate_training, job)
        
        return StandardResponse(
            success=True,
            message=f"Training job {job_id} created successfully",
            data={
                "job_id": job_id,
                "model": config.model,
                "mode": config.mode,
                "epochs": config.epochs,
                "status": job.status
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup training: {str(e)}")

@router.get("/training/status", response_model=Union[TrainingJob, Dict[str, Any]])
async def get_training_status(
    job_id: Optional[str] = Query(None, description="Specific job ID"),
    device_id: Optional[str] = Query(None, description="Filter by device ID")
):
    """Get real-time training status and metrics.
    
    Returns current epoch, loss, accuracy, and other metrics.
    If no job_id is provided, returns all active jobs.
    """
    try:
        if job_id:
            if job_id not in training_jobs:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
            
            job = training_jobs[job_id]
            
            # Get latest metrics
            latest_metrics = None
            if job_id in metrics_history and metrics_history[job_id]:
                latest_metrics = metrics_history[job_id][-1].model_dump()
            
            response = job.model_dump()
            response["latest_metrics"] = latest_metrics
            
            return response
        else:
            # Return all jobs, optionally filtered by device
            jobs = []
            for jid, job in training_jobs.items():
                if device_id and job["config"]["device_id"] != device_id:
                    continue
                jobs.append(job)
            
            return {
                "success": True,
                "jobs": jobs,
                "total": len(jobs)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.post("/training/control/{job_id}")
async def control_training(
    job_id: str,
    action: str = Query(..., description="Action to perform: pause, resume, cancel")
):
    """Control an active training job.
    
    Actions:
    - pause: Temporarily pause training
    - resume: Resume paused training
    - cancel: Cancel and cleanup training
    """
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        job = training_jobs[job_id]
        
        if action == "pause":
            if job.status != TrainingStatus.RUNNING:
                raise ValueError("Can only pause running jobs")
            job.status = TrainingStatus.PAUSED
            message = f"Training job {job_id} paused"
            
        elif action == "resume":
            if job.status != TrainingStatus.PAUSED:
                raise ValueError("Can only resume paused jobs")
            job.status = TrainingStatus.RUNNING
            message = f"Training job {job_id} resumed"
            
        elif action == "cancel":
            if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                raise ValueError("Cannot cancel completed or failed jobs")
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            message = f"Training job {job_id} cancelled"
            
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return StandardResponse(
            success=True,
            message=message,
            data={
                "job_id": job_id,
                "status": job.status,
                "action": action
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/federated/train", response_model=StandardResponse)
async def start_federated_training(
    config: FederatedConfig,
    background_tasks: BackgroundTasks
):
    """Start a federated learning session using Flower framework.
    
    Enables privacy-preserving collaborative training across multiple devices
    with optional differential privacy and secure aggregation.
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create federated session
        session = FederatedSession(
            session_id=session_id,
            config=config,
            status=TrainingStatus.PENDING,
            total_rounds=config.num_rounds,
            created_at=datetime.now()
        )
        
        # Store session
        federated_sessions[session_id] = session
        
        # Start federated training simulation
        async def run_federated_training():
            session.status = TrainingStatus.RUNNING
            session.started_at = datetime.now()
            
            # Simulate client connections
            for i in range(config.min_clients):
                client_id = f"client_{i+1}"
                session.connected_clients.append(client_id)
                
                # Create client record
                client = FederatedClient(
                    client_id=client_id,
                    device_id=f"thoth-{i+1:03d}",
                    session_id=session_id,
                    data_samples=random.randint(100, 1000),
                    last_update=datetime.now()
                )
                federated_clients[client_id] = client
            
            # Run federated rounds
            for round_num in range(1, config.num_rounds + 1):
                if session.status == TrainingStatus.CANCELLED:
                    break
                    
                await simulate_federated_round(session, round_num)
            
            # Complete session
            if session.status == TrainingStatus.RUNNING:
                session.status = TrainingStatus.COMPLETED
                session.completed_at = datetime.now()
                session.global_model_path = f"/models/federated/{session_id}/global_model.h5"
        
        background_tasks.add_task(run_federated_training)
        
        return StandardResponse(
            success=True,
            message=f"Federated session {session_id} created",
            data={
                "session_id": session_id,
                "session_name": config.session_name,
                "num_rounds": config.num_rounds,
                "min_clients": config.min_clients,
                "differential_privacy": config.differential_privacy,
                "status": session.status
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start federated training: {str(e)}")

@router.get("/federated/status", response_model=Union[FederatedSession, Dict[str, Any]])
async def get_federated_status(
    session_id: Optional[str] = Query(None, description="Specific session ID")
):
    """Monitor federated learning session and client contributions.
    
    Returns round progress, client participation, and aggregated metrics.
    """
    try:
        if session_id:
            if session_id not in federated_sessions:
                raise HTTPException(status_code=404, detail=f"Federated session {session_id} not found")
            
            session = federated_sessions[session_id]
            
            # Add client details
            client_details = []
            for client_id in session.connected_clients:
                if client_id in federated_clients:
                    client = federated_clients[client_id]
                    client_details.append({
                        "client_id": client_id,
                        "device_id": client.device_id,
                        "data_samples": client.data_samples,
                        "rounds_participated": len(client.rounds_participated),
                        "last_update": client.last_update.isoformat()
                    })
            
            response = session.model_dump()
            response["client_details"] = client_details
            
            return response
        else:
            # Return all sessions
            sessions = []
            for sid, session in federated_sessions.items():
                sessions.append({
                    "session_id": sid,
                    "session_name": session.config.session_name,
                    "status": session.status,
                    "progress": f"{session.current_round}/{session.total_rounds}",
                    "clients": len(session.connected_clients),
                    "created_at": session.created_at.isoformat()
                })
            
            return {
                "success": True,
                "total_sessions": len(sessions),
                "sessions": sessions
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get federated status: {str(e)}")

@router.post("/federated/{session_id}/join", response_model=StandardResponse)
async def join_federated_session(
    session_id: str,
    device_id: str,
    data_samples: int = Query(..., ge=1, description="Number of local data samples")
):
    """Join an existing federated learning session as a client.
    
    Allows devices to participate in collaborative training.
    """
    try:
        if session_id not in federated_sessions:
            raise HTTPException(status_code=404, detail=f"Federated session {session_id} not found")
        
        session = federated_sessions[session_id]
        
        if session.status != TrainingStatus.RUNNING:
            raise ValueError("Can only join running sessions")
        
        if len(session.connected_clients) >= session.config.max_clients:
            raise ValueError("Session has reached maximum client capacity")
        
        # Generate client ID
        client_id = f"client_{device_id}"
        
        if client_id in session.connected_clients:
            raise ValueError("Device already joined this session")
        
        # Add client to session
        session.connected_clients.append(client_id)
        
        # Create client record
        client = FederatedClient(
            client_id=client_id,
            device_id=device_id,
            session_id=session_id,
            data_samples=data_samples,
            last_update=datetime.now()
        )
        federated_clients[client_id] = client
        
        return StandardResponse(
            success=True,
            message=f"Device {device_id} joined federated session",
            data={
                "session_id": session_id,
                "client_id": client_id,
                "current_round": session.current_round,
                "total_rounds": session.total_rounds
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/training/models", response_model=Dict[str, Any])
async def list_trained_models(
    device_id: Optional[str] = Query(None, description="Filter by device ID")
):
    """List all trained models available for deployment.
    
    Returns model metadata including accuracy, size, and compatibility.
    """
    try:
        models = []
        
        # Get models from completed training jobs
        for job_id, job in training_jobs.items():
            if job.status != TrainingStatus.COMPLETED:
                continue
                
            if device_id and job.config.device_id != device_id:
                continue
            
            model_info = {
                "model_id": job_id,
                "model_name": job.config.model_name or f"{job.config.model}_model",
                "architecture": job.config.model,
                "training_mode": job.config.mode,
                "accuracy": job.best_metrics.get("val_accuracy"),
                "model_path": job.model_path,
                "created_at": job.completed_at.isoformat() if job.completed_at else None,
                "device_id": job.config.device_id,
                "size_mb": random.uniform(1, 50)  # Mock size
            }
            models.append(model_info)
        
        # Get models from federated sessions
        for session_id, session in federated_sessions.items():
            if session.status != TrainingStatus.COMPLETED:
                continue
                
            model_info = {
                "model_id": session_id,
                "model_name": f"federated_{session.config.session_name}",
                "architecture": session.config.training_config.model,
                "training_mode": "federated",
                "accuracy": max(session.round_metrics.values(), key=lambda x: x.get("avg_accuracy", 0)).get("avg_accuracy") if session.round_metrics else None,
                "model_path": session.global_model_path,
                "created_at": session.completed_at.isoformat() if session.completed_at else None,
                "num_clients": len(session.connected_clients),
                "size_mb": random.uniform(5, 100)  # Mock size
            }
            models.append(model_info)
        
        return {
            "success": True,
            "total_models": len(models),
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
