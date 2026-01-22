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
import logging
from collections import defaultdict

# Import shared models
from .models import StandardResponse

# Import real training functions
from server.ml_training import (
    train_model,
    IMUClassifier,
    load_dataset_from_db,
    save_model_to_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

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
# REAL TRAINING FUNCTIONS
# ============================================================================

async def run_training(job: TrainingJob, db_session=None):
    """Run real model training using PyTorch.
    
    Uses IMUClassifier from ml_training.py for actual neural network training.
    Requires a dataset_id in the job config to load real data.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    import time
    
    try:
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        logger.info(f"Starting training job {job.job_id}")
        logger.info(f"  Model: {job.config.model}, Epochs: {job.config.epochs}")
        
        # Check if we have a dataset_id to load real data
        dataset_id = getattr(job.config, 'dataset_id', None)
        
        if dataset_id and db_session:
            # Load real data from database
            logger.info(f"Loading dataset {dataset_id} from database...")
            X, y, class_names = load_dataset_from_db(
                db_session=db_session,
                dataset_id=dataset_id,
                window_size=128,
                output_shape='sequence'
            )
            logger.info(f"Loaded {len(X)} samples, {len(class_names)} classes")
        else:
            raise ValueError(
                "No dataset_id provided. Please create a dataset with labeled files "
                "and provide the dataset_id in the training configuration. "
                "Use the /datasets endpoints to create and manage datasets."
            )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=job.config.validation_split, 
            random_state=42, stratify=y
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=job.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=job.config.batch_size, shuffle=False)
        
        # Determine input shape
        seq_length = X_train.shape[1]
        input_channels = X_train.shape[2] if len(X_train.shape) > 2 else 1
        num_classes = len(class_names)
        
        # Create model
        model = IMUClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            architecture_size='medium'
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        # Training callback to update job progress
        def training_callback(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
            job.current_epoch = epoch
            
            # Create metrics record
            metrics = TrainingMetrics(
                job_id=job.job_id,
                epoch=epoch,
                batch=len(train_loader),
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=job.config.learning_rate,
                time_per_epoch=time.time() - job.started_at.timestamp() if job.started_at else 0,
                estimated_time_remaining=(total_epochs - epoch) * 2.0,
                memory_usage=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                gpu_usage=None
            )
            
            # Update job metrics
            for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
                if key not in job.metrics:
                    job.metrics[key] = []
                value = getattr(metrics, key)
                if value is not None:
                    job.metrics[key].append(value)
            
            # Track best metrics
            if val_acc is not None:
                if "val_accuracy" not in job.best_metrics or val_acc > job.best_metrics["val_accuracy"]:
                    job.best_metrics["val_accuracy"] = val_acc
                    job.best_metrics["best_epoch"] = epoch
            
            metrics_history[job.job_id].append(metrics)
            logger.info(f"Epoch {epoch}/{total_epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Run real training
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=job.config.epochs,
            learning_rate=job.config.learning_rate,
            device=device,
            callback=training_callback
        )
        
        # Save model
        if job.status == TrainingStatus.RUNNING:
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Save model bytes
            model_bytes = save_model_to_bytes(model, {
                'model_type': job.config.model.value,
                'num_classes': num_classes,
                'class_names': class_names,
                'input_channels': input_channels,
                'seq_length': seq_length
            })
            job.model_path = f"/models/{job.job_id}/model.pth"
            
            logger.info(f"Training completed. Best val accuracy: {results['best_val_accuracy']:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        job.status = TrainingStatus.FAILED
        job.error_message = str(e)

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
        
        # Start real training in background
        from server.db import get_db_session
        db_session = get_db_session()
        background_tasks.add_task(run_training, job, db_session)
        
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
    
    DEPRECATED: This endpoint is deprecated. Please use /fl/sessions endpoint instead
    for full Flower framework integration with real federated learning.
    
    This endpoint now redirects to the proper Flower FL implementation.
    """
    # Redirect to the proper Flower FL endpoints
    raise HTTPException(
        status_code=308,
        detail={
            "message": "This endpoint is deprecated. Use /fl/sessions for Flower-based federated learning.",
            "redirect_to": "/fl/sessions",
            "documentation": "POST /fl/sessions to create a new FL session with Flower framework"
        }
    )

@router.get("/federated/status", response_model=Union[FederatedSession, Dict[str, Any]])
async def get_federated_status(
    session_id: Optional[str] = Query(None, description="Specific session ID")
):
    """Monitor federated learning session and client contributions.
    
    DEPRECATED: Use /fl/sessions or /fl/sessions/{session_id} instead.
    """
    raise HTTPException(
        status_code=308,
        detail={
            "message": "This endpoint is deprecated. Use /fl/sessions for Flower-based federated learning.",
            "redirect_to": "/fl/sessions" if not session_id else f"/fl/sessions/{session_id}",
            "documentation": "GET /fl/sessions to list sessions, GET /fl/sessions/{id} for details"
        }
    )

@router.post("/federated/{session_id}/join", response_model=StandardResponse)
async def join_federated_session(
    session_id: str,
    device_id: str,
    data_samples: int = Query(..., ge=1, description="Number of local data samples")
):
    """Join an existing federated learning session as a client.
    
    DEPRECATED: Use /fl/sessions/{session_id}/clients endpoint instead.
    """
    raise HTTPException(
        status_code=308,
        detail={
            "message": "This endpoint is deprecated. Use /fl/sessions/{session_id}/clients for Flower-based federated learning.",
            "redirect_to": f"/fl/sessions/{session_id}/clients",
            "documentation": "POST /fl/sessions/{session_id}/clients to join a session"
        }
    )

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
            
            # Calculate actual model size if available
            model_size = None
            if hasattr(job, 'model_bytes') and job.model_bytes:
                model_size = len(job.model_bytes) / (1024 * 1024)  # Convert to MB
            
            model_info = {
                "model_id": job_id,
                "model_name": job.config.model_name or f"{job.config.model}_model",
                "architecture": job.config.model,
                "training_mode": job.config.mode,
                "accuracy": job.best_metrics.get("val_accuracy"),
                "model_path": job.model_path,
                "created_at": job.completed_at.isoformat() if job.completed_at else None,
                "device_id": job.config.device_id,
                "size_mb": model_size
            }
            models.append(model_info)
        
        # Note: Federated sessions are now handled by /fl/sessions endpoint
        # This legacy code is kept for backward compatibility but redirects to FL endpoints
        
        return {
            "success": True,
            "total_models": len(models),
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
