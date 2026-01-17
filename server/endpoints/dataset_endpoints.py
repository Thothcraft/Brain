"""Dataset and Cloud Training Endpoints.

This module handles:
- Dataset creation and management
- File labeling for training
- Cloud training job management
- Model evaluation and deployment
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
import uuid
import json
import asyncio
import random
import logging

from sqlalchemy.orm import selectinload

from ..db import get_db, TrainingDataset, DatasetFile, TrainingJob, TrainedModel, File
from ..auth import get_current_user
from .models import StandardResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateDatasetRequest(BaseModel):
    """Request to create a new training dataset."""
    name: str
    description: Optional[str] = None


class AddFilesToDatasetRequest(BaseModel):
    """Request to add files with labels to a dataset."""
    files: List[Dict[str, Any]]  # [{file_id: int, label: str}, ...]


class UpdateFileLabelRequest(BaseModel):
    """Request to update a file's label."""
    label: str


class CloudTrainingRequest(BaseModel):
    """Request to start cloud training."""
    dataset_id: int
    test_dataset_id: Optional[int] = None
    model_type: str = "cnn"  # cnn, lstm, transformer, linear
    model_architecture: str = "small"  # small, medium, large
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    model_name: Optional[str] = None
    window_size: int = 128  # Window size for time series data
    # Bayesian optimization settings
    use_bayesian_optimization: bool = False
    bayesian_trials: int = 20
    bayesian_epochs_per_trial: int = 3
    bayesian_lr_min: float = 0.00001
    bayesian_lr_max: float = 0.01
    bayesian_lr_scale: str = "log"  # 'log' or 'linear'
    bayesian_batch_sizes: List[int] = [16, 32, 64, 128]
    bayesian_weight_decay_min: float = 0.0
    bayesian_weight_decay_max: float = 0.01
    bayesian_optimizers: List[str] = ["adam", "adamw", "sgd"]
    bayesian_exploration_rate: float = 0.3
    bayesian_search_architecture: bool = False  # Whether to search architecture sizes


# ============================================================================
# DATASET ENDPOINTS
# ============================================================================

@router.post("/create", response_model=Dict[str, Any])
async def create_dataset(
    request: CreateDatasetRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new training dataset."""
    try:
        dataset = TrainingDataset(
            user_id=current_user.userId,
            name=request.name,
            description=request.description
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return {
            "success": True,
            "message": f"Dataset '{request.name}' created successfully",
            "dataset": dataset.to_dict()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@router.get("/list", response_model=Dict[str, Any])
async def list_datasets(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all datasets for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DATASETS] Starting datasets query for user {current_user.userId}")
        
        # Optimized query - select only necessary columns
        datasets = db.query(
            TrainingDataset.id,
            TrainingDataset.name,
            TrainingDataset.description,
            TrainingDataset.created_at,
            TrainingDataset.updated_at
        ).filter(
            TrainingDataset.user_id == current_user.userId
        ).order_by(TrainingDataset.created_at.desc()).limit(100).all()  # Add limit to prevent large result sets
        
        logger.info(f"[DATASETS] Query completed, found {len(datasets)} datasets")
        
        # Convert to dict efficiently
        dataset_list = []
        for d in datasets:
            dataset_list.append({
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            })
        
        return {
            "success": True,
            "datasets": dataset_list,
            "total": len(dataset_list),
            "operation": "list_datasets",
            "status": "completed"
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[DATASETS] Error listing datasets: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "datasets": [],
            "total": 0,
            "operation": "list_datasets",
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# CLOUD TRAINING ENDPOINTS (must be before /{dataset_id} catch-all)
# ============================================================================

def run_cloud_training(job_id: str, db_url: str):
    """Run real PyTorch training for IMU model in background."""
    import asyncio
    import traceback
    import sys
    
    print(f"[TRAINING-DEBUG] ========================================")
    print(f"[TRAINING-DEBUG] run_cloud_training CALLED for job {job_id}")
    print(f"[TRAINING-DEBUG] db_url: {db_url[:50]}...")
    print(f"[TRAINING-DEBUG] Thread: {__import__('threading').current_thread().name}")
    print(f"[TRAINING-DEBUG] ========================================")
    sys.stdout.flush()
    
    loop = None
    try:
        # Create new event loop for this thread
        print(f"[TRAINING-DEBUG] Creating new event loop...")
        sys.stdout.flush()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print(f"[TRAINING-DEBUG] Event loop created: {loop}")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Starting run_until_complete...")
        sys.stdout.flush()
        loop.run_until_complete(_run_cloud_training_async(job_id, db_url))
        print(f"[TRAINING-DEBUG] run_until_complete finished successfully")
        sys.stdout.flush()
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"[TRAINING-ERROR] Exception in run_cloud_training: {error_msg}")
        print(f"[TRAINING-ERROR] Traceback: {error_traceback}")
        sys.stdout.flush()
        
        # Mark job as failed in database
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job and job.status in ["pending", "running"]:
                job.status = "failed"
                job.error_message = f"Training crashed: {error_msg}"
                job.completed_at = datetime.utcnow()
                db.commit()
                print(f"[TRAINING-DEBUG] Job {job_id} marked as failed in database")
            db.close()
        except Exception as db_error:
            print(f"[TRAINING-ERROR] Failed to update job status in DB: {db_error}")
        sys.stdout.flush()
    finally:
        if loop:
            print(f"[TRAINING-DEBUG] Closing event loop...")
            sys.stdout.flush()
            loop.close()
            print(f"[TRAINING-DEBUG] Event loop closed")
            sys.stdout.flush()


async def _run_cloud_training_async(job_id: str, db_url: str):
    """Async implementation of cloud training."""
    import traceback
    import sys
    
    print(f"[TRAINING-DEBUG] _run_cloud_training_async started for job {job_id}")
    sys.stdout.flush()
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from server.ml_training import run_full_training
    
    print(f"[TRAINING-DEBUG] Imports successful")
    sys.stdout.flush()
    
    print(f"[TRAINING-DEBUG] Creating database engine...")
    sys.stdout.flush()
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    print(f"[TRAINING-DEBUG] Database session created")
    sys.stdout.flush()
    
    try:
        print(f"[TRAINING-DEBUG] Querying job {job_id}...")
        sys.stdout.flush()
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            print(f"[TRAINING-ERROR] Job {job_id} not found in database!")
            sys.stdout.flush()
            return
        
        print(f"[TRAINING-DEBUG] Job found: dataset_id={job.dataset_id}, model_type={job.model_type}")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Setting job status to 'running'...")
        sys.stdout.flush()
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        print(f"[TRAINING-DEBUG] Job status updated to 'running'")
        sys.stdout.flush()
        
        config = json.loads(job.config) if job.config else {}
        print(f"[TRAINING-DEBUG] Config loaded: {list(config.keys())}")
        sys.stdout.flush()
        
        # Define update callback for progress tracking
        async def update_callback(status, epoch, total, metrics):
            nonlocal job, db
            try:
                print(f"[TRAINING-DEBUG] Update callback: status={status}, epoch={epoch}")
                sys.stdout.flush()
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = status
                    if epoch > 0:
                        job.current_epoch = epoch
                    db.commit()
            except Exception as e:
                print(f"[TRAINING-WARNING] Failed to update job status: {e}")
                sys.stdout.flush()
        
        # Run real PyTorch training
        print(f"[TRAINING-DEBUG] ========================================")
        print(f"[TRAINING-DEBUG] CALLING run_full_training...")
        print(f"[TRAINING-DEBUG] job_id={job_id}")
        print(f"[TRAINING-DEBUG] dataset_id={job.dataset_id}")
        print(f"[TRAINING-DEBUG] model_type={job.model_type}")
        print(f"[TRAINING-DEBUG] ========================================")
        sys.stdout.flush()
        
        results = await run_full_training(
            job_id=job_id,
            dataset_id=job.dataset_id,
            db_session=db,
            model_type=job.model_type,
            config=config,
            update_callback=update_callback
        )
        
        # Store Bayesian trials if available
        if results.get('bayesian_trials_data'):
            config['bayesian_trials_results'] = results['bayesian_trials_data']
            job.config = json.dumps(config)
        
        # Store model bytes
        model_bytes = results.get('model_bytes')
        class_names = results.get('class_names', [])
        
        job.metrics = json.dumps({
            "loss": results["train_losses"],
            "accuracy": results["train_accuracies"],
            "val_loss": results["val_losses"],
            "val_accuracy": results["val_accuracies"]
        })
        
        best_metrics = {
            "val_accuracy": results["best_val_accuracy"],
            "best_epoch": results["best_epoch"],
            "per_class_metrics": results.get("per_class_metrics", {}),
            "confusion_matrix": results.get("confusion_matrix", []),
            "class_names": results.get("class_names", []),
            "roc_curves": results.get("roc_curves", {}),
            "pr_curves": results.get("pr_curves", {}),
            "model_architecture": results.get("model_architecture", {}),
            "learning_rate": results.get("model_architecture", {}).get("learning_rate", 0.001),
            "batch_size": results.get("model_architecture", {}).get("batch_size", 32),
            "optimizer": results.get("model_architecture", {}).get("optimizer", "adam")
        }
        
        if results.get("test_results"):
            best_metrics["test_accuracy"] = results["test_results"]["test_accuracy"]
            best_metrics["test_loss"] = results["test_results"]["test_loss"]
            best_metrics["test_dataset_id"] = results["test_results"]["test_dataset_id"]
        
        job.best_metrics = json.dumps(best_metrics)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        
        # Save trained model (for both mock and real training)
        try:
            from sqlalchemy import text
            try:
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS trained_model (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        job_id VARCHAR(255),
                        name VARCHAR(255) NOT NULL,
                        architecture VARCHAR(50),
                        accuracy FLOAT,
                        size_bytes BIGINT,
                        model_data BYTEA,
                        config TEXT,
                        is_pinned BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                db.commit()
                print(f"[INFO] Ensured trained_model table exists")
            except Exception as table_error:
                print(f"[WARNING] Could not create trained_model table: {table_error}")
            
            existing_models = db.query(TrainedModel).filter(
                TrainedModel.user_id == job.user_id,
                TrainedModel.is_pinned == False
            ).order_by(TrainedModel.created_at.desc()).all()
            
            if len(existing_models) >= 10:
                models_to_delete = existing_models[9:]
                for old_model in models_to_delete:
                    db.delete(old_model)
                    print(f"[INFO] Deleted old model {old_model.name} to maintain 10-model limit")
            
            # Use custom name from config if provided
            model_name = config.get("model_name") or f"{job.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get real model bytes from training
            if not model_bytes:
                raise ValueError("No model weights returned from training")
            
            actual_model_bytes = model_bytes
            actual_size_bytes = len(model_bytes)
            print(f"[INFO] Saving real model weights: {actual_size_bytes} bytes")
            
            trained_model = TrainedModel(
                user_id=job.user_id,
                job_id=job_id,
                name=model_name,
                architecture=job.model_type,
                accuracy=float(results["best_val_accuracy"]),
                size_bytes=actual_size_bytes,
                model_data=actual_model_bytes,
                config=json.dumps({
                    "training_results": {
                        "total_epochs": job.total_epochs,
                        "best_epoch": results["best_epoch"],
                        "final_train_loss": results["train_losses"][-1],
                        "final_val_loss": results["val_losses"][-1],
                        "final_train_acc": results["train_accuracies"][-1],
                        "final_val_acc": results["val_accuracies"][-1],
                        "best_val_acc": results["best_val_accuracy"],
                        "test_results": results.get("test_results")
                    }
                })
            )
            db.add(trained_model)
            db.commit()
            print(f"[INFO] Created trained model {model_name} with {results['best_val_accuracy']*100:.2f}% accuracy, size: {actual_size_bytes/1024/1024:.2f} MB")
        except Exception as e:
            print(f"[ERROR] Failed to create trained model: {str(e)}")
            import traceback
            traceback.print_exc()
        
        db.commit()
        
        print(f"[INFO] Training job {job_id} completed successfully")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Training job {job_id} failed: {str(e)}\n{error_details}")
        
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/train/cloud", response_model=Dict[str, Any])
async def start_cloud_training(
    request: CloudTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Start a cloud training job with a labeled dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == request.dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not dataset.files or len(dataset.files) == 0:
            raise HTTPException(status_code=400, detail="Dataset has no files")
        
        labels = set(f.label for f in dataset.files)
        if len(labels) < 2:
            raise HTTPException(status_code=400, detail="Dataset needs at least 2 different labels for classification")
        
        job_id = str(uuid.uuid4())
        config = {
            "model_type": request.model_type,
            "model_architecture": request.model_architecture,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "validation_split": request.validation_split,
            "model_name": request.model_name,
            "window_size": request.window_size,
            "num_classes": len(labels),
            "labels": list(labels),
            # Bayesian optimization settings
            "use_bayesian_optimization": request.use_bayesian_optimization,
            "bayesian_trials": request.bayesian_trials,
            "bayesian_epochs_per_trial": request.bayesian_epochs_per_trial,
            "bayesian_lr_min": request.bayesian_lr_min,
            "bayesian_lr_max": request.bayesian_lr_max,
            "bayesian_lr_scale": request.bayesian_lr_scale,
            "bayesian_batch_sizes": request.bayesian_batch_sizes,
            "bayesian_weight_decay_min": request.bayesian_weight_decay_min,
            "bayesian_weight_decay_max": request.bayesian_weight_decay_max,
            "bayesian_optimizers": request.bayesian_optimizers,
            "bayesian_exploration_rate": request.bayesian_exploration_rate,
            "bayesian_search_architecture": request.bayesian_search_architecture
        }
        
        job = TrainingJob(
            job_id=job_id,
            user_id=current_user.userId,
            dataset_id=request.dataset_id,
            test_dataset_id=request.test_dataset_id,
            model_type=request.model_type,
            training_mode="cloud",
            config=json.dumps(config),
            status="pending",
            total_epochs=request.epochs
        )
        db.add(job)
        db.commit()
        
        import os
        import sys
        db_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")
        
        print(f"[TRAINING-DEBUG] ========================================")
        print(f"[TRAINING-DEBUG] start_cloud_training endpoint called")
        print(f"[TRAINING-DEBUG] job_id: {job_id}")
        print(f"[TRAINING-DEBUG] dataset_id: {request.dataset_id}")
        print(f"[TRAINING-DEBUG] model_type: {request.model_type}")
        print(f"[TRAINING-DEBUG] use_bayesian_optimization: {request.use_bayesian_optimization}")
        print(f"[TRAINING-DEBUG] db_url: {db_url[:50]}...")
        print(f"[TRAINING-DEBUG] background_tasks object: {background_tasks}")
        print(f"[TRAINING-DEBUG] ========================================")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Adding background task...")
        sys.stdout.flush()
        background_tasks.add_task(run_cloud_training, job_id, db_url)
        print(f"[TRAINING-DEBUG] Background task ADDED for job {job_id}")
        print(f"[TRAINING-DEBUG] background_tasks.tasks count: {len(background_tasks.tasks) if hasattr(background_tasks, 'tasks') else 'N/A'}")
        sys.stdout.flush()
        
        return {
            "success": True,
            "message": f"Cloud training job started",
            "job": {
                "job_id": job_id,
                "dataset_id": request.dataset_id,
                "dataset_name": dataset.name,
                "model_type": request.model_type,
                "epochs": request.epochs,
                "num_classes": len(labels),
                "labels": list(labels),
                "status": "pending"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.get("/train/jobs", response_model=Dict[str, Any])
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all training jobs for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[JOBS] Starting jobs query for user {current_user.userId}, status={status}, limit={limit}")

        query = db.query(
            TrainingJob.job_id,
            TrainingJob.dataset_id,
            TrainingDataset.name.label("dataset_name"),
            TrainingJob.model_type,
            TrainingJob.training_mode,
            TrainingJob.status,
            TrainingJob.current_epoch,
            TrainingJob.total_epochs,
            TrainingJob.metrics,
            TrainingJob.best_metrics,
            TrainingJob.created_at,
            TrainingJob.started_at,
            TrainingJob.completed_at,
            TrainingJob.error_message,
        ).outerjoin(
            TrainingDataset,
            TrainingDataset.id == TrainingJob.dataset_id,
        ).filter(TrainingJob.user_id == current_user.userId)
        
        if status:
            query = query.filter(TrainingJob.status == status)
        
        jobs = query.order_by(TrainingJob.created_at.desc()).limit(limit).all()
        
        logger.info(f"[JOBS] Query completed, returned {len(jobs)} jobs")
        
        # Convert to dict efficiently
        job_list = []
        for j in jobs:
            progress = 0.0
            if j.total_epochs and j.total_epochs > 0:
                progress = (j.current_epoch / j.total_epochs) * 100

            metrics = {}
            if j.metrics:
                try:
                    metrics = json.loads(j.metrics)
                except Exception:
                    metrics = {}

            best_metrics = {}
            if j.best_metrics:
                try:
                    best_metrics = json.loads(j.best_metrics)
                except Exception:
                    best_metrics = {}

            job_list.append({
                "job_id": j.job_id,
                "dataset_id": j.dataset_id,
                "dataset_name": j.dataset_name,
                "model_type": j.model_type,
                "training_mode": j.training_mode,
                "status": j.status,
                "current_epoch": j.current_epoch,
                "total_epochs": j.total_epochs,
                "progress": progress,
                "metrics": metrics,
                "best_metrics": best_metrics,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "error_message": j.error_message,
            })
        
        return {
            "success": True,
            "jobs": job_list,
            "total": len(job_list),
            "limit": limit,
            "operation": "list_training_jobs",
            "status": "completed"
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[JOBS] Error listing jobs: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "limit": limit,
            "operation": "list_training_jobs",
            "status": "error",
            "error": str(e)
        }


@router.get("/train/jobs/{job_id}", response_model=Dict[str, Any])
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get details of a specific training job."""
    try:
        job = db.query(TrainingJob).filter(
            TrainingJob.job_id == job_id,
            TrainingJob.user_id == current_user.userId
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return {
            "success": True,
            "job": job.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.post("/train/jobs/{job_id}/cancel", response_model=StandardResponse)
async def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Cancel a running training job."""
    try:
        job = db.query(TrainingJob).filter(
            TrainingJob.job_id == job_id,
            TrainingJob.user_id == current_user.userId
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        if job.status not in ["pending", "running"]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job.status}'")
        
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Training job cancelled"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.delete("/train/jobs/{job_id}", response_model=StandardResponse)
async def delete_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a specific training job."""
    try:
        job = db.query(TrainingJob).filter(
            TrainingJob.job_id == job_id,
            TrainingJob.user_id == current_user.userId
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        db.delete(job)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Training job deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def list_trained_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all trained models for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[MODELS] Starting models query for user {current_user.userId}")
        
        # Query only necessary columns, exclude model_data to avoid loading large binaries
        logger.info("[MODELS] Executing models query")
        models = db.query(
            TrainedModel.id,
            TrainedModel.job_id,
            TrainedModel.name,
            TrainedModel.architecture,
            TrainedModel.accuracy,
            TrainedModel.size_bytes,
            TrainedModel.config,
            TrainedModel.is_pinned,
            TrainedModel.created_at
        ).filter(
            TrainedModel.user_id == current_user.userId
        ).order_by(TrainedModel.created_at.desc()).limit(50).all()
        
        logger.info(f"[MODELS] Query returned {len(models)} models")
        
        # Convert to dict manually since we're not loading the full model
        model_list = []
        for m in models:
            model_list.append({
                "id": m.id,
                "job_id": m.job_id,
                "name": m.name,
                "architecture": m.architecture,
                "accuracy": round(m.accuracy, 2) if m.accuracy else None,
                "size_mb": m.size_bytes / (1024 * 1024) if m.size_bytes else None,
                "config": json.loads(m.config) if m.config else {},
                "is_pinned": m.is_pinned,
                "created_at": m.created_at.isoformat() if m.created_at else None
            })
        
        logger.info(f"[MODELS] Successfully processed {len(model_list)} models")
        
        return {
            "success": True,
            "models": model_list,
            "total": len(model_list),
            "operation": "list_models",
            "status": "completed"
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[MODELS] Error listing models: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "models": [],
            "total": 0,
            "operation": "list_models",
            "status": "error",
            "error": str(e)
        }


@router.delete("/models/{model_id}", response_model=StandardResponse)
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a trained model."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        db.delete(model)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/models/{model_id}/pin", response_model=StandardResponse)
async def pin_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Pin a trained model to prevent auto-deletion."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_pinned = True
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model pinned successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to pin model: {str(e)}")


@router.post("/models/{model_id}/unpin", response_model=StandardResponse)
async def unpin_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Unpin a trained model to allow auto-deletion."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_pinned = False
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model unpinned successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to unpin model: {str(e)}")


class RenameModelRequest(BaseModel):
    """Request to rename a trained model."""
    name: str


@router.put("/models/{model_id}/rename", response_model=StandardResponse)
async def rename_model(
    model_id: int,
    request: RenameModelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Rename a trained model."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not request.name or len(request.name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Model name cannot be empty")
        
        model.name = request.name.strip()
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model renamed successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to rename model: {str(e)}")


@router.get("/models/{model_id}/download")
async def download_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Download a trained model."""
    from fastapi.responses import Response
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # If model_data exists, return it
        if model.model_data:
            return Response(
                content=model.model_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={model.name}.pth"
                }
            )
        else:
            # Return mock model file for now
            mock_model_data = b"Mock model weights data"
            return Response(
                content=mock_model_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={model.name}.pth"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


# ============================================================================
# DATASET DETAIL ENDPOINTS (catch-all routes must come last)
# ============================================================================

@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get dataset details including all files and labels."""
    try:
        dataset = db.query(TrainingDataset).options(
            selectinload(TrainingDataset.files).selectinload(DatasetFile.file)
        ).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId,
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        files_with_details = []
        for df in dataset.files or []:
            try:
                file_info = df.to_dict() if hasattr(df, "to_dict") else {
                    "id": getattr(df, "id", None),
                    "dataset_id": getattr(df, "dataset_id", None),
                    "file_id": getattr(df, "file_id", None),
                    "filename": None,
                    "label": getattr(df, "label", None),
                    "created_at": getattr(df, "created_at", None).isoformat() if getattr(df, "created_at", None) else None,
                }

                if df.file:
                    file_info["filename"] = df.file.filename
                    file_info["size"] = df.file.size
                    file_info["content_type"] = df.file.content_type
                    file_info["file_missing"] = False
                else:
                    file_info["file_missing"] = True

                files_with_details.append(file_info)
            except Exception as file_err:
                files_with_details.append({
                    "id": getattr(df, "id", None),
                    "dataset_id": getattr(df, "dataset_id", None),
                    "file_id": getattr(df, "file_id", None),
                    "filename": None,
                    "label": getattr(df, "label", None),
                    "created_at": getattr(df, "created_at", None).isoformat() if getattr(df, "created_at", None) else None,
                    "file_missing": True,
                    "error": str(file_err),
                })
        
        # Calculate label distribution
        label_counts = {}
        for df in dataset.files or []:
            try:
                if df.label is None:
                    continue
                label_counts[df.label] = label_counts.get(df.label, 0) + 1
            except Exception:
                continue
        
        result = dataset.to_dict()
        result["files"] = files_with_details
        result["label_distribution"] = label_counts
        
        return {
            "success": True,
            "dataset": result
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DATASET] Failed to get dataset {dataset_id} for user {getattr(current_user, 'userId', None)}: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")


@router.post("/{dataset_id}/files", response_model=Dict[str, Any])
async def add_files_to_dataset(
    dataset_id: int,
    request: AddFilesToDatasetRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add files with labels to a dataset."""
    import time
    start_time = time.time()
    logger.info(f"[ADD_FILES] Starting to add {len(request.files)} files to dataset {dataset_id} for user {current_user.userId}")
    
    try:
        # Query dataset
        query_start = time.time()
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        logger.info(f"[ADD_FILES] Dataset query took {(time.time() - query_start)*1000:.2f}ms")
        
        if not dataset:
            logger.warning(f"[ADD_FILES] Dataset {dataset_id} not found for user {current_user.userId}")
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        added_count = 0
        errors = []
        
        # Batch query only file IDs (avoid selecting large columns like file.content)
        file_ids = sorted({f.get("file_id") for f in request.files if f.get("file_id")})
        logger.info(f"[ADD_FILES] Querying {len(file_ids)} files in batch")
        
        query_start = time.time()
        existing_file_ids = set()
        if file_ids:
            rows = db.query(File.fileId).filter(
                File.fileId.in_(file_ids),
                File.userId == current_user.userId
            ).all()
            existing_file_ids = {r[0] for r in rows}
        logger.info(f"[ADD_FILES] Batch file-id query took {(time.time() - query_start)*1000:.2f}ms, found {len(existing_file_ids)} files")
        
        # Process files
        dataset_files_to_add = []
        for file_entry in request.files:
            file_id = file_entry.get("file_id")
            label = file_entry.get("label")
            
            if not file_id or not label:
                errors.append(f"Missing file_id or label in entry")
                continue
            
            # Check if file exists in our batch query results
            if file_id not in existing_file_ids:
                errors.append(f"File {file_id} not found")
                logger.warning(f"[ADD_FILES] File {file_id} not found for user {current_user.userId}")
                continue
            
            # Prepare dataset file entry
            dataset_file = DatasetFile(
                dataset_id=dataset_id,
                file_id=file_id,
                label=label
            )
            dataset_files_to_add.append(dataset_file)
            added_count += 1
        
        # Bulk add all dataset files
        if dataset_files_to_add:
            logger.info(f"[ADD_FILES] Adding {len(dataset_files_to_add)} dataset file entries")
            add_start = time.time()
            db.bulk_save_objects(dataset_files_to_add)
            logger.info(f"[ADD_FILES] Bulk add took {(time.time() - add_start)*1000:.2f}ms")
        
        # Commit transaction
        commit_start = time.time()
        db.commit()
        logger.info(f"[ADD_FILES] Commit took {(time.time() - commit_start)*1000:.2f}ms")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"[ADD_FILES] Successfully added {added_count} files to dataset {dataset_id} in {total_time:.2f}ms")
        
        return {
            "success": True,
            "message": f"Added {added_count} files to dataset",
            "added_count": added_count,
            "errors": errors if errors else None
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[ADD_FILES] Failed to add files to dataset {dataset_id}: {str(e)}\n{error_details}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add files: {str(e)}")


@router.delete("/{dataset_id}/files/{file_id}", response_model=StandardResponse)
async def remove_file_from_dataset(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Remove a file from a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_file = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id,
            DatasetFile.file_id == file_id
        ).first()
        
        if not dataset_file:
            raise HTTPException(status_code=404, detail="File not in dataset")
        
        db.delete(dataset_file)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="File removed from dataset"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")


@router.put("/{dataset_id}/files/{file_id}/label", response_model=StandardResponse)
async def update_file_label(
    dataset_id: int,
    file_id: int,
    request: UpdateFileLabelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update a file's label in a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_file = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id,
            DatasetFile.file_id == file_id
        ).first()
        
        if not dataset_file:
            raise HTTPException(status_code=404, detail="File not in dataset")
        
        dataset_file.label = request.label
        db.commit()
        
        return StandardResponse(
            success=True,
            message=f"Label updated to '{request.label}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update label: {str(e)}")


@router.delete("/{dataset_id}", response_model=StandardResponse)
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        db.delete(dataset)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Dataset deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
