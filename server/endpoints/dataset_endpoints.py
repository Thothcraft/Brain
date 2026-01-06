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

from ..db import get_db, TrainingDataset, DatasetFile, TrainingJob, TrainedModel, File
from ..auth import get_current_user
from .models import StandardResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


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
    use_bayesian_optimization: bool = False
    bayesian_trials: int = 20


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
        datasets = db.query(TrainingDataset).filter(
            TrainingDataset.user_id == current_user.userId
        ).order_by(TrainingDataset.created_at.desc()).all()
        
        return {
            "success": True,
            "datasets": [d.to_dict() for d in datasets],
            "total": len(datasets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


# ============================================================================
# CLOUD TRAINING ENDPOINTS (must be before /{dataset_id} catch-all)
# ============================================================================

async def run_cloud_training(job_id: str, db_url: str):
    """Run real PyTorch training for IMU model in background."""
    print(f"[INFO] Background task started for job {job_id}")
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from server.ml_training import run_full_training
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            print(f"[ERROR] Job {job_id} not found")
            return
        
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        config = json.loads(job.config) if job.config else {}
        
        # Define update callback for progress tracking
        async def update_callback(status, epoch, total, metrics):
            nonlocal job, db
            try:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = status
                    if epoch > 0:
                        job.current_epoch = epoch
                    db.commit()
            except Exception as e:
                print(f"[WARNING] Failed to update job status: {e}")
        
        # Run real PyTorch training
        print(f"[INFO] Starting real PyTorch training for job {job_id}")
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
            "num_classes": len(labels),
            "labels": list(labels),
            "use_bayesian_optimization": request.use_bayesian_optimization,
            "bayesian_trials": request.bayesian_trials
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
        db_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")
        
        print(f"[INFO] Starting background training task for job {job_id}")
        background_tasks.add_task(run_cloud_training, job_id, db_url)
        print(f"[INFO] Background task added for job {job_id}")
        
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
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all training jobs for the current user."""
    try:
        query = db.query(TrainingJob).filter(TrainingJob.user_id == current_user.userId)
        
        if status:
            query = query.filter(TrainingJob.status == status)
        
        jobs = query.order_by(TrainingJob.created_at.desc()).all()
        
        return {
            "success": True,
            "jobs": [j.to_dict() for j in jobs],
            "total": len(jobs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


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
        except Exception as table_error:
            print(f"[WARNING] Could not ensure trained_model table exists: {table_error}")
        
        models = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId
        ).order_by(TrainedModel.created_at.desc()).all()
        
        return {
            "success": True,
            "models": [m.to_dict() for m in models],
            "total": len(models)
        }
    except Exception as e:
        print(f"[ERROR] Failed to list models: {str(e)}")
        return {
            "success": True,
            "models": [],
            "total": 0
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
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get files with their details
        files_with_details = []
        for df in dataset.files:
            file_info = df.to_dict()
            if df.file:
                file_info["size"] = df.file.size
                file_info["content_type"] = df.file.content_type
            files_with_details.append(file_info)
        
        # Calculate label distribution
        label_counts = {}
        for df in dataset.files:
            label_counts[df.label] = label_counts.get(df.label, 0) + 1
        
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
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")


@router.post("/{dataset_id}/files", response_model=Dict[str, Any])
async def add_files_to_dataset(
    dataset_id: int,
    request: AddFilesToDatasetRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add files with labels to a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        added_count = 0
        errors = []
        
        for file_entry in request.files:
            file_id = file_entry.get("file_id")
            label = file_entry.get("label")
            
            if not file_id or not label:
                errors.append(f"Missing file_id or label in entry")
                continue
            
            # Verify file exists and belongs to user
            file = db.query(File).filter(
                File.fileId == file_id,
                File.userId == current_user.userId
            ).first()
            
            if not file:
                errors.append(f"File {file_id} not found")
                continue
            
            # Always add new entry - allow same file with different labels
            dataset_file = DatasetFile(
                dataset_id=dataset_id,
                file_id=file_id,
                label=label
            )
            db.add(dataset_file)
            
            added_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Added/updated {added_count} files to dataset",
            "added_count": added_count,
            "errors": errors if errors else None
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to add files: {str(e)}\n{error_details}")
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
