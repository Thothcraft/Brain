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
    model_type: str = "cnn"  # cnn, lstm, transformer, linear
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    model_name: Optional[str] = None


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
            
            # Check if already in dataset
            existing = db.query(DatasetFile).filter(
                DatasetFile.dataset_id == dataset_id,
                DatasetFile.file_id == file_id
            ).first()
            
            if existing:
                # Update label
                existing.label = label
            else:
                # Add new
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


# ============================================================================
# CLOUD TRAINING ENDPOINTS
# ============================================================================

async def run_cloud_training(job_id: str, db_url: str):
    """Simulate cloud training in background."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return
        
        config = json.loads(job.config) if job.config else {}
        epochs = job.total_epochs or 10
        
        # Update status to running
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        metrics = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Refresh job to check for cancellation
            db.refresh(job)
            if job.status == "cancelled":
                break
            
            # Simulate training time (2 seconds per epoch)
            await asyncio.sleep(2)
            
            # Generate realistic metrics
            progress = epoch / epochs
            base_loss = 1.0 - progress * 0.7
            base_acc = 0.5 + progress * 0.4
            
            loss = max(0.05, base_loss + random.uniform(-0.1, 0.1))
            accuracy = min(0.98, base_acc + random.uniform(-0.05, 0.05))
            val_loss = max(0.08, base_loss + 0.1 + random.uniform(-0.1, 0.1))
            val_accuracy = min(0.95, base_acc - 0.05 + random.uniform(-0.05, 0.05))
            
            metrics["loss"].append(round(loss, 4))
            metrics["accuracy"].append(round(accuracy, 4))
            metrics["val_loss"].append(round(val_loss, 4))
            metrics["val_accuracy"].append(round(val_accuracy, 4))
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch
            
            # Update job progress
            job.current_epoch = epoch
            job.metrics = json.dumps(metrics)
            job.best_metrics = json.dumps({
                "val_accuracy": round(best_val_acc, 4),
                "best_epoch": best_epoch
            })
            db.commit()
        
        # Complete training
        if job.status == "running":
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.model_path = f"/models/cloud/{job_id}/model.h5"
            
            # Create trained model record
            model_name = config.get("model_name") or f"{job.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            trained_model = TrainedModel(
                user_id=job.user_id,
                job_id=job_id,
                name=model_name,
                architecture=job.model_type,
                accuracy=int(best_val_acc * 10000),  # Store as int (e.g., 0.95 -> 9500)
                size_bytes=random.randint(1000000, 50000000),  # 1-50 MB
                config=job.config
            )
            db.add(trained_model)
            db.commit()
            
    except Exception as e:
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
        # Verify dataset exists and has files
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == request.dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not dataset.files or len(dataset.files) == 0:
            raise HTTPException(status_code=400, detail="Dataset has no files")
        
        # Check for unique labels
        labels = set(f.label for f in dataset.files)
        if len(labels) < 2:
            raise HTTPException(status_code=400, detail="Dataset needs at least 2 different labels for classification")
        
        # Create job
        job_id = str(uuid.uuid4())
        config = {
            "model_type": request.model_type,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "validation_split": request.validation_split,
            "model_name": request.model_name,
            "num_classes": len(labels),
            "labels": list(labels)
        }
        
        job = TrainingJob(
            job_id=job_id,
            user_id=current_user.userId,
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            training_mode="cloud",
            config=json.dumps(config),
            status="pending",
            total_epochs=request.epochs
        )
        db.add(job)
        db.commit()
        
        # Get database URL for background task
        import os
        db_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")
        
        # Start training in background
        background_tasks.add_task(run_cloud_training, job_id, db_url)
        
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


@router.get("/models", response_model=Dict[str, Any])
async def list_trained_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all trained models for the current user."""
    try:
        models = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId
        ).order_by(TrainedModel.created_at.desc()).all()
        
        return {
            "success": True,
            "models": [m.to_dict() for m in models],
            "total": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


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
