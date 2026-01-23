"""Activity Feed Endpoints.

This module provides endpoints for tracking and retrieving user activity,
including device events, training jobs, file uploads, and system events.
"""

from fastapi import APIRouter, Depends, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, func

from ..db import get_db, Device, File, TrainingJob, TrainedModel, Query as QueryModel
from ..auth import get_current_user

router = APIRouter(prefix="/activity", tags=["activity"])


@router.get("/recent", response_model=Dict[str, Any])
async def get_recent_activity(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of activities to return"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get recent activity feed for the current user.
    
    Returns a chronologically sorted list of recent events including:
    - Device connections/disconnections
    - File uploads
    - Training job status changes
    - AI queries
    """
    activities = []
    cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    
    try:
        # Get recent device activity - filter in DB, not Python
        devices = db.query(Device).filter(
            Device.userId == current_user.userId,
            Device.last_seen > cutoff_time
        ).limit(limit).all()
        
        for device in devices:
            activities.append({
                "type": "device",
                "action": "online" if device.online else "offline",
                "title": f"Device {'connected' if device.online else 'disconnected'}",
                "description": f"{device.device_name or device.device_uuid}",
                "timestamp": device.last_seen.isoformat(),
                "icon": "wifi" if device.online else "wifi-off",
                "color": "green" if device.online else "slate"
            })
        
        # Get recent file uploads
        files = db.query(File).filter(
            File.userId == current_user.userId,
            File.uploaded_at > cutoff_time
        ).order_by(desc(File.uploaded_at)).limit(limit).all()
        
        for file in files:
            # Extract original filename
            parts = file.filename.split('_', 3)
            original_filename = parts[-1] if len(parts) >= 4 else file.filename
            
            activities.append({
                "type": "file",
                "action": "upload",
                "title": "File uploaded",
                "description": original_filename,
                "timestamp": file.uploaded_at.isoformat(),
                "icon": "file",
                "color": "blue",
                "metadata": {
                    "size": file.size,
                    "content_type": file.content_type
                }
            })
        
        # Get recent training jobs
        jobs = db.query(TrainingJob).filter(
            TrainingJob.user_id == current_user.userId,
            or_(
                TrainingJob.created_at > cutoff_time,
                TrainingJob.completed_at > cutoff_time
            )
        ).order_by(desc(TrainingJob.created_at)).limit(limit).all()
        
        for job in jobs:
            if job.status == "completed" and job.completed_at:
                activities.append({
                    "type": "training",
                    "action": "completed",
                    "title": "Training completed",
                    "description": f"{job.model_type} model trained successfully",
                    "timestamp": job.completed_at.isoformat(),
                    "icon": "check-circle",
                    "color": "green",
                    "metadata": {
                        "job_id": job.job_id,
                        "model_type": job.model_type
                    }
                })
            elif job.status == "failed" and job.completed_at:
                activities.append({
                    "type": "training",
                    "action": "failed",
                    "title": "Training failed",
                    "description": f"{job.model_type} training encountered an error",
                    "timestamp": job.completed_at.isoformat(),
                    "icon": "x-circle",
                    "color": "red",
                    "metadata": {
                        "job_id": job.job_id,
                        "error": job.error_message
                    }
                })
            elif job.status == "running":
                activities.append({
                    "type": "training",
                    "action": "running",
                    "title": "Training in progress",
                    "description": f"{job.model_type} - Epoch {job.current_epoch}/{job.total_epochs}",
                    "timestamp": (job.started_at or job.created_at).isoformat(),
                    "icon": "loader",
                    "color": "blue",
                    "metadata": {
                        "job_id": job.job_id,
                        "progress": (job.current_epoch / job.total_epochs * 100) if job.total_epochs else 0
                    }
                })
            elif job.created_at > cutoff_time:
                activities.append({
                    "type": "training",
                    "action": "started",
                    "title": "Training job created",
                    "description": f"{job.model_type} model training queued",
                    "timestamp": job.created_at.isoformat(),
                    "icon": "brain",
                    "color": "purple",
                    "metadata": {
                        "job_id": job.job_id
                    }
                })
        
        # Get recent trained models
        models = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId,
            TrainedModel.created_at > cutoff_time
        ).order_by(desc(TrainedModel.created_at)).limit(limit).all()
        
        for model in models:
            activities.append({
                "type": "model",
                "action": "created",
                "title": "Model saved",
                "description": f"{model.name} ({model.accuracy*100:.1f}% accuracy)" if model.accuracy else model.name,
                "timestamp": model.created_at.isoformat(),
                "icon": "brain",
                "color": "purple",
                "metadata": {
                    "model_id": model.id,
                    "accuracy": model.accuracy,
                    "size_mb": round(model.size_bytes / 1024 / 1024, 2) if model.size_bytes else None
                }
            })
        
        # Get recent AI queries
        queries = db.query(QueryModel).filter(
            QueryModel.userId == current_user.userId,
            QueryModel.created_at > cutoff_time
        ).order_by(desc(QueryModel.created_at)).limit(limit).all()
        
        for query in queries:
            activities.append({
                "type": "query",
                "action": "asked",
                "title": "AI Query",
                "description": query.query_text[:100] + "..." if len(query.query_text) > 100 else query.query_text,
                "timestamp": query.created_at.isoformat(),
                "icon": "message-circle",
                "color": "indigo"
            })
        
        # Sort all activities by timestamp (newest first)
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limit to requested number
        activities = activities[:limit]
        
        return {
            "success": True,
            "activities": activities,
            "count": len(activities)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "activities": [],
            "count": 0,
            "error": str(e)
        }


@router.get("/stats", response_model=Dict[str, Any])
async def get_activity_stats(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get activity statistics for dashboard display.
    
    Returns counts and stats for devices, files, training jobs, and models.
    """
    try:
        # Device stats
        devices = db.query(Device).filter(Device.userId == current_user.userId).all()
        total_devices = len(devices)
        online_devices = sum(1 for d in devices if d.online)
        
        # File stats
        total_files = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like("file_%")
        ).count()
        
        # Training stats
        total_jobs = db.query(TrainingJob).filter(
            TrainingJob.user_id == current_user.userId
        ).count()
        
        active_jobs = db.query(TrainingJob).filter(
            TrainingJob.user_id == current_user.userId,
            TrainingJob.status.in_(["pending", "running", "optimizing"])
        ).count()
        
        completed_jobs = db.query(TrainingJob).filter(
            TrainingJob.user_id == current_user.userId,
            TrainingJob.status == "completed"
        ).count()
        
        # Model stats
        total_models = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId
        ).count()
        
        best_model = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId,
            TrainedModel.accuracy.isnot(None)
        ).order_by(desc(TrainedModel.accuracy)).first()
        
        best_accuracy = best_model.accuracy if best_model else None
        
        return {
            "success": True,
            "stats": {
                "devices": {
                    "total": total_devices,
                    "online": online_devices,
                    "offline": total_devices - online_devices
                },
                "files": {
                    "total": total_files
                },
                "training": {
                    "total_jobs": total_jobs,
                    "active_jobs": active_jobs,
                    "completed_jobs": completed_jobs
                },
                "models": {
                    "total": total_models,
                    "best_accuracy": best_accuracy
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "stats": {},
            "error": str(e)
        }
