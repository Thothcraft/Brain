"""Activity Feed Endpoints.

This module provides endpoints for tracking and retrieving user activity,
including device events, file uploads, and system events.
"""

from fastapi import APIRouter, Depends, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, func

from ..db import get_db, Device, File, TrainedModel, Query as QueryModel
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
    - Model registrations
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
        
        # Get recent registered models
        models = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId,
            TrainedModel.created_at > cutoff_time
        ).order_by(desc(TrainedModel.created_at)).limit(limit).all()
        
        for model in models:
            activities.append({
                "type": "model",
                "action": "created",
                "title": "Model saved",
                "description": f"{model.name} ({model.accuracy:.1f}% accuracy)" if model.accuracy else model.name,
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
    
    Returns counts and stats for devices, files, and models.
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
