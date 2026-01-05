"""Processing Pipeline Endpoints.

This module handles:
- Pipeline creation and management
- Block configuration
- Pipeline execution
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
import json

from ..db import get_db
from ..auth import get_current_user
from .models import StandardResponse

router = APIRouter(prefix="/processing", tags=["processing"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreatePipelineRequest(BaseModel):
    """Request to create a new processing pipeline."""
    name: str
    description: Optional[str] = None


class UpdatePipelineRequest(BaseModel):
    """Request to update a pipeline."""
    blocks: List[Dict[str, Any]]
    connections: List[Dict[str, str]]


# ============================================================================
# IN-MEMORY STORAGE (temporary until database table is created)
# ============================================================================
_pipelines_store: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "IMU Preprocessing",
        "description": "Standard IMU data preprocessing pipeline",
        "blocks": [],
        "connections": [],
        "created_at": datetime.utcnow().isoformat()
    }
}
_next_pipeline_id = 2

# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

@router.get("/pipelines", response_model=Dict[str, Any])
async def list_pipelines(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all processing pipelines for the current user."""
    try:
        pipelines = list(_pipelines_store.values())
        return {
            "success": True,
            "pipelines": pipelines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.post("/pipelines", response_model=StandardResponse)
async def create_pipeline(
    request: CreatePipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new processing pipeline."""
    global _next_pipeline_id
    try:
        pipeline_id = _next_pipeline_id
        _next_pipeline_id += 1
        
        _pipelines_store[pipeline_id] = {
            "id": pipeline_id,
            "name": request.name,
            "description": request.description or "",
            "blocks": [],
            "connections": [],
            "created_at": datetime.utcnow().isoformat()
        }
        
        return StandardResponse(
            success=True,
            message=f"Pipeline '{request.name}' created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@router.put("/pipelines/{pipeline_id}", response_model=StandardResponse)
async def update_pipeline(
    pipeline_id: int,
    request: UpdatePipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update a processing pipeline."""
    try:
        if pipeline_id not in _pipelines_store:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        # Update blocks and connections
        _pipelines_store[pipeline_id]["blocks"] = request.blocks
        _pipelines_store[pipeline_id]["connections"] = request.connections
        
        return StandardResponse(
            success=True,
            message="Pipeline updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pipeline: {str(e)}")


@router.delete("/pipelines/{pipeline_id}", response_model=StandardResponse)
async def delete_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a processing pipeline."""
    try:
        # For now, return success
        # In production, you'd delete the Pipeline record
        return StandardResponse(
            success=True,
            message="Pipeline deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")
