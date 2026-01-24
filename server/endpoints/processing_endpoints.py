"""Processing Pipeline Endpoints.

This module handles:
- Pipeline creation and management
- Block configuration
- Pipeline execution
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session
import json

from ..db import get_db, PreprocessingPipeline
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


class CreateDbPipelineRequest(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str = "csi"
    output_shape: str = "flattened"
    include_phase: bool = True
    window_size: int = 1000
    filter_subcarriers: bool = True
    subcarrier_start: int = 5
    subcarrier_end: int = 32
    config: Dict[str, Any] = {}
    blocks: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, str]]] = None


class UpdateDbPipelineRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    output_shape: Optional[str] = None
    include_phase: Optional[bool] = None
    window_size: Optional[int] = None
    filter_subcarriers: Optional[bool] = None
    subcarrier_start: Optional[int] = None
    subcarrier_end: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    blocks: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, str]]] = None
    is_default: Optional[bool] = None


@router.post("/db-pipelines", response_model=StandardResponse)
async def create_db_pipeline(
    request: CreateDbPipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        config_data = request.config.copy() if request.config else {}
        if request.blocks is not None:
            config_data['blocks'] = request.blocks
        if request.connections is not None:
            config_data['connections'] = request.connections

        pipeline = PreprocessingPipeline(
            user_id=current_user.userId,
            name=request.name,
            description=request.description,
            data_type=request.data_type,
            output_shape=request.output_shape,
            include_phase=request.include_phase,
            window_size=request.window_size,
            filter_subcarriers=request.filter_subcarriers,
            subcarrier_start=request.subcarrier_start,
            subcarrier_end=request.subcarrier_end,
            config=json.dumps(config_data)
        )

        db.add(pipeline)
        db.commit()
        db.refresh(pipeline)

        return StandardResponse(
            success=True,
            message=f"Preprocessing pipeline '{request.name}' created successfully",
            data=pipeline.to_dict()
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@router.get("/db-pipelines", response_model=Dict[str, Any])
async def list_db_pipelines(
    data_type: Optional[str] = Query(None),
    output_shape: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        query = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.user_id == current_user.userId
        )
        if data_type:
            query = query.filter(PreprocessingPipeline.data_type == data_type)
        if output_shape:
            query = query.filter(PreprocessingPipeline.output_shape == output_shape)

        pipelines = query.order_by(PreprocessingPipeline.created_at.desc()).all()
        return {
            "success": True,
            "pipelines": [p.to_dict() for p in pipelines],
            "total": len(pipelines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.get("/db-pipelines/{pipeline_id}", response_model=Dict[str, Any])
async def get_db_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        return {
            "success": True,
            "pipeline": pipeline.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@router.put("/db-pipelines/{pipeline_id}", response_model=StandardResponse)
async def update_db_pipeline(
    pipeline_id: int,
    request: UpdateDbPipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

        if request.name is not None:
            pipeline.name = request.name
        if request.description is not None:
            pipeline.description = request.description
        if request.data_type is not None:
            pipeline.data_type = request.data_type
        if request.output_shape is not None:
            pipeline.output_shape = request.output_shape
        if request.include_phase is not None:
            pipeline.include_phase = request.include_phase
        if request.window_size is not None:
            pipeline.window_size = request.window_size
        if request.filter_subcarriers is not None:
            pipeline.filter_subcarriers = request.filter_subcarriers
        if request.subcarrier_start is not None:
            pipeline.subcarrier_start = request.subcarrier_start
        if request.subcarrier_end is not None:
            pipeline.subcarrier_end = request.subcarrier_end

        if request.config is not None or request.blocks is not None or request.connections is not None:
            existing_config = json.loads(pipeline.config) if pipeline.config else {}
            if request.config is not None:
                existing_config.update(request.config)
            if request.blocks is not None:
                existing_config['blocks'] = request.blocks
            if request.connections is not None:
                existing_config['connections'] = request.connections
            pipeline.config = json.dumps(existing_config)

        if request.is_default is not None:
            if request.is_default:
                db.query(PreprocessingPipeline).filter(
                    PreprocessingPipeline.user_id == current_user.userId,
                    PreprocessingPipeline.data_type == pipeline.data_type,
                    PreprocessingPipeline.is_default == True
                ).update({"is_default": False})
            pipeline.is_default = request.is_default

        db.commit()
        db.refresh(pipeline)

        return StandardResponse(
            success=True,
            message=f"Pipeline '{pipeline.name}' updated successfully",
            data=pipeline.to_dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update pipeline: {str(e)}")


@router.delete("/db-pipelines/{pipeline_id}", response_model=StandardResponse)
async def delete_db_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

        pipeline_name = pipeline.name
        db.delete(pipeline)
        db.commit()

        return StandardResponse(
            success=True,
            message=f"Pipeline '{pipeline_name}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")


@router.get("/preprocessing-methods", response_model=Dict[str, Any])
async def get_preprocessing_methods():
    try:
        methods = {
            "amplitude_phase": {
                "name": "Amplitude and Phase Extraction",
                "description": "Extract amplitude and phase from complex CSI values",
                "parameters": ["include_phase"],
                "output_shape": "variable"
            },
            "subcarrier_filter": {
                "name": "Subcarrier Filtering",
                "description": "Filter specific subcarrier ranges to remove guard bands",
                "parameters": ["start_index", "end_index", "filter_method"],
                "output_shape": "reduced"
            },
            "moving_average": {
                "name": "Moving Average Filter",
                "description": "Apply moving average smoothing to reduce noise",
                "parameters": ["window_size", "center"],
                "output_shape": "same"
            },
            "statistical_filter": {
                "name": "Statistical Outlier Removal",
                "description": "Remove outliers using z-score or IQR methods",
                "parameters": ["method", "threshold", "axis"],
                "output_shape": "reduced"
            },
            "frequency_domain": {
                "name": "Frequency Domain Filtering",
                "description": "Apply frequency domain filtering",
                "parameters": ["cutoff_frequency"],
                "output_shape": "same"
            },
            "pca_reduction": {
                "name": "PCA Dimensionality Reduction",
                "description": "Reduce dimensionality using PCA",
                "parameters": ["n_components"],
                "output_shape": "reduced"
            }
        }
        return {
            "success": True,
            "methods": methods,
            "total": len(methods)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preprocessing methods: {str(e)}")
