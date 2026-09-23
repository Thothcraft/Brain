"""Dataset and Model Registry Endpoints.

This module handles:
- Dataset creation and management
- File labeling for datasets
- Model artifact registry, upload, and deployment
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Body, Response, UploadFile, File as FormFile, Form
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
import io
import uuid
import json
import asyncio
import random
import logging
import zipfile
import hashlib
import tempfile
from pathlib import Path

from sqlalchemy.orm import selectinload, load_only

from ..db import get_db, TrainingDataset, DatasetFile, TrainedModel, File, Device, DeviceDeployment, DeviceCommand
from ..auth import get_current_user
from ..entitlements import check_download_allowed, has_entitlement
from .models import StandardResponse
from ..model_contract import validate_torchscript, ModelContractError

router = APIRouter(prefix="/datasets", tags=["datasets"])

logger = logging.getLogger(__name__)


def _original_filename(stored_filename: str) -> str:
    parts = stored_filename.split('_', 3)
    return parts[-1] if len(parts) >= 4 else stored_filename


def _file_content(file_record: File) -> Optional[bytes]:
    if file_record.storage_path:
        try:
            from server.utils.supabase_storage import download_file_sync
            path_parts = file_record.storage_path.split('/', 1)
            if len(path_parts) == 2:
                bucket, path = path_parts
                success, content = download_file_sync(bucket, path)
                if success and content:
                    return content
        except Exception as exc:
            logger.warning("Failed to read file %s from storage: %s", file_record.fileId, exc)
    return file_record.content


def _deployment_requests_allowed_for_device(device) -> bool:
    try:
        hw_info = device.hardware_info
        if isinstance(hw_info, str):
            hw_info = json.loads(hw_info)
        if isinstance(hw_info, dict) and "deployment_requests_allowed" in hw_info:
            return bool(hw_info.get("deployment_requests_allowed", True))
    except Exception:
        logger.debug("Unable to read deployment flag from hardware_info", exc_info=True)
    return True


def _require_ai_model_plan(user) -> None:
    """Private model deployment is a Research-tier capability."""
    if not has_entitlement(user, "custom_models"):
        raise HTTPException(
            status_code=403,
            detail="Private AI model deployment requires the Research plan",
        )


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
            dataset_files = db.query(DatasetFile.label).filter(
                DatasetFile.dataset_id == d.id
            ).all()
            label_counts = {}
            for row in dataset_files:
                label = row[0]
                if label:
                    label_counts[label] = label_counts.get(label, 0) + 1
            dataset_list.append({
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "file_count": sum(label_counts.values()),
                "labels": sorted(label_counts.keys()),
                "label_counts": label_counts,
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
# FILE LINE COUNT ENDPOINT
# ============================================================================

@router.get("/files/{file_id}/line-count", response_model=Dict[str, Any])
async def get_file_line_count(
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get the number of data lines in a file (excluding header)."""
    from sqlalchemy import text
    try:
        # First, check file exists and get metadata without loading content
        file_meta = db.query(File.fileId, File.filename, File.content_type, File.size).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_meta:
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = file_meta.filename
        content_type = file_meta.content_type
        file_size = file_meta.size
        
        # For large files, estimate line count based on file size
        # Average CSI line is ~500 bytes, IMU line is ~100 bytes
        filename_lower = (filename or "").lower()
        is_csi = 'csi' in filename_lower or (content_type and 'csv' in content_type.lower())
        
        if file_size and file_size > 10_000_000:  # > 10MB, estimate instead
            avg_line_size = 500 if is_csi else 100
            estimated_lines = file_size // avg_line_size
            return {
                "success": True,
                "file_id": file_id,
                "filename": filename,
                "total_lines": estimated_lines,
                "data_lines": estimated_lines - 1 if is_csi else estimated_lines,
                "is_csi": is_csi,
                "estimated": True,
                "note": f"Estimated from file size ({file_size:,} bytes)"
            }
        
        # For smaller files, count lines efficiently using raw SQL to avoid ORM overhead
        try:
            # Use a raw SQL query with timeout to count newlines directly in the database
            result = db.execute(text("""
                SELECT 
                    LENGTH(content) - LENGTH(REPLACE(CONVERT_FROM(content, 'UTF8'), E'\\n', '')) + 1 as line_count
                FROM file 
                WHERE file_id = :file_id AND user_id = :user_id
            """), {"file_id": file_id, "user_id": current_user.userId}).fetchone()
            
            if result and result[0]:
                total_lines = result[0]
                data_lines = total_lines - 1 if is_csi else total_lines  # Subtract header for CSI
                return {
                    "success": True,
                    "file_id": file_id,
                    "filename": filename,
                    "total_lines": total_lines,
                    "data_lines": max(0, data_lines),
                    "is_csi": is_csi
                }
        except Exception as sql_err:
            # If raw SQL fails, fall back to estimation
            if file_size:
                avg_line_size = 500 if is_csi else 100
                estimated_lines = file_size // avg_line_size
                return {
                    "success": True,
                    "file_id": file_id,
                    "filename": filename,
                    "total_lines": estimated_lines,
                    "data_lines": estimated_lines - 1 if is_csi else estimated_lines,
                    "is_csi": is_csi,
                    "estimated": True,
                    "note": f"Estimated (query failed: {str(sql_err)[:50]})"
                }
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "total_lines": 0,
            "data_lines": 0,
            "error": "Could not determine line count"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get line count: {str(e)}")


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
                "processor_type": m.processor_type,
                "sensor": m.sensor,
                "task": m.task,
                "visibility": m.visibility,
                "registry_name": m.registry_name,
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
        logger.error(f"[MODELS] Error listing models: {str(e)}")
        return {
            "success": False,
            "models": [],
            "total": 0,
            "operation": "list_models",
            "status": "error",
            "error": str(e)
        }


@router.post("/models/upload", response_model=Dict[str, Any])
async def upload_torchscript_model(
    model: UploadFile = FormFile(...),
    metadata: str = Form(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Validate and store a self-contained user TorchScript classifier."""
    suffix = Path(model.filename or '').suffix.lower()
    if suffix not in {'.pt', '.pth'}:
        raise HTTPException(status_code=422, detail='Model filename must end in .pt or .pth')
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f'Invalid metadata JSON: {exc}') from exc
    raw = await model.read()
    if not raw:
        raise HTTPException(status_code=422, detail='Model artifact is empty')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(raw)
            temporary_path = Path(temporary.name)
        validated = validate_torchscript(temporary_path, parsed_metadata)
    except ModelContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    record = TrainedModel(
        user_id=current_user.userId,
        job_id=None,
        name=validated['name'],
        architecture='torchscript',
        accuracy=None,
        size_bytes=len(raw),
        model_data=raw,
        config=json.dumps({'source': 'user-upload', 'model_hash': digest, 'metadata': validated}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {'success': True, 'model': record.to_dict(), 'validation': {'status': 'valid', 'output_dimension': validated['output_dimension'], 'model_hash': digest}}


# ── model registry (processor ecosystem) ────────────────────────────────────

PROCESSOR_TYPES = {"rule", "classical", "torchscript", "fusion"}
VISIBILITIES = {"private", "community", "official"}


@router.get("/models/registry", response_model=Dict[str, Any])
async def list_model_registry(
    sensor: Optional[str] = None,
    task: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Public processor catalog: official + community models, plus the
    caller's own private models. Filterable by sensor modality and task."""
    query = db.query(TrainedModel).filter(
        (TrainedModel.visibility.in_(["official", "community"]))
        | (TrainedModel.user_id == current_user.userId)
    )
    if sensor:
        query = query.filter(
            (TrainedModel.sensor == sensor) | (TrainedModel.sensor == "any"))
    if task:
        query = query.filter(TrainedModel.task == task)
    models = query.order_by(TrainedModel.created_at.desc()).limit(200).all()
    return {"success": True, "models": [m.to_dict() for m in models],
            "total": len(models)}


@router.get("/models/registry/{registry_name:path}", response_model=Dict[str, Any])
async def resolve_registry_model(
    registry_name: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Resolve a registry name ('thothcraft/radar-occupancy-v2') to a model."""
    model = db.query(TrainedModel).filter(
        TrainedModel.registry_name == registry_name,
        (TrainedModel.visibility.in_(["official", "community"]))
        | (TrainedModel.user_id == current_user.userId),
    ).order_by(TrainedModel.created_at.desc()).first()
    if not model:
        raise HTTPException(status_code=404, detail="Registry model not found")
    return {"success": True, "model": model.to_dict()}


@router.post("/models/rule", response_model=Dict[str, Any], status_code=201)
async def create_rule_model(
    body: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Register a config-only rule processor — no artifact upload.

    Body: {name, rules: [{when, label, confidence?}], else?, params?,
           sensor?, task?, registry_name?, config_schema?}
    """
    name = str(body.get("name") or "").strip()
    rules = body.get("rules")
    if not name or not isinstance(rules, list) or not rules:
        raise HTTPException(
            status_code=422,
            detail="rule models require a name and a non-empty rules list")
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("when") or not rule.get("label"):
            raise HTTPException(
                status_code=422,
                detail="each rule needs 'when' (expression) and 'label'")
    else_label = body.get("else", "unknown")
    class_names = [r.get("label") for r in rules if r.get("label")]
    if else_label and else_label not in class_names:
        class_names.append(else_label)
    config = {
        "processor": "rule",
        "rules": rules,
        "rule_type": body.get("rule_type") or ("face_detection" if body.get("sensor") == "camera" else "threshold"),
        "else": else_label,
        "params": body.get("params") or {},
        "config_schema": body.get("config_schema") or {},
        "actuator": body.get("actuator"),
        "class_names": class_names,
    }
    record = TrainedModel(
        user_id=current_user.userId,
        name=name,
        architecture="rule",
        processor_type="rule",
        sensor=body.get("sensor"),
        task=body.get("task"),
        visibility="private",
        registry_name=body.get("registry_name"),
        size_bytes=0,
        model_data=None,
        config=json.dumps(config),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"success": True, "model": record.to_dict()}


@router.post("/models/{model_id}/publish", response_model=Dict[str, Any])
async def publish_model(
    model_id: int,
    visibility: str = "community",
    registry_name: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Publish a model to the community registry (or set official — admin only)."""
    model = db.query(TrainedModel).filter(
        TrainedModel.id == model_id,
        TrainedModel.user_id == current_user.userId,
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if visibility not in VISIBILITIES:
        raise HTTPException(status_code=422, detail=f"visibility must be one of {sorted(VISIBILITIES)}")
    if visibility == "official" and getattr(current_user, "role", None) != 1:
        raise HTTPException(status_code=403, detail="Only admins can publish official models")
    if registry_name:
        clash = db.query(TrainedModel).filter(
            TrainedModel.registry_name == registry_name,
            TrainedModel.id != model_id,
        ).first()
        if clash:
            raise HTTPException(status_code=409, detail="registry_name already taken")
        model.registry_name = registry_name
    model.visibility = visibility
    db.commit()
    return {"success": True, "model": model.to_dict()}


@router.get("/models/{model_id}/metadata", response_model=Dict[str, Any])
async def get_model_metadata(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Return the validated contract for one model owned by the account."""
    record = db.query(TrainedModel).filter(
        TrainedModel.id == model_id,
        TrainedModel.user_id == current_user.userId,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        config = json.loads(record.config or "{}")
    except (TypeError, json.JSONDecodeError):
        config = {}
    return {
        "success": True,
        "model_id": record.id,
        "model_hash": config.get("model_hash"),
        "metadata": config.get("metadata"),
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
    check_download_allowed(current_user)
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
            # No model weights available
            raise HTTPException(
                status_code=404, 
                detail=f"Model weights not found for model '{model.name}'. The model may not have been trained yet."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


# ============================================================================
# MODEL DEPLOYMENT ENDPOINTS
# ============================================================================

class DeployModelRequest(BaseModel):
    """Request to deploy a trained model to a device."""
    model_id: int
    device_id: str  # device_uuid
    config: Optional[Dict[str, Any]] = None  # trigger config, thresholds, etc.


class PretrainedDeployRequest(BaseModel):
    """Request to deploy a built-in pretrained model to a device."""
    device_id: str
    model_key: str
    config: Optional[Dict[str, Any]] = None


@router.post("/models/{model_id}/deploy")
async def deploy_model_to_device(
    model_id: int,
    request: DeployModelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Queue a trained model for deployment to a specific device.

    Uses a pull-based model: the deployment payload is stored in the DB and
    the device picks it up on its next register/heartbeat cycle.  This works
    correctly even when the device is behind NAT or a private network.
    """
    from ..db import Device, DeviceDeployment
    import base64
    try:
        # Validate model
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        is_rule = (model.architecture == "rule" or getattr(model, "processor_type", None) == "rule")
        if not is_rule and not model.model_data:
            raise HTTPException(status_code=400, detail="Model has no weights to deploy")

        # Validate device
        device = db.query(Device).filter(
            Device.device_uuid == request.device_id,
            Device.userId == current_user.userId
        ).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        if not _deployment_requests_allowed_for_device(device):
            raise HTTPException(status_code=403, detail="Model deployment requests are disabled on this device")

        # Build deployment config
        deployment_id = str(uuid.uuid4())
        stored_config = json.loads(model.config) if model.config else {}
        metadata = stored_config.get('metadata') if isinstance(stored_config, dict) else None
        model_hash = stored_config.get('model_hash') if isinstance(stored_config, dict) else None
        if not is_rule and (not isinstance(metadata, dict) or metadata.get('schema') != 'thoth-model/v1'):
            raise HTTPException(status_code=400, detail='Only validated thoth-model/v1 TorchScript models can be deployed')
        deploy_config = request.config or {}
        deploy_config.update({
            "deployment_id": deployment_id,
            "model_name": model.name,
            "model_type": model.architecture or "unknown",
            "processor_type": "rule" if is_rule else "torchscript",
            "sensor": getattr(model, "sensor", None) or "any",
            "deployed_at": datetime.utcnow().isoformat(),
        })

        if is_rule:
            class_names = stored_config.get("class_names") or [
                r.get("label") for r in stored_config.get("rules", []) if r.get("label")
            ]
            rule_metadata = {
                "schema": "thoth-rule/v1",
                "name": model.name,
                "version": "1.0.0",
                "processor_type": "rule",
                "sensor": getattr(model, "sensor", None) or "any",
                "class_names": class_names,
            }
            full_payload = {
                "deployment_id": deployment_id,
                "model_name": model.name,
                "model_type": "rule",
                "processor_type": "rule",
                "sensor": getattr(model, "sensor", None) or "any",
                "metadata": rule_metadata,
                "rule_config": stored_config,
                "config": deploy_config,
            }
        else:
            # Preprocessing/window info travels inside the thoth-model/v1
            # manifest stored with the artifact, not a server-side training job.
            if isinstance(stored_config, dict):
                if stored_config.get("preprocessing"):
                    deploy_config["preprocessing"] = stored_config["preprocessing"]
                if stored_config.get("class_names"):
                    deploy_config["class_names"] = stored_config["class_names"]

            # Build full payload (model weights encoded as base64)
            full_payload = {
                "deployment_id": deployment_id,
                "model_name": model.name,
                "model_type": model.architecture or "unknown",
                "model_data": base64.b64encode(model.model_data).decode("utf-8"),
                "model_hash": model_hash or hashlib.sha256(model.model_data).hexdigest(),
                "metadata": metadata,
                "config": deploy_config,
            }

        # Store in DB — device will pick it up on next register/heartbeat
        record = DeviceDeployment(
            deployment_id=deployment_id,
            device_uuid=request.device_id,
            model_id=model_id,
            user_id=current_user.userId,
            payload=json.dumps(full_payload),
            status="pending",
        )
        db.add(record)
        db.commit()

        logger.info(f"Deployment {deployment_id} queued for device {request.device_id}")
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": model.name,
            "device_name": device.device_name,
            "device_id": device.device_uuid,
            "message": f"'{model.name}' queued for deployment — device will receive it on next sync"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue deployment: {str(e)}")


@router.post("/models/pretrained/deploy")
async def deploy_pretrained_model_to_device(
    request: PretrainedDeployRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Built-in semantic detectors were removed in favor of user models."""
    raise HTTPException(status_code=410, detail="Built-in pretrained models are no longer available; upload a TorchScript model")
    """Legacy implementation retained below for database migration reference."""
    from ..db import Device, DeviceDeployment
    import base64
    _require_ai_model_plan(current_user)

    try:
        if request.model_key != "opencv_person_detector":
            raise HTTPException(status_code=400, detail="Unsupported pretrained model")

        device = db.query(Device).filter(
            Device.device_uuid == request.device_id,
            Device.userId == current_user.userId
        ).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        if not _deployment_requests_allowed_for_device(device):
            raise HTTPException(status_code=403, detail="Model deployment requests are disabled on this device")

        model = db.query(TrainedModel).filter(
            TrainedModel.user_id == current_user.userId,
            TrainedModel.architecture == "opencv_person_detector",
            TrainedModel.name == "OpenCV Person Detector"
        ).first()
        if not model:
            model = TrainedModel(
                user_id=current_user.userId,
                job_id=None,
                name="OpenCV Person Detector",
                architecture="opencv_person_detector",
                accuracy=None,
                size_bytes=64,
                model_data=json.dumps({
                    "pretrained": True,
                    "model_key": request.model_key,
                    "runtime": "opencv-hog-person-detector"
                }).encode("utf-8"),
                config=json.dumps({
                    "pretrained": True,
                    "model_key": request.model_key,
                    "data_type": "image",
                }),
            )
            db.add(model)
            db.commit()
            db.refresh(model)

        deployment_id = str(uuid.uuid4())
        deploy_config = request.config or {}
        deploy_config.update({
            "deployment_id": deployment_id,
            "model_name": model.name,
            "model_type": model.architecture or "unknown",
            "pretrained": True,
            "model_key": request.model_key,
            "deployed_at": datetime.utcnow().isoformat(),
        })

        payload = {
            "deployment_id": deployment_id,
            "model_name": model.name,
            "model_type": model.architecture or "unknown",
            "model_data": base64.b64encode(model.model_data or b"{}").decode("utf-8"),
            "config": deploy_config,
        }

        record = DeviceDeployment(
            deployment_id=deployment_id,
            device_uuid=request.device_id,
            model_id=model.id,
            user_id=current_user.userId,
            payload=json.dumps(payload),
            status="pending",
        )
        db.add(record)
        db.commit()

        logger.info(f"Pretrained deployment {deployment_id} queued for device {request.device_id}")
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": model.name,
            "device_name": device.device_name,
            "device_id": device.device_uuid,
            "message": f"'{model.name}' queued for deployment — device will receive it on next sync"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to queue pretrained deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue pretrained deployment: {str(e)}")


@router.get("/models/deployments")
async def list_deployments(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all model deployments for the current user."""
    from ..db import Device, DeviceDeployment
    
    try:
        deployment_rows = db.query(DeviceDeployment).filter(
            DeviceDeployment.user_id == current_user.userId
        ).order_by(DeviceDeployment.created_at.desc()).all()
        
        result = []
        for deployment in deployment_rows:
            try:
                payload = json.loads(deployment.payload) if deployment.payload else {}
            except Exception:
                payload = {}
            device = db.query(Device).filter(Device.device_uuid == deployment.device_uuid).first()
            model = db.query(TrainedModel).filter(TrainedModel.id == deployment.model_id).first()
            result.append({
                "deployment_id": deployment.deployment_id,
                "model_id": deployment.model_id,
                "model_name": model.name if model else payload.get("model_name") or "Unknown model",
                "model_type": (model.architecture if model and model.architecture else payload.get("model_type") or "unknown"),
                "device_id": device.device_uuid if device else deployment.device_uuid,
                "device_name": device.device_name if device else payload.get("device_name") or deployment.device_uuid,
                "status": deployment.status,
                "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
                "delivered_at": deployment.delivered_at.isoformat() if deployment.delivered_at else None,
                "declined_at": getattr(deployment, "declined_at", None).isoformat() if getattr(deployment, "declined_at", None) else None,
                "runtime_model_id": payload.get("runtime_model_id"),
                "activation": payload.get("activation"),
            })
        
        return {
            "success": True,
            "deployments": result
        }
        
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/deployments/{deployment_id}/activation", response_model=Dict[str, Any])
async def set_deployment_activation(
    deployment_id: str,
    body: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Queue remote enable/disable for a delivered user model."""
    deployment = db.query(DeviceDeployment).filter(
        DeviceDeployment.deployment_id == deployment_id,
        DeviceDeployment.user_id == current_user.userId,
    ).first()
    if not deployment:
        raise HTTPException(status_code=404, detail='Deployment not found')
    if deployment.status != 'delivered':
        raise HTTPException(status_code=409, detail='Model must be delivered before activation can change')
    try:
        deployment_payload = json.loads(deployment.payload or '{}')
    except (TypeError, json.JSONDecodeError):
        deployment_payload = {}
    runtime_model_id = deployment_payload.get('runtime_model_id')
    if not runtime_model_id:
        raise HTTPException(status_code=409, detail='Device has not reported its local model id')
    device = db.query(Device).filter(Device.device_uuid == deployment.device_uuid, Device.userId == current_user.userId).first()
    if not device:
        raise HTTPException(status_code=404, detail='Device not found')
    enabled = bool(body.get('enabled'))
    command = DeviceCommand(
        device_id=device.deviceId,
        user_id=current_user.userId,
        command='enable_model' if enabled else 'disable_model',
        payload=json.dumps({'model_id': runtime_model_id, 'deployment_id': deployment_id}, separators=(',', ':')),
    )
    db.add(command)
    deployment_payload['activation'] = {'requested': enabled, 'status': 'pending', 'updated_at': datetime.utcnow().isoformat()}
    deployment.payload = json.dumps(deployment_payload)
    db.commit()
    db.refresh(command)
    return {'success': True, 'deployment_id': deployment_id, 'enabled': enabled, 'command': command.to_dict()}


@router.get("/models/pending-deployments")
async def list_pending_deployments(
    current_user = Depends(get_current_user)
):
    """Return pending model deployment requests for the current user.
    
    Deployments are pushed directly to devices; no async queue exists yet.
    This endpoint returns an empty list so the portal stops 405-erroring.
    """
    return {"success": True, "deployments": []}


@router.delete("/models/deployments/{deployment_id}")
async def cancel_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Cancel a pending deployment."""
    from ..db import DeviceDeployment
    
    try:
        deployment = db.query(DeviceDeployment).filter(
            DeviceDeployment.deployment_id == deployment_id,
            DeviceDeployment.user_id == current_user.userId,
            DeviceDeployment.status == "pending"
        ).first()
        
        if not deployment:
            raise HTTPException(status_code=404, detail="Pending deployment not found")
        
        db.delete(deployment)
        db.commit()
        
        return {
            "success": True,
            "message": "Deployment cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cancel deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/deployments/{deployment_id}/confirm")
async def confirm_deployment(
    deployment_id: str,
    body: Dict[str, Any] = Body(default={}),
    current_user = Depends(get_current_user)
):
    """Confirm (accept/decline) a pending model deployment request."""
    accepted = body.get("accepted", True)
    return {
        "success": True,
        "deployment_id": deployment_id,
        "accepted": accepted,
        "message": "Deployment accepted" if accepted else "Deployment declined"
    }


# ============================================================================
# DATASET DETAIL ENDPOINTS (catch-all routes must come last)
# ============================================================================

@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Download a full dataset as one zip archive grouped by label."""
    check_download_allowed(current_user)
    try:
        dataset = db.query(TrainingDataset).options(
            selectinload(TrainingDataset.files).selectinload(DatasetFile.file)
        ).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId,
        ).first()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        archive = io.BytesIO()
        used_names = set()
        manifest = {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "files": [],
        }

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for df in dataset.files or []:
                if not df.file:
                    continue
                content = _file_content(df.file)
                if content is None:
                    continue
                label = str(df.label or "unlabeled").replace("/", "_").replace("\\", "_")
                original = _original_filename(df.file.filename)
                arcname = f"{label}/{original}"
                if arcname in used_names:
                    arcname = f"{label}/{df.file.fileId}_{original}"
                used_names.add(arcname)
                zf.writestr(arcname, content)
                manifest["files"].append({
                    "file_id": df.file.fileId,
                    "filename": original,
                    "label": df.label,
                    "path": arcname,
                    "size": df.file.size,
                    "content_type": df.file.content_type,
                })

            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        archive.seek(0)
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in dataset.name) or f"dataset_{dataset.id}"
        return Response(
            content=archive.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download dataset %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")


@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get dataset details including all files and labels."""
    try:
        dataset = db.query(TrainingDataset).options(
            selectinload(TrainingDataset.files).selectinload(DatasetFile.file).load_only(
                File.fileId,
                File.filename,
                File.size,
                File.content_type,
                File.uploaded_at,
            )
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
        
        # Query file labels for concatenation
        file_labels_map = {}
        if file_ids:
            file_rows = db.query(File.fileId, File.labels).filter(
                File.fileId.in_(file_ids),
                File.userId == current_user.userId
            ).all()
            for fid, labels_json in file_rows:
                if labels_json:
                    try:
                        file_labels_map[fid] = json.loads(labels_json)
                    except:
                        file_labels_map[fid] = []
                else:
                    file_labels_map[fid] = []
        
        # Process files
        dataset_files_to_add = []
        all_labels_used = set()
        
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
            
            # Get file's existing labels and create concatenated label if multiple
            file_labels = file_labels_map.get(file_id, [])
            final_label = label
            
            # If file has multiple labels, create concatenated label (label1_label2)
            if len(file_labels) > 1:
                # Concatenate all labels with underscore
                concatenated = "_".join(file_labels)
                final_label = concatenated
                all_labels_used.add(concatenated)
            elif len(file_labels) == 1:
                final_label = file_labels[0]
                all_labels_used.add(file_labels[0])
            else:
                all_labels_used.add(label)
            
            # Prepare dataset file entry
            dataset_file = DatasetFile(
                dataset_id=dataset_id,
                file_id=file_id,
                label=final_label
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
