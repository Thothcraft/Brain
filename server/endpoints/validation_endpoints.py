"""Data Validation Endpoints.

This module provides API endpoints for:
- Validating uploaded files against recognized types
- Validating metadata files
- Getting file statistics and type detection
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File as FastAPIFile
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.db import get_db, User, File
from server.auth import get_current_user
from server.data_validation import (
    FileType,
    DataLoaderRegistry,
    ValidationResult,
    validate_metadata_file,
    create_metadata_template,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validation", tags=["validation"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ValidateFileRequest(BaseModel):
    file_id: int
    metadata: Optional[Dict[str, Any]] = None


class ValidateContentRequest(BaseModel):
    content: str  # Base64 encoded
    filename: str
    metadata: Optional[Dict[str, Any]] = None


class MetadataTemplateRequest(BaseModel):
    file_type: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/file-types", response_model=Dict[str, Any])
async def list_file_types():
    """List all recognized file types and their requirements."""
    return {
        "success": True,
        "file_types": [
            {
                "type": "csi",
                "name": "CSI (Channel State Information)",
                "extensions": [".csv"],
                "description": "WiFi CSI data with 128 I/Q values per line",
                "required_metadata": ["file_type", "label"],
                "optional_metadata": ["description", "num_subcarriers", "sampling_rate_hz", "antenna_config", "bandwidth_mhz"],
            },
            {
                "type": "general_csv",
                "name": "General CSV",
                "extensions": [".csv"],
                "description": "Comma-separated values with optional header",
                "required_metadata": ["file_type", "label"],
                "optional_metadata": ["description", "has_header", "delimiter"],
            },
            {
                "type": "image",
                "name": "Image",
                "extensions": [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".webp", ".tiff"],
                "description": "Image files for classification",
                "required_metadata": ["file_type", "label"],
                "optional_metadata": ["description"],
            },
            {
                "type": "video",
                "name": "Video",
                "extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
                "description": "Video files for activity recognition",
                "required_metadata": ["file_type", "label"],
                "optional_metadata": ["description"],
            },
            {
                "type": "imu",
                "name": "IMU (Inertial Measurement Unit)",
                "extensions": [".json", ".jsonl"],
                "description": "6-axis IMU data (accelerometer + gyroscope)",
                "required_metadata": ["file_type", "label"],
                "optional_metadata": ["description", "sampling_rate_hz"],
            },
        ]
    }


@router.post("/validate-file", response_model=Dict[str, Any])
async def validate_file(
    request: ValidateFileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Validate an uploaded file against recognized types."""
    try:
        # Get file from database
        file = db.query(File).filter(
            File.fileId == request.file_id,
            File.userId == current_user.userId,
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file content
        content = file.content
        if not content:
            # Try to load from storage
            from server.utils.supabase_storage import download_file_sync
            if file.storage_path:
                bucket, path = file.storage_path.split('/', 1)
                success, content = download_file_sync(bucket, path)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load file content")
            else:
                raise HTTPException(status_code=400, detail="File has no content")
        
        # Validate file
        result = DataLoaderRegistry.validate_file(content, file.filename, request.metadata)
        
        # Update file validation status in database
        file.is_validated = result.is_valid
        file.validation_status = "valid" if result.is_valid else "invalid"
        file.data_type = result.file_type.value
        db.commit()
        
        return {
            "success": True,
            "validation": result.to_dict(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-type", response_model=Dict[str, Any])
async def detect_file_type(
    request: ValidateContentRequest,
    current_user: User = Depends(get_current_user),
):
    """Detect file type from content without full validation."""
    try:
        import base64
        content = base64.b64decode(request.content)
        
        file_type = DataLoaderRegistry.detect_file_type(content, request.filename)
        
        return {
            "success": True,
            "file_type": file_type.value,
            "filename": request.filename,
        }
        
    except Exception as e:
        logger.error(f"Error detecting file type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-metadata", response_model=Dict[str, Any])
async def validate_metadata(
    metadata: Dict[str, Any],
    file_type: str = Query(..., description="Expected file type"),
    current_user: User = Depends(get_current_user),
):
    """Validate a metadata object for a specific file type."""
    try:
        # Convert file_type string to enum
        try:
            ft = FileType(file_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown file type: {file_type}")
        
        # Validate metadata
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        is_valid, parsed, errors = validate_metadata_file(metadata_bytes, ft)
        
        return {
            "success": True,
            "is_valid": is_valid,
            "metadata": parsed,
            "errors": errors,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata-template/{file_type}", response_model=Dict[str, Any])
async def get_metadata_template(
    file_type: str,
    current_user: User = Depends(get_current_user),
):
    """Get a metadata template for a specific file type."""
    try:
        try:
            ft = FileType(file_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown file type: {file_type}")
        
        template = create_metadata_template(ft)
        
        return {
            "success": True,
            "file_type": file_type,
            "template": template,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file-statistics/{file_id}", response_model=Dict[str, Any])
async def get_file_statistics(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed statistics for a file."""
    try:
        file = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId,
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        content = file.content
        if not content:
            from server.utils.supabase_storage import download_file_sync
            if file.storage_path:
                bucket, path = file.storage_path.split('/', 1)
                success, content = download_file_sync(bucket, path)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load file content")
            else:
                raise HTTPException(status_code=400, detail="File has no content")
        
        # Detect type and get loader
        file_type = DataLoaderRegistry.detect_file_type(content, file.filename)
        loader = DataLoaderRegistry.get_loader(file_type)
        
        if not loader:
            raise HTTPException(status_code=400, detail=f"No loader for file type: {file_type.value}")
        
        # Extract statistics
        statistics = loader.extract_metadata(content, file.filename)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_type.value,
            "size": file.size,
            "statistics": statistics,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-validate", response_model=Dict[str, Any])
async def batch_validate_files(
    file_ids: List[int],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Validate multiple files at once."""
    try:
        results = []
        
        for file_id in file_ids:
            file = db.query(File).filter(
                File.fileId == file_id,
                File.userId == current_user.userId,
            ).first()
            
            if not file:
                results.append({
                    "file_id": file_id,
                    "is_valid": False,
                    "error": "File not found",
                })
                continue
            
            content = file.content
            if not content:
                results.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "is_valid": False,
                    "error": "No content",
                })
                continue
            
            # Validate
            result = DataLoaderRegistry.validate_file(content, file.filename)
            
            # Update database
            file.is_validated = result.is_valid
            file.validation_status = "valid" if result.is_valid else "invalid"
            file.data_type = result.file_type.value
            
            results.append({
                "file_id": file_id,
                "filename": file.filename,
                "is_valid": result.is_valid,
                "file_type": result.file_type.value,
                "errors": result.errors,
                "warnings": result.warnings,
            })
        
        db.commit()
        
        valid_count = sum(1 for r in results if r.get("is_valid"))
        
        return {
            "success": True,
            "total": len(file_ids),
            "valid": valid_count,
            "invalid": len(file_ids) - valid_count,
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
