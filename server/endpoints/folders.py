"""Folder management endpoints for file organization.

This module provides endpoints for:
- Creating, listing, updating, and deleting folders
- Moving files between folders
- Uploading entire folders with queued file processing
- Adding folders to datasets
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File as FastAPIFile, Form, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, Field

from server.db import get_db
from server.auth import get_current_user
from server.db import User, File, Folder
from server.utils.logging_utils import log_request_start, log_response, log_error
from server.utils.error_handler import (
    APIError, handle_api_error, not_found_error, validation_error,
    ErrorCode
)

router = APIRouter(prefix="/folders", tags=["folders"])

# Pydantic models for API
class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None
    description: Optional[str] = None

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[int] = None
    description: Optional[str] = None

class FolderResponse(BaseModel):
    id: int
    name: str
    parent_id: Optional[int]
    path: str
    created_at: datetime
    updated_at: datetime
    file_count: int
    subfolder_count: int
    size_bytes: int
    description: Optional[str] = None

class FileMoveRequest(BaseModel):
    file_id: int
    folder_id: Optional[int] = None  # None means root

class FolderUploadFile(BaseModel):
    filename: str
    content: str  # Base64 encoded
    relative_path: str  # Path within the folder

class FolderUploadRequest(BaseModel):
    folder_name: str
    files: List[FolderUploadFile]
    parent_id: Optional[int] = None
    use_folder_name_as_label: bool = True

class AddFolderToDatasetRequest(BaseModel):
    folder_id: int
    dataset_id: int
    label: Optional[str] = None  # If None, use folder name as label


def build_folder_path(db: Session, folder_id: int) -> str:
    """Build full path for a folder by traversing parent chain."""
    path_parts = []
    current_id = folder_id
    folder_cache = {}
    
    while current_id is not None:
        if current_id in folder_cache:
            folder = folder_cache[current_id]
        else:
            folder = db.query(Folder).filter(Folder.folderId == current_id).first()
            if folder:
                folder_cache[current_id] = folder
            else:
                break
        
        path_parts.append(folder.name)
        current_id = folder.parent_id
    
    return "/" + "/".join(reversed(path_parts)) if path_parts else "/"


def get_folder_stats(db: Session, folder_id: int) -> Dict[str, int]:
    """Get file count, subfolder count, and total size for a folder."""
    file_count = db.query(func.count(File.fileId)).filter(File.folder_id == folder_id).scalar() or 0
    subfolder_count = db.query(func.count(Folder.folderId)).filter(Folder.parent_id == folder_id).scalar() or 0
    size_bytes = db.query(func.sum(File.size)).filter(File.folder_id == folder_id).scalar() or 0
    
    return {
        "file_count": file_count,
        "subfolder_count": subfolder_count,
        "size_bytes": size_bytes
    }

@router.post("/", response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Create a new folder."""
    try:
        log_request_start("/folders/", "POST", None, None, current_user.userId)
        
        # Validate folder name
        if not folder.name or folder.name.strip() == "":
            raise HTTPException(status_code=400, detail="Folder name cannot be empty")
        
        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in folder.name for char in invalid_chars):
            raise HTTPException(status_code=400, detail=f"Folder name contains invalid characters")
        
        # Validate parent folder if specified
        if folder.parent_id is not None:
            parent = db.query(Folder).filter(
                Folder.folderId == folder.parent_id,
                Folder.userId == current_user.userId
            ).first()
            if not parent:
                raise HTTPException(status_code=404, detail="Parent folder not found")
        
        # Create folder in database
        new_folder = Folder(
            name=folder.name.strip(),
            parent_id=folder.parent_id,
            userId=current_user.userId,
            description=folder.description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_folder)
        db.commit()
        db.refresh(new_folder)
        
        path = build_folder_path(db, new_folder.folderId)
        
        log_response(201, f"Folder created: {folder.name}", "/folders/")
        
        return FolderResponse(
            id=new_folder.folderId,
            name=new_folder.name,
            parent_id=new_folder.parent_id,
            path=path,
            created_at=new_folder.created_at,
            updated_at=new_folder.updated_at,
            file_count=0,
            subfolder_count=0,
            size_bytes=0,
            description=new_folder.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create folder: {str(e)}")

@router.get("/", response_model=List[FolderResponse])
async def list_folders(
    parent_id: Optional[int] = Query(None, description="Filter by parent folder (use -1 for root)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[FolderResponse]:
    """List folders for the current user."""
    try:
        log_request_start("/folders/", "GET", None, None, current_user.userId)
        
        # Query folders from database
        query = db.query(Folder).filter(Folder.userId == current_user.userId)
        
        # Filter by parent_id (-1 or None means root level)
        if parent_id is None or parent_id == -1:
            query = query.filter(Folder.parent_id == None)
        else:
            query = query.filter(Folder.parent_id == parent_id)
        
        folders = query.order_by(Folder.name).all()
        
        folder_responses = []
        for folder in folders:
            path = build_folder_path(db, folder.folderId)
            stats = get_folder_stats(db, folder.folderId)
            
            folder_responses.append(FolderResponse(
                id=folder.folderId,
                name=folder.name,
                parent_id=folder.parent_id,
                path=path,
                created_at=folder.created_at or datetime.utcnow(),
                updated_at=folder.updated_at or datetime.utcnow(),
                file_count=stats["file_count"],
                subfolder_count=stats["subfolder_count"],
                size_bytes=stats["size_bytes"],
                description=folder.description
            ))
        
        log_response(200, f"Retrieved {len(folder_responses)} folders", "/folders/")
        return folder_responses
        
    except Exception as e:
        log_error(f"Error listing folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list folders: {str(e)}")

@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Get folder details."""
    try:
        log_request_start(f"/folders/{folder_id}", "GET", None, None, current_user.userId)
        
        folder = db.query(Folder).filter(
            Folder.folderId == folder_id,
            Folder.userId == current_user.userId
        ).first()
        
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        path = build_folder_path(db, folder.folderId)
        stats = get_folder_stats(db, folder.folderId)
        
        return FolderResponse(
            id=folder.folderId,
            name=folder.name,
            parent_id=folder.parent_id,
            path=path,
            created_at=folder.created_at or datetime.utcnow(),
            updated_at=folder.updated_at or datetime.utcnow(),
            file_count=stats["file_count"],
            subfolder_count=stats["subfolder_count"],
            size_bytes=stats["size_bytes"],
            description=folder.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get folder: {str(e)}")


@router.get("/{folder_id}/files")
async def get_folder_files(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get files in a folder."""
    try:
        # Verify folder exists and belongs to user
        folder = db.query(Folder).filter(
            Folder.folderId == folder_id,
            Folder.userId == current_user.userId
        ).first()
        
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Get files in folder
        files = db.query(File).filter(
            File.folder_id == folder_id,
            File.userId == current_user.userId
        ).all()
        
        file_list = []
        for f in files:
            file_list.append({
                "file_id": f.fileId,
                "filename": f.filename,
                "size": f.size,
                "content_type": f.content_type,
                "data_type": f.data_type,
                "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None,
                "labels": json.loads(f.labels) if f.labels else []
            })
        
        return {
            "success": True,
            "folder_id": folder_id,
            "folder_name": folder.name,
            "files": file_list,
            "file_count": len(file_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting folder files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get folder files: {str(e)}")


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    folder_update: FolderUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> FolderResponse:
    """Update folder details."""
    try:
        log_request_start(f"/folders/{folder_id}", "PUT", None, None, current_user.userId)
        
        folder = db.query(Folder).filter(
            Folder.folderId == folder_id,
            Folder.userId == current_user.userId
        ).first()
        
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        if folder_update.name:
            folder.name = folder_update.name.strip()
        if folder_update.parent_id is not None:
            # Validate parent folder
            if folder_update.parent_id != -1:  # -1 means move to root
                parent = db.query(Folder).filter(
                    Folder.folderId == folder_update.parent_id,
                    Folder.userId == current_user.userId
                ).first()
                if not parent:
                    raise HTTPException(status_code=404, detail="Parent folder not found")
                folder.parent_id = folder_update.parent_id
            else:
                folder.parent_id = None
        if folder_update.description is not None:
            folder.description = folder_update.description
        
        folder.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(folder)
        
        path = build_folder_path(db, folder.folderId)
        stats = get_folder_stats(db, folder.folderId)
        
        log_response(200, f"Folder updated: {folder.name}", f"/folders/{folder_id}")
        
        return FolderResponse(
            id=folder.folderId,
            name=folder.name,
            parent_id=folder.parent_id,
            path=path,
            created_at=folder.created_at or datetime.utcnow(),
            updated_at=folder.updated_at,
            file_count=stats["file_count"],
            subfolder_count=stats["subfolder_count"],
            size_bytes=stats["size_bytes"],
            description=folder.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error updating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update folder: {str(e)}")


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: int,
    force: bool = Query(False, description="Force delete even if folder has contents"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a folder."""
    try:
        log_request_start(f"/folders/{folder_id}", "DELETE", None, None, current_user.userId)
        
        folder = db.query(Folder).filter(
            Folder.folderId == folder_id,
            Folder.userId == current_user.userId
        ).first()
        
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Check if folder has contents
        stats = get_folder_stats(db, folder_id)
        
        if not force and (stats["file_count"] > 0 or stats["subfolder_count"] > 0):
            raise HTTPException(
                status_code=400, 
                detail=f"Folder contains {stats['file_count']} files and {stats['subfolder_count']} subfolders. Use force=true to delete."
            )
        
        # If force, move files to root (set folder_id to None)
        if force:
            db.query(File).filter(File.folder_id == folder_id).update({"folder_id": None})
            # Delete subfolders recursively would be needed here
        
        folder_name = folder.name
        db.delete(folder)
        db.commit()
        
        log_response(200, f"Folder deleted: {folder_name}", f"/folders/{folder_id}")
        
        return {
            "success": True,
            "message": f"Folder '{folder_name}' deleted successfully",
            "folder_id": folder_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")


@router.post("/move-file")
async def move_file_to_folder(
    request: FileMoveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Move a file to a folder."""
    try:
        log_request_start("/folders/move-file", "POST", None, None, current_user.userId)
        
        # Get file
        file_record = db.query(File).filter(
            File.fileId == request.file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Validate folder if specified
        folder_name = "root"
        if request.folder_id is not None:
            folder = db.query(Folder).filter(
                Folder.folderId == request.folder_id,
                Folder.userId == current_user.userId
            ).first()
            
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found")
            folder_name = folder.name
        
        # Update file folder
        file_record.folder_id = request.folder_id
        db.commit()
        
        log_response(200, f"File {request.file_id} moved to {folder_name}", "/folders/move-file")
        
        return {
            "success": True,
            "message": f"File moved to {folder_name}",
            "file_id": request.file_id,
            "folder_id": request.folder_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error moving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to move file: {str(e)}")


# ============================================================================
# FOLDER UPLOAD ENDPOINT - Upload entire folder with queued file processing
# ============================================================================

import base64
import time

class FolderUploadStatus(BaseModel):
    """Status of a folder upload operation."""
    upload_id: str
    folder_id: int
    folder_name: str
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    status: str  # "processing", "completed", "failed"
    errors: List[str] = []

# In-memory upload status tracking (in production, use Redis or database)
_upload_status: Dict[str, FolderUploadStatus] = {}


@router.post("/upload-folder")
async def upload_folder(
    request: FolderUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload an entire folder with all its files.
    
    Files are processed in a queue. Each file gets:
    - Auto-detected file type
    - Metadata created with folder name as label
    - Stored in the created folder
    
    Args:
        request: Folder upload request with folder name and files
        
    Returns:
        Upload status with folder_id and upload_id for tracking
    """
    try:
        log_request_start("/folders/upload-folder", "POST", None, None, current_user.userId)
        
        # Validate folder name
        if not request.folder_name or request.folder_name.strip() == "":
            raise HTTPException(status_code=400, detail="Folder name cannot be empty")
        
        if not request.files or len(request.files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create the folder
        new_folder = Folder(
            name=request.folder_name.strip(),
            parent_id=request.parent_id,
            userId=current_user.userId,
            description=f"Uploaded folder with {len(request.files)} files",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_folder)
        db.commit()
        db.refresh(new_folder)
        
        # Generate upload ID for tracking
        upload_id = f"upload_{current_user.userId}_{new_folder.folderId}_{int(time.time())}"
        
        # Initialize upload status
        upload_status = FolderUploadStatus(
            upload_id=upload_id,
            folder_id=new_folder.folderId,
            folder_name=new_folder.name,
            total_files=len(request.files),
            processed_files=0,
            successful_files=0,
            failed_files=0,
            status="processing",
            errors=[]
        )
        _upload_status[upload_id] = upload_status
        
        # Process files in background
        background_tasks.add_task(
            process_folder_files,
            upload_id=upload_id,
            folder_id=new_folder.folderId,
            folder_name=new_folder.name,
            files=request.files,
            user_id=current_user.userId,
            use_folder_name_as_label=request.use_folder_name_as_label
        )
        
        log_response(202, f"Folder upload started: {new_folder.name} ({len(request.files)} files)", "/folders/upload-folder")
        
        return {
            "success": True,
            "message": f"Folder '{new_folder.name}' created. Processing {len(request.files)} files...",
            "folder_id": new_folder.folderId,
            "upload_id": upload_id,
            "total_files": len(request.files),
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error uploading folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload folder: {str(e)}")


async def process_folder_files(
    upload_id: str,
    folder_id: int,
    folder_name: str,
    files: List[FolderUploadFile],
    user_id: int,
    use_folder_name_as_label: bool
):
    """Background task to process folder files one by one."""
    from server.db import SessionLocal
    from server.file_type_detector import detect_file_type, DetectedFileType
    
    db = SessionLocal()
    status = _upload_status.get(upload_id)
    
    if not status:
        return
    
    try:
        for file_info in files:
            try:
                # Decode base64 content
                content_bytes = base64.b64decode(file_info.content)
                
                # Detect file type
                detection = detect_file_type(content_bytes[:8192], file_info.filename)
                
                # Map detected type to data_type string
                type_mapping = {
                    DetectedFileType.CSI: "csi",
                    DetectedFileType.GENERAL_CSV: "csv",
                    DetectedFileType.IMU: "imu",
                    DetectedFileType.IMAGE: "image",
                    DetectedFileType.VIDEO: "video",
                    DetectedFileType.AUDIO: "audio",
                }
                data_type = type_mapping.get(detection.detected_type, "unknown")
                
                # Create file metadata
                file_labels = [folder_name] if use_folder_name_as_label else []
                file_metadata = {
                    "original_filename": file_info.filename,
                    "relative_path": file_info.relative_path,
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "detected_type": detection.detected_type.value,
                    "detection_confidence": detection.confidence,
                    "labels": file_labels,
                    "primary_label": folder_name if use_folder_name_as_label else "",
                    "upload_timestamp": datetime.utcnow().isoformat(),
                }
                
                # Generate unique filename
                timestamp = int(time.time() * 1000)
                unique_filename = f"folder_{folder_id}_{timestamp}_{file_info.filename}"
                
                # Create file record
                db_file = File(
                    userId=user_id,
                    filename=unique_filename,
                    content=content_bytes,
                    size=len(content_bytes),
                    content_type=detection.detected_type.value,
                    uploaded_at=datetime.utcnow(),
                    file_hash=json.dumps(file_metadata),
                    data_type=data_type,
                    folder_id=folder_id,
                    labels=json.dumps(file_labels)
                )
                db.add(db_file)
                db.commit()
                
                status.successful_files += 1
                
            except Exception as file_error:
                status.failed_files += 1
                status.errors.append(f"{file_info.filename}: {str(file_error)}")
            
            status.processed_files += 1
        
        # Update final status
        if status.failed_files == 0:
            status.status = "completed"
        elif status.successful_files == 0:
            status.status = "failed"
        else:
            status.status = "completed"  # Partial success
            
    except Exception as e:
        status.status = "failed"
        status.errors.append(f"Processing error: {str(e)}")
    finally:
        db.close()


@router.get("/upload-status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get the status of a folder upload operation."""
    status = _upload_status.get(upload_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return {
        "success": True,
        "upload_id": status.upload_id,
        "folder_id": status.folder_id,
        "folder_name": status.folder_name,
        "total_files": status.total_files,
        "processed_files": status.processed_files,
        "successful_files": status.successful_files,
        "failed_files": status.failed_files,
        "status": status.status,
        "progress_percent": (status.processed_files / status.total_files * 100) if status.total_files > 0 else 0,
        "errors": status.errors[:10]  # Limit errors returned
    }


# ============================================================================
# ADD FOLDER TO DATASET
# ============================================================================

@router.post("/add-to-dataset")
async def add_folder_to_dataset(
    request: AddFolderToDatasetRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Add all files from a folder to a dataset.
    
    Each file in the folder is added to the dataset with:
    - Label from folder name (or custom label if provided)
    - File's existing labels are preserved
    
    Args:
        request: Folder ID, Dataset ID, and optional label
        
    Returns:
        Number of files added to dataset
    """
    try:
        log_request_start("/folders/add-to-dataset", "POST", None, None, current_user.userId)
        
        # Verify folder exists and belongs to user
        folder = db.query(Folder).filter(
            Folder.folderId == request.folder_id,
            Folder.userId == current_user.userId
        ).first()
        
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Verify dataset exists
        from server.db import TrainingDataset, DatasetFile
        
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == request.dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get all files in folder
        files = db.query(File).filter(
            File.folder_id == request.folder_id,
            File.userId == current_user.userId
        ).all()
        
        if not files:
            raise HTTPException(status_code=400, detail="Folder has no files")
        
        # Determine label to use
        label = request.label if request.label else folder.name
        
        # Add files to dataset
        added_count = 0
        skipped_count = 0
        
        for file in files:
            # Check if file already in dataset
            existing = db.query(DatasetFile).filter(
                DatasetFile.dataset_id == request.dataset_id,
                DatasetFile.file_id == file.fileId
            ).first()
            
            if existing:
                skipped_count += 1
                continue
            
            # Add file to dataset
            dataset_file = DatasetFile(
                dataset_id=request.dataset_id,
                file_id=file.fileId,
                label=label,
                added_at=datetime.utcnow()
            )
            db.add(dataset_file)
            added_count += 1
        
        # Update dataset labels if needed
        if label not in (dataset.labels or []):
            current_labels = dataset.labels or []
            current_labels.append(label)
            dataset.labels = current_labels
        
        db.commit()
        
        log_response(200, f"Added {added_count} files from folder '{folder.name}' to dataset", "/folders/add-to-dataset")
        
        return {
            "success": True,
            "message": f"Added {added_count} files to dataset with label '{label}'",
            "folder_id": request.folder_id,
            "folder_name": folder.name,
            "dataset_id": request.dataset_id,
            "label": label,
            "files_added": added_count,
            "files_skipped": skipped_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error adding folder to dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add folder to dataset: {str(e)}")
