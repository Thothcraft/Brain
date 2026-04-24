"""File management endpoints."""

import base64
import json
import mimetypes
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request, status, UploadFile, File as FastAPIFile, Form
from sqlalchemy.orm import Session

from server.db import get_db
from server.auth import get_current_user
from server.db import User, File, DeviceFile, Device, DatasetFile, TrainingDataset
from server.utils.logging_utils import log_request_start, log_response, log_error
from server.utils.error_handler import (
    APIError, handle_api_error, file_error, validation_error, 
    not_found_error, ErrorCode
)
from .models import FileUploadSimpleRequest, FileUploadResponse, PaginatedResponse

router = APIRouter(prefix="/file", tags=["files"])

@router.get("/files", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def list_files(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    device_id: Optional[str] = Query(None, description="Filter by source device"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List files uploaded to cloud storage.
    
    Purpose: Retrieve paginated list of user's uploaded files, optionally filtered by device
    
    Args:
        limit: Maximum number of files to return (1-200)
        offset: Number of files to skip
        device_id: Optional filter by source device
        content_type: Optional filter by content type
        
    Returns:
        Dict containing files array, count, and pagination info
    """
    try:
        log_request_start("/file/files", "GET", None, None, current_user.userId)
        
        # Build query
        query = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like("file_%")  # Only get files uploaded via file endpoints
        )
        
        # Apply filters
        if device_id:
            query = query.filter(File.filename.like(f"file_{device_id}_%"))
        
        if content_type:
            query = query.filter(File.content_type == content_type)
        
        # Get files with pagination - exclude content column for performance
        files = query.with_entities(
            File.fileId,
            File.filename,
            File.size,
            File.content_type,
            File.uploaded_at,
            File.file_hash,
            File.labels,
            File.folder_id
        ).order_by(File.uploaded_at.desc()).offset(offset).limit(limit + 1).all()
        
        has_more = len(files) > limit
        if has_more:
            files = files[:limit]
        
        file_list = []
        for file in files:
            # file is now a tuple: (file_id, filename, size, content_type, uploaded_at, file_hash, labels, folder_id)
            file_id, filename, size, content_type, uploaded_at, file_hash, labels, folder_id = file
            
            # Extract original filename from stored filename
            parts = filename.split('_', 3)
            original_filename = parts[-1] if len(parts) >= 4 else filename
            
            # Extract device_id from filename
            extracted_device_id = None
            if len(parts) >= 4 and parts[1] != "user":
                extracted_device_id = parts[1]
            
            # Try to parse metadata from file_hash field
            metadata = {}
            if file_hash:
                try:
                    metadata = json.loads(file_hash)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Parse labels from JSON string
            parsed_labels = []
            if labels:
                try:
                    parsed_labels = json.loads(labels)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            file_info = {
                "file_id": file_id,
                "filename": original_filename,
                "size": size,
                "content_type": content_type,
                "uploaded_at": uploaded_at.isoformat(),
                "device_id": extracted_device_id,
                "on_cloud": True,  # Files in this list are always on cloud
                "metadata": metadata,
                "labels": parsed_labels,
                "folder_id": folder_id
            }
            file_list.append(file_info)
        
        log_response(200, f"Retrieved {len(file_list)} files", "/file/files")
        return {
            "success": True,
            "files": file_list,
            "total": len(file_list),
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "next_offset": offset + limit if has_more else None
            },
            "operation": "list_files",
            "status": "completed"
        }
    except Exception as e:
        log_error(f"Error listing files: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "files": [],
            "pagination": {
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "next_offset": None
            },
            "operation": "list_files",
            "status": "error",
            "error": str(e)
        }


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file_simple(
    request: FileUploadSimpleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    fastapi_request: Request = None
) -> Dict[str, Any]:
    """Upload a file with simplified interface.
    
    Purpose: Store files uploaded from devices or applications
    
    Args:
        request: File upload request with filename, content, and metadata
        
    Returns:
        FileUploadResponse: Upload confirmation with file_id and size
    """
    try:
        log_request_start(
            "/file/upload", 
            "POST", 
            fastapi_request, 
            remote_addr=None, 
            user_id=current_user.userId
        )
        
        # Decode content if base64
        if request.is_base64:
            try:
                content_bytes = base64.b64decode(request.content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")
        else:
            content_bytes = request.content.encode('utf-8')
        
        # Check file size (limit to 50MB for simple uploads)
        if len(content_bytes) > 52_428_800:  # 50MB
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Generate unique filename
        timestamp = int(time.time())
        if request.device_id:
            unique_filename = f"file_{request.device_id}_{timestamp}_{request.filename}"
        else:
            unique_filename = f"file_user_{timestamp}_{request.filename}"
        
        # Determine content type
        content_type = request.content_type or mimetypes.guess_type(request.filename)[0] or "application/octet-stream"
        
        # Check for existing file with same name from same device (optional deduplication)
        if request.device_id:
            existing_file = db.query(File).filter(
                File.userId == current_user.userId,
                File.filename.like(f"file_{request.device_id}_%_{request.filename}")
            ).first()
            
            if existing_file:
                # Update DeviceFile record even if file already exists
                try:
                    device = db.query(Device).filter(
                        Device.device_uuid == request.device_id
                    ).first()
                    
                    if device:
                        device_file = db.query(DeviceFile).filter(
                            DeviceFile.device_id == device.deviceId,
                            DeviceFile.filename == request.filename
                        ).first()
                        
                        if device_file:
                            device_file.on_cloud = True
                            device_file.upload_requested = False
                            device_file.cloud_file_id = existing_file.fileId
                            db.commit()
                            log_response(200, f"DeviceFile updated (existing): {request.filename} now on cloud", "/file/upload")
                except Exception as e:
                    log_error(f"Error updating DeviceFile for existing file: {e}")
                
                log_response(200, f"File already exists: {request.filename}", "/file/upload")
                return {
                    "success": True,
                    "file_id": existing_file.fileId,
                    "filename": request.filename,
                    "size": existing_file.size,
                    "message": "File already exists"
                }
        
        # Detect file type using content-based analysis (extension + first-line)
        from server.file_type_detector import detect_file_type, DetectedFileType
        
        detection = detect_file_type(content_bytes[:8192], request.filename)
        
        # Map detected type to data_type string
        type_to_data_type = {
            DetectedFileType.CSI: "csi",
            DetectedFileType.GENERAL_CSV: "csv",
            DetectedFileType.IMU: "imu",
            DetectedFileType.IMAGE: "image",
            DetectedFileType.VIDEO: "video",
            DetectedFileType.AUDIO: "audio",
        }
        data_type = type_to_data_type.get(detection.detected_type, "unknown")
        
        # Create file metadata with detection results
        file_metadata = {
            "original_filename": request.filename,
            "content_type": content_type,
            "upload_timestamp": datetime.now().isoformat(),
            "device_id": request.device_id,
            "user_id": current_user.userId,
            "is_base64_encoded": request.is_base64,
            # Detection metadata
            "detected_type": detection.detected_type.value,
            "detection_confidence": detection.confidence,
            "detection_method": detection.detection_method,
            "is_csi": detection.is_csi,
            # User-fillable fields (empty by default)
            "labels": [],
            "primary_label": "",
            "description": "",
            "subject_id": "",
            "environment": "",
            "activity": "",
        }
        
        # Add CSI-specific metadata
        if detection.is_csi:
            file_metadata["csi_array_length"] = detection.csi_array_length
            file_metadata["header_columns"] = detection.header_columns
        
        # Add CSV column info for general CSV
        if detection.detected_type == DetectedFileType.GENERAL_CSV:
            file_metadata["header_columns"] = detection.header_columns
            file_metadata["column_types"] = detection.statistics.get("column_types", {})
        
        # Add validation statistics
        if detection.statistics:
            file_metadata["statistics"] = detection.statistics
        
        log_response(200, f"Detected type for {request.filename}: {detection.detected_type.value} (confidence={detection.confidence})", "/file/upload")
        
        # Generate sample content for quick preview
        sample_content = None
        try:
            from server.ml_training import generate_file_sample
            sample_content, _ = generate_file_sample(content_bytes, request.filename)
            log_response(200, f"Generated sample for {request.filename}: sample_len={len(sample_content) if sample_content else 0}", "/file/upload")
        except Exception as sample_err:
            log_error(f"Failed to generate file sample: {sample_err}")
        
        # Save file record first (without content if using storage)
        db_file = File(
            userId=current_user.userId,
            filename=unique_filename,
            size=len(content_bytes),
            content_type=content_type,
            uploaded_at=datetime.now(),
            file_hash=json.dumps(file_metadata),
            sample_content=sample_content,
            data_type=data_type
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        # Try to upload to Supabase Storage first
        storage_path = None
        try:
            from server.utils.supabase_storage import upload_file_sync, generate_storage_path, is_storage_configured, BUCKET_FILES
            
            if is_storage_configured():
                path = generate_storage_path(current_user.userId, db_file.fileId, request.filename)
                success, result = upload_file_sync(BUCKET_FILES, path, content_bytes, content_type)
                
                if success:
                    storage_path = result
                    db_file.storage_path = storage_path
                    db_file.content = None  # Don't store in DB if in storage
                    db.commit()
                    log_response(200, f"File uploaded to Supabase Storage: {storage_path}", "/file/upload")
                else:
                    # Fall back to DB storage
                    log_error(f"Supabase Storage upload failed, falling back to DB: {result}")
                    db_file.content = content_bytes
                    db.commit()
            else:
                # No storage configured, use DB
                db_file.content = content_bytes
                db.commit()
        except Exception as storage_error:
            log_error(f"Storage error, falling back to DB: {storage_error}")
            db_file.content = content_bytes
            db.commit()
        
        # If this upload is from a device, update the DeviceFile record
        if request.device_id:
            try:
                # Find the device by UUID
                device = db.query(Device).filter(
                    Device.device_uuid == request.device_id
                ).first()
                
                if device:
                    # Find the DeviceFile record for this file
                    device_file = db.query(DeviceFile).filter(
                        DeviceFile.device_id == device.deviceId,
                        DeviceFile.filename == request.filename
                    ).first()
                    
                    if device_file:
                        device_file.on_cloud = True
                        device_file.upload_requested = False
                        device_file.cloud_file_id = db_file.fileId
                        db.commit()
                        log_response(200, f"DeviceFile updated: {request.filename} now on cloud (file_id={db_file.fileId})", "/file/upload")
            except Exception as e:
                log_error(f"Error updating DeviceFile record: {e}")
                # Don't fail the upload if DeviceFile update fails
        
        log_response(200, f"File uploaded: {request.filename} ({len(content_bytes)} bytes)", "/file/upload")
        return {
            "success": True,
            "file_id": db_file.fileId,
            "filename": request.filename,
            "size": len(content_bytes),
            "message": "File uploaded successfully"
        }
        
    except HTTPException as he:
        log_error(f"HTTPException in file upload: {str(he.detail)}")
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_error(f"Error uploading file: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/upload-multipart", response_model=FileUploadResponse)
async def upload_file_multipart(
    file: UploadFile = FastAPIFile(...),
    device_id: Optional[str] = Form(None),
    labels: Optional[str] = Form(None),
    folder_id: Optional[int] = Form(None),
    relative_path: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    fastapi_request: Request = None
) -> Dict[str, Any]:
    """Upload a file using multipart form data.
    
    Purpose: Store files uploaded from web browser with progress support
    
    Args:
        file: The uploaded file (multipart form data)
        device_id: Optional device identifier
        labels: Optional JSON array of labels for the file
        folder_id: Optional folder ID to place the file in
        relative_path: Optional relative path within folder
        
    Returns:
        FileUploadResponse: Upload confirmation with file_id and size
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        log_request_start(
            "/file/upload-multipart", 
            "POST", 
            fastapi_request, 
            remote_addr=None, 
            user_id=current_user.userId
        )
        
        raw_filename = file.filename or "unnamed_file"
        logger.info(f"Starting multipart upload: {raw_filename}")
        
        # Extract base filename from path (frontend may send relative paths like "train/drink.csv")
        # Use the last component as the actual filename
        filename = os.path.basename(raw_filename.replace('\\', '/'))
        if not filename:
            filename = "unnamed_file"
        
        # Store the original relative path for folder organization
        if relative_path is None and '/' in raw_filename:
            relative_path = raw_filename
        
        # Basic security check on the base filename only
        if '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename - path traversal not allowed")
        
        # Check for dangerous extensions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
        if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
            raise HTTPException(status_code=400, detail="File type not allowed")
        
        # Read file content in chunks to avoid memory issues
        logger.info(f"Reading file content: {filename}")
        chunks = []
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            total_size += len(chunk)
            # Check size limit during read
            max_size = current_user.max_file_size if hasattr(current_user, 'max_file_size') else 104_857_600  # Default 100MB
            if total_size > max_size:
                max_size_mb = max_size / (1024 * 1024)
                raise HTTPException(status_code=413, detail=f"File too large (max {max_size_mb:.1f}MB)")
        
        content_bytes = b''.join(chunks)
        logger.info(f"File read complete: {filename} ({len(content_bytes)} bytes)")
        
        # Generate unique filename
        timestamp = int(time.time())
        if device_id:
            unique_filename = f"file_{device_id}_{timestamp}_{filename}"
        else:
            unique_filename = f"file_user_{timestamp}_{filename}"
        
        # Determine content type
        content_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        
        # Detect file type using content-based analysis (extension + first-line)
        from server.file_type_detector import detect_file_type, DetectedFileType
        
        detection = detect_file_type(content_bytes[:8192], filename)
        
        # Map detected type to data_type string
        type_to_data_type = {
            DetectedFileType.CSI: "csi",
            DetectedFileType.GENERAL_CSV: "csv",
            DetectedFileType.IMU: "imu",
            DetectedFileType.IMAGE: "image",
            DetectedFileType.VIDEO: "video",
            DetectedFileType.AUDIO: "audio",
        }
        data_type = type_to_data_type.get(detection.detected_type, "unknown")
        
        # Parse labels from JSON string if provided
        parsed_labels = []
        if labels:
            try:
                parsed_labels = json.loads(labels)
                if not isinstance(parsed_labels, list):
                    parsed_labels = [str(parsed_labels)]
            except json.JSONDecodeError:
                parsed_labels = [labels]  # Use as single label if not valid JSON
        
        # Create file metadata with detection results
        file_metadata = {
            "original_filename": filename,
            "content_type": content_type,
            "upload_timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "user_id": current_user.userId,
            "upload_method": "multipart",
            "folder_id": folder_id,
            "relative_path": relative_path,
            # Detection metadata
            "detected_type": detection.detected_type.value,
            "detection_confidence": detection.confidence,
            "detection_method": detection.detection_method,
            "is_csi": detection.is_csi,
            # User-fillable fields
            "labels": parsed_labels,
            "primary_label": parsed_labels[0] if parsed_labels else "",
            "description": "",
            "subject_id": "",
            "environment": "",
            "activity": "",
        }
        
        # Add CSI-specific metadata
        if detection.is_csi:
            file_metadata["csi_array_length"] = detection.csi_array_length
            file_metadata["header_columns"] = detection.header_columns
        
        # Add CSV column info for general CSV
        if detection.detected_type == DetectedFileType.GENERAL_CSV:
            file_metadata["header_columns"] = detection.header_columns
            file_metadata["column_types"] = detection.statistics.get("column_types", {})
        
        # Add validation statistics
        if detection.statistics:
            file_metadata["statistics"] = detection.statistics
        
        logger.info(f"Detected type for {filename}: {detection.detected_type.value} (confidence={detection.confidence})")
        
        # Generate sample content for quick preview
        sample_content = None
        try:
            from server.ml_training import generate_file_sample
            sample_content, _ = generate_file_sample(content_bytes, filename)
            # Ensure no NUL characters in sample_content (PostgreSQL text fields can't contain NUL)
            if sample_content:
                sample_content = sample_content.replace('\x00', '')
            logger.info(f"Generated sample for {filename}: sample_len={len(sample_content) if sample_content else 0}")
        except Exception as sample_err:
            logger.warning(f"Failed to generate file sample: {sample_err}")
        
        # Save file to database
        logger.info(f"Saving to database: {filename}")
        db_file = File(
            userId=current_user.userId,
            filename=unique_filename,
            content=content_bytes,
            size=len(content_bytes),
            content_type=content_type,
            uploaded_at=datetime.now(),
            file_hash=json.dumps(file_metadata),
            sample_content=sample_content,
            data_type=data_type,
            folder_id=folder_id,
            labels=json.dumps(parsed_labels) if parsed_labels else None
        )
        db.add(db_file)
        
        try:
            db.commit()
            logger.info(f"Database commit successful: {filename}")
        except Exception as db_error:
            db.rollback()
            logger.error(f"Database commit failed: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        db.refresh(db_file)
        
        log_response(200, f"File uploaded (multipart): {filename} ({len(content_bytes)} bytes)", "/file/upload-multipart")
        return {
            "success": True,
            "file_id": db_file.fileId,
            "filename": filename,
            "size": len(content_bytes),
            "message": "File uploaded successfully"
        }
        
    except HTTPException as he:
        log_error(f"HTTPException in multipart file upload: {str(he.detail)}")
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_error(f"Error uploading file (multipart): {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/{file_id}")
async def download_file_simple(
    file_id: int,
    download: bool = Query(True, description="Whether to download as attachment"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download a file by its ID.
    
    Purpose: Retrieve file content for download or viewing
    
    Args:
        file_id: The unique file identifier
        download: Whether to force download (vs inline viewing)
        
    Returns:
        File content with appropriate headers
    """
    try:
        log_request_start("GET", f"/file/{file_id}", current_user.userId)
        
        # Get file record
        file_record = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract original filename
        parts = file_record.filename.split('_', 3)
        original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
        
        # Get file content - try Supabase Storage first, then DB
        file_content = None
        if file_record.storage_path:
            try:
                from server.utils.supabase_storage import download_file_sync, BUCKET_FILES
                # Extract path from storage_path (format: "bucket/path")
                path_parts = file_record.storage_path.split('/', 1)
                if len(path_parts) == 2:
                    bucket, path = path_parts
                    success, content = download_file_sync(bucket, path)
                    if success and content:
                        file_content = content
                        log_response(200, f"File retrieved from Supabase Storage: {file_record.storage_path}", f"/file/{file_id}")
            except Exception as storage_error:
                log_error(f"Failed to download from storage, trying DB: {storage_error}")
        
        # Fall back to DB content
        if file_content is None:
            file_content = file_record.content
        
        if file_content is None:
            raise HTTPException(status_code=404, detail="File content not available")
        
        # Prepare headers
        headers = {
            "Content-Length": str(len(file_content))
        }
        
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{original_filename}"'
        else:
            headers["Content-Disposition"] = f'inline; filename="{original_filename}"'
        
        log_response(200, f"File {'downloaded' if download else 'viewed'}: {original_filename}", f"/file/{file_id}")
        
        return Response(
            content=file_content,
            media_type=file_record.content_type or "application/octet-stream",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error retrieving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file")


@router.get("/{file_id}/sample")
async def get_file_sample(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get sample preview data for a file.
    
    Purpose: Retrieve quick preview data for preprocessing and training UI
    
    Args:
        file_id: The unique file identifier
        
    Returns:
        Dict containing sample content, data type, and format info
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        log_request_start("GET", f"/file/{file_id}/sample", current_user.userId)
        
        # Get file record
        file_record = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # If sample already exists in DB, return it
        if file_record.sample_content:
            return {
                "success": True,
                "file_id": file_id,
                "filename": file_record.filename,
                "sample_content": file_record.sample_content,
                "data_type": file_record.data_type or "unknown",
                "file_size": file_record.size,
                "cached": True
            }
        
        # Otherwise, generate sample from content
        file_content = file_record.content
        
        # Try to get from storage if not in DB
        if file_content is None and file_record.storage_path:
            try:
                from server.utils.supabase_storage import download_file_sync, BUCKET_FILES
                success, result = download_file_sync(BUCKET_FILES, file_record.storage_path)
                if success:
                    file_content = result
            except Exception as storage_err:
                logger.warning(f"Failed to download from storage for sample: {storage_err}")
        
        if file_content is None:
            raise HTTPException(status_code=404, detail="File content not available for preview")
        
        # Generate sample
        from server.ml_training import generate_file_sample, get_file_sample_info
        
        sample_content, data_type = generate_file_sample(file_content, file_record.filename)
        sample_info = get_file_sample_info(file_content, file_record.filename)
        
        # Cache the sample in DB for future requests
        try:
            file_record.sample_content = sample_content
            file_record.data_type = data_type
            db.commit()
            logger.info(f"Cached sample for file {file_id}")
        except Exception as cache_err:
            logger.warning(f"Failed to cache sample: {cache_err}")
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file_record.filename,
            "sample_content": sample_content,
            "data_type": data_type,
            "file_size": file_record.size,
            "sample_info": sample_info,
            "cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file sample: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file sample: {str(e)}")


@router.delete("/{file_id}")
async def delete_file_simple(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a file by its ID.
    
    Purpose: Remove a file from storage
    
    Args:
        file_id: The unique file identifier
        
    Returns:
        Dict containing deletion confirmation
    """
    try:
        log_request_start("DELETE", f"/file/{file_id}", current_user.userId)
        
        # Get file record
        file_record = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise handle_api_error(not_found_error("File", str(file_id)))
        
        # Extract original filename for logging
        parts = file_record.filename.split('_', 3)
        original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
        
        # Check if file is used in any datasets
        dataset_refs = db.query(DatasetFile).filter(DatasetFile.file_id == file_id).all()
        if dataset_refs:
            # Get dataset names for error message
            dataset_ids = list(set(ref.dataset_id for ref in dataset_refs))
            datasets = db.query(TrainingDataset).filter(TrainingDataset.id.in_(dataset_ids)).all()
            dataset_names = [ds.name for ds in datasets]
            
            raise handle_api_error(file_error(
                ErrorCode.FILE_DELETE_FAILED,
                f"Cannot delete file '{original_filename}' - it is used in {len(dataset_refs)} dataset(s)",
                {
                    "file_id": file_id,
                    "datasets": dataset_names,
                    "hint": "Remove the file from all datasets first, or delete the datasets"
                }
            ))
        
        # Delete the file
        try:
            db.delete(file_record)
            db.commit()
        except Exception as db_error:
            db.rollback()
            raise handle_api_error(file_error(
                ErrorCode.FILE_DELETE_FAILED,
                f"Failed to delete file '{original_filename}'",
                {"file_id": file_id, "db_error": str(db_error)}
            ))
        
        log_response(200, f"File deleted: {original_filename} (ID: {file_id})", f"/file/{file_id}")
        return {
            "success": True,
            "message": f"File '{original_filename}' deleted successfully",
            "file_id": file_id
        }
        
    except APIError as ae:
        log_error(f"APIError in file deletion: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


@router.get("/check/{device_id}/{filename}")
async def check_file_on_cloud(
    device_id: str,
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Check if a specific file from a device exists on cloud.
    
    Purpose: Check cloud availability before downloading from device
    
    Args:
        device_id: The device identifier
        filename: The original filename to check
        
    Returns:
        Dict containing on_cloud status and file_id if available
    """
    try:
        log_request_start("GET", f"/file/check/{device_id}/{filename}", current_user.userId)
        
        # Search for file with matching device_id and filename
        file_record = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like(f"file_{device_id}_%_{filename}")
        ).first()
        
        if file_record:
            log_response(200, f"File found on cloud: {filename}", f"/file/check/{device_id}/{filename}")
            return {
                "success": True,
                "on_cloud": True,
                "file_id": file_record.fileId,
                "size": file_record.size,
                "uploaded_at": file_record.uploaded_at.isoformat()
            }
        else:
            log_response(200, f"File not on cloud: {filename}", f"/file/check/{device_id}/{filename}")
            return {
                "success": True,
                "on_cloud": False,
                "file_id": None
            }
            
    except Exception as e:
        log_error(f"Error checking file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check file status")


@router.get("/{file_id}/info")
async def get_file_info(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get file information without downloading content.
    
    Purpose: Retrieve file metadata and information
    
    Args:
        file_id: The unique file identifier
        
    Returns:
        Dict containing file metadata
    """
    try:
        log_request_start("GET", f"/file/{file_id}/info", current_user.userId)
        
        # Get file record
        file_record = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract original filename and device_id
        parts = file_record.filename.split('_', 3)
        original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
        device_id = parts[1] if len(parts) >= 4 and parts[1] != "user" else None
        
        # Parse metadata if available
        metadata = {}
        if file_record.file_hash:
            try:
                metadata = json.loads(file_record.file_hash)
            except (json.JSONDecodeError, TypeError):
                pass
        
        file_info = {
            "success": True,
            "file_id": file_record.fileId,
            "filename": original_filename,
            "size": file_record.size,
            "content_type": file_record.content_type,
            "uploaded_at": file_record.uploaded_at.isoformat(),
            "device_id": device_id,
            "on_cloud": True,
            "metadata": metadata
        }
        
        log_response(200, "File info retrieved", f"/file/{file_id}/info")
        return file_info
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting file info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get file info")


@router.post("/upload-from-device")
async def upload_file_from_device(
    device_file_id: int = Query(..., description="ID of the DeviceFile record"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a file from a device to cloud storage.
    
    This endpoint fetches the file content from the Thoth device and stores it in the cloud.
    The device must be online and reachable.
    
    Args:
        device_file_id: The ID of the DeviceFile record to upload
        
    Returns:
        Dict with success status and cloud file ID
    """
    import requests as http_requests
    from server.db import DeviceFile, Device
    
    try:
        log_request_start("POST", "/file/upload-from-device", current_user.userId)
        
        # Get the DeviceFile record
        device_file = db.query(DeviceFile).filter(
            DeviceFile.id == device_file_id,
            DeviceFile.user_id == current_user.userId
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device file not found")
        
        if device_file.on_cloud:
            return {
                "success": True,
                "message": "File already on cloud",
                "cloud_file_id": device_file.cloud_file_id
            }
        
        # Get the device
        device = db.query(Device).filter(Device.deviceId == device_file.device_id).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        
        if not device.online or not device.ip_address:
            raise HTTPException(status_code=400, detail="Device is offline or IP address unknown")
        
        # Fetch file content from device
        device_url = f"http://{device.ip_address}:5000/api/files/download/{device_file.filename}"
        
        try:
            response = http_requests.get(device_url, timeout=60, stream=True)
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to fetch file from device: {response.status_code}")
            
            content_bytes = response.content
        except http_requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Could not reach device: {str(e)}")
        
        # Determine content type
        content_type = mimetypes.guess_type(device_file.filename)[0] or "application/octet-stream"
        
        # Generate unique filename
        timestamp = int(time.time())
        unique_filename = f"file_{device.device_uuid}_{timestamp}_{device_file.filename}"
        
        # Create file metadata
        file_metadata = {
            "original_filename": device_file.filename,
            "content_type": content_type,
            "upload_timestamp": datetime.now().isoformat(),
            "device_id": device.device_uuid,
            "device_file_id": device_file_id,
            "user_id": current_user.userId
        }
        
        # Save to database
        db_file = File(
            userId=current_user.userId,
            filename=unique_filename,
            content=content_bytes,
            size=len(content_bytes),
            content_type=content_type,
            uploaded_at=datetime.now(),
            file_hash=json.dumps(file_metadata)
        )
        db.add(db_file)
        
        # Update DeviceFile to mark as on_cloud
        device_file.on_cloud = True
        device_file.cloud_file_id = db_file.fileId
        
        db.commit()
        db.refresh(db_file)
        
        log_response(200, f"File uploaded from device: {device_file.filename}", "/file/upload-from-device")
        return {
            "success": True,
            "cloud_file_id": db_file.fileId,
            "filename": device_file.filename,
            "size": len(content_bytes),
            "message": "File uploaded to cloud successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error uploading file from device: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# ============================================================================
# FILE METADATA ENDPOINTS
# ============================================================================

from pydantic import BaseModel, Field
from typing import List

class FileMetadataRequest(BaseModel):
    """Request model for updating file metadata."""
    labels: Optional[List[str]] = Field(None, description="List of labels for the file")
    primary_label: Optional[str] = Field(None, description="Primary classification label")
    description: Optional[str] = Field(None, description="File description")
    subject_id: Optional[str] = Field(None, description="Subject identifier")
    environment: Optional[str] = Field(None, description="Environment description")
    activity: Optional[str] = Field(None, description="Activity label")


@router.get("/{file_id}/metadata")
async def get_file_metadata(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get metadata for a file including auto-detected type and user labels.
    
    Returns both auto-detected metadata (file type, statistics) and user-provided
    metadata (labels, description, etc.).
    
    Args:
        file_id: The file ID
        
    Returns:
        Dict containing file metadata
    """
    try:
        log_request_start(f"/file/{file_id}/metadata", "GET", None, None, current_user.userId)
        
        # Get the file
        file = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file content for type detection
        content = None
        if file.content:
            content = file.content
        elif file.storage_path:
            # Try to get from storage
            try:
                from server.utils.supabase_storage import download_file_sync, BUCKET_FILES
                success, result = download_file_sync(BUCKET_FILES, file.storage_path)
                if success:
                    content = result
            except Exception as e:
                log_error(f"Failed to download file for metadata: {e}")
        
        # Detect file type
        from server.file_type_detector import detect_file_type, create_brain_metadata
        
        metadata_response = {
            "file_id": file.fileId,
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "uploaded_at": file.uploaded_at.isoformat() if file.uploaded_at else None,
            "data_type": file.data_type,
        }
        
        # Add detection results if we have content
        if content:
            detection = detect_file_type(content[:8192], file.filename)
            metadata_response["detected_type"] = detection.detected_type.value
            metadata_response["detection_confidence"] = detection.confidence
            metadata_response["detection_method"] = detection.detection_method
            metadata_response["is_csi"] = detection.is_csi
            
            if detection.header_columns:
                metadata_response["header_columns"] = detection.header_columns
            if detection.statistics:
                metadata_response["statistics"] = detection.statistics
        
        # Parse existing metadata from file_hash
        existing_metadata = {}
        if file.file_hash:
            try:
                existing_metadata = json.loads(file.file_hash)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Add user-provided metadata
        metadata_response["labels"] = existing_metadata.get("labels", [])
        metadata_response["primary_label"] = existing_metadata.get("primary_label", "")
        metadata_response["description"] = existing_metadata.get("description", "")
        metadata_response["subject_id"] = existing_metadata.get("subject_id", "")
        metadata_response["environment"] = existing_metadata.get("environment", "")
        metadata_response["activity"] = existing_metadata.get("activity", "")
        
        log_response(200, f"Retrieved metadata for file {file_id}", f"/file/{file_id}/metadata")
        return {
            "success": True,
            "metadata": metadata_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting file metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file metadata: {str(e)}")


@router.put("/{file_id}/metadata")
async def update_file_metadata(
    file_id: int,
    request: FileMetadataRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update user-provided metadata for a file.
    
    Allows users to add labels, description, and other metadata to files.
    Labels can be used when adding files to datasets.
    
    Args:
        file_id: The file ID
        request: Metadata update request
        
    Returns:
        Dict containing updated metadata
    """
    try:
        log_request_start(f"/file/{file_id}/metadata", "PUT", None, None, current_user.userId)
        
        # Get the file
        file = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Parse existing metadata
        existing_metadata = {}
        if file.file_hash:
            try:
                existing_metadata = json.loads(file.file_hash)
            except (json.JSONDecodeError, TypeError):
                existing_metadata = {}
        
        # Update metadata fields
        if request.labels is not None:
            existing_metadata["labels"] = request.labels
        if request.primary_label is not None:
            existing_metadata["primary_label"] = request.primary_label
        if request.description is not None:
            existing_metadata["description"] = request.description
        if request.subject_id is not None:
            existing_metadata["subject_id"] = request.subject_id
        if request.environment is not None:
            existing_metadata["environment"] = request.environment
        if request.activity is not None:
            existing_metadata["activity"] = request.activity
        
        # Add update timestamp
        existing_metadata["metadata_updated_at"] = datetime.now().isoformat()
        
        # Save back to file_hash
        file.file_hash = json.dumps(existing_metadata)
        db.commit()
        
        log_response(200, f"Updated metadata for file {file_id}", f"/file/{file_id}/metadata")
        return {
            "success": True,
            "message": "File metadata updated successfully",
            "metadata": existing_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error updating file metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update file metadata: {str(e)}")


@router.post("/{file_id}/detect-type")
async def detect_file_type_endpoint(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Detect and return the file type based on content analysis.
    
    Uses extension + first-line content analysis to determine file type.
    CSI files are identified by their specific header format.
    General CSV files have columns treated as time-dependent features.
    
    Args:
        file_id: The file ID
        
    Returns:
        Dict containing detection results
    """
    try:
        log_request_start(f"/file/{file_id}/detect-type", "POST", None, None, current_user.userId)
        
        # Get the file
        file = db.query(File).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file content
        content = None
        if file.content:
            content = file.content
        elif file.storage_path:
            try:
                from server.utils.supabase_storage import download_file_sync, BUCKET_FILES
                success, result = download_file_sync(BUCKET_FILES, file.storage_path)
                if success:
                    content = result
            except Exception as e:
                log_error(f"Failed to download file for detection: {e}")
        
        if not content:
            raise HTTPException(status_code=400, detail="File content not available for detection")
        
        # Detect file type
        from server.file_type_detector import detect_file_type, DetectedFileType
        
        detection = detect_file_type(content[:8192], file.filename)
        
        # Update file's data_type field
        type_to_data_type = {
            DetectedFileType.CSI: "csi",
            DetectedFileType.GENERAL_CSV: "csv",
            DetectedFileType.IMU: "imu",
            DetectedFileType.IMAGE: "image",
            DetectedFileType.VIDEO: "video",
            DetectedFileType.AUDIO: "audio",
        }
        
        new_data_type = type_to_data_type.get(detection.detected_type, "unknown")
        if file.data_type != new_data_type:
            file.data_type = new_data_type
            db.commit()
        
        response = {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "detected_type": detection.detected_type.value,
            "confidence": detection.confidence,
            "detection_method": detection.detection_method,
            "is_csi": detection.is_csi,
        }
        
        if detection.header_columns:
            response["header_columns"] = detection.header_columns
        if detection.csi_array_length:
            response["csi_array_length"] = detection.csi_array_length
        if detection.statistics:
            response["statistics"] = detection.statistics
        if detection.errors:
            response["errors"] = detection.errors
        if detection.warnings:
            response["warnings"] = detection.warnings
        
        log_response(200, f"Detected type for file {file_id}: {detection.detected_type.value}", f"/file/{file_id}/detect-type")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error detecting file type: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect file type: {str(e)}")

