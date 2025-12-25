"""File management endpoints."""

import base64
import json
import mimetypes
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request, status
from sqlalchemy.orm import Session

from server.db import get_db
from server.auth import get_current_user
from server.db import User, File
from server.utils.logging_utils import log_request_start, log_response, log_error
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
        
        # Get files with pagination
        files = query.order_by(File.uploaded_at.desc()).offset(offset).limit(limit + 1).all()
        
        has_more = len(files) > limit
        if has_more:
            files = files[:limit]
        
        file_list = []
        for file in files:
            # Extract original filename from stored filename
            parts = file.filename.split('_', 3)
            original_filename = parts[-1] if len(parts) >= 4 else file.filename
            
            # Extract device_id from filename
            extracted_device_id = None
            if len(parts) >= 4 and parts[1] != "user":
                extracted_device_id = parts[1]
            
            # Try to parse metadata from file_hash field
            metadata = {}
            if file.file_hash:
                try:
                    metadata = json.loads(file.file_hash)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            file_info = {
                "file_id": file.fileId,
                "filename": original_filename,
                "size": file.size,
                "content_type": file.content_type,
                "uploaded_at": file.uploaded_at.isoformat(),
                "device_id": extracted_device_id,
                "on_cloud": True,  # Files in this list are always on cloud
                "metadata": metadata
            }
            file_list.append(file_info)
        
        log_response(200, f"Retrieved {len(file_list)} files", "/file/files")
        return {
            "success": True,
            "files": file_list,
            "count": len(file_list),
            "has_more": has_more,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "next_offset": offset + limit if has_more else None
            }
        }
    except Exception as e:
        log_error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list files")


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
                log_response(200, f"File already exists: {request.filename}", "/file/upload")
                return {
                    "success": True,
                    "file_id": existing_file.fileId,
                    "filename": request.filename,
                    "size": existing_file.size,
                    "message": "File already exists"
                }
        
        # Create file metadata
        file_metadata = {
            "original_filename": request.filename,
            "content_type": content_type,
            "upload_timestamp": datetime.now().isoformat(),
            "device_id": request.device_id,
            "user_id": current_user.userId,
            "is_base64_encoded": request.is_base64
        }
        
        # Save file
        db_file = File(
            userId=current_user.userId,
            filename=unique_filename,
            content=content_bytes,
            size=len(content_bytes),
            content_type=content_type,
            uploaded_at=datetime.now(),
            # Store metadata as JSON in a comment field if available
            file_hash=json.dumps(file_metadata)  # Reusing file_hash field for metadata
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
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
        
        # Prepare headers
        headers = {
            "Content-Length": str(file_record.size)
        }
        
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{original_filename}"'
        else:
            headers["Content-Disposition"] = f'inline; filename="{original_filename}"'
        
        log_response(200, f"File {'downloaded' if download else 'viewed'}: {original_filename}", f"/file/{file_id}")
        
        return Response(
            content=file_record.content,
            media_type=file_record.content_type or "application/octet-stream",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error retrieving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file")


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
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract original filename for logging
        parts = file_record.filename.split('_', 3)
        original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
        
        # Delete the file
        db.delete(file_record)
        db.commit()
        
        log_response(200, f"File deleted: {original_filename} (ID: {file_id})", f"/file/{file_id}")
        return {
            "success": True,
            "message": f"File '{original_filename}' deleted successfully",
            "file_id": file_id
        }
        
    except HTTPException as he:
        log_error(f"HTTPException in file deletion: {str(he.detail)}")
        raise
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

