"""File management endpoints."""

import base64
import json
import mimetypes
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from server.db import get_db
from server.auth import get_current_user
from server.db import User, DBFile
from server.utils.logging_utils import log_request_start, log_response, log_error
from .models import FileUploadSimpleRequest, FileUploadResponse, PaginatedResponse

router = APIRouter(prefix="/file", tags=["files"])

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file_simple(
    request: FileUploadSimpleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a file with simplified interface.
    
    Purpose: Store files uploaded from devices or applications
    
    Args:
        request: File upload request with filename, content, and metadata
        
    Returns:
        FileUploadResponse: Upload confirmation with file_id and size
    """
    try:
        log_request_start("POST", "/file/upload", current_user.userId)
        
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
            existing_file = db.query(DBFile).filter(
                DBFile.userId == current_user.userId,
                DBFile.filename.like(f"file_{request.device_id}_%_{request.filename}")
            ).first()
            
            if existing_file:
                log_response(f"File already exists: {request.filename}", 200)
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
        db_file = DBFile(
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
        
        log_response(f"File uploaded: {request.filename} ({len(content_bytes)} bytes)", 200)
        return {
            "success": True,
            "file_id": db_file.fileId,
            "filename": request.filename,
            "size": len(content_bytes),
            "message": "File uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@router.get("/list")
async def list_user_files(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    device_id: Optional[str] = Query(None, description="Filter by source device"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List files uploaded by the authenticated user.
    
    Purpose: Retrieve paginated list of user's uploaded files
    
    Args:
        limit: Maximum number of files to return (1-200)
        offset: Number of files to skip
        device_id: Optional filter by source device
        content_type: Optional filter by content type
        
    Returns:
        Dict containing files array, count, and pagination info
    """
    try:
        log_request_start("GET", "/file/list", current_user.userId)
        
        # Build query
        query = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename.like("file_%")  # Only get files uploaded via file endpoints
        )
        
        # Apply filters
        if device_id:
            query = query.filter(DBFile.filename.like(f"file_{device_id}_%"))
        
        if content_type:
            query = query.filter(DBFile.content_type == content_type)
        
        # Get files with pagination
        files = query.order_by(DBFile.uploaded_at.desc()).offset(offset).limit(limit + 1).all()
        
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
                "metadata": metadata
            }
            file_list.append(file_info)
        
        log_response(f"Retrieved {len(file_list)} files", 200)
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
        file_record = db.query(DBFile).filter(
            DBFile.fileId == file_id,
            DBFile.userId == current_user.userId
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
        
        log_response(f"File {'downloaded' if download else 'viewed'}: {original_filename}", 200)
        
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
        file_record = db.query(DBFile).filter(
            DBFile.fileId == file_id,
            DBFile.userId == current_user.userId
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
            "metadata": metadata
        }
        
        log_response("File info retrieved", 200)
        return file_info
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting file info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get file info")

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
        file_record = db.query(DBFile).filter(
            DBFile.fileId == file_id,
            DBFile.userId == current_user.userId
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Extract original filename for logging
        parts = file_record.filename.split('_', 3)
        original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
        
        # Delete file
        db.delete(file_record)
        db.commit()
        
        log_response(f"File deleted: {original_filename}", 200)
        return {
            "success": True,
            "file_id": file_id,
            "filename": original_filename,
            "message": "File deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

@router.delete("/bulk")
async def delete_files_bulk(
    file_ids: list[int],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete multiple files by their IDs.
    
    Purpose: Remove multiple files from storage in a single operation
    
    Args:
        file_ids: List of file IDs to delete
        
    Returns:
        Dict containing bulk deletion results
    """
    try:
        log_request_start("DELETE", "/file/bulk", current_user.userId)
        
        if not file_ids or len(file_ids) == 0:
            raise HTTPException(status_code=400, detail="No file IDs provided")
        
        if len(file_ids) > 100:  # Prevent extremely large bulk operations
            raise HTTPException(status_code=400, detail="Too many files (max 100 per request)")
        
        # Get file records
        file_records = db.query(DBFile).filter(
            DBFile.fileId.in_(file_ids),
            DBFile.userId == current_user.userId
        ).all()
        
        if not file_records:
            raise HTTPException(status_code=404, detail="No files found")
        
        deleted_files = []
        for file_record in file_records:
            parts = file_record.filename.split('_', 3)
            original_filename = parts[-1] if len(parts) >= 4 else file_record.filename
            deleted_files.append({
                "file_id": file_record.fileId,
                "filename": original_filename
            })
            db.delete(file_record)
        
        db.commit()
        
        log_response(f"Bulk deleted {len(deleted_files)} files", 200)
        return {
            "success": True,
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files,
            "message": f"Successfully deleted {len(deleted_files)} files"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error in bulk file deletion: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete files")
