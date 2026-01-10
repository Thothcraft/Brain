"""Resumable upload endpoints for large files."""

import hashlib
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File as FastAPIFile, Form, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.db import get_db
from server.auth import get_current_user
from server.db import User
from server.utils.logging_utils import log_request_start, log_response, log_error
from server.utils.error_handler import (
    APIError, handle_api_error, file_error, validation_error,
    database_error, ErrorCode
)

router = APIRouter(prefix="/upload", tags=["resumable-upload"])

# In-memory storage for upload sessions (in production, use Redis)
UPLOAD_SESSIONS: Dict[str, Dict[str, Any]] = {}

class UploadInitRequest(BaseModel):
    filename: str
    total_size: int
    chunk_size: int = 1024 * 1024  # 1MB default
    file_hash: Optional[str] = None  # SHA-256 hash of entire file
    content_type: Optional[str] = None

class UploadInitResponse(BaseModel):
    success: bool
    upload_id: str
    chunk_size: int
    total_chunks: int
    uploaded_chunks: list[int]
    expires_at: datetime

class ChunkUploadResponse(BaseModel):
    success: bool
    chunk_number: int
    uploaded_chunks: list[int]
    remaining_chunks: int

class UploadCompleteResponse(BaseModel):
    success: bool
    file_id: int
    filename: str
    size: int
    upload_url: str

def generate_upload_id(user_id: int, filename: str) -> str:
    """Generate unique upload ID."""
    timestamp = datetime.utcnow().isoformat()
    unique_string = f"{user_id}-{filename}-{timestamp}"
    return hashlib.sha256(unique_string.encode()).hexdigest()[:32]

def cleanup_expired_sessions():
    """Remove expired upload sessions."""
    now = datetime.utcnow()
    expired_ids = []
    
    for upload_id, session in UPLOAD_SESSIONS.items():
        if now > session.get('expires_at', now):
            expired_ids.append(upload_id)
    
    for upload_id in expired_ids:
        # Clean up temporary files
        temp_dir = session.get('temp_dir')
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        del UPLOAD_SESSIONS[upload_id]

@router.post("/init", response_model=UploadInitResponse)
async def init_resumable_upload(
    request: UploadInitRequest,
    current_user: User = Depends(get_current_user)
) -> UploadInitResponse:
    """Initialize a resumable upload session.
    
    Args:
        request: Upload initialization parameters
        
    Returns:
        Upload session information
    """
    try:
        cleanup_expired_sessions()
        
        # Validate file size
        max_size = 100 * 1024 * 1024 * 1024  # 100GB
        if request.total_size > max_size:
            raise handle_api_error(file_error(
                ErrorCode.FILE_TOO_LARGE,
                f"File size {request.total_size} exceeds maximum {max_size}",
                {"max_size": max_size, "requested_size": request.total_size}
            ))
        
        # Generate upload ID
        upload_id = generate_upload_id(current_user.userId, request.filename)
        
        # Calculate chunk count
        total_chunks = (request.total_size + request.chunk_size - 1) // request.chunk_size
        
        # Create upload session
        session = {
            'upload_id': upload_id,
            'user_id': current_user.userId,
            'filename': request.filename,
            'total_size': request.total_size,
            'chunk_size': request.chunk_size,
            'total_chunks': total_chunks,
            'uploaded_chunks': [],
            'content_type': request.content_type,
            'file_hash': request.file_hash,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=24),  # 24 hour expiry
            'temp_dir': f"/tmp/uploads/{upload_id}"
        }
        
        # Create temporary directory
        os.makedirs(session['temp_dir'], exist_ok=True)
        
        UPLOAD_SESSIONS[upload_id] = session
        
        log_response(200, f"Upload session created: {upload_id}", "/upload/init")
        
        return UploadInitResponse(
            success=True,
            upload_id=upload_id,
            chunk_size=request.chunk_size,
            total_chunks=total_chunks,
            uploaded_chunks=[],
            expires_at=session['expires_at']
        )
        
    except APIError as ae:
        log_error(f"APIError in upload init: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error initializing upload: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.FILE_UPLOAD_FAILED,
            "Failed to initialize upload",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.get("/status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get upload session status.
    
    Args:
        upload_id: Upload session identifier
        
    Returns:
        Current upload status
    """
    try:
        session = UPLOAD_SESSIONS.get(upload_id)
        
        if not session:
            raise handle_api_error(not_found_error("Upload session", upload_id))
        
        if session['user_id'] != current_user.userId:
            raise handle_api_error(APIError(
                ErrorCode.INSUFFICIENT_PERMISSIONS,
                "Access denied to upload session",
                status.HTTP_403_FORBIDDEN
            ))
        
        # Check if expired
        if datetime.utcnow() > session['expires_at']:
            del UPLOAD_SESSIONS[upload_id]
            raise handle_api_error(APIError(
                ErrorCode.RESOURCE_EXPIRED,
                "Upload session has expired",
                status.HTTP_410_GONE
            ))
        
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": session['filename'],
            "total_chunks": session['total_chunks'],
            "uploaded_chunks": session['uploaded_chunks'],
            "progress": len(session['uploaded_chunks']) / session['total_chunks'] * 100,
            "expires_at": session['expires_at']
        }
        
    except APIError as ae:
        log_error(f"APIError getting upload status: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error getting upload status: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to get upload status",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.post("/chunk/{upload_id}", response_model=ChunkUploadResponse)
async def upload_chunk(
    upload_id: str,
    chunk_number: int = Query(..., ge=0),
    chunk_hash: str = Query(..., description="SHA-256 hash of chunk data"),
    chunk: UploadFile = FastAPIFile(...),
    current_user: User = Depends(get_current_user)
) -> ChunkUploadResponse:
    """Upload a chunk of data.
    
    Args:
        upload_id: Upload session identifier
        chunk_number: Zero-based chunk index
        chunk_hash: SHA-256 hash of chunk data
        chunk: Chunk data
        
    Returns:
        Upload progress after this chunk
    """
    try:
        session = UPLOAD_SESSIONS.get(upload_id)
        
        if not session:
            raise handle_api_error(not_found_error("Upload session", upload_id))
        
        if session['user_id'] != current_user.userId:
            raise handle_api_error(APIError(
                ErrorCode.INSUFFICIENT_PERMISSIONS,
                "Access denied to upload session",
                status.HTTP_403_FORBIDDEN
            ))
        
        # Validate chunk number
        if chunk_number >= session['total_chunks']:
            raise handle_api_error(validation_error(
                f"Invalid chunk number {chunk_number}. Max: {session['total_chunks'] - 1}"
            ))
        
        # Check if already uploaded
        if chunk_number in session['uploaded_chunks']:
            return ChunkUploadResponse(
                success=True,
                chunk_number=chunk_number,
                uploaded_chunks=session['uploaded_chunks'],
                remaining_chunks=session['total_chunks'] - len(session['uploaded_chunks'])
            )
        
        # Read chunk data
        chunk_data = await chunk.read()
        
        # Verify chunk hash
        calculated_hash = hashlib.sha256(chunk_data).hexdigest()
        if calculated_hash != chunk_hash:
            raise handle_api_error(validation_error(
                "Chunk hash mismatch. Data may be corrupted.",
                {"expected": chunk_hash, "calculated": calculated_hash}
            ))
        
        # Save chunk to temporary file
        chunk_path = os.path.join(session['temp_dir'], f"chunk_{chunk_number:06d}")
        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)
        
        # Update session
        if chunk_number not in session['uploaded_chunks']:
            session['uploaded_chunks'].append(chunk_number)
            session['uploaded_chunks'].sort()
        
        remaining = session['total_chunks'] - len(session['uploaded_chunks'])
        
        log_response(200, f"Chunk {chunk_number} uploaded for {upload_id}", f"/upload/chunk/{upload_id}")
        
        return ChunkUploadResponse(
            success=True,
            chunk_number=chunk_number,
            uploaded_chunks=session['uploaded_chunks'],
            remaining_chunks=remaining
        )
        
    except APIError as ae:
        log_error(f"APIError uploading chunk: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error uploading chunk: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.FILE_UPLOAD_FAILED,
            "Failed to upload chunk",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.post("/complete/{upload_id}", response_model=UploadCompleteResponse)
async def complete_upload(
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UploadCompleteResponse:
    """Complete the resumable upload and assemble the file.
    
    Args:
        upload_id: Upload session identifier
        
    Returns:
        Completed file information
    """
    try:
        session = UPLOAD_SESSIONS.get(upload_id)
        
        if not session:
            raise handle_api_error(not_found_error("Upload session", upload_id))
        
        if session['user_id'] != current_user.userId:
            raise handle_api_error(APIError(
                ErrorCode.INSUFFICIENT_PERMISSIONS,
                "Access denied to upload session",
                status.HTTP_403_FORBIDDEN
            ))
        
        # Check if all chunks are uploaded
        if len(session['uploaded_chunks']) != session['total_chunks']:
            raise handle_api_error(validation_error(
                f"Upload incomplete. {len(session['uploaded_chunks'])}/{session['total_chunks']} chunks uploaded."
            ))
        
        # Assemble file from chunks
        temp_file_path = os.path.join(session['temp_dir'], session['filename'])
        total_size = 0
        file_hash = hashlib.sha256()
        
        with open(temp_file_path, 'wb') as output_file:
            for chunk_num in range(session['total_chunks']):
                chunk_path = os.path.join(session['temp_dir'], f"chunk_{chunk_num:06d}")
                with open(chunk_path, 'rb') as chunk_file:
                    chunk_data = chunk_file.read()
                    output_file.write(chunk_data)
                    file_hash.update(chunk_data)
                    total_size += len(chunk_data)
        
        # Verify file size and hash
        if total_size != session['total_size']:
            raise handle_api_error(APIError(
                ErrorCode.FILE_CORRUPTED,
                f"File size mismatch. Expected: {session['total_size']}, Actual: {total_size}",
                status.HTTP_400_BAD_REQUEST
            ))
        
        if session['file_hash'] and file_hash.hexdigest() != session['file_hash']:
            raise handle_api_error(APIError(
                ErrorCode.FILE_CORRUPTED,
                "File hash mismatch. File may be corrupted.",
                status.HTTP_400_BAD_REQUEST
            ))
        
        # Save to database (reuse existing file upload logic)
        from server.endpoints.file_endpoints import save_file_to_database
        
        try:
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()
            
            file_record = await save_file_to_database(
                db=db,
                user_id=current_user.userId,
                filename=session['filename'],
                content=file_content,
                content_type=session['content_type'] or 'application/octet-stream',
                device_id=None
            )
            
            file_id = file_record.fileId
            
        except Exception as db_error:
            raise handle_api_error(database_error(
                ErrorCode.DATABASE_CONNECTION_ERROR,
                "Failed to save file to database",
                {"db_error": str(db_error)}
            ))
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(session['temp_dir'], ignore_errors=True)
        del UPLOAD_SESSIONS[upload_id]
        
        log_response(200, f"Upload completed: {session['filename']} (ID: {file_id})", f"/upload/complete/{upload_id}")
        
        return UploadCompleteResponse(
            success=True,
            file_id=file_id,
            filename=session['filename'],
            size=total_size,
            upload_url=f"/file/{file_id}"
        )
        
    except APIError as ae:
        log_error(f"APIError completing upload: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error completing upload: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.FILE_UPLOAD_FAILED,
            "Failed to complete upload",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))

@router.delete("/cancel/{upload_id}")
async def cancel_upload(
    upload_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel an upload session and clean up temporary files.
    
    Args:
        upload_id: Upload session identifier
        
    Returns:
        Cancellation confirmation
    """
    try:
        session = UPLOAD_SESSIONS.get(upload_id)
        
        if not session:
            raise handle_api_error(not_found_error("Upload session", upload_id))
        
        if session['user_id'] != current_user.userId:
            raise handle_api_error(APIError(
                ErrorCode.INSUFFICIENT_PERMISSIONS,
                "Access denied to upload session",
                status.HTTP_403_FORBIDDEN
            ))
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(session['temp_dir'], ignore_errors=True)
        del UPLOAD_SESSIONS[upload_id]
        
        log_response(200, f"Upload cancelled: {upload_id}", f"/upload/cancel/{upload_id}")
        
        return {
            "success": True,
            "message": "Upload cancelled successfully",
            "upload_id": upload_id
        }
        
    except APIError as ae:
        log_error(f"APIError cancelling upload: {ae.message}")
        raise handle_api_error(ae)
    except Exception as e:
        log_error(f"Error cancelling upload: {str(e)}")
        raise handle_api_error(APIError(
            ErrorCode.SYSTEM_1701,
            "Failed to cancel upload",
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ))
