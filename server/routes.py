"""API Routes Module for AI-Powered Backend Platform.

This module defines all the API endpoints for the platform, including:
- User authentication and management
- File operations (upload, download, listing, deletion)
- AI query handling and conversation management
- Device management and tracking
- Twilio integration for SMS/voice
- User profile and settings

All endpoints are protected with JWT authentication unless explicitly marked as public.
"""

import os
import re
import logging
import json
import uuid
import mimetypes
import base64
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote

# Configure logging
logger = logging.getLogger(__name__)

# Local imports
from server.utils.logging_utils import (
    log_request_start, 
    log_response,
    log_error,
    log_something,
    log_ai_call,
    log_ai_response,
    log_request_payload,
    logger
)

# AI Components - using local package
import sys
import os
from pathlib import Path

from twilio.twiml.voice_response import VoiceResponse


# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)


from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
from aiagent.handler.query import query_openai, summarize_conversation, update_memory
    
# Create a simple wrapper class for the query handler
class AIQueryHandler:
    def __init__(self):
        self.query_openai = query_openai
        self.summarize_conversation = summarize_conversation
        self.update_memory = update_memory

ai_query_handler = AIQueryHandler()
logger.info("AI Query Handler initialized successfully")
    


from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    UploadFile as FastAPIFile, 
    File, 
    Request, 
    Form, 
    status,
    Path,
    Body,
    Header,
    Depends
)
from fastapi import UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator, HttpUrl, field_validator
from typing import Dict, List, Optional, Any, Union

# Application imports
from . import services
from server.utils import compute_sha256
from .db import User, File as DBFile, Query, Device, Session, SessionLocal
from pathlib import Path
import os
from .auth import (
    get_db, 
    get_password_hash, 
    authenticate_user, 
    create_access_token,
    get_current_user, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password
)

# Import schemas
from .schemas.health import HealthCheckResponse
from pydantic import BaseModel
from typing import Optional


class LoginRequest(BaseModel):
    username: str
    password: str
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    device_type: Optional[str] = None
from .schemas import (
    RegisterRequest, 
    RegisterResponse, 
    LoginRequest, 
    TokenResponse,
    UserResponse,
    FileUploadResponse,
    QueryRequest,
    QueryResponse,
    DeviceHeartbeatRequest,
    DeviceInfo,
    DeviceListResponse,
    MessageRequest,
    MessageResponse,
    IncomingMessage,
    CallResponse,
    TranscriptionResponse,
    FileInfo,
    FileListResponse,
    FileUpdateRequest,
    FileType
)

# Initialize router
router = APIRouter(
    prefix="",
    tags=["api"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid authentication"},
        status.HTTP_403_FORBIDDEN: {"description": "Not enough permissions"},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found"},
    },
)

# Start the scheduler when the module loads
services.start_scheduler()

# ===========================================
# API Endpoints
# ===========================================



@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the API is running and healthy.",
    tags=["system"]
)
async def health_check() -> HealthCheckResponse:
    """Check the health status of the API.
    
    This endpoint performs a basic health check of the API and its dependencies.
    
    Returns:
        HealthCheckResponse: The health check response containing status, timestamp, and version.
    """
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0"
    )


@router.post(
    "/register", 
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with the provided credentials.",
    tags=["auth"],
    responses={
        400: {"description": "Username or phone number already exists"},
        422: {"description": "Validation error in request data"}
    }
)
async def register(
    req: RegisterRequest, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Register a new user in the system.
    
    Creates a new user account with the provided username, password, and optional phone number.
    The password will be hashed before storage.
    
    Args:
        req: The registration request containing user details.
        db: Database session dependency.
        
    Returns:
        Dict[str, Any]: Registration confirmation with user ID and username.
        
    Raises:
        HTTPException: 400 if username or phone number already exists.
    """
    # Check if username already exists
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if phone number already exists (if provided)
    if req.phone_number and db.query(User).filter(User.phone_number == req.phone_number).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone number already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(req.password)
    db_user = User(
        username=req.username,
        hashed_password=hashed_password,
        phone_number=req.phone_number
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {
        "user_id": db_user.userId,
        "username": db_user.username,
        "message": "User registered successfully"
    }



@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate a user and retrieve an access token.",
    tags=["auth"],
    responses={
        400: {"description": "Incorrect username or password"},
        422: {"description": "Validation error in request data"}
    }
)
async def login(
    request: Request, 
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    logger.info(f"[LOGIN] Login attempt with device_id={login_data.device_id}, device_name={login_data.device_name}, device_type={login_data.device_type}")
    logger.info(f"[LOGIN] Request URL: {request.url}")
    logger.info(f"[LOGIN] Request headers: {dict(request.headers)}")
    
    try:
        username = login_data.username
        password = login_data.password
        device_name = login_data.device_name
        device_type = login_data.device_type
        device_uuid = login_data.device_id  # client-provided stable id
        
        logger.info(f"[LOGIN] Parsed credentials - username: {username}, device_uuid: {device_uuid}")
        
        if not username or not password:
            error_msg = "Username and password are required"
            logger.error(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Missing credentials",
                    "message": error_msg,
                    "hint": "Please provide both username and password"
                }
            )
            
        logger.info(f"[LOGIN] Attempting to authenticate user: {username}")
        
        # Get user from database
        user = db.query(User).filter(User.username == username).first()
        if not user:
            error_msg = f"User not found: {username}"
            logger.warning(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Authentication failed",
                    "message": "Incorrect username or password",
                    "hint": "Please check your credentials and try again"
                }
            )
            
        logger.info(f"[LOGIN] Found user in database: {user.userId}")
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            error_msg = f"Invalid password for user: {username}"
            logger.warning(f"[LOGIN] {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Authentication failed",
                    "message": "Incorrect username or password",
                    "hint": "Please check your credentials and try again"
                }
            )
            
        logger.info(f"[LOGIN] Password verified for user: {user.userId}")
        
        # Generate access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"[LOGIN] Generated access token for user: {user.userId}")
        
        # Update or create device record
        if device_uuid:
            try:
                # Try to validate the device_uuid if it's provided
                uuid.UUID(device_uuid)
                
                device = db.query(Device).filter(
                    Device.userId == user.userId,
                    Device.device_uuid == device_uuid
                ).first()
                
                now = datetime.utcnow()
                
                if device:
                    # Update existing device
                    device.last_seen = now
                    device.online = True
                    if device_name:
                        device.device_name = device_name
                    if device_type:
                        device.device_type = device_type
                    logger.info(f"[LOGIN] Updated existing device: {device.deviceId} - {device.device_name}")
                else:
                    # Create new device
                    device = Device(
                        userId=user.userId,
                        device_uuid=device_uuid,
                        device_name=device_name or "Unknown Device",
                        device_type=device_type or "unknown",
                        last_seen=now,
                        online=True
                    )
                    db.add(device)
                    logger.info(f"[LOGIN] Created new device: {device_uuid} - {device_name or 'Unnamed Device'}")
                
                db.commit()
                logger.info(f"[LOGIN] Device record {'created' if not device.deviceId else 'updated'} successfully")
                
            except ValueError as e:
                logger.error(f"[LOGIN] Invalid device UUID format: {device_uuid} - {str(e)}")
                # Don't fail the login, just log the error
                device = None
        else:
            logger.warning("[LOGIN] No device_id provided in login request")
            device = None
        
        # Get user role (default to 'user' if not set) and ensure it's a string
        user_role = str(getattr(user, 'role', 'user'))
        
        # Prepare response according to TokenResponse model
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(access_token_expires.total_seconds()),
            "user_id": user.userId,
            "username": user.username,
            "role": user_role,  # This is now guaranteed to be a string
            # These fields are not in the TokenResponse model but we'll keep them
            # for backward compatibility
            "device_id": device.device_uuid if device else None,
            "device_name": device.device_name if device else None
        }
        
        logger.info(f"[LOGIN] Login successful for user: {user.userId}")
        return response_data
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions as they're already properly formatted
        logger.error(f"[LOGIN] HTTP Exception: {str(http_exc)}")
        raise http_exc
        
    except Exception as e:
        # Log any unexpected errors
        error_msg = f"Unexpected error during login: {str(e)}"
        logger.error(f"[LOGIN] {error_msg}", exc_info=True)
        
        # Return a 500 error with detailed information in development
        import traceback
        error_detail = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during login",
            "detail": str(e),
            "traceback": traceback.format_exc() if os.getenv("ENV") == "development" else None
        }
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )


class FileUploadRequest(BaseModel):
    filename: str
    content_type: str
    content: str  # Encoded file content
    size: int
    file_hash: Optional[str] = None
    is_base64_encoded: bool = True

@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a file",
    description="Upload a file to the server",
    tags=["files"],
    responses={
        400: {"description": "Invalid file or upload failed"},
        401: {"description": "Not authenticated"},
        413: {"description": "File too large"}
    }
)
async def upload_file(
    file_data: FileUploadRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    logger.info(f"[UPLOAD] Starting file upload for user {user.userId}")
    logger.debug(f"[UPLOAD] File info - Name: {file_data.filename}, Size: {file_data.size} bytes")
    logger.debug(f"[UPLOAD] Content type: {file_data.content_type}")
    
    try:
        # Check if file with same hash already exists for this user (if provided)
        if file_data.file_hash:
            existing_file = db.query(DBFile).filter(
                DBFile.userId == user.userId,
                DBFile.file_hash == file_data.file_hash
            ).first()
            
            if existing_file:
                # Update the existing file's metadata
                existing_file.filename = file_data.filename
                existing_file.content_type = file_data.content_type
                existing_file.size = file_data.size
                existing_file.last_modified = datetime.utcnow()
                db.commit()
                
                return {
                    "message": "File already exists with same content",
                    "file_id": existing_file.fileId,
                    "filename": existing_file.filename,
                    "size": existing_file.size,
                    "hash": existing_file.file_hash
                }
        
        # Log database connection status
        # Verify database connectivity; if unavailable, fall back to filesystem storage
        db_available = True
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1")).scalar()
            logger.debug("[UPLOAD] Database connection verified")
        except Exception as db_err:
            db_available = False
            logger.warning(f"[UPLOAD] Database unavailable, falling back to file storage: {db_err}")
        
        # Decode content (base64 for binary files, raw for text)
        logger.debug("[UPLOAD] Processing uploaded content")
        decode_start = time.time()
        try:
            if file_data.is_base64_encoded:
                contents = base64.b64decode(file_data.content)
            else:
                contents = file_data.content.encode('utf-8')
            decode_time = time.time() - decode_start
            logger.debug(f"[UPLOAD] Content processed in {decode_time:.2f}s, size: {len(contents)} bytes")
        except Exception as decode_err:
            logger.error(f"[UPLOAD] Failed to process content: {str(decode_err)}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid file content encoding") from decode_err

        file_size = len(contents)
        logger.debug(f"[UPLOAD] Processing file of size: {file_size} bytes")
        
        # Compute file hash
        hash_start = time.time()
        try:
            file_hash = compute_sha256(contents)
            hash_time = time.time() - hash_start
            logger.debug(f"[UPLOAD] File hash computed in {hash_time:.2f}s: {file_hash}")
        except Exception as hash_err:
            logger.error(f"[UPLOAD] Failed to compute file hash: {str(hash_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process file") from hash_err

        # Check file size limit
        max_file_size = user.max_file_size or 524_288_000  # 500 MB default
        logger.debug(f"[UPLOAD] Max allowed file size: {max_file_size} bytes")
        
        if file_size > max_file_size:
            error_msg = f"File size {file_size} exceeds maximum allowed size {max_file_size}"
            logger.warning(f"[UPLOAD] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Check for existing file
        logger.debug(f"[UPLOAD] Checking for existing file with name: {file_data.filename}")
        db_query_start = time.time()
        try:
            existing = db.query(DBFile).filter(
                DBFile.userId == user.userId,
                DBFile.filename == file_data.filename
            ).first()
            db_query_time = time.time() - db_query_start
            logger.debug(f"[UPLOAD] Database query completed in {db_query_time:.2f}s")
            if existing:
                logger.debug(f"[UPLOAD] Found existing file with ID: {existing.fileId}")
            else:
                logger.debug("[UPLOAD] No existing file found with this name")
        except Exception as query_err:
            logger.error(f"[UPLOAD] Database query failed: {str(query_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Database error") from query_err

        content_type = mimetypes.guess_type(file_data.filename)[0] or "application/octet-stream"

        if db_available and existing:
            logger.debug("[UPLOAD] Handling existing file update in database")
            if existing.file_hash == file_hash:
                return {
                    "message": "File already exists",
                    "file_id": existing.fileId,
                    "filename": existing.filename,
                    "size": existing.size,
                    "hash": existing.file_hash
                }
            else:
                existing.content = contents
                existing.size = file_size
                existing.file_hash = file_hash
                existing.uploaded_at = datetime.utcnow()
                existing.content_type = content_type
                db.commit()
                return {
                    "message": "File updated",
                    "file_id": existing.fileId,
                    "filename": existing.filename,
                    "size": existing.size,
                    "hash": existing.file_hash
                }

        # New file or fallback storage
        if db_available:
            logger.debug("[UPLOAD] Creating new file record in database")
            try:
                new_file = DBFile(
                    filename=file_data.filename,
                    userId=user.userId,
                    size=file_size,
                    content=contents,
                    file_hash=file_hash,
                    content_type=content_type,
                    uploaded_at=datetime.utcnow()
                )
                db_start = time.time()
                db.add(new_file)
                db.commit()
                db.refresh(new_file)
                db_time = time.time() - db_start
                logger.debug(f"[UPLOAD] Database operation completed in {db_time:.2f}s, new file ID: {new_file.fileId}")
                return {
                    "message": "File uploaded",
                    "file_id": new_file.fileId,
                    "filename": new_file.filename,
                    "size": new_file.size,
                    "hash": new_file.file_hash
                }
            except Exception as db_err:
                logger.error(f"[UPLOAD] Failed to save file to database: {db_err}", exc_info=True)
                db.rollback()
                # Fall back to filesystem storage
                db_available = False
                logger.warning("[UPLOAD] Falling back to file storage due to DB save failure")

        # Filesystem fallback storage
        storage_root = Path(os.getenv('FILE_STORAGE_DIR', 'uploaded_files'))
        storage_root.mkdir(parents=True, exist_ok=True)
        dest = storage_root / f"{user.userId}_{file_data.filename}"
        try:
            with open(dest, 'wb') as f:
                f.write(contents)
            logger.info(f"[UPLOAD] File stored on filesystem at {dest}")
        except Exception as fs_err:
            logger.error(f"[UPLOAD] Filesystem write failed: {fs_err}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to store file") from fs_err
        return {
            "message": "File uploaded (filesystem fallback)",
            "filename": file_data.filename,
            "size": file_size,
            "hash": file_hash,
            "path": str(dest)
        }

    except Exception as e:
        db.rollback()
        log_error(f"Upload failed: {str(e)}", e, endpoint="/upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process AI Query",
    description="""
    Process a user query using the AI model and return a response.
    Maintains conversation context using chat_id for multi-turn conversations.
    """,
    tags=["AI"],
    responses={
        400: {"description": "Invalid request parameters"},
        401: {"description": "Not authenticated"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "AI service error"}
    }
)
async def query_endpoint(
    query_data: QueryRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Process an AI query from a user.
    
    Sends the query to the OpenAI API and returns the response. The query and response
    are associated with a chat ID for maintaining conversation context.
    All queries and responses are stored in the database for future reference.
    
    Args:
        query_data: The query parameters including the user's message
        request: The HTTP request object
        user: The authenticated user making the query
        db: Database session dependency
        
    Returns:
        Dict containing the AI's response and metadata
        
    Raises:
        HTTPException: If there's an error processing the query
    """
    
    # Try to parse the JSON body
    try:
        body = await request.json()
        if not body:
            return JSONResponse({"error": "Empty request body"}, status_code=400)
    except json.JSONDecodeError as json_err:
        return JSONResponse({"error": f"Invalid JSON in request body: {str(json_err)}"}, status_code=400)
        

    try:
        # Check for required fields
        if not body.get("query"):
            return JSONResponse(status_code=400, content={"error": "No query provided"})
            
        # Check for chat_id field - Make sure this runs BEFORE the try-except for the AI agent
        # Handle both camelCase (chatId) and snake_case (chat_id) formats for compatibility
        chat_id = body.get("chat_id") or body.get("chatId")
        if chat_id is None or chat_id == "":
            return JSONResponse(status_code=400, content={"error": "No chat ID provided"})
        
        user_query = body.get("query")
        # check pageContent field
        page_content = body.get("pageContent")
        if page_content :
            user_query += "\n\n" + "Here is the page content: " + page_content

        
        # Create a new query record in the database (without response yet)
        db_query = Query(
            userId=user.userId,
            chatId=chat_id,
            query_text=user_query
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        
        # Helper function to get or create memory file
        def get_or_create_memory_file(filename, default_content='{}'):
            file = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == filename).first()
            if not file:
                # Create the file if it doesn't exist
                file = DBFile(
                    userId=user.userId,
                    filename=filename,
                    content=default_content.encode('utf-8'),
                    content_type='application/json',
                    size=len(default_content)
                )
                db.add(file)
                db.commit()
                db.refresh(file)
                logger.info(f"Created new memory file: {filename} for user {user.userId}")
            return file
            
        # Get or create memory files
        try:
            # Long-term memory
            longterm_memory_file = get_or_create_memory_file("long_term_memory.json")
            longterm_content_str = longterm_memory_file.content.decode('utf-8') if longterm_memory_file.content else "{}"
            
            # Short-term memory
            shortterm_memory_file = get_or_create_memory_file("short_term_memory.json")
            shortterm_content_str = shortterm_memory_file.content.decode('utf-8') if shortterm_memory_file.content else "{}"
            
            # Parse JSON content
            longterm_memory_data = json.loads(longterm_content_str) if longterm_content_str.strip() else {}
            shortterm_memory_data = json.loads(shortterm_content_str) if shortterm_content_str.strip() else {}
            
            # Initialize memory managers
            long_term_memory = LongTermMemoryManager(memory_content=longterm_memory_data)
            short_term_memory = ShortTermMemoryManager(memory_content=shortterm_memory_data)
            
        except Exception as e:
            logger.error(f"Error initializing memory managers for user {user.userId}: {str(e)}")
            # Fallback to empty memories
            long_term_memory = LongTermMemoryManager()
            short_term_memory = ShortTermMemoryManager()
            logger.info("Initialized empty memory managers due to error")

        # Include user markdown notes in the query context
        try:
            notes_files = (
                db.query(DBFile)
                .filter(DBFile.userId == user.userId, DBFile.filename.ilike('%.md'))
                .limit(50)
                .all()
            )
            notes_parts = []
            for nf in notes_files:
                try:
                    note_text = (nf.content or b'').decode('utf-8', errors='ignore')
                except Exception:
                    note_text = ''
                notes_parts.append(f"<BEGIN NOTE>{nf.filename}:{note_text}<END NOTE>")
            if notes_parts:
                concatenated_notes = "_".join(notes_parts)
                user_query += "\n\nThese are the notes of the user. Notes are separated by _ and each note is in the format <BEGIN NOTE>filename:note<END NOTE> ::" 
                user_query += concatenated_notes
        except Exception as e:
            logger.error(f"Error loading user notes for query: {e}")

        # Call your AI agent with try/except to handle Vercel environment limitations
        try:
            try:
                
                logger.info(f"Sending query to AI: {user_query[:100]}...")
                # Send query to AI agent using query_openai instead of ask_ai
                response = query_openai(
                    query=user_query,
                    long_term_memory=long_term_memory,
                    short_term_memory=short_term_memory,
                    max_tokens=10000,  # Default max tokens
                    temperature=0.7,  # Default temperature
                    aux_data={"current_user_id": user.userId}
                )
                logger.info("Successfully received response from AI")
            except Exception as e:
                logger.error(f"Error during AI query processing: {str(e)}", exc_info=True)
                raise
            
            # If the query was successful, update the memory
            if not response.startswith("Error:"):
                # Update shortterm memory
                conversations = shortterm_memory_data.get("conversations", [])
                
                summary = ai_query_handler.summarize_conversation(user_query, response)
                updated = ai_query_handler.update_memory(user_query, response, long_term_memory) 
                
                # Update conversations
                conversations += [{
                    "query": user_query, 
                    "response": response, 
                    "summary": summary
                }]
                # limit conversations to 50
                if len(conversations) > 50:
                    conversations = conversations[-50:]
                shortterm_memory_data["conversations"] = conversations

                # save updated longterm memory
                if updated : 
                    log_something("Updated longterm memory:"+str(long_term_memory.get_content()), "queryEndpoint")
                    if longterm_memory_file:
                        updated_content = long_term_memory.get_content()
                        longterm_memory_file.content = json.dumps(updated_content).encode('utf-8')
                        longterm_memory_file.size = len(longterm_memory_file.content)
                        # db.add(longterm_memory_file)
                        db.commit()
                    else:

                        # long_term_memory.json didn't exist for this user, create it now
                        new_longterm_file = DBFile(
                            filename="long_term_memory.json",
                            userId=user.userId,
                            content=json.dumps(long_term_memory.get_content()).encode('utf-8'),
                            content_type="application/json"
                        )
                        new_longterm_file.size = len(new_longterm_file.content)
                        db.add(new_longterm_file)
                        db.commit()
                else :
                    log_something("Longterm memory not updated", "queryEndpoint")
                
                # Save updated shortterm memory back to the database
                if shortterm_memory_file:
                    shortterm_memory_file.content = json.dumps(shortterm_memory_data).encode('utf-8')
                    shortterm_memory_file.size = len(shortterm_memory_file.content)
                else:
                    # short_term_memory.json didn't exist for this user, create it now
                    new_shortterm_file = DBFile(
                        filename="short_term_memory.json",
                        userId=user.userId,
                        content=json.dumps(shortterm_memory_data).encode('utf-8'),
                        content_type="application/json"
                    )
                    new_shortterm_file.size = len(new_shortterm_file.content)
                    db.add(new_shortterm_file)
                db.commit()

        except FileNotFoundError as file_err:
            # Handle missing files in Vercel environment
            log_error(f"AI agent file not found: {str(file_err)}", exc=file_err, endpoint="/query")
            raise file_err
        
        # Log the response
        log_ai_response(response, "/query")
        
        # Update the query record with the response
        db_query.response = response
        db.commit()
        
        # Get the model name from the environment or use a default
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Estimate tokens used (this is a rough estimate since we don't have exact token count)
        tokens_used = len(response.split()) + len(user_query.split())
        
        return {
            "response": response,
            "query": user_query,
            "chat_id": chat_id,
            "queryId": db_query.queryId,
            "tokens_used": tokens_used,
            "model": model_name
        }
    except Exception as e:
        db.rollback()  # Rollback transaction on error
        error_msg = f"AI query failed: {str(e)}. Type: {type(e).__name__}. Args: {e.args}"
        log_error(error_msg, exc=e, endpoint="/query")
        
        # Include more detailed error information in development
        detail_msg = f"Error processing query: {str(e)}"
        if not os.environ.get("PRODUCTION"):
            detail_msg += f"\nType: {type(e).__name__}\nArgs: {e.args}"
            import traceback
            detail_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            
        # Always return JSON response
        return JSONResponse(
            status_code=500,
            content={
                "error": detail_msg,
                "type": type(e).__name__,
                "args": str(e.args)
            }
        )
        


@router.post("/active")
async def update_active_item(
    request: Request, 
    user: User = Depends(get_current_user),
    db: SessionLocal = Depends(get_db)
):
    """Update or add an active item for the authenticated user.
    
    Records the device, path, and title of the active item being viewed.
    If an entry with the same device exists, it will be updated.
    
    Args:
        request: The HTTP request containing the active item data
        user: The authenticated user
        db: Database session
        
    Returns:
        dict: Success status and updated active items
        
    Raises:
        HTTPException: 400 if required fields are missing
        HTTPException: 404 if user's short-term memory is not found
    """
    try:
        # Log request start
        log_request_start('/active', request.method, dict(request.headers), request.client.host if request.client else None)
        
        # Parse request body
        try:
            data = await request.json()
            log_request_payload(data, '/active')
        except json.JSONDecodeError as json_err:
            log_error(f"Invalid JSON: {str(json_err)}", json_err, {"endpoint": "/active"}, "/active")
            return JSONResponse({"error": f"Invalid JSON in request body: {str(json_err)}"}, status_code=400)
            
        # Validate required fields
        device = data.get('device')
        if not device:
            log_error("Missing device identifier", None, {"endpoint": "/active"}, "/active")
            return JSONResponse({"error": "Missing device identifier"}, status_code=400)

        # Accept multiple shapes
        # Browser: title/url/content
        # Desktop/Mobile: app/content
        kind = 'browser' if ('url' in data or 'title' in data) else 'app'
        title = data.get('title', '') if kind == 'browser' else data.get('app', '')
        url = data.get('url', '') if kind == 'browser' else ''
        content = data.get('content', '') or ''
        focused = bool(data.get('focused', True))
        path = data.get('path', url)  # keep backward compat; prefer url as path when browser
        # Truncate large content
        if isinstance(content, str) and len(content) > 8000:
            content = content[:8000]
            
        # Get user's short-term memory file
        stm_file = db.query(DBFile).filter(
            DBFile.userId == user.userId,
            DBFile.filename == "short_term_memory.json"
        ).first()
        
        if not stm_file:
            log_error("Short-term memory not found", None, {"userId": user.userId}, "/active")
            raise HTTPException(status_code=404, detail="Short-term memory not found")
            
        # Parse existing memory
        try:
            memory = json.loads(stm_file.content.decode('utf-8'))
        except json.JSONDecodeError as e:
            log_error(f"Invalid short-term memory format: {str(e)}", e, {"userId": user.userId}, "/active")
            raise HTTPException(status_code=500, detail="Invalid short-term memory format")
            
        # Initialize structures if missing (use only active_history to be consistent)
        if 'active_history' not in memory or not isinstance(memory['active_history'], dict):
            memory['active_history'] = {}

        # Create new active item
        new_item = {
            'device': device,
            'type': kind,
            'path': path,
            'title': title,
            'url': url,
            'content_preview': (content[:200] if isinstance(content, str) else ''),
            'focused': focused,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Update per-device history (most recent first, cap 20 beyond current)
        history = memory['active_history'].get(device, [])
        history.insert(0, new_item)
        # Deduplicate consecutive identical URLs/titles to avoid noise
        deduped = []
        seen = set()
        for h in history:
            key = (h.get('title', ''), h.get('url', ''), h.get('path', ''))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
            if len(deduped) >= 21:
                break
        memory['active_history'][device] = deduped

        # Persist STM changes
        stm_file.content = json.dumps(memory).encode('utf-8')
        stm_file.size = len(stm_file.content)
        db.commit()

        # If not focused, do not trigger AI/SMS; just persist STM and return
        if not focused:
            response = {
                "status": "success",
                "active_history": memory['active_history'].get(device, []),
                "chat_id": f"active:{device}",
                "queryId": None,
                "response": "",
                "show_notification": False,
            }
            log_response(200, response, '/active')
            return JSONResponse(response, status_code=200)

        # Build custom prompt referencing current and up to 20 previous items
        prev_items = deduped[1:21] if len(deduped) > 1 else []
        prev_lines = []
        for it in prev_items:
            if it.get('type') == 'browser':
                prev_lines.append(f"- {it.get('title','')} | {it.get('url','')} @ {it.get('timestamp','')}")
            else:
                prev_lines.append(f"- {it.get('title','')} (app) @ {it.get('timestamp','')}")
        prev_block = "\n".join(prev_lines) if prev_lines else "(no previous items)"

        current_line = (
            f"Title: {new_item.get('title','')}\n"
            f"URL/Path: {new_item.get('url') or new_item.get('path','')}\n"
            f"Type: {new_item.get('type','')}\n"
            f"When: {new_item.get('timestamp','')}\n"
        )
        if content:
            current_line += f"Content (may be truncated):\n{content}"

        instruction = (
            "You are assisting the user across devices. The user's phone number is stored in long-term memory. "
            "If the current website indicates the user is a youtube video, send an SMS informing the user that he is watching a video. "
            "If the current website indicates the user is an academic article, send an SMS informing the user that he is reading an academic article. "
        )

        user_query = (
            f"{instruction}\n\n"
            f"Current Item:\n{current_line}\n\n"
            f"Previous (up to 20):\n{prev_block}"
        )

        # Prepare memory managers from DB files (similar to query_endpoint)
        def get_or_create_memory_file(filename, default_content='{}'):
            file = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == filename).first()
            if not file:
                file = DBFile(
                    userId=user.userId,
                    filename=filename,
                    content=default_content.encode('utf-8'),
                    content_type='application/json',
                    size=len(default_content),
                )
                db.add(file)
                db.commit()
                db.refresh(file)
            return file

        try:
            longterm_file = get_or_create_memory_file("long_term_memory.json")
            shortterm_file = get_or_create_memory_file("short_term_memory.json")
            lt_str = longterm_file.content.decode('utf-8') if longterm_file.content else "{}"
            st_str = shortterm_file.content.decode('utf-8') if shortterm_file.content else "{}"
            lt_data = json.loads(lt_str) if lt_str.strip() else {}
            st_data = json.loads(st_str) if st_str.strip() else {}
            long_term_memory = LongTermMemoryManager(memory_content=lt_data)
            short_term_memory = ShortTermMemoryManager(memory_content=st_data)
        except Exception as e:
            logger.error(f"[ACTIVE] Memory init error for user {user.userId}: {e}")
            long_term_memory = LongTermMemoryManager()
            short_term_memory = ShortTermMemoryManager()
            st_data = {}

        # Persist the prompt as a Query row and call AI (mirrors query_endpoint flow)
        chat_id = f"active:{device}"
        db_query = Query(userId=user.userId, chatId=chat_id, query_text=user_query)
        db.add(db_query)
        db.commit()
        db.refresh(db_query)

        ai_response = query_openai(
            query=user_query,
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            max_tokens=2000,
            temperature=0.5,
            aux_data={"current_user_id": user.userId},
        )

        # Update conversations in STM file
        try:
            conversations = st_data.get("conversations", []) if isinstance(st_data, dict) else []
            summary = ai_query_handler.summarize_conversation(user_query, ai_response)
            updated_lt = ai_query_handler.update_memory(user_query, ai_response, long_term_memory)
            conversations.append({"query": user_query, "response": ai_response, "summary": summary})
            if len(conversations) > 50:
                conversations = conversations[-50:]
            if isinstance(st_data, dict):
                st_data["conversations"] = conversations
                shortterm_file.content = json.dumps(st_data).encode('utf-8')
                shortterm_file.size = len(shortterm_file.content)
                db.commit()
            if updated_lt and longterm_file:
                longterm_file.content = json.dumps(long_term_memory.get_content()).encode('utf-8')
                longterm_file.size = len(longterm_file.content)
                db.commit()
        except Exception as e:
            logger.error(f"[ACTIVE] Failed to update conversations/memory: {e}")

        show_notification = bool(ai_response and isinstance(ai_response, str) and ai_response.strip())
        empty_response = ""
        show_notification = False
        response = {
            "status": "success",
            "active_history": memory['active_history'].get(device, []),
            "chat_id": chat_id,
            "queryId": db_query.queryId,
            "response": empty_response, 
            "show_notification": show_notification,
        }
        log_response(200, response, '/active')
        return JSONResponse(response, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(str(e), e, {"endpoint": "/active"}, "/active")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/files")
def list_files(user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """List all files uploaded by the authenticated user.
    
    Retrieves files from the database instead of checking the filesystem directly.
    
    Args:
        user: The authenticated user whose files to list
        db: Database session dependency
        
    Returns:
        dict: List of files with their metadata
    """
    try:
        # Query the database for files owned by this user
        files = db.query(DBFile).filter(DBFile.userId == user.userId).all()
        
        # Format the response with file metadata
        file_list = [
            {
                "fileId": file.fileId,
                "filename": file.filename,
                "size": file.size,
                "uploaded_at": file.uploaded_at.isoformat() if file.uploaded_at else None,
                "file_hash": file.file_hash,
                "last_modified": file.last_modified.isoformat() if file.last_modified else None
            } for file in files
        ]
        
        return {"files": file_list, "count": len(file_list)}
    except Exception as e:
        log_error(f"Error listing files: {str(e)}", exc=e, endpoint="/files")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.get("/download/{fileId}")
def download_file(fileId: int, user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Download a specific file by its ID.
    
    Retrieves the file record from the database before accessing the filesystem.
    
    Args:
        fileId: The ID of the file to download
        user: The authenticated user requesting the download
        db: Database session dependency
        
    Returns:
        FileResponse: The file content as a download
        
    Raises:
        HTTPException: 404 if file not found
        HTTPException: 403 if trying to access another user's file
    """
    try:
        # Query the database for the file record
        file_record = db.query(DBFile).filter(DBFile.fileId == fileId).first()
        
        # Check if file exists and belongs to the user
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
            
        if file_record.userId != user.userId:
            raise HTTPException(status_code=403, detail="You don't have permission to access this file")
        
        # First try to get content directly from the database
        if file_record.content is not None:
            file_content = file_record.content
            content_type = file_record.content_type or "application/octet-stream"
        else:
            # Fall back to the filesystem if content is not in the database
            filepath = file_record.path
            if filepath and os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    file_content = f.read()
                content_type = file_record.content_type or "application/octet-stream"
            else:
                raise HTTPException(status_code=404, detail="File content not found")
        
        # Guess content type from filename if not set
        if not content_type or content_type == "application/octet-stream":
            import mimetypes
            content_type = mimetypes.guess_type(file_record.filename)[0] or "application/octet-stream"
        
        # Stream the file from memory
        def iterfile():
            yield file_content
                
        return StreamingResponse(
            iterfile(), 
            media_type=content_type, 
            headers={"Content-Disposition": f"attachment; filename={quote(file_record.filename)}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error downloading file: {str(e)}", exc=e, endpoint=f"/download/{fileId}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@router.delete("/delete/{fileId}")
def delete_file(fileId: int, user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    """Delete a specific file by its ID.
    
    Deletes both the database record and the file on disk.
    
    Args:
        fileId: The ID of the file to delete
        user: The authenticated user requesting the deletion
        db: Database session dependency
        
    Returns:
        dict: Confirmation message
        
    Raises:
        HTTPException: 404 if file not found
        HTTPException: 403 if trying to delete another user's file
    """
    try:
        # Query the database for the file record
        file_record = db.query(DBFile).filter(DBFile.fileId == fileId).first()
        
        # Check if file exists and belongs to the user
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
            
        if file_record.userId != user.userId:
            raise HTTPException(status_code=403, detail="You don't have permission to delete this file")
        

        # Delete the database record
        db.delete(file_record)
        db.commit()
        
        return {"message": f"File '{filename}' deleted successfully.", "fileId": fileId}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting file: {str(e)}", exc=e, endpoint=f"/delete/{fileId}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get('/profile', response_model=UserResponse)
def profile(current_user: User = Depends(get_current_user)):
    """Get profile information for the currently authenticated user.
    
    Returns:
        UserResponse: The authenticated user's profile data, including their role and phone number.
    """
    # The current_user object (SQLAlchemy model) will be automatically
    # converted to a UserResponse Pydantic model by FastAPI.
    return UserResponse(
        userId=current_user.userId,
        username=current_user.username,
        max_file_size=current_user.max_file_size,
        role=current_user.role,
        phone_number=current_user.phone_number
    )



# --- Device Models ---

class DeviceBase(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    
    @validator('device_id')
    def validate_device_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Device ID is required")
        return v.strip()

class DeviceHeartbeatRequest(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    current_app: Optional[str] = None
    current_page: Optional[str] = None
    current_url: Optional[str] = None
    focused: Optional[bool] = None
    
    @validator('device_id')
    def validate_device_id(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # If not a UUID, check if it's a Chrome extension ID (alphanumeric 32 chars)
            if len(v) == 32 and v.isalnum():
                return v
            # Special case for 'chrome-extension://' prefixed IDs
            if v.startswith('chrome-extension://'):
                return v
            raise ValueError('device_id must be a valid UUID or Chrome extension ID')

class DeviceLogoutRequest(DeviceBase):
    pass

# --- Device Endpoints ---

@router.post("/device/heartbeat")
async def device_heartbeat(
    request: Request,
    heartbeat_data: DeviceHeartbeatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Handle device heartbeat with detailed logging and validation.
    
    This endpoint is called periodically by client devices to indicate they are still active.
    It updates the device's last_seen timestamp and can track the current app/page/URL.
    """
    start_time = datetime.utcnow()
    endpoint = "/device/heartbeat"
    logger = logging.getLogger('thoth')
    
    # Log incoming request with all relevant context
    logger.info(
        "[HEARTBEAT] Received from user_id=%s, device_id=%s, app=%s, page=%s",
        user.userId,
        heartbeat_data.device_id,
        heartbeat_data.current_app or 'unknown',
        heartbeat_data.current_page or 'unknown'
    )
    
    try:
        # Extract fields from the request model
        device_id = heartbeat_data.device_id
        device_name = heartbeat_data.device_name
        device_type = heartbeat_data.device_type
        current_app = heartbeat_data.current_app
        current_page = heartbeat_data.current_page
        current_url = heartbeat_data.current_url
        
        # Log detailed request info for debugging
        logger.debug(
            "[HEARTBEAT] Request details - device_name=%s, device_type=%s, url=%s",
            device_name, device_type, current_url
        )
        
        # Resolve device record prioritizing device_uuid
        device = None
        
        # First try exact match (for backward compatibility)
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == device_id
        ).first()
        
        # If not found and device_id starts with 'chrome-', try without the prefix
        if not device and device_id and device_id.startswith('chrome-'):
            clean_device_id = device_id[7:]  # Remove 'chrome-' prefix
            logger.debug("Trying with cleaned device_id: %s", clean_device_id)
            device = db.query(Device).filter(
                Device.userId == user.userId,
                Device.device_uuid == clean_device_id
            ).first()
        
        # If found with cleaned ID, update device_uuid to include the prefix for future requests
        if device:
            logger.info("Found device with cleaned ID, updating to include prefix")
            device.device_uuid = device_id  # Update to include prefix
            db.commit()
            
        # If still not found, create a new device record
        if not device:
            logger.info("Creating new device record for device_id: %s", device_id)
            try:
                device = Device(
                    userId=user.userId,
                    device_uuid=device_id,
                    device_name=device_name or f"Device-{str(uuid.uuid4())[:8]}",
                    device_type=device_type or "unknown",
                    last_seen=start_time,
                    created_at=start_time,
                    updated_at=start_time
                )
                db.add(device)
                db.commit()
                db.refresh(device)
                logger.info("Created new device with ID: %s", device.id)
            except Exception as e:
                db.rollback()
                logger.error("Failed to create device record: %s", str(e), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to create device record",
                        "message": str(e)
                    }
                )
        
        try:
            # Update device info and last_seen timestamp
            logger.debug("Updating device %s last_seen timestamp", device.deviceId)
            device.last_seen = start_time
            if device_name:
                device.device_name = device_name
            if device_type:
                device.device_type = device_type
            
            # Update online status based on last_seen
            time_since_last_seen = (datetime.utcnow() - device.last_seen).total_seconds()
            device.online = time_since_last_seen < 300  # 5 minutes
            
            db.commit()
            
            # Log the update
            logger.info(
                "[HEARTBEAT] Updated device %s (last_seen=%s, online=%s)",
                device.deviceId,
                device.last_seen.isoformat(),
                device.online
            )
            
            # # Create or update device activity
            # activity = db.query(DeviceActivity).filter(
            #     DeviceActivity.device_id == device.deviceId,
            #     func.date(DeviceActivity.activity_date) == start_time.date()
            # ).first()
            
            # if activity:
            #     activity.activity_count += 1
            #     activity.last_activity = start_time
            #     activity.updated_at = start_time
            # else:
            #     activity = DeviceActivity(
            #         device_id=device.deviceId,
            #         activity_date=start_time.date(),
            #         activity_count=1,
            #         last_activity=start_time,
            #         created_at=start_time,
            #         updated_at=start_time
            #     )
            #     db.add(activity)
            
            # db.commit()
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Log successful heartbeat with performance metrics
            logger.info(
                "Heartbeat processed for device %s (user_id=%s) in %.2f ms",
                device.deviceId,
                user.userId,
                processing_time
            )
            
            return {
                "status": "success",
                "message": "Heartbeat received",
                "device_id": str(device.deviceId),
                "timestamp": start_time.isoformat(),
                "processing_time_ms": round(processing_time, 2)
            }
            
        except Exception as e:
            # Log the error with device context
            logger.error(
                "[HEARTBEAT] Error updating device %s: %s",
                device.deviceId if device and hasattr(device, 'deviceId') else 'unknown',
                str(e),
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to update device activity",
                    "message": str(e)
                }
            )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Unexpected error in device_heartbeat: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }
        )

    # ------------------------------------------------------------------
    # Persist foreground context to the <device_id>_<timestamp>.json file
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{device.deviceId}_{timestamp}.json"
    file_entry = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename == filename
    ).first()

    if file_entry and file_entry.content:
        try:
            data = json.loads(file_entry.content.decode("utf-8"))
        except Exception:
            data = {
                "deviceId": device.deviceId,
                "device_name": device_name,
                "device_type": device_type,
                "events": []
            }
    else:
        data = {
            "deviceId": device.deviceId,
            "device_name": device_name,
            "device_type": device_type,
            "events": []
        }

    # Append the current foreground information as a new event
    data.setdefault("events", []).append({
        "timestamp": now.isoformat(),
        "current_app": current_app,
        "current_page": current_page,
        "current_url": current_url
    })

    updated_bytes = json.dumps(data).encode("utf-8")

    # remove old device files (10 days old)
    old_files = db.query(DBFile).filter(
        DBFile.userId == user.userId,
        DBFile.filename.like(f"{device.deviceId}_%")
    ).filter(
        DBFile.uploaded_at < (now - timedelta(days=10))
    ).all()
    for old_file in old_files:
        db.delete(old_file)



    if file_entry:
        file_entry.content = updated_bytes
        file_entry.size = len(updated_bytes)
        file_entry.file_hash = compute_sha256(updated_bytes)
        file_entry.uploaded_at = now
    else:
        new_file = DBFile(
            filename=filename,
            userId=user.userId,
            size=len(updated_bytes),
            content=updated_bytes,
            file_hash=compute_sha256(updated_bytes),
            content_type="application/json",
            uploaded_at=now
        )
        db.add(new_file)

    db.commit()

    return {
        "message": "Device heartbeat received and state updated",
        "deviceId": device.deviceId,
        "device_uuid": device.device_uuid,
        "last_seen": device.last_seen.isoformat()
    }


@router.post(
    "/device/logout",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Logout device",
    description="Mark a device as logged out",
    tags=["devices"]
)
async def device_logout(
    logout_data: DeviceLogoutRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    device_id = logout_data.device_id
    device_name = logout_data.device_name
    # Resolve device record prioritising device_uuid when supplied
    device = None
    if device_id:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_uuid == device_id
        ).first()

    if not device and device_name:
        device = db.query(Device).filter(
            Device.userId == user.userId,
            Device.device_name == device_name
        ).first()

    if device:
        device.online = False
        device.last_seen = datetime.utcnow()
        logger.info(f"[device_logout] Set device '{device.device_name}' offline")
        db.commit()

    return {
        "status": "logged out",
        "deviceId": device.deviceId if device else None,
        "device_uuid": device.device_uuid if device else None,
    }


@router.get("/devices")
async def list_devices(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List all devices registered for the authenticated user.

    A device is considered **online** if its `last_seen` timestamp is within
    the last five minutes. Otherwise, it's treated as offline.
    """
    try:
        # Log the incoming request and authenticated user for debugging
        logger.debug("[DEVICES] Incoming headers: %s", dict(request.headers))
        logger.info(
            "[DEVICES] Listing devices for user '%s' (userId=%s)",
            user.username,
            user.userId,
        )
        threshold = datetime.utcnow() - timedelta(minutes=5)
        devices = db.query(Device).filter(Device.userId == user.userId).all()

        device_list = []
        for device in devices:
            # Determine online status based on last_seen timestamp rather than the persisted boolean
            online = bool(device.last_seen and device.last_seen >= threshold)
            device_list.append({
                "deviceId": device.deviceId,
                "name": device.device_name,
                "type": device.device_type,
                "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                "online": online,
            })

        return {"devices": device_list, "count": len(device_list)}

    except Exception as e:
        log_error(f"Error listing devices: {str(e)}", exc=e, endpoint="/devices")
        raise HTTPException(status_code=500, detail="Error listing devices")


# --- Twilio Webhook Endpoints ---

@router.post("/api/webhooks/twilio/incoming-message")
async def handle_twilio_incoming_message(
    request: Request, 
    db: Session = Depends(get_db)
):
    """Handle incoming SMS messages from Twilio and forward them to the query endpoint.
    
    Args:
        request: The incoming HTTP request containing Twilio webhook data
        db: Database session
    """
    # Parse the form data from the request
    form_data = await request.form()
    from_number = form_data.get('From', '')
    body = form_data.get('Body', '').strip()
    
    endpoint_name = "/api/webhooks/twilio/incoming-message"
    log_request_start(endpoint_name, "POST", dict(request.headers), request.client.host if request.client else "unknown")
    
    # Log the incoming message
    logger.info(f"[{endpoint_name}] Incoming message from {from_number}: {body}")
    
    try:
        # Normalize phone number (remove non-digits)
        normalized_from = re.sub(r'\D', '', from_number)
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None
        
        if not user:
            logger.warning(f"[{endpoint_name}] Unauthorized access attempt from {from_number}")
            return Response(
                content="<Response><Message>Unauthorized. Please register first.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
        
        # Create a chat ID for this conversation
        chat_id = f"sms_{normalized_from}"
        user_query_text = body
        
        # Log the query
        logger.info(f"[{endpoint_name}] Processing query from user {user.userId}: {user_query_text}")
        
        # Save the query to database
        try:
            db_query = Query(
                userId=user.userId,
                chatId=chat_id,
                query_text=user_query_text,
                response=None  # Will be updated with AI response
            )
            db.add(db_query)
            db.commit()
            db.refresh(db_query)
            logger.info(f"[{endpoint_name}] Successfully created query with ID: {db_query.queryId}")
            log_ai_call(user_query_text, "default_sms_model", endpoint_name)
        except Exception as e:
            logger.error(f"[{endpoint_name}] Error creating query record: {str(e)}", exc_info=True)
            db.rollback()
            raise

        # Load conversation history
        conversation_history = []
        try:
            # Get recent messages for context
            recent_messages = db.query(Query).filter(
                Query.userId == user.userId,
                Query.chatId == chat_id
            ).order_by(Query.created_at.desc()).limit(5).all()
            
            # Reverse to get chronological order
            for msg in reversed(recent_messages):
                if msg.query_text:
                    conversation_history.append({"role": "user", "content": msg.query_text})
                if msg.response:
                    conversation_history.append({"role": "assistant", "content": msg.response})
            
            logger.info(f"[{endpoint_name}] Loaded {len(conversation_history)} messages from history")
        except Exception as e:
            logger.error(f"[{endpoint_name}] Error loading conversation history: {str(e)}")
            conversation_history = []
        
        # Create a request object that will work with the query_endpoint
        class JSONRequest:
            def __init__(self, query_text, chat_id, conversation_history):
                self._json = {
                    "query": query_text,
                    "chat_id": chat_id,
                    "conversation_history": conversation_history,
                    "user_id": user.userId
                }
                self.method = "POST"
                self.headers = {"content-type": "application/json"}
                self.url = "http://localhost/query"  # This is just a placeholder
                
            async def json(self):
                return self._json
                
            def __getattr__(self, name):
                # Default None for any other attributes that might be accessed
                return None
        
        # Create the JSON request
        json_request = JSONRequest(user_query_text, chat_id, conversation_history)
        
        try:
            logger.info(f"[{endpoint_name}] Forwarding query to /query endpoint")
            
            # Call the query endpoint with the JSON request
            ai_response = await query_endpoint(
                query_data=QueryRequest(
                    query=user_query_text,
                    chat_id=chat_id,
                    conversation_history=conversation_history,
                    user_id=user.userId
                ),
                request=json_request,
                user=user,
                db=db
            )
            
            response_text = ai_response.get("response", "Sorry, I couldn't process your request at the moment.")
            logger.info(f"[{endpoint_name}] Received response from /query endpoint")
            
            # Update the query with the response
            db_query.response = response_text
            db.commit()
            
            # Format TwiML response
            twiml_response = f"""
            <Response>
                <Message>{response_text}</Message>
            </Response>
            """
            
            log_response(200, "TwiML reply sent", endpoint_name)
            return Response(
                content=twiml_response.strip(),
                media_type="application/xml",
                status_code=200
            )

        except Exception as e:
            logger.error(f"[{endpoint_name}] Error calling /query endpoint: {str(e)}", exc_info=True)
            db.rollback()
            twiml_error_reply = "<Response><Message>Sorry, an internal error occurred while processing your message.</Message></Response>"
            return Response(content=twiml_error_reply, media_type="application/xml", status_code=500)

    except Exception as e:
        #logger.error(f"[{endpoint_name}] Error processing AI query for SMS: {e}", exc_info=True)
        db.rollback() # Rollback any partial DB changes on error
        twiml_error_reply = "<Response><Message>Sorry, an internal error occurred while processing your message.</Message></Response>"
        return Response(content=twiml_error_reply, media_type="application/xml", status_code=500)


_STATUS_ORDER = {
    "queued": 0,
    "accepted": 1,
    "sending": 2,
    "sent": 3,
    "delivered": 4,
    "undelivered": 5,
    "failed": 6,
}

# Best-effort in-memory caches (per-process). For full durability, persist to DB.
_LAST_STATUS: Dict[str, str] = {}
_IDEMPOTENCY_SEEN: Dict[str, float] = {}
_IDEMPOTENCY_TTL_SECONDS = 60 * 15  # 15 minutes

def _cleanup_idempotency_cache(now_ts: float) -> None:
    stale = [k for k, ts in _IDEMPOTENCY_SEEN.items() if now_ts - ts > _IDEMPOTENCY_TTL_SECONDS]
    for k in stale:
        _IDEMPOTENCY_SEEN.pop(k, None)

def _validate_twilio_signature(request: Request, auth_token: str, form_data: Dict[str, Any]) -> bool:
    """Validate Twilio signature per docs using HMAC-SHA1 over URL+params."""
    try:
        signature = request.headers.get("x-twilio-signature", "")
        # URL must be the full URL Twilio requested
        url = str(request.url)
        # Concatenate URL and params sorted by key
        data = url + ''.join([k + str(form_data[k]) for k in sorted(form_data.keys())])
        mac = hmac.new(auth_token.encode("utf-8"), msg=data.encode("utf-8"), digestmod=hashlib.sha1)
        expected = base64.b64encode(mac.digest()).decode()
        return hmac.compare_digest(signature, expected)
    except Exception:
        return False


@router.post("/api/webhooks/twilio/message-status")
async def handle_twilio_message_status(
    request: Request,
    MessageSid: str = Form(...),
    MessageStatus: str = Form(...),
    To: str = Form(None),
    From: str = Form(None),
    ErrorCode: str = Form(None)
):
    """Handle message status callbacks from Twilio with validation, dedupe, and monotonic state updates."""
    endpoint = "/api/webhooks/twilio/message-status"
    client_host = request.client.host if request.client else "unknown_client"
    headers = dict(request.headers)
    log_request_start(endpoint, "POST", headers, client_host)

    # Parse the full form to use for signature verification
    form = await request.form()
    form_dict = {k: v for k, v in form.items()}

    # Validate Twilio signature
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not auth_token:
        logger.warning(f"[{endpoint}] TWILIO_AUTH_TOKEN not set; skipping signature validation")
    else:
        if not _validate_twilio_signature(request, auth_token, form_dict):
            log_error("Invalid Twilio signature", context={"MessageSid": MessageSid}, endpoint=endpoint)
            return JSONResponse(status_code=403, content={"detail": "invalid signature"})

    # Idempotency dedupe
    idem = headers.get("i-twilio-idempotency-token") or headers.get("I-Twilio-Idempotency-Token")
    now_ts = time.time()
    _cleanup_idempotency_cache(now_ts)
    if idem:
        if idem in _IDEMPOTENCY_SEEN:
            logger.info(f"[{endpoint}] Duplicate webhook ignored (idempotency): token={idem}, sid={MessageSid}")
            return {"status": "ok"}
        _IDEMPOTENCY_SEEN[idem] = now_ts

    # Monotonic status enforcement
    old_status = _LAST_STATUS.get(MessageSid)
    new_status = (MessageStatus or "").lower()
    if old_status is not None:
        old_rank = _STATUS_ORDER.get(old_status, -1)
        new_rank = _STATUS_ORDER.get(new_status, -1)
        if new_rank < old_rank:
            logger.info(f"[{endpoint}] Ignoring regressive status for {MessageSid}: {old_status} -> {new_status}")
            log_response(200, "OK", endpoint)
            return {"status": "ok", "ignored": True}

    _LAST_STATUS[MessageSid] = new_status

    # Log the status update
    logger.info(f"Twilio Message Status Update - SID: {MessageSid}, Status: {MessageStatus}")
    if ErrorCode:
        logger.error(f"Message {MessageSid} failed with error code: {ErrorCode}")

    log_response(200, "OK", endpoint)
    return {"status": "ok"}



@router.post("/webhooks/twilio/incoming-call")
async def handle_twilio_incoming_call(
    request: Request,
    From: str = Form(None),
    SpeechResult: str = Form(None),  # Speech-to-text result from Twilio
    RecordingUrl: str = Form(None),  # URL of the recording if available
    db: Session = Depends(get_db)
):
    """Handle incoming calls from Twilio and respond with AI-generated speech.
    
    Args:
        request: The incoming HTTP request
        From: The caller's phone number (from Twilio form data)
        SpeechResult: The transcribed text from the user's speech (if available)
        RecordingUrl: URL of the recording (if available)
        db: Database session
    """
    endpoint_name = "/webhooks/twilio/incoming-call"
    log_request_start(endpoint_name, "POST", dict(request.headers), request.client.host if request.client else "unknown")
    
    try:
        # Normalize phone number (remove non-digits)
        normalized_from = re.sub(r'\D', '', From) if From else ''
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from and normalized_from.isdigit() else None
        
        # Create a voice response
        resp = VoiceResponse()
        
        if not user:
            logger.warning(f"[{endpoint_name}] Unauthorized access attempt from {From}")
            resp.say("Sorry, you are not authorized to use this service. Please contact support for assistance.")
            log_response(200, "Unauthorized response sent", endpoint_name)
            return Response(content=str(resp), media_type="application/xml", status_code=200)
        
        # If this is a transcription callback with speech result
        if SpeechResult:
            logger.info(f"[{endpoint_name}] Processing speech input from user {user.userId}: {SpeechResult}")
            
            # Create a chat ID for this conversation
            chat_id = f"call_{normalized_from}"
            
            # Save the query to database
            try:
                db_query = Query(
                    userId=user.userId,
                    chatId=chat_id,
                    query_text=SpeechResult,
                    response=None  # Will be updated with AI response
                )
                db.add(db_query)
                db.commit()
                db.refresh(db_query)
                logger.info(f"[{endpoint_name}] Created query with ID: {db_query.queryId}")
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error creating query record: {str(e)}", exc_info=True)
                db.rollback()
                resp.say("Sorry, there was an error processing your request.")
                return Response(content=str(resp), media_type="application/xml", status_code=200)
            
            # Load conversation history
            conversation_history = []
            try:
                # Get recent messages for context
                recent_messages = db.query(Query).filter(
                    Query.userId == user.userId,
                    Query.chatId == chat_id
                ).order_by(Query.created_at.desc()).limit(5).all()
                
                # Reverse to get chronological order
                for msg in reversed(recent_messages):
                    if msg.query_text:
                        conversation_history.append({"role": "user", "content": msg.query_text})
                    if msg.response:
                        conversation_history.append({"role": "assistant", "content": msg.response})
                
                logger.info(f"[{endpoint_name}] Loaded {len(conversation_history)} messages from history")
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error loading conversation history: {str(e)}")
                conversation_history = []
            
            # Prepare the query for the /query endpoint
            query_data = QueryRequest(
                query=SpeechResult,
                chat_id=chat_id,
                conversation_history=conversation_history,
                user_id=user.userId
            )
            
            try:
                # Call the query endpoint
                logger.info(f"[{endpoint_name}] Forwarding query to /query endpoint")
                ai_response = await query_endpoint(
                    query_data=query_data,
                    user=user,
                    db=db
                )
                
                response_text = ai_response.response
                logger.info(f"[{endpoint_name}] Received response from /query endpoint")
                
                # Update the query with the response
                db_query.response = response_text
                db.commit()
                
                # Speak the response to the user
                resp.say(response_text)
                
                # Ask if there's anything else we can help with
                resp.say("Is there anything else I can help you with?")
                
                # Record the user's response
                resp.record(
                    action=f"/webhooks/twilio/incoming-call?From={From}",
                    method="POST",
                    finish_on_key="#",
                    transcribe=True,
                    transcribe_callback=f"/webhooks/twilio/incoming-call?From={From}"
                )
                
                log_response(200, "AI response spoken to user", endpoint_name)
                return Response(content=str(resp), media_type="application/xml", status_code=200)
                
            except Exception as e:
                logger.error(f"[{endpoint_name}] Error calling /query endpoint: {str(e)}", exc_info=True)
                db.rollback()
                resp.say("I'm sorry, I encountered an error processing your request. Please try again later.")
                return Response(content=str(resp), media_type="application/xml", status_code=200)
        
        # Initial greeting for new calls
        logger.info(f"[{endpoint_name}] New call from user {user.userId} ({user.username})")
        resp.say(f"Hello {user.username}, this is Gad's assistant. How can I help you today?")
        
        # Record the user's message
        resp.record(
            action=f"/webhooks/twilio/incoming-call?From={From}",
            method="POST",
            finish_on_key="#",
            transcribe=True,
            transcribe_callback=f"/webhooks/twilio/incoming-call?From={From}"
        )
        
        log_response(200, "Initial greeting sent", endpoint_name)
        return Response(content=str(resp), media_type="application/xml", status_code=200)
        
    except Exception as e:
        logger.error(f"[{endpoint_name}] Unexpected error: {str(e)}", exc_info=True)
        resp = VoiceResponse()
        resp.say("I'm sorry, an unexpected error occurred. Please try your call again later.")
        return Response(content=str(resp), media_type="application/xml", status_code=200)


@router.post("/webhooks/twilio/transcription-callback")
async def handle_transcription_callback(
    request: Request,
    From: str = Form(...),
    TranscriptionText: str = Form(...),
    RecordingUrl: str = Form(None),
    db: Session = Depends(get_db)
):
    normalized_from = re.sub(r"\D", "", From)
    user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None

    if not user:
        return Response(status_code=204)

    chat_id = f"voice_{normalized_from}"
    user_transcript = TranscriptionText

    # Save query
    db_query = Query(userId=user.userId, chatId=chat_id, query_text=user_transcript)
    db.add(db_query)
    db.commit()
    db.refresh(db_query)

    # Load memory
    shortterm = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "short_term_memory.json").first()
    longterm = db.query(DBFile).filter(DBFile.userId == user.userId, DBFile.filename == "long_term_memory.json").first()
    short_content = json.loads(shortterm.content.decode('utf-8')) if shortterm and shortterm.content else {}
    long_content = json.loads(longterm.content.decode('utf-8')) if longterm and longterm.content else {}

    short_term_memory = ShortTermMemoryManager(memory_content=short_content)
    long_term_memory = LongTermMemoryManager(memory_content=long_content)

    aux_data = {
        "username": user.username,
        "user_id": user.userId,
        "chat_id": chat_id,
        "query_id": db_query.queryId,
        "client_info": {"max_tokens": 256, "temperature": 0.5}
    }

    ai_response = ai_query_handler.query_openai(
        query=user_transcript,
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        aux_data=aux_data,
        max_tokens=256,
        temperature=0.5,
    )

    db_query.response = ai_response
    db.commit()

    summary = ai_query_handler.summarize_conversation(user_transcript, ai_response)
    short_content.setdefault("conversations", []).append({
        "query": user_transcript, "response": ai_response, "summary": summary
    })
    if shortterm:
        shortterm.content = json.dumps(short_content).encode('utf-8')
        shortterm.size = len(shortterm.content)
        db.commit()

    # Optionally send an SMS response or set up Twilio to redirect for the next round
    # Or ignore (Twilio will just hang up or timeout)
    return Response(status_code=200)



# send_twilio_message has been moved to services.py
