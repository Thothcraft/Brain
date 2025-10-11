"""Authentication endpoints for user login, registration, and token management."""

from datetime import datetime, timedelta
from typing import Dict, Any
import logging
from fastapi import Request

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator

from server.db import get_db, User
from server.auth import get_current_user
from server.utils.logging_utils import log_request_start, log_response, log_error

router = APIRouter(prefix="", tags=["auth"])

# Request/Response Models
class LoginRequest(BaseModel):
    username: str
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class RegisterRequest(BaseModel):
    username: str
    password: str
    phone_number: int = None
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    username: str

class RegisterResponse(BaseModel):
    success: bool
    user_id: int
    username: str
    message: str

class UserResponse(BaseModel):
    userId: int
    username: str
    email: str
    phone_number: int = None
    created_at: str

@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and return access token",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Invalid request format"}
    }
)
async def login_for_access_token(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Authenticate user and return JWT access token.
    
    Purpose: Authenticate users and provide access tokens for API access
    
    Args:
        login_data: Username and password credentials
        db: Database session
        
    Returns:
        TokenResponse: Access token and user information
        
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    try:
        log_request_start("POST", "/token", None)
        
        # Import auth functions here to avoid circular imports
        from server.auth import authenticate_user, create_access_token
        
        # Authenticate user with detailed logging
        logging.info(f"Attempting to authenticate user: {login_data.username}")
        user = authenticate_user(db, login_data.username, login_data.password)
        
        if not user:
            # Log more details about the failure
            db_user = db.query(User).filter(User.username == login_data.username).first()
            if not db_user:
                logging.warning(f"User not found: {login_data.username}")
            else:
                logging.warning(f"User found but password verification failed for: {login_data.username}")
                logging.debug(f"Stored hash: {db_user.hashed_password}")
            
            log_error("Authentication failed for user", None, {"username": login_data.username}, "/token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        logging.info(f"Successfully authenticated user: {user.username} (ID: {user.userId})")
        
        # Create access token
        access_token_expires = timedelta(days=30)  # Token expires in 30 days
        access_token = create_access_token(
            data={"sub": str(user.userId)}, expires_delta=access_token_expires
        )
        
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 2592000,  # 30 days in seconds (30 * 24 * 60 * 60)
            "user_id": user.userId,
            "username": user.username
        }
        
        log_response(200, response_data, "/token")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_error(f"Login error: {str(e)}\n{error_details}", e, endpoint="/token")
        logging.error(f"Full error details: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post(
    "/register", 
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User registration",
    description="Register a new user account",
    responses={
        201: {"description": "Registration successful"},
        400: {"description": "User already exists or invalid data"},
        422: {"description": "Invalid request format"}
    }
)
async def register_user(
    register_data: RegisterRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Register a new user account.
    
    Purpose: Create new user accounts with validation
    
    Args:
        register_data: User registration information
        db: Database session
        
    Returns:
        RegisterResponse: Registration confirmation and user info
        
    Raises:
        HTTPException: 400 if user already exists
    """
    try:
        log_request_start("POST", "/register", None)
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            User.username == register_data.username
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Import password hashing function
        from server.auth import get_password_hash
        
        # Create new user
        new_user = User(
            username=register_data.username,
            hashed_password=get_password_hash(register_data.password),
            phone_number=register_data.phone_number
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        response_data = {
            "success": True,
            "user_id": new_user.userId,
            "username": new_user.username,
            "message": "User registered successfully"
        }
        
        log_response(201, "User registered successfully", "/register")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Registration error: {str(e)}", e, endpoint="/register")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.get(
    '/profile', 
    response_model=UserResponse,
    summary="Get user profile",
    description="Get profile information for the authenticated user",
    responses={
        200: {"description": "Profile retrieved successfully"},
        401: {"description": "Not authenticated"}
    }
)
async def get_user_profile(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get profile information for the currently authenticated user.
    
    Purpose: Retrieve user profile data for authenticated users
    
    Args:
        current_user: Authenticated user from JWT token
        
    Returns:
        UserResponse: User profile information
    """
    try:
        # Log the request with headers
        headers = dict(request.headers) if hasattr(request, "headers") else {}
        log_request_start("GET", "/profile", headers)
        
        # Create profile data with all required fields
        profile_data = {
            "userId": current_user.userId,
            "username": current_user.username,
            "email": f"{current_user.username}@example.com",  # Default email since it's not in the model
            "phone_number": current_user.phone_number or 0,  # Default to 0 if None
            "created_at": datetime.utcnow().isoformat()  # Add current timestamp
        }
        
        log_response(200, "Profile retrieved successfully", "/profile")
        return profile_data
        
    except Exception as e:
        log_error(f"Profile retrieval error: {str(e)}", e, endpoint="/profile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )
