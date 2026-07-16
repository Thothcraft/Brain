"""Authentication endpoints for user login, registration, and token management."""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import os
import requests
import asyncio
from fastapi import Request

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator

from server.db import get_db, User
from server.auth import get_current_user
from server.utils.logging_utils import log_request_start, log_response, log_error

router = APIRouter(prefix="", tags=["auth"])


def _supabase_signup_key() -> str:
    """Resolve the server-side key across legacy and current Supabase names."""
    return next((
        os.getenv(name, "")
        for name in (
            "SUPABASE_ANON_KEY",
            "SUPABASE_PUBLISHABLE_KEY",
            "SUPABASE_KEY",
            "SUPABASE_SERVICE_KEY",
        )
        if os.getenv(name, "")
    ), "")

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
    email: str
    password: str
    phone_number: Optional[int] = None
    
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

    @validator('email')
    def validate_email(cls, v):
        value = str(v or '').strip().lower()
        if '@' not in value or value.startswith('@') or value.endswith('@') or len(value) > 320:
            raise ValueError('Enter a valid email address')
        return value

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    username: str
    role: int = 0
    plan: str = "free"

class RegisterResponse(BaseModel):
    success: bool
    user_id: int
    username: str
    message: str
    verification_required: bool = True

class UserResponse(BaseModel):
    userId: int
    username: str
    email: Optional[str] = None
    email_verified: bool = False
    phone_number: Optional[int] = None
    role: int = 0
    plan: str = "free"
    org_name: Optional[str] = None
    created_at: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    username: Optional[str] = None
    phone_number: Optional[int] = None
    org_name: Optional[str] = None

    @validator('username')
    def validate_optional_username(cls, value):
        if value is None:
            return value
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError('Username must be at least 3 characters')
        return cleaned

class ResendVerificationRequest(BaseModel):
    email: str

    @validator('email')
    def validate_resend_email(cls, value):
        cleaned = str(value or '').strip().lower()
        if '@' not in cleaned or len(cleaned) > 320:
            raise ValueError('Enter a valid email address')
        return cleaned

@router.get('/registration-status', summary="Email registration capability")
async def registration_status() -> Dict[str, Any]:
    return {
        "email_registration_configured": bool(os.getenv("SUPABASE_URL") and _supabase_signup_key()),
        "email_verification_check_configured": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY")),
        "redirect_url": os.getenv("SUPABASE_EMAIL_REDIRECT_URL", "https://portal-three-rho.vercel.app/auth?verified=1"),
    }

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
            
            log_error("Authentication failed for user", None, {"username": login_data.username}, "/token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if user.supabase_auth_user_id and not user.email_verified:
            service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
            supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
            if not service_key or not supabase_url:
                raise HTTPException(status_code=503, detail="Email verification service is unavailable")
            verification = await asyncio.to_thread(
                requests.get,
                f"{supabase_url}/auth/v1/admin/users/{user.supabase_auth_user_id}",
                headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                timeout=8,
            )
            if verification.status_code != 200:
                raise HTTPException(status_code=503, detail="Unable to verify email status")
            if not verification.json().get("email_confirmed_at"):
                raise HTTPException(status_code=403, detail="Confirm your email before signing in")
            user.email_verified = True
            db.commit()
            
        logging.info(f"Successfully authenticated user: {user.username} (ID: {user.userId})")
        
        access_token_days = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "3650"))
        access_token_expires = timedelta(days=access_token_days)
        access_token = create_access_token(
            data={"sub": str(user.userId), "role": user.role}, expires_delta=access_token_expires
        )
        expires_in = int(access_token_expires.total_seconds())
        
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "user_id": user.userId,
            "username": user.username,
            "role": user.role,
            "plan": user.plan or "free",
        }
        
        log_response(200, {"user_id": user.userId, "username": user.username}, "/token")
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
            (User.username == register_data.username) | (User.email == register_data.email)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )

        supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        anon_key = _supabase_signup_key()
        if not supabase_url or not anon_key:
            raise HTTPException(status_code=503, detail="Email registration is not configured")
        redirect_url = os.getenv("SUPABASE_EMAIL_REDIRECT_URL", "https://portal-three-rho.vercel.app/auth?verified=1")
        signup = await asyncio.to_thread(
            requests.post,
            f"{supabase_url}/auth/v1/signup",
            params={"redirect_to": redirect_url},
            headers={"apikey": anon_key, "Authorization": f"Bearer {anon_key}", "Content-Type": "application/json"},
            json={
                "email": register_data.email,
                "password": register_data.password,
                "data": {"username": register_data.username},
            },
            timeout=10,
        )
        if signup.status_code not in (200, 201):
            logging.warning("Supabase signup failed with status %s", signup.status_code)
            raise HTTPException(status_code=400, detail="Unable to register that email address")
        auth_user = (signup.json() or {}).get("user") or signup.json()
        auth_user_id = auth_user.get("id") if isinstance(auth_user, dict) else None
        if not auth_user_id:
            raise HTTPException(status_code=502, detail="Verification provider returned an invalid response")
        
        # Import password hashing function
        from server.auth import get_password_hash
        
        # Create new user
        new_user = User(
            username=register_data.username,
            email=register_data.email,
            email_verified=bool(auth_user.get("email_confirmed_at")),
            supabase_auth_user_id=str(auth_user_id),
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
            "message": "Check your email to verify your account, then sign in.",
            "verification_required": not new_user.email_verified,
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

@router.post(
    '/logout',
    summary="User logout",
    description="Logout the current user (client should clear token)",
    responses={
        200: {"description": "Logout successful"},
        401: {"description": "Not authenticated"}
    }
)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Logout endpoint - primarily for client-side token invalidation.
    The actual token invalidation happens client-side by removing the stored token.
    This endpoint confirms the logout action was received.
    """
    log_request_start("POST", "/logout", dict(request.headers) if hasattr(request, "headers") else {})
    log_response(200, "Logout successful", "/logout")
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@router.post('/resend-verification', summary="Resend signup verification email")
async def resend_verification(payload: ResendVerificationRequest) -> Dict[str, Any]:
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    anon_key = _supabase_signup_key()
    if not supabase_url or not anon_key:
        raise HTTPException(status_code=503, detail="Email registration is not configured")
    redirect_url = os.getenv("SUPABASE_EMAIL_REDIRECT_URL", "https://portal-three-rho.vercel.app/auth?verified=1")
    response = await asyncio.to_thread(
        requests.post,
        f"{supabase_url}/auth/v1/resend",
        headers={"apikey": anon_key, "Authorization": f"Bearer {anon_key}", "Content-Type": "application/json"},
        json={"type": "signup", "email": payload.email, "options": {"emailRedirectTo": redirect_url}},
        timeout=10,
    )
    if response.status_code not in (200, 201):
        logging.warning("Supabase verification resend failed with status %s", response.status_code)
    return {"success": True, "message": "If the account is awaiting confirmation, a new verification email has been sent."}

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
            "role": current_user.role,
            "plan": current_user.plan or "free",
            "org_name": current_user.org_name,
            "email": current_user.email,
            "email_verified": bool(current_user.email_verified),
            "phone_number": current_user.phone_number,
            "created_at": None,
        }
        
        log_response(200, "Profile retrieved successfully", "/profile")
        return profile_data
        
    except Exception as e:
        log_error(f"Profile retrieval error: {str(e)}", e, endpoint="/profile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve profile: {str(e)}"
        )

@router.put('/profile', response_model=UserResponse, summary="Update user profile")
async def update_user_profile(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if payload.username and payload.username != current_user.username:
        duplicate = db.query(User).filter(
            User.username == payload.username,
            User.userId != current_user.userId,
        ).first()
        if duplicate:
            raise HTTPException(status_code=409, detail="Username is already in use")
        current_user.username = payload.username
    if 'phone_number' in payload.__fields_set__:
        current_user.phone_number = payload.phone_number
    if 'org_name' in payload.__fields_set__ and current_user.role == 2:
        current_user.org_name = (payload.org_name or '').strip() or None
    db.commit()
    db.refresh(current_user)
    return {
        "userId": current_user.userId,
        "username": current_user.username,
        "email": current_user.email,
        "email_verified": bool(current_user.email_verified),
        "phone_number": current_user.phone_number,
        "role": current_user.role,
        "plan": current_user.plan or "free",
        "org_name": current_user.org_name,
        "created_at": None,
    }
