"""Authentication Module for LMS Platform.

This module handles all aspects of user authentication including:
- Password hashing and verification
- JWT token generation and validation
- User authentication logic
- Dependency injection for database and current user
"""

import logging
import bcrypt
from jose import jwt, JWTError
from fastapi import status
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
import os
from .db import User, SessionLocal

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))  # Default to 24 hours
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load secret key; fall back to a default (development) key to avoid runtime errors
# NOTE: In production, ALWAYS set the SECRET_KEY environment variable to a strong, random value.
# If SECRET_KEY is missing, we log a warning and use a weak default to keep the API online instead of crashing.
SECRET_KEY = os.getenv("SECRET_KEY") or "__insecure_dev_key_change_me__"
if SECRET_KEY == "__insecure_dev_key_change_me__":
    import logging
    logging.getLogger("lms.server").warning("[Auth] SECRET_KEY env var not set. Using insecure default key. Set SECRET_KEY for production!")

def get_db():
    """Create and yield a database session.
    
    This function serves as a FastAPI dependency for database access.
    It ensures the database session is properly closed after use.
    
    Yields:
        Session: A SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt.
    
    Args:
        password: The plain text password to hash
        
    Returns:
        str: The hashed password as a string
    """
    # Ensure password is not too long for bcrypt (72 bytes max)
    if len(password) > 72:
        password = password[:72]
        
    try:
        # Generate salt and hash the password using bcrypt directly
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        logging.error(f"[Auth] Error hashing password: {str(e)}")
        raise

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password to compare against
        
    Returns:
        bool: True if password matches, False otherwise
    """
    logging.info("[Auth] Starting password verification")
    
    # Input validation
    if not plain_password:
        logging.warning("[Auth] verify_password: No plain password provided")
        return False
        
    if not hashed_password:
        logging.warning("[Auth] verify_password: No hash provided")
        return False
    
    # Log the inputs (be careful with sensitive data in production)
    logging.debug(f"[Auth] Plain password length: {len(plain_password)}")
    logging.debug(f"[Auth] Hashed password: {hashed_password[:10]}...")
    
    # Ensure plain_password is not too long for bcrypt
    if len(plain_password) > 72:
        logging.warning("[Auth] Password exceeds 72 bytes, truncating")
        plain_password = plain_password[:72]
    
    try:
        # Encode both strings to bytes
        plain_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        
        logging.debug("[Auth] Calling bcrypt.checkpw")
        result = bcrypt.checkpw(plain_bytes, hash_bytes)
        logging.info(f"[Auth] bcrypt.checkpw result: {result}")
        return result
        
    except ValueError as ve:
        logging.error(f"[Auth] ValueError in verify_password: {str(ve)}")
        logging.error(f"[Auth] Hash format may be invalid")
        return False
    except Exception as e:
        logging.error(f"[Auth] Unexpected error in verify_password: {str(e)}")
        logging.error(f"[Auth] Error type: {type(e).__name__}", exc_info=True)
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: The data to encode in the token, typically includes the 'sub' field
        expires_delta: Optional expiration time, either as timedelta or minutes (int)
        
    Returns:
        str: The encoded JWT token
        
    Raises:
        HTTPException: If there's an error encoding the JWT token
    """
    try:
        to_encode = data.copy()
        # If expires_delta is an int (minutes), convert to timedelta
        if isinstance(expires_delta, int):
            expires_delta = timedelta(minutes=expires_delta)
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        
        # Ensure SECRET_KEY is set
        if not SECRET_KEY:
            logging.error("[Auth] SECRET_KEY is not set")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error"
            )
            
        # Encode the JWT token
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
        
    except JWTError as e:
        logging.error(f"[Auth] Error encoding JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )
    except Exception as e:
        logging.error(f"[Auth] Unexpected error in create_access_token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password.
    
    Args:
        db: Database session
        username: The username to authenticate
        password: The password to verify
        
    Returns:
        Optional[User]: The authenticated user object or None if authentication fails
    """
    try:
        logging.info(f"[Auth] Starting authentication for user: {username}")
        
        # Find user by username
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            logging.warning(f"[Auth] User not found: {username}")
            return None
            
        if not user.hashed_password:
            logging.warning(f"[Auth] User {username} has no password set")
            return None
            
        logging.info(f"[Auth] Found user: {user.username} (ID: {user.userId})")
        
        # Verify the password
        try:
            password_matches = verify_password(password, user.hashed_password)
            logging.info(f"[Auth] Password verification result for {username}: {password_matches}")
            
            if password_matches:
                logging.info(f"[Auth] Successful authentication for user: {username}")
                return user
            else:
                logging.warning(f"[Auth] Password verification failed for user: {username}")
                return None
                
        except Exception as verify_error:
            logging.error(f"[Auth] Error during password verification for {username}: {str(verify_error)}")
            logging.error(f"[Auth] Error type: {type(verify_error).__name__}")
            return None
        
    except Exception as e:
        logging.error(f"[Auth] Unexpected error during authentication for {username}: {str(e)}")
        logging.error(f"[Auth] Error type: {type(e).__name__}", exc_info=True)
        return None

async def get_user_from_token(token: str) -> dict:
    """Get user information from a JWT token without requiring a database session.
    
    This is a lighter version of get_current_user that doesn't hit the database.
    Use this when you only need basic user info from the token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        dict: User information from the token
        
    Raises:
        HTTPException: 401 error if token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
        # Return basic user info from the token
        return {
            "username": username,
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "scopes": payload.get("scopes", [])
        }
    except JWTError as e:
        logging.error(f"[Auth] JWT validation error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logging.error(f"[Auth] Error in get_user_from_token: {str(e)}")
        raise credentials_exception


def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from a JWT token.
    
    This function is used as a FastAPI dependency to inject the current user
    into route handlers that require authentication.
    
    Args:
        token: The JWT token from the Authorization header
        db: Database session
        
    Returns:
        User: The authenticated user object
        
    Raises:
        HTTPException: 401 error if token is invalid or user doesn't exist
    """
    credentials_exception = HTTPException(status_code=401, detail="Invalid credentials")
    # Log the raw Authorization header for debugging auth failures
    logging.getLogger("lms.server").debug(
        "[AUTH] get_current_user Authorization header: %s",
        request.headers.get("Authorization"),
    )
    try:
        # Log the token value (truncated) for debugging
        logging.getLogger("lms.server").debug(
            "[AUTH] Decoding token: %s...",
            token[:10] + '...' if token else None,
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Log the full payload for debugging
        logging.getLogger("lms.server").debug(
            "[AUTH] Decoded token payload: %s",
            payload
        )
        
        # Try to get user by username or user_id from the token
        username = payload.get("username")
        user_id = payload.get("sub")
        
        if username:
            user = db.query(User).filter(User.username == username).first()
        elif user_id:
            # Try to get user by ID if username is not in the token
            user = db.query(User).filter(User.userId == int(user_id)).first()
        else:
            logging.error("[AUTH] No username or user_id found in token")
            raise credentials_exception
            
        if user is None:
            logging.error(f"[AUTH] User not found in database. Username: {username}, User ID: {user_id}")
            raise credentials_exception
            
        return user
        
    except JWTError as e:
        logging.error(f"[AUTH] JWT decoding error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logging.error(f"[AUTH] Unexpected error in get_current_user: {str(e)}")
        raise credentials_exception
