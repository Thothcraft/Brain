"""Authentication Module for LMS Platform.

This module handles all aspects of user authentication including:
- Password hashing and verification
- JWT token generation and validation
- User authentication logic
- Dependency injection for database and current user
"""

import logging
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
import os
from .db import User, SessionLocal

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))  # Default to 24 hours
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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
        str: The hashed password
    """
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain: The plain text password
        hashed: The hashed password to compare against
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: The data to encode in the token, typically includes the 'sub' field
        expires_delta: Optional expiration time, either as timedelta or minutes (int)
        
    Returns:
        str: The encoded JWT token
    """
    to_encode = data.copy()
    # If expires_delta is an int (minutes), convert to timedelta
    if isinstance(expires_delta, int):
        expires_delta = timedelta(minutes=expires_delta)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password.
    
    Args:
        db: Database session
        username: The username to authenticate
        password: The password to verify
        
    Returns:
        Optional[User]: The authenticated user object or None if authentication fails
    """
    user = db.query(User).filter(User.username == username).first()
    # If the user record is missing or the stored hash is NULL / empty, treat as invalid credentials.
    if not user or not user.hashed_password:
        return None

    # Verify the supplied password against the stored bcrypt hash. Wrap in a
    # try/except so any unexpected passlib errors (e.g. invalid hash format)
    # don't bubble up as a 500 Internal Server Error – instead, fail the
    # authentication gracefully.
    try:
        if not verify_password(password, user.hashed_password):
            return None
    except Exception:
        # Log at DEBUG level via the app logger if available, but avoid
        # exposing details to the client.
        import logging
        logging.getLogger("lms.server").debug("[Auth] Password verification error for user '%s'", username, exc_info=True)
        return None

    # Successful authentication
    return user

async def get_current_user(
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
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
