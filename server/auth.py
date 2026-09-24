"""Authentication Module for LMS Platform.

This module handles all aspects of user authentication including:
- Password hashing and verification
- JWT token generation and validation
- User authentication logic
- Dependency injection for database and current user
"""

import logging
import hashlib
import json
import secrets
import bcrypt
from jose import jwt, JWTError
from fastapi import status
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session
from typing import Optional
import os
from .db import User, SessionLocal, AutomationKey, get_db

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))  # Default to 24 hours
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# ── credential domains ────────────────────────────────────────────────────────
# Browsers, developer clients and physical devices have different security
# requirements, so each uses its own signing domain rather than one master
# secret. SESSION_SECRET signs HttpOnly browser-session tokens, DEVICE_SECRET
# signs device-scoped credentials, SECRET_KEY signs user access tokens.
# Each falls back to SECRET_KEY so existing deployments keep working until
# distinct secrets are configured.
_ENVIRONMENT = os.getenv("ENVIRONMENT", os.getenv("BRAIN_ENV", "development")).lower()
_IS_PRODUCTION = _ENVIRONMENT in {"production", "prod"}

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if _IS_PRODUCTION:
        raise RuntimeError(
            "SECRET_KEY environment variable is required in production. "
            "Refusing to start with an insecure signing key."
        )
    SECRET_KEY = "__insecure_dev_key_change_me__"
    logging.getLogger("lms.server").warning(
        "[Auth] SECRET_KEY env var not set. Using insecure default key (development only)."
    )

SESSION_SECRET = os.getenv("SESSION_SECRET") or SECRET_KEY
DEVICE_SECRET = os.getenv("DEVICE_SECRET") or SECRET_KEY

_DOMAIN_SECRETS = {
    "user": SECRET_KEY,
    "session": SESSION_SECRET,
    "device": DEVICE_SECRET,
}

SESSION_COOKIE_NAME = "thoth_session"
SESSION_COOKIE_SECURE = _IS_PRODUCTION
SESSION_EXPIRE_DAYS = int(os.getenv("SESSION_EXPIRE_DAYS", "14"))


def _secret_for_domain(domain: str) -> str:
    return _DOMAIN_SECRETS.get(domain, SECRET_KEY)


def decode_token_any(token: str) -> tuple:
    """Verify a token against each credential domain.

    Returns ``(payload, domain)``. Raises JWTError if no domain verifies.
    """
    last_err: Optional[Exception] = None
    for domain, secret in _DOMAIN_SECRETS.items():
        try:
            return jwt.decode(token, secret, algorithms=[ALGORITHM]), domain
        except JWTError as e:
            last_err = e
    raise last_err or JWTError("token verification failed")

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

def create_session_token(user: User) -> str:
    """Create a browser-session token for the HttpOnly cookie domain."""
    return create_access_token(
        data={
            "sub": str(user.userId),
            "username": user.username,
            "role": user.role,
            "typ": "session",
        },
        expires_delta=timedelta(days=SESSION_EXPIRE_DAYS),
        domain="session",
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, domain: str = "user") -> str:
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
            
        # Encode the JWT token under the requested credential domain
        encoded_jwt = jwt.encode(to_encode, _secret_for_domain(domain), algorithm=ALGORITHM)
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
        
        # The verified email and username are equivalent login identities for
        # both the device dashboard and thothHUB.
        identity = str(username or '').strip()
        user = db.query(User).filter(or_(
            User.username == identity,
            and_(
                func.lower(User.email) == identity.lower(),
                User.email_verified.is_(True),
            ),
        )).first()
        
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

class TokenUser:
    """Simple class to hold user info from token with attribute access."""
    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __getitem__(self, key):
        return self._data[key]


async def get_user_from_token(token: str) -> TokenUser:
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
        payload, domain = decode_token_any(token)
        sub: str = payload.get("sub")
        if sub is None:
            raise credentials_exception
        
        # The 'sub' field contains the user ID (as string)
        # Try to parse it as an integer user_id
        try:
            user_id = int(sub)
            username = payload.get("username")
        except (ValueError, TypeError):
            # If sub is not numeric, treat it as username
            user_id = payload.get("user_id")
            username = sub
            
        # Return basic user info from the token
        return TokenUser({
            "username": username,
            "user_id": user_id,
            "userId": user_id,  # Alias for compatibility
            "email": payload.get("email"),
            "scopes": payload.get("scopes", []),
            "device_id": payload.get("device_id"),
            "domain": domain,
        })
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

    # Scoped automation credential: an opaque X-Api-Key resolves to an
    # AutomationPrincipal carrying the key's scopes. Checked first so a key
    # never needs a JWT.
    api_key = request.headers.get("x-api-key")
    if api_key:
        key = db.query(AutomationKey).filter(
            AutomationKey.key_hash == hash_automation_key(api_key),
            AutomationKey.revoked == False,  # noqa: E712
        ).first()
        if not key:
            raise credentials_exception
        user = db.query(User).filter(User.userId == key.user_id).first()
        if not user:
            raise credentials_exception
        key.last_used_at = datetime.utcnow()
        db.commit()
        return AutomationPrincipal(user, key)

    # Dual auth: bearer token (CLI/SDK/Flutter) or HttpOnly session cookie
    # (browser/thothHUB). Bearer takes precedence when both are present.
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token and not cookie_token:
        raise credentials_exception

    try:
        if token:
            payload, domain = decode_token_any(token)
            if domain == "session":
                # Session tokens are only valid via the cookie channel
                raise credentials_exception
        else:
            payload = jwt.decode(cookie_token, SESSION_SECRET, algorithms=[ALGORITHM])
            domain = "session"
            if payload.get("typ") != "session":
                raise credentials_exception

        # Device-scoped credentials are accepted only by endpoints that
        # explicitly use get_user_from_token and validate the device claim.
        if domain == "device" or "device" in (payload.get("scopes") or []):
            logging.warning("[AUTH] Device-scoped token rejected by user endpoint")
            raise credentials_exception
        
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

        # Carry the token's scope claim so scoped endpoints can enforce it.
        # None = unconstrained user token (full access, backwards compatible).
        try:
            user.auth_scopes = payload.get("scopes")
        except Exception:
            pass
        return user

    except JWTError as e:
        logging.error(f"[AUTH] JWT decoding error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logging.error(f"[AUTH] Unexpected error in get_current_user: {str(e)}")
        raise credentials_exception


# ── scoped automation credentials (§9.4 / §17) ──────────────────────────────
# Operation scopes advertised by the SDK. An automation key carries a subset
# and may only perform the matching operations.
AUTOMATION_SCOPES = {"sensor:stream", "model:deploy", "capture",
                     "device:read", "predictions:read", "context:write"}


def hash_automation_key(raw_key: str) -> str:
    """SHA-256 the raw key — the plaintext is never stored."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_automation_key() -> str:
    """Return a new opaque automation key (shown to the user once)."""
    return "tc_" + secrets.token_urlsafe(32)


class AutomationPrincipal:
    """A `User`-shaped principal backed by a scoped automation key.

    Delegates attribute access to the owning ``User`` so existing handlers
    that read ``.userId``/``.username`` keep working, while ``scopes`` and
    ``is_automation`` expose the credential's restrictions.
    """

    def __init__(self, user: User, key: AutomationKey):
        self._user = user
        self._key = key
        self.userId = user.userId
        self.scopes = key.scope_list()
        self.is_automation = True

    def __getattr__(self, name):
        return getattr(self._user, name)


def get_scoped_principal(required_scope: Optional[str] = None):
    """Dependency factory: authenticate then enforce an operation scope.

    ``get_current_user`` resolves the caller to a ``User`` (JWT/session) or an
    ``AutomationPrincipal`` (X-Api-Key). The resolved principal's scopes —
    ``AutomationPrincipal.scopes`` or the JWT's ``auth_scopes`` claim — must
    include ``required_scope``. A principal with no scope claim (``None``)
    is an unconstrained user token and retains full access.
    """
    def dependency(
        principal=Depends(get_current_user),
    ):
        if required_scope is None:
            return principal
        scopes = getattr(principal, "scopes", None)
        if scopes is None:
            scopes = getattr(principal, "auth_scopes", None)
        if scopes is not None and required_scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"credential missing scope '{required_scope}'")
        return principal

    return dependency
