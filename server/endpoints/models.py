"""Request/Response models for all endpoints."""

import uuid
from ipaddress import ip_address as validate_ip
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Any

# ============================================================================
# DEVICE MODELS
# ============================================================================

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

class DeviceRegisterRequest(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_type: str = "thoth"
    hardware_info: Optional[Dict[str, Any]] = {}
    ip_address: Optional[str] = None
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Device ID must be at least 3 characters')
        return v.strip()
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v is None:
            return v
        try:
            validate_ip(v)
            return v
        except ValueError:
            raise ValueError('Invalid IP address format')

class DeviceStatusRequest(BaseModel):
    status: str
    battery_level: Optional[int] = None
    wifi_connected: Optional[bool] = None
    collection_active: Optional[bool] = None
    
    @validator('status')
    def validate_status(cls, v):
        allowed = ['online', 'offline', 'error', 'maintenance']
        if v not in allowed:
            raise ValueError(f'Status must be one of: {allowed}')
        return v
    
    @validator('battery_level')
    def validate_battery_level(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Battery level must be between 0 and 100')
        return v

# ============================================================================
# DATA MODELS
# ============================================================================

class DataUploadRequest(BaseModel):
    device_id: str
    data_type: str = "sensor"
    data: List[Dict[str, Any]]
    timestamp: Optional[str] = None
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Device ID must be at least 3 characters')
        return v.strip()
    
    @validator('data')
    def validate_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Data array cannot be empty')
        if len(v) > 10000:  # Prevent extremely large uploads
            raise ValueError('Data array too large (max 10,000 items)')
        return v
    
    @validator('data_type')
    def validate_data_type(cls, v):
        allowed_types = ['sensor', 'log', 'event', 'metric', 'diagnostic']
        if v not in allowed_types:
            raise ValueError(f'Data type must be one of: {allowed_types}')
        return v

# ============================================================================
# FILE MODELS
# ============================================================================

class FileUploadRequest(BaseModel):
    filename: str
    content_type: str
    content: str  # Encoded file content
    size: int
    file_hash: Optional[str] = None
    is_base64_encoded: bool = True

class FileUploadSimpleRequest(BaseModel):
    filename: str
    content: str
    content_type: Optional[str] = None
    device_id: Optional[str] = None
    is_base64: bool = False
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Filename is required')
        # Basic security check
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename - no path separators allowed')
        # Check for dangerous extensions
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
        if any(v.lower().endswith(ext) for ext in dangerous_extensions):
            raise ValueError('File type not allowed')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        if not v:
            raise ValueError('File content cannot be empty')
        return v
    
    @validator('device_id')
    def validate_device_id(cls, v):
        if v is not None and len(v.strip()) < 3:
            raise ValueError('Device ID must be at least 3 characters')
        return v.strip() if v else None

# ============================================================================
# AUTH MODELS
# ============================================================================

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
    email: str
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
    
    @validator('email')
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Valid email address required')
        return v.strip().lower()

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

# ============================================================================
# AI MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:  # Prevent extremely long queries
            raise ValueError('Query too long (max 10,000 characters)')
        return v.strip()

class QueryResponse(BaseModel):
    success: bool
    query_id: int
    response: str
    chat_id: str
    timestamp: str

# ============================================================================
# SYSTEM MODELS
# ============================================================================

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class StandardResponse(BaseModel):
    success: bool
    message: str

class DeviceResponse(StandardResponse):
    device_id: str
    device_name: Optional[str] = None

class DataUploadResponse(StandardResponse):
    upload_id: str
    device_id: str
    data_count: int

class FileUploadResponse(StandardResponse):
    file_id: int
    filename: str
    size: int

class PaginatedResponse(BaseModel):
    success: bool
    count: int
    has_more: bool
