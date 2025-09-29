"""
Pydantic models for request/response validation and serialization.

This package contains schemas organized by domain:
- auth.py: Authentication and user management schemas
- ai.py: AI and chat-related schemas
- files.py: File and device management schemas
- twilio.py: Twilio integration schemas
- health.py: Health check related schemas
"""

# Re-export commonly used models for easier imports
from .auth import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    TokenResponse,
    UserResponse,
    PasswordResetRequest,
    PasswordResetConfirm,
    ChangePasswordRequest,
    UpdateProfileRequest,
    TokenData,
    ErrorResponse
)

from .ai import (
    QueryRequest,
    QueryResponse,
    ChatMessage,
    ChatHistory,
    AIConfig,
    AITrainingExample,
    AITrainingRequest
)

from .files import (
    FileType,
    FileUploadResponse,
    FileInfo,
    FileListResponse,
    FileUpdateRequest,
    DeviceType,
    DeviceStatus,
    DeviceInfo,
    DeviceListResponse,
    DeviceHeartbeatRequest
)

from .twilio import (
    MessageStatus,
    MessageDirection,
    MessageType,
    MessageRequest,
    MessageResponse,
    IncomingMessage,
    CallStatus,
    CallRequest,
    CallResponse,
    TranscriptionStatus,
    TranscriptionResponse
)

from .health import HealthCheckResponse

__all__ = [
    # Auth
    'RegisterRequest',
    'RegisterResponse',
    'LoginRequest',
    'TokenResponse',
    'UserResponse',
    'PasswordResetRequest',
    'PasswordResetConfirm',
    'ChangePasswordRequest',
    'UpdateProfileRequest',
    'TokenData',
    'ErrorResponse',
    
    # AI
    'QueryRequest',
    'QueryResponse',
    'ChatMessage',
    'ChatHistory',
    'AIConfig',
    'AITrainingExample',
    'AITrainingRequest',
    
    # Files & Devices
    'FileType',
    'FileUploadResponse',
    'FileInfo',
    'FileListResponse',
    'FileUpdateRequest',
    'DeviceType',
    'DeviceStatus',
    'DeviceInfo',
    'DeviceListResponse',
    'DeviceHeartbeatRequest',
    
    # Twilio
    'MessageStatus',
    'MessageDirection',
    'MessageType',
    'MessageRequest',
    'MessageResponse',
    'IncomingMessage',
    'CallStatus',
    'CallRequest',
    'CallResponse',
    'TranscriptionStatus',
    'TranscriptionResponse',
    
    # Health
    'HealthCheckResponse'
]
