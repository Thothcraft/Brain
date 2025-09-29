"""Pydantic models for file and device management."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    """Supported file types."""
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OTHER = "other"


class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    message: str = Field(..., example="File uploaded successfully")
    file_id: int = Field(..., example=123)
    size: int = Field(..., example=1024, description="File size in bytes")
    hash: str = Field(..., example="a1b2c3d4e5f6...", 
                     description="SHA-256 hash of the file contents")
    filename: str = Field(..., example="document.pdf",
                          description="Original name of the uploaded file")
    content_type: Optional[str] = Field(None, example="application/pdf",
                                      description="MIME type of the file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "File uploaded successfully",
                "file_id": 123,
                "size": 1024,
                "hash": "a1b2c3d4e5f6...",
                "filename": "document.pdf",
                "content_type": "application/pdf"
            }
        }


class FileInfo(BaseModel):
    """Detailed information about a file."""
    id: int = Field(..., example=123, description="Unique file ID")
    name: str = Field(..., example="document.pdf", 
                     description="Original file name")
    size: int = Field(..., example=1024, 
                     description="File size in bytes")
    content_type: Optional[str] = Field(None, example="application/pdf",
                                      description="MIME type of the file")
    hash: str = Field(..., example="a1b2c3d4e5f6...", 
                     description="SHA-256 hash of the file contents")
    file_type: FileType = Field(..., example=FileType.DOCUMENT,
                              description="Type/category of the file")
    uploaded_at: datetime = Field(..., 
                                example="2023-01-01T12:00:00Z",
                                description="When the file was uploaded")
    is_public: bool = Field(False, 
                           description="Whether the file is publicly accessible")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                    description="Additional file metadata")
    
    class Config:
        orm_mode = True


class FileListResponse(BaseModel):
    """Response model for listing files."""
    files: List[FileInfo] = Field(..., description="List of files")
    total: int = Field(..., example=5, 
                      description="Total number of files available")
    page: int = Field(1, example=1, 
                     description="Current page number")
    page_size: int = Field(20, example=20, 
                         description="Number of items per page")


class FileUpdateRequest(BaseModel):
    """Request model for updating file metadata."""
    name: Optional[str] = Field(None, example="new-name.pdf",
                               description="New name for the file")
    is_public: Optional[bool] = Field(None, 
                                     description="Set file public/private")
    metadata: Optional[Dict[str, Any]] = Field(None,
                                             description="Additional metadata to update")


class DeviceType(str, Enum):
    """Supported device types."""
    WEB = "web"
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    IOT = "iot"
    OTHER = "other"


class DeviceStatus(str, Enum):
    """Device status values."""
    ONLINE = "online"
    OFFLINE = "offline"
    AWAY = "away"
    BUSY = "busy"
    DND = "dnd"  # Do Not Disturb


class DeviceInfo(BaseModel):
    """Information about a user's device."""
    id: str = Field(..., example="device_abc123", 
                   description="Unique device identifier")
    name: str = Field(..., example="John's iPhone", 
                     description="User-assigned device name")
    type: DeviceType = Field(..., example=DeviceType.MOBILE,
                           description="Type of device")
    status: DeviceStatus = Field(DeviceStatus.ONLINE,
                               description="Current device status")
    last_seen: datetime = Field(..., 
                              example="2023-01-01T12:00:00Z",
                              description="Last activity timestamp")
    ip_address: Optional[str] = Field(None, example="192.168.1.100",
                                     description="Last known IP address")
    user_agent: Optional[str] = Field(None, 
                                     example="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)",
                                     description="Device user agent string")
    app_version: Optional[str] = Field(None, example="1.2.3",
                                      description="Application version")
    os: Optional[str] = Field(None, example="iOS 15.0",
                             description="Operating system and version")
    model: Optional[str] = Field(None, example="iPhone 13",
                                description="Device model")
    is_current: bool = Field(False, 
                           description="Whether this is the current device")
    
    class Config:
        orm_mode = True


class DeviceListResponse(BaseModel):
    """Response model for listing user devices."""
    devices: List[DeviceInfo] = Field(..., 
                                     description="List of user's devices")
    current_device_id: Optional[str] = Field(None,
                                           example="device_abc123",
                                           description="ID of the current device")


class DeviceHeartbeatRequest(BaseModel):
    """Request model for device heartbeat."""
    device_id: str = Field(..., example="device_abc123",
                         description="Unique device identifier")
    device_name: Optional[str] = Field(None, example="John's iPhone",
                                      description="User-assigned device name")
    device_type: Optional[DeviceType] = Field(DeviceType.OTHER,
                                            description="Type of device")
    current_app: Optional[str] = Field(None, example="com.example.app",
                                      description="Current application")
    current_page: Optional[str] = Field(None, example="/dashboard",
                                       description="Current page/route")
    current_url: Optional[HttpUrl] = Field(None, 
                                          example="https://example.com/dashboard",
                                          description="Current URL")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                    description="Additional device metadata")
