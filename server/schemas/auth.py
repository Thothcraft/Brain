"""Pydantic models for authentication and user management."""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class RegisterRequest(BaseModel):
    """Request model for user registration."""
    username: str = Field(..., min_length=3, max_length=50, example="johndoe")
    password: str = Field(..., min_length=8, example="securepassword123")
    phone_number: Optional[str] = Field(None, example="+1234567890")
    email: Optional[EmailStr] = Field(None, example="user@example.com")
    full_name: Optional[str] = Field(None, example="John Doe")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "password": "securepassword123",
                "phone_number": "+1234567890",
                "email": "john.doe@example.com",
                "full_name": "John Doe"
            }
        }


class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., example="johndoe")
    password: str = Field(..., example="securepassword123")
    device_id: Optional[str] = Field(None, example="device-12345")
    device_name: Optional[str] = Field(None, example="John's iPhone")
    device_type: Optional[str] = Field(None, example="ios")


class TokenResponse(BaseModel):
    """Authentication token response model."""
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field(..., example="bearer")
    expires_in: int = Field(..., example=3600)
    user_id: int = Field(..., example=1)
    username: str = Field(..., example="johndoe")
    role: str = Field(..., example="user")


class UserResponse(BaseModel):
    """User profile response model."""
    id: int = Field(..., example=1)
    username: str = Field(..., example="johndoe")
    email: Optional[str] = Field(None, example="john.doe@example.com")
    phone_number: Optional[str] = Field(None, example="+1234567890")
    full_name: Optional[str] = Field(None, example="John Doe")
    is_active: bool = Field(..., example=True)
    role: str = Field(..., example="user")
    created_at: datetime = Field(..., example="2023-01-01T12:00:00Z")
    updated_at: Optional[datetime] = Field(None, example="2023-01-01T12:00:00Z")

    class Config:
        from_attributes = True


class RegisterResponse(BaseModel):
    """Registration response model."""
    user_id: int = Field(..., example=1)
    username: str = Field(..., example="johndoe")
    message: str = Field(..., example="User registered successfully")


class PasswordResetRequest(BaseModel):
    """Request model for password reset."""
    email: EmailStr = Field(..., example="user@example.com")


class PasswordResetConfirm(BaseModel):
    """Request model for confirming password reset."""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class ChangePasswordRequest(BaseModel):
    """Request model for changing password."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class UpdateProfileRequest(BaseModel):
    """Request model for updating user profile."""
    email: Optional[EmailStr] = Field(None, example="new.email@example.com")
    phone_number: Optional[str] = Field(None, example="+1987654321")
    full_name: Optional[str] = Field(None, example="John Smith")
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        if v is not None and not v.startswith('+'):
            raise ValueError('Phone number must start with +')
        return v


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: List[str] = []


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., example="Error message describing what went wrong")
