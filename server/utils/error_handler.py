"""Structured error handling utilities."""

from enum import Enum
from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel
import time


class ErrorCode(str, Enum):
    """Standardized error codes for the API."""
    
    # Authentication errors (1000-1099)
    INVALID_TOKEN = "AUTH_1001"
    TOKEN_EXPIRED = "AUTH_1002"
    INSUFFICIENT_PERMISSIONS = "AUTH_1003"
    USER_NOT_FOUND = "AUTH_1004"
    INVALID_CREDENTIALS = "AUTH_1005"
    
    # Validation errors (1100-1199)
    INVALID_INPUT = "VALIDATION_1101"
    MISSING_REQUIRED_FIELD = "VALIDATION_1102"
    INVALID_FILE_TYPE = "VALIDATION_1103"
    FILE_TOO_LARGE = "VALIDATION_1104"
    INVALID_DATE_RANGE = "VALIDATION_1105"
    
    # Resource errors (1200-1299)
    RESOURCE_NOT_FOUND = "RESOURCE_1201"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_1202"
    RESOURCE_IN_USE = "RESOURCE_1203"
    RESOURCE_LOCKED = "RESOURCE_1204"
    RESOURCE_EXPIRED = "RESOURCE_1205"
    
    # Database errors (1300-1399)
    DATABASE_CONNECTION_ERROR = "DB_1301"
    DATABASE_TIMEOUT = "DB_1302"
    DATABASE_CONSTRAINT_ERROR = "DB_1303"
    
    # File system errors (1400-1499)
    FILE_NOT_FOUND = "FILE_1401"
    FILE_UPLOAD_FAILED = "FILE_1402"
    FILE_DELETE_FAILED = "FILE_1403"
    FILE_CORRUPTED = "FILE_1404"
    STORAGE_QUOTA_EXCEEDED = "FILE_1405"
    
    # Network errors (1500-1599)
    NETWORK_TIMEOUT = "NETWORK_1501"
    SERVICE_UNAVAILABLE = "NETWORK_1502"
    RATE_LIMIT_EXCEEDED = "NETWORK_1503"
    
    # ML/Training errors (1600-1699)
    TRAINING_FAILED = "ML_1601"
    MODEL_NOT_FOUND = "ML_1602"
    INSUFFICIENT_DATA = "ML_1603"
    MODEL_CORRUPTED = "ML_1604"
    
    # System errors (1700-1799)
    INTERNAL_SERVER_ERROR = "SYSTEM_1701"
    SERVICE_MAINTENANCE = "SYSTEM_1702"
    CONFIGURATION_ERROR = "SYSTEM_1703"


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float
    request_id: Optional[str] = None


class APIError(Exception):
    """Custom API error with structured information."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.request_id = request_id
        super().__init__(message)


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error_code=error_code.value,
        message=message,
        details=details,
        timestamp=time.time(),
        request_id=request_id
    )


def handle_api_error(error: APIError) -> HTTPException:
    """Convert APIError to HTTPException with structured response."""
    response = create_error_response(
        error_code=error.error_code,
        message=error.message,
        details=error.details,
        request_id=error.request_id
    )
    
    return HTTPException(
        status_code=error.status_code,
        detail=response.dict()
    )


# Common error creators
def auth_error(message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None) -> APIError:
    """Create authentication error."""
    return APIError(
        error_code=ErrorCode.INVALID_CREDENTIALS,
        message=message,
        status_code=status.HTTP_401_UNAUTHORIZED,
        details=details
    )


def validation_error(message: str = "Invalid input", details: Optional[Dict[str, Any]] = None) -> APIError:
    """Create validation error."""
    return APIError(
        error_code=ErrorCode.INVALID_INPUT,
        message=message,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=details
    )


def not_found_error(resource: str, identifier: str = "") -> APIError:
    """Create not found error."""
    message = f"{resource} not found"
    if identifier:
        message += f": {identifier}"
    
    return APIError(
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        message=message,
        status_code=status.HTTP_404_NOT_FOUND,
        details={"resource": resource, "identifier": identifier}
    )


def file_error(error_code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None) -> APIError:
    """Create file-related error."""
    status_map = {
        ErrorCode.FILE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
        ErrorCode.FILE_UPLOAD_FAILED: status.HTTP_400_BAD_REQUEST,
        ErrorCode.FILE_DELETE_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCode.FILE_TOO_LARGE: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        ErrorCode.STORAGE_QUOTA_EXCEEDED: status.HTTP_507_INSUFFICIENT_STORAGE,
    }
    
    return APIError(
        error_code=error_code,
        message=message,
        status_code=status_map.get(error_code, status.HTTP_500_INTERNAL_SERVER_ERROR),
        details=details
    )


def database_error(error_code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None) -> APIError:
    """Create database-related error."""
    return APIError(
        error_code=error_code,
        message=message,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        details=details
    )
