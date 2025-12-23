"""System endpoints for health checks and system status."""

import os
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.utils.logging_utils import log_request_start, log_response, log_error, logger

router = APIRouter(prefix="", tags=["system"])

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns the health status of the API",
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "API is unhealthy"}
    }
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify API status.
    
    Purpose: Provide system health status for monitoring and load balancers
    
    Returns:
        HealthCheckResponse: System health information
    """
    try:
        log_request_start("GET", "/health", None)
        
        # You can add more sophisticated health checks here
        # For example, checking database connectivity, external services, etc.
        
        response_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
        
        log_response(200, "Health check successful", "/health")
        return response_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        log_error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )

@router.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information",
    tags=["system"]
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.
    
    Purpose: Provide basic API information and welcome message
    
    Returns:
        Dict: API information and available endpoints
    """
    try:
        log_request_start("GET", "/", None)
        
        response_data = {
            "message": "Welcome to the AI-Powered Backend API",
            "version": "1.0.0",
            "documentation": {
                "swagger": "/api-docs",
                "redoc": "/api-redoc"
            },
            "endpoints": {
                "health": "/health",
                "auth": "/token",
                "devices": "/device/*",
                "data": "/data/*", 
                "files": "/file/*"
            }
        }
        
        log_response(200, "Root endpoint accessed", "/")
        return response_data
        
    except Exception as e:
        log_error(f"Root endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
