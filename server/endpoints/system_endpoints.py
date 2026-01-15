"""System endpoints for health checks and system status."""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.utils.logging_utils import log_request_start, log_response, log_error, logger
from server.db import test_database_connection
try:
    from server.db_health_monitor import get_database_health_status, force_database_health_check
except ImportError:
    logger.warning("Database health monitor not available")
    get_database_health_status = None
    force_database_health_check = None

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
        
        # Check database connectivity
        db_status = test_database_connection()
        
        # Get detailed health monitor status if available
        health_monitor_status = None
        if get_database_health_status:
            health_monitor_status = get_database_health_status()
        
        # Determine overall health
        overall_status = "healthy"
        if db_status.get("status") != "connected":
            overall_status = "unhealthy"
        elif health_monitor_status and health_monitor_status.get("failure_count", 0) > 0:
            overall_status = "degraded"
        
        response_data = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "database": db_status,
            "health_monitor": health_monitor_status
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

@router.get(
    "/database/health",
    summary="Database health check",
    description="Detailed database connectivity and performance status",
    tags=["database", "system"]
)
async def database_health_check(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Detailed database health check endpoint.
    
    Purpose: Provide comprehensive database health information for monitoring
    
    Args:
        force_refresh: Force a new health check instead of returning cached status
        
    Returns:
        Dict: Detailed database health information
    """
    try:
        log_request_start("GET", "/database/health", None)
        
        # Force new check if requested
        if force_refresh and force_database_health_check:
            await force_database_health_check()
        
        # Get current database status
        db_status = test_database_connection()
        
        # Get health monitor status if available
        monitor_status = None
        if get_database_health_status:
            monitor_status = get_database_health_status()
        
        response_data = {
            "database": db_status,
            "monitor": monitor_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        log_response(200, "Database health check successful", "/database/health")
        return response_data
        
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        log_error(f"Database health check failed: {str(e)}")
        return {
            "database": {"status": "error", "error": str(e)},
            "monitor": monitor_status,
            "timestamp": datetime.utcnow().isoformat()
        }
