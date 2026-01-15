"""
Application startup utilities.

Handles initialization of background services like database health monitoring.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.db_health_monitor import start_database_health_monitor, stop_database_health_monitor
from server.utils.logging_utils import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting application...")
    
    # Start database health monitoring
    try:
        await start_database_health_monitor()
        logger.info("Database health monitor started successfully")
    except Exception as e:
        logger.error(f"Failed to start database health monitor: {e}")
        # Continue without health monitor - don't crash the app
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop database health monitoring
    try:
        await stop_database_health_monitor()
        logger.info("Database health monitor stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping database health monitor: {e}")
    
    logger.info("Application shutdown complete")

def create_app_with_lifespan() -> FastAPI:
    """
    Create FastAPI app with lifespan management.
    
    Returns:
        FastAPI: Application instance with lifespan events configured
    """
    from fastapi import FastAPI
    
    app = FastAPI(
        title="AI-Powered Backend API",
        description="Backend API for AI-powered research platform",
        version="1.0.0",
        lifespan=lifespan
    )
    
    return app
