"""Service layer for handling business logic and utility functions.

This module contains non-endpoint functions that handle core business logic,
background tasks, and utility functions used across the application.
"""

import os
import logging
import json
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from urllib.parse import unquote, quote

# FastAPI
from fastapi import HTTPException, status, UploadFile
from fastapi.responses import FileResponse

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Twilio
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Application imports
from server.db import SessionLocal, Device, User, File as DBFile, Query, Session
from server.config import settings
from server.utils import (
    log_something,
    log_error,
    log_server_health,
    log_server_lifecycle,
    compute_sha256
)
from server.auth import get_password_hash, create_access_token

# AI Agent imports
from aiagent.handler import query as ai_query_handler
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context.reference import read_references

# Initialize logger
logger = logging.getLogger(__name__)


# Initialize scheduler
scheduler = None


def get_status_message() -> str:
    """Generate a status message with current time and user count.
    
    Returns:
        str: Status message
    """
    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Thoth API is running. Users: {user_count}, Time: {current_time}"
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return f"Thoth API is running. Error getting status: {e}"
    finally:
        db.close()




def send_status(message: str = "", to_phone_number: str = ""):
    """Send status update to all connected devices."""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        default_message = f"Thoth API Status: Running as of {current_time}"
        message = message or default_message
        recipient_phone = "+18073587137" 
        recipient_phone = to_phone_number or recipient_phone  # Hardcoded E.164 format
        success = send_twilio_message(recipient_phone, message)
        if not success:
            log_error(f"[send_status] Failed to send SMS to {recipient_phone}")
        else:
            log_something(f"[send_status] SMS sent to {recipient_phone} at {current_time}", endpoint="send_status")
    except Exception as e:
        log_error(f"[send_status] Error: {e}")


def auto_disconnect_stale_devices():
    """Mark devices as offline if they haven't sent a heartbeat recently."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError, DBAPIError
    
    db = None
    start_time = datetime.utcnow()
    logger.info(f"[SERVICE] Starting auto_disconnect_stale_devices at {start_time}")
    
    try:
        # Use a fresh connection with timeout
        db = SessionLocal()
        
        # Quick connection test with timeout
        try:
            result = db.execute(text("SELECT 1"))
            logger.info("[SERVICE] Database connection test passed")
        except (OperationalError, DBAPIError) as conn_err:
            logger.warning(f"[SERVICE] Database connection test failed, skipping this run: {conn_err}")
            return 0
        
        # Check for devices that haven't been seen in the last 5 minutes
        stale_time = datetime.utcnow() - timedelta(minutes=5)
        
        # Get devices that haven't been seen recently and are currently online
        stale_devices = db.query(Device).filter(
            Device.last_seen < stale_time,
            Device.online == True
        ).all()
        
        logger.info(f"[SERVICE] Found {len(stale_devices)} stale devices")
        
        if stale_devices:
            for device in stale_devices:
                device.online = False
                if hasattr(device, 'disconnected_at'):
                    device.disconnected_at = datetime.utcnow()
                if hasattr(device, 'updated_at'):
                    device.updated_at = datetime.utcnow()
                
                logger.info(f"[SERVICE] Marked device {getattr(device, 'id', 'unknown')} as offline")
            
            try:
                db.commit()
                logger.info(f"[SERVICE] Successfully marked {len(stale_devices)} devices as offline")
            except Exception as commit_err:
                logger.error(f"[SERVICE] Failed to commit device updates: {commit_err}")
                db.rollback()
                return 0
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"[SERVICE] Completed auto_disconnect_stale_devices in {duration:.2f}s")
        return len(stale_devices) if stale_devices else 0
        
    except (OperationalError, DBAPIError) as db_err:
        logger.warning(f"[SERVICE] Database error in auto_disconnect_stale_devices: {db_err}")
        if db:
            try:
                db.rollback()
                logger.info("[SERVICE] Database rollback completed")
            except Exception as rollback_err:
                logger.error(f"[SERVICE] Failed to rollback: {rollback_err}")
        return 0
    except Exception as e:
        logger.error(f"[SERVICE] Unexpected error in auto_disconnect_stale_devices: {e}")
        if db:
            try:
                db.rollback()
            except:
                pass
        return 0
    finally:
        if db:
            try:
                db.close()
                logger.info("[SERVICE] Database connection closed")
            except:
                pass


def stop_scheduler():
    """Stop the background scheduler gracefully."""
    global scheduler
    if scheduler is not None:
        try:
            if scheduler.running:
                scheduler.shutdown(wait=False)
                logger.info("Scheduler stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping scheduler: {e}")
        finally:
            scheduler = None


def start_scheduler():
    """Start the background scheduler for periodic tasks."""
    global scheduler
    
    # Stop existing scheduler if running
    if scheduler is not None:
        if scheduler.running:
            logger.warning("Scheduler already running")
            return scheduler
        else:
            # Scheduler exists but not running, clean it up
            stop_scheduler()
    
    try:
        from apscheduler.executors.pool import ThreadPoolExecutor
        
        # Configure with explicit executor settings
        executors = {
            'default': ThreadPoolExecutor(max_workers=2)
        }
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 60
        }
        
        scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults
        )
        
        # Add device status check job (every 2 minutes to reduce database load)
        scheduler.add_job(
            auto_disconnect_stale_devices,
            trigger=IntervalTrigger(seconds=120),
            id='auto_disconnect_job',
            name='Auto disconnect stale devices',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("Scheduler started successfully with jobs: %s", 
                   [job.name for job in scheduler.get_jobs()])
        
        # Register shutdown handler
        import atexit
        atexit.register(stop_scheduler)
        
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}", exc_info=True)
        stop_scheduler()
        raise


def send_twilio_message(to_phone_number: str, message: str) -> Dict[str, Any]:
    """Send an SMS message using Twilio.
    
    Args:
        to_phone_number: Recipient's phone number in E.164 format
        message: The message content to send
        
    Returns:
        Dict containing the message SID and status if successful
        
    Raises:
        HTTPException: If there's an error sending the message
    """
    try:
        # Initialize Twilio client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        # Send message
        twilio_message = client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )
        
        return {
            "message_sid": twilio_message.sid,
            "status": twilio_message.status,
            "to": twilio_message.to,
            "date_created": twilio_message.date_created.isoformat() if twilio_message.date_created else None
        }
        
    except TwilioRestException as e:
        logger.error(f"Twilio API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error sending Twilio message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending the message"
        )
