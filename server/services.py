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
    db = SessionLocal()
    try:
        # Check for devices that haven't been seen in the last 5 minutes
        stale_time = datetime.utcnow() - timedelta(minutes=5)
        
        # Get devices that haven't been seen recently and are currently online
        stale_devices = db.query(Device).filter(
            Device.last_seen < stale_time,
            Device.online == True
        ).all()
        
        if stale_devices:
            for device in stale_devices:
                device.online = False
                if hasattr(device, 'disconnected_at'):
                    device.disconnected_at = datetime.utcnow()
                if hasattr(device, 'updated_at'):
                    device.updated_at = datetime.utcnow()
                
                logger.info(f"Marked device {getattr(device, 'id', 'unknown')} as offline")
            
            db.commit()
            logger.info(f"Marked {len(stale_devices)} devices as offline")
        
        return len(stale_devices)
        
    except Exception as e:
        logger.error(f"Error in auto_disconnect_stale_devices: {e}")
        db.rollback()
        return 0
    finally:
        db.close()


def start_scheduler():
    """Start the background scheduler for periodic tasks."""
    global scheduler
    if scheduler is not None:
        logger.warning("Scheduler already running")
        return scheduler
    
    try:
        scheduler = BackgroundScheduler()
        
        # Add status update job (every ~3.3 hours)
        scheduler.add_job(
            send_status,
            trigger=IntervalTrigger(minutes=200),
            id='send_status_job',
            name='Send status periodically',
            replace_existing=True
        )
        
        # Add device status check job (every 2 minutes)
        scheduler.add_job(
            auto_disconnect_stale_devices,
            trigger=IntervalTrigger(minutes=2),
            id='auto_disconnect_job',
            name='Auto disconnect stale devices',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("Scheduler started successfully with jobs: %s", 
                   [job.name for job in scheduler.get_jobs()])
        
        # Register shutdown handler
        import atexit
        atexit.register(lambda: scheduler.shutdown() if scheduler else None)
        
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}", exc_info=True)
        if scheduler is not None:
            try:
                scheduler.shutdown()
            except:
                pass
            scheduler = None
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
