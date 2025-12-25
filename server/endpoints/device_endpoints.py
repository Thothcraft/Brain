"""
Device Management Endpoints

This module provides endpoints for device registration, status updates, and management.
It handles the communication between Thoth devices and the Brain server.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import uuid as uuid_lib
from ipaddress import ip_address, IPv4Address

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.orm import Session
from sqlalchemy import or_

from server.db import get_db, Device, User, File
from server.auth import get_current_user, get_user_from_token
from server.utils.logging_utils import log_request_start, log_response, log_error
from .models import (
    DeviceRegisterRequest, 
    DeviceStatusRequest, 
    DeviceResponse, 
    StandardResponse,
    DeviceHeartbeatRequest
)

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/device", tags=["devices"])

# Cache for device authentication tokens
device_auth_cache = {}

# Rate limiting for device registration (new devices only)
# Existing device updates are not rate limited as strictly
REGISTRATION_RATE_LIMIT = {
    'max_attempts': 60,  # Allow frequent updates from registered devices
    'window_seconds': 60  # 1 minute window
}

class DeviceRegistrationError(Exception):
    """Custom exception for device registration errors."""
    pass

def validate_ip_address(ip_str: str) -> bool:
    """Validate an IP address string."""
    try:
        return bool(ip_address(ip_str))
    except ValueError:
        return False

def get_client_ip(request: Request) -> str:
    """Get the client's IP address from the request."""
    if not request:
        return None
        
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        # Get the first IP in the list
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.client.host
    
    return ip if validate_ip_address(ip) else None


def _fetch_device_files_async(device_id: int, ip_address: str, user_id: int, device_uuid: str):
    """Fetch file list from device and store in database (runs in background thread)."""
    import threading
    import requests as http_requests
    
    def fetch_files():
        try:
            # Request file list from Thoth device
            device_url = f"http://{ip_address}:5000/api/files"
            logger.info(f"Fetching files from device at {device_url}")
            
            response = http_requests.get(device_url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch files from device {device_uuid}: {response.status_code}")
                return
            
            data = response.json()
            if data.get('status') != 'success':
                logger.warning(f"Device returned error: {data}")
                return
            
            files = data.get('data', [])
            logger.info(f"Received {len(files)} files from device {device_uuid}")
            
            # Store files in database
            from server.db import SessionLocal, DeviceFile
            db = SessionLocal()
            try:
                for file_info in files:
                    # Skip directories
                    if file_info.get('type') == 'directory':
                        continue
                    
                    filename = file_info.get('name', '')
                    if not filename:
                        continue
                    
                    # Determine file type from prefix
                    file_type = 'other'
                    lower_name = filename.lower()
                    if lower_name.startswith('imu_'):
                        file_type = 'imu'
                    elif lower_name.startswith('csi_'):
                        file_type = 'csi'
                    elif lower_name.startswith('mfcw_'):
                        file_type = 'mfcw'
                    elif lower_name.startswith('img_'):
                        file_type = 'img'
                    elif lower_name.startswith('vid_'):
                        file_type = 'vid'
                    
                    # Parse timestamps
                    created_at = None
                    modified_at = None
                    try:
                        if file_info.get('created'):
                            created_at = datetime.fromisoformat(file_info['created'].replace('Z', '+00:00'))
                        if file_info.get('modified'):
                            modified_at = datetime.fromisoformat(file_info['modified'].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        pass
                    
                    # Check if file already exists
                    existing = db.query(DeviceFile).filter(
                        DeviceFile.device_id == device_id,
                        DeviceFile.filename == filename
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.size = file_info.get('size', 0)
                        existing.modified_at = modified_at
                        existing.on_device = True
                        existing.last_synced = datetime.utcnow()
                    else:
                        # Create new record
                        device_file = DeviceFile(
                            device_id=device_id,
                            user_id=user_id,
                            filename=filename,
                            size=file_info.get('size', 0),
                            file_type=file_type,
                            created_at=created_at,
                            modified_at=modified_at,
                            on_device=True,
                            on_cloud=False,
                            last_synced=datetime.utcnow()
                        )
                        db.add(device_file)
                
                db.commit()
                logger.info(f"Synced {len(files)} files for device {device_uuid}")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Error storing device files: {e}")
            finally:
                db.close()
                
        except http_requests.exceptions.RequestException as e:
            logger.warning(f"Could not reach device {device_uuid} at {ip_address}: {e}")
        except Exception as e:
            logger.error(f"Error fetching device files: {e}")
    
    # Run in background thread to not block registration response
    thread = threading.Thread(target=fetch_files, daemon=True)
    thread.start()

@router.post("/register", response_model=DeviceResponse)
async def register_device(
    request: DeviceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    user_agent: str = Header(None),
    request_obj: Request = None
) -> Dict[str, Any]:
    """
    Register a new device for the authenticated user.
    
    This endpoint allows devices to register with the system. If the device already exists,
    its information will be updated. The device will be marked as online upon registration.
    After successful registration, it will attempt to fetch the list of files from the device.
    """
    try:
        log_request_start("POST", "/device/register", current_user.userId)
        
        # Rate limiting check
        ip = get_client_ip(request_obj)
        if ip:
            current_time = datetime.utcnow()
            cache_key = f"reg_attempt:{ip}"
            
            attempts = device_auth_cache.get(cache_key, [])
            # Remove old attempts outside the time window
            attempts = [t for t in attempts if current_time - t < timedelta(seconds=REGISTRATION_RATE_LIMIT['window_seconds'])]
            
            if len(attempts) >= REGISTRATION_RATE_LIMIT['max_attempts']:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many registration attempts. Please try again later."
                )
            
            attempts.append(current_time)
            device_auth_cache[cache_key] = attempts
        
        # Process device information
        try:
            # Generate a friendly device name if not provided
            device_name = request.device_name or f"{request.device_type or 'Device'}-{request.device_id[:8]}"
            
            # Get additional device info from request
            mac_address = getattr(request, 'mac_address', None)
            
            # Get IP address from request or use the one provided in the request
            ip_address = request.ip_address or get_client_ip(request_obj)
            
            # Store IP in hardware info if available
            hardware_info = request.hardware_info or {}
            if ip_address:
                hardware_info['ip_address'] = ip_address
            if mac_address:
                hardware_info['mac_address'] = mac_address
            
            # Check if device already exists for this user
            existing_device = db.query(Device).filter(
                Device.device_uuid == request.device_id,
                Device.userId == current_user.userId
            ).first()
            
            # Convert device_id to UUID if it's not already in UUID format
            try:
                device_uuid = str(uuid_lib.UUID(request.device_id)) if not isinstance(request.device_id, uuid_lib.UUID) else request.device_id
            except (ValueError, AttributeError):
                # If conversion fails, create a UUID from the string
                device_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, request.device_id))
            
            now = datetime.utcnow()
            
            if existing_device:
                # Update existing device
                existing_device.device_name = device_name
                existing_device.device_type = request.device_type or existing_device.device_type
                existing_device.ip_address = ip_address or existing_device.ip_address
                existing_device.mac_address = mac_address or existing_device.mac_address
                existing_device.last_seen = now
                existing_device.online = True
                
                db.commit()
                db.refresh(existing_device)
                
                # Fetch files from device in background
                if ip_address:
                    _fetch_device_files_async(existing_device.deviceId, ip_address, current_user.userId, device_uuid)
                
                logger.info(f"Device updated: {device_uuid} for user {current_user.userId}")
                log_response(200, "Device updated successfully", "/device/register")
                
                return {
                    "success": True,
                    "device_id": device_uuid,
                    "device_name": device_name,
                    "ip_address": ip_address,
                    "message": "Device updated successfully"
                }
            
            # Create new device record
            new_device = Device(
                userId=current_user.userId,
                device_uuid=device_uuid,
                device_name=device_name,
                device_type=request.device_type or "unknown",
                ip_address=ip_address,
                mac_address=mac_address,
                last_seen=now,
                online=True
            )
            
            db.add(new_device)
            db.commit()
            db.refresh(new_device)
            
            # Fetch files from device in background
            if ip_address:
                _fetch_device_files_async(new_device.deviceId, ip_address, current_user.userId, device_uuid)
            
            logger.info(f"New device registered: {device_uuid} for user {current_user.userId}")
            log_response(201, "Device registered successfully", "/device/register")
            
            return {
                "success": True,
                "device_id": device_uuid,
                "device_name": device_name,
                "ip_address": ip_address,
                "message": "Device registered successfully"
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing device registration: {str(e)}", exc_info=True)
            raise DeviceRegistrationError(f"Error processing registration: {str(e)}")
        
        log_response(200, {"success": True, "device_id": request.device_id, "device_name": device_name}, "/device/register")
    except HTTPException:
        raise
    except DeviceRegistrationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        db.rollback()
        error_msg = f"Failed to register device: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while registering the device. Please try again."
        )

@router.get("/list")
async def list_user_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    include_offline: bool = False
) -> Dict[str, Any]:
    """List all registered devices for the authenticated user.
    
    Args:
        current_user: The authenticated user
        db: Database session
        include_offline: Whether to include offline devices in the results
        
    Returns:
        Dict containing devices array, count, and success status
        
    Raises:
        HTTPException: 500 if there's an error retrieving the device list
    """
    log_request_start("GET", "/device/list", current_user.userId)
    
    try:
        # Build query based on include_offline parameter
        query = db.query(Device).filter(Device.userId == current_user.userId)
        if not include_offline:
            query = query.filter(Device.online == True)
            
        devices = query.all()
        device_list = [device.to_dict() for device in devices]
        
        log_response(200, {
            "success": True,
            "count": len(device_list),
            "devices": device_list
        }, "/device/list")
        
        return {
            "success": True,
            "count": len(device_list),
            "devices": device_list,
            "message": f"Found {len(device_list)} devices"
        }
            
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "user_id": getattr(current_user, 'userId', 'unknown'),
            "user_type": type(current_user).__name__,
            "user_attrs": [attr for attr in dir(current_user) if not attr.startswith('_')]
        }
        log_error(f"Error listing devices: {error_details}")
        
        # For debugging, return the full error details
        # In production, you might want to limit what's returned to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve device list",
                "details": str(e),
                "type": type(e).__name__
            }
        )

@router.put("/{device_id}/status")
async def update_device_status(
    device_id: str,
    request: DeviceStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request_obj: Request = None
) -> StandardResponse:
    """
    Update the status of a specific device.
    
    This endpoint allows updating various status fields of a device, such as
    battery level, WiFi status, and collection status.
    
    Args:
        device_id: The unique device identifier (UUID)
        request: Status update data
        current_user: Authenticated user
        db: Database session
        request_obj: The incoming request object
        
    Returns:
        StandardResponse: Update confirmation
        
    Raises:
        HTTPException: 404 if device not found, 500 on server error
    """
    try:
        log_request_start("PUT", f"/device/{device_id}/status", current_user.userId)
        
        # Get the device
        device = db.query(Device).filter(
            Device.device_uuid == str(device_id),
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            log_error(f"Device not found for user {current_user.userId} and device ID {device_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found or access denied"
            )
        
        # Update device status
        now = datetime.utcnow()
        update_data = {
            "last_seen": now,
            "online": request.status.lower() == "online" if hasattr(request, 'status') else device.online,
            "updated_at": now
        }
        
        # Update optional fields if provided
        if hasattr(request, 'battery_level') and request.battery_level is not None:
            update_data["battery_level"] = request.battery_level
        if hasattr(request, 'wifi_connected') and request.wifi_connected is not None:
            update_data["wifi_connected"] = request.wifi_connected
        if hasattr(request, 'collection_active') and request.collection_active is not None:
            update_data["collection_active"] = request.collection_active
        
        # Update IP address if available
        ip = get_client_ip(request_obj)
        if ip:
            update_data["ip_address"] = ip
        
        # Apply updates
        db.query(Device).filter(Device.deviceId == device.deviceId).update(update_data)
        db.commit()
        
        log_response(200, {"success": True, "message": "Device status updated successfully"}, f"/device/{device_id}/status")
        return {
            "success": True,
            "message": "Device status updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error updating device status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update device status: {str(e)}"
        )

@router.delete("/{device_id}", response_model=StandardResponse)
async def delete_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a device and all its associated data.
    
    Purpose: Remove a device registration and clean up all related data
    
    Args:
        device_id: The unique device identifier (UUID)
        
    Returns:
        StandardResponse: Deletion confirmation
    """
    try:
        log_request_start("DELETE", f"/device/{device_id}", current_user.userId)
        
        # Find the device
        device = db.query(Device).filter(
            Device.device_uuid == device_id,
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found or access denied"
            )
        
        try:
            # First, try to delete the device directly
            db.delete(device)
            db.commit()
            
            log_response(200, {"success": True, "message": "Device deleted"}, f"/device/{device_id}")
            return {
                "success": True,
                "message": "Device deleted successfully"
            }
            
        except Exception as e:
            db.rollback()
            log_error(f"Error during device deletion: {str(e)}")
            
            # If there's a foreign key constraint error, try to delete related records first
            if "foreign key constraint" in str(e).lower() or "violates foreign key" in str(e).lower():
                try:
                    # Use raw SQL to delete related records
                    db.execute("""
                        DELETE FROM device_activity 
                        WHERE device_id = :device_id
                    """, {"device_id": device.device_id})
                    
                    # Now try to delete the device again
                    db.delete(device)
                    db.commit()
                    
                    log_response(200, {"success": True, "message": "Device deleted with cleanup"}, f"/device/{device_id}")
                    return {
                        "success": True,
                        "message": "Device and related data deleted successfully"
                    }
                    
                except Exception as cleanup_error:
                    db.rollback()
                    log_error(f"Error during device deletion cleanup: {str(cleanup_error)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to clean up device data: {str(cleanup_error)}"
                    )
            
            # If it's a different error, re-raise it
            raise
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete device: {str(e)}"
        )


@router.get("/{device_uuid}/files", response_model=Dict[str, Any])
async def get_device_files(
    device_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get all files for a specific device.
    
    Returns files from the DeviceFile table which tracks files on the device.
    Files are marked as on_device=True when on the device, on_cloud=True when uploaded.
    """
    from server.db import DeviceFile
    
    log_request_start("GET", f"/device/{device_uuid}/files", current_user.userId)
    
    try:
        # Find the device
        device = db.query(Device).filter(
            Device.device_uuid == device_uuid,
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        
        # Get all files for this device
        files = db.query(DeviceFile).filter(
            DeviceFile.device_id == device.deviceId
        ).order_by(DeviceFile.modified_at.desc()).all()
        
        file_list = [f.to_dict() for f in files]
        
        return {
            "success": True,
            "device_id": device_uuid,
            "device_name": device.device_name,
            "device_online": device.online,
            "files": file_list,
            "count": len(file_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting device files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get device files: {str(e)}"
        )
