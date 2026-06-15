"""
Device Management Endpoints

This module provides endpoints for device registration, status updates, and management.
It handles the communication between Thoth devices and the Brain server.
"""

import json
import time
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import uuid as uuid_lib
from ipaddress import ip_address, IPv4Address
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header, Body
from sqlalchemy.orm import Session
from sqlalchemy import or_

from server.db import get_db, Device, User, File, DeviceFile, DeviceDeployment
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
FREE_PLAN_MAX_ONLINE_DEVICES = 1

# Rate limiting for device registration (new devices only)
# Existing device updates are not rate limited as strictly
REGISTRATION_RATE_LIMIT = {
    'max_attempts': 60,  # Allow frequent updates from registered devices
    'window_seconds': 60  # 1 minute window
}

class DeviceRegistrationError(Exception):
    """Custom exception for device registration errors."""
    pass


def _is_free_plan_user(user: Union[User, Any]) -> bool:
    """
    Treat role=0 as Free plan.
    Paid/organization users can use role values above 0.
    """
    try:
        return int(getattr(user, "role", 0) or 0) == 0
    except (TypeError, ValueError):
        return True


def _can_mark_device_online(
    db: Session,
    user_id: int,
    current_device_id: Optional[int] = None
) -> bool:
    """Return True when user can bring another device online."""
    online_count_query = db.query(Device).filter(
        Device.userId == user_id,
        Device.online == True
    )
    if current_device_id is not None:
        online_count_query = online_count_query.filter(Device.deviceId != current_device_id)
    online_count = online_count_query.count()
    return online_count < FREE_PLAN_MAX_ONLINE_DEVICES

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


def _scan_device_files(device_uuid: str, data_path: str = None, require_metadata: bool = False) -> List[Dict[str, Any]]:
    """Scan device data directory for files and return file information.
    
    Uses content-based file type detection (extension + first-line analysis)
    instead of filename prefix conventions. Files are identified by:
    1. File extension
    2. First-line/header content analysis for CSV files
    3. Content structure analysis for JSON files
    
    Args:
        device_uuid: Device UUID for identification
        data_path: Path to scan (defaults to thoth/data if None)
        require_metadata: If True, only return files with valid .meta.json
    
    Returns:
        List of file information dictionaries with detected types
    """
    from server.file_type_detector import (
        detect_file_type, 
        DetectedFileType,
        get_thoth_metadata_filename,
        validate_thoth_metadata
    )
    
    if not data_path:
        # Default to thoth/data relative to current working directory
        data_path = os.path.join("thoth", "data")
    
    files = []
    
    try:
        if not os.path.exists(data_path):
            logger.warning(f"Data directory not found: {data_path}")
            return files
        
        data_dir = Path(data_path)
        
        # Scan for files in the data directory
        for file_path in data_dir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                
                # Skip metadata files, config files, and system files
                if filename.endswith('.meta.json') or filename.endswith('.brain.json'):
                    continue
                if filename in ['device_id.txt']:
                    continue
                if filename.startswith('.'):
                    continue
                
                try:
                    stat = file_path.stat()
                    
                    # Check for corresponding thoth metadata file
                    meta_filename = get_thoth_metadata_filename(filename)
                    meta_path = data_dir / meta_filename
                    has_metadata = meta_path.exists()
                    
                    # If metadata is required but missing, skip this file
                    if require_metadata and not has_metadata:
                        logger.debug(f"Skipping {filename}: no metadata file")
                        continue
                    
                    # Read file content for type detection (first 8KB)
                    with open(file_path, 'rb') as f:
                        content_sample = f.read(8192)
                    
                    # Detect file type using content-based analysis
                    detection = detect_file_type(content_sample, filename)
                    
                    # Map detected type to file_type string
                    type_mapping = {
                        DetectedFileType.CSI: 'csi',
                        DetectedFileType.GENERAL_CSV: 'csv',
                        DetectedFileType.IMU: 'imu',
                        DetectedFileType.IMAGE: 'image',
                        DetectedFileType.VIDEO: 'video',
                        DetectedFileType.AUDIO: 'audio',
                        DetectedFileType.NUMPY: 'numpy',
                        DetectedFileType.UNKNOWN: 'other',
                    }
                    file_type = type_mapping.get(detection.detected_type, 'other')
                    
                    # Load thoth metadata if available
                    thoth_metadata = None
                    metadata_valid = False
                    if has_metadata:
                        is_valid, thoth_metadata, meta_errors = validate_thoth_metadata(meta_path)
                        metadata_valid = is_valid
                        if not is_valid:
                            logger.warning(f"Invalid metadata for {filename}: {meta_errors}")
                    
                    file_info = {
                        'name': filename,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_type,
                        'path': str(file_path),
                        # New fields from content-based detection
                        'detected_type': detection.detected_type.value,
                        'detection_confidence': detection.confidence,
                        'detection_method': detection.detection_method,
                        'has_metadata': has_metadata,
                        'metadata_valid': metadata_valid,
                    }
                    
                    # Add CSI-specific info if detected
                    if detection.is_csi:
                        file_info['is_csi'] = True
                        file_info['csi_array_length'] = detection.csi_array_length
                        file_info['header_columns'] = detection.header_columns
                    
                    # Add CSV column info for general CSV
                    if detection.detected_type == DetectedFileType.GENERAL_CSV:
                        file_info['header_columns'] = detection.header_columns
                        file_info['column_types'] = detection.statistics.get('column_types', {})
                    
                    # Add validation statistics
                    if detection.statistics:
                        file_info['statistics'] = detection.statistics
                    
                    # Include thoth metadata labels if available
                    if thoth_metadata:
                        labels = thoth_metadata.get('labels', {})
                        if labels:
                            file_info['activity'] = labels.get('activity')
                            file_info['subject_id'] = labels.get('subject_id')
                            file_info['class_name'] = labels.get('class_name')
                    
                    files.append(file_info)
                    
                except Exception as e:
                    logger.error(f"Error scanning file {file_path}: {e}")
                    continue
        
        logger.info(f"Scanned {len(files)} files in {data_path} (content-based detection)")
        
    except Exception as e:
        logger.error(f"Error scanning device files: {e}")
    
    return files

def _auto_sync_device_files(device_id: int, user_id: int, device_uuid: str, db: Session, data_path: str = None):
    """Automatically sync files from device data directory.
    
    Args:
        device_id: Internal device ID
        user_id: User ID who owns the device
        device_uuid: Device UUID string
        db: Database session
        data_path: Path to scan (defaults to thoth/data if None)
    """
    try:
        # Scan files
        scanned_files = _scan_device_files(device_uuid, data_path)
        
        if not scanned_files:
            logger.info(f"No files found to sync for device {device_uuid}")
            return
        
        # Store files using existing function
        _store_device_files(device_id, user_id, device_uuid, scanned_files, db)
        
        logger.info(f"Auto-synced {len(scanned_files)} files for device {device_uuid}")
        
    except Exception as e:
        logger.error(f"Error auto-syncing device files: {e}")

def _get_pending_uploads(device_id: int, db: Session) -> list:
    """Get list of files that have been requested for upload to cloud.
    
    Returns list of filenames that need to be uploaded.
    """
    try:
        pending = db.query(DeviceFile).filter(
            DeviceFile.device_id == device_id,
            DeviceFile.upload_requested == True,
            DeviceFile.on_cloud == False
        ).all()
        
        return [f.filename for f in pending]
    except Exception as e:
        logger.error(f"Error getting pending uploads: {e}")
        return []


def _get_pending_deployments(device_uuid: str, db: Session) -> list:
    """Return pending model deployments for a device (payload without model_data for size)."""
    try:
        records = db.query(DeviceDeployment).filter(
            DeviceDeployment.device_uuid == device_uuid,
            DeviceDeployment.status == "pending"
        ).all()
        result = []
        for r in records:
            try:
                p = json.loads(r.payload)
                result.append(p)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.error(f"Error fetching pending deployments: {e}")
        return []


def _get_file_type_from_extension(filename: str) -> str:
    """Determine file type based on extension.
    
    Returns one of: image, video, audio, sensor, timelapse, other
    """
    import os
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.heic'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    SENSOR_EXTENSIONS = {'.json', '.csv'}
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in SENSOR_EXTENSIONS:
        return 'sensor'
    else:
        return 'other'


def _store_device_files(device_id: int, user_id: int, device_uuid: str, files: list, db: Session):
    """Store file list pushed from device into database.
    
    Args:
        device_id: Internal device ID (primary key)
        user_id: User ID who owns the device
        device_uuid: Device UUID string
        files: List of file info dicts from the device
        db: Database session
    """
    from server.db import DeviceFile
    
    if not files:
        return
    
    try:
        stored_count = 0
        for file_info in files:
            # Skip directories (unless it's a timelapse folder)
            file_type_val = file_info.type if hasattr(file_info, 'type') else file_info.get('type')
            filename = file_info.name if hasattr(file_info, 'name') else file_info.get('name', '')
            
            if not filename:
                continue
            
            # Handle timelapse folders
            if file_type_val == 'timelapse' or filename.startswith('timelapse_'):
                file_type = 'timelapse'
            elif file_type_val == 'directory':
                continue
            else:
                # Determine file type from extension (not prefix)
                # First check if data_type was provided by the device
                data_type = file_info.get('data_type') if isinstance(file_info, dict) else None
                if data_type:
                    file_type = data_type
                else:
                    file_type = _get_file_type_from_extension(filename)
            
            # Parse timestamps
            created_at = None
            modified_at = None
            try:
                created_str = file_info.created if hasattr(file_info, 'created') else file_info.get('created')
                modified_str = file_info.modified if hasattr(file_info, 'modified') else file_info.get('modified')
                if created_str:
                    created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                if modified_str:
                    modified_at = datetime.fromisoformat(modified_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
            
            size = file_info.size if hasattr(file_info, 'size') else file_info.get('size', 0)
            
            # Check if file already exists
            existing = db.query(DeviceFile).filter(
                DeviceFile.device_id == device_id,
                DeviceFile.filename == filename
            ).first()
            
            if existing:
                # Update existing record
                existing.size = size or 0
                existing.modified_at = modified_at
                existing.on_device = True
                existing.last_synced = datetime.utcnow()
            else:
                # Create new record
                device_file = DeviceFile(
                    device_id=device_id,
                    user_id=user_id,
                    filename=filename,
                    size=size or 0,
                    file_type=file_type,
                    created_at=created_at,
                    modified_at=modified_at,
                    on_device=True,
                    on_cloud=False,
                    last_synced=datetime.utcnow()
                )
                db.add(device_file)
            stored_count += 1
        
        db.commit()
        logger.info(f"Stored {stored_count} files for device {device_uuid}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error storing device files: {e}")

@router.post("/register", response_model=DeviceResponse)
async def register_device(
    request: DeviceRegisterRequest,
    db: Session = Depends(get_db),
    user_agent: str = Header(None),
    request_obj: Request = None,
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Register a new device for the authenticated user.
    
    This endpoint allows devices to register with the system. If the device already exists,
    its information will be updated. The device will be marked as online upon registration.
    After successful registration, it will attempt to fetch the list of files from the device.
    
    Authentication is required for device registration.
    """
    try:
        current_user = None
        
        # Require authentication for device registration
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for device registration"
            )
        
        try:
            # Strip "Bearer " prefix if present
            token = authorization
            if authorization.lower().startswith("bearer "):
                token = authorization[7:]
            current_user = await get_user_from_token(token)
            log_request_start("POST", "/device/register", current_user.userId)
        except Exception as e:
            logger.warning(f"Device auth token invalid: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token"
            )
        
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
            
            # Check if device already exists (by UUID for this user)
            user_id = current_user.userId
            existing_device = db.query(Device).filter(
                Device.device_uuid == request.device_id,
                Device.userId == user_id
            ).first()
            
            # Convert device_id to UUID if it's not already in UUID format
            try:
                device_uuid = str(uuid_lib.UUID(request.device_id)) if not isinstance(request.device_id, uuid_lib.UUID) else request.device_id
            except (ValueError, AttributeError):
                # If conversion fails, create a UUID from the string
                device_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, request.device_id))
            
            now = datetime.utcnow()
            
            if existing_device:
                if _is_free_plan_user(current_user) and not _can_mark_device_online(
                    db,
                    user_id,
                    current_device_id=existing_device.deviceId
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Free plan allows only one online device. Disconnect another device or upgrade your plan."
                    )

                # Update existing device
                existing_device.device_name = device_name
                existing_device.device_type = request.device_type or existing_device.device_type
                existing_device.ip_address = ip_address or existing_device.ip_address
                existing_device.mac_address = mac_address or existing_device.mac_address
                existing_device.last_seen = now
                existing_device.online = True
                existing_device.approved = True
                
                # Store hardware_info as JSON if provided
                if hardware_info:
                    existing_device.hardware_info = json.dumps(hardware_info)
                
                db.commit()
                db.refresh(existing_device)
                
                # Store files pushed from device (if provided)
                if request.files:
                    _store_device_files(existing_device.deviceId, user_id, device_uuid, request.files, db)
                else:
                    # Auto-scan files if none provided
                    _auto_sync_device_files(existing_device.deviceId, user_id, device_uuid, db)
                
                # Get pending upload requests and deployments for this device
                pending_uploads = _get_pending_uploads(existing_device.deviceId, db)
                pending_deployments = _get_pending_deployments(device_uuid, db)
                
                logger.info(f"Device updated: {device_uuid} for user {user_id}")
                if pending_uploads:
                    logger.info(f"Pending uploads for device {device_uuid}: {pending_uploads}")
                if pending_deployments:
                    logger.info(f"Pending deployments for device {device_uuid}: {[p.get('deployment_id') for p in pending_deployments]}")
                log_response(200, "Device updated successfully", "/device/register")
                
                return {
                    "success": True,
                    "device_id": device_uuid,
                    "device_name": device_name,
                    "ip_address": ip_address,
                    "message": "Device updated successfully",
                    "pending_uploads": pending_uploads,
                    "pending_deployments": pending_deployments
                }
            
            if _is_free_plan_user(current_user) and not _can_mark_device_online(db, user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Free plan allows only one online device. Disconnect another device or upgrade your plan."
                )
            # Create new device record — not yet approved; user must confirm in portal
            new_device = Device(
                userId=user_id,
                device_uuid=device_uuid,
                device_name=device_name,
                device_type=request.device_type or "unknown",
                ip_address=ip_address,
                mac_address=mac_address,
                last_seen=now,
                online=True,
                approved=True,
                hardware_info=json.dumps(hardware_info) if hardware_info else None
            )
            
            db.add(new_device)
            db.commit()
            db.refresh(new_device)
            
            # Store files pushed from device (if provided)
            if request.files:
                _store_device_files(new_device.deviceId, user_id, device_uuid, request.files, db)
            else:
                # Auto-scan files if none provided
                _auto_sync_device_files(new_device.deviceId, user_id, device_uuid, db)
            
            logger.info(f"New device registered: {device_uuid} for user {user_id}")
            log_response(201, "Device registered successfully", "/device/register")
            
            return {
                "success": True,
                "device_id": device_uuid,
                "device_name": device_name,
                "ip_address": ip_address,
                "message": "Device registered successfully",
                "pending_uploads": [],
                "pending_deployments": []
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

@router.get("/pending")
async def list_pending_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List devices awaiting approval for the authenticated user."""
    try:
        devices = db.query(Device).filter(
            Device.userId == current_user.userId,
            Device.approved == False
        ).all()
        return {
            "success": True,
            "pending": [d.to_dict() for d in devices],
            "count": len(devices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{device_id}/approve")
async def approve_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Approve a pending device so it shows in the main device list."""
    device = db.query(Device).filter(
        Device.device_uuid == device_id,
        Device.userId == current_user.userId
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    device.approved = True
    db.commit()
    logger.info(f"Device {device_id} approved by user {current_user.userId}")
    return {"success": True, "message": "Device approved", "device_id": device_id}


@router.post("/{device_id}/offline")
async def device_offline(
    device_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Device calls this on graceful shutdown or uninstall to immediately mark itself offline.

    No strong auth required — the device already has its own device_id as identity.
    An invalid token just means we skip user-id validation.
    """
    try:
        device = db.query(Device).filter(Device.device_uuid == device_id).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        device.online = False
        db.commit()
        logger.info(f"Device {device_id} marked offline via explicit signal")
        return {"success": True, "message": "Device marked offline"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error marking device offline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{device_id}/deployment/{deployment_id}/ack")
async def ack_deployment(
    device_id: str,
    deployment_id: str,
    status: str = "delivered",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Device calls this after receiving a deployment.
    
    No user auth required — the device uses its own device_id as identity.
    Supports 'delivered', 'declined', and 'pending_confirmation' statuses.
    """
    record = db.query(DeviceDeployment).filter(
        DeviceDeployment.deployment_id == deployment_id,
        DeviceDeployment.device_uuid == device_id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    if status not in ["delivered", "declined", "pending_confirmation"]:
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'delivered', 'declined', or 'pending_confirmation'")
    
    record.status = status
    if status == "delivered":
        record.delivered_at = datetime.utcnow()
        logger.info(f"Deployment {deployment_id} acknowledged by device {device_id}")
        return {"success": True, "message": "Deployment acknowledged"}
    elif status == "declined":
        record.declined_at = datetime.utcnow()
        logger.info(f"Deployment {deployment_id} declined by device {device_id}")
        return {"success": True, "message": "Deployment declined"}
    else:
        logger.info(f"Deployment {deployment_id} received and pending confirmation by device {device_id}")
        return {"success": True, "message": "Deployment received, pending confirmation"}


@router.post("/{device_id}/reject")
async def reject_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Reject (delete) a pending device registration."""
    device = db.query(Device).filter(
        Device.device_uuid == device_id,
        Device.userId == current_user.userId
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    db.query(DeviceFile).filter(DeviceFile.device_id == device.deviceId).delete()
    db.delete(device)
    db.commit()
    logger.info(f"Device {device_id} rejected and removed by user {current_user.userId}")
    return {"success": True, "message": "Device rejected and removed", "device_id": device_id}


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
        # Build query: only show approved devices
        query = db.query(Device).filter(
            Device.userId == current_user.userId,
            Device.approved == True
        )
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
        requested_online = request.status.lower() == "online" if hasattr(request, 'status') else device.online
        if requested_online and _is_free_plan_user(current_user) and not _can_mark_device_online(
            db,
            current_user.userId,
            current_device_id=device.deviceId
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Free plan allows only one online device. Disconnect another device or upgrade your plan."
            )

        update_data = {
            "last_seen": now,
            "online": requested_online,
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


@router.delete("/all", response_model=StandardResponse)
async def delete_all_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete all devices for the current user.
    
    This endpoint removes all registered devices and their associated files
    from the database for the authenticated user.
    """
    try:
        log_request_start("DELETE", "/device/all", current_user.userId)
        
        # Get all devices for this user
        devices = db.query(Device).filter(Device.userId == current_user.userId).all()
        
        if not devices:
            return {
                "success": True,
                "message": "No devices found to delete"
            }
        
        deleted_count = 0
        for device in devices:
            # Delete associated device files first
            db.query(DeviceFile).filter(DeviceFile.device_id == device.deviceId).delete()
            # Delete file device updates
            from server.db import FileDeviceUpdate
            db.query(FileDeviceUpdate).filter(FileDeviceUpdate.deviceId == device.deviceId).delete()
            # Delete the device
            db.delete(device)
            deleted_count += 1
        
        db.commit()
        
        logger.info(f"Deleted {deleted_count} devices for user {current_user.userId}")
        log_response(200, {"success": True, "deleted_count": deleted_count}, "/device/all")
        
        return {
            "success": True,
            "message": f"Successfully deleted {deleted_count} devices",
            "data": {"deleted_count": deleted_count}
        }
        
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting all devices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete devices: {str(e)}"
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


@router.post("/{device_uuid}/sync-files", response_model=StandardResponse)
async def sync_device_files(
    device_uuid: str,
    data_path: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Manually sync files from device data directory.
    
    This endpoint scans the device's data directory and updates the file registry.
    If no data_path is provided, it defaults to 'thoth/data'.
    
    Args:
        device_uuid: The unique device identifier
        data_path: Optional custom path to scan (defaults to thoth/data)
        current_user: Authenticated user
        db: Database session
        
    Returns:
        StandardResponse: Sync result with file count
    """
    try:
        log_request_start("POST", f"/device/{device_uuid}/sync-files", current_user.userId)
        
        # Find the device
        device = db.query(Device).filter(
            Device.device_uuid == device_uuid,
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found or access denied"
            )
        
        # Perform auto-sync
        _auto_sync_device_files(device.deviceId, current_user.userId, device_uuid, db, data_path)
        
        # Get updated file count
        from server.db import DeviceFile
        file_count = db.query(DeviceFile).filter(
            DeviceFile.device_id == device.deviceId,
            DeviceFile.on_device == True
        ).count()
        
        log_response(200, {"success": True, "files_synced": file_count}, f"/device/{device_uuid}/sync-files")
        
        return {
            "success": True,
            "message": f"Files synced successfully for device {device_uuid}",
            "data": {
                "device_uuid": device_uuid,
                "files_synced": file_count,
                "data_path": data_path or "thoth/data"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error syncing device files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync device files: {str(e)}"
        )


@router.get("/scan-files", response_model=Dict[str, Any])
async def scan_local_files(
    data_path: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Scan local data directory for files without requiring a device.
    
    This endpoint scans the data directory and returns file information.
    Useful for testing and general file discovery.
    
    Args:
        data_path: Optional custom path to scan (defaults to thoth/data)
        current_user: Authenticated user
        
    Returns:
        Dict with file information
    """
    try:
        log_request_start("GET", "/device/scan-files", current_user.userId)
        
        # Use a generic device UUID for scanning
        scanner_uuid = "file-scanner"
        
        # Scan files
        scanned_files = _scan_device_files(scanner_uuid, data_path)
        
        # Organize files by type
        files_by_type = {}
        for file_info in scanned_files:
            file_type = file_info['type']
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file_info)
        
        log_response(200, {"files_found": len(scanned_files)}, "/device/scan-files")
        
        return {
            "success": True,
            "message": f"Scanned {len(scanned_files)} files",
            "data": {
                "total_files": len(scanned_files),
                "data_path": data_path or "thoth/data",
                "files_by_type": files_by_type,
                "all_files": scanned_files
            }
        }
        
    except Exception as e:
        log_error(f"Error scanning local files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan local files: {str(e)}"
        )


@router.post("/heartbeat", response_model=StandardResponse)
async def device_heartbeat(
    request: DeviceHeartbeatRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    request_obj: Request = None
) -> Dict[str, Any]:
    """
    Receive heartbeat from a Thoth device.
    
    This endpoint allows devices to send periodic heartbeats to indicate they are online.
    The device's online status and last_seen timestamp are updated in the database.
    
    Args:
        request: Heartbeat data including device_id and optional status updates
        authorization: Device authentication token
        db: Database session
        request_obj: The incoming request object
        
    Returns:
        StandardResponse: Heartbeat confirmation
        
    Raises:
        HTTPException: 401 if not authenticated, 404 if device not found, 500 on server error
    """
    try:
        # Authenticate device
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        try:
            # Strip "Bearer " prefix if present
            token = authorization
            if authorization.lower().startswith("bearer "):
                token = authorization[7:]
            current_user = await get_user_from_token(token)
        except Exception as e:
            logger.warning(f"Device heartbeat auth failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token"
            )
        
        # Get the device
        device = db.query(Device).filter(
            Device.device_uuid == str(request.device_id),
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            logger.warning(f"Heartbeat from unknown device: {request.device_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        
        # Update device status
        now = datetime.utcnow()
        requested_online = request.online if hasattr(request, 'online') and request.online is not None else True
        if requested_online and _is_free_plan_user(current_user) and not _can_mark_device_online(
            db,
            current_user.userId,
            current_device_id=device.deviceId
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Free plan allows only one online device. Disconnect another device or upgrade your plan."
            )

        update_data = {
            "last_seen": now,
            "online": requested_online,
            "updated_at": now
        }
        
        # Update optional fields if provided
        if hasattr(request, 'battery_level') and request.battery_level is not None:
            update_data["battery_level"] = request.battery_level
        if hasattr(request, 'wifi_connected') and request.wifi_connected is not None:
            update_data["wifi_connected"] = request.wifi_connected
        if hasattr(request, 'collection_active') and request.collection_active is not None:
            update_data["collection_active"] = request.collection_active
        if hasattr(request, 'online') and request.online is not None:
            update_data["online"] = request.online
        
        # Update IP address if available
        ip = get_client_ip(request_obj)
        if ip:
            update_data["ip_address"] = ip
        
        # Apply updates
        db.query(Device).filter(Device.deviceId == device.deviceId).update(update_data)
        db.commit()
        
        logger.debug(f"Heartbeat received from device {request.device_id}")
        
        return {
            "success": True,
            "message": "Heartbeat received",
            "data": {
                "device_id": str(request.device_id),
                "timestamp": now.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error processing heartbeat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process heartbeat: {str(e)}"
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


@router.post("/file/{device_file_id}/request-upload")
async def request_file_upload(
    device_file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Request a file to be uploaded from device to cloud.
    
    Sets upload_requested=True on the DeviceFile record.
    The device will see this in its next registration response and upload the file.
    """
    try:
        log_request_start("POST", f"/device/file/{device_file_id}/request-upload", current_user.userId)
        
        # Get the device file
        device_file = db.query(DeviceFile).filter(
            DeviceFile.id == device_file_id,
            DeviceFile.user_id == current_user.userId
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device file not found")
        
        if device_file.on_cloud:
            return {
                "success": True,
                "message": "File already on cloud",
                "cloud_file_id": device_file.cloud_file_id
            }
        
        # Mark for upload
        device_file.upload_requested = True
        db.commit()
        
        logger.info(f"Upload requested for file {device_file.filename} (id={device_file_id})")
        
        return {
            "success": True,
            "message": "Upload requested. File will be uploaded on next device sync.",
            "filename": device_file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error requesting file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to request upload: {str(e)}")


@router.patch("/file/{device_file_id}/type")
async def update_file_type(
    device_file_id: int,
    file_type: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Manually update the type of a device file.
    
    Allowed types: image, video, audio, sensor, timelapse, other
    """
    ALLOWED_TYPES = {'image', 'video', 'audio', 'sensor', 'timelapse', 'other'}
    
    if file_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Must be one of: {', '.join(ALLOWED_TYPES)}"
        )
    
    try:
        log_request_start("PATCH", f"/device/file/{device_file_id}/type", current_user.userId)
        
        device_file = db.query(DeviceFile).filter(
            DeviceFile.id == device_file_id,
            DeviceFile.user_id == current_user.userId
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device file not found")
        
        old_type = device_file.file_type
        device_file.file_type = file_type
        db.commit()
        
        logger.info(f"Updated file type for {device_file.filename}: {old_type} -> {file_type}")
        
        return {
            "success": True,
            "message": f"File type updated to '{file_type}'",
            "file_id": device_file_id,
            "filename": device_file.filename,
            "old_type": old_type,
            "new_type": file_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error updating file type: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update file type: {str(e)}")


@router.delete("/{device_uuid}/files/{file_id}", response_model=Dict[str, Any])
async def delete_device_file(
    device_uuid: str,
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a specific file from a device's file registry.
    
    This removes the file record from the database. It does not delete
    the actual file from the device.
    
    Args:
        device_uuid: The unique device identifier
        file_id: The device file ID to delete
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Dict with deletion confirmation
    """
    try:
        log_request_start("DELETE", f"/device/{device_uuid}/files/{file_id}", current_user.userId)
        
        # Find the device
        device = db.query(Device).filter(
            Device.device_uuid == device_uuid,
            Device.userId == current_user.userId
        ).first()
        
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found or access denied"
            )
        
        # Find and delete the file
        device_file = db.query(DeviceFile).filter(
            DeviceFile.id == file_id,
            DeviceFile.device_id == device.deviceId
        ).first()
        
        if not device_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        filename = device_file.filename
        db.delete(device_file)
        db.commit()
        
        logger.info(f"Deleted device file: {filename} (id={file_id}) from device {device_uuid}")
        
        return {
            "success": True,
            "message": f"File '{filename}' deleted from device registry",
            "file_id": file_id,
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_error(f"Error deleting device file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete device file: {str(e)}"
        )
