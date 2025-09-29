"""Device management endpoints."""

import json
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.db import get_db
from server.auth import get_current_user
from server.db import User, File
from server.utils.logging_utils import log_request_start, log_response, log_error
from .models import DeviceRegisterRequest, DeviceStatusRequest, DeviceResponse, StandardResponse

router = APIRouter(prefix="/device", tags=["devices"])

@router.post("/register", response_model=DeviceResponse)
async def register_device(
    request: DeviceRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Register a new device for the authenticated user.
    
    Purpose: Register any type of device (Thoth, mobile, desktop, etc.) with the system
    
    Args:
        request: Device registration data
        current_user: Authenticated user
        db: Database session
        
    Returns:
        DeviceResponse: Registration result with device_id and status
        
    Raises:
        HTTPException: 400 if device already exists, 500 on server error
    """
    try:
        log_request_start("POST", "/device/register", current_user.userId)
        
        device_name = request.device_name or f"{request.device_type.title()}-{request.device_id[:8]}"
        
        # Check if device already exists
        existing_device = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename == f"device_{request.device_id}.json"
        ).first()
        
        if existing_device:
            raise HTTPException(status_code=400, detail="Device already registered")
        
        # Create device record
        device_record = {
            "device_id": request.device_id,
            "device_name": device_name,
            "device_type": request.device_type,
            "hardware_info": request.hardware_info or {},
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "status": "online",
            "created_by": current_user.userId
        }
        
        # Save device record
        content = json.dumps(device_record, indent=2).encode('utf-8')
        db_file = File(
            userId=current_user.userId,
            filename=f"device_{request.device_id}.json",
            content=content,
            size=len(content),
            content_type="application/json",
            uploaded_at=datetime.now()
        )
        db.add(db_file)
        db.commit()
        
        log_response("Device registered successfully", 200)
        return {
            "success": True,
            "device_id": request.device_id,
            "device_name": device_name,
            "message": "Device registered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error registering device: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to register device")

@router.get("/list")
async def list_user_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List all registered devices for the authenticated user.
    
    Purpose: Retrieve all devices registered by the current user with their status
    
    Returns:
        Dict containing devices array, count, and success status
    """
    try:
        log_request_start("GET", "/device/list", current_user.userId)
        
        # Get all device files for this user
        device_files = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like("device_%.json")
        ).all()
        
        devices = []
        for file in device_files:
            try:
                device_data = json.loads(file.content.decode('utf-8'))
                # Calculate online status based on last_seen (within 5 minutes)
                last_seen_str = device_data.get('last_seen', '2000-01-01T00:00:00')
                try:
                    last_seen = datetime.fromisoformat(last_seen_str.replace('Z', '+00:00'))
                except ValueError:
                    # Handle different datetime formats
                    last_seen = datetime.fromisoformat(last_seen_str)
                
                is_online = (datetime.now() - last_seen).total_seconds() < 300
                
                devices.append({
                    "device_id": device_data.get("device_id"),
                    "device_name": device_data.get("device_name"),
                    "device_type": device_data.get("device_type"),
                    "status": "online" if is_online else "offline",
                    "last_seen": device_data.get("last_seen"),
                    "hardware_info": device_data.get("hardware_info", {}),
                    "registered_at": device_data.get("registered_at")
                })
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                log_error(f"Error parsing device file {file.filename}: {str(e)}")
                continue
        
        log_response(f"Retrieved {len(devices)} devices", 200)
        return {
            "success": True,
            "devices": devices,
            "count": len(devices)
        }
        
    except Exception as e:
        log_error(f"Error listing devices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve devices")

@router.get("/{device_id}/status")
async def get_device_status(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed status of a specific device.
    
    Purpose: Retrieve comprehensive status information for a registered device
    
    Args:
        device_id: The unique device identifier
        
    Returns:
        Dict containing detailed device status information
    """
    try:
        log_request_start("GET", f"/device/{device_id}/status", current_user.userId)
        
        # Get device record
        device_file = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename == f"device_{device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device_data = json.loads(device_file.content.decode('utf-8'))
        
        # Get latest data upload summary
        data_files = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like(f"data_{device_id}_%.json")
        ).order_by(File.uploaded_at.desc()).limit(5).all()
        
        data_summary = {
            "recent_uploads": len(data_files),
            "last_upload": data_files[0].uploaded_at.isoformat() if data_files else None,
            "total_data_files": len(data_files)
        }
        
        # Calculate online status
        last_seen_str = device_data.get('last_seen', '2000-01-01T00:00:00')
        try:
            last_seen = datetime.fromisoformat(last_seen_str.replace('Z', '+00:00'))
        except ValueError:
            last_seen = datetime.fromisoformat(last_seen_str)
        
        is_online = (datetime.now() - last_seen).total_seconds() < 300
        
        status_info = {
            "success": True,
            "device_id": device_id,
            "device_name": device_data.get("device_name"),
            "device_type": device_data.get("device_type"),
            "status": "online" if is_online else "offline",
            "last_seen": device_data.get("last_seen"),
            "registered_at": device_data.get("registered_at"),
            "hardware_info": device_data.get("hardware_info", {}),
            "battery_level": device_data.get("battery_level"),
            "wifi_connected": device_data.get("wifi_connected"),
            "collection_active": device_data.get("collection_active"),
            "data_summary": data_summary
        }
        
        log_response("Device status retrieved", 200)
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting device status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get device status")

@router.put("/{device_id}/status", response_model=StandardResponse)
async def update_device_status(
    device_id: str,
    request: DeviceStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update the status of a specific device.
    
    Purpose: Allow devices to report their current status and health metrics
    
    Args:
        device_id: The unique device identifier
        request: Status update data
        
    Returns:
        StandardResponse: Update confirmation
    """
    try:
        log_request_start("PUT", f"/device/{device_id}/status", current_user.userId)
        
        # Get device record
        device_file = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename == f"device_{device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not found")
        
        device_data = json.loads(device_file.content.decode('utf-8'))
        
        # Update device status
        device_data["status"] = request.status
        device_data["last_seen"] = datetime.now().isoformat()
        device_data["updated_at"] = datetime.now().isoformat()
        
        # Update optional fields if provided
        if request.battery_level is not None:
            device_data["battery_level"] = request.battery_level
        if request.wifi_connected is not None:
            device_data["wifi_connected"] = request.wifi_connected
        if request.collection_active is not None:
            device_data["collection_active"] = request.collection_active
        
        # Save updated device record
        content = json.dumps(device_data, indent=2).encode('utf-8')
        device_file.content = content
        device_file.size = len(content)
        device_file.uploaded_at = datetime.now()  # Update modification time
        db.commit()
        
        log_response("Device status updated", 200)
        return {
            "success": True,
            "device_id": device_id,
            "message": "Device status updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error updating device status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update device status")

@router.delete("/{device_id}", response_model=StandardResponse)
async def delete_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a device and all its associated data.
    
    Purpose: Remove a device registration and clean up all related data files
    
    Args:
        device_id: The unique device identifier
        
    Returns:
        StandardResponse: Deletion confirmation with file count
    """
    try:
        log_request_start("DELETE", f"/device/{device_id}", current_user.userId)
        
        # Get device record
        device_file = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename == f"device_{device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not found")
        
        deleted_count = 0
        
        # Delete device record
        db.delete(device_file)
        deleted_count += 1
        
        # Delete all data files for this device
        data_files = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like(f"data_{device_id}_%.json")
        ).all()
        
        for file in data_files:
            db.delete(file)
            deleted_count += 1
        
        # Delete all file uploads from this device
        file_uploads = db.query(File).filter(
            File.userId == current_user.userId,
            File.filename.like(f"file_{device_id}_%.%")
        ).all()
        
        for file in file_uploads:
            db.delete(file)
            deleted_count += 1
        
        db.commit()
        
        log_response(f"Device and {deleted_count} files deleted", 200)
        return {
            "success": True,
            "device_id": device_id,
            "deleted_files": deleted_count,
            "message": "Device deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error deleting device: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete device")
