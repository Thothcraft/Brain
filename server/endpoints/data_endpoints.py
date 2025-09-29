"""Data management endpoints."""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from server.db import get_db
from server.auth import get_current_user
from server.db import User, DBFile
from server.utils.logging_utils import log_request_start, log_response, log_error
from .models import DataUploadRequest, DataUploadResponse

router = APIRouter(prefix="/data", tags=["data"])

@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    request: DataUploadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Upload data from a registered device.
    
    Purpose: Store structured data (sensor readings, logs, etc.) from devices
    
    Args:
        request: Data upload request with device_id, data_type, and data array
        
    Returns:
        DataUploadResponse: Upload confirmation with upload_id and data_count
    """
    try:
        log_request_start("POST", "/data/upload", current_user.userId)
        
        upload_timestamp = request.timestamp or datetime.now().isoformat()
        
        # Verify device is registered
        device_file = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename == f"device_{request.device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not registered")
        
        # Create data file with timestamp
        upload_id = f"data_{request.device_id}_{int(time.time())}"
        data_filename = f"{upload_id}.json"
        
        # Validate data structure
        for i, data_point in enumerate(request.data):
            if not isinstance(data_point, dict):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Data point {i} must be an object/dictionary"
                )
        
        data_record = {
            "upload_id": upload_id,
            "device_id": request.device_id,
            "data_type": request.data_type,
            "upload_timestamp": upload_timestamp,
            "data_count": len(request.data),
            "data": request.data,
            "user_id": current_user.userId,
            "version": "1.0"
        }
        
        # Save data file
        content = json.dumps(data_record, indent=2).encode('utf-8')
        db_file = DBFile(
            userId=current_user.userId,
            filename=data_filename,
            content=content,
            size=len(content),
            content_type="application/json",
            uploaded_at=datetime.now()
        )
        db.add(db_file)
        
        # Update device last_seen
        try:
            device_data = json.loads(device_file.content.decode('utf-8'))
            device_data["last_seen"] = upload_timestamp
            device_data["last_data_upload"] = upload_timestamp
            device_file.content = json.dumps(device_data, indent=2).encode('utf-8')
            device_file.size = len(device_file.content)
        except (json.JSONDecodeError, KeyError) as e:
            log_error(f"Error updating device last_seen: {str(e)}")
            # Continue with data upload even if device update fails
        
        db.commit()
        
        log_response(f"Uploaded {len(request.data)} data points", 200)
        return {
            "success": True,
            "upload_id": upload_id,
            "device_id": request.device_id,
            "data_count": len(request.data),
            "message": "Data uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload data")

@router.get("/{device_id}")
async def get_device_data(
    device_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Retrieve data uploaded by a specific device.
    
    Purpose: Fetch paginated data records from a registered device
    
    Args:
        device_id: The device identifier
        limit: Maximum number of records to return (1-1000)
        offset: Number of records to skip
        data_type: Optional filter by data type
        
    Returns:
        Dict containing data_records, total_count, and pagination info
    """
    try:
        log_request_start("GET", f"/data/{device_id}", current_user.userId)
        
        # Verify device exists
        device_file = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename == f"device_{device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # Build query for data files
        query = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename.like(f"data_{device_id}_%.json")
        )
        
        # Get data files for the device with pagination
        data_files = query.order_by(DBFile.uploaded_at.desc()).offset(offset).limit(limit + 1).all()
        
        has_more = len(data_files) > limit
        if has_more:
            data_files = data_files[:limit]
        
        data_records = []
        total_count = 0
        
        for file in data_files:
            try:
                data_record = json.loads(file.content.decode('utf-8'))
                
                # Filter by data type if specified
                if data_type and data_record.get("data_type") != data_type:
                    continue
                
                # Add metadata and sanitize response
                record_info = {
                    "upload_id": data_record.get("upload_id"),
                    "data_type": data_record.get("data_type"),
                    "upload_timestamp": data_record.get("upload_timestamp"),
                    "data_count": data_record.get("data_count", 0),
                    "data": data_record.get("data", []),
                    "file_size": file.size,
                    "uploaded_at": file.uploaded_at.isoformat()
                }
                data_records.append(record_info)
                total_count += record_info["data_count"]
                
            except (json.JSONDecodeError, KeyError) as e:
                log_error(f"Error parsing data file {file.filename}: {str(e)}")
                continue
        
        log_response(f"Retrieved {len(data_records)} data batches with {total_count} total points", 200)
        return {
            "success": True,
            "device_id": device_id,
            "data_records": data_records,
            "total_count": total_count,
            "returned_batches": len(data_records),
            "has_more": has_more,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "next_offset": offset + limit if has_more else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error retrieving device data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data")

@router.get("/analytics/{device_id}")
async def get_data_analytics(
    device_id: str,
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get analytics for device data uploads.
    
    Purpose: Provide statistical analysis of data uploads from a device
    
    Args:
        device_id: The device identifier
        days: Number of days to analyze (1-365)
        
    Returns:
        Dict containing analytics data including totals, averages, and breakdowns
    """
    try:
        log_request_start("GET", f"/data/analytics/{device_id}", current_user.userId)
        
        # Verify device exists
        device_file = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename == f"device_{device_id}.json"
        ).first()
        
        if not device_file:
            raise HTTPException(status_code=404, detail="Device not found")
        
        # Get data files from the specified period
        cutoff_date = datetime.now() - timedelta(days=days)
        data_files = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename.like(f"data_{device_id}_%.json"),
            DBFile.uploaded_at >= cutoff_date
        ).all()
        
        total_uploads = len(data_files)
        total_data_points = 0
        data_by_day = {}
        data_types = {}
        upload_sizes = []
        
        for file in data_files:
            try:
                data_record = json.loads(file.content.decode('utf-8'))
                data_count = data_record.get("data_count", 0)
                data_type = data_record.get("data_type", "unknown")
                
                total_data_points += data_count
                upload_sizes.append(file.size)
                
                # Group by day
                day_key = file.uploaded_at.strftime("%Y-%m-%d")
                if day_key not in data_by_day:
                    data_by_day[day_key] = {"uploads": 0, "data_points": 0}
                data_by_day[day_key]["uploads"] += 1
                data_by_day[day_key]["data_points"] += data_count
                
                # Group by data type
                if data_type not in data_types:
                    data_types[data_type] = {"uploads": 0, "data_points": 0}
                data_types[data_type]["uploads"] += 1
                data_types[data_type]["data_points"] += data_count
                
            except (json.JSONDecodeError, KeyError) as e:
                log_error(f"Error parsing data file {file.filename}: {str(e)}")
                continue
        
        # Calculate additional statistics
        avg_upload_size = sum(upload_sizes) / len(upload_sizes) if upload_sizes else 0
        max_upload_size = max(upload_sizes) if upload_sizes else 0
        min_upload_size = min(upload_sizes) if upload_sizes else 0
        
        analytics = {
            "success": True,
            "device_id": device_id,
            "analysis_period": {
                "days": days,
                "start_date": cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "summary": {
                "total_uploads": total_uploads,
                "total_data_points": total_data_points,
                "average_per_day": total_data_points / max(days, 1),
                "average_per_upload": total_data_points / max(total_uploads, 1)
            },
            "upload_statistics": {
                "average_size_bytes": int(avg_upload_size),
                "max_size_bytes": max_upload_size,
                "min_size_bytes": min_upload_size,
                "total_size_bytes": sum(upload_sizes)
            },
            "data_by_day": data_by_day,
            "data_types": data_types
        }
        
        log_response("Analytics retrieved", 200)
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error getting data analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.delete("/{device_id}/batch/{upload_id}")
async def delete_data_batch(
    device_id: str,
    upload_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a specific data upload batch.
    
    Purpose: Remove a specific data upload batch from storage
    
    Args:
        device_id: The device identifier
        upload_id: The upload batch identifier
        
    Returns:
        Dict containing deletion confirmation
    """
    try:
        log_request_start("DELETE", f"/data/{device_id}/batch/{upload_id}", current_user.userId)
        
        # Find the specific data file
        data_file = db.query(DBFile).filter(
            DBFile.userId == current_user.userId,
            DBFile.filename == f"{upload_id}.json"
        ).first()
        
        if not data_file:
            raise HTTPException(status_code=404, detail="Data batch not found")
        
        # Verify it belongs to the specified device
        try:
            data_record = json.loads(data_file.content.decode('utf-8'))
            if data_record.get("device_id") != device_id:
                raise HTTPException(status_code=403, detail="Data batch does not belong to specified device")
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Corrupted data file")
        
        # Delete the file
        db.delete(data_file)
        db.commit()
        
        log_response("Data batch deleted", 200)
        return {
            "success": True,
            "device_id": device_id,
            "upload_id": upload_id,
            "message": "Data batch deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error deleting data batch: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete data batch")
