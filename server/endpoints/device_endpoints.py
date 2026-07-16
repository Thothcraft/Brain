"""
Device Management Endpoints

This module provides endpoints for device registration, status updates, and management.
It handles the communication between Thoth devices and the Brain server.
"""

import json
import time
import logging
import os
import re
import asyncio
from datetime import datetime, timedelta, timezone
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
from server.calibration import REGIONS as CALIBRATION_REGIONS, derive_thresholds
from server.inventory import is_newer_snapshot
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
DEVICE_ONLINE_TIMEOUT_SECONDS = max(30, int(os.getenv("DEVICE_ONLINE_TIMEOUT_SECONDS", "90")))


def _expire_stale_devices(db: Session, user_id: Optional[int] = None) -> int:
    """Make persisted device state agree with the heartbeat freshness contract."""
    cutoff = datetime.utcnow() - timedelta(seconds=DEVICE_ONLINE_TIMEOUT_SECONDS)
    query = db.query(Device).filter(Device.online == True).filter(
        (Device.last_seen.is_(None)) | (Device.last_seen < cutoff)
    )
    if user_id is not None:
        query = query.filter(Device.userId == user_id)
    changed = query.update({Device.online: False}, synchronize_session=False)
    if changed:
        db.commit()
    return changed
MINUTE_DIR_RE = re.compile(r"^\d{8}_\d{4}$")
MINUTE_FILE_RE = re.compile(r"^\d{8}_\d{4}_.+")
MINUTE_PATH_RE = re.compile(r"(^|/)\d{8}_\d{4}(/|_|$)")
CAPTURE_SENSOR_KEYS = ("usb_camera", "dreamhat_radar", "esp32_csi", "sense_hat")
DEFAULT_CAPTURE_SETTINGS = {
    "labels": [],
    "sensors": {
        "usb_camera": True,
        "dreamhat_radar": True,
        "esp32_csi": True,
        "sense_hat": True,
    },
    "radar_detection_threshold_normalized": 0.45,
    "occupancy_threshold_percent": 50.0,
    "yellow_threshold_percent": 20.0,
    "green_threshold_percent": 60.0,
    "auto_occupancy_label_enabled": True,
    "chunk_seconds": 10.0,
    "system_mode": "balanced",
    "occupancy_vote_chunks": 1,
    "prediction_label_style": "occupancy",
    "people_count_label_enabled": False,
    "sleep_study_enabled": False,
    "calibrations": {},
    "revision": 0,
    "updated_at": None,
}

# Rate limiting for device registration (new devices only)
# Existing device updates are not rate limited as strictly
REGISTRATION_RATE_LIMIT = {
    'max_attempts': 60,  # Allow frequent updates from registered devices
    'window_seconds': 60  # 1 minute window
}

class DeviceRegistrationError(Exception):
    """Custom exception for device registration errors."""
    pass


PLAN_DEVICE_LIMITS = {"free": 1, "home": 5, "pro": 10, "research": 10}
PLAN_FEATURES = {
    "free": {"basic_occupancy", "maps", "home_assistant", "labels", "data_export", "calibration"},
    "home": {"basic_occupancy", "presence", "maps", "home_assistant", "labels", "data_export", "calibration", "predictions", "zones", "spaces", "multi_device"},
    "pro": {"basic_occupancy", "presence", "maps", "home_assistant", "labels", "data_export", "calibration", "predictions", "zones", "spaces", "multi_device", "har", "people_count", "ai_models", "federated_learning"},
    "research": {"basic_occupancy", "presence", "maps", "home_assistant", "labels", "data_export", "calibration", "predictions", "zones", "spaces", "multi_device", "har", "people_count", "ai_models", "federated_learning", "detailed_labels", "academy", "assistant"},
}


def _product_plan(user: Union[User, Any]) -> str:
    """Resolve product access from verified billing state, never admin role."""
    aliases = {"researcher": "research", "organization": "pro"}
    plan = aliases.get(str(getattr(user, "plan", "free") or "free").lower(), str(getattr(user, "plan", "free") or "free").lower())
    return plan if plan in PLAN_DEVICE_LIMITS else "free"


def _require_feature(user: Union[User, Any], feature: str) -> None:
    if feature not in PLAN_FEATURES[_product_plan(user)]:
        raise HTTPException(status_code=403, detail=f"{feature.replace('_', ' ').title()} requires a paid plan")


def _portal_upload_allowed_for_device(device: Device) -> bool:
    """Return whether the device allows portal-initiated uploads."""
    try:
        if not device or not device.hardware_info:
            return True
        hw_info = json.loads(device.hardware_info) if isinstance(device.hardware_info, str) else device.hardware_info
        if isinstance(hw_info, dict) and "portal_upload_allowed" in hw_info:
            return bool(hw_info.get("portal_upload_allowed", True))
    except Exception:
        logger.debug("Unable to read portal upload flag from hardware_info", exc_info=True)
    return True


def _is_minute_dir(path: Path) -> bool:
    return path.is_dir() and MINUTE_DIR_RE.match(path.name) is not None


def _is_minute_file_name(filename: str) -> bool:
    value = filename or ""
    return MINUTE_FILE_RE.match(value) is not None or MINUTE_PATH_RE.search(value) is not None


def _device_hardware_info(device: Device) -> Dict[str, Any]:
    try:
        if device and device.hardware_info:
            loaded = json.loads(device.hardware_info) if isinstance(device.hardware_info, str) else device.hardware_info
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        logger.debug("Unable to parse hardware_info for device %s", getattr(device, "device_uuid", None), exc_info=True)
    return {}


def _normalize_capture_settings(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    labels_value = source.get("labels")
    if isinstance(labels_value, str):
        labels = [item.strip() for item in labels_value.split(",") if item.strip()]
    elif isinstance(labels_value, list):
        labels = [str(item).strip() for item in labels_value if str(item).strip()]
    else:
        label = str(source.get("label") or "").strip()
        labels = [label] if label else []

    raw_sensors = source.get("sensors") if isinstance(source.get("sensors"), dict) else {}
    sensors = dict(DEFAULT_CAPTURE_SETTINGS["sensors"])
    for key in CAPTURE_SENSOR_KEYS:
        if key in raw_sensors:
            sensors[key] = bool(raw_sensors.get(key))

    try:
        radar_value = source.get("radar_detection_threshold_normalized")
        if radar_value is None and "radar_detection_threshold_db" in source:
            radar_value = float(source["radar_detection_threshold_db"]) / 10.0
        radar_threshold = min(0.95, max(0.05, float(radar_value if radar_value is not None else 0.45)))
    except (TypeError, ValueError):
        radar_threshold = 0.45
    try:
        occupancy_threshold = min(100.0, max(0.0, float(source.get("occupancy_threshold_percent", 50.0))))
    except (TypeError, ValueError):
        occupancy_threshold = 50.0
    try:
        yellow_threshold = min(100.0, max(0.0, float(source.get("yellow_threshold_percent", 20.0))))
        green_threshold = min(100.0, max(0.0, float(source.get("green_threshold_percent", 60.0))))
        if yellow_threshold >= green_threshold:
            raise ValueError("yellow threshold must be below green threshold")
    except (TypeError, ValueError):
        yellow_threshold, green_threshold = 20.0, 60.0
    auto_label = source.get("auto_occupancy_label_enabled", True)
    if isinstance(auto_label, str):
        auto_label = auto_label.strip().lower() in {"1", "true", "yes", "on"}
    try:
        chunk_seconds = min(30.0, max(2.0, float(source.get("chunk_seconds", 10.0))))
    except (TypeError, ValueError):
        chunk_seconds = 10.0
    system_mode = str(source.get("system_mode") or "balanced").strip().lower()
    if system_mode not in {"responsive", "balanced", "precision"}:
        system_mode = "balanced"
    try:
        occupancy_vote_chunks = min(60, max(1, int(source.get("occupancy_vote_chunks", 1))))
    except (TypeError, ValueError):
        occupancy_vote_chunks = 1
    prediction_label_style = str(source.get("prediction_label_style") or "occupancy").strip().lower()
    if prediction_label_style not in {"occupancy", "presence"}:
        prediction_label_style = "occupancy"
    people_count_label = source.get("people_count_label_enabled", False)
    sleep_study = source.get("sleep_study_enabled", False)
    if isinstance(people_count_label, str):
        people_count_label = people_count_label.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(sleep_study, str):
        sleep_study = sleep_study.strip().lower() in {"1", "true", "yes", "on"}
    try:
        revision = max(0, int(source.get("revision") or 0))
    except (TypeError, ValueError):
        revision = 0

    return {
        "labels": labels,
        "sensors": sensors,
        "radar_detection_threshold_normalized": radar_threshold,
        "occupancy_threshold_percent": occupancy_threshold,
        "yellow_threshold_percent": yellow_threshold,
        "green_threshold_percent": green_threshold,
        "auto_occupancy_label_enabled": bool(auto_label),
        "chunk_seconds": chunk_seconds,
        "system_mode": system_mode,
        "occupancy_vote_chunks": occupancy_vote_chunks,
        "prediction_label_style": prediction_label_style,
        "people_count_label_enabled": bool(people_count_label),
        "sleep_study_enabled": bool(sleep_study),
        "calibrations": source.get("calibrations") if isinstance(source.get("calibrations"), dict) else {},
        "revision": revision,
        "updated_at": str(source.get("updated_at")) if source.get("updated_at") else None,
    }


def _capture_settings_sort_key(value: Dict[str, Any]) -> tuple[int, str]:
    normalized = _normalize_capture_settings(value)
    return int(normalized.get("revision") or 0), str(normalized.get("updated_at") or "")


def _reconcile_capture_settings(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Select the latest revision so offline device edits cannot be lost."""
    current_normalized = _normalize_capture_settings(current)
    incoming_normalized = _normalize_capture_settings(incoming)
    return incoming_normalized if _capture_settings_sort_key(incoming_normalized) > _capture_settings_sort_key(current_normalized) else current_normalized


def _capture_settings_for_device(device: Device) -> Dict[str, Any]:
    hardware_info = _device_hardware_info(device)
    return _normalize_capture_settings(hardware_info.get("capture_settings"))


def _set_capture_settings_for_device(
    device: Device, updates: Dict[str, Any], db: Session, *, increment_revision: bool = True
) -> Dict[str, Any]:
    hardware_info = _device_hardware_info(device)
    current = _capture_settings_for_device(device)
    incoming = dict(updates or {})
    if (
        "radar_detection_threshold_normalized" not in incoming
        and "radar_detection_threshold_db" in incoming
    ):
        try:
            incoming["radar_detection_threshold_normalized"] = (
                float(incoming["radar_detection_threshold_db"]) / 10.0
            )
        except (TypeError, ValueError):
            pass
    merged = _normalize_capture_settings({
        **current,
        **incoming,
    })
    if increment_revision:
        merged["revision"] = max(int(current.get("revision") or 0), int(merged.get("revision") or 0)) + 1
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    hardware_info["capture_settings"] = merged
    device.hardware_info = json.dumps(hardware_info)
    db.commit()
    db.refresh(device)
    return merged


def _can_mark_device_online(
    db: Session,
    user: Union[User, Any],
    current_device_id: Optional[int] = None
) -> bool:
    """Return True when user can bring another device online."""
    online_count_query = db.query(Device).filter(
        Device.userId == user.userId,
        Device.online == True
    )
    if current_device_id is not None:
        online_count_query = online_count_query.filter(Device.deviceId != current_device_id)
    online_count = online_count_query.count()
    return online_count < PLAN_DEVICE_LIMITS[_product_plan(user)]

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
        data_path = os.path.join("thoth", "data")

    files: List[Dict[str, Any]] = []

    try:
        if not os.path.exists(data_path):
            # Only log as debug since missing local data directory is expected for remote devices
            logger.debug(f"Data directory not found: {data_path} (expected for remote devices)")
            return files

        data_dir = Path(data_path)
        if _is_minute_dir(data_dir):
            minute_dirs = [data_dir]
        else:
            minute_dirs = sorted([item for item in data_dir.iterdir() if _is_minute_dir(item)], key=lambda item: item.name)

        for minute_dir in minute_dirs:
            minute_label = None
            if minute_dir.parent != data_dir and minute_dir.parent.name and not _is_minute_dir(minute_dir.parent):
                minute_label = minute_dir.parent.name
            for file_path in sorted(minute_dir.iterdir()):
                if not file_path.is_file():
                    continue

                raw_name = file_path.name
                if raw_name.endswith('.meta.json') or raw_name.endswith('.brain.json'):
                    continue
                if raw_name in ['device_id.txt'] or raw_name.startswith('.'):
                    continue

                filename = f"{minute_dir.name}_{raw_name}"

                try:
                    stat = file_path.stat()
                    relative_path = file_path.relative_to(data_dir).as_posix()

                    meta_filename = get_thoth_metadata_filename(raw_name)
                    meta_path = minute_dir / meta_filename
                    has_metadata = meta_path.exists()

                    if require_metadata and not has_metadata:
                        logger.debug(f"Skipping {filename}: no metadata file")
                        continue

                    with open(file_path, 'rb') as f:
                        content_sample = f.read(8192)

                    detection = detect_file_type(content_sample, raw_name)

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

                    thoth_metadata = None
                    metadata_valid = False
                    if has_metadata:
                        is_valid, thoth_metadata, meta_errors = validate_thoth_metadata(meta_path)
                        metadata_valid = is_valid
                        if not is_valid:
                            logger.warning(f"Invalid metadata for {filename}: {meta_errors}")

                    file_info = {
                        'name': filename,
                        'minute': minute_dir.name,
                        'relative_name': raw_name,
                        'relative_path': relative_path,
                        'label': minute_label,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': file_type,
                        'path': str(file_path),
                        'detected_type': detection.detected_type.value,
                        'detection_confidence': detection.confidence,
                        'detection_method': detection.detection_method,
                        'has_metadata': has_metadata,
                        'metadata_valid': metadata_valid,
                    }

                    if detection.is_csi:
                        file_info['is_csi'] = True
                        file_info['csi_array_length'] = detection.csi_array_length
                        file_info['header_columns'] = detection.header_columns

                    if detection.detected_type == DetectedFileType.GENERAL_CSV:
                        file_info['header_columns'] = detection.header_columns
                        file_info['column_types'] = detection.statistics.get('column_types', {})

                    if detection.statistics:
                        file_info['statistics'] = detection.statistics

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

        logger.info(f"Scanned {len(files)} minute files in {data_path} (minute-only detection)")

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

        scanned_names = {str(file_info.get('name', '')) for file_info in scanned_files if isinstance(file_info, dict)}
        stale_files = db.query(DeviceFile).filter(DeviceFile.device_id == device_id).all()
        removed = 0
        updated = 0
        for record in stale_files:
            if not _is_minute_file_name(record.filename):
                db.delete(record)
                removed += 1
                continue
            if record.filename not in scanned_names and record.on_device:
                record.on_device = False
                record.last_synced = datetime.utcnow()
                updated += 1

        if removed or updated:
            db.commit()
            logger.info(
                f"Reconciled device files for {device_uuid}: removed={removed}, marked_offline={updated}"
            )
        
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


def _store_device_files(device_id: int, user_id: int, device_uuid: str, files: list, db: Session, *, commit: bool = True):
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
        device = db.query(Device).filter(Device.deviceId == device_id).first()
        hardware_info = _device_hardware_info(device) if device else {}
        capture_metadata = hardware_info.get("capture_file_metadata") if isinstance(hardware_info.get("capture_file_metadata"), dict) else {}
        for file_info in files:
            # Skip directories (unless it's a timelapse folder)
            file_type_val = file_info.type if hasattr(file_info, 'type') else file_info.get('type')
            filename = file_info.name if hasattr(file_info, 'name') else file_info.get('name', '')
            
            if not filename:
                continue
            def field(name, default=None):
                return getattr(file_info, name, default) if not isinstance(file_info, dict) else file_info.get(name, default)
            labels = field('labels') if isinstance(field('labels'), list) else []
            label = str(field('label') or '').strip()
            if label and label not in labels:
                labels.append(label)
            capture_metadata[filename] = {
                'labels': [str(item).strip() for item in labels if str(item).strip()],
                'label': label or (str(labels[0]) if labels else None),
                'occupancy': field('occupancy') if isinstance(field('occupancy'), dict) else None,
                'progress': field('progress') if isinstance(field('progress'), dict) else None,
            }
            
            # Handle timelapse folders
            if file_type_val == 'timelapse' or filename.startswith('timelapse_'):
                file_type = 'timelapse'
            elif file_type_val == 'directory':
                continue
            else:
                # Determine file type from extension (not prefix)
                # First check if data_type was provided by the device
                data_type = (
                    getattr(file_info, 'data_type', None)
                    if not isinstance(file_info, dict)
                    else file_info.get('data_type')
                )
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
        
        if device:
            hardware_info['capture_file_metadata'] = capture_metadata
            device.hardware_info = json.dumps(hardware_info)
        if commit:
            db.commit()
        logger.info(f"Stored {stored_count} files for device {device_uuid}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error storing device files: {e}")
        if not commit:
            raise


def _apply_inventory_snapshot(device: Device, user_id: int, request: DeviceHeartbeatRequest, db: Session) -> bool:
    """Atomically reconcile a complete, monotonically revisioned device inventory."""
    if not request.inventory_complete or request.files is None or request.inventory_revision is None:
        return False
    hardware_info = _device_hardware_info(device)
    current = hardware_info.get("inventory_snapshot") if isinstance(hardware_info.get("inventory_snapshot"), dict) else {}
    incoming_revision = int(request.inventory_revision)
    incoming_timestamp = str(request.inventory_timestamp or "")
    if not is_newer_snapshot(incoming_revision, incoming_timestamp, current):
        logger.info("Ignored stale inventory snapshot for %s at revision %s", device.device_uuid, incoming_revision)
        return False

    _store_device_files(device.deviceId, user_id, str(device.device_uuid), request.files, db, commit=False)
    present_names = {str(item.name) for item in request.files if item.name}
    now = datetime.utcnow()
    for record in db.query(DeviceFile).filter(DeviceFile.device_id == device.deviceId).all():
        if record.filename not in present_names and record.on_device:
            record.on_device = False
            record.last_synced = now
    device = db.query(Device).filter(Device.deviceId == device.deviceId).first()
    hardware_info = _device_hardware_info(device)
    metadata = hardware_info.get("capture_file_metadata") if isinstance(hardware_info.get("capture_file_metadata"), dict) else {}
    hardware_info["capture_file_metadata"] = {name: value for name, value in metadata.items() if name in present_names}
    hardware_info["inventory_snapshot"] = {
        "revision": incoming_revision, "timestamp": incoming_timestamp,
        "received_at": datetime.now(timezone.utc).isoformat(), "count": len(present_names),
    }
    device.hardware_info = json.dumps(hardware_info)
    db.commit()
    return True

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
            if 'portal_upload_allowed' not in hardware_info:
                hardware_info['portal_upload_allowed'] = True
            if 'deployment_requests_allowed' not in hardware_info:
                hardware_info['deployment_requests_allowed'] = True
            if 'cloud_sync_allowed' not in hardware_info:
                hardware_info['cloud_sync_allowed'] = True
            
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
                existing_capture_settings = _capture_settings_for_device(existing_device)
                if 'capture_settings' not in hardware_info:
                    hardware_info['capture_settings'] = existing_capture_settings
                else:
                    hardware_info['capture_settings'] = _reconcile_capture_settings(
                        existing_capture_settings, hardware_info.get('capture_settings') or {}
                    )

                if not _can_mark_device_online(
                    db,
                    current_user,
                    current_device_id=existing_device.deviceId
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"{_product_plan(current_user).title()} plan device limit reached. Disconnect another device or upgrade your plan."
                    )

                # Portal/user renames are canonical. Routine device
                # re-registration must not reset them to the local default.
                device_name = existing_device.device_name or device_name
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
                capture_settings = _capture_settings_for_device(existing_device)
                
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
                    "pending_deployments": pending_deployments,
                    "capture_settings": capture_settings,
                    "data": {"capture_settings": capture_settings}
                }
            
            if not _can_mark_device_online(db, current_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"{_product_plan(current_user).title()} plan device limit reached. Disconnect another device or upgrade your plan."
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
            capture_settings = _set_capture_settings_for_device(
                new_device, hardware_info.get("capture_settings") or DEFAULT_CAPTURE_SETTINGS, db, increment_revision=False
            )
            
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
                "pending_deployments": [],
                "capture_settings": capture_settings,
                "data": {"capture_settings": capture_settings}
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
        _expire_stale_devices(db, current_user.userId)
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
        if requested_online and not _can_mark_device_online(
            db,
            current_user,
            current_device_id=device.deviceId
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{_product_plan(current_user).title()} plan device limit reached. Disconnect another device or upgrade your plan."
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
    mode: str = "detach",
    confirmation: Optional[str] = None,
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
    mode = str(mode or "detach").lower()
    if mode not in {"detach", "erase"}:
        raise HTTPException(status_code=422, detail="Deletion mode must be detach or erase")
    device = db.query(Device).filter(
        Device.device_uuid == device_id, Device.userId == current_user.userId
    ).first()
    # Both operations are deliberately idempotent.
    if not device:
        return {"success": True, "message": "Device is already detached"}
    if mode == "erase" and confirmation != f"ERASE {device.device_name}":
        raise HTTPException(status_code=422, detail=f"Type ERASE {device.device_name} to permanently erase this device")
    try:
        records = db.query(DeviceFile).filter(DeviceFile.device_id == device.deviceId).all()
        cloud_file_ids = [record.cloud_file_id for record in records if record.cloud_file_id]
        db.query(DeviceFile).filter(DeviceFile.device_id == device.deviceId).delete(synchronize_session=False)
        from server.db import FileDeviceUpdate
        db.query(FileDeviceUpdate).filter(FileDeviceUpdate.deviceId == device.deviceId).delete(synchronize_session=False)
        db.query(DeviceDeployment).filter(DeviceDeployment.device_uuid == device_id).delete(synchronize_session=False)
        if mode == "erase" and cloud_file_ids:
            db.query(File).filter(File.fileId.in_(cloud_file_ids), File.userId == current_user.userId).delete(synchronize_session=False)
        db.delete(device)
        db.commit()
        return {
            "success": True,
            "message": "Device and cloud captures permanently erased" if mode == "erase" else "Device detached; uploaded cloud files retained",
            "data": {"mode": mode, "device_id": device_id},
        }
    except Exception as exc:
        db.rollback()
        logger.exception("Device %s %s failed", device_id, mode)
        raise HTTPException(status_code=500, detail=f"Device deletion failed: {exc}")


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
        if requested_online and not _can_mark_device_online(
            db,
            current_user,
            current_device_id=device.deviceId
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{_product_plan(current_user).title()} plan device limit reached. Disconnect another device or upgrade your plan."
            )

        update_data = {
            "last_seen": now,
            "online": requested_online,
        }

        # Prepare hardware_info updates
        hardware_info_updates = {}
        if hasattr(request, 'wifi_connected') and request.wifi_connected is not None:
            hardware_info_updates["wifi_connected"] = request.wifi_connected
        if hasattr(request, 'collection_active') and request.collection_active is not None:
            hardware_info_updates["collection_active"] = request.collection_active

        # Update optional fields if provided
        if hasattr(request, 'battery_level') and request.battery_level is not None:
            update_data["battery_level"] = request.battery_level
        if hasattr(request, 'online') and request.online is not None:
            update_data["online"] = request.online

        # Merge hardware_info updates
        if hardware_info_updates or (hasattr(request, 'hardware_info') and request.hardware_info):
            existing_hardware_info = _device_hardware_info(device)
            if hasattr(request, 'hardware_info') and request.hardware_info:
                existing_hardware_info.update(dict(request.hardware_info))
            existing_hardware_info.update(hardware_info_updates)

            # Preserve capture_settings
            existing_capture_settings = _capture_settings_for_device(device)
            if "capture_settings" not in existing_hardware_info:
                existing_hardware_info["capture_settings"] = existing_capture_settings
            else:
                existing_hardware_info["capture_settings"] = _reconcile_capture_settings(
                    existing_capture_settings, existing_hardware_info.get("capture_settings") or {}
                )

            update_data["hardware_info"] = json.dumps(existing_hardware_info)
        
        # Update IP address if available
        ip = get_client_ip(request_obj)
        if ip:
            update_data["ip_address"] = ip
        
        # Apply updates
        db.query(Device).filter(Device.deviceId == device.deviceId).update(update_data)
        db.commit()
        if request.inventory_complete:
            _apply_inventory_snapshot(device, current_user.userId, request, db)
        elif request.files:
            # Backward-compatible partial metadata update from older Thoth builds.
            _store_device_files(device.deviceId, current_user.userId, str(request.device_id), request.files, db)
        
        logger.debug(f"Heartbeat received from device {request.device_id}")
        db.refresh(device)
        capture_settings = _capture_settings_for_device(device)
        pending_uploads = _get_pending_uploads(device.deviceId, db)
        
        return {
            "success": True,
            "message": "Heartbeat received",
            "data": {
                "device_id": str(request.device_id),
                "timestamp": now.isoformat(),
                "capture_settings": capture_settings,
                "pending_uploads": pending_uploads,
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


@router.get("/{device_uuid}/capture-settings", response_model=Dict[str, Any])
async def get_device_capture_settings(
    device_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    device = db.query(Device).filter(
        Device.device_uuid == device_uuid,
        Device.userId == current_user.userId
    ).first()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")

    return {
        "success": True,
        "device_id": device_uuid,
        "capture_settings": _capture_settings_for_device(device),
    }


@router.put("/{device_uuid}/capture-settings", response_model=Dict[str, Any])
async def update_device_capture_settings(
    device_uuid: str,
    payload: Dict[str, Any] = Body(default={}),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    device = db.query(Device).filter(
        Device.device_uuid == device_uuid,
        Device.userId == current_user.userId
    ).first()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")

    current = _capture_settings_for_device(device)
    if "revision" in payload:
        try:
            expected_revision = int(payload["revision"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Settings revision must be an integer")
        if expected_revision != int(current.get("revision") or 0):
            raise HTTPException(status_code=409, detail="Settings changed on another client; refresh and try again")

    if "yellow_threshold_percent" in payload or "green_threshold_percent" in payload:
        try:
            yellow = float(payload.get("yellow_threshold_percent", current["yellow_threshold_percent"]))
            green = float(payload.get("green_threshold_percent", current["green_threshold_percent"]))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Detection thresholds must be numeric")
        if not 0.0 <= yellow < green <= 100.0:
            raise HTTPException(status_code=422, detail="Thresholds must satisfy 0 <= yellow < green <= 100")

    settings = _set_capture_settings_for_device(device, payload, db)
    return {
        "success": True,
        "message": "Capture settings updated",
        "device_id": device_uuid,
        "capture_settings": settings,
    }


def _owned_device(device_uuid: str, user: User, db: Session) -> Device:
    device = db.query(Device).filter(Device.device_uuid == device_uuid, Device.userId == user.userId).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


@router.post("/{device_uuid}/calibrations", response_model=Dict[str, Any])
async def create_device_calibration(
    device_uuid: str, payload: Dict[str, Any] = Body(default={}),
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_feature(current_user, "calibration")
    device = _owned_device(device_uuid, current_user, db)
    mode = str(payload.get("processing_mode") or "balanced").lower()
    if mode not in {"responsive", "balanced", "precision"}:
        raise HTTPException(status_code=422, detail="Unknown processing mode")
    hardware = _device_hardware_info(device)
    runs = hardware.get("calibration_runs") if isinstance(hardware.get("calibration_runs"), dict) else {}
    run_id = str(uuid_lib.uuid4())
    runs[run_id] = {
        "id": run_id, "processing_mode": mode, "status": "collecting",
        "created_at": datetime.now(timezone.utc).isoformat(), "regions": {},
        "base_settings_revision": int(_capture_settings_for_device(device).get("revision") or 0),
    }
    hardware["calibration_runs"] = runs
    device.hardware_info = json.dumps(hardware)
    db.commit()
    return {"success": True, "calibration": runs[run_id]}


@router.post("/{device_uuid}/calibrations/{run_id}/regions/{region}/start", response_model=Dict[str, Any])
async def record_calibration_region(
    device_uuid: str, run_id: str, region: str, payload: Dict[str, Any] = Body(default={}),
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_feature(current_user, "calibration")
    device = _owned_device(device_uuid, current_user, db)
    if region not in CALIBRATION_REGIONS:
        raise HTTPException(status_code=422, detail="Region must be red, yellow, or green")
    hardware = _device_hardware_info(device)
    runs = hardware.get("calibration_runs") if isinstance(hardware.get("calibration_runs"), dict) else {}
    run = runs.get(run_id)
    if not isinstance(run, dict) or run.get("status") != "collecting":
        raise HTTPException(status_code=409, detail="Calibration run is unavailable")
    values = payload.get("ratio_percent_samples")
    if not isinstance(values, list) or not values:
        raise HTTPException(status_code=422, detail="A completed minute of ratio samples is required")
    try:
        cleaned = [float(value) for value in values]
        if any(value < 0.0 or value > 100.0 for value in cleaned):
            raise ValueError
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Ratio samples must be between 0 and 100")
    run.setdefault("regions", {})[region] = {
        "ratio_percent_samples": cleaned, "completed_at": datetime.now(timezone.utc).isoformat(),
        "calibration_only": True,
    }
    device.hardware_info = json.dumps(hardware)
    db.commit()
    return {"success": True, "calibration": run}


@router.get("/{device_uuid}/calibrations/{run_id}", response_model=Dict[str, Any])
async def get_device_calibration(
    device_uuid: str, run_id: str, current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_feature(current_user, "calibration")
    device = _owned_device(device_uuid, current_user, db)
    run = (_device_hardware_info(device).get("calibration_runs") or {}).get(run_id)
    if not isinstance(run, dict):
        raise HTTPException(status_code=404, detail="Calibration run not found")
    return {"success": True, "calibration": run}


@router.post("/{device_uuid}/calibrations/{run_id}/commit", response_model=Dict[str, Any])
async def commit_device_calibration(
    device_uuid: str, run_id: str, current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_feature(current_user, "calibration")
    device = _owned_device(device_uuid, current_user, db)
    hardware = _device_hardware_info(device)
    run = (hardware.get("calibration_runs") or {}).get(run_id)
    if not isinstance(run, dict) or run.get("status") != "collecting":
        raise HTTPException(status_code=409, detail="Calibration run is unavailable")
    current = _capture_settings_for_device(device)
    if int(current.get("revision") or 0) != int(run.get("base_settings_revision") or 0):
        raise HTTPException(status_code=409, detail="Capture settings changed during calibration")
    try:
        derived = derive_thresholds({
            region: ((run.get("regions") or {}).get(region) or {}).get("ratio_percent_samples")
            for region in CALIBRATION_REGIONS
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    committed_at = datetime.now(timezone.utc).isoformat()
    calibration = {
        **derived, "run_id": run_id, "processing_mode": run["processing_mode"],
        "calibrated_at": committed_at,
    }
    calibrations = current.get("calibrations") if isinstance(current.get("calibrations"), dict) else {}
    calibrations[run["processing_mode"]] = calibration
    settings = _set_capture_settings_for_device(device, {
        "yellow_threshold_percent": derived["yellow_threshold_percent"],
        "green_threshold_percent": derived["green_threshold_percent"],
        "calibrations": calibrations,
    }, db)
    device = db.query(Device).filter(Device.deviceId == device.deviceId).first()
    hardware = _device_hardware_info(device)
    hardware["calibration_runs"][run_id].update({"status": "committed", "committed_at": committed_at, "result": calibration})
    device.hardware_info = json.dumps(hardware)
    db.commit()
    return {"success": True, "calibration": hardware["calibration_runs"][run_id], "capture_settings": settings}


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
            DeviceFile.device_id == device.deviceId,
            DeviceFile.on_device == True,
        ).order_by(DeviceFile.modified_at.desc()).all()

        capture_metadata = _device_hardware_info(device).get('capture_file_metadata', {})
        file_list = []
        for record in files:
            if not _is_minute_file_name(record.filename):
                continue
            item = record.to_dict()
            metadata = capture_metadata.get(record.filename, {}) if isinstance(capture_metadata, dict) else {}
            if isinstance(metadata, dict):
                item.update({
                    'label': metadata.get('label'),
                    'labels': metadata.get('labels') or [],
                    'occupancy': metadata.get('occupancy'),
                    'progress': metadata.get('progress'),
                })
            file_list.append(item)
        
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


@router.post("/{device_uuid}/captures/{minute}/request-upload", response_model=Dict[str, Any])
async def request_capture_upload(
    device_uuid: str,
    minute: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Request the complete minute from its owning Thoth device."""
    device = db.query(Device).filter(
        Device.device_uuid == device_uuid,
        Device.userId == current_user.userId,
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    record = db.query(DeviceFile).filter(
        DeviceFile.device_id == device.deviceId,
        DeviceFile.filename == minute,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Capture minute not found")
    if not _portal_upload_allowed_for_device(device):
        raise HTTPException(status_code=403, detail="Portal-initiated uploads are disabled on this device")
    record.upload_requested = True
    db.commit()
    return {"success": True, "minute": minute, "upload_requested": True}


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

        device = db.query(Device).filter(
            Device.deviceId == device_file.device_id,
            Device.userId == current_user.userId
        ).first()
        if device and not _portal_upload_allowed_for_device(device):
            raise HTTPException(
                status_code=403,
                detail="Portal-initiated uploads are disabled on this device"
            )

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
