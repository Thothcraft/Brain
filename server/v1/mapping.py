"""Map Brain ORM objects to v1 contract dicts.

Single place where the legacy schema is translated into the versioned
public contract - clients never see legacy field names.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Legacy deployment status -> v1 deployment state machine (section 19).
_DEPLOYMENT_STATE_MAP = {
    "pending": "queued",
    "queued": "queued",
    "delivered": "acknowledged",
    "received": "received",
    "validated": "validated",
    "installed": "installed",
    "acknowledged": "acknowledged",
    "active": "active",
    "activated": "active",
    "restarted": "active",
    "failed": "failed",
    "declined": "declined",
    "rejected": "declined",
}


def _hw_info(device: Any) -> Dict[str, Any]:
    try:
        return json.loads(device.hardware_info) if device.hardware_info else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def sensors_from_hardware(device: Any) -> List[Dict[str, Any]]:
    """Extract the sensor inventory reported by the node."""
    hw = _hw_info(device)
    raw = hw.get("sensors") or hw.get("capabilities") or []
    sensors: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        raw = [{"id": k, **(v if isinstance(v, dict) else {"type": k})}
               for k, v in raw.items()]
    for item in raw:
        if isinstance(item, str):
            sensors.append({"id": f"{item}-0", "type": item})
        elif isinstance(item, dict):
            sid = str(item.get("id") or item.get("sensor_id")
                      or f"{item.get('type', 'sensor')}-0")
            sensors.append({
                "id": sid,
                "type": str(item.get("type") or item.get("sensor_type") or sid.rsplit("-", 1)[0]),
                "driver": str(item.get("driver") or ""),
                "driver_version": str(item.get("driver_version") or ""),
                "sample_rate": item.get("sample_rate"),
                "units": dict(item.get("units") or {}),
                "online": bool(item.get("online", True)),
                "capabilities": list(item.get("capabilities") or []),
                "metadata": dict(item.get("metadata") or {}),
            })
    return sensors


def device_to_v1(device: Any, *, include_sensors: bool = True) -> Dict[str, Any]:
    """Device ORM -> DeviceV1 contract dict."""
    hw = _hw_info(device)
    d = device.to_dict() if hasattr(device, "to_dict") else {}
    return {
        "id": str(device.device_uuid),
        "stable_uuid": str(device.device_uuid),
        "name": device.device_name,
        "owner": str(device.userId) if device.userId is not None else None,
        "platform": str(hw.get("platform") or hw.get("os") or ""),
        "architecture": str(hw.get("architecture") or ""),
        "software_version": str(hw.get("software_version") or hw.get("version") or ""),
        "whispy_version": str(hw.get("whispy_version") or ""),
        "online": bool(d.get("online")),
        "last_seen": d.get("last_seen"),
        "capabilities": list(hw.get("capabilities") or []),
        "sensors": sensors_from_hardware(device) if include_sensors else [],
        "health": dict(hw.get("health") or {}),
    }


def model_to_v1(model: Any) -> Dict[str, Any]:
    """TrainedModel ORM -> ModelV1 contract dict."""
    d = model.to_dict() if hasattr(model, "to_dict") else {}
    config = d.get("config") or {}
    manifest = config.get("manifest") if isinstance(config, dict) else None
    return {
        "id": str(model.id),
        "name": model.name,
        "processor": getattr(model, "processor_type", None) or "torchscript",
        "sensor": getattr(model, "sensor", None),
        "task": getattr(model, "task", None),
        "visibility": getattr(model, "visibility", None) or "private",
        "manifest": manifest,
        "created_at": d.get("created_at"),
    }


def deployment_to_v1(dep: Any) -> Dict[str, Any]:
    """DeviceDeployment ORM -> DeploymentV1 contract dict.

    The v1 state machine rides in the payload JSON so older rows still
    map cleanly: ``payload.v1_state`` wins, else legacy ``status`` maps.
    """
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(dep.payload) if dep.payload else {}
    except (TypeError, json.JSONDecodeError):
        payload = {}
    state = payload.get("v1_state") or _DEPLOYMENT_STATE_MAP.get(
        str(dep.status or "pending"), "queued")
    return {
        "deployment_id": str(dep.deployment_id),
        "device_id": str(dep.device_uuid),
        "model_id": str(dep.model_id),
        "state": state,
        "runtime_model_id": str(payload.get("runtime_model_id") or ""),
        "failure": payload.get("failure"),
        "created_at": dep.created_at.isoformat() + "Z" if getattr(dep, "created_at", None) else None,
        "updated_at": dep.delivered_at.isoformat() + "Z" if getattr(dep, "delivered_at", None) else None,
    }


def capture_to_v1(capture: Any, device_uuid: str) -> Dict[str, Any]:
    """DeviceCapture ORM -> CaptureV1 contract dict."""
    try:
        sensors = json.loads(capture.sensors or "[]")
    except (TypeError, json.JSONDecodeError):
        sensors = []
    try:
        counts = json.loads(capture.sample_counts or "{}")
    except (TypeError, json.JSONDecodeError):
        counts = {}
    return {
        "id": str(capture.capture_id),
        "device_id": str(device_uuid),
        "state": str(capture.state or "requested"),
        "sensors": sensors,
        "sample_counts": counts,
        "started_at": capture.started_at.timestamp() if getattr(capture, "started_at", None) else None,
        "stopped_at": capture.stopped_at.timestamp() if getattr(capture, "stopped_at", None) else None,
        "metadata": {"created_at": capture.created_at.isoformat() + "Z"
                     if getattr(capture, "created_at", None) else None},
    }


def chunk_to_samples(chunk: Any, device_uuid: str,
                     sensor_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flatten a live capture chunk into SensorSampleV1-shaped dicts.

    Chunks carry ``features``/``model_predictions`` payloads; each
    sensor-bearing entry becomes a real sample - never an availability
    boolean.
    """
    try:
        payload = json.loads(chunk.payload) if isinstance(chunk.payload, str) else dict(chunk.payload or {})
    except (TypeError, json.JSONDecodeError):
        payload = {}
    ts = chunk.updated_at.timestamp() if getattr(chunk, "updated_at", None) else 0.0
    samples: List[Dict[str, Any]] = []

    # Preferred path: the node uploaded real SensorSamples. Relay them with
    # their original identity, source timestamp, sequence, payload type,
    # units and sample rate intact — never re-derived from the chunk row.
    raw_samples = payload.get("samples")
    if isinstance(raw_samples, list):
        for s in raw_samples:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("sensor_id") or "")
            if sensor_id and sid != sensor_id:
                continue
            samples.append({
                "device_id": str(s.get("device_id") or device_uuid),
                "sensor_id": sid,
                "sensor_type": str(s.get("sensor_type")
                                   or sid.rsplit("-", 1)[0]),
                "timestamp": float(s.get("timestamp") or ts),
                "sequence": int(s.get("sequence") or 0),
                "payload_type": str(s.get("payload_type") or "json"),
                "payload": s.get("payload"),
                "sample_rate": s.get("sample_rate"),
                "units": dict(s.get("units") or {}),
                "metadata": {"minute": getattr(chunk, "minute", None),
                             **dict(s.get("metadata") or {})},
            })
        return samples

    # Legacy fallback: flatten a feature map into samples (loses the
    # original timing/sequence/units — retained only for old chunks).
    features = payload.get("features") or {}
    seq = int(getattr(chunk, "chunk_index", 0) or 0)
    for key, value in features.items():
        if sensor_id and key != sensor_id:
            continue
        samples.append({
            "device_id": device_uuid,
            "sensor_id": key,
            "sensor_type": key.rsplit("-", 1)[0],
            "timestamp": ts,
            "sequence": seq,
            "payload_type": "json",
            "payload": value,
            "metadata": {"minute": getattr(chunk, "minute", None)},
        })
    return samples
