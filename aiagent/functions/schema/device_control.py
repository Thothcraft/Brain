import json

from aiagent.functions.metadata import function_schema

try:
    from server.db import Device, DeviceCommand, SessionLocal
except Exception:  # pragma: no cover - registry must load without database extras
    Device = None
    DeviceCommand = None
    SessionLocal = None


def _current_user_id():
    from aiagent.handler import query as query_module

    context = getattr(query_module, "CURRENT_USER_ID", None)
    return context.get() if context is not None else None


def _queue(command, device_id="", label=""):
    if not (Device and DeviceCommand and SessionLocal):
        return {"success": False, "error": "Database layer unavailable"}
    user_id = _current_user_id()
    if user_id is None:
        return {"success": False, "error": "No authenticated user context"}
    db = SessionLocal()
    try:
        query = db.query(Device).filter(Device.userId == int(user_id), Device.approved == True)
        if device_id:
            query = query.filter(Device.device_uuid == str(device_id))
        device = query.order_by(Device.online.desc(), Device.last_seen.desc()).first()
        if device is None:
            return {"success": False, "error": "No matching device"}
        payload = {"label": str(label).strip()} if label else {}
        item = DeviceCommand(
            device_id=device.deviceId,
            user_id=int(user_id),
            command=command,
            payload=json.dumps(payload, separators=(",", ":")),
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return {
            "success": True,
            "command_id": item.id,
            "device_id": device.device_uuid,
            "command": command,
        }
    finally:
        db.close()


@function_schema(
    "start_collection",
    "Start minute-based data collection on a user's Thoth device.",
    [],
    ["device_id"],
)
def start_collection(device_id=""):
    return _queue("start_collection", device_id=device_id)


@function_schema(
    "stop_collection",
    "Stop data collection on a user's Thoth device.",
    [],
    ["device_id"],
)
def stop_collection(device_id=""):
    return _queue("stop_collection", device_id=device_id)


@function_schema(
    "label_current_chunk",
    "Apply a label to the current and subsequent 10-frame chunks in the active minute.",
    ["label"],
    ["device_id"],
)
def label_current_chunk(label, device_id=""):
    return _queue("label_current_chunk", device_id=device_id, label=label)
