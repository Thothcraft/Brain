"""Brain v1 API router - the versioned public contract.

Mounted at ``/v1`` alongside the legacy ``/api`` namespace. Every remote
client (Whispy, Thoth, thothHUB, mobile) converges on these endpoints.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from server.auth import get_current_user, get_scoped_principal
from server.db import (
    AutomationKey, Device, DeviceCapture, DeviceCaptureChunk, DeviceCommand,
    DeviceDeployment, TrainedModel, User, get_db,
)
from .mapping import (
    capture_to_v1, chunk_to_samples, deployment_to_v1, device_to_v1,
    model_to_v1, sensors_from_hardware,
)
from .schemas import (
    AccountV1, CaptureListV1, CaptureStartRequestV1, CaptureV1,
    DeploymentCreateV1, DeploymentListV1, DeploymentV1, DeviceListV1,
    DeviceV1, ModelListV1, PredictionListV1, PredictionV1, SensorListV1,
    StreamPageV1,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["v1"])

_PROCESSOR_TYPES = {"rule", "torchscript", "fusion"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _owned_device(device_id: str, user: User, db: Session) -> Device:
    """Resolve a device the caller owns - tenant isolation everywhere."""
    device = db.query(Device).filter(
        Device.device_uuid == device_id,
        Device.userId == user.userId,
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@router.get("/account", response_model=AccountV1)
async def get_account(current_user: User = Depends(get_current_user)) -> AccountV1:
    return AccountV1(
        user_id=current_user.userId,
        username=current_user.username,
        email=getattr(current_user, "email", None),
        plan=getattr(current_user, "plan", "free") or "free",
    )


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

@router.get("/devices", response_model=DeviceListV1)
async def list_devices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeviceListV1:
    """All approved devices - online AND offline (mobile/hub parity)."""
    devices = db.query(Device).filter(
        Device.userId == current_user.userId,
        Device.approved == True,  # noqa: E712
    ).all()
    out = [device_to_v1(d) for d in devices]
    return DeviceListV1(devices=out, count=len(out))


@router.get("/devices/{device_id}", response_model=DeviceV1)
async def get_device(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeviceV1:
    return DeviceV1(**device_to_v1(_owned_device(device_id, current_user, db)))


@router.get("/devices/{device_id}/sensors", response_model=SensorListV1)
async def get_device_sensors(
    device_id: str,
    current_user: User = Depends(get_scoped_principal("device:read")),
    db: Session = Depends(get_db),
) -> SensorListV1:
    device = _owned_device(device_id, current_user, db)
    return SensorListV1(device_id=device_id,
                        sensors=sensors_from_hardware(device))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@router.get("/devices/{device_id}/predictions", response_model=PredictionListV1)
async def get_device_predictions(
    device_id: str,
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_scoped_principal("predictions:read")),
    db: Session = Depends(get_db),
) -> PredictionListV1:
    """Recent predictions extracted from the device's live chunks."""
    device = _owned_device(device_id, current_user, db)
    rows = db.query(DeviceCaptureChunk).filter(
        DeviceCaptureChunk.device_id == device.deviceId,
    ).order_by(DeviceCaptureChunk.updated_at.desc()).limit(limit).all()

    predictions: List[PredictionV1] = []
    for row in rows:
        try:
            payload = json.loads(row.payload or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        for pred in payload.get("model_predictions") or []:
            if not isinstance(pred, dict):
                continue
            predictions.append(PredictionV1(
                id=str(pred.get("id") or ""),
                device_id=device_id,
                runtime_model_id=str(pred.get("runtime_model_id")
                                     or pred.get("model_id") or ""),
                timestamp=float(pred.get("timestamp") or 0.0),
                label=str(pred.get("label") or pred.get("class") or ""),
                confidence=float(pred.get("confidence") or 0.0),
                scores=dict(pred.get("scores") or {}),
                metadata={"minute": row.minute,
                          **dict(pred.get("metadata") or {})},
            ))
    return PredictionListV1(device_id=device_id, predictions=predictions)


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

def _capture_to_schema(capture: DeviceCapture, db: Session) -> CaptureV1:
    device = db.query(Device).filter(Device.deviceId == capture.device_id).first()
    dev_uuid = device.device_uuid if device else ""
    return CaptureV1(**capture_to_v1(capture, dev_uuid))


@router.get("/devices/{device_id}/captures", response_model=CaptureListV1)
async def list_device_captures(
    device_id: str,
    current_user: User = Depends(get_scoped_principal("capture")),
    db: Session = Depends(get_db),
) -> CaptureListV1:
    device = _owned_device(device_id, current_user, db)
    rows = db.query(DeviceCapture).filter(
        DeviceCapture.device_id == device.deviceId,
        DeviceCapture.user_id == current_user.userId,
    ).order_by(DeviceCapture.created_at.desc()).all()
    return CaptureListV1(
        captures=[_capture_to_schema(c, db) for c in rows])


@router.post("/devices/{device_id}/captures", response_model=CaptureV1,
             status_code=201)
async def start_device_capture(
    device_id: str,
    request: CaptureStartRequestV1,
    current_user: User = Depends(get_scoped_principal("capture")),
    db: Session = Depends(get_db),
) -> CaptureV1:
    """Create a durable capture and queue capture-start on its device.

    The capture enters the ``requested`` state — it is *not* reported active
    until the node acknowledges it. The same ``capture_id`` is used by the
    client, the node command, and storage so it reconciles end to end.
    """
    device = _owned_device(device_id, current_user, db)
    capture_id = uuid.uuid4().hex[:12]
    capture = DeviceCapture(
        capture_id=capture_id,
        device_id=device.deviceId,
        user_id=current_user.userId,
        state="requested",
        sensors=json.dumps(request.sensors or []),
    )
    db.add(capture)
    command = DeviceCommand(
        device_id=device.deviceId,
        user_id=current_user.userId,
        command="capture_start",
        payload=json.dumps({"capture_id": capture_id,
                            "sensors": request.sensors}),
        status="pending",
    )
    db.add(command)
    db.commit()
    db.refresh(capture)
    return _capture_to_schema(capture, db)


@router.get("/captures", response_model=CaptureListV1)
async def list_captures(
    device_id: Optional[str] = Query(None),
    current_user: User = Depends(get_scoped_principal("capture")),
    db: Session = Depends(get_db),
) -> CaptureListV1:
    query = db.query(DeviceCapture).filter(
        DeviceCapture.user_id == current_user.userId)
    if device_id:
        device = db.query(Device).filter(
            Device.device_uuid == device_id,
            Device.userId == current_user.userId).first()
        if not device:
            return CaptureListV1(captures=[])
        query = query.filter(DeviceCapture.device_id == device.deviceId)
    rows = query.order_by(DeviceCapture.created_at.desc()).all()
    return CaptureListV1(
        captures=[_capture_to_schema(c, db) for c in rows])


@router.post("/captures/{capture_id}/stop")
async def stop_capture(
    capture_id: str,
    current_user: User = Depends(get_scoped_principal("capture")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Queue a capture-stop for the device that owns this exact capture.

    The capture→device binding is looked up from the durable record, so a
    stop always targets the right device and unknown IDs return 404. The
    state becomes ``stopping`` — it is not reported stopped until the node
    confirms.
    """
    capture = db.query(DeviceCapture).filter(
        DeviceCapture.capture_id == capture_id,
        DeviceCapture.user_id == current_user.userId,
    ).first()
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    device = db.query(Device).filter(
        Device.deviceId == capture.device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Capture not found")
    capture.state = "stopping"
    command = DeviceCommand(
        device_id=device.deviceId,
        user_id=current_user.userId,
        command="capture_stop",
        payload=json.dumps({"capture_id": capture_id}),
        status="pending",
    )
    db.add(command)
    db.commit()
    return {"id": capture_id, "device_id": device.device_uuid,
            "state": "stopping"}


# ---------------------------------------------------------------------------
# Streams (authorized remote sensor access section 9.4)
# ---------------------------------------------------------------------------

@router.get("/devices/{device_id}/streams/{sensor_id}",
            response_model=StreamPageV1)
async def stream_device_sensor(
    device_id: str,
    sensor_id: str,
    cursor: Optional[str] = Query(None),
    current_user: User = Depends(get_scoped_principal("sensor:stream")),
    db: Session = Depends(get_db),
) -> StreamPageV1:
    """Cursor-paged real samples for one sensor.

    Brain authenticates the caller and verifies device ownership before
    any sample is returned. The cursor is the last-seen chunk timestamp;
    clients poll with it for incremental delivery.
    """
    device = _owned_device(device_id, current_user, db)

    query = db.query(DeviceCaptureChunk).filter(
        DeviceCaptureChunk.device_id == device.deviceId)
    if cursor:
        # Resume strictly after the last delivered row across ALL minutes —
        # not just the latest — so a poll spanning a minute boundary never
        # discards unread older-minute data.
        try:
            after = datetime.fromisoformat(
                cursor.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            raise HTTPException(status_code=422,
                                detail="cursor must be an ISO timestamp")
        query = query.filter(DeviceCaptureChunk.updated_at > after)
    else:
        # No cursor → start from the latest minute only.
        latest_minute = db.query(DeviceCaptureChunk.minute).filter(
            DeviceCaptureChunk.device_id == device.deviceId
        ).order_by(DeviceCaptureChunk.minute.desc()).limit(1).scalar()
        if not latest_minute:
            state = "ok" if device.online else "disconnected"
            return StreamPageV1(device_id=device_id, sensor_id=sensor_id,
                                samples=[], cursor=cursor, state=state)
        query = query.filter(DeviceCaptureChunk.minute == latest_minute)

    rows = query.order_by(
        DeviceCaptureChunk.updated_at.asc(),
        DeviceCaptureChunk.chunk_index.asc()).limit(500).all()

    samples: List[Dict[str, Any]] = []
    for row in rows:
        samples.extend(chunk_to_samples(row, device_id, sensor_id))

    # The cursor is the newest timestamp among the rows actually returned —
    # never a separate "latest" query that could leap past unreturned rows.
    newest = max((r.updated_at for r in rows), default=None)
    return StreamPageV1(
        device_id=device_id, sensor_id=sensor_id, samples=samples,
        cursor=newest.isoformat() + "Z" if newest else cursor,
        state="ok" if device.online else "disconnected")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelListV1)
async def list_models(
    current_user: User = Depends(get_scoped_principal("model:deploy")),
    db: Session = Depends(get_db),
) -> ModelListV1:
    models = db.query(TrainedModel).filter(
        TrainedModel.user_id == current_user.userId).all()
    return ModelListV1(models=[model_to_v1(m) for m in models])


@router.post("/models", response_model=Dict[str, Any], status_code=201)
async def register_model(
    manifest: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_scoped_principal("model:deploy")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Register a model with a validated ``whispy-model/v1`` manifest.

    Legacy ``thoth-model/v1`` manifests are accepted during the rename
    transition.
    """
    errors: List[str] = []
    if manifest.get("format") not in ("whispy-model/v1", "thoth-model/v1"):
        errors.append(
            "format must be 'whispy-model/v1' (legacy 'thoth-model/v1' "
            "accepted)")
    if not manifest.get("name"):
        errors.append("name is required")
    processor = manifest.get("processor")
    if processor not in _PROCESSOR_TYPES:
        errors.append(f"processor must be one of {sorted(_PROCESSOR_TYPES)}")
    inputs = manifest.get("inputs") or []
    if not inputs:
        errors.append("at least one input is required")
    sha = manifest.get("artifact_sha256") or ""
    if sha and len(sha) != 64:
        errors.append("artifact_sha256 must be a 64-char hex digest")
    if errors:
        raise HTTPException(status_code=422,
                            detail={"manifest_errors": errors})

    record = TrainedModel(
        user_id=current_user.userId,
        name=manifest["name"],
        processor_type=processor,
        sensor=(inputs[0].get("sensor") if inputs else None),
        task=manifest.get("metadata", {}).get("task"),
        config=json.dumps({"manifest": manifest}),
        model_data=None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"model": model_to_v1(record)}


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------

@router.get("/deployments", response_model=DeploymentListV1)
async def list_deployments_v1(
    device_id: Optional[str] = Query(None),
    current_user: User = Depends(get_scoped_principal("model:deploy")),
    db: Session = Depends(get_db),
) -> DeploymentListV1:
    query = db.query(DeviceDeployment).filter(
        DeviceDeployment.user_id == current_user.userId)
    if device_id:
        query = query.filter(DeviceDeployment.device_uuid == device_id)
    deps = query.order_by(DeviceDeployment.created_at.desc()).all()
    return DeploymentListV1(
        deployments=[DeploymentV1(**deployment_to_v1(d)) for d in deps])


@router.post("/deployments", response_model=DeploymentV1, status_code=201)
async def create_deployment(
    request: DeploymentCreateV1,
    current_user: User = Depends(get_scoped_principal("model:deploy")),
    db: Session = Depends(get_db),
) -> DeploymentV1:
    """Queue a model deployment - enters the v1 state machine at ``queued``."""
    device = _owned_device(request.device_id, current_user, db)
    try:
        model_id = int(request.model_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="model_id must be an integer")
    model = db.query(TrainedModel).filter(
        TrainedModel.id == model_id,
        TrainedModel.user_id == current_user.userId,
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    dep = DeviceDeployment(
        deployment_id=uuid.uuid4().hex,
        device_uuid=device.device_uuid,
        model_id=model.id,
        user_id=current_user.userId,
        payload=json.dumps({
            "v1_state": "queued",
            "model": model_to_v1(model),
        }),
        status="pending",
    )
    db.add(dep)
    db.commit()
    db.refresh(dep)
    return DeploymentV1(**deployment_to_v1(dep))


# ---------------------------------------------------------------------------
# Automation keys (scoped programmatic credentials, §9.4 / §17)
# ---------------------------------------------------------------------------

@router.post("/automation/keys", status_code=201)
async def create_automation_key(
    request: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Mint a scoped automation key. The raw key is returned ONCE."""
    from server.auth import (AUTOMATION_SCOPES, generate_automation_key,
                             hash_automation_key)
    name = str(request.get("name") or "")
    scopes = [s for s in (request.get("scopes") or []) if s in AUTOMATION_SCOPES]
    if not scopes:
        raise HTTPException(
            status_code=422,
            detail=f"scopes must be a non-empty subset of {sorted(AUTOMATION_SCOPES)}")
    raw = generate_automation_key()
    key = AutomationKey(
        key_hash=hash_automation_key(raw),
        user_id=current_user.userId,
        name=name,
        scopes=json.dumps(scopes),
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    out = key.to_dict()
    out["key"] = raw  # shown once — never stored or returned again
    return out


@router.get("/automation/keys")
async def list_automation_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    keys = db.query(AutomationKey).filter(
        AutomationKey.user_id == current_user.userId).all()
    return {"keys": [k.to_dict() for k in keys]}


@router.delete("/automation/keys/{key_id}")
async def revoke_automation_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    key = db.query(AutomationKey).filter(
        AutomationKey.id == key_id,
        AutomationKey.user_id == current_user.userId).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    key.revoked = True
    db.commit()
    return {"ok": True, "id": key_id, "revoked": True}
