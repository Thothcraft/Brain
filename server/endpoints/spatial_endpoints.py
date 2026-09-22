"""Spatial endpoints — spaces, zones, device placement, live spatial state.

A Space is a named physical area with an optional floor plan. Zones are
polygonal regions inside it. DevicePlacement positions a node's sensor
field-of-view on the plan so predictions (occupancy labels, radar XY
points) can be evaluated per-zone.

Occupancy rule v1 is deliberately simple: a zone is occupied when the
latest device predictions report occupancy, or when radar XY points
(transformed into plan coordinates) fall inside the zone polygon.
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import (
    Device, DeviceCaptureChunk, DevicePlacement, Space, User, Zone, get_db,
)
from server.entitlements import check_space_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/spaces", tags=["spaces"])

OCCUPIED_LABELS = {"occupied", "person", "presence", "moving", "activity"}


# ── request models ──────────────────────────────────────────────────────────

class SpaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    parent_id: Optional[int] = None
    floor_plan_file_id: Optional[int] = None
    width_m: Optional[float] = Field(default=None, gt=0)
    height_m: Optional[float] = Field(default=None, gt=0)


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    polygon: List[List[float]] = Field(min_length=3)


class PlacementIn(BaseModel):
    space_id: int
    x: float = 0.0
    y: float = 0.0
    rotation_deg: float = 0.0
    fov_deg: float = Field(default=90.0, gt=0, le=360)
    range_m: float = Field(default=8.0, gt=0)


# ── helpers ─────────────────────────────────────────────────────────────────

def _get_space(db: Session, user: User, space_id: int) -> Space:
    space = db.query(Space).filter(
        Space.id == space_id, Space.user_id == user.userId).first()
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    return space


def _get_device(db: Session, user: User, device_uuid: str) -> Device:
    device = db.query(Device).filter(
        Device.device_uuid == device_uuid,
        Device.userId == user.userId).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _to_plan_coords(px: float, py: float, placement: DevicePlacement) -> tuple:
    """Rotate+translate a sensor-local XY point into plan coordinates."""
    rad = math.radians(placement.rotation_deg)
    return (
        placement.x + px * math.cos(rad) - py * math.sin(rad),
        placement.y + px * math.sin(rad) + py * math.cos(rad),
    )


def _latest_predictions(db: Session, device: Device) -> List[Dict[str, Any]]:
    """Model predictions from the device's newest live chunks."""
    minute = db.query(DeviceCaptureChunk.minute).filter(
        DeviceCaptureChunk.device_id == device.deviceId
    ).order_by(DeviceCaptureChunk.minute.desc()).limit(1).scalar()
    if not minute:
        return []
    rows = db.query(DeviceCaptureChunk).filter(
        DeviceCaptureChunk.device_id == device.deviceId,
        DeviceCaptureChunk.minute == minute,
    ).order_by(DeviceCaptureChunk.chunk_index.desc()).limit(12).all()
    preds: List[Dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row.payload or "{}")
        except (TypeError, ValueError):
            continue
        for pred in payload.get("model_predictions") or []:
            if isinstance(pred, dict):
                preds.append(pred)
        labels = payload.get("labels")
        if isinstance(labels, dict):
            preds.extend(
                {"label": k, "confidence": v} for k, v in labels.items())
    return preds


def _space_state(db: Session, space: Space) -> Dict[str, Any]:
    """Compute live spatial state for one space from placed devices."""
    zone_states: Dict[str, Dict[str, Any]] = {
        z.name: {"occupied": False, "people_count": 0, "confidence": 0.0}
        for z in space.zones
    }
    space_occupied = False
    people = 0
    best_conf = 0.0
    last_activity: Optional[str] = None

    for placement in space.placements:
        device = placement.device
        if not device:
            continue
        preds = _latest_predictions(db, device)
        for pred in preds:
            label = str(pred.get("label") or pred.get("prediction") or "").lower()
            conf = float(pred.get("confidence") or 0.0)
            occupied = label in OCCUPIED_LABELS
            count = int(pred.get("people_count") or (1 if occupied else 0))
            xy = pred.get("xy") or pred.get("points") or []

            if occupied:
                space_occupied = True
                best_conf = max(best_conf, conf)
                people = max(people, count)

            # Assign XY points (plan coords) to zones
            zone_hits: Dict[str, int] = {name: 0 for name in zone_states}
            for pt in xy:
                try:
                    px, py = _to_plan_coords(float(pt[0]), float(pt[1]), placement)
                except (TypeError, ValueError, IndexError):
                    continue
                for zone in space.zones:
                    try:
                        polygon = json.loads(zone.polygon_json)
                    except (TypeError, ValueError):
                        continue
                    if _point_in_polygon(px, py, polygon):
                        zone_hits[zone.name] += 1

            for name, hits in zone_hits.items():
                if hits > 0 or (occupied and not xy):
                    zs = zone_states[name]
                    zs["occupied"] = True
                    zs["people_count"] = max(zs["people_count"], hits or count)
                    zs["confidence"] = max(zs["confidence"], conf)

        if device.last_seen:
            ts = device.last_seen.isoformat() + "Z"
            if last_activity is None or ts > last_activity:
                last_activity = ts

    return {
        "space_id": space.id,
        "name": space.name,
        "occupied": space_occupied,
        "people_count": people,
        "confidence": round(best_conf, 3),
        "zones": zone_states,
        "last_activity": last_activity,
    }


# ── spaces CRUD ─────────────────────────────────────────────────────────────

@router.get("", response_model=Dict[str, Any])
async def list_spaces(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    spaces = db.query(Space).filter(Space.user_id == current_user.userId).all()
    return {"success": True, "spaces": [s.to_dict() for s in spaces]}


@router.post("", response_model=Dict[str, Any], status_code=201)
async def create_space(
    body: SpaceIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    count = db.query(Space).filter(Space.user_id == current_user.userId).count()
    check_space_limit(current_user, count)
    if body.parent_id is not None:
        _get_space(db, current_user, body.parent_id)
    space = Space(user_id=current_user.userId, **body.model_dump())
    db.add(space)
    db.commit()
    db.refresh(space)
    return {"success": True, "space": space.to_dict()}


@router.get("/state", response_model=Dict[str, Any])
async def all_spaces_state(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Live spatial state for every space — the context API."""
    spaces = db.query(Space).filter(Space.user_id == current_user.userId).all()
    return {"success": True,
            "spaces": [_space_state(db, s) for s in spaces]}


@router.get("/{space_id}", response_model=Dict[str, Any])
async def get_space(
    space_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return {"success": True, "space": _get_space(db, current_user, space_id).to_dict()}


@router.put("/{space_id}", response_model=Dict[str, Any])
async def update_space(
    space_id: int,
    body: SpaceIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    space = _get_space(db, current_user, space_id)
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(space, key, value)
    db.commit()
    return {"success": True, "space": space.to_dict()}


@router.delete("/{space_id}", response_model=Dict[str, Any])
async def delete_space(
    space_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    space = _get_space(db, current_user, space_id)
    db.delete(space)
    db.commit()
    return {"success": True}


# ── zones ───────────────────────────────────────────────────────────────────

@router.post("/{space_id}/zones", response_model=Dict[str, Any], status_code=201)
async def create_zone(
    space_id: int,
    body: ZoneIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    space = _get_space(db, current_user, space_id)
    zone = Zone(space_id=space.id, name=body.name,
                polygon_json=json.dumps(body.polygon))
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return {"success": True, "zone": zone.to_dict()}


@router.delete("/{space_id}/zones/{zone_id}", response_model=Dict[str, Any])
async def delete_zone(
    space_id: int,
    zone_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    space = _get_space(db, current_user, space_id)
    zone = db.query(Zone).filter(
        Zone.id == zone_id, Zone.space_id == space.id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    db.delete(zone)
    db.commit()
    return {"success": True}


# ── device placement ────────────────────────────────────────────────────────

@router.put("/devices/{device_uuid}/placement", response_model=Dict[str, Any])
async def set_device_placement(
    device_uuid: str,
    body: PlacementIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    device = _get_device(db, current_user, device_uuid)
    space = _get_space(db, current_user, body.space_id)
    placement = db.query(DevicePlacement).filter(
        DevicePlacement.device_id == device.deviceId).first()
    if placement is None:
        placement = DevicePlacement(device_id=device.deviceId, space_id=space.id)
        db.add(placement)
    placement.space_id = space.id
    placement.x, placement.y = body.x, body.y
    placement.rotation_deg = body.rotation_deg
    placement.fov_deg = body.fov_deg
    placement.range_m = body.range_m
    db.commit()
    return {"success": True, "placement": placement.to_dict()}


@router.delete("/devices/{device_uuid}/placement", response_model=Dict[str, Any])
async def remove_device_placement(
    device_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    device = _get_device(db, current_user, device_uuid)
    placement = db.query(DevicePlacement).filter(
        DevicePlacement.device_id == device.deviceId).first()
    if placement:
        db.delete(placement)
        db.commit()
    return {"success": True}


# ── spatial state ───────────────────────────────────────────────────────────

@router.get("/{space_id}/state", response_model=Dict[str, Any])
async def space_state(
    space_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    space = _get_space(db, current_user, space_id)
    return {"success": True, "state": _space_state(db, space)}
