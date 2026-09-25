"""Actuation dispatch — Brain → Thoth node local API (§17).

An :class:`ActionRequest` row is the durable, idempotent envelope; this
module performs the real dispatch over the node's authenticated local
API::

    POST http://{host}:{port}/api/v1/actuators/{actuator_id}/actions
    Authorization: Bearer <node local token>
    {"operation": ..., "params": {...}, "timeout_seconds": ...,
     "action_id": ..., "expires_at": ..., "origin": ...}

The node deduplicates on ``action_id`` and refuses expired actions, so a
replayed dispatch is handled idempotently end to end. Dispatch failures
leave the request ``queued`` with ``attempts`` incremented — bounded
retries, never silent infinite retry.

Device reachability comes from ``Device.hardware_info["local_api"]``::

    {"host": "10.0.0.88", "port": 5001, "token": "<node local token>"}

provisioned at pairing/registration time.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from .db import ActionRequest, Device

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "unsupported", "expired"})


def device_endpoint(device: Device) -> Optional[Dict[str, Any]]:
    """Resolve a device's local-API endpoint from its reported inventory."""
    try:
        hw = json.loads(device.hardware_info) if device.hardware_info else {}
    except (TypeError, json.JSONDecodeError):
        hw = {}
    api = hw.get("local_api") or {}
    host = api.get("host") or device.ip_address
    port = api.get("port")
    token = api.get("token")
    if not host or not port or not token:
        return None
    return {"host": host, "port": int(port), "token": token}


def _post_action(endpoint: Dict[str, Any], actuator_id: str,
                 body: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    url = (f"http://{endpoint['host']}:{endpoint['port']}"
           f"/api/v1/actuators/{actuator_id}/actions")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {endpoint['token']}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = res.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def dispatch_action(db: Session, action: ActionRequest,
                    *, now: Optional[datetime] = None) -> ActionRequest:
    """Attempt one dispatch of a queued ActionRequest.

    Idempotent: terminal statuses return immediately; ``action_id`` is
    sent to the node so a replayed dispatch is deduplicated there too.
    Expired requests transition to ``expired`` without executing.
    """
    now = now or datetime.utcnow()
    if action.status in TERMINAL_STATUSES:
        return action
    if action.expires_at and now > action.expires_at:
        action.status = "expired"
        action.completed_at = now
        action.last_error = "expired before dispatch"
        action.result_json = json.dumps({
            "status": "expired",
            "detail": "action expired before dispatch"})
        db.commit()
        return action
    if action.attempts >= (action.max_attempts or 3):
        action.status = "failed"
        action.completed_at = now
        action.last_error = (action.last_error or "") + \
            " | max attempts reached"
        action.result_json = json.dumps({
            "status": "failed",
            "detail": f"max attempts ({action.max_attempts}) reached"})
        db.commit()
        return action

    device = db.query(Device).filter(
        Device.device_uuid == action.device_id).first()
    endpoint = device_endpoint(device) if device else None
    if endpoint is None:
        action.attempts += 1
        action.last_error = ("device unreachable: no local_api endpoint "
                             f"for {action.device_id}")
        db.commit()
        return action

    action.attempts += 1
    action.dispatched_at = now
    action.status = "dispatched"
    body = {
        "operation": action.operation,
        "params": json.loads(action.params or "{}"),
        "timeout_seconds": 30.0,
        "action_id": action.action_id,
        "expires_at": action.expires_at.timestamp()
                      if action.expires_at else None,
        "origin": action.origin,
    }
    try:
        result = _post_action(endpoint, action.actuator_id, body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        action.last_error = f"node HTTP {exc.code}: {detail}"
        if exc.code in (401, 403):
            # Auth failures are not retryable-transient — mark failed.
            action.status = "failed"
            action.completed_at = now
            action.result_json = json.dumps(
                {"status": "failed", "detail": action.last_error})
        else:
            action.status = "queued"
        db.commit()
        return action
    except (urllib.error.URLError, OSError) as exc:
        action.status = "queued"
        action.last_error = f"device unreachable: {exc}"
        db.commit()
        return action

    status = str(result.get("status") or "failed")
    action.status = status if status in TERMINAL_STATUSES else "failed"
    action.completed_at = now
    action.result_json = json.dumps(result)
    action.last_error = None if status == "succeeded" \
        else str(result.get("detail") or "")[:500]
    db.commit()
    return action


def retry_queued(db: Session, user_id: int, limit: int = 50) -> int:
    """Dispatch every queued, unexpired action for a user (bounded)."""
    rows = db.query(ActionRequest).filter(
        ActionRequest.user_id == user_id,
        ActionRequest.status == "queued",
    ).order_by(ActionRequest.created_at.asc()).limit(limit).all()
    dispatched = 0
    for row in rows:
        before = row.status
        dispatch_action(db, row)
        if row.status != "queued" or row.attempts > 0:
            dispatched += 1 if row.status != before or row.attempts else 0
    return dispatched


__all__ = ["dispatch_action", "retry_queued", "device_endpoint",
           "TERMINAL_STATUSES"]
