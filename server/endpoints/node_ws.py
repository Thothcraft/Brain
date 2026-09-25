"""Node↔Brain channel — plans/CONTRACT.md §2–§4.

One outbound-from-node WebSocket per device at ``WS /v1/node/ws`` carries
request/response frames (portal→node command tunnel) plus node→Brain push
frames (``event``/``room_changed``/``metadata``). REST fallback endpoints
exist for nodes that cannot hold a socket and for clients polling the
notification feed.

    WS   /v1/node/ws?device_id&token       node side connects here
    POST /v1/nodes/{id}/api                REST→WS relay (single choke point)
    GET/PUT /v1/nodes/{id}/room            cached room doc / write-through
    POST /v1/events                        node posts trigger/room/metadata
    GET  /v1/events?device_id=&limit=      portal/mobile poll
    POST /v1/usage                         record an api_usage row
    GET  /v1/usage?device_id=              usage rows for the caller

Frames (contract §2)::

    → {"type":"api_request","id":"<uuid>","method":"GET",
       "path":"/api/v1/metadata","body":null}
    ← {"type":"api_response","id":"<uuid>","status":200,"body":{...}}
       (binary replies use "body_b64" + "content_type" instead)
    ← {"type":"event","kind":"trigger_fired","data":{...}}
    ← {"type":"room_changed","data":{room/v1 doc}}
    ← {"type":"metadata","data":{...}}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from datetime import datetime
from time import perf_counter, time
from typing import Any, Dict, Optional, Tuple

from fastapi import (
    APIRouter, Body, Depends, HTTPException, Query, Request, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response
from jose import JWTError
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.auth import decode_token_any, get_current_user
from server.db import (
    ApiUsage, Device, NodeEvent, NodeRoom, User, get_db, get_db_session,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["v1", "node-channel"])

NODE_API_TIMEOUT_S = float(os.getenv("NODE_API_TIMEOUT_S", "30"))
USAGE_KINDS = {"prediction", "deploy", "capture", "automation", "actuate"}
USAGE_SOURCES = {"api", "portal", "dashboard", "mobile"}


# ---------------------------------------------------------------------------
# Connection registry
# ---------------------------------------------------------------------------

class _NodeConn:
    """One live node socket + its in-flight api_request futures."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.send_lock = asyncio.Lock()
        self.pending: Dict[str, asyncio.Future] = {}

    async def send(self, frame: Dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_json(frame)


class NodeWSManager:
    """In-memory ``{device_uuid → _NodeConn}`` map.

    Single-process Brain deployments hold the tunnel here; horizontal
    scaling would swap this for a shared broker (documented TODO).
    """

    def __init__(self) -> None:
        self._conns: Dict[str, _NodeConn] = {}

    def register(self, device_id: str, conn: _NodeConn) -> None:
        self._conns[device_id] = conn

    def unregister(self, device_id: str, conn: _NodeConn) -> None:
        if self._conns.get(device_id) is conn:
            self._conns.pop(device_id, None)
        for fut in conn.pending.values():
            if not fut.done():
                fut.cancel()
        conn.pending.clear()

    def get(self, device_id: str) -> Optional[_NodeConn]:
        return self._conns.get(device_id)

    def online(self, device_id: str) -> bool:
        return device_id in self._conns

    async def request(
        self,
        device_id: str,
        method: str,
        path: str,
        body: Any = None,
        timeout: float = NODE_API_TIMEOUT_S,
    ) -> Optional[Dict[str, Any]]:
        """Send an api_request frame and await its api_response.

        Returns the node's response frame, ``None`` when the node has no
        tunnel, and raises ``asyncio.TimeoutError`` on a hung request.
        """
        conn = self._conns.get(device_id)
        if conn is None:
            return None
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        conn.pending[request_id] = fut
        try:
            await conn.send({
                "type": "api_request",
                "id": request_id,
                "method": method.upper(),
                "path": path,
                "body": body,
            })
            return await asyncio.wait_for(fut, timeout)
        finally:
            conn.pending.pop(request_id, None)


manager = NodeWSManager()


# ---------------------------------------------------------------------------
# Small DB helpers (short-lived sessions — safe inside the WS loop)
# ---------------------------------------------------------------------------

def _mark_online(device_uuid: str, online: bool) -> None:
    try:
        with get_db_session() as db:
            db.query(Device).filter(Device.device_uuid == device_uuid).update(
                {"online": online, "last_seen": datetime.utcnow()})
    except Exception as exc:  # never let bookkeeping kill the tunnel
        logger.warning("[node-ws] online update failed for %s: %s",
                       device_uuid, exc)


def _touch_seen(device_uuid: str) -> None:
    try:
        with get_db_session() as db:
            db.query(Device).filter(Device.device_uuid == device_uuid).update(
                {"last_seen": datetime.utcnow()})
    except Exception as exc:
        logger.debug("[node-ws] last_seen update failed for %s: %s",
                     device_uuid, exc)


def _write_event(
    user_id: int,
    device_uuid: str,
    kind: str,
    data: Any,
    ts: Optional[float] = None,
    external_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert one node_event row; idempotent on (user, device, external_id)."""
    row = NodeEvent(
        user_id=user_id,
        device_id=device_uuid,
        kind=str(kind or "event")[:80],
        data=json.dumps(data if data is not None else {}),
        ts=float(ts or time()),
        external_id=str(external_id)[:255] if external_id else None,
    )
    with get_db_session() as db:
        db.add(row)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            existing = db.query(NodeEvent).filter(
                NodeEvent.user_id == user_id,
                NodeEvent.device_id == device_uuid,
                NodeEvent.external_id == row.external_id,
            ).first()
            if existing is not None:
                return existing.to_dict()
            raise
        doc = row.to_dict()
    # Push to live subscribers (SSE /v1/events/stream) — never block the
    # node channel on a slow consumer.
    event_bus.publish(user_id, doc)
    try:
        from server.event_delivery import enqueue
        enqueue(user_id, doc)
    except Exception:
        logger.debug("[node-ws] webhook enqueue failed", exc_info=True)
    return doc


def _apply_prediction_context(user_id: int, device_uuid: str,
                              data: Any) -> None:
    """Project a node ``prediction`` event into the context store.

    Normalized context (CONTRACT §5): every model output reduces to a
    ``ContextState`` keyed ``prediction`` per device whose value is the
    label string. A changed value emits a ``ContextEvent`` and the
    automation engine evaluates the user's rules — so
    ``empty → occupied`` needs nothing but a ``when: {key: "prediction",
    equals: "occupied"}`` rule. Non-dict payloads and errors degrade to
    a logged no-op; prediction events must never break the node channel.
    """
    if not isinstance(data, dict):
        return
    label = data.get("label") or data.get("prediction")
    if not label:
        return
    try:
        from server.automation import evaluate_user_rules
        from server.db import ContextEvent, ContextState
        now = time()
        with get_db_session() as db:
            state = db.query(ContextState).filter(
                ContextState.user_id == user_id,
                ContextState.state_key == "prediction",
                ContextState.entity_id == device_uuid).first()
            previous = None
            event_type = ""
            if state is None:
                state = ContextState(
                    user_id=user_id, state_key="prediction",
                    entity_id=device_uuid, since=now)
                db.add(state)
                event_type = "entered"
            else:
                previous = json.loads(state.value) if state.value else None
                if previous != label:
                    event_type = "changed"
                    state.since = now
            state.value = json.dumps(label)
            state.confidence = float(data.get("confidence") or 0.0)
            state.estimator = str(data.get("runtime_model_id")
                                  or data.get("model_id") or "node")
            db.commit()
            db.refresh(state)
            if event_type:
                event = ContextEvent(
                    user_id=user_id, event_key="prediction",
                    event_type=event_type, entity_id=device_uuid,
                    state_id=str(state.id), value=json.dumps(label),
                    previous_value=json.dumps(previous),
                    confidence=state.confidence, timestamp=now,
                    provenance=json.dumps({"estimator": state.estimator}))
                db.add(event)
                db.commit()
                db.refresh(event)
                evaluate_user_rules(db, user_id, trigger={
                    "state_id": state.id, "event_id": event.id,
                    "state": state.to_dict()})
    except Exception:
        logger.exception("[node-ws] prediction→context failed for %s",
                         device_uuid)


def _cache_room(user_id: int, device_uuid: str, doc: Dict[str, Any]) -> None:
    with get_db_session() as db:
        row = db.query(NodeRoom).filter(
            NodeRoom.device_id == device_uuid).first()
        if row is None:
            row = NodeRoom(user_id=user_id, device_id=device_uuid,
                           doc=json.dumps(doc))
            db.add(row)
        else:
            row.doc = json.dumps(doc)
            row.user_id = user_id


def _record_usage(
    db: Session,
    *,
    user_id: int,
    device_uuid: str,
    source: str,
    kind: str,
    model_id: Optional[str] = None,
    latency_ms: Optional[float] = None,
    tokens: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    ts: Optional[float] = None,
) -> ApiUsage:
    row = ApiUsage(
        user_id=user_id,
        device_id=device_uuid,
        ts=float(ts or time()),
        source=(source or "api")[:40],
        kind=(kind or "api")[:40],
        model_id=str(model_id)[:255] if model_id else None,
        latency_ms=latency_ms,
        tokens=tokens,
        meta=json.dumps(meta or {}),
    )
    db.add(row)
    return row


def _infer_kind(path: str) -> Optional[str]:
    """Map a node API path to a usage kind; ``None`` = not metered."""
    p = (path or "").lower()
    if "predict" in p or "/inference" in p:
        return "prediction"
    if "deploy" in p or "install" in p or "activate" in p:
        return "deploy"
    if "capture" in p or "minutes" in p:
        return "capture"
    if "automation" in p:
        return "automation"
    if "actuator" in p or "/actions" in p:
        return "actuate"
    return None


def _owned_device(device_id: str, user: User, db: Session) -> Device:
    device = db.query(Device).filter(
        Device.device_uuid == device_id,
        Device.userId == user.userId,
    ).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device


def _authenticate_device_token(
    token: str,
    device_id: Optional[str],
    db: Session,
) -> Tuple[Device, Dict[str, Any]]:
    """Validate a ``domain='device'`` JWT against the target device."""
    try:
        payload, domain = decode_token_any(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid device token")
    # The scope claim is the discriminator (same as the heartbeat path) —
    # the credential *domain* can misreport when deployments share secrets.
    if "device" not in (payload.get("scopes") or []):
        raise HTTPException(status_code=401, detail="Not a device credential")
    claim = str(payload.get("device_id") or "")
    if device_id and claim and claim != device_id:
        raise HTTPException(status_code=403,
                            detail="Device token does not match this device")
    target = device_id or claim
    device = db.query(Device).filter(
        Device.device_uuid == target).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device, payload


def _principal(
    request: Request,
    db: Session,
) -> Tuple[str, Any, Optional[Device]]:
    """Resolve the caller for dual-auth endpoints.

    Returns ``("device", payload, Device)`` for a node JWT or
    ``("user", User, None)`` for a user credential (bearer/session/x-api-key).
    """
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:]
        try:
            payload, _domain = decode_token_any(token)
            if "device" in (payload.get("scopes") or []):
                device = db.query(Device).filter(
                    Device.device_uuid == str(payload.get("device_id") or "")
                ).first()
                if device:
                    return "device", payload, device
        except JWTError:
            pass
    # User credential path — get_current_user enforces token/session/api-key.
    token = auth[7:] if auth.lower().startswith("bearer ") else None
    user = get_current_user(request=request, token=token, db=db)
    return "user", user, None


# ---------------------------------------------------------------------------
# WebSocket endpoint (node side)
# ---------------------------------------------------------------------------

@router.websocket("/node/ws")
async def node_ws(
    websocket: WebSocket,
    device_id: str = Query(""),
    token: str = Query(""),
):
    """The node's outbound tunnel. Auth happens before accept so a bad
    credential gets a clean WS close instead of an ambiguous HTTP error."""
    device_uuid = str(device_id or "")
    with get_db_session() as db:
        try:
            device, _payload = _authenticate_device_token(
                token, device_uuid, db)
        except HTTPException:
            await websocket.close(code=4401)
            return
        user_id = device.userId
        device_uuid = device.device_uuid

    await websocket.accept()
    conn = _NodeConn(websocket)
    manager.register(device_uuid, conn)
    await asyncio.to_thread(_mark_online, device_uuid, True)
    logger.info("[node-ws] node %s connected", device_uuid)
    try:
        while True:
            frame = await websocket.receive_json()
            if not isinstance(frame, dict):
                continue
            ftype = frame.get("type")
            if ftype == "api_response":
                fut = conn.pending.get(str(frame.get("id") or ""))
                if fut is not None and not fut.done():
                    fut.set_result(frame)
            elif ftype == "room_changed":
                doc = frame.get("data")
                if isinstance(doc, dict):
                    await asyncio.to_thread(
                        _cache_room, user_id, device_uuid, doc)
                    await asyncio.to_thread(
                        _write_event, user_id, device_uuid,
                        "room_changed", doc, frame.get("ts"),
                        frame.get("id"))
            elif ftype in ("event", "metadata"):
                kind = frame.get("kind") or (
                    "metadata" if ftype == "metadata" else "event")
                await asyncio.to_thread(
                    _write_event, user_id, device_uuid, kind,
                    frame.get("data"), frame.get("ts"), frame.get("id"))
                if kind == "prediction":
                    await asyncio.to_thread(
                        _apply_prediction_context, user_id,
                        device_uuid, frame.get("data") or {})
            await asyncio.to_thread(_touch_seen, device_uuid)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("[node-ws] node %s socket error: %s", device_uuid, exc)
    finally:
        manager.unregister(device_uuid, conn)
        await asyncio.to_thread(_mark_online, device_uuid, False)
        logger.info("[node-ws] node %s disconnected", device_uuid)


# ---------------------------------------------------------------------------
# REST→WS relay
# ---------------------------------------------------------------------------

class NodeApiRequest(BaseModel):
    method: str = "GET"
    path: str
    body: Any = None


@router.post("/nodes/{device_id}/api")
async def node_api_relay(
    device_id: str,
    request: NodeApiRequest,
    raw: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Forward ``{method,path,body}`` to the node over its tunnel and return
    the node's status + body verbatim — portal/mobile's single choke point."""
    device = _owned_device(device_id, current_user, db)
    if not request.path.startswith("/"):
        raise HTTPException(status_code=422,
                            detail="path must start with '/'")

    started = perf_counter()
    source = (raw.headers.get("x-thoth-source") or "api")[:40]
    try:
        resp = await manager.request(
            device.device_uuid, request.method, request.path, request.body)
    except asyncio.TimeoutError:
        resp = {"__timeout__": True}
    latency_ms = (perf_counter() - started) * 1000.0

    kind = _infer_kind(request.path)
    if kind:
        model_id = None
        if isinstance(request.body, dict):
            model_id = (request.body.get("model_id")
                        or request.body.get("runtime_model_id"))
        _record_usage(
            db, user_id=current_user.userId, device_uuid=device.device_uuid,
            source=source, kind=kind, model_id=model_id, latency_ms=latency_ms,
            meta={"method": request.method.upper(), "path": request.path,
                  "status": (resp or {}).get("status")})
        db.commit()

    if resp is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "node offline (no ws tunnel)",
                     "device_id": device.device_uuid, "online": False})
    if resp.get("__timeout__"):
        return JSONResponse(
            status_code=504,
            content={"detail": "node api_request timed out",
                     "device_id": device.device_uuid})

    status_code = int(resp.get("status") or 200)
    if resp.get("body_b64") is not None:
        try:
            payload = base64.b64decode(str(resp["body_b64"]))
        except Exception:
            payload = b""
        return Response(
            content=payload, status_code=status_code,
            media_type=str(resp.get("content_type") or "application/octet-stream"))
    return JSONResponse(content=resp.get("body"), status_code=status_code)


# ---------------------------------------------------------------------------
# Room document (§1.2 — node authoritative, Brain caches)
# ---------------------------------------------------------------------------

@router.get("/nodes/{device_id}/room")
async def get_node_room(
    device_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    device = _owned_device(device_id, current_user, db)
    row = db.query(NodeRoom).filter(
        NodeRoom.device_id == device.device_uuid).first()
    return {
        "device_id": device.device_uuid,
        "room": row.to_dict() if row else None,
        "cached": row is not None,
        "updated_at": row.updated_at.isoformat() + "Z" if row and row.updated_at else None,
    }


@router.put("/nodes/{device_id}/room")
async def put_node_room(
    device_id: str,
    room: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Write-through: relay ``PUT /api/v1/room`` to the authoritative node;
    the cache updates only after the node accepts."""
    device = _owned_device(device_id, current_user, db)
    resp = await manager.request(
        device.device_uuid, "PUT", "/api/v1/room", room)
    if resp is None:
        raise HTTPException(status_code=503,
                            detail="node offline (no ws tunnel)")
    status_code = int(resp.get("status") or 200)
    if status_code >= 400:
        return JSONResponse(status_code=status_code,
                            content=resp.get("body")
                            or {"detail": "node rejected room document"})
    doc = resp.get("body") if isinstance(resp.get("body"), dict) else room
    _cache_room(current_user.userId, device.device_uuid, doc)
    return {"ok": True, "device_id": device.device_uuid, "room": doc}


# ---------------------------------------------------------------------------
# Events (§6 — notification feed)
# ---------------------------------------------------------------------------

class EventIn(BaseModel):
    device_id: Optional[str] = None
    kind: str
    data: Any = None
    ts: Optional[float] = None
    event_id: Optional[str] = None


@router.post("/events", status_code=201)
async def post_event(
    request: EventIn,
    raw: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Node-side REST fallback for event delivery (the WS frame is primary).

    Device JWTs are bound to their own device; user credentials may post
    for any owned device (used by tests and future producers).
    """
    kind, principal, device = _principal(raw, db)
    if kind == "device":
        target = device
        if request.device_id and request.device_id != target.device_uuid:
            raise HTTPException(status_code=403,
                                detail="Device token does not match device_id")
        user_id = target.userId
    else:
        target = _owned_device(request.device_id or "", principal, db)
        user_id = principal.userId
    if not request.kind:
        raise HTTPException(status_code=422, detail="kind is required")
    row = _write_event(user_id, target.device_uuid, request.kind,
                       request.data, request.ts, request.event_id)
    if request.kind == "prediction":
        await asyncio.to_thread(
            _apply_prediction_context, user_id,
            target.device_uuid, request.data or {})
    return {"ok": True, "event": row}


@router.get("/events")
async def list_events(
    device_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    since: Optional[str] = Query(
        None, description="event id (int) or epoch ts (float) — strict >"),
    since_ts: Optional[float] = Query(None),
    kind: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(NodeEvent).filter(NodeEvent.user_id == current_user.userId)
    if device_id:
        device = _owned_device(device_id, current_user, db)
        query = query.filter(NodeEvent.device_id == device.device_uuid)
    if kind:
        query = query.filter(NodeEvent.kind == kind)
    if since:
        try:
            query = query.filter(NodeEvent.id > int(since))
        except (TypeError, ValueError):
            try:
                query = query.filter(NodeEvent.ts > float(since))
            except (TypeError, ValueError):
                raise HTTPException(status_code=422,
                                    detail="since must be an event id or ts")
    if since_ts is not None:
        query = query.filter(NodeEvent.ts > since_ts)
    rows = query.order_by(NodeEvent.id.desc()).limit(limit).all()
    events = [r.to_dict() for r in rows]
    events.reverse()  # chronological for the notification feed
    return {"events": events, "count": len(events)}


# ---------------------------------------------------------------------------
# API usage metering (§4)
# ---------------------------------------------------------------------------

class UsageIn(BaseModel):
    device_id: Optional[str] = None
    source: str = "api"
    kind: str
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    ts: Optional[float] = None


@router.post("/usage", status_code=201)
async def post_usage(
    request: UsageIn,
    raw: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    kind, principal, device = _principal(raw, db)
    if kind == "device":
        target = device
        if request.device_id and request.device_id != target.device_uuid:
            raise HTTPException(status_code=403,
                                detail="Device token does not match device_id")
        user_id = target.userId
    else:
        target = _owned_device(request.device_id or "", principal, db)
        user_id = principal.userId
    if request.kind not in USAGE_KINDS and request.kind != "api":
        raise HTTPException(
            status_code=422,
            detail=f"kind must be one of {sorted(USAGE_KINDS | {'api'})}")
    row = _record_usage(
        db, user_id=user_id, device_uuid=target.device_uuid,
        source=request.source if request.source in USAGE_SOURCES else "api",
        kind=request.kind, model_id=request.model_id,
        latency_ms=request.latency_ms, tokens=request.tokens,
        meta=request.meta, ts=request.ts)
    db.commit()
    db.refresh(row)
    return {"ok": True, "usage": row.to_dict()}


@router.get("/usage")
async def list_usage(
    device_id: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    since_ts: Optional[float] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(ApiUsage).filter(
        ApiUsage.user_id == current_user.userId)
    if device_id:
        device = _owned_device(device_id, current_user, db)
        query = query.filter(ApiUsage.device_id == device.device_uuid)
    if kind:
        query = query.filter(ApiUsage.kind == kind)
    if source:
        query = query.filter(ApiUsage.source == source)
    if since_ts is not None:
        query = query.filter(ApiUsage.ts > since_ts)
    rows = query.order_by(ApiUsage.ts.desc()).limit(limit).all()
    return {"usage": [r.to_dict() for r in rows], "count": len(rows)}
