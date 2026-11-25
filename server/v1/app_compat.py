"""Mobile/desktop app surface — the endpoints the Thothcraft app's home
tabs load on start (conversations, memories, action items, folders,
people). Without them the app shows "something went wrong".

Documents are stored per user as JSON in ``app_doc`` (kind, doc_id) so
conversations created by any device — phone, desktop, Room1 — appear in
one history. Shapes match the app's decoders: conversation/memory/folder
lists are bare JSON arrays; action items use ``{"action_items", "has_more"}``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import User, get_db

router = APIRouter(tags=["app"])          # mounted under /v1
root_router = APIRouter(tags=["app"])     # mounted at / (v2/v3 paths)

_KINDS = {"conversation", "memory", "action_item", "folder", "person", "activity"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list(db: Session, user: User, kind: str, limit: int, offset: int) -> List[Dict[str, Any]]:
    rows = db.execute(text(
        "SELECT body FROM app_doc WHERE user_id = :u AND kind = :k "
        "ORDER BY created_at DESC LIMIT :l OFFSET :o"),
        {"u": user.userId, "k": kind, "l": limit, "o": offset}).fetchall()
    out = []
    for (body,) in rows:
        try:
            out.append(json.loads(body))
        except (TypeError, ValueError):
            continue
    return out


def _get(db: Session, user: User, kind: str, doc_id: str) -> Dict[str, Any]:
    row = db.execute(text(
        "SELECT body FROM app_doc WHERE user_id = :u AND kind = :k AND doc_id = :d"),
        {"u": user.userId, "k": kind, "d": doc_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"{kind} not found")
    return json.loads(row[0])


def _put(db: Session, user: User, kind: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = str(doc.get("id") or uuid.uuid4())
    doc["id"] = doc_id
    doc.setdefault("created_at", _now())
    db.execute(text(
        "INSERT INTO app_doc (user_id, kind, doc_id, body, created_at) "
        "VALUES (:u, :k, :d, :b, CURRENT_TIMESTAMP) "
        "ON CONFLICT (user_id, kind, doc_id) DO UPDATE SET body = EXCLUDED.body"),
        {"u": user.userId, "k": kind, "d": doc_id, "b": json.dumps(doc)})
    db.commit()
    return doc


def _delete(db: Session, user: User, kind: str, doc_id: str) -> None:
    db.execute(text(
        "DELETE FROM app_doc WHERE user_id = :u AND kind = :k AND doc_id = :d"),
        {"u": user.userId, "k": kind, "d": doc_id})
    db.commit()


# -- conversations --------------------------------------------------------------

@router.get("/conversations")
async def list_conversations(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                             user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _list(db, user, "conversation", limit, offset)


@router.post("/conversations")
async def create_conversation(payload: Dict[str, Any] = Body(...),
                              user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    payload.setdefault("structured", {"title": "", "overview": "", "emoji": "", "category": "other",
                                      "action_items": [], "events": []})
    payload.setdefault("transcript_segments", [])
    payload.setdefault("status", "completed")
    payload.setdefault("discarded", False)
    payload.setdefault("source", "phone")
    return _put(db, user, "conversation", payload)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user: User = Depends(get_current_user),
                           db: Session = Depends(get_db)):
    return _get(db, user, "conversation", conversation_id)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user: User = Depends(get_current_user),
                              db: Session = Depends(get_db)):
    _delete(db, user, "conversation", conversation_id)
    return {"status": "ok"}


# -- action items / folders / people -------------------------------------------

@router.get("/action-items")
async def list_action_items(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                            user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = _list(db, user, "action_item", limit + 1, offset)
    return {"action_items": items[:limit], "has_more": len(items) > limit}


@router.post("/action-items")
async def create_action_item(payload: Dict[str, Any] = Body(...),
                             user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    payload.setdefault("completed", False)
    return _put(db, user, "action_item", payload)


@router.get("/folders")
async def list_folders(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _list(db, user, "folder", 200, 0)


@router.get("/users/people")
async def list_people(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _list(db, user, "person", 500, 0)


# -- activity timeline (physical + digital, every device) ----------------------
# One event = {ts, device: phone|desktop|room1|watch|necklace|earbuds,
#              kind: physical|digital, label, detail?, duration_s?}.

@router.post("/activity")
async def post_activity(payload: Dict[str, Any] = Body(...),
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    events = payload.get("events") if isinstance(payload.get("events"), list) else [payload]
    stored = 0
    for ev in events[:500]:
        if not isinstance(ev, dict) or not ev.get("label"):
            continue
        ev = {k: ev[k] for k in ("id", "ts", "device", "kind", "label", "detail", "duration_s") if k in ev}
        ev.setdefault("ts", _now())
        ev["kind"] = ev.get("kind") if ev.get("kind") in ("physical", "digital") else "digital"
        _put(db, user, "activity", ev)
        stored += 1
    return {"status": "ok", "stored": stored}


@router.get("/activity")
async def get_activity(limit: int = Query(200, ge=1, le=1000),
                       user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    events = _list(db, user, "activity", limit, 0)
    return {"events": sorted(events, key=lambda e: str(e.get("ts", "")), reverse=True)}


# -- memories (v3 namespace) -----------------------------------------------------

@root_router.get("/v3/memories")
async def list_memories(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _list(db, user, "memory", limit, offset)


@root_router.post("/v3/memories")
async def create_memory(payload: Dict[str, Any] = Body(...),
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    payload.setdefault("category", "other")
    payload.setdefault("visibility", "private")
    return _put(db, user, "memory", payload)


@root_router.delete("/v3/memories/{memory_id}")
async def delete_memory(memory_id: str, user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    _delete(db, user, "memory", memory_id)
    return {"status": "ok"}
