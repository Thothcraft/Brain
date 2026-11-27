"""Mobile/desktop app surface — the endpoints the Thothcraft app's home
tabs load on start (conversations, memories, action items, folders,
people). Without them the app shows "something went wrong".

Documents are stored per user as JSON in ``app_doc`` (kind, doc_id) so
conversations created by any device — phone, desktop, Room1 — appear in
one history. Shapes match the app's decoders: conversation/memory/folder
lists are bare JSON arrays; action items use ``{"action_items", "has_more"}``.
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import threading
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (APIRouter, BackgroundTasks, Body, Depends, File, Header,
                     HTTPException, Query, UploadFile)
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from server import privacy
from server.auth import get_current_user
from server.db import User, get_db, get_db_session

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


# -- WAL audio sync (v2) -------------------------------------------------------
#
# The app's pendant WAL upload path: POST /v2/sync-local-files accepts
# framed .bin files ([u32 LE pkt_len][opus packet]* @16 kHz mono), answers
# 202 + job_id, and the reconciler polls GET /v2/sync-local-files/{job_id}
# in the app's SyncJobStatusResponse shape. The job decodes opus → WAV,
# transcribes via the same OpenAI path as /v1/chat/transcribe, and stores
# a conversation doc so the recording lands in the user's history.

_SYNC_JOBS: Dict[str, Dict[str, Any]] = {}
_SYNC_JOBS_LOCK = threading.Lock()
_MAX_SYNC_FRAME_BYTES = 4 * 1024 * 1024
_SYNC_TMP = Path(tempfile.gettempdir()) / "thoth_sync"
_OPUS_RATE = 16000


def _decode_wal_bin(path: str) -> tuple:
    """framed .bin → (wav_bytes, duration_seconds)."""
    try:
        import opuslib
    except Exception as exc:
        raise RuntimeError(
            "opuslib unavailable — libopus is required on the server image") from exc
    decoder = opuslib.Decoder(_OPUS_RATE, 1)
    pcm = bytearray()
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (n,) = struct.unpack("<I", hdr)
            if n == 0 or n > _MAX_SYNC_FRAME_BYTES:
                break
            pkt = f.read(n)
            if len(pkt) < n:
                break
            pcm += decoder.decode(pkt, frame_size=5760)
    if not pcm:
        raise ValueError("no decodable opus frames")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_OPUS_RATE)
        w.writeframes(bytes(pcm))
    return buf.getvalue(), len(pcm) / (2 * _OPUS_RATE)


def _title_for(transcript: str) -> str:
    """Short conversation title via the chat model; falls back silently."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_TITLE_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            max_tokens=16,
            messages=[
                {"role": "system", "content":
                 "Title this transcript in 6 words or fewer. Reply with the title only."},
                {"role": "user", "content": transcript[:4000]},
            ])
        return (resp.choices[0].message.content or "").strip().strip('"')
    except Exception:
        return ""


def _run_sync_job(job_id: str, user_id: str, paths: List[str],
                  started_s: Optional[float], finished_s: Optional[float],
                  geolocation: Optional[str]) -> None:
    job = _SYNC_JOBS.get(job_id)
    if job is None:
        return
    from .chat import _openai_transcribe  # deferred — avoids an import cycle
    texts: List[str] = []
    durations: List[float] = []
    try:
        with get_db_session() as db:
            user = db.query(User).filter(User.userId == user_id).first()
            if user is None:
                raise RuntimeError("user row missing")
            for path in paths:
                try:
                    wav, dur = _decode_wal_bin(path)
                    text_out = _openai_transcribe("audio.wav", wav, "audio/wav")
                    if text_out.strip():
                        texts.append(text_out.strip())
                        durations.append(dur)
                        job["successful_segments"] += 1
                except Exception:
                    job["failed_segments"] += 1
                job["processed_segments"] += 1
            if texts:
                transcript = " ".join(texts)
                start_dt = (datetime.fromtimestamp(started_s, timezone.utc)
                            if started_s else datetime.now(timezone.utc))
                end_dt = (datetime.fromtimestamp(finished_s, timezone.utc)
                          if finished_s else datetime.now(timezone.utc))
                conv: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    "created_at": _now(),
                    "started_at": start_dt.isoformat(),
                    "finished_at": end_dt.isoformat(),
                    "source": "device_sync",
                    "status": "completed",
                    "discarded": False,
                    "transcript_segments": [
                        {"id": str(uuid.uuid4()), "text": t, "start": 0.0,
                         "end": d, "is_user": False, "speaker": "SPEAKER_00",
                         "speech_profile_processed": True}
                        for t, d in zip(texts, durations)],
                    "structured": {
                        "title": _title_for(transcript) or "Recording",
                        "overview": transcript[:400],
                        "emoji": "",
                        "category": "other",
                        "action_items": [],
                        "events": [],
                    },
                }
                if geolocation:
                    try:
                        conv["geolocation"] = json.loads(geolocation)
                    except (ValueError, TypeError):
                        pass
                _put(db, user, "conversation", conv)
                job["conversation_id"] = conv["id"]
            failed = job["failed_segments"]
            job["status"] = ("failed" if failed and not texts else
                             "partial_failure" if failed else "completed")
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)[:300]
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        job["result"] = {"total_segments": job["total_segments"],
                         "failed_segments": job["failed_segments"],
                         "new_memories": [], "updated_memories": []}


@root_router.post("/v2/sync-local-files")
async def sync_local_files(background_tasks: BackgroundTasks,
                           files: List[UploadFile] = File(...),
                           user: User = Depends(get_current_user),
                           audio_start_seconds: Optional[float] = Query(None),
                           audio_end_seconds: Optional[float] = Query(None),
                           x_omi_conversation_geolocation: Optional[str] =
                           Header(None)):
    privacy.require(user, "cloud_processing")
    if not files:
        raise HTTPException(422, "no files")
    _SYNC_TMP.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    paths: List[str] = []
    try:
        for f in files:
            data = await f.read()
            if not data:
                continue
            p = _SYNC_TMP / f"{job_id}-{len(paths)}.bin"
            p.write_bytes(data)
            paths.append(str(p))
    except Exception:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        raise
    if not paths:
        raise HTTPException(422, "empty upload")
    geo = x_omi_conversation_geolocation
    with _SYNC_JOBS_LOCK:
        _SYNC_JOBS[job_id] = {
            "job_id": job_id, "user_id": user.userId, "lane": "fresh",
            "status": "processing", "total_segments": len(paths),
            "processed_segments": 0, "successful_segments": 0,
            "failed_segments": 0, "poll_after_ms": 10000,
        }
    background_tasks.add_task(_run_sync_job, job_id, user.userId, paths,
                              audio_start_seconds, audio_end_seconds, geo)
    return JSONResponse(202, {"job_id": job_id, "lane": "fresh",
                            "poll_after_ms": 10000, "status": "processing",
                            "total_files": len(paths),
                            "total_segments": len(paths)})


@root_router.get("/v2/sync-local-files/{job_id}")
async def sync_job_status(job_id: str,
                          user: User = Depends(get_current_user)):
    job = _SYNC_JOBS.get(job_id)
    if job is None or job.get("user_id") != user.userId:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "user_id"}
