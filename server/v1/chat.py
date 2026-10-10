"""Brain v1 chat — context-grounded assistant with voice + files.

``POST /v1/chat`` answers questions using the same bundle the context
builder feeds its map model — the semantic map (entities/relationships/
states with ``confirmed`` flags), recent descriptor aggregates and
scenes — plus optional attachments (images go to the vision model,
text/pdf files are inlined). The reply is a structured "rich answer"
(``answer`` markdown + ``widgets``) the client renders inline; the
response echoes ``context_used`` so the UI can show exactly what the
model was given.

``GET /v1/chat/context`` returns that bundle on its own — the answer
to "what does the model see?" without spending an inference call.

``POST /v1/chat/transcribe`` and ``POST /v1/chat/tts`` proxy the OpenAI
voice APIs (speech→text and text→speech) so keys stay server-side.

Every hosted call is authorized + metered through
``server.inference_auth`` (kinds: ``chat_query``, ``chat_transcribe``,
``chat_tts``).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter, Depends, File, HTTPException, Response, UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user, get_scoped_principal
from server.db import ContextEntity, User, get_db
from server.inference_auth import (
    INFERENCE_SOURCE, inference_call, inference_status,
)
from .context_builder import BUILDER_ENTITY, build_bundle, map_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["v1", "chat"])

_DEFAULT_MODEL = "gpt-4o-mini"
_ADVANCED_MODEL = "gpt-4o"
# Numbered user-facing model picks — the app/hub chooser sends "4"|"5"|"6";
# each maps to a concrete chat model, env-overridable per pick
# (CHAT_MODEL_4 / CHAT_MODEL_5 / CHAT_MODEL_6).
_TIER_MODELS = {
    "4": "gpt-4o",
    "5": "gpt-5",
    "6": "gpt-6",
}
_TIERS = ("standard", "advanced", *_TIER_MODELS)
# Picks that get the widened context bundle + bigger message budget.
_WIDE_TIERS = frozenset({"advanced", "5", "6"})
_MAX_HISTORY = 12
_MAX_HISTORY_CHARS = 2000
_MAX_ATTACHMENTS = 5
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_FILE_CHARS = 16_000
_MAX_TTS_CHARS = 4096
_MAX_AUDIO_BYTES = 25 * 1024 * 1024

_TEXT_EXTS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".log", ".py",
    ".yaml", ".yml", ".xml", ".html", ".ini", ".cfg", ".sql", ".sh",
    ".ps1", ".js", ".ts", ".dart", ".cpp", ".c", ".h", ".ino", ".toml",
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AttachmentIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    mime: str = "application/octet-stream"
    data_b64: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str                       # user | assistant
    content: str


class ChatRequest(BaseModel):
    message: str = ""
    history: List[ChatMessage] = Field(default_factory=list)
    attachments: List[AttachmentIn] = Field(default_factory=list)
    include_context: bool = True
    model: str = Field(
        default="standard",
        description="Model pick: '4'|'5'|'6' (user-facing chooser — "
                    "4 = gpt-4o-class fast, 5/6 = deeper reasoning + a "
                    "much larger context bundle) or the legacy tiers "
                    "'standard'/'advanced'. Bogus pick → 422.")


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=_MAX_TTS_CHARS)
    voice: Optional[str] = None


# ---------------------------------------------------------------------------
# The context bundle the model sees
# ---------------------------------------------------------------------------

def _chat_context(db: Session, user_id: int, user: Optional[User],
                  now: Optional[float] = None, *,
                  detail: str = "standard") -> Dict[str, Any]:
    """The evidence bundle behind an answer — semantic map + fresh
    descriptor aggregates + builder status + quota. Identical inputs to
    the context builder's own bundle so the assistant sees what the map
    sees. ``advanced`` widens every cap — more descriptor groups, more
    scenes and the builder's last proposal — for the bigger model."""
    now = now or time.time()
    bundle = build_bundle(db, user_id, now, window_s=900.0)
    builder_ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == BUILDER_ENTITY).first()
    battrs = {}
    if builder_ent and builder_ent.attributes:
        try:
            battrs = json.loads(builder_ent.attributes)
        except (TypeError, ValueError):
            battrs = {}
    advanced = detail in _WIDE_TIERS
    builder: Dict[str, Any] = {
        "builds": battrs.get("builds", 0),
        "last_build_at": battrs.get("last_build_at"),
        "last_summary": battrs.get("last_summary"),
        "pending": len(battrs.get("pending", {}) or {}),
    }
    if advanced:
        builder["last_proposal"] = battrs.get("last_proposal")
    return {
        "generated_at": now,
        "window_s": bundle.get("window_s", 900.0),
        "map": bundle["map"],
        "devices": (bundle.get("devices") or [])
                   [:30 if advanced else 12],
        "descriptors": (bundle.get("descriptors") or [])
                       [:32 if advanced else 12],
        "scenes": (bundle.get("scenes") or [])
                  [:10 if advanced else 5],
        "reference": bundle.get("reference"),
        "builder": builder,
        "usage": inference_status(db, user) if user else None,
    }


# ---------------------------------------------------------------------------
# Attachments → model content parts
# ---------------------------------------------------------------------------

def _b64_bytes(att: AttachmentIn) -> bytes:
    try:
        return base64.b64decode(att.data_b64, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(422, f"attachment {att.name!r} is not base64")


def _pdf_text(data: bytes) -> str:
    try:
        import PyPDF2  # already a platform dep
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join(
            (page.extract_text() or "") for page in reader.pages[:10])
    except Exception:
        logger.debug("pdf text extraction failed", exc_info=True)
        return ""


def _attachment_parts(
    att: AttachmentIn,
) -> Tuple[List[Dict[str, Any]], str]:
    """One attachment → (OpenAI content parts, receipt note)."""
    mime = (att.mime or "").lower()
    ext = os.path.splitext(att.name or "")[1].lower()
    if mime.startswith("image/"):
        if len(att.data_b64) * 3 // 4 > _MAX_IMAGE_BYTES:
            raise HTTPException(413, f"image {att.name!r} exceeds 8 MB")
        return ([{"type": "image_url",
                  "image_url": {"url":
                                f"data:{mime};base64,{att.data_b64}"}}],
                f"image:{att.name}")
    raw = _b64_bytes(att)
    if mime == "application/pdf" or ext == ".pdf":
        text = _pdf_text(raw)
        if not text.strip():
            return ([{"type": "text",
                      "text": f"[attached file '{att.name}' — a PDF whose "
                              "text could not be extracted]"}],
                    f"pdf:{att.name}")
        return ([{"type": "text",
                  "text": f"[attached file '{att.name}']\n"
                          + text[:_MAX_FILE_CHARS]}],
                f"pdf:{att.name}")
    if mime.startswith("text/") or ext in _TEXT_EXTS:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return ([{"type": "text",
                  "text": f"[attached file '{att.name}']\n"
                          + (text[:_MAX_FILE_CHARS]
                             or "(empty or unreadable)")}],
                f"file:{att.name}")
    return ([{"type": "text",
              "text": f"[attached file '{att.name}' ({mime or 'unknown'}) "
                      "— this file type cannot be read and was not "
                      "included]"}],
            f"skipped:{att.name}")


# ---------------------------------------------------------------------------
# Rich-answer call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Thothcraft assistant answering questions about the user's
physical spaces, devices and activity. You receive a context bundle —
the same semantic map the context builder maintains (entities,
relationships, derived states; entries marked `confirmed: true` are
user-confirmed facts), a `devices` registry telling you which physical
machine each `device:<uuid>` / scene `device` id is (name, type, MAC,
hardware — anything not listed is an unknown emitter), recent evidence
descriptor aggregates and scene summaries — sensor entries carry
physical `fields` (mean/min/max/n for radar SNR, CSI variance, RSSI,
audio level…) plus node `estimates` and `location`, and `reference` —
measured empty-vs-occupied stats for the radar/CSI fields (E1 rig:
empty room snr mean ~7.6 dB tight, occupancy shows as snr_max spikes
>15-40 dB + map-mean lifts — a modest median SNR alone is weak) — then
the user's question and optional attachments.

Rules:
  * Answer ONLY from the bundle + attachments + conversation — never
    invent telemetry. Say when evidence is missing, stale or weak.
  * Confirmed facts win over inferred ones; describe inferred entries as
    tentative ("the system believes…").
  * You can explain what the model was shown — the user can ask "what
    context did you use?"; reference the bundle fields directly.
  * Be concise. No prose outside the JSON object.

Respond ONLY with this JSON shape:
{"answer": "<markdown text>",
 "widgets": [
   {"type": "list", "title": "…", "items": ["…"]},
   {"type": "key_values", "title": "…",
    "items": [{"label": "…", "value": "…"}]},
   {"type": "table", "title": "…",
    "columns": ["…"], "rows": [["…", "…"]]},
   {"type": "states", "title": "…",
    "items": [{"key": "…", "entity": "…", "value": "…",
               "confidence": 0.9, "confirmed": true}]},
   {"type": "questions", "items": ["natural follow-up question?"]}
 ]}

Widget guidance — include only what genuinely helps: list for
enumerations, key_values for summaries, table for comparisons, states
when quoting live context verbatim, questions for 2–4 natural
follow-ups. `answer` is always required; widgets may be an empty array.
"""


def _model_for_tier(tier: str) -> str:
    """Slider tier → concrete model id, env-overridable per tier
    (``CHAT_MODEL``, ``CHAT_MODEL_ADVANCED``). Unknown → 422."""
    tier = (tier or "standard").lower()
    if tier == "standard":
        return (os.getenv("CHAT_MODEL")
                or os.getenv("CONTEXT_INFER_MODEL")
                or os.getenv("MODEL_NAME") or _DEFAULT_MODEL)
    if tier == "advanced":
        return os.getenv("CHAT_MODEL_ADVANCED") or _ADVANCED_MODEL
    if tier in _TIER_MODELS:
        return (os.getenv(f"CHAT_MODEL_{tier}")
                or _TIER_MODELS[tier])
    raise HTTPException(422, f"unknown model tier {tier!r} "
                             f"— one of {list(_TIERS)}")


def _openai_answer(messages: List[Dict[str, Any]], *,
                   tier: str = "standard",
                   usage_out: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """One JSON-mode completion → parsed rich answer."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = _model_for_tier(tier)
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3,
        response_format={"type": "json_object"})
    if usage_out is not None:
        usage_out["model_id"] = model
        usage_out["tokens"] = getattr(
            getattr(resp, "usage", None), "total_tokens", None)
    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not an object")
    except (ValueError, TypeError):
        parsed = {"answer": raw, "widgets": []}
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        parsed["answer"] = raw or "(empty reply)"
    widgets = parsed.get("widgets")
    parsed["widgets"] = ([w for w in widgets if isinstance(w, dict)]
                         if isinstance(widgets, list) else [])
    return parsed


def _openai_transcribe(filename: str, data: bytes, mime: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = os.getenv("CHAT_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    client = OpenAI(api_key=api_key)
    result = client.audio.transcriptions.create(
        model=model, file=(filename or "audio.webm", data,
                           mime or "application/octet-stream"))
    return getattr(result, "text", "") or ""


def _openai_tts(text: str, voice: str) -> bytes:
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = os.getenv("CHAT_TTS_MODEL", "gpt-4o-mini-tts")
    client = OpenAI(api_key=api_key)
    resp = client.audio.speech.create(
        model=model, voice=voice or os.getenv("CHAT_TTS_VOICE", "alloy"),
        input=text)
    return b"".join(resp.iter_bytes())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_scoped_principal("chat:write")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Question (+ attachments) → context-grounded rich answer."""
    tier = (body.model or "standard").lower()
    if tier not in _TIERS:
        raise HTTPException(422, f"unknown model tier {tier!r} "
                                 f"— one of {list(_TIERS)}")
    context_used = (_chat_context(db, current_user.userId, current_user,
                                  detail=tier)
                    if body.include_context else None)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT}]
    if context_used is not None:
        messages.append({
            "role": "user",
            "content": "[context bundle]\n"
                       + json.dumps(context_used, default=str)
                       [:120_000 if tier in _WIDE_TIERS else 60_000]})
        messages.append({"role": "assistant",
                         "content": "Understood — I will answer only from "
                                    "that bundle plus what you send."})
    for m in (body.history or [])[-_MAX_HISTORY:]:
        if m.role not in ("user", "assistant") or not m.content:
            continue
        messages.append({"role": m.role,
                         "content": m.content[:_MAX_HISTORY_CHARS]})

    parts: List[Dict[str, Any]] = []
    attachment_notes: List[str] = []
    for i, att in enumerate((body.attachments or [])[:_MAX_ATTACHMENTS]):
        att_parts, note = _attachment_parts(att)
        parts.extend(att_parts)
        attachment_notes.append(note)
    text = (body.message or "").strip()
    if text:
        parts.insert(0, {"type": "text", "text": text})
    if not parts:
        raise HTTPException(422, "message or attachment required")
    messages.append({"role": "user",
                     "content": parts if len(parts) > 1
                     or parts[0]["type"] != "text" else text})

    with inference_call(db, current_user, kind="chat_query") as meter:
        parsed = _openai_answer(messages, tier=tier, usage_out=meter)

    return {
        "answer": parsed["answer"],
        "widgets": parsed["widgets"],
        "context_used": context_used,
        "attachments": attachment_notes,
        "model": {"tier": tier,
                  "id": _model_for_tier(tier)},
        "generated_at": time.time(),
    }


@router.get("/context")
async def chat_context(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The bundle the assistant would be given — what the model sees."""
    return _chat_context(db, current_user.userId, current_user)


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    current_user: User = Depends(get_scoped_principal("chat:write")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Audio clip → transcript (OpenAI speech-to-text, proxied)."""
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty audio")
    if len(data) > _MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio clip exceeds 25 MB")
    with inference_call(db, current_user,
                        kind="chat_transcribe") as meter:
        text = _openai_transcribe(
            file.filename or "audio.webm", data,
            file.content_type or "audio/webm")
    return {"text": text}


@router.post("/tts")
async def tts(
    body: TtsRequest,
    current_user: User = Depends(get_scoped_principal("chat:write")),
    db: Session = Depends(get_db),
) -> Response:
    """Text → spoken audio/mpeg (OpenAI text-to-speech, proxied)."""
    with inference_call(db, current_user, kind="chat_tts"):
        audio = _openai_tts(body.text[:_MAX_TTS_CHARS],
                            body.voice or "alloy")
    return Response(content=audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})
