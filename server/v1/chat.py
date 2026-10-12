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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter, Depends, File, HTTPException, Response, UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user, get_scoped_principal
from server.db import ContextEntity, NodeRoom, User, get_db
from server.inference_auth import (
    INFERENCE_SOURCE, inference_call, inference_status,
)
from .context_builder import (
    BUILDER_ENTITY, _device_fp_index, _fingerprints, build_bundle,
    seed_devices,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["v1", "chat"])

_DEFAULT_MODEL = "gpt-4o-mini"
_ADVANCED_MODEL = "gpt-4o"
# Numbered user-facing model picks — the app/hub chooser sends "4"|"5"|"6";
# each maps to a concrete chat model, env-overridable per pick
# (CHAT_MODEL_4 / CHAT_MODEL_5 / CHAT_MODEL_6).
_TIER_MODELS = {
    "4": "gpt-4o",
    "5": "gpt-4.1",
    "6": "gpt-5",
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


class ChatConfig(BaseModel):
    """Per-user assistant tuning — what the bundle covers. Persisted on
    the ``chat:config`` system entity so the app/hub surfaces stay in
    sync; the context payload echoes it under ``config`` and documents
    every bound under ``provenance``."""
    window_s: float = Field(default=900.0, ge=60.0, le=86400.0)
    max_bundle_chars: int = Field(default=0, ge=0, le=250_000,
        description="0 = use the per-tier cap (60k / 120k wide)")
    include_scenes: bool = True
    include_descriptors: bool = True
    include_reference: bool = True
    include_builder: bool = True
    include_usage: bool = True
    include_persons_places: bool = True


_CONFIG_ENTITY = "chat:config"


def _load_config(db: Session, user_id: int) -> ChatConfig:
    ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == _CONFIG_ENTITY).first()
    if not ent or not ent.attributes:
        return ChatConfig()
    try:
        raw = json.loads(ent.attributes)
    except (TypeError, ValueError):
        return ChatConfig()
    try:
        return ChatConfig(**raw) if isinstance(raw, dict) else ChatConfig()
    except Exception:
        return ChatConfig()


# ---------------------------------------------------------------------------
# The context bundle the model sees
# ---------------------------------------------------------------------------

def _chat_context(db: Session, user_id: int, user: Optional[User],
                  now: Optional[float] = None, *,
                  detail: str = "standard",
                  cfg: Optional[ChatConfig] = None,
                  ) -> Dict[str, Any]:
    """The evidence bundle behind an answer — semantic map + fresh
    descriptor aggregates + builder status + quota. Identical inputs to
    the context builder's own bundle so the assistant sees what the map
    sees. ``advanced`` widens every cap — more descriptor groups, more
    scenes and the builder's last proposal — for the bigger model.
    Per-user ``cfg`` sets the evidence window and section toggles; the
    result echoes it plus a ``provenance`` block naming every section
    the model was actually shown."""
    cfg = cfg or _load_config(db, user_id)
    now = now or time.time()
    def is_device_gps(value: Any) -> bool:
        if isinstance(value, dict) and isinstance(value.get("value"), dict):
            value = value["value"]
        return isinstance(value, dict) and str(value.get("source") or "").casefold() in {
            "gps", "device_gps", "gnss"}
    # Same anchors the builder maintains — registry devices (with MAC/
    # hostname fingerprints) + person:owner/place:home — so the assistant
    # never answers against an unseeded map.
    seed_devices(db, user_id, now)
    bundle = build_bundle(db, user_id, now, window_s=cfg.window_s)
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

    # Devices + fingerprint table → which emitter is already KNOWN and
    # WHO owns it. Registry devices belong to the account owner by
    # definition; the fingerprint index links each registry row to its
    # canonical context entity (so "device:aa-bb" sightings fold onto
    # "device:<uuid>" the user owns).
    fp_idx = _device_fp_index(db, user_id)
    devices: List[Dict[str, Any]] = []
    for d in (bundle.get("devices") or [])[:30 if advanced else 12]:
        d = dict(d)
        refs = [str(d.get("uuid") or ""), str(d.get("name") or ""),
                str(d.get("mac") or ""), f"device:{d.get('uuid')}"]
        hits = [fp_idx.get(fp) for v in refs
                for fp in _fingerprints(v)]
        ent_key = next((h for h in hits if h), None)
        d["entity_key"] = ent_key
        d["owned"] = True
        d["owner"] = "person:owner"
        devices.append(d)

    persons, places = [], []
    if cfg.include_persons_places:
        for e in (bundle["map"].get("entities") or []):
            if e.get("kind") == "person":
                persons.append(e)
            elif e.get("kind") == "place":
                places.append(e)

    calibration: Dict[str, Any] = {}
    if cfg.include_reference and bundle.get("reference"):
        # The reference map IS the calibration — radar_occupancy /
        # csi_link / watch_imu_worn at top level (the old extra wrap
        # buried everything one level deeper than the prompt promises).
        calibration = dict(bundle["reference"])

    out: Dict[str, Any] = {
        "generated_at": now,
        "now_iso": datetime.fromtimestamp(
            now, tz=timezone.utc).isoformat(),
        "window_s": cfg.window_s,
        "map": bundle["map"],
        "devices": devices,
        "config": cfg.model_dump(),
        "provenance": {
            "sections": [],   # filled below — names every block sent
            "history": {"max_turns": _MAX_HISTORY,
                        "max_chars_per_turn": _MAX_HISTORY_CHARS},
            "attachments": {"max": _MAX_ATTACHMENTS,
                            "image_bytes": _MAX_IMAGE_BYTES,
                            "file_chars": _MAX_FILE_CHARS},
            "model_picks": {t: _model_for_tier(t) for t in _TIERS},
        },
    }
    if persons:
        out["persons"] = persons
    if places:
        out["places"] = places
    spaces = bundle.get("spaces") or []
    if spaces:
        out["spaces"] = spaces
    # The node's detailed room/v1 plan is a separate source from Brain's
    # account-space placements. Include it so chat sees furniture, room
    # geometry, device positions and manually placed people as authored.
    room_layouts = []
    try:
        for row in (db.query(NodeRoom).filter(NodeRoom.user_id == user_id)
                    .order_by(NodeRoom.updated_at.desc()).limit(20).all()):
            try:
                doc = json.loads(row.doc or "{}")
            except (TypeError, ValueError):
                doc = {}
            room_layouts.append({
                "device_id": row.device_id,
                "updated_at": row.updated_at.isoformat() + "Z"
                    if row.updated_at else None,
                "doc": doc,
            })
    except Exception:
        logger.exception("chat room layout load failed for user %s", user_id)
    if room_layouts:
        out["room_layouts"] = room_layouts

    # `valid_until` is a retention boundary, not proof that volatile data
    # remains true. Keep the map for auditability and expose a conservative
    # summary the assistant can use for present-tense claims.
    entities = bundle["map"].get("entities") or []
    entity_by_alias: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        for alias in (entity.get("id"), *(entity.get("aliases") or [])):
            if alias:
                entity_by_alias[str(alias).casefold()] = entity
    current_device_facts: List[Dict[str, Any]] = []
    now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
    for device in devices:
        entity = entity_by_alias.get(
            str(device.get("entity_key") or "").casefold())
        aliases = [str(device.get(k) or "").casefold()
                   for k in ("uuid", "name", "entity", "entity_key")]
        if entity:
            aliases.extend(str(a).casefold()
                           for a in (entity.get("aliases") or []))
        linked = [s for s in (bundle["map"].get("states") or [])
                  if str(s.get("entity_id") or "").casefold() in aliases]
        for state in linked:
            key = str(state.get("key") or "")
            age = max(0.0, float(state.get("age_s") or 0.0))
            freshness_limit = 120.0 if key.startswith("location.") else 300.0
            state["freshness"] = "stale" if age > freshness_limit else "current"
            state["is_stale"] = age > freshness_limit
        try:
            seen = datetime.fromisoformat(
                str(device.get("last_seen") or "").replace("Z", "+00:00"))
            online_age = max(0.0, (now_dt - seen).total_seconds())
        except (TypeError, ValueError):
            online_age = None
        gps_state = next((s for s in linked
                          if str(s.get("key") or "").startswith("location.geo")
                          and not s.get("is_stale")
                          and is_device_gps(s.get("value"))), None)
        fresh_scene_location = next((scene for scene in (bundle.get("scenes") or [])
            if str(scene.get("device") or "").casefold() in aliases
            and scene.get("location")
            and isinstance(scene.get("location"), dict)
            and str(scene["location"].get("source") or "").casefold() in {"gps", "device_gps", "gnss"}
            and float(scene["location"].get("age_s") or 0) <= 120), None)
        current_device_facts.append({
            "entity": device.get("entity_key") or device.get("entity"),
            "name": device.get("name"),
            "node_online": bool(device.get("online"))
                and online_age is not None and online_age <= 90,
            "node_online_age_s": round(online_age, 1)
                if online_age is not None else None,
            "status_states": linked,
            "location": (
                {"source": "device GPS state", "age_s": gps_state.get("age_s"),
                 "value": gps_state.get("value")}
                if gps_state else
                {"source": "device GPS", "age_s": fresh_scene_location["location"].get("age_s"),
                 "value": fresh_scene_location["location"]}
                if fresh_scene_location else None
            ),
        })
        if current_device_facts[-1]["location"] is None:
            refs = {str(device.get("entity_key") or "").casefold(),
                    str(device.get("entity") or "").casefold(),
                    f"device:{str(device.get('uuid') or '').casefold()}"}
            placement = next((
                {"source": "user-drawn room placement", "space": space.get("name"),
                 "entity": space.get("entity")}
                for space in spaces
                for placed in (space.get("devices") or [])
                if str(placed.get("ref") or "").casefold() in refs
            ), None)
            if placement is None:
                for layout in room_layouts:
                    doc = layout.get("doc") or {}
                    device_row = next((d for d in (doc.get("devices") or [])
                                       if str(d.get("device_id") or "").casefold()
                                       in aliases), None)
                    if device_row is None:
                        continue
                    room_id = str(device_row.get("room_id") or "")
                    room = next((r for r in (doc.get("rooms") or [])
                                 if str(r.get("room_id") or "") == room_id), None)
                    placement = {
                        "source": "node room plan",
                        "space": (room or {}).get("name") or
                            (doc.get("name") if not room_id else room_id) or
                            "primary room",
                        "device_id": layout.get("device_id"),
                    }
                    break
            current_device_facts[-1]["location"] = placement or {
                "status": "unknown; no trusted device GPS or room placement"}
    out["current_device_facts"] = current_device_facts
    if cfg.include_descriptors:
        out["descriptors"] = (bundle.get("descriptors") or []) \
                             [:32 if advanced else 12]
    if cfg.include_scenes:
        out["scenes"] = (bundle.get("scenes") or []) \
                        [:10 if advanced else 5]
    # Always included — the agent must know which sensor classes are
    # actually live and which tasks they can/cannot answer right now.
    out["sensors"] = bundle.get("sensors") or []
    out["capabilities"] = bundle.get("capabilities") or []
    if cfg.include_reference:
        out["reference"] = bundle.get("reference")
    if calibration:
        out["calibration"] = calibration
    if cfg.include_builder:
        out["builder"] = builder
    if cfg.include_usage:
        out["usage"] = inference_status(db, user) if user else None
    out["provenance"]["sections"] = [k for k in out
                                     if k not in ("provenance", "config")]
    return out


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
physical `fields` (mean/std/min/max/n for radar SNR, CSI variance,
RSSI, IMU, audio level…) plus node `estimates` and `location`, and
`reference`/`calibration` — measured baselines the judgements stand
on (`calibration.radar_occupancy` + `calibration.csi_link`: empty-vs-
occupied stats for the radar/CSI fields — compare the current
window's tail, max spikes and widened std, against them; the median
alone is weak evidence. `calibration.watch_imu_worn`: measured
unworn-watch IMU baseline plus the worn signature to compare `acc_std`
and gravity-axis behaviour against.
`calibration.user_calibrations`: the user's OWN labeled baselines
measured on their nodes — e.g. `sleep`, `empty_room` — per-sensor
descriptor stats (fields/scan) collected while the label held. To
infer an activity or state, compare the CURRENT window's
descriptors/scene fields against the labeled baseline (near → the
label applies; far → it doesn't). A `sleep` baseline whose BLE scan
carries the owner's phone and whose IMU/audio are quiet is the
reference for "the owner is asleep".)
— then the user's question and optional attachments.

Time — `now_iso`/`generated_at` is the CURRENT time. Every temporal
field elsewhere is an absolute epoch/`since` + `age_s` (seconds old at
bundle time). Always translate ages into plain words ("3 minutes ago",
"14 hours ago") and reason about recency: a state with `age_s` of
hours may reflect a situation that has since changed — say how old the
evidence behind an answer is, and treat states whose `age_s` is large
as stale even while `valid_until` keeps them technically live.

Sources — every claim has a SOURCE you must name when it matters:
`map.states[].estimator` is the estimator/model that produced it
(e.g. `context-builder/1`, `ble-presence/1`, `rm-<hex>`, a node
estimator), `devices[]`/`descriptors` attribute evidence to a physical
node, scenes carry `device`, states carry `since`/`age_s`. Never say
"the system indicates" — say which estimator or sensor said it and how
old that observation is.

Current status and location rules — `current_device_facts` is the
conservative summary of registry reachability and rapidly changing
states. `node_online` means only that the node checked in recently; it
does NOT mean that the laptop/person is physically inside the home or a
room. `location.geo.v1` marked `source=public_ip_geolocation` is a
coarse internet-egress estimate, not GPS; devices on the same network
can report identical coordinates. It never proves a device is at home,
inside a building, or in a room. Physical location requires a fresh
device GPS state marked with source `gps`, or an explicit user-drawn
room placement in `spaces`. Do not infer room
presence from online status, ownership, a stale `located_in` edge, or
the fact that a device was seen by its own node. For `watch.link.v1`,
only its fresh `connected` field establishes the link; RSSI alone does
not mean connected. `connected: false` means disconnected. For
`watch.worn.v1`, `worn: true` means worn and `worn: false` means not
worn; low IMU/acceleration variance means little movement, not unworn.
Never override a fresh explicit watch state with RSSI or motion variance.
Read these values literally. If a state is marked
`is_stale: true`, describe it only as the last report and do not use it
as the present state. When current evidence conflicts, show the
timestamps and say that the current status is unknown.

Capability — `sensors[]` lists the sensor CLASSES actually producing
data in this window per device; `capabilities[]` maps tasks →
sensor classes and marks each `available`/`unavailable` (with `missing`
alternatives). Different tasks need different sensors — occupancy can
come from radar/CSI fluctuation and motion. Radio or watch sightings
identify nearby emitters but do not by themselves prove room occupancy;
activity recognition
(sit/stand/sleep/walk) needs IMU-bearing devices (watch/phone/laptop)
plus device/app state; scene perception needs a camera or mic. When a
task is `unavailable` — or the sensor that could answer hasn't
produced evidence in this window — do NOT guess: say plainly that the
available sensors can't answer it and name what WOULD enable it
(e.g. "I can't tell the room is empty — no radar/CSI/motion evidence
flowed recently; the BLE presence state is 14h old"). A state still
answering may itself be stale — combine capability with age.

Identity — WHO and WHICH THINGS belong to the account:
  * `persons`/`places` list the account's people and address→building→
    room hierarchy; `person:owner` is the account owner, `place:home`
    the home address, its `rooms` the drawn layout.
  * Every `devices[]` row (registry nodes/apps/watches) is OWNED by
    `person:owner` unless evidence says otherwise — `entity_key` links
    it to its `device:<uuid>` map entity. Emitters absent from
    `devices` are unknowns.
  * `spaces[]` is the account-space layout; `room_layouts[]` contains
    node-authored `room/v1` geometry, furniture, room assignments and
    manually placed person markers. Use both plans as authored and do
    not treat a marker as proof someone is currently present.
  * `spaces[]` is the USER-DRAWN layout (authoritative): each space
    has `name`, `size_m`, `zones[]` polygons, `devices[]` placements
    ({ref → `device:<uuid>`, x, y, rotation_deg, fov_deg, range_m}).
    A device placed in a space is IN that space; an emitter seen by a
    node inside a space is inside it too. Scan emitters carrying
    `device_ref`+`known` are anchored to the registry — `known`
    emitters near a placed node share that node's space.
  * Assume owned devices sit in the owner's identified room/place
    unless located_in edges or fresher evidence place them elsewhere.
  * `config`/`provenance` tell you which bundle sections exist and the
    window/caps the user configured.

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
 "assessments": [{"claim": "<factual statement you made>",
                  "probability": 0.0,
                  "basis": "<bundle field/evidence supporting it>"}],
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

Assessments — one entry per consequential factual claim in `answer`,
with a calibrated probability (0–1) grounded in the bundle: fresh
direct evidence → high; stale/inferred/absent → low. These make
answers actionable downstream, so always emit them for factual claims.

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


# Strict structured-output contract — guarantees the probabilistic
# fields (`assessments[].probability`) exist and are well-typed so
# answers stay machine-actionable. Falls back to JSON-object mode on
# models that reject schema mode.
_WIDGET_SCALAR: Dict[str, Any] = {
    "type": ["string", "number", "boolean", "null"]}
_WIDGET_KV: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {"label": {"type": "string"},
                   "value": {"type": "string"}},
    "required": ["label", "value"]}
_WIDGET_STATE: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {"key": {"type": "string"}, "entity": {"type": "string"},
                   "value": {"type": "string"},
                   "confidence": {"type": "number"},
                   "confirmed": {"type": "boolean"}},
    "required": ["key", "entity", "value", "confidence", "confirmed"]}
_ANSWER_SCHEMA: Dict[str, Any] = {
    "name": "thoth_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string"},
                        "probability": {"type": "number"},
                        "basis": {"type": "string"},
                    },
                    "required": ["claim", "probability", "basis"],
                },
            },
            "widgets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string"},
                        "title": {"type": "string"},
                        "items": {"type": "array", "items": {
                            "anyOf": [{"type": "string"},
                                      _WIDGET_KV, _WIDGET_STATE]}},
                        "columns": {"type": "array",
                                    "items": {"type": "string"}},
                        "rows": {"type": "array",
                                 "items": {"type": "array",
                                           "items": _WIDGET_SCALAR}},
                    },
                    "required": ["type", "title", "items",
                                 "columns", "rows"],
                },
            },
        },
        "required": ["answer", "assessments", "widgets"],
    },
}


def _openai_answer(messages: List[Dict[str, Any]], *,
                   tier: str = "standard",
                   usage_out: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """One structured completion → parsed rich answer."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = _model_for_tier(tier)
    client = OpenAI(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_schema",
                            "json_schema": _ANSWER_SCHEMA},
    }
    # Reasoning-class models (gpt-5, o-series…) reject `temperature`
    # and/or `response_format` — degrade on BadRequestError instead of
    # failing the user's turn.
    from openai import APIStatusError, BadRequestError, RateLimitError
    try:
        resp = client.chat.completions.create(**kwargs)
    except BadRequestError:
        kwargs.pop("temperature", None)
        try:
            resp = client.chat.completions.create(**kwargs)
        except BadRequestError:
            kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**kwargs)
            except BadRequestError:
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)
        # A BadRequestError chain aside, upstream failures (dead key,
        # exhausted balance, provider outage) must surface as a clean
        # status — never a bare 500 the app can only call "chat failed".
    except RateLimitError as exc:
        raise HTTPException(429, "model quota exhausted — top up the "
                                 "OpenAI balance or retry later") \
            from exc
    except APIStatusError as exc:
        raise HTTPException(502, f"model upstream error "
                                 f"({exc.status_code})") from exc
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
    asmts = parsed.get("assessments")
    parsed["assessments"] = ([a for a in asmts if isinstance(a, dict)]
                             if isinstance(asmts, list) else [])
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
    context_used: Optional[Dict[str, Any]] = None
    if body.include_context:
        try:
            context_used = _chat_context(
                db, current_user.userId, current_user,
                detail=tier, cfg=_load_config(db, current_user.userId))
        except Exception:
            # A broken context leg must not kill the turn — answer
            # context-free and log loud so ops sees it.
            logger.exception("chat context build failed for user %s",
                             current_user.userId)

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
        "assessments": [a for a in (parsed.get("assessments") or [])
                        if isinstance(a, dict)],
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
    """The bundle the assistant would be given — what the model sees,
    plus ``config`` and ``provenance`` describing every section."""
    return _chat_context(db, current_user.userId, current_user,
                         cfg=_load_config(db, current_user.userId))


@router.get("/config")
async def get_chat_config(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """User's assistant context configuration (see ChatConfig)."""
    return {"config": _load_config(db, current_user.userId).model_dump()}


@router.put("/config")
async def put_chat_config(
    body: ChatConfig,
    current_user: User = Depends(get_scoped_principal("chat:write")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Persist the user's assistant context configuration."""
    ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == current_user.userId,
        ContextEntity.entity_key == _CONFIG_ENTITY).first()
    if ent is None:
        ent = ContextEntity(user_id=current_user.userId,
                            entity_key=_CONFIG_ENTITY, kind="system",
                            name="chat config", attributes="{}")
        db.add(ent)
    ent.attributes = body.model_dump_json()
    db.commit()
    return {"config": body.model_dump()}


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
