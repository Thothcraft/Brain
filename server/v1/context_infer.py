"""Brain v1 context inference — OpenAI function-calling estimator.

``POST /v1/context/infer`` feeds per-window physical descriptors plus
per-class calibration statistics to an OpenAI model. The model's only
output channel is the ``submit_context_form`` tool (``tool_choice`` is
forced), so every request deterministically yields one complete,
schema-validated "context layer form": entities, relationships,
evidence, and derived states.

Submission is format-sensitive by construction — the tool arguments
are validated against the same pydantic models (``EntityIn``,
``RelationshipIn``, ``EvidenceIn``, ``StateIn``) and applied through
the same ``apply_*`` helpers the REST endpoints use, so a form that
passes is guaranteed to be a legal context submission.

States in the form link evidence by *form-local* refs (``evidence_refs``
→ entries in the form's ``evidence`` list); the server resolves them to
evidence row ids after insertion, since the model can't know DB ids.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from server.auth import get_current_user, get_scoped_principal
from server.db import ContextEntity, User, get_db
from server.inference_auth import inference_call, inference_status
from .context import (
    EntityIn, EvidenceIn, RelationshipIn, StateIn,
    apply_entity, apply_evidence_items, apply_relationship, apply_state,
    is_confirmed_source,
)
from .context_builder import (
    _FP_ATTR_KEYS, _device_fp_index, _resolve_device,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["v1", "context"])

_DEFAULT_MODEL = "gpt-4o-mini"

# Models the caller may pick explicitly (the "meter" the user sets —
# gpt-4o class or better, all ≥128k context). Env-overridable comma list.
_MODEL_SET_ENV = "CONTEXT_INFER_MODELS"
_DEFAULT_MODEL_SET = (
    "gpt-4o-mini", "gpt-4o",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "o4-mini",
)

# Auditable trail on the ``infer:last`` system entity — what the model
# saw and what it produced, for the UI.
LAST_ENTITY = "infer:last"
INPUT_KEEP = 24000
OUTPUT_KEEP = 12000
SEEN_KEEP = 220000  # response echo budget


# ---------------------------------------------------------------------------
# Request model — calibration stats + physical descriptors of the window
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    """One inference request = one time window.

    ``calibration`` holds per-target-class stats (means/stds/counts/
    thresholds — arbitrary JSON keyed by class). ``descriptors`` is the
    physical description of the window being classified (sensor
    features, RSSI/CSI summaries, occupancy probabilities, motion
    variance — arbitrary JSON). The model must fill the form from
    these alone.
    """
    window: Dict[str, Any] = Field(
        default_factory=dict,
        description="Window bounds/metadata, e.g. "
                    "{start_ts, end_ts, device_id, room_id}")
    calibration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-target-class calibration statistics — the "
                    "reference distributions the window is judged "
                    "against (per-class feature mean/std/count, RSSI "
                    "fingerprint anchor means, radar/CSI descriptor "
                    "ranges)")
    descriptors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical descriptors measured over the window")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Account/device context around the window — "
                    "entity inventory with kinds/names/confirmed "
                    "attributes, device metadata (platform, firmware, "
                    "capabilities), digital context (app, network, "
                    "peripherals, traffic summaries). The model may "
                    "infer device type/state from metadata + traffic "
                    "but lands only proposals, never confirmed facts.")
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prior windows / previously submitted form "
                    "summaries for temporal context (caller bounds "
                    "size; ~20 items is a good budget)")
    coverage: List[str] = Field(
        default_factory=list,
        description="Form sections the caller wants emphasized "
                    "('entities', 'relationships', 'evidence', "
                    "'states'); empty = all relevant")
    thinking: str = Field(
        default="standard",
        description="Reasoning tier → model route: 'quick' (cheap,"
                    " fast), 'standard' (default), 'deep' (harder "
                    "reconciliation; billed the same per call but "
                    "uses the CONTEXT_INFER_MODEL_DEEP model)")
    model: Optional[str] = Field(
        default=None,
        description="User-set explicit model id — overrides the tier "
                    "route. Must be one of the allowed set "
                    "(GET /context/infer/options); 422 otherwise.")
    devices: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Caller-supplied device roster — which device is "
                    "which. Entries carry {entity/uuid/name/type/mac/"
                    "hardware/association}; association 'registered' "
                    "(or 'user') marks account-owned nodes, anything "
                    "else (e.g. 'unknown', 'observed') marks emitters "
                    "only seen in scans.")
    gather_window_s: float = Field(
        default=0.0, ge=0.0, le=86400.0,
        description="When >0, Brain assembles the comprehensive bundle "
                    "server-side from stored evidence: the registered "
                    "device roster, the latest per-node scene uplinks "
                    "(sensor text, cues, fields incl. radar snr_db/"
                    "range/xy stats, radio scans, digital context), "
                    "per-key descriptor aggregates and the current "
                    "context map — the model sees all of it.")
    entity_hint: Optional[str] = Field(
        default=None,
        description="Canonical entity id the window concerns "
                    "(e.g. 'person:gad' or the device uuid)")
    dry_run: bool = Field(
        default=False,
        description="Return the predicted form without writing it")


def _allowed_models() -> List[str]:
    raw = os.getenv(_MODEL_SET_ENV, "")
    items = [m.strip() for m in raw.split(",") if m.strip()]
    return items or list(_DEFAULT_MODEL_SET)


# ---------------------------------------------------------------------------
# The tool — a complete context layer form
# ---------------------------------------------------------------------------

def _ctx_tool_schema() -> Dict[str, Any]:
    """OpenAI tool definition. Field names mirror the /v1/context
    request models exactly (id/kind/name/attributes; subject/predicate/
    object; key/value/timestamp; state key/value/entity_id...) so the
    validated arguments can be applied verbatim."""
    return {
        "type": "function",
        "function": {
            "name": "submit_context_form",
            "description": (
                "Submit the completed context layer form for this window. "
                "This is the ONLY output channel — fill every section that "
                "the evidence supports. The form is validated and applied "
                "to the context store exactly as the /v1/context REST "
                "endpoints would apply it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One sentence: what the window "
                                       "most likely depicts."},
                    "analysis": {
                        "type": "string",
                        "description": "A short plain-language "
                                       "narrative of your reasoning — "
                                       "which sensors drove the call, "
                                       "which devices you identified, "
                                       "what stayed ambiguous. This is "
                                       "the text the UI shows the "
                                       "user."},
                    "entities": {
                        "type": "array",
                        "description": "Canonical objects the window "
                                       "implies (persons, spaces, devices).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string",
                                       "description": "e.g. 'person:gad', "
                                                      "'space:living-room'"},
                                "kind": {"type": "string"},
                                "name": {"type": "string"},
                                "attributes": {"type": "object"},
                            },
                            "required": ["id", "kind"],
                        }},
                    "relationships": {
                        "type": "array",
                        "description": "subject–predicate–object edges "
                                       "between entity ids.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string",
                                              "description": "e.g. "
                                                             "'located_in', "
                                                             "'carries', "
                                                             "'observed_by'"},
                                "object": {"type": "string"},
                                "confidence": {"type": "number",
                                               "minimum": 0, "maximum": 1},
                                "valid_from": {"type": "number"},
                                "valid_until": {"type": "number"},
                            },
                            "required": ["subject", "predicate", "object"],
                        }},
                    "evidence": {
                        "type": "array",
                        "description": "Predictions/observations behind "
                                       "this form — provenance, never "
                                       "truth.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref": {"type": "string",
                                        "description": "Form-local id so "
                                                       "states can cite "
                                                       "it, e.g. 'ev1'"},
                                "key": {"type": "string",
                                        "description": "Versioned key, "
                                                       "e.g. "
                                                       "'occupancy.prob.v1'"},
                                "value": {},
                                "timestamp": {"type": "number"},
                                "confidence": {"type": "number",
                                               "minimum": 0, "maximum": 1},
                                "model_id": {"type": "string"},
                                "model_version": {"type": "string"},
                                "source_id": {"type": "string"},
                                "device_id": {"type": "string"},
                                "provenance": {"type": "object"},
                            },
                            "required": ["ref", "key"],
                        }},
                    "states": {
                        "type": "array",
                        "description": "Derived context states — the "
                                       "estimator's final answer per "
                                       "key.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string",
                                        "description": "e.g. "
                                                       "'occupancy.v1', "
                                                       "'activity.v1'"},
                                "value": {},
                                "entity_id": {"type": "string"},
                                "confidence": {"type": "number",
                                               "minimum": 0, "maximum": 1},
                                "since": {"type": "number"},
                                "valid_until": {"type": "number"},
                                "transition": {"type": "string",
                                               "enum": ["entered", "exited",
                                                        "changed"]},
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "refs of this form's "
                                                   "evidence items"},
                            },
                            "required": ["key"],
                        }},
                    "uncertainties": {
                        "type": "array",
                        "description": "Claims you considered but could "
                                       "NOT support — curiosity, not "
                                       "silence. The platform reviews "
                                       "these instead of guessing.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string",
                                        "description": "the state/claim "
                                                       "key left "
                                                       "unresolved"},
                                "reason": {"type": "string"},
                            },
                            "required": ["key", "reason"],
                        }},
                    "questions": {
                        "type": "array",
                        "description": "Specific observations you'd "
                                       "request next to resolve the "
                                       "uncertainties (what to "
                                       "measure, on which device, "
                                       "for how long).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "target_key": {"type": "string"},
                            },
                            "required": ["question"],
                        }},
                    "notes": {
                        "type": "array",
                        "description": "Curated memory: durable "
                                       "observations about entities/"
                                       "relationships worth keeping "
                                       "for future windows (never "
                                       "confirmed facts).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["subject", "text"],
                        }},
                    "devices": {
                        "type": "array",
                        "description": "The device inventory as YOU see "
                                       "it — every registered node and "
                                       "every persistent unknown "
                                       "emitter, ONE entry per physical "
                                       "device (fingerprints dedup "
                                       "server-side; never emit two "
                                       "entries for the same MAC/uuid/"
                                       "hostname). Reuse the canonical "
                                       "ref (device:<uuid> when "
                                       "registered) and carry the "
                                       "richest name the metadata "
                                       "supports.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "device": {
                                    "type": "string",
                                    "description": "canonical ref — "
                                                   "device:<uuid>, uuid, "
                                                   "MAC or beacon id"},
                                "name": {
                                    "type": "string",
                                    "description": "rich human name "
                                                   "from MAC vendor + "
                                                   "advertised name + "
                                                   "hostname + observed "
                                                   "behavior, e.g. "
                                                   "'living-room radar "
                                                   "rig (thoth-chen, "
                                                   "d8:3a:dd)'"},
                                "kind": {
                                    "type": "string",
                                    "description": "device | object | "
                                                   "person — what the "
                                                   "emitter most likely "
                                                   "is"},
                                "role": {
                                    "type": "string",
                                    "description": "inferred function: "
                                                   "radar-node | phone | "
                                                   "watch | laptop | tv | "
                                                   "beacon | unknown-"
                                                   "emitter"},
                                "mac": {"type": "string"},
                                "vendor": {"type": "string"},
                                "hostname": {"type": "string"},
                                "model": {"type": "string"},
                                "attributes": {
                                    "type": "object",
                                    "description": "extra inferred "
                                                   "facts (os, transport, "
                                                   "seen_near, sensors)"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0, "maximum": 1},
                                "rationale": {"type": "string"},
                            },
                            "required": ["device", "name"],
                        }},
                    "persons": {
                        "type": "array",
                        "description": "Who is where — always include "
                                       "person:owner (the user). "
                                       "associated_devices are the "
                                       "carried devices (watch/phone/"
                                       "laptop); location resolves "
                                       "geo evidence against the "
                                       "place addresses, then radar/"
                                       "BLE/CSI narrows it to a room.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string",
                                       "description": "person:<slug>"},
                                "name": {"type": "string"},
                                "is_user": {"type": "boolean"},
                                "address": {
                                    "type": "string",
                                    "description": "the person's home "
                                                   "address"},
                                "associated_devices": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "entity refs of "
                                                   "carried devices "
                                                   "(watch/phone/laptop)"},
                                "location": {
                                    "type": "object",
                                    "description": "{place, room — "
                                                   "entity ids; "
                                                   "address; "
                                                   "granularity: "
                                                   "indoor_room | "
                                                   "building | address | "
                                                   "street | unknown; "
                                                   "on_street bool}"},
                                "activity": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0, "maximum": 1},
                                "rationale": {"type": "string"},
                            },
                            "required": ["id"],
                        }},
                    "places": {
                        "type": "array",
                        "description": "Buildings and the indoor spaces "
                                       "inside them. A place with an "
                                       "`address` is a building; its "
                                       "`rooms` carry layout + manual "
                                       "device placement/orientation "
                                       "(user-set entries are "
                                       "authoritative — echo them "
                                       "verbatim).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string",
                                       "description": "place:<slug>"},
                                "name": {"type": "string"},
                                "address": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "description": "building | room | "
                                                   "outdoor"},
                                "parent": {
                                    "type": "string",
                                    "description": "building place id "
                                                   "for rooms"},
                                "rooms": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"},
                                            "layout": {"type": "object"},
                                            "devices": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "ref": {
                                                            "type": "string"},
                                                        "placement": {
                                                            "type": "object"},
                                                        "orientation": {
                                                            "type": "object"},
                                                        "manual": {
                                                            "type": "boolean"},
                                                    },
                                                    "required": ["ref"],
                                                }},
                                        },
                                    }},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0, "maximum": 1},
                                "rationale": {"type": "string"},
                            },
                            "required": ["id"],
                        }},
                    "device_updates": {
                        "type": "array",
                        "description": "Device identification / "
                                       "naming proposals — your room "
                                       "to update device names from "
                                       "MAC vendors, advertised names, "
                                       "hostnames or other metadata. "
                                       "Reference a registered device "
                                       "by its entity id (device:<uuid>) "
                                       "or uuid; reference an unknown "
                                       "emitter by its MAC/beacon id.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "device": {
                                    "type": "string",
                                    "description": "device:<uuid>, raw "
                                                   "uuid, MAC or beacon "
                                                   "id being named"},
                                "proposed_name": {
                                    "type": "string",
                                    "description": "human-readable name "
                                                   "the evidence "
                                                   "supports"},
                                "proposed_kind": {
                                    "type": "string",
                                    "description": "device | object | "
                                                   "person — what the "
                                                   "emitter most likely "
                                                   "is (phone, watch, "
                                                   "TV, beacon...)"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0, "maximum": 1},
                                "rationale": {
                                    "type": "string",
                                    "description": "why — the MAC "
                                                   "vendor, advertised "
                                                   "name, hostname, "
                                                   "behavioral pattern"},
                            },
                            "required": ["device", "proposed_name"],
                        }},
                },
                "required": ["summary", "states"],
            },
        },
    }


_SYSTEM_PROMPT = """\
You are the CONTEXT ESTIMATOR of a Thoth deployment — the model that
turns one observation window into a structured context-layer form.

THE SYSTEM. Thoth is a sensor-fusion platform. Edge "nodes" — the
user's registered devices (Raspberry Pis, laptops, phones) — run sensor
adapters: mmWave radar (BGT60TR13C: snr_db, range_profile, xy_map),
Wi-Fi CSI amplitude, BLE/Wi-Fi/Zigbee radio scans (MAC, RSSI, advertised
names), cameras (face/person detectors), microphones (level, speech),
IMU and host telemetry. On-device models emit predictions; nodes uplink
physical descriptors (per-field mean/min/max stats), predictions and
digital context (foreground app, network state, battery) to Brain.
Brain keeps a context layer — entities (persons, spaces, devices),
relationships, evidence, derived states — that agents and automations
read. Your role: fuse this window's evidence into occupancy, presence,
activity, location and device-identity answers, then submit them via
submit_context_form — your ONLY output channel. You never see raw
sensor frames, only descriptors and stats.

For each request you receive some of:
  * `window`/`entity_hint` — the window's bounds and the entity it
    concerns.
  * `calibration` — REFERENCE distributions learned during calibration
    (per-class feature mean/std/count; RSSI fingerprints {anchor:
    {mean,std,count}}). Judge a window by distance to these — a
    descriptor several std from every class centroid is low-confidence
    or unknown, never a forced label.
  * `descriptors` — physical descriptors of THIS window (radar stats,
    CSI variance, RSSI, audio level, per-sensor fields).
  * `devices` — the user's REGISTERED nodes: entity (device:<uuid>),
    name, type, mac, hardware, online, association="registered". These
    tell you which physical machine each `device` id/uuid in the data
    is. Emitters appearing in radio scans that match NO registered
    device and NO existing map entity are UNKNOWN devices — candidates
    for device_updates, not silently-owned hardware.
  * `scenes` — the latest uplink per node: a one-line scene summary,
    per-sensor `text` + structured `cues` (speech, people count,
    recognized face, motion, strongest emitter) + physical `fields`
    (per-field mean/min/max/n — e.g. radar snr_db/range/xy, CSI
    amplitude variance, RSSI) with n/rate_hz/age_s, on-device
    `predictions`, `estimates`, `location` and `room`.
  * `observations` — aggregated evidence stats per (key, device);
    `map` — the current semantic map (entities carry `confirmed`:
    user-confirmed facts you must not contradict); `context` — caller-
    supplied account/device/digital context; `history` — prior windows.
  * `reference` — built-in dataset prior: measured empty-vs-occupied
    stats for the radar/CSI fields (E1 rig, BGT60TR13C). When the
    caller left `calibration` empty it is seeded FROM this reference —
    treat it as the default class distributions.
  * `coverage` — if non-empty, fill only the named form sections.

READING OCCUPANCY. The `reference` prior (E1 rig, BGT60TR13C @ ~10 Hz,
5 s windows, ~2150 windows per class): an EMPTY room reads snr mean
~7.6 dB and stays TIGHT (p95 <8, max 8.3; window snr_max ~10 dB, never
>~13); OCCUPANCY lives in the tail — window snr_max spikes >15-40 dB
(occupied p95 ~40.6), snr mean p95 ~22 dB, and the ra/re/xy map means
lift ~+0.1-0.2 log-power while *std90 spreads widen ~2x. A modest
median snr alone is weak evidence (AUC ~0.7 — medians barely move);
call occupied on peak spikes + map-mean lifts vs the node's own empty
baseline, not on a fixed threshold. Flat fields while the reporting
sensors are fresh (small age_s) is positive evidence the space is
EMPTY; a stale or missing sensor contributes nothing, not a vote for
empty. Weight camera face/person detections, BLE/RSSI of carried
devices, and audio activity as corroborating signals.

IDENTITY LAW — one entity per physical device. A MAC, hostname, uuid or
beacon id is a FINGERPRINT: an emitter matching any fingerprint of an
existing entity IS that entity — reference it by its canonical
device:<uuid> (or existing map id), never mint a second id. Emitters in
scans that match nothing get ONE entry keyed by their best stable id
(prefer the MAC). Every device entry carries the richest name the
metadata supports — compose it from MAC vendor (OUI), advertised name,
hostname, model and observed role/traffic ("thoth-chen — Pi5 radar rig",
"Sarah's iPhone 15 (Apple, -58 dBm, carried)") — plus a rationale
naming the evidence.

PLACES & PEOPLE — the account owns at least `person:owner` (the user)
and `place:home` (the user's address). A place with an `address` is a
building; rooms inside it are places with `parent` set to the building
id and carry `layout` plus `devices` with {ref, placement, orientation,
manual} — user-set placement/orientation is authoritative, echo it
verbatim. Resolve WHERE things are top-down: current geo evidence
(GPS/`location.geo.v1`) versus each place's `address` decides which
building (or "outside — on-street/different address"); inside the
building, radar/CSI occupancy + BLE RSSI of carried devices narrows to
a `room`. persons[].location.granularity says how far you resolved:
indoor_room | building | address | street | unknown.

Output rules for submit_context_form:
  * `analysis` — a few sentences of plain-language reasoning for the
    UI: which sensors drove the call, which devices you identified,
    what stayed ambiguous.
  * states[] — one entry per target key (occupancy.v1, activity.v1,
    location.v1, presence.v1); value = predicted label/object;
    confidence calibrated by distance to class statistics and sensor
    freshness, not raw probability.
  * evidence[] — what you based each prediction on (model
    probabilities, descriptors, calibration references).
  * entities[]/relationships[] — WHO/WHERE/WHAT the window implies;
    create when confident, omit when unsupported.
  * devices[] — the deduplicated inventory: every registered node and
    persistent unknown emitter, fingerprint-keyed, richly named.
  * persons[] — who is where (person:owner always present), with
    associated carried devices and resolved location.
  * places[] — buildings (addressed) and rooms (layout + manual
    device placement preserved).
  * device_updates[] — legacy naming/identification proposals; prefer
    devices[] for new submissions.
  * uncertainties[]/questions[]/notes[] — what you considered but
    couldn't support, the observations that would resolve it, and
    durable facts for future windows. Silent ambiguity is worse than
    an explicit question.
  * All numbers are floats (epoch seconds); no prose inside values.
"""


# ---------------------------------------------------------------------------
# LLM call + form application
# ---------------------------------------------------------------------------

_THINKING_TIERS = ("quick", "standard", "deep")


def _model_for_tier(tier: str, explicit: Optional[str] = None) -> str:
    """Resolve a thinking tier — or the user's explicit pick — to a
    concrete model id.

    An explicit ``model`` (the meter the user sets) wins over the tier
    route but must belong to the allowed set — every entry is a
    large-context (≥128k) gpt-4o-class model or better. ``standard``
    keeps the historical env knobs (``CONTEXT_INFER_MODEL`` →
    ``MODEL_NAME`` → default); ``quick``/``deep`` have their own
    overrides so operators can price the meter. Unknown tiers and
    disallowed models are client errors."""
    if explicit:
        allowed = _allowed_models()
        if explicit not in allowed:
            raise HTTPException(
                422, f"model {explicit!r} is not in the allowed set "
                     f"{allowed} (env {_MODEL_SET_ENV})")
        return explicit
    tier = (tier or "standard").lower()
    if tier == "standard":
        return (os.getenv("CONTEXT_INFER_MODEL")
                or os.getenv("MODEL_NAME") or _DEFAULT_MODEL)
    if tier == "quick":
        return (os.getenv("CONTEXT_INFER_MODEL_QUICK")
                or _DEFAULT_MODEL)
    if tier == "deep":
        return (os.getenv("CONTEXT_INFER_MODEL_DEEP") or "gpt-4o")
    raise HTTPException(
        422, f"unknown thinking tier {tier!r} "
             f"— one of {list(_THINKING_TIERS)}")


def _assemble_payload(request: InferRequest, db: Session,
                      user_id: int) -> Dict[str, Any]:
    """Build the exact user-message payload the model sees.

    Caller-supplied sections are always honored. When
    ``gather_window_s`` > 0 the comprehensive bundle (registered device
    roster, latest per-node scenes with radar/CSI/radio stats, per-key
    evidence aggregates, the context map) is assembled server-side from
    stored evidence — the same bundle the context builder reads."""
    payload: Dict[str, Any] = {
        "window": request.window,
        "entity_hint": request.entity_hint,
        "calibration": request.calibration,
        "descriptors": request.descriptors,
        "context": request.context,
        "history": request.history[:20],
        "coverage": request.coverage,
    }
    if request.devices:
        payload["devices"] = request.devices
    if request.gather_window_s > 0:
        from .context_builder import build_bundle, seed_devices
        now = time.time()
        # Same anchors the builder maintains: registered device entities
        # (with MAC/hostname fingerprints) plus person:owner/place:home —
        # the map the model reasons against is seeded, not empty.
        seed_devices(db, user_id, now)
        bundle = build_bundle(db, user_id, now, request.gather_window_s)
        devices = []
        for dev in bundle.get("devices") or []:
            d = dict(dev)
            d.setdefault("association", "registered")
            devices.append(d)
        payload["devices"] = (request.devices or []) + devices
        payload["scenes"] = bundle.get("scenes") or []
        payload["observations"] = bundle.get("descriptors") or []
        payload["map"] = bundle.get("map") or {}
        payload["gathered_window_s"] = bundle.get("window_s")
        payload["evidence_rows"] = bundle.get("evidence_rows")
        reference = bundle.get("reference") or {}
        if reference:
            payload["reference"] = reference
            # The dataset prior is the default calibration — a caller-
            # supplied calibration always wins.
            if not payload["calibration"]:
                payload["calibration"] = reference
    return payload


def _bounded_seen(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Response echo of what the model saw — the full payload unless it
    exceeds the response budget, then per-section sizes so the UI can
    still show what was included."""
    try:
        size = len(json.dumps(payload, default=str))
    except (TypeError, ValueError):
        return {"_error": "payload not serializable"}
    if size <= SEEN_KEEP:
        return payload
    return {"_truncated": True, "bytes": size,
            "sections": {k: len(json.dumps(v, default=str))
                         for k, v in payload.items()}}


def _openai_form(request: InferRequest, *,
                 payload: Optional[Dict[str, Any]] = None,
                 usage_out: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Force the model to fill the form — returns ``{"form": args,
    "model_text": <any stray assistant text>}``.

    ``payload`` is the assembled user message (from
    ``_assemble_payload``); when omitted a minimal payload is built
    from the request fields. ``usage_out``, when supplied, is filled
    with metering fields (``model_id``, ``tokens``, ``tier``) for the
    inference ledger."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = _model_for_tier(request.thinking, request.model)

    if payload is None:
        payload = {
            "window": request.window,
            "entity_hint": request.entity_hint,
            "calibration": request.calibration,
            "descriptors": request.descriptors,
            "context": request.context,
            "history": request.history[:20],
            "coverage": request.coverage,
            "devices": request.devices,
        }
    client = OpenAI(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
        "tools": [_ctx_tool_schema()],
        "tool_choice": {"type": "function",
                        "function": {"name": "submit_context_form"}},
        "temperature": 0.2,
        # room for analysis + device_updates + big forms
        "max_tokens": 8192,
    }
    if re.match(r"^o\d", model):
        # reasoning family: no custom temperature, different token knob
        kwargs.pop("temperature")
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    resp = client.chat.completions.create(**kwargs)
    if usage_out is not None:
        usage_out["model_id"] = model
        usage_out["tier"] = request.thinking
        usage_out["tokens"] = getattr(
            getattr(resp, "usage", None), "total_tokens", None)
    msg = resp.choices[0].message
    model_text = msg.content or None
    calls = msg.tool_calls or []
    for call in calls:
        if call.function.name == "submit_context_form":
            try:
                form = json.loads(call.function.arguments or "{}")
                return {"form": form, "model_text": model_text}
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    502, f"model returned malformed form JSON: {exc}")
    raise HTTPException(
        502, "model did not call submit_context_form "
             f"(finish_reason={resp.choices[0].finish_reason})")


def _merge_rooms(existing: Any, proposed: List[Any],
                 fp_idx: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fold proposed place.rooms[] into the stored set, keyed by room id
    or name. Device refs resolve through fingerprints (a MAC/beacon ref
    lands on the canonical device:<uuid>); user-set ``manual`` placement/
    orientation entries are authoritative and survive any proposal."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in existing or []:
        if isinstance(r, dict):
            key = str(r.get("id") or r.get("name") or "").lower()
            if key:
                out[key] = dict(r)
    for r in proposed or []:
        if not isinstance(r, dict):
            continue
        key = str(r.get("id") or r.get("name") or "").lower()
        if not key:
            continue
        prev = out.get(key, {})
        devs: Dict[str, Dict[str, Any]] = {}
        for d in prev.get("devices") or []:
            if isinstance(d, dict):
                ref = str(d.get("ref") or d.get("device") or "")
                canon = _resolve_device(fp_idx, ref) or ref
                if canon:
                    devs[canon] = {**d, "ref": canon}
        for d in r.get("devices") or []:
            if isinstance(d, dict):
                ref = str(d.get("ref") or d.get("device") or "")
                canon = _resolve_device(fp_idx, ref) or ref
                if not canon:
                    continue
                base = devs.get(canon, {})
                base.update({k: v for k, v in d.items()
                             if v is not None})
                base["ref"] = canon
                devs[canon] = base
        merged = {**prev,
                  **{k: v for k, v in r.items() if v is not None}}
        if devs:
            merged["devices"] = list(devs.values())
        out[key] = merged
    return list(out.values())


def _load_attrs(db: Session, user_id: int, key: str) -> Dict[str, Any]:
    row = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == key).first()
    try:
        a = json.loads(row.attributes) if row and row.attributes else {}
    except (TypeError, ValueError):
        a = {}
    return a if isinstance(a, dict) else {}


def _apply_form(db: Session, user_id: int,
                form: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + persist a predicted form via the shared apply_*
    helpers. Raises HTTPException(422) on the first format violation —
    the submission is all-or-nothing per section order:
    entities → relationships → evidence → states."""
    receipt: Dict[str, Any] = {"entities": [], "relationships": [],
                               "evidence": [], "states": [],
                               "events": []}
    errors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    def _conflict(exc: HTTPException, section: str, index: int) -> bool:
        """409 confirmed_fact → ``skipped`` (precedence, not malformed)."""
        if exc.status_code == 409:
            skipped.append({"section": section, "index": index,
                            "reason": "confirmed_fact",
                            "detail": exc.detail})
            return True
        return False

    # 1. entities ---------------------------------------------------------
    seen_entities = set()
    for i, raw in enumerate(form.get("entities") or []):
        raw = dict(raw)
        attrs = raw.get("attributes")
        if isinstance(attrs, dict) and "_confirmed" in attrs:
            attrs = dict(attrs)
            del attrs["_confirmed"]   # reserved — inference can't confirm
            raw["attributes"] = attrs
            errors.append({"section": "entities", "index": i,
                           "error": "reserved attribute '_confirmed' "
                                    "stripped"})
        try:
            ent = apply_entity(db, user_id, EntityIn(**raw))
            seen_entities.add(ent.entity_key)
            receipt["entities"].append(ent.entity_key)
        except HTTPException as exc:
            if not _conflict(exc, "entities", i):
                errors.append({"section": "entities", "index": i,
                               "error": exc.detail})
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "entities", "index": i,
                           "error": str(exc)[:300]})

    # 2. relationships ----------------------------------------------------
    for i, raw in enumerate(form.get("relationships") or []):
        raw = dict(raw)
        if is_confirmed_source(str(raw.get("source") or "")):
            raw["source"] = ""            # reserved prefix — strip
            errors.append({"section": "relationships", "index": i,
                           "error": "reserved confirmed source stripped"})
        prov = raw.get("provenance")
        if isinstance(prov, dict) and "confirmed" in prov:
            prov = dict(prov)
            del prov["confirmed"]
            raw["provenance"] = prov
            errors.append({"section": "relationships", "index": i,
                           "error": "reserved provenance.confirmed "
                                    "stripped"})
        try:
            rel = apply_relationship(
                db, user_id, RelationshipIn(**raw))
            receipt["relationships"].append(rel.id)
        except HTTPException as exc:
            if not _conflict(exc, "relationships", i):
                errors.append({"section": "relationships", "index": i,
                               "error": exc.detail})
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "relationships", "index": i,
                           "error": str(exc)[:300]})

    # 3. evidence ----------------------------------------------------------
    ref_to_id: Dict[str, int] = {}
    ev_items: List[Dict[str, Any]] = []
    for i, raw in enumerate(form.get("evidence") or []):
        raw = dict(raw)
        ref = str(raw.pop("ref", f"ev{i}"))
        try:
            ev = EvidenceIn(**raw)
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "evidence", "index": i,
                           "error": str(exc)[:300]})
            continue
        ev_items.append((ref, ev))
    if ev_items:
        rows = apply_evidence_items(
            db, user_id, [e.model_dump(exclude_none=True)
                          for _, e in ev_items])
        for (ref, _), row in zip(ev_items, rows):
            ref_to_id[ref] = row.id
            receipt["evidence"].append({"ref": ref, "id": row.id})

    # 4. states ------------------------------------------------------------
    for i, raw in enumerate(form.get("states") or []):
        raw = dict(raw)
        refs = raw.pop("evidence_refs", []) or []
        raw["evidence_ids"] = [str(ref_to_id[r]) for r in refs
                               if r in ref_to_id]
        if is_confirmed_source(str(raw.get("estimator") or "")):
            raw["estimator"] = ""         # LLM can't self-assert confirmed
            errors.append({"section": "states", "index": i,
                           "error": "reserved confirmed estimator "
                                    "stripped"})
        if not raw.get("estimator"):
            raw["estimator"] = "openai-context-form/1"
        try:
            st = apply_state(db, user_id, StateIn(**raw))
            receipt["states"].append(st.to_dict())
        except HTTPException as exc:
            if not _conflict(exc, "states", i):
                errors.append({"section": "states", "index": i,
                               "error": exc.detail})
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "states", "index": i,
                           "error": str(exc)[:300]})

    # 5. curation — uncertainties / questions / notes land as evidence
    # rows so the memory is inspectable and future windows can pick
    # them up; none of them assert truth.
    curation_keys = {"uncertainties": "context.uncertainty.v1",
                     "questions": "context.question.v1",
                     "notes": "context.note.v1"}
    for section, key in curation_keys.items():
        items = form.get(section) or []
        if not items:
            continue
        evs = []
        for raw in items:
            raw = dict(raw) if isinstance(raw, dict) else {"text": str(raw)}
            evs.append({"key": key, "value": raw,
                        "timestamp": time.time(),
                        "source_id": "context_infer",
                        "provenance": {"kind": section[:-1],
                                       "from": "openai-context-form"}})
        try:
            rows = apply_evidence_items(db, user_id, evs)
            receipt["curation"] = receipt.get("curation", [])
            receipt["curation"] += [{"kind": section, "id": r.id}
                                    for r in rows]
        except HTTPException as exc:
            errors.append({"section": section, "index": 0,
                           "error": exc.detail})

    fp_idx = _device_fp_index(db, user_id)

    # 6. places — buildings (address) and the rooms inside them. Room
    # data merges into the building entity's ``rooms`` attr AND lands as
    # its own place entity (part_of the building) so located_in edges
    # can point at rooms directly. User-set manual placement/orientation
    # on room devices is preserved by _merge_rooms.
    for i, raw in enumerate(form.get("places") or []):
        raw = dict(raw) if isinstance(raw, dict) else {}
        pid = str(raw.get("id") or "").strip()
        if not pid:
            errors.append({"section": "places", "index": i,
                           "error": "id required"})
            continue
        attrs = _load_attrs(db, user_id, pid)
        for k in ("address", "kind", "parent", "building", "layout"):
            if raw.get(k) is not None:
                attrs[k] = raw[k]
        rooms = [r for r in (raw.get("rooms") or [])
                 if isinstance(r, dict)]
        if rooms:
            attrs["rooms"] = _merge_rooms(attrs.get("rooms"), rooms,
                                          fp_idx)
        if raw.get("confidence") is not None:
            attrs["infer_confidence"] = raw["confidence"]
        if raw.get("rationale"):
            attrs["infer_rationale"] = raw["rationale"]
        try:
            ent = apply_entity(db, user_id, EntityIn(
                id=pid, kind="place",
                name=str(raw.get("name")) if raw.get("name") else None,
                attributes=attrs))
            receipt.setdefault("places", []).append(ent.entity_key)
        except HTTPException as exc:
            if not _conflict(exc, "places", i):
                errors.append({"section": "places", "index": i,
                               "error": exc.detail})
            continue
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "places", "index": i,
                           "error": str(exc)[:300]})
            continue
        for room in rooms:
            rid = str(room.get("id") or "").strip()
            rname = str(room.get("name") or "").strip()
            if not rid and rname:
                rid = f"{pid}-" + re.sub(
                    r"[^a-z0-9]+", "-", rname.lower()).strip("-")
            if not rid or rid == pid:
                continue
            rattrs = _load_attrs(db, user_id, rid)
            rattrs["building"] = pid
            if room.get("layout") is not None:
                rattrs["layout"] = room["layout"]
            rdevs = []
            for d in room.get("devices") or []:
                if isinstance(d, dict):
                    ref = str(d.get("ref") or d.get("device") or "")
                    canon = _resolve_device(fp_idx, ref) or ref
                    if canon:
                        rdevs.append({**d, "ref": canon})
            if rdevs:
                rattrs["devices"] = rdevs
            try:
                apply_entity(db, user_id, EntityIn(
                    id=rid, kind="place",
                    name=rname or None, attributes=rattrs))
                rel = apply_relationship(db, user_id, RelationshipIn(
                    subject=rid, predicate="part_of", object=pid,
                    confidence=float(raw.get("confidence") or 0.6),
                    source="openai-context-form"))
                receipt["relationships"].append(rel.id)
            except HTTPException as exc:
                if not _conflict(exc, "places", i):
                    errors.append({"section": "places", "index": i,
                                   "error": exc.detail})
            except (ValidationError, TypeError) as exc:
                errors.append({"section": "places", "index": i,
                               "error": str(exc)[:300]})

    # 7. persons — WHO is where. associated_devices resolve through
    # fingerprints (watch/phone/laptop land on canonical ids); the
    # resolved location becomes a located_in edge + location.v1 state.
    for i, raw in enumerate(form.get("persons") or []):
        raw = dict(raw) if isinstance(raw, dict) else {}
        pid = str(raw.get("id") or "").strip()
        if not pid:
            errors.append({"section": "persons", "index": i,
                           "error": "id required"})
            continue
        attrs = _load_attrs(db, user_id, pid)
        if raw.get("address"):
            attrs["address"] = raw["address"]
        if raw.get("is_user") is not None:
            attrs["is_user"] = bool(raw["is_user"])
        assoc = raw.get("associated_devices")
        if isinstance(assoc, list):
            attrs["associated_devices"] = [
                _resolve_device(fp_idx, d) or str(d)
                for d in assoc if d]
        if raw.get("activity"):
            attrs["activity"] = raw["activity"]
        if raw.get("confidence") is not None:
            attrs["infer_confidence"] = raw["confidence"]
        if raw.get("rationale"):
            attrs["infer_rationale"] = raw["rationale"]
        try:
            ent = apply_entity(db, user_id, EntityIn(
                id=pid, kind="person",
                name=str(raw.get("name")) if raw.get("name") else None,
                attributes=attrs))
            receipt.setdefault("persons", []).append(ent.entity_key)
        except HTTPException as exc:
            if not _conflict(exc, "persons", i):
                errors.append({"section": "persons", "index": i,
                               "error": exc.detail})
            continue
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "persons", "index": i,
                           "error": str(exc)[:300]})
            continue
        loc = raw.get("location")
        if isinstance(loc, dict):
            loc = dict(loc)
        elif isinstance(loc, str) and loc.strip():
            loc = {"place": loc.strip()}
        else:
            loc = {}
        target = loc.get("room") or loc.get("place")
        conf = float(raw.get("confidence") or 0.6)
        if target and str(target) != pid:
            try:
                rel = apply_relationship(db, user_id, RelationshipIn(
                    subject=pid, predicate="located_in",
                    object=str(target), confidence=conf,
                    source="openai-context-form"))
                receipt["relationships"].append(rel.id)
            except HTTPException as exc:
                if not _conflict(exc, "persons", i):
                    errors.append({"section": "persons", "index": i,
                                   "error": exc.detail})
            except (ValidationError, TypeError) as exc:
                errors.append({"section": "persons", "index": i,
                               "error": str(exc)[:300]})
        if loc:
            try:
                st = apply_state(db, user_id, StateIn(
                    key="location.v1", entity_id=pid, value=loc,
                    confidence=conf,
                    estimator="openai-context-form/1"))
                receipt["states"].append(st.to_dict())
            except HTTPException as exc:
                if not _conflict(exc, "persons", i):
                    errors.append({"section": "persons", "index": i,
                                   "error": exc.detail})
            except (ValidationError, TypeError) as exc:
                errors.append({"section": "persons", "index": i,
                               "error": str(exc)[:300]})

    # 8. devices + legacy device_updates — the model's room to name
    # things. Each entry lands as an entity upsert on the CANONICAL key:
    # fingerprints (uuid/mac/hostname/name) resolve refs onto existing
    # entities BEFORE any slug is minted, so the same hardware can't be
    # registered twice under different spellings. Registry devices keep
    # their device:<uuid> anchor; unknown emitters get device:<slug>
    # carrying the MAC in attributes + aliases. Confirmed entities are
    # protected by apply_entity → skipped.
    dev_items = ([(i, raw, "devices")
                  for i, raw in enumerate(form.get("devices") or [])]
                 + [(i, raw, "device_updates")
                    for i, raw in enumerate(
                        form.get("device_updates") or [])])
    for i, raw, section in dev_items:
        raw = dict(raw) if isinstance(raw, dict) else {"device": str(raw)}
        ref = str(raw.get("device") or raw.get("id")
                  or raw.get("mac") or "").strip()
        name = str(raw.get("name") or raw.get("proposed_name")
                   or "").strip()
        if not ref or not name:
            errors.append({"section": section, "index": i,
                           "error": "device + name required"})
            continue
        extra_attrs = raw.get("attributes")
        fp_refs = [ref, raw.get("mac"), raw.get("hostname")]
        if isinstance(extra_attrs, dict):
            fp_refs += [extra_attrs.get(k) for k in _FP_ATTR_KEYS]
        fp_hit = _resolve_device(fp_idx, *fp_refs)
        if fp_hit:
            key = fp_hit
        else:
            key = ref if ref.startswith("device:") else \
                "device:" + re.sub(r"[^A-Za-z0-9_.-]+", "-", ref)\
                                 .strip("-").lower()
        attrs = _load_attrs(db, user_id, key)
        aliases = set(attrs.get("aliases") or [])
        if ref != key:
            aliases.add(ref)
        for k in ("mac", "hostname", "model"):
            if isinstance(raw.get(k), str) and raw[k]:
                aliases.add(raw[k])
        extra_attrs = raw.get("attributes")
        if isinstance(extra_attrs, dict):
            for k, v in extra_attrs.items():
                if v is not None:
                    attrs[k] = v
        attrs.update({
            "aliases": sorted(a for a in aliases if a and a != key),
            "proposed_name": name,
            "identified_by": "context_infer",
            "identified_at": time.time(),
        })
        for k in ("rationale", "mac", "vendor", "hostname", "model",
                  "role", "registered"):
            if raw.get(k) is not None:
                attrs[k] = raw[k]
        if raw.get("confidence") is not None:
            attrs["identification_confidence"] = raw["confidence"]
        kind = str(raw.get("kind") or raw.get("proposed_kind")
                   or "device")
        try:
            ent = apply_entity(db, user_id, EntityIn(
                id=key, kind=kind, name=name, attributes=attrs))
            receipt.setdefault("device_updates", []).append({
                "entity": ent.entity_key, "name": name,
                "ref": ref, "resolved": fp_hit})
        except HTTPException as exc:
            if not _conflict(exc, section, i):
                errors.append({"section": section, "index": i,
                               "error": exc.detail})
        except (ValidationError, TypeError) as exc:
            errors.append({"section": section, "index": i,
                           "error": str(exc)[:300]})

    if skipped:
        receipt["skipped"] = skipped
    if errors:
        receipt["errors"] = errors
    return receipt


def _persist_last(db: Session, user_id: int, *,
                  payload: Dict[str, Any], result: Dict[str, Any],
                  now: float) -> None:
    """Retain what the model saw + produced on the ``infer:last``
    system entity — the audit the UI renders (bounded, never
    authoritative)."""
    ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == LAST_ENTITY).first()
    if ent is None:
        ent = ContextEntity(user_id=user_id, entity_key=LAST_ENTITY,
                            kind="system", name="context infer — last run",
                            attributes="{}")
        db.add(ent)
    try:
        attrs = json.loads(ent.attributes) if ent.attributes else {}
        if not isinstance(attrs, dict):
            attrs = {}
    except (TypeError, ValueError):
        attrs = {}
    attrs.update({
        "at": now,
        "model_id": result.get("model_id"),
        "tier": result.get("thinking"),
        "dry_run": bool(result.get("dry_run")),
        "summary": result.get("summary"),
        "analysis": result.get("analysis"),
        "input": json.dumps(payload, default=str)[:INPUT_KEEP],
        "output": json.dumps(result.get("form") or {},
                             default=str)[:OUTPUT_KEEP],
        "model_text": (result.get("model_text") or "")[:4000],
    })
    ent.attributes = json.dumps(attrs, default=str)
    ent.retired_at = None
    db.commit()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/infer/options")
async def infer_options(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """What the UI needs to drive /infer: the reasoning tiers, the
    user-selectable model set (all large-context), and the resolved
    default per tier."""
    return {
        "tiers": list(_THINKING_TIERS),
        "models": _allowed_models(),
        "defaults": {t: _model_for_tier(t) for t in _THINKING_TIERS},
        "gather_default_s": 900,
    }


@router.get("/infer/last")
async def infer_last(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The last infer run's retained input/output — what the model saw
    and what it produced — or 404 before the first run."""
    ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == current_user.userId,
        ContextEntity.entity_key == LAST_ENTITY).first()
    if ent is None or not ent.attributes:
        raise HTTPException(404, "no infer run recorded yet")
    try:
        attrs = json.loads(ent.attributes)
    except (TypeError, ValueError):
        attrs = {}
    return {"entity": LAST_ENTITY, **(attrs if isinstance(attrs, dict)
                                      else {})}


@router.post("/infer")
async def infer_context(
    body: InferRequest,
    current_user: User = Depends(get_scoped_principal("context:write")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """One window → one complete context layer form.

    The prediction is the function-call arguments themselves; when
    ``dry_run`` is false the form is also submitted to the context
    store. Format violations per section are reported in the receipt
    (valid sections still apply) — a malformed form can never corrupt
    the store because every section is validated against the same
    models as the REST endpoints.

    ``gather_window_s`` > 0 makes Brain assemble the comprehensive
    sensor bundle server-side (device roster, node scenes with radar/
    radio stats, evidence aggregates, context map). The response echoes
    ``seen`` — the exact payload the model received — plus ``analysis``
    and ``model_text`` (what the model produced), and the run is
    retained on the ``infer:last`` entity for the UI.

    This is a hosted-inference boundary: the call is authorized and
    metered through ``server.inference_auth`` before the model runs.
    """
    # 422 on bogus tier/model BEFORE quota reservation.
    model_id = _model_for_tier(body.thinking, body.model)
    payload = _assemble_payload(body, db, current_user.userId)
    with inference_call(db, current_user,
                        kind=f"context_infer:{body.thinking}",
                        model_id=model_id) as meter:
        out = _openai_form(body, payload=payload, usage_out=meter)
    # Test seams may return a bare form; the real call returns
    # {"form", "model_text"}.
    form = out.get("form", out) if isinstance(out, dict) else {}
    model_text = out.get("model_text") if isinstance(out, dict) else None
    result: Dict[str, Any] = {
        "form": form,
        "summary": form.get("summary"),
        "analysis": form.get("analysis"),
        "model_text": model_text,
        "seen": _bounded_seen(payload),
        "thinking": body.thinking,
        "model_id": model_id,
        "questions": form.get("questions") or [],
        "uncertainties": form.get("uncertainties") or [],
        "device_updates": form.get("device_updates") or [],
        "dry_run": body.dry_run,
        "generated_at": time.time(),
    }
    if not body.dry_run:
        result["receipt"] = _apply_form(db, current_user.userId, form)
    _persist_last(db, current_user.userId,
                  payload=payload, result=result,
                  now=result["generated_at"])
    return result


@router.get("/inference/usage")
async def get_inference_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Hosted-inference quota consumption for the current period."""
    return inference_status(db, current_user)
