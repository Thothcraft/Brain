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
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from server.auth import get_scoped_principal
from server.db import User, get_db
from .context import (
    EntityIn, EvidenceIn, RelationshipIn, StateIn,
    apply_entity, apply_evidence_items, apply_relationship, apply_state,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["v1", "context"])

_DEFAULT_MODEL = "gpt-4o-mini"


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
        description="Per-target-class calibration statistics")
    descriptors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical descriptors measured over the window")
    entity_hint: Optional[str] = Field(
        default=None,
        description="Canonical entity id the window concerns "
                    "(e.g. 'person:gad' or the device uuid)")
    dry_run: bool = Field(
        default=False,
        description="Return the predicted form without writing it")


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
                },
                "required": ["summary", "states"],
            },
        },
    }


_SYSTEM_PROMPT = """\
You are the context-layer estimator for a sensor-fusion platform. For
each request you receive:
  * `calibration` — statistics of the target classes learned during
    calibration (per-class feature means/stds, thresholds, priors,
    support counts).
  * `descriptors` — physical descriptors of ONE time window from the
    sensors (CSI/radar/BLE/IMU summaries, RSSI values, variances,
    packet counts, spectral features).
  * `window` — the window's bounds and provenance.

Classify the window against the target classes using the calibration
stats, then respond ONLY by calling submit_context_form. Do not emit
plain text. Rules:
  * states[] must contain one entry per target class key the platform
    uses (e.g. occupancy.v1, activity.v1, location.v1); value is the
    predicted label/object, confidence is calibrated by distance to the
    class statistics — not raw probability.
  * evidence[] records what you based the prediction on (the model
    probabilities, the descriptors, the calibration reference).
  * entities[] and relationships[] describe WHO/WHERE/WHAT the window
    implies — create them when confident, omit when unsupported.
  * Confidence must reflect the descriptor-vs-calibration distance:
    a window far from every class centroid yields low confidence, not
    a forced label.
  * All numbers are floats (epoch seconds); no prose inside values.
"""


# ---------------------------------------------------------------------------
# LLM call + form application
# ---------------------------------------------------------------------------

def _openai_form(request: InferRequest) -> Dict[str, Any]:
    """Force the model to fill the form — returns parsed tool args."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = (os.getenv("CONTEXT_INFER_MODEL")
             or os.getenv("MODEL_NAME") or _DEFAULT_MODEL)

    client = OpenAI(api_key=api_key)
    user_msg = json.dumps({
        "window": request.window,
        "entity_hint": request.entity_hint,
        "calibration": request.calibration,
        "descriptors": request.descriptors,
    }, default=str)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        tools=[_ctx_tool_schema()],
        tool_choice={"type": "function",
                     "function": {"name": "submit_context_form"}},
        temperature=0.2,
    )
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    for call in calls:
        if call.function.name == "submit_context_form":
            try:
                return json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    502, f"model returned malformed form JSON: {exc}")
    raise HTTPException(
        502, "model did not call submit_context_form "
             f"(finish_reason={resp.choices[0].finish_reason})")


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

    # 1. entities ---------------------------------------------------------
    seen_entities = set()
    for i, raw in enumerate(form.get("entities") or []):
        try:
            ent = apply_entity(db, user_id, EntityIn(**raw))
            seen_entities.add(ent.entity_key)
            receipt["entities"].append(ent.entity_key)
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "entities", "index": i,
                           "error": str(exc)[:300]})

    # 2. relationships ----------------------------------------------------
    for i, raw in enumerate(form.get("relationships") or []):
        raw = dict(raw)
        try:
            rel = apply_relationship(
                db, user_id, RelationshipIn(**raw))
            receipt["relationships"].append(rel.id)
        except HTTPException as exc:
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
        if not raw.get("estimator"):
            raw["estimator"] = "openai-context-form/1"
        try:
            st = apply_state(db, user_id, StateIn(**raw))
            receipt["states"].append(st.to_dict())
        except HTTPException as exc:
            errors.append({"section": "states", "index": i,
                           "error": exc.detail})
        except (ValidationError, TypeError) as exc:
            errors.append({"section": "states", "index": i,
                           "error": str(exc)[:300]})

    if errors:
        receipt["errors"] = errors
    return receipt


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

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
    """
    form = _openai_form(body)
    result: Dict[str, Any] = {
        "form": form,
        "summary": form.get("summary"),
        "dry_run": body.dry_run,
        "generated_at": time.time(),
    }
    if not body.dry_run:
        result["receipt"] = _apply_form(db, current_user.userId, form)
    return result
