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

from server.auth import get_current_user, get_scoped_principal
from server.db import User, get_db
from server.inference_auth import inference_call, inference_status
from .context import (
    EntityIn, EvidenceIn, RelationshipIn, StateIn,
    apply_entity, apply_evidence_items, apply_relationship, apply_state,
    is_confirmed_source,
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
                },
                "required": ["summary", "states"],
            },
        },
    }


_SYSTEM_PROMPT = """\
You are the context-layer estimator for a sensor-fusion platform. For
each request you receive:
  * `calibration` — the REFERENCE distributions: statistics of the
    target classes learned during calibration. Entries carry per-class
    feature mean/std/count; RSSI fingerprints are {anchor: {mean,std,
    count}}; radar descriptors carry per-range-bin/angle statistics.
    Judge a window by its distance to these references — a descriptor
    several std from every class centroid is a low-confidence or
    unknown classification, never a forced label.
  * `descriptors` — physical descriptors of ONE time window from the
    sensors (CSI/radar/BLE/IMU summaries, RSSI values, variances,
    packet counts, spectral features).
  * `context` — the account's assets around the window: entity
    inventory (names/kinds/confirmed attributes), device metadata
    (platform, capabilities, firmware), digital context (app/network/
    peripheral state, traffic summaries). Device type/state may be
    INFERRED from metadata + traffic — land it as a proposal entity
    attribute, never confirmed.
  * `history` — recent prior windows/forms, for temporal consistency
    (a label that contradicts the last stable window needs stronger
    evidence).
  * `coverage` — if non-empty, fill only the named form sections.
  * `window` — the window's bounds and provenance.

Classify the window against the calibration references, then respond
ONLY by calling submit_context_form. Do not emit plain text. Rules:
  * states[] must contain one entry per target class key the platform
    uses (e.g. occupancy.v1, activity.v1, location.v1); value is the
    predicted label/object, confidence is calibrated by distance to the
    class statistics — not raw probability.
  * evidence[] records what you based the prediction on (the model
    probabilities, the descriptors, the calibration reference).
  * entities[] and relationships[] describe WHO/WHERE/WHAT the window
    implies — create them when confident, omit when unsupported.
  * uncertainties[] reports claims you considered but rejected — what
    was unresolved and why; questions[] asks for the specific
    observations that would resolve them; notes[] records durable
    entity/relationship observations for future windows. Use these
    channels — silent ambiguity is worse than an explicit question.
  * All numbers are floats (epoch seconds); no prose inside values.
"""


# ---------------------------------------------------------------------------
# LLM call + form application
# ---------------------------------------------------------------------------

_THINKING_TIERS = ("quick", "standard", "deep")


def _model_for_tier(tier: str) -> str:
    """Resolve a thinking tier to a concrete model id.

    ``standard`` keeps the historical env knobs
    (``CONTEXT_INFER_MODEL`` → ``MODEL_NAME`` → default); ``quick``
    and ``deep`` have their own overrides so operators can price the
    meter. Unknown tiers are a client error."""
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


def _openai_form(request: InferRequest, *,
                 usage_out: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Force the model to fill the form — returns parsed tool args.

    ``usage_out``, when supplied, is filled with metering fields
    (``model_id``, ``tokens``, ``tier``) for the inference ledger."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = _model_for_tier(request.thinking)

    client = OpenAI(api_key=api_key)
    user_msg = json.dumps({
        "window": request.window,
        "entity_hint": request.entity_hint,
        "calibration": request.calibration,
        "descriptors": request.descriptors,
        "context": request.context,
        "history": request.history[:20],
        "coverage": request.coverage,
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
    if usage_out is not None:
        usage_out["model_id"] = model
        usage_out["tier"] = request.thinking
        usage_out["tokens"] = getattr(
            getattr(resp, "usage", None), "total_tokens", None)
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

    if skipped:
        receipt["skipped"] = skipped
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

    This is a hosted-inference boundary: the call is authorized and
    metered through ``server.inference_auth`` before the model runs.
    """
    model_id = _model_for_tier(body.thinking)  # 422 on bogus tier
    with inference_call(db, current_user,
                        kind=f"context_infer:{body.thinking}",
                        model_id=model_id) as meter:
        form = _openai_form(body, usage_out=meter)
    result: Dict[str, Any] = {
        "form": form,
        "summary": form.get("summary"),
        "thinking": body.thinking,
        "model_id": model_id,
        "questions": form.get("questions") or [],
        "uncertainties": form.get("uncertainties") or [],
        "dry_run": body.dry_run,
        "generated_at": time.time(),
    }
    if not body.dry_run:
        result["receipt"] = _apply_form(db, current_user.userId, form)
    return result


@router.get("/inference/usage")
async def get_inference_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Hosted-inference quota consumption for the current period."""
    return inference_status(db, current_user)
