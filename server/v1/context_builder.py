"""Brain v1 context builder — a stable semantic map from heterogeneous
physical descriptors.

The builder follows the E2 ``jev`` pattern: raw sensor data never reaches
the LLM. Each build summarises the user's recent evidence into compact,
rounded descriptor aggregates per (evidence key, device), adds the current
map snapshot, and asks the model to *update* the map through one forced
tool call (``update_context_map``).

The model proposes; the stabilizer decides. Stability guarantees:

* **Alias resolution** — every proposed id is resolved through the
  aliases recorded on existing entities, so renames don't fork nodes.
* **Hysteresis** — changing an exclusive relationship (``located_in``,
  ``doing``) or a state's value needs ``CONFIRM_BUILDS`` consecutive
  proposals, or one proposal at ``STRONG_CONFIDENCE``.
* **Confidence floor** — proposals under ``MIN_CONFIDENCE`` are ignored.
* **Decay** — builder-owned relationships unconfirmed for ``REL_TTL_S``
  end; persons/activities/objects unseen for their TTL retire. Devices
  are seeded deterministically from the device registry, never by the
  LLM, and never decay.
* **Merges** require both entities to exist with the same kind.

Builder memory (pending candidates, last summary) lives on a single
``system`` entity ``map:builder`` so it survives restarts and is
auditable through the normal context tables.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user, get_scoped_principal
from server.db import (
    ContextEntity, ContextEvidence, ContextRelationship, ContextState,
    Device, User, get_db,
)
from .context import StateIn, apply_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["v1", "context"])

ESTIMATOR = "context-builder/1"
SOURCE = "context-builder"
BUILDER_ENTITY = "map:builder"

KINDS = ("person", "place", "device", "activity", "object")
EXCLUSIVE_PREDICATES = {"located_in", "doing"}
ENTITY_TTL_S = {"person": 7 * 86400.0, "activity": 86400.0,
                "object": 7 * 86400.0, "place": 30 * 86400.0}

MIN_CONFIDENCE = 0.5
STRONG_CONFIDENCE = 0.85
CONFIRM_BUILDS = 2
REL_TTL_S = 1800.0
STATE_TTL_S = 1800.0
DEFAULT_WINDOW_S = 900.0
MAX_EVIDENCE_ROWS = 3000
MAX_AGGREGATES = 60
MAX_FIELDS = 8

_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Bundle — compact descriptor aggregates + current map snapshot
# ---------------------------------------------------------------------------

def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in list(v.items())[:MAX_FIELDS * 2]}
    if isinstance(v, list):
        return [_round(x) for x in v[:MAX_FIELDS]]
    return v


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[Tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield (prefix or "value", float(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k))


def _aggregate(rows: List[ContextEvidence]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        k = (row.evidence_key, row.device_id or "")
        g = groups.setdefault(k, {"key": k[0], "device": k[1] or None,
                                  "n": 0, "first_ts": row.timestamp,
                                  "last_ts": row.timestamp, "conf": [],
                                  "fields": {}, "latest": None})
        g["n"] += 1
        g["first_ts"] = min(g["first_ts"], row.timestamp)
        if row.timestamp >= g["last_ts"]:
            g["last_ts"] = row.timestamp
            try:
                g["latest"] = json.loads(row.value) if row.value else None
            except (TypeError, ValueError):
                g["latest"] = None
        if row.confidence is not None:
            g["conf"].append(row.confidence)
        try:
            value = json.loads(row.value) if row.value else None
        except (TypeError, ValueError):
            value = None
        for name, x in _numeric_leaves(value):
            g["fields"].setdefault(name, []).append(x)
    out = []
    for g in sorted(groups.values(), key=lambda g: -g["n"])[:MAX_AGGREGATES]:
        fields = {}
        for name, xs in list(g["fields"].items())[:MAX_FIELDS]:
            mean = sum(xs) / len(xs)
            fields[name] = {"mean": round(mean, 3), "min": round(min(xs), 3),
                            "max": round(max(xs), 3), "n": len(xs)}
        out.append({
            "key": g["key"], "device": g["device"], "n": g["n"],
            "span_s": round(g["last_ts"] - g["first_ts"], 1),
            "age_s": None,
            "last_ts": g["last_ts"],
            "mean_confidence": (round(sum(g["conf"]) / len(g["conf"]), 3)
                                if g["conf"] else None),
            "fields": fields,
            "latest": _round(g["latest"]),
        })
    return out


def _entity_attrs(entity: ContextEntity) -> Dict[str, Any]:
    try:
        return json.loads(entity.attributes) if entity.attributes else {}
    except (TypeError, ValueError):
        return {}


def _live_entities(db: Session, user_id: int) -> List[ContextEntity]:
    return db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.retired_at.is_(None),
        ContextEntity.kind != "system").all()


def _active_relationships(db: Session, user_id: int,
                          now: float) -> List[ContextRelationship]:
    return db.query(ContextRelationship).filter(
        ContextRelationship.user_id == user_id,
        ContextRelationship.valid_from <= now,
        (ContextRelationship.valid_until.is_(None)) |
        (ContextRelationship.valid_until > now)).all()


def _active_states(db: Session, user_id: int, now: float) -> List[ContextState]:
    return db.query(ContextState).filter(
        ContextState.user_id == user_id,
        (ContextState.valid_until.is_(None)) |
        (ContextState.valid_until > now)).all()


def map_snapshot(db: Session, user_id: int,
                 now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    entities = []
    for e in _live_entities(db, user_id):
        attrs = _entity_attrs(e)
        entities.append({"id": e.entity_key, "kind": e.kind, "name": e.name,
                         "aliases": attrs.get("aliases", []),
                         "confidence": attrs.get("confidence"),
                         "last_seen": attrs.get("last_seen")})
    rels = [{"subject": r.subject, "predicate": r.predicate,
             "object": r.object, "confidence": round(r.confidence or 0, 3)}
            for r in _active_relationships(db, user_id, now)]
    states = []
    for s in _active_states(db, user_id, now):
        try:
            value = json.loads(s.value) if s.value else None
        except (TypeError, ValueError):
            value = None
        states.append({"key": s.state_key, "entity_id": s.entity_id or None,
                       "value": value,
                       "confidence": round(s.confidence or 0, 3)})
    return {"entities": entities, "relationships": rels, "states": states}


def seed_devices(db: Session, user_id: int, now: float) -> List[str]:
    """Devices come from the registry, not the LLM — stable anchors."""
    seeded = []
    for dev in db.query(Device).filter(Device.userId == user_id).all():
        key = f"device:{dev.device_uuid}"
        ent = db.query(ContextEntity).filter(
            ContextEntity.user_id == user_id,
            ContextEntity.entity_key == key).first()
        if ent is None:
            ent = ContextEntity(user_id=user_id, entity_key=key, kind="device")
            db.add(ent)
        attrs = _entity_attrs(ent)
        aliases = set(attrs.get("aliases", []))
        aliases.update({dev.device_uuid, dev.device_name})
        attrs.update({"aliases": sorted(a for a in aliases if a),
                      "device_type": dev.device_type,
                      "registry": True, "confidence": 1.0})
        ent.name = dev.device_name
        ent.kind = "device"
        ent.retired_at = None
        ent.attributes = json.dumps(attrs)
        seeded.append(key)
    db.commit()
    return seeded


def build_bundle(db: Session, user_id: int, now: float,
                 window_s: float = DEFAULT_WINDOW_S) -> Dict[str, Any]:
    rows = db.query(ContextEvidence).filter(
        ContextEvidence.user_id == user_id,
        ContextEvidence.timestamp >= now - window_s,
        ~ContextEvidence.evidence_key.like("context.map.%"),
    ).order_by(ContextEvidence.timestamp.desc()).limit(MAX_EVIDENCE_ROWS).all()
    aggregates = _aggregate(rows)
    for a in aggregates:
        a["age_s"] = round(now - a.pop("last_ts"), 1)
    return {
        "now": now,
        "window_s": window_s,
        "evidence_rows": len(rows),
        "descriptors": aggregates,
        "map": map_snapshot(db, user_id, now),
    }


# ---------------------------------------------------------------------------
# LLM — forced tool call that proposes map updates
# ---------------------------------------------------------------------------

def _map_tool_schema() -> Dict[str, Any]:
    conf = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        "type": "function",
        "function": {
            "name": "update_context_map",
            "description": (
                "Propose updates to the semantic context map. This is the "
                "ONLY output channel. Reuse existing entity ids whenever an "
                "existing entity matches; propose only what the descriptors "
                "support."),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "entities": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string",
                                   "description": "<kind>:<slug>, e.g. "
                                                  "person:gad, place:kitchen"},
                            "kind": {"type": "string", "enum": list(KINDS)},
                            "name": {"type": "string"},
                            "aliases": {"type": "array",
                                        "items": {"type": "string"}},
                            "attributes": {"type": "object"},
                            "confidence": conf,
                        },
                        "required": ["id", "kind", "confidence"]}},
                    "relationships": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string",
                                          "description": "located_in | doing "
                                                         "| uses | carries | "
                                                         "observed_by | "
                                                         "part_of | near"},
                            "object": {"type": "string"},
                            "confidence": conf,
                        },
                        "required": ["subject", "predicate", "object",
                                     "confidence"]}},
                    "states": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string",
                                    "description": "occupancy.v1, "
                                                   "activity.v1, presence.v1"},
                            "entity_id": {"type": "string"},
                            "value": {},
                            "confidence": conf,
                        },
                        "required": ["key", "value", "confidence"]}},
                    "merges": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"source": {"type": "string"},
                                       "target": {"type": "string"},
                                       "reason": {"type": "string"}},
                        "required": ["source", "target"]}},
                    "retire": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"},
                                       "reason": {"type": "string"}},
                        "required": ["id"]}},
                },
                "required": ["summary"],
            },
        },
    }


_SYSTEM_PROMPT = """\
You maintain a STABLE semantic context map of persons, places, devices,
activities and objects for one household/lab. You never see raw sensor
data — only compact physical descriptor aggregates per (evidence key,
device): counts, field mean/min/max, latest value, mean confidence, age.
Typical keys: occupancy/presence probabilities, radar SNR and range,
CSI amplitude variance, BLE/Wi-Fi RSSI sightings (with decoded beacon
identities), IMU motion variance, face/person detections, audio level,
model predictions (key 'prediction'), device metadata/room placement.

Update the map by calling update_context_map exactly once:
  * Reuse existing entity ids from `map` whenever they match — never
    invent a new id for something already on the map. Put alternative
    names/MACs/face ids in `aliases`.
  * Devices are pre-seeded from the registry (device:<uuid>); relate
    them, don't recreate them.
  * Use located_in (person/device/object -> place) and doing
    (person -> activity:<slug>) for the current situation; one active
    place and one activity per subject.
  * States: occupancy.v1 per place ({"occupied": bool}), presence.v1 per
    person ({"present": bool}), activity.v1 per person ({"activity": str}).
  * Confidence must reflect descriptor support: weak, stale (large
    age_s) or conflicting evidence → low confidence. Omit what the
    descriptors don't support; absence of evidence is not evidence of
    absence unless the relevant sensor is fresh and reports it.
  * Only propose merges for clear duplicates and retire only entities the
    evidence contradicts.
"""


def _llm_update(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Forced tool call — returns the parsed proposal."""
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(503, "openai package not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured")
    model = (os.getenv("CONTEXT_BUILDER_MODEL")
             or os.getenv("CONTEXT_INFER_MODEL") or _DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": json.dumps(bundle, default=str)}],
        tools=[_map_tool_schema()],
        tool_choice={"type": "function",
                     "function": {"name": "update_context_map"}},
        temperature=0.1,
    )
    for call in resp.choices[0].message.tool_calls or []:
        if call.function.name == "update_context_map":
            try:
                return json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise HTTPException(502, f"malformed map update JSON: {exc}")
    raise HTTPException(502, "model did not call update_context_map")


# ---------------------------------------------------------------------------
# Stabilizer
# ---------------------------------------------------------------------------

class _Memory:
    """Builder memory persisted on the ``map:builder`` system entity."""

    def __init__(self, db: Session, user_id: int):
        self.db = db
        ent = db.query(ContextEntity).filter(
            ContextEntity.user_id == user_id,
            ContextEntity.entity_key == BUILDER_ENTITY).first()
        if ent is None:
            ent = ContextEntity(user_id=user_id, entity_key=BUILDER_ENTITY,
                                kind="system", name="context builder",
                                attributes="{}")
            db.add(ent)
            db.flush()
        self.entity = ent
        self.data = _entity_attrs(ent)
        self.data.setdefault("pending", {})
        self.data.setdefault("builds", 0)

    def confirm(self, slot: str, candidate: Any, confidence: float) -> bool:
        """True when ``candidate`` may replace the current value of
        ``slot`` (strong proposal or confirmed across builds)."""
        if confidence >= STRONG_CONFIDENCE:
            self.data["pending"].pop(slot, None)
            return True
        pend = self.data["pending"].get(slot)
        token = json.dumps(candidate, sort_keys=True, default=str)
        if pend and pend.get("candidate") == token:
            pend["count"] += 1
        else:
            pend = {"candidate": token, "count": 1}
            self.data["pending"][slot] = pend
        if pend["count"] >= CONFIRM_BUILDS:
            self.data["pending"].pop(slot, None)
            return True
        return False

    def clear(self, slot: str) -> None:
        self.data["pending"].pop(slot, None)

    def save(self, summary: str, now: float) -> None:
        self.data["builds"] = int(self.data.get("builds", 0)) + 1
        self.data["last_build_at"] = now
        self.data["last_summary"] = summary
        self.entity.attributes = json.dumps(self.data, default=str)
        self.db.commit()


def _alias_index(entities: List[ContextEntity]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for e in entities:
        idx[e.entity_key.lower()] = e.entity_key
        for a in _entity_attrs(e).get("aliases", []):
            idx.setdefault(str(a).lower(), e.entity_key)
    return idx


def _resolve(idx: Dict[str, str], key: Optional[str]) -> Optional[str]:
    if not key:
        return key
    return idx.get(str(key).lower(), key)


def _upsert_entity(db: Session, user_id: int, raw: Dict[str, Any],
                   now: float) -> Optional[ContextEntity]:
    kind = raw.get("kind")
    key = str(raw.get("id") or "").strip()
    if kind not in KINDS or not key or key == BUILDER_ENTITY:
        return None
    ent = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == key).first()
    if ent is None:
        ent = ContextEntity(user_id=user_id, entity_key=key, kind=kind)
        db.add(ent)
    elif ent.kind == "device" and kind != "device":
        return ent  # registry devices keep their kind
    attrs = _entity_attrs(ent)
    aliases = set(attrs.get("aliases", [])) | set(raw.get("aliases") or [])
    aliases.discard(key)
    extra = raw.get("attributes") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in ("aliases", "registry", "pending"):
                attrs[k] = v
    prev = attrs.get("confidence")
    c = float(raw.get("confidence") or 0)
    attrs["confidence"] = round(c if prev is None else 0.7 * prev + 0.3 * c, 3)
    attrs["aliases"] = sorted(str(a) for a in aliases if a)
    attrs["last_seen"] = now
    ent.kind = ent.kind if ent.kind == "device" else kind
    ent.name = raw.get("name") or ent.name
    ent.retired_at = None
    ent.attributes = json.dumps(attrs, default=str)
    db.flush()
    return ent


def _touch(db: Session, user_id: int, keys: Iterable[str], now: float) -> None:
    for key in set(keys):
        ent = db.query(ContextEntity).filter(
            ContextEntity.user_id == user_id,
            ContextEntity.entity_key == key).first()
        if ent is None:
            continue
        attrs = _entity_attrs(ent)
        attrs["last_seen"] = now
        ent.attributes = json.dumps(attrs, default=str)


def _new_rel(db: Session, user_id: int, s: str, p: str, o: str,
             conf: float, now: float) -> ContextRelationship:
    rel = ContextRelationship(
        user_id=user_id, subject=s, predicate=p, object=o,
        valid_from=now, valid_until=None, confidence=round(conf, 3),
        source=SOURCE, provenance=json.dumps({"confirmed_at": now}))
    db.add(rel)
    return rel


def apply_map_update(db: Session, user_id: int, update: Dict[str, Any],
                     now: Optional[float] = None) -> Dict[str, Any]:
    """Apply a proposed update with alias resolution, hysteresis and
    decay. Returns a receipt of what changed and what is pending."""
    now = now or time.time()
    mem = _Memory(db, user_id)
    receipt: Dict[str, Any] = {"entities": [], "merged": [], "retired": [],
                               "relationships": {"created": [], "refreshed": [],
                                                 "ended": [], "pending": []},
                               "states": {"applied": [], "pending": []},
                               "skipped": []}

    entities = _live_entities(db, user_id)
    idx = _alias_index(entities)

    # 1. entities — resolve proposals onto existing ids first
    for raw in update.get("entities") or []:
        if float(raw.get("confidence") or 0) < MIN_CONFIDENCE:
            receipt["skipped"].append({"entity": raw.get("id"),
                                       "reason": "low confidence"})
            continue
        raw = dict(raw)
        resolved = _resolve(idx, raw.get("id"))
        if resolved != raw.get("id"):
            raw["aliases"] = list(raw.get("aliases") or []) + [raw["id"]]
            raw["id"] = resolved
        ent = _upsert_entity(db, user_id, raw, now)
        if ent is None:
            receipt["skipped"].append({"entity": raw.get("id"),
                                       "reason": "invalid kind/id"})
            continue
        receipt["entities"].append(ent.entity_key)
        idx[ent.entity_key.lower()] = ent.entity_key
        for a in _entity_attrs(ent).get("aliases", []):
            idx[str(a).lower()] = ent.entity_key
    db.commit()

    live = {e.entity_key: e for e in _live_entities(db, user_id)}

    # 2. merges — same kind only; aliases + active relationships move
    for m in update.get("merges") or []:
        src = _resolve(idx, m.get("source"))
        dst = _resolve(idx, m.get("target"))
        a, b = live.get(src), live.get(dst)
        if not a or not b or a is b or a.kind != b.kind or a.kind == "device":
            receipt["skipped"].append({"merge": [src, dst],
                                       "reason": "invalid merge"})
            continue
        battrs = _entity_attrs(b)
        aliases = set(battrs.get("aliases", [])) | set(
            _entity_attrs(a).get("aliases", [])) | {a.entity_key}
        battrs["aliases"] = sorted(aliases)
        b.attributes = json.dumps(battrs, default=str)
        a.retired_at = now
        for rel in _active_relationships(db, user_id, now):
            if src in (rel.subject, rel.object):
                rel.valid_until = now
                _new_rel(db, user_id,
                         dst if rel.subject == src else rel.subject,
                         rel.predicate,
                         dst if rel.object == src else rel.object,
                         rel.confidence or 0.5, now)
        for alias in aliases:
            idx[str(alias).lower()] = dst
        live.pop(src, None)
        receipt["merged"].append({"source": src, "target": dst})
    db.commit()

    # 3. relationships — dedupe, hysteresis on exclusive predicates
    active = _active_relationships(db, user_id, now)
    seen_entities: List[str] = list(receipt["entities"])
    for raw in update.get("relationships") or []:
        s = _resolve(idx, raw.get("subject"))
        p = str(raw.get("predicate") or "").strip()
        o = _resolve(idx, raw.get("object"))
        conf = float(raw.get("confidence") or 0)
        if conf < MIN_CONFIDENCE or not p:
            receipt["skipped"].append({"relationship": [s, p, o],
                                       "reason": "low confidence"})
            continue
        if s not in live or o not in live:
            receipt["skipped"].append({"relationship": [s, p, o],
                                       "reason": "unresolved endpoint"})
            continue
        seen_entities += [s, o]
        same = next((r for r in active if r.subject == s and r.predicate == p
                     and r.object == o), None)
        slot = f"rel|{s}|{p}"
        if same is not None:
            same.confidence = round(0.7 * (same.confidence or 0) + 0.3 * conf, 3)
            prov = json.loads(same.provenance) if same.provenance else {}
            prov["confirmed_at"] = now
            same.provenance = json.dumps(prov)
            mem.clear(slot)
            receipt["relationships"]["refreshed"].append(same.id)
            continue
        rivals = ([r for r in active if r.subject == s and r.predicate == p]
                  if p in EXCLUSIVE_PREDICATES else [])
        if rivals and not mem.confirm(slot, o, conf):
            receipt["relationships"]["pending"].append([s, p, o])
            continue
        for r in rivals:
            r.valid_until = now
            receipt["relationships"]["ended"].append(r.id)
            active.remove(r)
        rel = _new_rel(db, user_id, s, p, o, conf, now)
        db.flush()
        active.append(rel)
        receipt["relationships"]["created"].append(rel.id)
    db.commit()

    # 4. states — hysteresis on value changes, TTL so stale states expire
    current = {(st.state_key, st.entity_id or ""): st
               for st in _active_states(db, user_id, now)}
    for raw in update.get("states") or []:
        key = str(raw.get("key") or "").strip()
        ent = _resolve(idx, raw.get("entity_id")) or ""
        conf = float(raw.get("confidence") or 0)
        if not key or conf < MIN_CONFIDENCE:
            receipt["skipped"].append({"state": key, "reason": "low confidence"})
            continue
        if ent and ent not in live:
            receipt["skipped"].append({"state": key,
                                       "reason": "unresolved entity"})
            continue
        value = raw.get("value")
        prev = current.get((key, ent))
        slot = f"state|{key}|{ent}"
        if prev is not None:
            try:
                prev_value = json.loads(prev.value) if prev.value else None
            except (TypeError, ValueError):
                prev_value = None
            if prev_value != value and not mem.confirm(slot, value, conf):
                prev.valid_until = now + STATE_TTL_S  # keep current alive
                receipt["states"]["pending"].append(
                    {"key": key, "entity_id": ent or None, "value": value})
                continue
            if prev_value == value:
                mem.clear(slot)
        st = apply_state(db, user_id, StateIn(
            key=key, value=value, entity_id=ent, confidence=conf,
            valid_until=now + STATE_TTL_S, estimator=ESTIMATOR))
        if ent:
            seen_entities.append(ent)
        receipt["states"]["applied"].append(st.to_dict())

    # 5. explicit retirements — never registry devices
    for raw in update.get("retire") or []:
        key = _resolve(idx, raw.get("id"))
        ent = live.get(key)
        if ent is None or ent.kind == "device" or key in seen_entities:
            receipt["skipped"].append({"retire": key,
                                       "reason": "protected or seen"})
            continue
        ent.retired_at = now
        live.pop(key, None)
        receipt["retired"].append(key)

    _touch(db, user_id, seen_entities, now)
    db.commit()

    # 6. decay
    receipt["decayed"] = decay(db, user_id, now)
    mem.save(str(update.get("summary") or ""), now)
    receipt["summary"] = update.get("summary")
    return receipt


def decay(db: Session, user_id: int, now: float) -> Dict[str, List[Any]]:
    ended, retired = [], []
    for rel in _active_relationships(db, user_id, now):
        if rel.source != SOURCE:
            continue
        prov = json.loads(rel.provenance) if rel.provenance else {}
        if now - float(prov.get("confirmed_at") or rel.valid_from) > REL_TTL_S:
            rel.valid_until = now
            ended.append(rel.id)
    for ent in _live_entities(db, user_id):
        ttl = ENTITY_TTL_S.get(ent.kind)
        if ttl is None:
            continue
        last = _entity_attrs(ent).get("last_seen")
        if last is not None and now - float(last) > ttl:
            ent.retired_at = now
            retired.append(ent.entity_key)
    db.commit()
    return {"relationships": ended, "entities": retired}


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------

LLMFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def run_build(db: Session, user_id: int, *, now: Optional[float] = None,
              window_s: float = DEFAULT_WINDOW_S, dry_run: bool = False,
              llm: Optional[LLMFn] = None) -> Dict[str, Any]:
    now = now or time.time()
    seed_devices(db, user_id, now)
    bundle = build_bundle(db, user_id, now, window_s)
    update = (llm or _llm_update)(bundle)
    result: Dict[str, Any] = {"generated_at": now, "dry_run": dry_run,
                              "evidence_rows": bundle["evidence_rows"],
                              "descriptor_groups": len(bundle["descriptors"]),
                              "proposal": update}
    if not dry_run:
        result["receipt"] = apply_map_update(db, user_id, update, now)
    return result


def grouped_map(db: Session, user_id: int,
                now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    snap = map_snapshot(db, user_id, now)
    plural = {"person": "persons", "place": "places", "device": "devices",
              "activity": "activities", "object": "objects"}
    out: Dict[str, Any] = {name: [] for name in plural.values()}
    out["other"] = []
    for e in snap["entities"]:
        out[plural.get(e["kind"], "other")].append(e)
    builder = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == BUILDER_ENTITY).first()
    battrs = _entity_attrs(builder) if builder else {}
    return {
        **out,
        "relationships": snap["relationships"],
        "states": snap["states"],
        "builder": {"builds": battrs.get("builds", 0),
                    "last_build_at": battrs.get("last_build_at"),
                    "last_summary": battrs.get("last_summary"),
                    "pending": len(battrs.get("pending", {}))},
        "generated_at": now,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class RebuildRequest(BaseModel):
    window_s: float = Field(default=DEFAULT_WINDOW_S, gt=0, le=86400)
    dry_run: bool = False


@router.get("/map")
async def get_context_map(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """The stable semantic map: persons, places, devices, activities,
    objects + active relationships/states + builder status."""
    return grouped_map(db, current_user.userId)


@router.post("/rebuild")
async def rebuild_context_map(
    body: RebuildRequest = RebuildRequest(),
    current_user: User = Depends(get_scoped_principal("context:write")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Run one builder pass now over the last ``window_s`` of evidence."""
    result = await asyncio.to_thread(
        run_build, db, current_user.userId,
        window_s=body.window_s, dry_run=body.dry_run)
    result["map"] = grouped_map(db, current_user.userId)
    return result


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

def _interval_s() -> float:
    try:
        return float(os.getenv("CONTEXT_BUILDER_INTERVAL_S", "300"))
    except ValueError:
        return 300.0


def build_due_users(session_factory, now: Optional[float] = None) -> List[int]:
    """One scheduler tick: build for every user with fresh evidence."""
    now = now or time.time()
    interval = _interval_s()
    built = []
    db = session_factory()
    try:
        users = [u for (u,) in db.query(ContextEvidence.user_id).filter(
            ContextEvidence.timestamp >= now - interval,
            ~ContextEvidence.evidence_key.like("context.map.%"),
        ).distinct().all()]
        for uid in users:
            try:
                run_build(db, uid, now=now, window_s=max(interval * 3, 300.0))
                built.append(uid)
            except HTTPException as exc:
                logger.warning("context build user %s: %s", uid, exc.detail)
                db.rollback()
                if exc.status_code == 503:
                    break
            except Exception:
                logger.exception("context build failed for user %s", uid)
                db.rollback()
    finally:
        db.close()
    return built


async def context_builder_loop(session_factory) -> None:
    interval = _interval_s()
    if interval <= 0 or not os.getenv("OPENAI_API_KEY"):
        logger.info("context builder disabled (interval=%s, key=%s)",
                    interval, bool(os.getenv("OPENAI_API_KEY")))
        return
    logger.info("context builder running every %.0fs", interval)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(build_due_users, session_factory)
        except Exception:
            logger.exception("context builder tick failed")
