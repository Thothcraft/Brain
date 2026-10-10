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
* **Confirmed-fact precedence** — entities/relationships/states the user
  confirmed (see ``context.py``) are never mutated, contradicted, ended
  or retired by a proposal; they appear in the receipt as skipped
  ``confirmed_fact`` entries. A confirmed write supersedes through the
  normal context API, not the builder.

Builder memory (pending candidates, last summary) lives on a single
``system`` entity ``map:builder`` so it survives restarts and is
auditable through the normal context tables.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from server.inference_auth import inference_call
from .context import (
    StateIn, apply_state, entity_is_confirmed, is_confirmed_source,
    rel_is_confirmed,
)

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
MAX_AGGREGATES = 80
MAX_FIELDS = 12

# The builder reads a lot of physical evidence — a tier above the
# infer path pays for itself in map quality. Env-overridable per
# deployment; the actual model used is metered on every call.
_DEFAULT_MODEL = "gpt-4o"

# How much of the model's input/output text is retained on the
# map:builder entity for audit ("what the model saw / produced").
INPUT_KEEP = 12000
PROPOSAL_KEEP = 8000


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
                         "last_seen": attrs.get("last_seen"),
                         "confirmed": entity_is_confirmed(e)})
    rels = [{"subject": r.subject, "predicate": r.predicate,
             "object": r.object, "confidence": round(r.confidence or 0, 3),
             "confirmed": rel_is_confirmed(r)}
            for r in _active_relationships(db, user_id, now)]
    states = []
    for s in _active_states(db, user_id, now):
        try:
            value = json.loads(s.value) if s.value else None
        except (TypeError, ValueError):
            value = None
        states.append({"key": s.state_key, "entity_id": s.entity_id or None,
                       "value": value,
                       "confidence": round(s.confidence or 0, 3),
                       "confirmed": is_confirmed_source(s.estimator or "")})
    return {"entities": entities, "relationships": rels, "states": states}


def seed_devices(db: Session, user_id: int, now: float) -> List[str]:
    """Devices come from the registry, not the LLM — stable anchors.

    MAC and hostname join the alias set so a fingerprint (``aa:bb:…``,
    ``rpi-e1``) proposed by either model resolves onto the registered
    ``device:<uuid>`` anchor instead of forking a second entity.
    """
    seeded = []
    for dev in db.query(Device).filter(Device.userId == user_id).all():
        key = f"device:{dev.device_uuid}"
        ent = db.query(ContextEntity).filter(
            ContextEntity.user_id == user_id,
            ContextEntity.entity_key == key).first()
        if ent is None:
            ent = ContextEntity(user_id=user_id, entity_key=key, kind="device")
            db.add(ent)
        if entity_is_confirmed(ent):
            # User-curated entity — refresh presence only; never clobber
            # the confirmed name/attributes.
            attrs = _entity_attrs(ent)
            attrs["last_seen"] = now
            ent.attributes = json.dumps(attrs, default=str)
            ent.retired_at = None
            seeded.append(key)
            continue
        attrs = _entity_attrs(ent)
        aliases = set(attrs.get("aliases", []))
        aliases.update({dev.device_uuid, dev.device_name, dev.mac_address})
        if dev.hardware_info:
            try:
                hw = json.loads(dev.hardware_info)
                if isinstance(hw, dict):
                    aliases.add(hw.get("hostname"))
                    aliases.add(hw.get("mac"))
                    aliases.add(hw.get("mac_address"))
            except (TypeError, ValueError):
                pass
        aliases.update({m for m in (_norm_mac(a) for a in aliases) if m})
        attrs.update({"aliases": sorted(a for a in aliases if a),
                      "device_type": dev.device_type,
                      "registry": True, "confidence": 1.0})
        ent.name = dev.device_name
        ent.kind = "device"
        ent.retired_at = None
        ent.attributes = json.dumps(attrs)
        seeded.append(key)
    # Account anchors — every user owns at least one person (themselves)
    # and one indoor place (their address). Seeded unconfirmed so the
    # model refines/renames them as site/geo evidence lands.
    owner = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == "person:owner").first()
    if owner is None:
        name = None
        try:
            user = db.query(User).filter(User.userId == user_id).first()
            if user is not None:
                name = (user.username or (user.email or "").split("@")[0]
                        or None)
        except Exception:
            name = None
        owner = ContextEntity(
            user_id=user_id, entity_key="person:owner", kind="person",
            name=name or "owner",
            attributes=json.dumps(
                {"owner": True, "seeded": True,
                 "associated_devices": [], "last_seen": now}))
        db.add(owner)
    else:
        attrs = _entity_attrs(owner)
        attrs["last_seen"] = now
        owner.attributes = json.dumps(attrs, default=str)
        owner.retired_at = None
    home = db.query(ContextEntity).filter(
        ContextEntity.user_id == user_id,
        ContextEntity.entity_key == "place:home").first()
    if home is None:
        home = ContextEntity(
            user_id=user_id, entity_key="place:home", kind="place",
            name="home",
            attributes=json.dumps(
                {"seeded": True, "role": "primary_address",
                 "rooms": [], "last_seen": now}))
        db.add(home)
    else:
        attrs = _entity_attrs(home)
        attrs["last_seen"] = now
        home.attributes = json.dumps(attrs, default=str)
        home.retired_at = None
    seeded += ["person:owner", "place:home"]
    db.commit()
    return seeded


# ---------------------------------------------------------------------------
# Device fingerprints — one entity per physical device
# ---------------------------------------------------------------------------

_MAC_RE = re.compile(r"(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}", re.I)
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
# Attribute keys that hold hardware identifiers — folded into aliases so
# any later proposal naming the same hardware resolves, not duplicates.
_FP_ATTR_KEYS = ("mac", "mac_address", "bluetooth_address", "bt_mac",
                 "wifi_mac", "bssid", "uuid", "serial", "hostname")


def _norm_mac(value: Any) -> Optional[str]:
    """Canonical MAC (lowercase, ``:`` separators) or None."""
    if not isinstance(value, str):
        return None
    m = _MAC_RE.search(value.lower().replace("-", ":"))
    return m.group(0) if m else None


def _fingerprints(*values: Any) -> List[str]:
    """Fingerprint keys for a ref/attr value: mac:/uuid:/name:/id:."""
    out: List[str] = []
    for v in values:
        if not isinstance(v, str) or not v.strip():
            continue
        s = v.strip().lower()
        mac = _norm_mac(s)
        if mac:
            out.append(f"mac:{mac}")
        else:
            for m in _UUID_RE.findall(s):
                out.append(f"uuid:{m}")
            s2 = s.split(":", 1)[-1] if s.startswith(("device:", "ble:",
                                                      "wifi:", "bt:")) else s
            out.append(f"id:{s2}")
            out.append(f"name:{s2}")
    return out


def _device_fp_index(db: Session, user_id: int) -> Dict[str, str]:
    """fingerprint -> canonical entity_key, from the live map (device
    kind only — persons/places sharing a hostname is not a match) plus
    the registry (so unseeded ``device:<uuid>`` anchors still resolve)."""
    idx: Dict[str, str] = {}
    for e in _live_entities(db, user_id):
        if e.kind != "device":
            continue
        attrs = _entity_attrs(e)
        refs = [e.entity_key, e.name] + list(attrs.get("aliases") or [])
        refs += [attrs.get(k) for k in _FP_ATTR_KEYS]
        for fp in _fingerprints(*refs):
            # first claim wins — registered anchors were seeded earlier
            idx.setdefault(fp, e.entity_key)
    for dev in db.query(Device).filter(Device.userId == user_id).all():
        key = f"device:{dev.device_uuid}"
        refs = [dev.device_uuid, dev.device_name, dev.mac_address, key]
        if dev.hardware_info:
            try:
                hw = json.loads(dev.hardware_info)
                if isinstance(hw, dict):
                    refs += [hw.get("hostname"), hw.get("mac"),
                             hw.get("mac_address")]
            except (TypeError, ValueError):
                pass
        for fp in _fingerprints(*refs):
            idx.setdefault(fp, key)
    return idx


def _resolve_device(fp_idx: Dict[str, str], *refs: Any) -> Optional[str]:
    """Best-effort map of a device ref (id, uuid, MAC, hostname, beacon
    id, name) onto the canonical entity_key — None when nothing matches."""
    for fp in _fingerprints(*refs):
        hit = fp_idx.get(fp)
        if hit:
            return hit
    return None


DESCRIPTOR_KEY = "context.descriptors.v1"
MAX_SCENES = 12

# Dataset-derived empty-vs-occupied reference for the physical fields
# nodes actually uplink. Source: E1 multilink capture (desktop/radar/E1,
# Oct 2026) — BGT60TR13C mmWave @ ~10 Hz + dual-link CSI, 5 s windows,
# 2154 empty / 2160 occupied windows, RF cv_acc ~1.0.
# A prior for THIS sensor class, not per-site calibration: absolute map
# means are mount-dependent (the legacy rig reversed ra_mean's
# direction) — the transferable shape is that medians barely move while
# occupancy lives in the window TAIL (snr_max spikes, std90 spreads
# widen ~2x, ra/re/xy means lift ~+0.1-0.2 log-power).
RADAR_REFERENCE: Dict[str, Any] = {
    "dataset": "e1_multilink_5s",
    "sensor": "BGT60TR13C mmWave radar + Wi-Fi CSI",
    "windows": {"empty": 2154, "occupied": 2160, "window_s": 5,
                "frame_hz": 10},
    "how_to_read": (
        "empty-room stats are TIGHT (snr_db.mean p95<8, max<8.3; "
        "snr_db.max never >~13); occupancy shows in the TAIL of the "
        "window — snr_db.max spikes >15-40 dB, ra/re/xy map means lift "
        "~+0.1-0.2, *std90 spreads widen ~2x. A modest snr mean alone "
        "is weak (AUC~0.7): call occupied on spikes + map lifts, not a "
        "fixed median threshold. Compare a node's lift vs ITS OWN empty "
        "baseline — absolute values shift with mounting."),
    "fields": {
        # node `fields.snr_db.{mean,max}` == window snr_mean/snr_max
        "snr_db.mean": {"empty": {"mean": 7.62, "std": 0.20,
                                  "p95": 7.93, "max": 8.28},
                        "occupied": {"p50": 7.80, "p75": 8.47,
                                     "p95": 22.37, "max": 31.69},
                        "d": 0.63, "auc": 0.71},
        "snr_db.max": {"empty": {"mean": 10.07, "std": 0.62,
                                  "p95": 11.23, "max": 12.88},
                       "occupied": {"p50": 10.59, "p75": 17.77,
                                    "p95": 40.60, "max": 51.89},
                       "d": 0.80, "auc": 0.71},
        # range-azimuth / range-elevation / xy map window means
        "ra_mean": {"empty": {"p50": 5.693, "p95": 5.734,
                              "max": 5.750},
                    "occupied": {"p50": 5.830, "p75": 5.888,
                                 "p95": 5.982},
                    "d": 3.08, "auc": 1.00},
        "re_mean": {"empty": {"p50": 5.788, "p95": 5.808},
                    "occupied": {"p50": 5.926, "p75": 5.970,
                                 "p95": 6.080},
                    "d": 2.91, "auc": 0.99},
        "xy_mean": {"empty": {"p50": 3.400, "p95": 3.412},
                    "occupied": {"p50": 3.446, "p75": 3.476,
                                 "p95": 3.506},
                    "d": 2.04, "auc": 0.94},
        "re_p90": {"empty": {"p50": 7.822, "p95": 8.058},
                   "occupied": {"p50": 8.316, "p75": 8.417,
                                "p95": 8.654},
                   "d": 1.98, "auc": 0.92},
        "xy_peak": {"empty": {"p50": 14.184, "p95": 14.202},
                    "occupied": {"p50": 14.212, "p75": 14.218,
                                 "p95": 14.223},
                    "d": 1.88, "auc": 0.90},
        # within-window spread of the map tails — ~2x wider when occupied
        "re_std90": {"empty": {"p50": 0.108, "p95": 0.113},
                     "occupied": {"p50": 0.220, "p75": 0.330,
                                  "p95": 0.873},
                     "d": 1.14, "auc": 0.97},
        "xy_std90": {"empty": {"p50": 0.108, "p95": 0.112},
                     "occupied": {"p50": 0.145, "p75": 0.185,
                                  "p95": 0.444},
                     "d": 0.99, "auc": 0.98},
        "rd_mean": {"empty": {"p50": 6.597, "p95": 6.613},
                    "occupied": {"p50": 6.610, "p75": 6.624,
                                 "p95": 6.715},
                    "d": 0.82},
        # CSI amplitude mean DROPPED under occupancy on this rig
        # (shadowing) — direction flips with link geometry, treat as
        # magnitude-of-change evidence only
        "csi_amp_mean": {"empty": {"p50": 27.65, "p95": 38.12},
                         "occupied": {"p50": 24.81, "p95": 29.91},
                         "d": -1.54, "auc": 0.88,
                         "direction": "mount-dependent"},
    },
}


def _devices(db: Session, user_id: int) -> List[Dict[str, Any]]:
    """The device registry, keyed by the map entity id each scene's
    `device` uuid joins to — so the model always knows which physical
    machine a uuid is (name, type, MAC, hardware) and can tell a
    registered node from an unknown emitter seen only in scans."""
    out = []
    seen_fp: Dict[str, str] = {}
    for dev in db.query(Device).filter(Device.userId == user_id).all():
        hw: Dict[str, Any] = {}
        if dev.hardware_info:
            try:
                parsed = json.loads(dev.hardware_info)
                if isinstance(parsed, dict):
                    hw = parsed
            except (TypeError, ValueError):
                hw = {}
        key = f"device:{dev.device_uuid}"
        entry: Dict[str, Any] = {
            "entity": key,
            "uuid": dev.device_uuid,
            "name": dev.device_name,
            "type": dev.device_type,
            "mac": dev.mac_address,
            "online": bool(dev.online),
            "last_seen": (dev.last_seen.isoformat() + "Z"
                          if dev.last_seen else None),
            "hardware": {k: hw.get(k) for k in
                         ("hostname", "model", "os", "platform",
                          "chip") if hw.get(k)},
        }
        # Two registry rows with the same fingerprint = the same physical
        # device double-registered — flag the earlier entry as canonical
        # so the model never treats them as two machines.
        for fp in _fingerprints(dev.device_uuid, dev.mac_address,
                                hw.get("mac"), hw.get("mac_address"),
                                hw.get("hostname")):
            if fp in seen_fp:
                entry["duplicate_of"] = seen_fp[fp]
                break
            seen_fp[fp] = key
        out.append(entry)
    return out


def _scenes(rows: List[ContextEvidence], now: float) -> List[Dict[str, Any]]:
    """Latest node uplink per device → textual cues (scene line, per-
    sensor sentences + cues, predictions) PLUS the physical descriptors
    (fields mean/min/max, emitter scans) — the numeric evidence the
    model needs for judgements like "radar flat → room empty"."""
    latest: Dict[str, ContextEvidence] = {}
    for row in rows:
        dev = row.device_id or ""
        if dev not in latest or row.timestamp > latest[dev].timestamp:
            latest[dev] = row
    out = []
    for dev, row in sorted(latest.items(), key=lambda kv: -kv[1].timestamp):
        try:
            body = json.loads(row.value) if row.value else {}
        except (TypeError, ValueError):
            continue
        v = body.get("value") if isinstance(body.get("value"), dict) else body
        sensors = {}
        for sid, d in (v.get("sensors") or {}).items():
            if not isinstance(d, dict):
                continue
            entry: Dict[str, Any] = {"type": d.get("type"),
                                     "text": d.get("text")}
            for k in ("n", "rate_hz", "age_s", "state"):
                if d.get(k) is not None:
                    entry[k] = d[k]
            if d.get("fields"):
                entry["fields"] = _round(d["fields"])
            if d.get("scan"):
                entry["scan"] = _round(d["scan"])
            if d.get("cues"):
                entry["cues"] = _round(d["cues"])
            sensors[sid] = entry
        scene: Dict[str, Any] = {"device": dev or None,
                                 "age_s": round(now - row.timestamp, 1),
                                 "scene": v.get("scene"),
                                 "predictions": _round(
                                     v.get("predictions") or {}),
                                 "sensors": sensors}
        # Node-reported geo + site — how "this node is at <address>"
        # reaches the map. ``estimates`` carry the node's own
        # occupancy/zone/activity states verbatim.
        if v.get("location"):
            scene["location"] = _round(v["location"])
        if v.get("estimates"):
            scene["estimates"] = _round(
                [e for e in v["estimates"] if isinstance(e, dict)][:16])
        if v.get("room"):
            scene["room"] = _round(v["room"])
        out.append(scene)
    return out[:MAX_SCENES]


def build_bundle(db: Session, user_id: int, now: float,
                 window_s: float = DEFAULT_WINDOW_S) -> Dict[str, Any]:
    rows = db.query(ContextEvidence).filter(
        ContextEvidence.user_id == user_id,
        ContextEvidence.timestamp >= now - window_s,
        ~ContextEvidence.evidence_key.like("context.map.%"),
    ).order_by(ContextEvidence.timestamp.desc()).limit(MAX_EVIDENCE_ROWS).all()
    uplinks = [r for r in rows if r.evidence_key == DESCRIPTOR_KEY]
    aggregates = _aggregate([r for r in rows
                             if r.evidence_key != DESCRIPTOR_KEY])
    for a in aggregates:
        a["age_s"] = round(now - a.pop("last_ts"), 1)
    return {
        "now": now,
        "window_s": window_s,
        "evidence_rows": len(rows),
        "devices": _devices(db, user_id),
        "scenes": _scenes(uplinks, now),
        "descriptors": aggregates,
        "map": map_snapshot(db, user_id, now),
        "reference": {"radar_occupancy": RADAR_REFERENCE},
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
                            "name": {"type": "string",
                                     "description": "human name — set or "
                                                    "refine it whenever "
                                                    "metadata (MAC vendor, "
                                                    "hostname, beacon id) "
                                                    "identifies the thing "
                                                    "better than its id"},
                            "aliases": {"type": "array",
                                        "items": {"type": "string"}},
                            "attributes": {
                                "type": "object",
                                "description": "Kind-shaped facts. "
                                    "device: mac, vendor, hostname, "
                                    "inferred_model, role. person: "
                                    "address (home address), "
                                    "associated_devices (carried "
                                    "watch/phone/laptop entity ids). "
                                    "place: address (buildings), "
                                    "building (parent place id for "
                                    "rooms), rooms[] each "
                                    "{name, layout, devices[] each "
                                    "{ref, placement, orientation, "
                                    "manual}} — user-set "
                                    "placement/orientation is "
                                    "authoritative, carry it "
                                    "forward verbatim."},
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
You are the MAP BUILDER model in a Thoth deployment. The system: edge
nodes (the user's registered devices) sample radar, CSI, radio (BLE/
Wi-Fi/Zigbee), camera, microphone and IMU, compute physical sensor
descriptors on-device, and uplink them to Brain as evidence rows.
Brain aggregates those rows into this bundle once per build tick. Your
role: fuse that physical evidence into ONE stable semantic map of
persons, places, devices, activities and objects — the shared context
every downstream agent reads. You never see raw sensor data.

`devices` lists the user's REGISTERED nodes: `entity` is the map id
each scene's `device` uuid joins to, and name/type/mac/hardware say
which physical machine it is (e.g. which Pi runs radar+CSI, which
laptop runs camera+mic). Emitters and MACs inside sensor `scan`s that
match no registered device and no existing alias are UNKNOWN devices
— for persistent unknowns propose `device:`/`object:`/`person:`
entities (keep the raw MAC/beacon id in `aliases`). You may also refine
a known device: propose the same device:<uuid> id with a better `name`
or identifying `attributes` (hostname, MAC vendor, model) — the
registry remains the anchor, never create a second entity for it.

IDENTITY LAW — one entity per physical device. A MAC, hostname, uuid,
or beacon id is a FINGERPRINT: any emitter matching a fingerprint of an
existing entity (check `map` aliases and `devices` uuid/mac/hardware)
IS that entity — reuse its id, add the new identifier to `aliases`,
and fold richer metadata into `attributes` (vendor from the MAC OUI,
model/role from advertised name + observed traffic). Never propose a
second entity for hardware already on the map; never create
`device:<mac-slug>` beside a `device:<uuid>` carrying that MAC. When
two map entities are provably one device, use `merges` — never let
duplicates stand. Refine `name` whenever metadata identifies the thing
better ("b8:27:eb" → "thoth-april — living-room Pi3").

PLACES — a `place:` with an `address` is a building; `place:`s inside
it (rooms) carry `building` (the parent's id) plus `layout` and
`devices` (each {ref, placement, orientation, manual}) — user-set
placement/orientation entries are authoritative, preserve them
verbatim. Current `location.geo.v1`/geo evidence versus the place
address decides where things ARE: a device/person whose geo sits
outside every known address is `located_in` a different/unknown place
(on-street/away), and inside the address, radar/CSI/BLE evidence
narrows it to a specific room.

PERSONS — the account always owns at least `person:owner` (the user,
pre-seeded) and `place:home` (their address). Track who the carried
devices belong to via `associated_devices` (watch, phone, laptop), and
keep `presence.v1`/`location.v1` states + `located_in` edges current
for every person the evidence supports.

You get:
  * `scenes` — the latest uplink per node with TEXTUAL cues already
    computed on the device: a one-line `scene`, a sentence per sensor
    (`text`) and structured `cues` (speech transcript, people count,
    recognized face identity, motion level, strongest radio emitter),
    plus on-device `predictions`. Each sensor entry ALSO carries its
    physical `fields` (per-field mean/min/max/n — e.g. radar `snr_db`,
    `range`, CSI amplitude variance, RSSI, audio level) with `n`,
    `rate_hz` and `age_s`, plus the node's `estimates` (its own
    occupancy/zone/activity verdicts) and `location` (self-resolved geo
    + user-set `site` address + `room`). Prefer these — they are the
    most direct evidence.
  * `descriptors` — compact physical aggregates per (evidence key,
    device): counts, field mean/min/max, latest value, mean confidence,
    age. `location.geo.v1` entries describe where a node is (lat/lon,
    postal_code, city, site).
  * `devices` — the device registry (see above); `map` — the current
    semantic map snapshot you are updating; `reference` — measured
    empty-vs-occupied stats for the radar/CSI fields (E1 rig): treat as
    the default prior — absolute map means are mount-dependent, so weigh
    a node's lift vs ITS OWN empty baseline; the transferable shape is
    occupancy-in-the-tail (snr spikes + std90 spread + map-mean lifts),
    not a higher median.
Typical keys: occupancy/presence probabilities, radar SNR and range,
CSI amplitude variance, BLE/Wi-Fi RSSI sightings (with decoded beacon
identities), IMU motion variance, face/person detections, audio level,
model predictions (key 'prediction'), device metadata/room placement.

A `site` on a node's location is a user-set address — anchor it: create
or reuse a `place:` entity for the site (e.g. place:home named after
the address), relate the reporting `device:` with `located_in`, and
relate other persons/devices the node observes there when the evidence
supports it. When radar/CSI fields are flat — snr_db.max under ~13 dB,
tight range profile, near-zero CSI variance, zero emitters — while the
node is fresh and reporting, that is positive evidence the space is
EMPTY — set occupancy.v1 {"occupied": false} on the place with matching
confidence. (Reference: on the E1 rig an empty room reads snr mean
~7.6 dB, p95 <8; occupancy appears as window snr_max spikes >15-40 dB
and ra/re/xy map lifts — see `reference.radar_occupancy`.)

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
  * Map entries marked `confirmed: true` are user-confirmed facts — do
    not rename, move, contradict, merge or retire them; your proposals
    for them are dropped automatically.
  * Only propose merges for clear duplicates and retire only entities the
    evidence contradicts.
"""


def _llm_update(bundle: Dict[str, Any], *,
                usage_out: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Forced tool call — returns the parsed proposal.

    ``usage_out``, when supplied, is filled with metering fields for the
    inference ledger."""
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
    if usage_out is not None:
        usage_out["model_id"] = model
        usage_out["tokens"] = getattr(
            getattr(resp, "usage", None), "total_tokens", None)
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

    def save(self, summary: str, now: float, *,
             bundle: Optional[Dict[str, Any]] = None,
             proposal: Optional[Dict[str, Any]] = None) -> None:
        self.data["builds"] = int(self.data.get("builds", 0)) + 1
        self.data["last_build_at"] = now
        self.data["last_summary"] = summary
        # Audit trail — what the model saw and what it produced.
        if bundle is not None:
            self.data["last_input"] = json.dumps(
                bundle, default=str)[:INPUT_KEEP]
        if proposal is not None:
            self.data["last_proposal"] = json.dumps(
                proposal, default=str)[:PROPOSAL_KEEP]
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
    if entity_is_confirmed(ent):
        # User-confirmed entity — the proposal may refresh presence and
        # add aliases (identification aids) but must not rename it or
        # mutate the confirmed attributes.
        aliases = set(attrs.get("aliases", [])) | set(
            raw.get("aliases") or [])
        aliases.discard(key)
        attrs["aliases"] = sorted(str(a) for a in aliases if a)
        attrs["last_seen"] = now
        ent.retired_at = None
        ent.attributes = json.dumps(attrs, default=str)
        db.flush()
        return ent
    aliases = set(attrs.get("aliases", [])) | set(raw.get("aliases") or [])
    aliases.discard(key)
    extra = raw.get("attributes") or {}
    if isinstance(extra, dict):
        for k, v in extra.items():
            if k not in ("aliases", "registry", "pending"):
                attrs[k] = v
        # Hardware identifiers become aliases — any later proposal naming
        # the same MAC/hostname/uuid resolves onto THIS entity instead of
        # forking a duplicate.
        if ent.kind == "device":
            for k in _FP_ATTR_KEYS:
                v = extra.get(k)
                if isinstance(v, str) and v:
                    aliases.add(v)
                    mac = _norm_mac(v)
                    if mac:
                        aliases.add(mac)
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
                     now: Optional[float] = None, *,
                     bundle: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Apply a proposed update with alias resolution, hysteresis and
    decay. Returns a receipt of what changed and what is pending.
    ``bundle`` is retained on the builder entity alongside the proposal
    so both sides of the last build stay auditable."""
    now = now or time.time()
    mem = _Memory(db, user_id)
    receipt: Dict[str, Any] = {"entities": [], "merged": [], "retired": [],
                               "relationships": {"created": [], "refreshed": [],
                                                 "ended": [], "pending": []},
                               "states": {"applied": [], "pending": []},
                               "skipped": []}

    entities = _live_entities(db, user_id)
    idx = _alias_index(entities)
    fp_idx = _device_fp_index(db, user_id)

    # 1. entities — resolve proposals onto existing ids first
    for raw in update.get("entities") or []:
        if float(raw.get("confidence") or 0) < MIN_CONFIDENCE:
            receipt["skipped"].append({"entity": raw.get("id"),
                                       "reason": "low confidence"})
            continue
        raw = dict(raw)
        attrs = raw.get("attributes") if isinstance(
            raw.get("attributes"), dict) else {}
        # Device fingerprints outrank plain alias resolution: a proposal
        # that names the same MAC/hostname/uuid as an existing device IS
        # that device, whatever id slug the model invented. Device-only —
        # a person/place must never collapse onto hardware.
        fp_hit = None
        if raw.get("kind") == "device" or _norm_mac(str(raw.get("id"))):
            fp_hit = _resolve_device(
                fp_idx, raw.get("id"), *(raw.get("aliases") or []),
                *(attrs.get(k) for k in _FP_ATTR_KEYS))
        resolved = fp_hit or _resolve(idx, raw.get("id"))
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
        if (entity_is_confirmed(a) or entity_is_confirmed(b)
                or any(rel_is_confirmed(r)
                       for r in _active_relationships(db, user_id, now)
                       if src in (r.subject, r.object))):
            # Merging retires the source and rewires its relationships —
            # never touch a user-confirmed entity or a confirmed edge.
            receipt["skipped"].append({"merge": [src, dst],
                                       "reason": "confirmed_fact"})
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
            if rel_is_confirmed(same):
                # Re-asserting a user-confirmed edge — dedupe without
                # mutating the confirmed row.
                mem.clear(slot)
                receipt["relationships"]["refreshed"].append(same.id)
                continue
            same.confidence = round(0.7 * (same.confidence or 0) + 0.3 * conf, 3)
            prov = json.loads(same.provenance) if same.provenance else {}
            prov["confirmed_at"] = now
            same.provenance = json.dumps(prov)
            mem.clear(slot)
            receipt["relationships"]["refreshed"].append(same.id)
            continue
        rivals = ([r for r in active if r.subject == s and r.predicate == p]
                  if p in EXCLUSIVE_PREDICATES else [])
        if any(rel_is_confirmed(r) for r in rivals):
            # A confirmed edge holds this subject+predicate — a builder
            # proposal cannot displace it, hysteresis or not.
            receipt["skipped"].append(
                {"relationship": [s, p, o], "reason": "confirmed_fact"})
            continue
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
        if (prev is not None
                and is_confirmed_source(prev.estimator or "")):
            # Live confirmed state — builder proposals can't touch it
            # (same rule apply_state enforces for the form path; skip
            # before the hysteresis branch would extend its validity).
            receipt["skipped"].append(
                {"state": key, "entity_id": ent or None,
                 "reason": "confirmed_fact"})
            continue
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
        try:
            st = apply_state(db, user_id, StateIn(
                key=key, value=value, entity_id=ent, confidence=conf,
                valid_until=now + STATE_TTL_S, estimator=ESTIMATOR))
        except HTTPException as exc:
            # e.g. a confirmed state written between snapshot and apply —
            # record the block instead of aborting the whole build.
            receipt["skipped"].append(
                {"state": key, "entity_id": ent or None,
                 "reason": "confirmed_fact" if exc.status_code == 409
                 else f"http_{exc.status_code}"})
            continue
        if ent:
            seen_entities.append(ent)
        receipt["states"]["applied"].append(st.to_dict())

    # 5. explicit retirements — never registry devices or confirmed facts
    for raw in update.get("retire") or []:
        key = _resolve(idx, raw.get("id"))
        ent = live.get(key)
        if (ent is None or ent.kind == "device" or key in seen_entities
                or entity_is_confirmed(ent)):
            receipt["skipped"].append({"retire": key,
                                       "reason": "protected or seen"
                                       if not (ent is not None
                                               and entity_is_confirmed(ent))
                                       else "confirmed_fact"})
            continue
        ent.retired_at = now
        live.pop(key, None)
        receipt["retired"].append(key)

    _touch(db, user_id, seen_entities, now)
    db.commit()

    # 6. decay
    receipt["decayed"] = decay(db, user_id, now)
    mem.save(str(update.get("summary") or ""), now,
             bundle=bundle, proposal=update)
    receipt["summary"] = update.get("summary")
    return receipt


def decay(db: Session, user_id: int, now: float) -> Dict[str, List[Any]]:
    ended, retired = [], []
    for rel in _active_relationships(db, user_id, now):
        if rel.source != SOURCE or rel_is_confirmed(rel):
            continue
        prov = json.loads(rel.provenance) if rel.provenance else {}
        if now - float(prov.get("confirmed_at") or rel.valid_from) > REL_TTL_S:
            rel.valid_until = now
            ended.append(rel.id)
    for ent in _live_entities(db, user_id):
        if entity_is_confirmed(ent):
            continue
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
    if llm is None:
        # Hosted call — authorized + metered like every inference path
        # (an injected ``llm`` is local and bypasses metering).
        with inference_call(db, user_id,
                            kind="context_build") as meter:
            update = _llm_update(bundle, usage_out=meter)
    else:
        update = llm(bundle)
    result: Dict[str, Any] = {"generated_at": now, "dry_run": dry_run,
                              "evidence_rows": bundle["evidence_rows"],
                              "descriptor_groups": len(bundle["descriptors"]),
                              "proposal": update}
    if not dry_run:
        result["receipt"] = apply_map_update(db, user_id, update, now,
                                             bundle=bundle)
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
                    "last_input": battrs.get("last_input"),
                    "last_proposal": battrs.get("last_proposal"),
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
