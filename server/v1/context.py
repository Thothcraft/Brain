"""Brain v1 context API — entities, relationships, evidence, state, events.

The context model (Architecture §30–§35): entities are canonical objects,
relationships are subject–predicate–object edges with validity windows,
evidence wraps observations/predictions with provenance (predictions are
evidence, never truth), states are derived statements with evidence links,
and events are the discrete transitions emitted when a state changes.

All rows are user-scoped — tenant isolation everywhere.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import (
    ContextEntity, ContextEvent, ContextEvidence, ContextRelationship,
    ContextState, User, get_db,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["v1", "context"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EntityIn(BaseModel):
    id: str = Field(min_length=1, max_length=255)          # e.g. "person:gad"
    kind: str = Field(min_length=1, max_length=80)
    name: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class RelationshipIn(BaseModel):
    subject: str
    predicate: str
    object: str
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    confidence: float = 1.0
    source: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)


class EvidenceIn(BaseModel):
    key: str                                               # versioned key
    value: Any = None
    timestamp: Optional[float] = None
    source_id: str = ""
    device_id: str = ""
    prediction_id: str = ""
    observation_id: str = ""
    model_id: str = ""
    model_version: str = ""
    confidence: Optional[float] = None
    execution_class: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)


class StateIn(BaseModel):
    key: str
    value: Any = None
    entity_id: str = ""
    confidence: float = 1.0
    since: Optional[float] = None
    valid_until: Optional[float] = None
    evidence_ids: List[str] = Field(default_factory=list)
    estimator: str = ""


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@router.get("/entities")
async def list_entities(
    kind: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ContextEntity).filter(
        ContextEntity.user_id == current_user.userId)
    if kind:
        q = q.filter(ContextEntity.kind == kind)
    return {"entities": [e.to_dict() for e in
                         q.order_by(ContextEntity.entity_key).all()]}


@router.post("/entities", status_code=201)
async def upsert_entity(
    body: EntityIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    entity = db.query(ContextEntity).filter(
        ContextEntity.user_id == current_user.userId,
        ContextEntity.entity_key == body.id).first()
    if entity is None:
        entity = ContextEntity(user_id=current_user.userId,
                               entity_key=body.id, kind=body.kind)
        db.add(entity)
    entity.name = body.name or entity.name
    entity.attributes = json.dumps(body.attributes or {})
    db.commit()
    db.refresh(entity)
    return entity.to_dict()


@router.delete("/entities/{entity_key}")
async def delete_entity(
    entity_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    entity = db.query(ContextEntity).filter(
        ContextEntity.user_id == current_user.userId,
        ContextEntity.entity_key == entity_key).first()
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    db.delete(entity)
    db.commit()
    return {"ok": True, "id": entity_key}


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

@router.get("/relationships")
async def list_relationships(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    active_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ContextRelationship).filter(
        ContextRelationship.user_id == current_user.userId)
    if subject:
        q = q.filter(ContextRelationship.subject == subject)
    if predicate:
        q = q.filter(ContextRelationship.predicate == predicate)
    if active_only:
        q = q.filter(ContextRelationship.valid_until.is_(None))
    return {"relationships": [r.to_dict() for r in
                              q.order_by(ContextRelationship.id).all()]}


@router.post("/relationships", status_code=201)
async def create_relationship(
    body: RelationshipIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rel = ContextRelationship(
        user_id=current_user.userId,
        subject=body.subject, predicate=body.predicate, object=body.object,
        valid_from=body.valid_from or time.time(),
        valid_until=body.valid_until,
        confidence=body.confidence, source=body.source,
        provenance=json.dumps(body.provenance or {}))
    db.add(rel)
    db.commit()
    db.refresh(rel)
    return rel.to_dict()


@router.delete("/relationships/{rel_id}")
async def end_relationship(
    rel_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """End a relationship (sets valid_until) — history is preserved."""
    rel = db.query(ContextRelationship).filter(
        ContextRelationship.id == rel_id,
        ContextRelationship.user_id == current_user.userId).first()
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    rel.valid_until = time.time()
    db.commit()
    return {"ok": True, "id": str(rel_id), "valid_until": rel.valid_until}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@router.post("/evidence", status_code=201)
async def ingest_evidence(
    body: Any = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Ingest one evidence item or a batch (``{"items": [...]}``)."""
    items = body.get("items") if isinstance(body, dict) else None
    if items is None:
        items = [body]
    out = []
    for raw in items:
        ev = EvidenceIn(**raw)
        row = ContextEvidence(
            user_id=current_user.userId,
            evidence_key=ev.key,
            value=json.dumps(ev.value),
            timestamp=ev.timestamp or time.time(),
            source_id=ev.source_id or None,
            device_id=ev.device_id or None,
            prediction_id=ev.prediction_id or None,
            observation_id=ev.observation_id or None,
            model_id=ev.model_id or None,
            model_version=ev.model_version or None,
            confidence=ev.confidence,
            execution_class=ev.execution_class or None,
            provenance=json.dumps(ev.provenance or {}))
        db.add(row)
        out.append(row)
    db.commit()
    for row in out:
        db.refresh(row)
    return {"evidence": [r.to_dict() for r in out]}


@router.get("/evidence")
async def list_evidence(
    key: Optional[str] = None,
    source_id: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = Query(default=200, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ContextEvidence).filter(
        ContextEvidence.user_id == current_user.userId)
    if key:
        q = q.filter(ContextEvidence.evidence_key == key)
    if source_id:
        q = q.filter(ContextEvidence.source_id == source_id)
    if since:
        q = q.filter(ContextEvidence.timestamp >= since)
    rows = q.order_by(ContextEvidence.timestamp.desc()).limit(limit).all()
    return {"evidence": [r.to_dict() for r in rows]}


# ---------------------------------------------------------------------------
# State + events
# ---------------------------------------------------------------------------

@router.get("/state")
async def get_state(
    key: Optional[str] = None,
    entity_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ContextState).filter(
        ContextState.user_id == current_user.userId)
    if key:
        q = q.filter(ContextState.state_key == key)
    if entity_id:
        q = q.filter(ContextState.entity_id == entity_id)
    return {"states": [s.to_dict() for s in
                       q.order_by(ContextState.state_key).all()]}


@router.post("/state")
async def upsert_state(
    body: StateIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Upsert a derived state; emits a ContextEvent on value change."""
    now = time.time()
    state = db.query(ContextState).filter(
        ContextState.user_id == current_user.userId,
        ContextState.state_key == body.key,
        ContextState.entity_id == (body.entity_id or None)).first()
    previous = None
    event_type = "changed"
    if state is None:
        state = ContextState(
            user_id=current_user.userId, state_key=body.key,
            entity_id=body.entity_id or None, since=body.since or now)
        db.add(state)
        event_type = "entered"
    else:
        previous = json.loads(state.value) if state.value else None
        if previous == body.value:
            event_type = ""  # no transition
        elif previous in (None, "absent", "vacant", "empty", "off") \
                and body.value not in (None, "absent", "vacant", "empty", "off"):
            event_type = "entered"
        elif body.value in (None, "absent", "vacant", "empty", "off"):
            event_type = "exited"
    state.value = json.dumps(body.value)
    state.confidence = body.confidence
    state.valid_until = body.valid_until
    state.evidence_ids = json.dumps(body.evidence_ids or [])
    state.estimator = body.estimator or state.estimator
    db.commit()
    db.refresh(state)

    if event_type:
        event = ContextEvent(
            user_id=current_user.userId, event_key=body.key,
            event_type=event_type, entity_id=body.entity_id or None,
            state_id=str(state.id), value=json.dumps(body.value),
            previous_value=json.dumps(previous),
            confidence=body.confidence, timestamp=now,
            provenance=json.dumps({"estimator": body.estimator}))
        db.add(event)
        db.commit()
    return state.to_dict()


@router.get("/events")
async def list_events(
    key: Optional[str] = None,
    since: Optional[float] = None,
    limit: int = Query(default=200, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ContextEvent).filter(
        ContextEvent.user_id == current_user.userId)
    if key:
        q = q.filter(ContextEvent.event_key == key)
    if since:
        q = q.filter(ContextEvent.timestamp >= since)
    rows = q.order_by(ContextEvent.timestamp.desc()).limit(limit).all()
    return {"events": [r.to_dict() for r in rows]}


@router.get("/snapshot")
async def context_snapshot(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Full context snapshot: entities + active relationships + states."""
    uid = current_user.userId
    entities = db.query(ContextEntity).filter(
        ContextEntity.user_id == uid).all()
    rels = db.query(ContextRelationship).filter(
        ContextRelationship.user_id == uid,
        ContextRelationship.valid_until.is_(None)).all()
    states = db.query(ContextState).filter(
        ContextState.user_id == uid).all()
    return {
        "entities": [e.to_dict() for e in entities],
        "relationships": [r.to_dict() for r in rels],
        "states": [s.to_dict() for s in states],
        "generated_at": time.time(),
    }
