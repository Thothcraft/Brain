"""Brain v1 automation API — declarative rules evaluated server-side.

Automation lives in Brain (reasoning/orchestration layer): rules keep
firing while control surfaces are offline. ``POST /v1/automation/evaluate``
runs the caller's enabled rules against the current context snapshot —
the same non-expired states ``/v1/context/snapshot`` serves.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.automation import AutomationEngine
from server.db import (
    AutomationRule, ContextState, User, get_db,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/automation", tags=["v1", "automation"])


class RuleIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    when: Dict[str, Any]
    then: Dict[str, Any]
    cooldown_s: float = Field(default=0.0, ge=0.0)
    enabled: bool = True


def _current_states(db: Session, user_id: int) -> List[Dict[str, Any]]:
    """Non-expired states — identical semantics to /v1/context/snapshot."""
    now = time.time()
    rows = db.query(ContextState).filter(
        ContextState.user_id == user_id,
        (ContextState.valid_until.is_(None)) |
        (ContextState.valid_until > now)).all()
    return [s.to_dict() for s in rows]


@router.get("/rules")
async def list_rules(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rows = db.query(AutomationRule).filter(
        AutomationRule.user_id == current_user.userId).order_by(
        AutomationRule.name).all()
    return {"rules": [r.to_dict() for r in rows]}


@router.post("/rules", status_code=201)
async def upsert_rule(
    body: RuleIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not isinstance(body.when, dict) or not body.when.get("key"):
        raise HTTPException(status_code=422,
                            detail="when{} needs at least a state key")
    if not isinstance(body.then, dict) or not body.then.get("actuator_id"):
        raise HTTPException(status_code=422,
                            detail="then{} needs an actuator_id")
    rule = db.query(AutomationRule).filter(
        AutomationRule.user_id == current_user.userId,
        AutomationRule.name == body.name).first()
    if rule is None:
        rule = AutomationRule(user_id=current_user.userId, name=body.name)
        db.add(rule)
    rule.when = json.dumps(body.when)
    rule.then = json.dumps(body.then)
    rule.cooldown_s = body.cooldown_s
    rule.enabled = body.enabled
    db.commit()
    db.refresh(rule)
    return rule.to_dict()


@router.delete("/rules/{name}")
async def delete_rule(
    name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rule = db.query(AutomationRule).filter(
        AutomationRule.user_id == current_user.userId,
        AutomationRule.name == name).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    db.delete(rule)
    db.commit()
    return {"ok": True, "name": name}


@router.post("/evaluate")
async def evaluate_rules(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Evaluate the caller's enabled rules against current context.

    Actuation dispatch is record-only for now — results are returned as
    queued ActionResults; wiring to device local APIs is a follow-up.
    """
    rows = db.query(AutomationRule).filter(
        AutomationRule.user_id == current_user.userId,
        AutomationRule.enabled == True).all()  # noqa: E712
    engine = AutomationEngine()
    for r in rows:
        engine.add_rule({
            "name": r.name,
            "when": json.loads(r.when) if r.when else {},
            "then": json.loads(r.then) if r.then else {},
            "cooldown_s": r.cooldown_s or 0.0})
    fired = engine.evaluate({"states": _current_states(db,
                                                       current_user.userId)})
    return {"fired": fired, "evaluated_rules": len(rows),
            "evaluated_at": time.time()}
