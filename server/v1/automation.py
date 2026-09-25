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
from server.automation import PersistentAutomation
from server.actuation import dispatch_action, retry_queued
from server.db import (
    ActionRequest, AutomationExecution, AutomationRule, ContextState,
    User, get_db,
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
    """Evaluate the caller's enabled rules against current context
    through the persistent runtime — durable rule state, recorded
    executions, and real dispatch to Thoth nodes. The context/event path
    invokes the same machinery; this endpoint exists for inspection.
    """
    runtime = PersistentAutomation(db, current_user.userId)
    fired = runtime.evaluate(
        {"states": _current_states(db, current_user.userId)})
    return {"fired": fired,
            "evaluated_rules": len(runtime.engine._rules),
            "evaluated_at": time.time()}


@router.get("/executions")
async def list_executions(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Rule-firing history — each execution links the triggering context
    to the actions it dispatched (provenance chain)."""
    rows = db.query(AutomationExecution).filter(
        AutomationExecution.user_id == current_user.userId
    ).order_by(AutomationExecution.created_at.desc()).limit(limit).all()
    return {"executions": [r.to_dict() for r in rows]}


@router.post("/retry")
async def retry_queued_actions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Dispatch every queued, unexpired action — bounded, never silent
    infinite retry."""
    return {"attempted": retry_queued(db, current_user.userId)}


# ---------------------------------------------------------------------------
# Action requests — canonical ActionRequest/ActionResult history
# ---------------------------------------------------------------------------

@router.get("/actions")
async def list_actions(
    limit: int = 100,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(ActionRequest).filter(
        ActionRequest.user_id == current_user.userId)
    if status:
        q = q.filter(ActionRequest.status == status)
    rows = q.order_by(ActionRequest.created_at.desc()).limit(limit).all()
    return {"actions": [r.to_dict() for r in rows]}


@router.get("/actions/{action_id}")
async def get_action(
    action_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    row = db.query(ActionRequest).filter(
        ActionRequest.action_id == action_id,
        ActionRequest.user_id == current_user.userId).first()
    if not row:
        raise HTTPException(status_code=404, detail="Action not found")
    return row.to_dict()


@router.post("/actions/{action_id}/dispatch")
async def redispatch_action(
    action_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retry dispatch of a queued action — idempotent on ``action_id``."""
    row = db.query(ActionRequest).filter(
        ActionRequest.action_id == action_id,
        ActionRequest.user_id == current_user.userId).first()
    if not row:
        raise HTTPException(status_code=404, detail="Action not found")
    dispatch_action(db, row)
    return row.to_dict()
