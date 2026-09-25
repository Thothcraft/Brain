"""/v1/subscriptions — outbound webhook endpoints for node events.

Each subscription POSTs matching ``node_event`` rows to ``url`` as they
land (via ``event_delivery.enqueue`` on ``_write_event``). Bodies are
signed ``X-Thoth-Signature: sha256=<hmac(secret, body)>`` when a secret
is set; deliveries retry with backoff and are inspectable via
``GET /v1/subscriptions/{id}/deliveries``.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import (
    EventDelivery, EventSubscription, User, get_db,
)

router = APIRouter(prefix="/subscriptions", tags=["v1", "subscriptions"])


class SubscriptionIn(BaseModel):
    url: str
    secret: Optional[str] = None
    kinds: Optional[List[str]] = None
    device_id: Optional[str] = None
    enabled: bool = True


def _owned(sub_id: int, user: User, db: Session) -> EventSubscription:
    sub = db.query(EventSubscription).filter(
        EventSubscription.id == sub_id,
        EventSubscription.user_id == user.userId).first()
    if sub is None:
        raise HTTPException(status_code=404, detail="subscription not found")
    return sub


@router.get("")
async def list_subscriptions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rows = db.query(EventSubscription).filter(
        EventSubscription.user_id == current_user.userId).all()
    return {"subscriptions": [r.to_dict() for r in rows]}


@router.post("", status_code=201)
async def create_subscription(
    body: SubscriptionIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422,
                            detail="url must be http(s)")
    sub = EventSubscription(
        user_id=current_user.userId,
        url=body.url[:1024],
        secret=body.secret or secrets.token_urlsafe(24),
        kinds=json.dumps(body.kinds or []),
        device_id=body.device_id,
        enabled=body.enabled)
    db.add(sub)
    db.commit()
    db.refresh(sub)
    out = sub.to_dict()
    out["secret"] = sub.secret   # show once at creation
    return {"subscription": out}


@router.delete("/{sub_id}")
async def delete_subscription(
    sub_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sub = _owned(sub_id, current_user, db)
    db.delete(sub)
    db.commit()
    return {"ok": True}


@router.put("/{sub_id}")
async def update_subscription(
    sub_id: int,
    body: SubscriptionIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sub = _owned(sub_id, current_user, db)
    sub.url = body.url[:1024]
    if body.secret:
        sub.secret = body.secret
    sub.kinds = json.dumps(body.kinds or [])
    sub.device_id = body.device_id
    sub.enabled = body.enabled
    db.commit()
    return {"subscription": sub.to_dict()}


@router.get("/{sub_id}/deliveries")
async def list_deliveries(
    sub_id: int,
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sub = _owned(sub_id, current_user, db)
    rows = db.query(EventDelivery).filter(
        EventDelivery.subscription_id == sub.id).order_by(
        EventDelivery.id.desc()).limit(limit).all()
    return {"deliveries": [r.to_dict() for r in rows]}
