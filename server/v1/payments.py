"""omi-compatible plans, payments and usage routes (D-09).

Shapes follow omi's ``backend/routers/payment.py`` / ``users.py`` so the
forked omi apps' paywall and usage screens work against Brain:

    GET    /v1/payments/available-plans
    POST   /v1/payments/checkout-session        {price_id}
    POST   /v1/payments/billing-portal
    DELETE /v1/payments/subscription            cancel at period end
    POST   /v1/stripe/webhook                   (same handler as /api/stripe/webhook)
    GET    /v1/users/me/subscription
    GET    /v1/users/me/usage
    GET    /v1/entitlements/me                  user or Room1 device token
    POST   /v1/usage/report                     Room1 device token, counters only
    POST   /v1/auth/firebase                    Firebase ID token → Brain session

omi encodes "unlimited" as ``0`` in ``*_limit`` integer fields; the same
sentinel is used here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server import plans
from server.auth import create_access_token, get_current_user
from server.db import User, get_db
from server.endpoints import stripe_endpoints

logger = logging.getLogger(__name__)
router = APIRouter(tags=["v1", "payments"])


def _stripe():
    if not stripe_endpoints.STRIPE_AVAILABLE or not os.getenv("STRIPE_SECRET_KEY"):
        raise HTTPException(status_code=503, detail="Payment service not available")
    return stripe_endpoints.stripe


def _limit(value: Optional[float]) -> int:
    return 0 if value is None else int(value)


# ---------------------------------------------------------------------------
# Plans and checkout
# ---------------------------------------------------------------------------

@router.get("/payments/available-plans")
async def available_plans(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {"plans": plans.available_plans(current_user),
            "current_plan": plans.plan_type(current_user)}


class CheckoutIn(BaseModel):
    price_id: str


@router.post("/payments/checkout-session")
async def checkout_session(
    body: CheckoutIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    stripe = _stripe()
    if body.price_id not in plans.sellable_price_ids():
        raise HTTPException(status_code=400, detail="This plan is not available for purchase")
    if current_user.stripe_subscription_id:
        raise HTTPException(
            status_code=409,
            detail="You already have a subscription. Manage it in the billing portal.")

    def create() -> str:
        if not current_user.stripe_customer_id:
            params: Dict[str, Any] = {"metadata": {"user_id": current_user.userId}}
            if current_user.email:
                params["email"] = current_user.email
            current_user.stripe_customer_id = stripe.Customer.create(**params).id
            db.commit()
        session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            line_items=[{"price": body.price_id, "quantity": 1}],
            mode="subscription",
            allow_promotion_codes=True,
            automatic_tax={"enabled": True},
            customer_update={"address": "auto"},
            success_url=os.getenv("STRIPE_SUCCESS_URL", "https://thothcraft.com/settings?success=true"),
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "https://thothcraft.com/settings?cancel=true"),
            metadata={"plan": "pro", "user_id": current_user.userId},
        )
        return session.url

    try:
        url = await asyncio.to_thread(create)
    except HTTPException:
        raise
    except Exception:
        logger.exception("checkout session failed for user %s", current_user.userId)
        raise HTTPException(status_code=502, detail="Unable to start checkout")
    return {"url": url}


@router.post("/payments/billing-portal")
async def billing_portal(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    stripe = _stripe()
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account for this user")
    session = await asyncio.to_thread(
        stripe.billing_portal.Session.create,
        customer=current_user.stripe_customer_id,
        return_url=os.getenv("STRIPE_PORTAL_RETURN_URL", "https://thothcraft.com/settings"),
    )
    return {"url": session.url}


@router.delete("/payments/subscription")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    stripe = _stripe()
    if not current_user.stripe_subscription_id:
        raise HTTPException(status_code=404, detail="No active subscription")
    sub = await asyncio.to_thread(
        stripe.Subscription.modify, current_user.stripe_subscription_id,
        cancel_at_period_end=True)
    return {"status": "ok", "cancel_at_period_end": True,
            "current_period_end": getattr(sub, "current_period_end", None)}


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    return await stripe_endpoints.stripe_webhook(request)


# ---------------------------------------------------------------------------
# Subscription and usage (omi shapes)
# ---------------------------------------------------------------------------

def _period_end(user: User) -> Optional[int]:
    value = getattr(user, "plan_expires_at", None)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value.timestamp())
    except Exception:
        return None


@router.get("/users/me/subscription")
async def my_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    pid = plans.plan_type(current_user)
    spec = plans.CATALOG[pid]
    alloc = plans.allocations(db, current_user)
    chat = alloc["chat_questions"]
    return {
        "subscription": {
            "plan": pid,
            "status": "active",
            "current_period_end": _period_end(current_user),
            "stripe_subscription_id": current_user.stripe_subscription_id,
            "current_price_id": None,
            "features": spec["features"],
            "cancel_at_period_end": False,
            "limits": {
                "transcription_seconds": spec["allocations"]["transcription_seconds"],
                "chat_questions_per_month": spec["allocations"]["chat_questions"],
            },
            "deprecated": spec["lifecycle"] == "deprecated",
        },
        "transcription_seconds_used": int(alloc["transcription_seconds"]["used"]),
        "transcription_seconds_limit": _limit(alloc["transcription_seconds"]["limit"]),
        "words_transcribed_used": int(alloc["words_transcribed"]["used"]),
        "words_transcribed_limit": _limit(alloc["words_transcribed"]["limit"]),
        "insights_gained_used": int(alloc["memories_created"]["used"]),
        "insights_gained_limit": _limit(alloc["memories_created"]["limit"]),
        "available_plans": plans.available_plans(current_user),
        "show_subscription_ui": True,
        "trial_eligible": pid == "basic" and not current_user.stripe_customer_id,
        "chat_usage_quota": {
            "plan": spec["display_name"],
            "plan_type": pid,
            "unit": "questions",
            "used": chat["used"],
            "limit": chat["limit"],
            "percent": chat["percent"],
            "allowed": chat["allowed"],
            "reset_at": plans.period_reset_at(),
            "is_overage_plan": False,
        },
    }


@router.get("/users/me/usage")
async def my_usage(
    period: str = "monthly",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    used = plans.usage(db, current_user.userId)
    stats = {
        "transcription_seconds": int(used["transcription_seconds"]),
        "words_transcribed": int(used["words_transcribed"]),
        "insights_gained": int(used["memories_created"]),
        "memories_created": int(used["memories_created"]),
        "chat_questions": int(used["chat_questions"]),
    }
    return {"monthly": stats, "period": plans.current_period()}


# ---------------------------------------------------------------------------
# Room1 ↔ Brain: entitlement cache + counter reports
# ---------------------------------------------------------------------------

def _caller(request: Request, db: Session):
    from server.endpoints.node_ws import _principal
    kind, principal, device = _principal(request, db)
    if kind == "device":
        user = db.query(User).filter(User.userId == device.userId).first()
        if not user:
            raise HTTPException(status_code=404, detail="Device owner not found")
        return user, device
    return principal, None


@router.get("/entitlements/me")
async def my_entitlements(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    user, _device = _caller(request, db)
    return plans.entitlement_snapshot(db, user)


class UsageReportIn(BaseModel):
    period: str = Field(pattern=r"^\d{4}-\d{2}$")
    seq: int = Field(ge=0)
    transcription_seconds: float = Field(default=0, ge=0, le=31 * 24 * 3600)
    chat_questions: float = Field(default=0, ge=0, le=100000)
    words_transcribed: float = Field(default=0, ge=0)
    memories_created: float = Field(default=0, ge=0)


@router.post("/usage/report")
async def usage_report(
    body: UsageReportIn,
    request: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    user, device = _caller(request, db)
    if device is None:
        raise HTTPException(status_code=403, detail="Usage reports require a device credential")
    applied = plans.record_report(
        db, user.userId, device.device_uuid, body.period, body.seq,
        {"transcription_seconds": body.transcription_seconds,
         "chat_questions": body.chat_questions,
         "words_transcribed": body.words_transcribed,
         "memories_created": body.memories_created})
    return {"applied": applied, **plans.entitlement_snapshot(db, user)}


# ---------------------------------------------------------------------------
# Firebase sign-in exchange
# ---------------------------------------------------------------------------

class FirebaseIn(BaseModel):
    id_token: str


@router.post("/auth/firebase")
async def firebase_exchange(body: FirebaseIn, db: Session = Depends(get_db)) -> Dict[str, Any]:
    from server.firebase_auth import firebase_project_id, resolve_firebase_user, verify_firebase_token
    if not firebase_project_id():
        raise HTTPException(status_code=503, detail="Firebase sign-in is not configured")
    claims = verify_firebase_token(body.id_token)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    user = resolve_firebase_user(db, claims)
    token = create_access_token(
        data={"sub": str(user.userId), "username": user.username, "email": user.email},
        expires_delta=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")))
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user.userId, "username": user.username, "email": user.email,
                     "plan": plans.plan_type(user)}}
