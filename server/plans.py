"""omi-style plan catalog, allocations and usage ledger for Thothcraft.

Mirrors omi's ``backend/config/plan_catalog.json`` model (D-09):

* ``basic`` (display "Free"): transcription 18 000 s/month, chat 30
  questions/month (hard cap).
* ``pro`` (display "Thothcraft Pro"): unlimited transcription, chat 1 000
  questions/month (hard cap). The only plan sold; monthly + annual Stripe
  prices from ``STRIPE_PRICE_ID_PRO_MONTHLY`` / ``STRIPE_PRICE_ID_PRO_ANNUAL``.
* ``home`` / ``research``: deprecated, existing subscribers only, treated as
  Pro-equivalent allocations (grandfathered).

Processing happens on Room1; Brain only stores **counters** reported by
the user's devices (``usage_counter``), never content. Caps reset at the
start of each UTC month.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.db import UsageCounter, UsageReport, User
from server.entitlements import normalize_plan

METRICS = ("transcription_seconds", "chat_questions",
           "words_transcribed", "memories_created")

# limit None = unlimited
CATALOG: Dict[str, Dict[str, Any]] = {
    "basic": {
        "display_name": "Free",
        "lifecycle": "current",
        "is_paid": False,
        "allocations": {"transcription_seconds": 18000, "chat_questions": 30,
                        "words_transcribed": None, "memories_created": None},
        "features": ["Everything runs on your Room1", "5 hours of transcription / month",
                     "30 chat questions / month", "Unlimited memories and notes"],
    },
    "pro": {
        "display_name": "Thothcraft Pro",
        "lifecycle": "current",
        "is_paid": True,
        "allocations": {"transcription_seconds": None, "chat_questions": 1000,
                        "words_transcribed": None, "memories_created": None},
        "features": ["Unlimited transcription", "1,000 chat questions / month",
                     "Agentic tasks with Codex", "Remote access relay",
                     "Priority support"],
        "prices": [("month", "STRIPE_PRICE_ID_PRO_MONTHLY"),
                   ("year", "STRIPE_PRICE_ID_PRO_ANNUAL")],
    },
    "home": {
        "display_name": "Home (legacy)",
        "lifecycle": "deprecated",
        "is_paid": True,
        "allocations": {"transcription_seconds": None, "chat_questions": 1000,
                        "words_transcribed": None, "memories_created": None},
        "features": ["Grandfathered plan"],
        "prices": [("month", "STRIPE_PRICE_ID_HOME_MONTHLY"),
                   ("year", "STRIPE_PRICE_ID_HOME_ANNUAL")],
    },
    "research": {
        "display_name": "Research (legacy)",
        "lifecycle": "deprecated",
        "is_paid": True,
        "allocations": {"transcription_seconds": None, "chat_questions": 1000,
                        "words_transcribed": None, "memories_created": None},
        "features": ["Grandfathered plan"],
        "prices": [("month", "STRIPE_PRICE_ID_RESEARCH_MONTHLY"),
                   ("year", "STRIPE_PRICE_ID_RESEARCH_ANNUAL")],
    },
}


def plan_type(user: Any) -> str:
    """Catalog id for a user's stored plan (``free`` → ``basic``)."""
    plan = normalize_plan(getattr(user, "plan", None))
    return "basic" if plan == "free" else plan


def current_period(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def period_reset_at(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def price_ids(plan_id: str) -> List[Dict[str, str]]:
    out = []
    for interval, env in CATALOG.get(plan_id, {}).get("prices", []):
        price_id = os.getenv(env)
        if price_id:
            out.append({"interval": interval, "price_id": price_id})
    return out


def plan_for_price(price_id: Optional[str]) -> Optional[str]:
    if not price_id:
        return None
    for plan_id in CATALOG:
        if any(p["price_id"] == price_id for p in price_ids(plan_id)):
            return plan_id
    return None


def sellable_price_ids() -> set:
    return {p["price_id"] for p in price_ids("pro")}


def available_plans(user: Any) -> List[Dict[str, Any]]:
    """omi audience rule: sell Pro; show a retired plan only to its subscriber."""
    current = plan_type(user)
    plans = []
    for plan_id, spec in CATALOG.items():
        if not spec["is_paid"]:
            continue
        if spec["lifecycle"] == "deprecated" and plan_id != current:
            continue
        prices = [{
            "id": p["price_id"],
            "title": "Monthly" if p["interval"] == "month" else "Annual",
            "description": None,
            "price_string": "",
        } for p in price_ids(plan_id)]
        alloc = spec["allocations"]
        plans.append({
            "id": plan_id,
            "title": spec["display_name"],
            "subtitle": f"{alloc['chat_questions']:,} questions per month" if alloc["chat_questions"] else "Unlimited",
            "features": spec["features"],
            "prices": prices,
            "legacy": spec["lifecycle"] == "deprecated",
        })
    return plans


def usage(db: Session, user_id: int, period: Optional[str] = None) -> Dict[str, float]:
    period = period or current_period()
    rows = db.query(UsageCounter).filter(
        UsageCounter.user_id == user_id, UsageCounter.period == period).all()
    out = {m: 0.0 for m in METRICS}
    for row in rows:
        out[row.metric] = float(row.value or 0)
    return out


def allocations(db: Session, user: Any, period: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    used = usage(db, user.userId, period)
    limits = CATALOG[plan_type(user)]["allocations"]
    out = {}
    for metric in METRICS:
        limit = limits.get(metric)
        value = used.get(metric, 0.0)
        out[metric] = {
            "used": value,
            "limit": limit,
            "remaining": None if limit is None else max(0.0, limit - value),
            "allowed": limit is None or value < limit,
            "percent": 0.0 if not limit else round(min(100.0, 100.0 * value / limit), 2),
        }
    return out


def add_usage(db: Session, user_id: int, deltas: Dict[str, float],
              period: Optional[str] = None) -> None:
    """Increment counters (caller commits)."""
    period = period or current_period()
    for metric, delta in deltas.items():
        if metric not in METRICS or not delta or delta < 0:
            continue
        row = db.query(UsageCounter).filter(
            UsageCounter.user_id == user_id, UsageCounter.period == period,
            UsageCounter.metric == metric).first()
        if row is None:
            row = UsageCounter(user_id=user_id, period=period, metric=metric, value=0.0)
            db.add(row)
        row.value = float(row.value or 0) + float(delta)
        row.updated_at = datetime.utcnow()


def record_report(db: Session, user_id: int, device_uuid: str, period: str,
                  seq: int, deltas: Dict[str, float]) -> bool:
    """Apply a device usage report once. Returns False for a duplicate."""
    if db.query(UsageReport).filter(
            UsageReport.device_uuid == device_uuid, UsageReport.period == period,
            UsageReport.seq == seq).first():
        return False
    db.add(UsageReport(user_id=user_id, device_uuid=device_uuid,
                       period=period, seq=seq))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return False
    add_usage(db, user_id, deltas, period)
    db.commit()
    return True


def entitlement_snapshot(db: Session, user: Any) -> Dict[str, Any]:
    """What Room1 caches to enforce caps offline."""
    pid = plan_type(user)
    return {
        "plan": pid,
        "plan_display": CATALOG[pid]["display_name"],
        "is_paid": CATALOG[pid]["is_paid"],
        "period": current_period(),
        "reset_at": period_reset_at(),
        "allocations": allocations(db, user),
    }
