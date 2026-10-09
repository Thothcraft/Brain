"""Hosted inference authorization and metering.

Every path that calls a hosted model — ``POST /v1/context/infer``,
background context-builder builds, the AI assistant — must go through
:func:`inference_call`. It is the single point that:

1. **authorizes** the call against the plan's monthly inference
   allowance (``inference_quota`` in ``server.entitlements.PLANS``,
   overridable per-deploy via ``INFERENCE_MONTHLY_QUOTA``), and
2. **accounts** it in the shared ``api_usage`` ledger with
   ``source="inference"`` and a lifecycle ``status``
   (``reserved`` → ``ok``/``error``/``rejected``) so usage survives
   crashes and concurrent calls can't silently bypass the quota.

Reservation is insert-then-count: the reserved row is committed before
the model is called, then rows in (reserved, ok, error) are counted
for the period; if the count exceeds the allowance the new row is
flipped to ``rejected`` and the call raises 402. A boundary race can
over-reject (conservative) but can never let a call through once the
allowance is spent. Idempotent retries and worker crashes therefore
can't double-consume or bypass allowance — a crashed call stays
``reserved`` until the period rolls over, and a call that fails after
the model ran consumed its reservation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional, Union

from fastapi import HTTPException
from sqlalchemy.orm import Session

from server.db import ApiUsage, User
from server.entitlements import get_entitlements

logger = logging.getLogger(__name__)

INFERENCE_SOURCE = "inference"


def inference_allowance(user: Optional[User]) -> Optional[int]:
    """Monthly call allowance for the user's plan; ``None`` = unlimited.

    The allowance is the plan's ``inference_quota`` entitlement
    (``server.entitlements`` is the single source of truth);
    ``INFERENCE_MONTHLY_QUOTA`` overrides it deployment-wide (-1 =
    unlimited)."""
    override = os.getenv("INFERENCE_MONTHLY_QUOTA")
    if override:
        try:
            n = int(override)
            return None if n < 0 else n
        except ValueError:
            logger.warning("invalid INFERENCE_MONTHLY_QUOTA=%r", override)
    return get_entitlements(user).get("inference_quota")


def _plan_key(user: Optional[User]) -> str:
    from server.entitlements import normalize_plan
    return normalize_plan(getattr(user, "plan", None) if user else None)


def period_start(now: Optional[float] = None) -> float:
    """Epoch of the first instant of the current UTC calendar month."""
    d = datetime.fromtimestamp(now if now is not None else time.time(),
                               tz=timezone.utc)
    return datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()


def inference_used(db: Session, user_id: int,
                   since: Optional[float] = None) -> int:
    """Calls consumed this period — every reservation counts
    (``reserved``/``ok``/``error``); only ``rejected`` rows are free."""
    rows = db.query(ApiUsage.meta).filter(
        ApiUsage.user_id == user_id,
        ApiUsage.source == INFERENCE_SOURCE,
        ApiUsage.ts >= (since if since is not None else period_start()),
    ).all()
    return sum(1 for (meta,) in rows
               if _row_status_meta(meta) in ("reserved", "ok", "error"))


def _row_status_meta(meta: Optional[str]) -> str:
    try:
        return str((json.loads(meta or "{}") or {}).get("status") or "")
    except (TypeError, ValueError):
        return ""


def _insert_row(db: Session, user_id: int, kind: str,
                status: str) -> ApiUsage:
    row = ApiUsage(
        user_id=user_id, device_id="cloud", ts=time.time(),
        source=INFERENCE_SOURCE, kind=kind[:40],
        meta=json.dumps({"status": status}))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _finish_row(db: Session, row: ApiUsage, status: str,
                meter: Dict[str, Any], error: Any = None) -> None:
    meta = {"status": status}
    if error is not None:
        meta["error"] = str(getattr(error, "detail", error))[:200]
    row.meta = json.dumps(meta)
    row.model_id = (meter.get("model_id") or row.model_id)
    row.tokens = meter.get("tokens")
    row.latency_ms = meter.get("latency_ms")
    db.commit()


@contextmanager
def inference_call(db: Session,
                   principal: Union[User, int, None], *,
                   kind: str,
                   model_id: Optional[str] = None
                   ) -> Iterator[Dict[str, Any]]:
    """Authorize + meter one hosted model call.

    Yields a mutable ``meter`` dict the caller may fill with
    ``model_id``/``tokens``/``latency_ms``; fields are reconciled onto
    the ledger row when the call finishes. Raises HTTP 402 when the
    plan allowance is exhausted.
    """
    user = (principal if isinstance(principal, User)
            else db.get(User, principal) if principal is not None
            else None)
    user_id = int(user.userId) if user is not None else int(principal or 0)
    allowance = inference_allowance(user)
    start = period_start()

    row = _insert_row(db, user_id, kind, "reserved")
    if model_id:
        row.model_id = model_id[:255]
        db.commit()
    if allowance is not None and inference_used(db, user_id, start) > allowance:
        _finish_row(db, row, "rejected", {})
        ent = get_entitlements(user) if user is not None else {}
        raise HTTPException(
            status_code=402,
            detail={
                "error": "inference_quota_exceeded",
                "detail": "Monthly hosted-inference allowance reached; "
                          "upgrade for more.",
                "allowance": allowance,
                "used": inference_used(db, user_id, start),
                "plan": _plan_key(user),
                "sdk": ent.get("sdk"),
            })

    meter: Dict[str, Any] = {}
    t0 = time.time()
    try:
        yield meter
    except HTTPException as exc:
        meter.setdefault("latency_ms", round((time.time() - t0) * 1000, 1))
        _finish_row(db, row, "error", meter, error=exc)
        raise
    except Exception as exc:
        meter.setdefault("latency_ms", round((time.time() - t0) * 1000, 1))
        _finish_row(db, row, "error", meter, error=exc)
        raise
    else:
        meter.setdefault("latency_ms", round((time.time() - t0) * 1000, 1))
        _finish_row(db, row, "ok", meter)


def inference_status(db: Session, user: User) -> Dict[str, Any]:
    """Quota consumption for the current period — the usage surface."""
    start = period_start()
    allowance = inference_allowance(user)
    used = inference_used(db, user.userId, start)
    return {
        "plan": _plan_key(user),
        "period_start": start,
        "allowance": allowance,            # None = unlimited
        "used": used,
        "remaining": (None if allowance is None
                      else max(0, allowance - used)),
        "generated_at": time.time(),
    }
