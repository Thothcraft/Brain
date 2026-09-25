"""Outbound webhook delivery for :class:`EventSubscription` rows.

When a ``node_event`` row commits, :func:`enqueue` fans it out to every
enabled subscription owned by the event's user (``kinds``/``device_id``
filtered). Each delivery is a durable :class:`EventDelivery` row; the
scheduler calls :func:`drain` every second to POST due rows — signed
``X-Thoth-Signature: sha256=<hmac(secret, raw_body)>`` — with bounded
exponential backoff. Failures never drop silently: the row carries the
last error/status code for inspection and dead-letter filtering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List

from .db import (
    EventDelivery, EventSubscription, get_db_session,
)

logger = logging.getLogger(__name__)

_BACKOFF_S = (5, 30, 120, 600)          # attempts 2..5 delays
_TIMEOUT_S = 10.0
_MAX_BODY_BYTES = 64 * 1024


def _matches(sub: EventSubscription, event: Dict[str, Any]) -> bool:
    try:
        kinds = json.loads(sub.kinds) if sub.kinds else []
    except (TypeError, json.JSONDecodeError):
        kinds = []
    if kinds and event.get("kind") not in kinds:
        return False
    if sub.device_id and event.get("device_id") != sub.device_id:
        return False
    return True


def enqueue(user_id: int, event: Dict[str, Any]) -> int:
    """Queue one delivery per matching subscription. Returns count."""
    if not event or not event.get("id"):
        return 0
    with get_db_session() as db:
        subs = db.query(EventSubscription).filter(
            EventSubscription.user_id == user_id,
            EventSubscription.enabled == True).all()  # noqa: E712
        n = 0
        for sub in subs:
            if not _matches(sub, event):
                continue
            db.add(EventDelivery(
                subscription_id=sub.id,
                event_id=str(event["id"]),
                next_attempt_at=time.time()))
            n += 1
        return n


def _deliver_one(db, delivery: EventDelivery) -> None:
    sub = db.query(EventSubscription).get(delivery.subscription_id)
    if sub is None or not sub.enabled:
        delivery.status = "dropped"
        delivery.completed_at = datetime.utcnow()
        delivery.last_error = "subscription disabled or removed"
        return
    body = json.dumps({
        "event_id": delivery.event_id,
        "subscription_id": sub.id,
        "delivered_at": time.time(),
    }).encode("utf-8")
    # The event payload travels with the delivery — join lazily so a
    # missing event row can't wedge the worker.
    from .db import NodeEvent
    try:
        event = db.query(NodeEvent).get(int(delivery.event_id))
        payload = {
            "id": delivery.event_id,
            "kind": event.kind if event else None,
            "device_id": event.device_id if event else None,
            "ts": event.ts if event else None,
            "data": json.loads(event.data) if event and event.data else {},
        }
    except Exception:
        payload = {"id": delivery.event_id}
    payload["delivery_id"] = delivery.id
    body = json.dumps(payload).encode("utf-8")[:_MAX_BODY_BYTES]

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "thoth-brain-webhook/1",
        "X-Thoth-Event-Id": delivery.event_id,
    }
    if sub.secret:
        sig = hmac.new(sub.secret.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Thoth-Signature"] = f"sha256={sig}"

    delivery.attempts += 1
    try:
        req = urllib.request.Request(sub.url, data=body, headers=headers,
                                     method="POST")
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as res:
            delivery.last_status_code = res.status
            if 200 <= res.status < 300:
                delivery.status = "succeeded"
                delivery.completed_at = datetime.utcnow()
                delivery.last_error = None
            else:
                raise urllib.error.HTTPError(
                    sub.url, res.status, "non-2xx", res.headers, None)
    except urllib.error.HTTPError as exc:
        delivery.last_status_code = exc.code
        _fail_or_retry(delivery, f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError) as exc:
        delivery.last_status_code = None
        _fail_or_retry(delivery, str(exc)[:480])
    except Exception as exc:  # never let one bad sub kill the drain loop
        delivery.last_status_code = None
        _fail_or_retry(delivery, f"{type(exc).__name__}: {exc}"[:480])


def _fail_or_retry(delivery: EventDelivery, error: str) -> None:
    delivery.last_error = error
    if delivery.attempts >= (delivery.max_attempts or 5):
        delivery.status = "failed"
        delivery.completed_at = datetime.utcnow()
        return
    delay = _BACKOFF_S[min(delivery.attempts - 1, len(_BACKOFF_S) - 1)]
    delivery.status = "queued"
    delivery.next_attempt_at = time.time() + delay


def drain(limit: int = 50) -> int:
    """Attempt all due deliveries; called by the scheduler."""
    now = time.time()
    with get_db_session() as db:
        due = db.query(EventDelivery).filter(
            EventDelivery.status == "queued",
            EventDelivery.next_attempt_at <= now).order_by(
            EventDelivery.id.asc()).limit(limit).all()
        for delivery in due:
            try:
                _deliver_one(db, delivery)
            except Exception:
                logger.exception("event delivery %s crashed", delivery.id)
        return len(due)


__all__ = ["enqueue", "drain"]
