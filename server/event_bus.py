"""In-process event fan-out for user-facing event streams.

``node_ws._write_event`` is the single choke point where node events land
(WS frames + REST fallback). It calls :func:`publish` with the persisted
event dict; SSE subscribers registered via :func:`subscribe` receive it
immediately — no polling, no broker.

Single-process deployments only: the bus is per-process state (same
constraint as the node WS registry). Safe to call from sync code running
inside a worker thread (``asyncio.to_thread``) or the event loop — delivery
uses ``loop.call_soon_threadsafe`` when a subscriber lives on a different
loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# subscriber queue -> (loop it must be signaled on, queue)
_Subscriber = Tuple[asyncio.AbstractEventLoop, "asyncio.Queue[Dict[str, Any]]"]
_lock = threading.Lock()
_subscribers: Dict[int, List[_Subscriber]] = {}


def subscribe(user_id: int) -> "asyncio.Queue[Dict[str, Any]]":
    """Register a subscriber for ``user_id``'s events.

    Must be called on the event loop that will consume the queue.
    Caller must ``unsubscribe`` when done.
    """
    q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=500)
    loop = asyncio.get_running_loop()
    with _lock:
        _subscribers.setdefault(user_id, []).append((loop, q))
    return q


def unsubscribe(user_id: int, q: "asyncio.Queue[Dict[str, Any]]") -> None:
    with _lock:
        subs = _subscribers.get(user_id)
        if subs:
            _subscribers[user_id] = [s for s in subs if s[1] is not q]
            if not _subscribers[user_id]:
                _subscribers.pop(user_id, None)


def publish(user_id: int, event: Dict[str, Any]) -> int:
    """Deliver ``event`` to all live subscribers of ``user_id``.

    Returns the number of subscribers notified. Never raises — a dead
    queue is dropped silently.
    """
    with _lock:
        subs = list(_subscribers.get(user_id) or [])
    delivered = 0
    for loop, q in subs:
        try:
            if loop.is_closed():
                continue
            if loop.is_running():
                loop.call_soon_threadsafe(_safe_put, q, event)
            else:
                _safe_put(q, event)
            delivered += 1
        except Exception:
            continue
    return delivered


def _safe_put(q: "asyncio.Queue[Dict[str, Any]]",
              event: Dict[str, Any]) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        # Slow consumer: drop oldest rather than block the node channel.
        try:
            q.get_nowait()
            q.put_nowait(event)
        except Exception:
            pass


def subscriber_count(user_id: Optional[int] = None) -> int:
    with _lock:
        if user_id is not None:
            return len(_subscribers.get(user_id) or [])
        return sum(len(v) for v in _subscribers.values())


__all__ = ["subscribe", "unsubscribe", "publish", "subscriber_count"]
