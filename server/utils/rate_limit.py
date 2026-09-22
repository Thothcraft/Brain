"""Lightweight in-memory rate limiting for sensitive endpoints.

Fixed-window counter keyed by (bucket, client key). Suitable for a
single-process deployment; swap for Redis-backed limiting if Brain ever
runs multiple workers.
"""

import threading
import time
from typing import Callable, Optional

from fastapi import HTTPException, Request, status

_WINDOWS: dict = {}
_LOCK = threading.Lock()


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def hit(bucket: str, key: str, max_attempts: int, window_seconds: int) -> bool:
    """Record a hit; return True if within limit, False if exceeded."""
    now = time.monotonic()
    full_key = f"{bucket}:{key}"
    with _LOCK:
        window_start, count = _WINDOWS.get(full_key, (now, 0))
        if now - window_start > window_seconds:
            window_start, count = now, 0
        count += 1
        _WINDOWS[full_key] = (window_start, count)
        # Opportunistic cleanup of stale buckets
        if len(_WINDOWS) > 10_000:
            stale = [k for k, (ts, _) in _WINDOWS.items() if now - ts > window_seconds]
            for k in stale:
                del _WINDOWS[k]
    return count <= max_attempts


def rate_limit(
    bucket: str,
    max_attempts: int,
    window_seconds: int,
    key_fn: Optional[Callable[[Request], str]] = None,
) -> Callable:
    """FastAPI dependency factory enforcing a fixed-window rate limit.

    Usage::

        @router.post("/token")
        async def login(..., _=Depends(rate_limit("auth", 10, 60))):
    """
    async def _checker(request: Request) -> None:
        key = key_fn(request) if key_fn else _client_key(request)
        if not hit(bucket, key, max_attempts, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Try again later.",
                headers={"Retry-After": str(window_seconds)},
            )

    return _checker
