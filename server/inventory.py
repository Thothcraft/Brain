"""Ordering rules for complete device inventory snapshots."""

from __future__ import annotations

from datetime import datetime


def is_newer_snapshot(incoming_revision: int, incoming_timestamp: str, current: dict) -> bool:
    """Require both monotonic revision and non-regressing device time."""
    current_revision = int(current.get("revision") or -1)
    if int(incoming_revision) <= current_revision:
        return False
    current_timestamp = str(current.get("timestamp") or "")
    if current_timestamp and incoming_timestamp:
        try:
            incoming_time = datetime.fromisoformat(incoming_timestamp.replace("Z", "+00:00"))
            current_time = datetime.fromisoformat(current_timestamp.replace("Z", "+00:00"))
            if incoming_time < current_time:
                return False
        except ValueError:
            return False
    return True
