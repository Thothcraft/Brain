"""Privacy-first settings and consent gates (NFR-PRV-01).

Raw content (audio, transcripts, raw sensor minutes) is processed and
stored on the user's Room1. Brain accepts raw content, or processes it in
the cloud, only with explicit consent:

* ``raw_upload``        — device-originated raw sensor/minute uploads
* ``cloud_processing``  — cloud STT/LLM/TTS on user content (``/v1/chat*``)
* ``cloud_backup``      — encrypted backups stored in Brain storage
* ``analytics``         — product analytics beyond operational metrics

Accounts created before privacy-first (``privacy_json IS NULL``) keep their
previous behaviour until the user saves settings; new accounts (Firebase
sign-up) start with everything off.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import HTTPException

KEYS = ("raw_upload", "cloud_processing", "cloud_backup", "analytics")
LEGACY_DEFAULTS = {key: True for key in KEYS}
STRICT_DEFAULTS = {key: False for key in KEYS}


def settings(user: Any) -> Dict[str, Any]:
    raw = getattr(user, "privacy_json", None)
    if not raw:
        return {**LEGACY_DEFAULTS, "legacy": True}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data = {}
    return {**STRICT_DEFAULTS, **{k: bool(data.get(k)) for k in KEYS if k in data}, "legacy": False}


def update(user: Any, changes: Dict[str, Any]) -> Dict[str, Any]:
    current = {k: v for k, v in settings(user).items() if k in KEYS}
    current.update({k: bool(v) for k, v in changes.items() if k in KEYS})
    target = getattr(user, "_user", user)
    target.privacy_json = json.dumps(current)
    return settings(target)


def allowed(user: Any, key: str) -> bool:
    return bool(settings(user).get(key))


def require(user: Any, key: str) -> None:
    """403 unless the user consented to ``key``."""
    if not allowed(user, key):
        raise HTTPException(status_code=403, detail={
            "code": "privacy_consent_required",
            "consent": key,
            "message": "This data stays on your Room1. Enable it in Settings › Privacy to use the cloud.",
        })
