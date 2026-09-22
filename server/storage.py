"""Server-side cloud storage accounting.

Quota is always computed from owned objects in the database — never
from client-reported values. Counted objects:

* ``File`` rows (portal uploads, stored content)
* ``DeviceFile`` rows flagged ``on_cloud`` (synced device captures)
* ``TrainedModel.size_bytes`` (user-trained models)
"""

import logging
from typing import Any, Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from server.db import DeviceFile, File, TrainedModel, User
from server.entitlements import get_entitlements

logger = logging.getLogger(__name__)

WARN_THRESHOLD = 0.9


def user_storage_bytes(db: Session, user: User) -> int:
    """Total cloud bytes owned by the user, computed server-side."""
    uploads = db.query(func.coalesce(func.sum(File.size), 0)).filter(
        File.userId == user.userId
    ).scalar() or 0

    device_files = db.query(func.coalesce(func.sum(DeviceFile.size), 0)).filter(
        DeviceFile.user_id == user.userId,
        DeviceFile.on_cloud == True,
    ).scalar() or 0

    models = db.query(func.coalesce(func.sum(TrainedModel.size_bytes), 0)).filter(
        TrainedModel.user_id == user.userId
    ).scalar() or 0

    return int(uploads) + int(device_files) + int(models)


def storage_status(db: Session, user: User) -> Dict[str, Any]:
    """Usage/quota summary for the user's plan."""
    ent = get_entitlements(user)
    used = user_storage_bytes(db, user)
    quota = ent.get("storage_bytes")

    status: Dict[str, Any] = {
        "used_bytes": used,
        "quota_bytes": quota,
        "plan": user.plan or "free",
        "warning": False,
        "uploads_blocked": False,
    }

    if quota is None:
        # Free: bounded by minute retention, not bytes
        status["minute_retention"] = ent.get("minute_retention")
        return status

    status["used_fraction"] = used / quota if quota else 0
    status["warning"] = used >= WARN_THRESHOLD * quota
    # At 100% new cloud raw-data uploads stop; local sensing never stops.
    status["uploads_blocked"] = used >= quota
    return status


def check_upload_allowed(db: Session, user: User, incoming_bytes: int = 0) -> None:
    """Raise 413 when the upload would exceed the plan's storage quota.

    Free has no byte quota (bounded by minute retention) — always allowed.
    Local device storage is never affected; this only gates cloud uploads.
    """
    from fastapi import HTTPException

    ent = get_entitlements(user)
    quota = ent.get("storage_bytes")
    if quota is None:
        return
    used = user_storage_bytes(db, user)
    if used + incoming_bytes > quota:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Cloud storage quota exceeded "
                f"({used // (1024 ** 2)}MB of {quota // (1024 ** 3)}GB used). "
                "Upgrade your plan or delete cloud files."
            ),
        )
