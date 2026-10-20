"""Server-side cloud storage accounting.

Quota is always computed from owned objects in the database — never
from client-reported values. Counted objects:

* ``File`` rows (portal uploads, stored content)
* ``DeviceFile`` rows flagged ``on_cloud`` (synced device captures)
* ``TrainedModel.size_bytes`` (user-trained models)

On top of per-plan quotas this module enforces *project-wide* ceilings
sized to the backing Supabase project's quota (free tier: 1 GB Storage,
~5 GB/month egress, 500 MB database). The project ceiling is the real
hard limit — exceeding it suspends the project for every account, so it
is checked first. Budgets come from ``server.entitlements`` and are
env-overridable.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from server.db import ApiUsage, DeviceFile, File, TrainedModel, User
from server.entitlements import (
    PROJECT_DB_BUDGET_BYTES,
    PROJECT_MONTHLY_EGRESS_BUDGET_BYTES,
    PROJECT_STORAGE_BUDGET_BYTES,
    get_entitlements,
)

logger = logging.getLogger(__name__)

WARN_THRESHOLD = 0.9
EGRESS_KIND = "egress"


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


# ── project-wide usage (the Supabase quota side) ──────────────────────────────

def project_storage_bytes(db: Session) -> Optional[int]:
    """Total bytes held in Supabase Storage across all accounts.

    ``storage.objects`` lives in the same Postgres as the app tables on a
    Supabase deployment; returns None on non-Supabase databases so the
    caller can skip the project guard.
    """
    try:
        return int(db.execute(text(
            "SELECT COALESCE(SUM((metadata->>'size')::bigint), 0) "
            "FROM storage.objects"
        )).scalar() or 0)
    except Exception:
        logger.debug("storage.objects not readable — non-Supabase DB?", exc_info=True)
        return None


def project_db_bytes(db: Session) -> Optional[int]:
    """Current database size in bytes (None when unsupported)."""
    try:
        return int(db.execute(
            text("SELECT pg_database_size(current_database())")).scalar() or 0)
    except Exception:
        logger.debug("pg_database_size unavailable", exc_info=True)
        return None


def _month_start_epoch() -> float:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()


def _egress_rows(db: Session, user_id: Optional[int] = None) -> list:
    query = db.query(ApiUsage.meta).filter(
        ApiUsage.kind == EGRESS_KIND,
        ApiUsage.ts >= _month_start_epoch(),
    )
    if user_id is not None:
        query = query.filter(ApiUsage.user_id == user_id)
    return query.all()


def _meta_bytes(meta: Optional[str]) -> int:
    try:
        return int((json.loads(meta or "{}") or {}).get("bytes") or 0)
    except (TypeError, ValueError):
        return 0


def user_monthly_egress_bytes(db: Session, user: User) -> int:
    """Metered egress for this user in the current UTC month."""
    return sum(_meta_bytes(meta) for (meta,) in _egress_rows(db, user.userId))


def project_monthly_egress_bytes(db: Session) -> int:
    """Metered egress across all users in the current UTC month."""
    return sum(_meta_bytes(meta) for (meta,) in _egress_rows(db))


def record_egress(
    db: Session,
    user_id: int,
    byte_count: int,
    *,
    source: str = "api",
    device_id: str = "cloud",
) -> None:
    """Ledger one served-download byte count into ``api_usage``."""
    if byte_count <= 0:
        return
    try:
        db.add(ApiUsage(
            user_id=user_id, device_id=device_id, ts=time.time(),
            source=str(source)[:40], kind=EGRESS_KIND,
            meta=json.dumps({"bytes": int(byte_count)}),
        ))
        db.commit()
    except Exception:
        # Metering must never break a legitimate download.
        logger.debug("egress metering failed", exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass


def effective_storage_quota(db: Session, user: User) -> Optional[int]:
    """The user's byte quota clamped to the remaining project budget.

    A plan cap larger than what the Supabase project can still hold is
    meaningless — the project budget always wins.
    """
    quota = get_entitlements(user).get("storage_bytes")
    project_used = project_storage_bytes(db)
    if project_used is None:
        return quota
    remaining = max(0, PROJECT_STORAGE_BUDGET_BYTES - project_used)
    return remaining if quota is None else min(quota, remaining)


def storage_status(db: Session, user: User) -> Dict[str, Any]:
    """Usage/quota summary for the user's plan."""
    ent = get_entitlements(user)
    used = user_storage_bytes(db, user)
    quota = ent.get("storage_bytes")
    effective_quota = effective_storage_quota(db, user)

    project_used = project_storage_bytes(db)
    db_bytes = project_db_bytes(db)
    egress_used = user_monthly_egress_bytes(db, user)
    egress_quota = ent.get("monthly_egress_bytes")

    status: Dict[str, Any] = {
        "used_bytes": used,
        "quota_bytes": quota,
        "effective_quota_bytes": effective_quota,
        "plan": user.plan or "free",
        "warning": False,
        "uploads_blocked": False,
        "egress": {
            "used_bytes": egress_used,
            "quota_bytes": egress_quota,
            "monthly": True,
        },
        "project": {
            "storage_used_bytes": project_used,
            "storage_budget_bytes": PROJECT_STORAGE_BUDGET_BYTES,
            "db_used_bytes": db_bytes,
            "db_budget_bytes": PROJECT_DB_BUDGET_BYTES,
            "egress_used_bytes": project_monthly_egress_bytes(db),
            "egress_budget_bytes": PROJECT_MONTHLY_EGRESS_BUDGET_BYTES,
        },
    }

    if quota is None:
        # Free: bounded by minute retention, not bytes
        status["minute_retention"] = ent.get("minute_retention")
    else:
        status["used_fraction"] = used / quota if quota else 0
        status["warning"] = used >= WARN_THRESHOLD * quota

    if effective_quota is not None:
        status["warning"] = status["warning"] or used >= WARN_THRESHOLD * effective_quota
        # At 100% new cloud raw-data uploads stop; local sensing never stops.
        status["uploads_blocked"] = used >= effective_quota
    if db_bytes is not None and db_bytes >= WARN_THRESHOLD * PROJECT_DB_BUDGET_BYTES:
        status["db_warning"] = True
    return status


def check_upload_allowed(db: Session, user: User, incoming_bytes: int = 0) -> None:
    """Raise 413 when the upload would exceed the storage budget.

    Checks the *project* Supabase Storage ceiling first (the hard quota
    that would suspend the whole deployment), then the per-plan quota.
    Free has no byte quota (bounded by minute retention) — always allowed.
    Local device storage is never affected; this only gates cloud uploads.
    """
    from fastapi import HTTPException

    project_used = project_storage_bytes(db)
    if project_used is not None and \
            project_used + incoming_bytes > PROJECT_STORAGE_BUDGET_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Supabase project storage budget exhausted "
                f"({project_used // (1024 ** 2)}MB of "
                f"{PROJECT_STORAGE_BUDGET_BYTES // (1024 ** 2)}MB used). "
                "Delete cloud files or raise SUPABASE_PROJECT_STORAGE_BYTES."
            ),
        )

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
                f"({used // (1024 ** 2)}MB of {quota // (1024 ** 2)}MB used). "
                "Upgrade your plan or delete cloud files."
            ),
        )


def check_egress_allowed(db: Session, user: User, incoming_bytes: int = 0) -> None:
    """Raise 429 when a download would exceed the monthly egress budget.

    Project budget (Supabase egress allowance) is checked first, then the
    plan's ``monthly_egress_bytes``. Metering lives in ``api_usage`` rows
    written by :func:`record_egress`.
    """
    from fastapi import HTTPException

    project_egress = project_monthly_egress_bytes(db)
    if project_egress + incoming_bytes > PROJECT_MONTHLY_EGRESS_BUDGET_BYTES:
        raise HTTPException(
            status_code=429,
            detail=(
                "Supabase project monthly egress budget exhausted "
                f"({project_egress // (1024 ** 2)}MB of "
                f"{PROJECT_MONTHLY_EGRESS_BUDGET_BYTES // (1024 ** 3)}GB used). "
                "Raise SUPABASE_MONTHLY_EGRESS_BUDGET_BYTES or wait for next month."
            ),
        )

    quota = get_entitlements(user).get("monthly_egress_bytes")
    if quota is None:
        return
    used = user_monthly_egress_bytes(db, user)
    if used + incoming_bytes > quota:
        raise HTTPException(
            status_code=429,
            detail=(
                "Monthly download quota exceeded "
                f"({used // (1024 ** 2)}MB of {quota // (1024 ** 3)}GB used). "
                "Upgrade your plan or wait for next month."
            ),
        )
