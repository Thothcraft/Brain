"""Centralized plan entitlements for ThothCraft.

Single source of truth for what each subscription plan allows.
All feature gating must go through this module — never scatter
``user.plan == "..."`` conditionals across endpoints.

Plans
-----
free     : functionally capable; 1 device, no raw-data download,
           rolling retention of the most recent 400 minutes.
home     : 5 devices, 10 GB cloud storage, data download/export.
research : 10 devices, 100 GB cloud storage, datasets, Labs,
           notebook submission/grading, full SDK use.
"""

import logging
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from server.db import User, get_db

logger = logging.getLogger(__name__)

GB = 1024 ** 3

# ── plan definitions ──────────────────────────────────────────────────────────

PLANS: Dict[str, Dict[str, Any]] = {
    "free": {
        "device_limit": 1,
        "space_limit": 1,
        # Free keeps only the most recent N minutes server-side.
        "minute_retention": 400,
        "storage_bytes": None,          # bounded by minute_retention, not bytes
        "download_data": False,
        "labs": False,
        "datasets": False,
        "notebook_submission": False,
        "custom_models": False,         # private model deployment
        "sdk": "read_control",          # read data + control devices
    },
    "home": {
        "device_limit": 5,
        "space_limit": 5,
        "minute_retention": None,       # bounded by storage_bytes
        "storage_bytes": 10 * GB,
        "download_data": True,
        "labs": False,
        "datasets": False,
        "notebook_submission": False,
        "custom_models": False,
        "sdk": "full",
    },
    "research": {
        "device_limit": 10,
        "space_limit": None,          # unlimited
        "minute_retention": None,
        "storage_bytes": 100 * GB,
        "download_data": True,
        "labs": True,
        "datasets": True,
        "notebook_submission": True,
        "custom_models": True,
        "sdk": "full",
    },
}

DEFAULT_PLAN = "free"

# Stripe price metadata may still carry historical names.
PLAN_ALIASES = {
    "researcher": "research",
}


def normalize_plan(plan: Optional[str]) -> str:
    """Map legacy/alias plan names to a canonical plan key."""
    if not plan:
        return DEFAULT_PLAN
    plan = PLAN_ALIASES.get(plan, plan)
    return plan if plan in PLANS else DEFAULT_PLAN


def get_entitlements(user: User) -> Dict[str, Any]:
    """Return the entitlement dict for a user's current plan."""
    return PLANS[normalize_plan(getattr(user, "plan", None))]


def has_entitlement(user: User, feature: str) -> bool:
    """Boolean check for a single entitlement flag."""
    return bool(get_entitlements(user).get(feature))


def require_entitlement(feature: str) -> Callable:
    """FastAPI dependency factory: 403 unless the user's plan grants `feature`.

    Usage::

        @router.get("/labs")
        async def list_labs(user: User = Depends(require_entitlement("labs"))):
            ...
    """
    from server.auth import get_current_user  # deferred to avoid circular import

    async def _checker(current_user: User = Depends(get_current_user)) -> User:
        if not has_entitlement(current_user, feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your plan does not include '{feature}'. Upgrade to access this feature.",
            )
        return current_user

    return _checker


def check_device_limit(user: User, current_device_count: int) -> None:
    """Raise 403 if registering/pairing another device would exceed the plan."""
    limit = get_entitlements(user)["device_limit"]
    if current_device_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Device limit reached ({limit}). Upgrade your plan to add more devices.",
        )


def check_space_limit(user: User, current_space_count: int) -> None:
    """Raise 403 if creating another space would exceed the plan."""
    limit = get_entitlements(user).get("space_limit")
    if limit is not None and current_space_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Space limit reached ({limit}). Upgrade your plan to add more spaces.",
        )


def check_download_allowed(user: User) -> None:
    """Raise 403 if the plan does not allow raw-data download/export."""
    if not has_entitlement(user, "download_data"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Raw-data download is not available on the Free plan.",
        )


def check_storage_quota(user: User, used_bytes: int, incoming_bytes: int = 0) -> None:
    """Raise 413 if a byte-quota plan would be exceeded.

    Free is bounded by minute retention instead of bytes, so this is a no-op
    for plans where ``storage_bytes`` is None.
    """
    quota = get_entitlements(user)["storage_bytes"]
    if quota is not None and used_bytes + incoming_bytes > quota:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Cloud storage quota exceeded. Delete data or upgrade your plan.",
        )
