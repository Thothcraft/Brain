"""Centralized plan entitlements for ThothCraft.

Single source of truth for what each subscription plan allows.
All feature gating must go through this module — never scatter
``user.plan == "..."`` conditionals across endpoints.

Plans
-----
free     : functionally capable; 1 device, no raw-data download,
           rolling retention of the most recent 400 minutes,
           200 hosted-inference calls/month.
home     : 5 devices, cloud storage + data download/export bounded by
           the Supabase project budget, 5,000 hosted-inference calls/month.
research : 10 devices, larger cloud storage share, datasets, Labs,
           notebook submission/grading, full SDK use,
           20,000 hosted-inference calls/month.

``inference_quota`` = hosted model calls per UTC calendar month (None
= unlimited), enforced by ``server.inference_auth``; the provisional
values here are placeholders for the paid-inference phase and can be
overridden deployment-wide via ``INFERENCE_MONTHLY_QUOTA``.

Supabase budget
---------------
Per-plan ``storage_bytes``/``monthly_egress_bytes`` are sized relative to
the backing Supabase project's quota, which is the real hard limit. The
deployment below is a **free** project (500 MB database, 1 GB Storage,
~5 GB/month egress), so plan caps are small fractions of the project
budget and ``server.storage`` additionally enforces project-wide ceilings
so one account can never push the project itself over quota. Override
the project budgets via env when the Supabase plan changes:

* ``SUPABASE_PROJECT_STORAGE_BYTES`` — total Storage budget (default 1 GiB)
* ``SUPABASE_MONTHLY_EGRESS_BUDGET_BYTES`` — total monthly egress budget
  (default 4 GiB, margin under the ~5 GB free allowance)
* ``SUPABASE_DB_BUDGET_BYTES`` — database size budget (default 460 MiB)
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from server.db import User, get_db

logger = logging.getLogger(__name__)

GB = 1024 ** 3
MB = 1024 ** 2

# ── Supabase project budgets (deployment-wide hard ceilings) ─────────────────
# Defaults match a Supabase *free* project; raise via env on paid tiers.
PROJECT_STORAGE_BUDGET_BYTES = int(
    os.getenv("SUPABASE_PROJECT_STORAGE_BYTES", str(1 * GB)))
PROJECT_MONTHLY_EGRESS_BUDGET_BYTES = int(
    os.getenv("SUPABASE_MONTHLY_EGRESS_BUDGET_BYTES", str(4 * GB)))
PROJECT_DB_BUDGET_BYTES = int(
    os.getenv("SUPABASE_DB_BUDGET_BYTES", str(460 * MB)))

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
        "inference_quota": 200,          # hosted model calls / month
        "monthly_egress_bytes": 250 * MB,
    },
    "home": {
        "device_limit": 5,
        "space_limit": 5,
        "minute_retention": None,       # bounded by storage_bytes
        # Sized to a free-tier Supabase project (1 GB storage total); the
        # project-wide budget guard in server.storage is the backstop.
        "storage_bytes": 400 * MB,
        "download_data": True,
        "labs": False,
        "datasets": False,
        "notebook_submission": False,
        "custom_models": False,
        "sdk": "full",
        "inference_quota": 5000,
        "monthly_egress_bytes": 1 * GB,
    },
    "research": {
        "device_limit": 10,
        "space_limit": None,          # unlimited
        "minute_retention": None,
        "storage_bytes": 1 * GB,
        "download_data": True,
        "labs": True,
        "datasets": True,
        "notebook_submission": True,
        "custom_models": True,
        "sdk": "full",
        "inference_quota": 20000,
        "monthly_egress_bytes": 3 * GB,
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
