"""Organization endpoints — invite codes, membership management."""

import json
import logging
import secrets
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import InviteCode, OrgMembership, User, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/org", tags=["organization"])


def _require_org(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != 2:
        raise HTTPException(status_code=403, detail="Organization account required")
    return current_user


def _gen_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _member_dict(m: OrgMembership) -> Dict[str, Any]:
    return {
        "id": m.id,
        "member_id": m.member_id,
        "username": m.member.username if m.member else None,
        "status": m.status,
        "invited_at": m.invited_at.isoformat() if m.invited_at else None,
        "approved_at": m.approved_at.isoformat() if m.approved_at else None,
        "invite_code": m.invite_code,
    }


# ─── request models ───────────────────────────────────────────────────────────

class MemberActionRequest(BaseModel):
    member_id: int


class JoinOrgRequest(BaseModel):
    invite_code: str


# ─── invite code management ───────────────────────────────────────────────────

@router.get("/invite-codes")
async def list_invite_codes(
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    codes = db.query(InviteCode).filter(
        InviteCode.org_id == org.userId
    ).order_by(desc(InviteCode.created_at)).all()
    return {
        "codes": [
            {
                "id": c.id,
                "code": c.code,
                "is_active": c.is_active,
                "uses_count": c.uses_count,
                "max_uses": c.max_uses,
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                "created_at": c.created_at.isoformat(),
            }
            for c in codes
        ]
    }


@router.post("/invite-codes")
async def create_invite_code(
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # Generate unique code
    for _ in range(10):
        code = _gen_code(8)
        if not db.query(InviteCode).filter(InviteCode.code == code).first():
            break
    invite = InviteCode(code=code, org_id=org.userId)
    db.add(invite)
    db.commit()
    return {"success": True, "code": code, "id": invite.id}


@router.delete("/invite-codes/{code_id}")
async def deactivate_invite_code(
    code_id: int,
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    invite = db.query(InviteCode).filter(
        InviteCode.id == code_id,
        InviteCode.org_id == org.userId
    ).first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite code not found")
    invite.is_active = False
    db.commit()
    return {"success": True}


# ─── member management (org view) ────────────────────────────────────────────

@router.get("/members")
async def list_members(
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
    status_filter: Optional[str] = Query(None, alias="status"),
) -> Dict[str, Any]:
    q = db.query(OrgMembership).filter(OrgMembership.org_id == org.userId)
    if status_filter:
        q = q.filter(OrgMembership.status == status_filter)
    members = q.order_by(desc(OrgMembership.invited_at)).all()
    return {"members": [_member_dict(m) for m in members]}


@router.post("/members/{member_id}/approve")
async def approve_member(
    member_id: int,
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    m = db.query(OrgMembership).filter(
        OrgMembership.org_id == org.userId,
        OrgMembership.member_id == member_id,
        OrgMembership.status == "pending"
    ).first()
    if not m:
        raise HTTPException(status_code=404, detail="Pending membership not found")
    m.status = "approved"
    m.approved_at = datetime.utcnow()
    db.commit()
    return {"success": True}


@router.post("/members/{member_id}/decline")
async def decline_member(
    member_id: int,
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    m = db.query(OrgMembership).filter(
        OrgMembership.org_id == org.userId,
        OrgMembership.member_id == member_id,
    ).first()
    if not m:
        raise HTTPException(status_code=404, detail="Membership not found")
    m.status = "declined"
    db.commit()
    return {"success": True}


@router.delete("/members/{member_id}")
async def remove_member(
    member_id: int,
    org: User = Depends(_require_org),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    m = db.query(OrgMembership).filter(
        OrgMembership.org_id == org.userId,
        OrgMembership.member_id == member_id,
    ).first()
    if not m:
        raise HTTPException(status_code=404, detail="Membership not found")
    db.delete(m)
    db.commit()
    return {"success": True}


# ─── member self-service (any user joining via invite code) ──────────────────

@router.post("/join")
async def join_org(
    data: JoinOrgRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Any authenticated user can request to join an org via invite code."""
    invite = db.query(InviteCode).filter(
        InviteCode.code == data.invite_code.strip().upper(),
        InviteCode.is_active == True
    ).first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invalid or expired invite code")
    if invite.expires_at and invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invite code has expired")
    if invite.uses_count >= invite.max_uses:
        raise HTTPException(status_code=400, detail="Invite code has reached maximum uses")

    # Check existing membership
    existing = db.query(OrgMembership).filter(
        OrgMembership.org_id == invite.org_id,
        OrgMembership.member_id == current_user.userId
    ).first()
    if existing:
        return {"success": False, "message": f"Already {existing.status} for this organization"}

    membership = OrgMembership(
        org_id=invite.org_id,
        member_id=current_user.userId,
        status="pending",
        invite_code=invite.code,
    )
    invite.uses_count += 1
    db.add(membership)
    db.commit()
    return {"success": True, "message": "Join request submitted — awaiting approval"}


@router.get("/my-memberships")
async def my_memberships(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return all orgs the current user belongs to (any status)."""
    memberships = db.query(OrgMembership).filter(
        OrgMembership.member_id == current_user.userId
    ).all()
    return {
        "memberships": [
            {
                "id": m.id,
                "org_id": m.org_id,
                "org_name": m.org.org_name or m.org.username if m.org else None,
                "status": m.status,
                "invited_at": m.invited_at.isoformat() if m.invited_at else None,
                "approved_at": m.approved_at.isoformat() if m.approved_at else None,
            }
            for m in memberships
        ]
    }
