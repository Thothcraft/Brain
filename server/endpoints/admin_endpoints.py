"""Admin endpoints — role=1 required for all routes."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import (
    Device, File, Lab, OrgMembership, Payment, TrainedModel, User, get_db
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

PLAN_PRICES = {
    "free": 0,
    "researcher": 2900,       # $29/mo in cents
    "organization": 9900,     # $99/mo in cents
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != 1:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _user_dict(u: User) -> Dict[str, Any]:
    return {
        "user_id": u.userId,
        "username": u.username,
        "role": u.role,
        "plan": u.plan or "free",
        "org_name": u.org_name,
        "stripe_customer_id": u.stripe_customer_id,
        "stripe_subscription_id": u.stripe_subscription_id,
        "plan_expires_at": u.plan_expires_at.isoformat() if u.plan_expires_at else None,
        "phone_number": u.phone_number,
    }


# ─── request models ───────────────────────────────────────────────────────────

class UpdateUserRequest(BaseModel):
    role: Optional[int] = None       # 0=user, 1=admin, 2=organization
    plan: Optional[str] = None       # free | researcher | organization
    org_name: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    plan_expires_at: Optional[str] = None  # ISO datetime string


class CreatePaymentRequest(BaseModel):
    user_id: int
    amount: int           # cents
    currency: str = "usd"
    plan: str
    status: str = "succeeded"
    stripe_payment_intent: Optional[str] = None
    stripe_invoice_id: Optional[str] = None


# ─── dashboard stats ──────────────────────────────────────────────────────────

@router.get("/stats")
async def admin_stats(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Overall platform statistics for the admin dashboard."""
    total_users = db.query(func.count(User.userId)).scalar() or 0
    total_orgs = db.query(func.count(User.userId)).filter(User.role == 2).scalar() or 0
    total_admins = db.query(func.count(User.userId)).filter(User.role == 1).scalar() or 0
    total_devices = db.query(func.count(Device.deviceId)).scalar() or 0
    total_files = db.query(func.count(File.fileId)).scalar() or 0
    total_models = db.query(func.count(TrainedModel.id)).scalar() or 0
    total_revenue = db.query(func.sum(Payment.amount)).filter(
        Payment.status == "succeeded"
    ).scalar() or 0

    plan_counts = {}
    for plan in ("free", "researcher", "organization"):
        cnt = db.query(func.count(User.userId)).filter(User.plan == plan).scalar() or 0
        plan_counts[plan] = cnt

    recent_payments = (
        db.query(Payment).order_by(desc(Payment.created_at)).limit(10).all()
    )

    return {
        "total_users": total_users,
        "total_orgs": total_orgs,
        "total_admins": total_admins,
        "total_devices": total_devices,
        "total_files": total_files,
        "total_models": total_models,
        "total_revenue_cents": total_revenue,
        "plan_counts": plan_counts,
        "recent_payments": [
            {
                "id": p.id,
                "user_id": p.user_id,
                "amount": p.amount,
                "currency": p.currency,
                "plan": p.plan,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
            }
            for p in recent_payments
        ],
    }


# ─── user management ─────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None),
    role: Optional[int] = Query(None),
    plan: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    q = db.query(User)
    if search:
        q = q.filter(User.username.ilike(f"%{search}%"))
    if role is not None:
        q = q.filter(User.role == role)
    if plan:
        q = q.filter(User.plan == plan)
    total = q.count()
    users = q.order_by(desc(User.userId)).offset(offset).limit(limit).all()

    result = []
    for u in users:
        d = _user_dict(u)
        d["device_count"] = db.query(func.count(Device.deviceId)).filter(
            Device.userId == u.userId
        ).scalar() or 0
        d["file_count"] = db.query(func.count(File.fileId)).filter(
            File.userId == u.userId
        ).scalar() or 0
        result.append(d)

    return {"users": result, "total": total}


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    u = db.query(User).filter(User.userId == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    d = _user_dict(u)
    d["device_count"] = db.query(func.count(Device.deviceId)).filter(Device.userId == u.userId).scalar() or 0
    d["file_count"] = db.query(func.count(File.fileId)).filter(File.userId == u.userId).scalar() or 0
    d["payments"] = [
        {
            "id": p.id,
            "amount": p.amount,
            "currency": p.currency,
            "plan": p.plan,
            "status": p.status,
            "created_at": p.created_at.isoformat(),
        }
        for p in db.query(Payment).filter(Payment.user_id == u.userId).order_by(desc(Payment.created_at)).all()
    ]
    return d


@router.patch("/users/{user_id}")
async def update_user(
    user_id: int,
    data: UpdateUserRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    u = db.query(User).filter(User.userId == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if data.role is not None:
        if data.role not in (0, 1, 2):
            raise HTTPException(status_code=400, detail="role must be 0, 1, or 2")
        u.role = data.role
    if data.plan is not None:
        if data.plan not in ("free", "researcher", "organization"):
            raise HTTPException(status_code=400, detail="Invalid plan")
        u.plan = data.plan
    if data.org_name is not None:
        u.org_name = data.org_name
    if data.stripe_customer_id is not None:
        u.stripe_customer_id = data.stripe_customer_id
    if data.stripe_subscription_id is not None:
        u.stripe_subscription_id = data.stripe_subscription_id
    if data.plan_expires_at is not None:
        try:
            u.plan_expires_at = datetime.fromisoformat(data.plan_expires_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid plan_expires_at format")
    db.commit()
    return {"success": True, "user": _user_dict(u)}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    u = db.query(User).filter(User.userId == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    if u.userId == admin.userId:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    db.delete(u)
    db.commit()
    return {"success": True, "message": f"User {user_id} deleted"}


# ─── payment management ───────────────────────────────────────────────────────

@router.get("/payments")
async def list_payments(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    total = db.query(func.count(Payment.id)).scalar() or 0
    payments = (
        db.query(Payment)
        .order_by(desc(Payment.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "payments": [
            {
                "id": p.id,
                "user_id": p.user_id,
                "username": p.user.username if p.user else None,
                "amount": p.amount,
                "currency": p.currency,
                "plan": p.plan,
                "status": p.status,
                "stripe_payment_intent": p.stripe_payment_intent,
                "stripe_invoice_id": p.stripe_invoice_id,
                "created_at": p.created_at.isoformat(),
            }
            for p in payments
        ],
        "total": total,
    }


@router.post("/payments")
async def create_payment(
    data: CreatePaymentRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    u = db.query(User).filter(User.userId == data.user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    payment = Payment(
        user_id=data.user_id,
        amount=data.amount,
        currency=data.currency,
        plan=data.plan,
        status=data.status,
        stripe_payment_intent=data.stripe_payment_intent,
        stripe_invoice_id=data.stripe_invoice_id,
    )
    db.add(payment)
    if data.status == "succeeded":
        u.plan = data.plan
    db.commit()
    return {"success": True, "payment_id": payment.id}


# ─── lab management (admin CRUD) ──────────────────────────────────────────────

class CreateLabRequest(BaseModel):
    title: str
    description: Optional[str] = None
    sensor_type: str  # camera | wifi_sensing | cwmf
    difficulty: str = "beginner"
    questions: List[Dict[str, Any]]
    max_score: int = 100
    is_published: bool = True


@router.get("/labs")
async def admin_list_labs(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    labs = db.query(Lab).order_by(desc(Lab.created_at)).all()
    return {
        "labs": [
            {
                "id": l.id,
                "title": l.title,
                "description": l.description,
                "sensor_type": l.sensor_type,
                "difficulty": l.difficulty,
                "max_score": l.max_score,
                "is_published": l.is_published,
                "submission_count": len(l.submissions),
                "created_at": l.created_at.isoformat(),
            }
            for l in labs
        ]
    }


@router.post("/labs")
async def create_lab(
    data: CreateLabRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = Lab(
        title=data.title,
        description=data.description,
        sensor_type=data.sensor_type,
        difficulty=data.difficulty,
        questions=json.dumps(data.questions),
        max_score=data.max_score,
        is_published=data.is_published,
        created_by=admin.userId,
    )
    db.add(lab)
    db.commit()
    return {"success": True, "lab_id": lab.id}


@router.put("/labs/{lab_id}")
async def update_lab(
    lab_id: int,
    data: CreateLabRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = db.query(Lab).filter(Lab.id == lab_id).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    lab.title = data.title
    lab.description = data.description
    lab.sensor_type = data.sensor_type
    lab.difficulty = data.difficulty
    lab.questions = json.dumps(data.questions)
    lab.max_score = data.max_score
    lab.is_published = data.is_published
    db.commit()
    return {"success": True}


@router.delete("/labs/{lab_id}")
async def delete_lab(
    lab_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = db.query(Lab).filter(Lab.id == lab_id).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    db.delete(lab)
    db.commit()
    return {"success": True}
