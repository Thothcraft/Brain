"""Labs endpoints — accessible by approved org members and admins."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import Lab, LabSubmission, OrgMembership, User, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labs", tags=["labs"])


def _approved_org_ids(user_id: int, db: Session) -> List[int]:
    """Return list of org_ids the user is an approved member of."""
    rows = db.query(OrgMembership.org_id).filter(
        OrgMembership.member_id == user_id,
        OrgMembership.status == "approved"
    ).all()
    return [r[0] for r in rows]


def _lab_dict(lab: Lab, include_answers: bool = False) -> Dict[str, Any]:
    questions = json.loads(lab.questions) if lab.questions else []
    if not include_answers:
        for q in questions:
            q.pop("correct_answer", None)
    return {
        "id": lab.id,
        "title": lab.title,
        "description": lab.description,
        "sensor_type": lab.sensor_type,
        "difficulty": lab.difficulty,
        "max_score": lab.max_score,
        "is_published": lab.is_published,
        "questions": questions,
        "created_at": lab.created_at.isoformat(),
    }


# ─── request models ───────────────────────────────────────────────────────────

class SubmitLabRequest(BaseModel):
    answers: Dict[str, Any]   # {question_id: answer}


# ─── list labs ────────────────────────────────────────────────────────────────

@router.get("")
async def list_labs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    sensor_type: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Return published labs.
    - Admins see all.
    - Org accounts see all (to track member progress).
    - Regular users see labs only if they are an approved member of at least one org.
    """
    if current_user.role == 1:
        # Admin sees everything
        q = db.query(Lab).filter(Lab.is_published == True)
    elif current_user.role == 2:
        # Org account sees all published labs
        q = db.query(Lab).filter(Lab.is_published == True)
    else:
        # Regular user — must be approved member of at least one org
        org_ids = _approved_org_ids(current_user.userId, db)
        if not org_ids:
            return {"labs": [], "message": "Join an organization to access labs"}
        q = db.query(Lab).filter(Lab.is_published == True)

    if sensor_type:
        q = q.filter(Lab.sensor_type == sensor_type)

    labs = q.order_by(Lab.created_at).all()

    # Attach user's own submission info
    result = []
    for lab in labs:
        d = _lab_dict(lab)
        sub = db.query(LabSubmission).filter(
            LabSubmission.lab_id == lab.id,
            LabSubmission.user_id == current_user.userId
        ).first()
        d["my_submission"] = {
            "submitted": sub is not None,
            "score": sub.score if sub else None,
            "max_score": sub.max_score if sub else lab.max_score,
            "submitted_at": sub.submitted_at.isoformat() if sub else None,
        }
        result.append(d)

    return {"labs": result}


@router.get("/{lab_id}")
async def get_lab(
    lab_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = db.query(Lab).filter(Lab.id == lab_id, Lab.is_published == True).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")

    # Permission check for regular users
    if current_user.role == 0:
        org_ids = _approved_org_ids(current_user.userId, db)
        if not org_ids:
            raise HTTPException(status_code=403, detail="Must be an approved org member")

    # Admin sees answers; others do not
    include_answers = current_user.role == 1
    return _lab_dict(lab, include_answers=include_answers)


# ─── submit answers ───────────────────────────────────────────────────────────

@router.post("/{lab_id}/submit")
async def submit_lab(
    lab_id: int,
    data: SubmitLabRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = db.query(Lab).filter(Lab.id == lab_id, Lab.is_published == True).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")

    # Must be approved org member
    org_ids = _approved_org_ids(current_user.userId, db)
    if not org_ids and current_user.role != 1:
        raise HTTPException(status_code=403, detail="Must be an approved org member")

    # Use first org if multiple
    org_id = org_ids[0] if org_ids else current_user.userId

    # Check for existing submission
    existing = db.query(LabSubmission).filter(
        LabSubmission.lab_id == lab_id,
        LabSubmission.user_id == current_user.userId
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already submitted this lab")

    # Auto-grade
    questions = json.loads(lab.questions) if lab.questions else []
    score = 0.0
    total = 0
    for q in questions:
        qid = str(q["id"])
        correct = q.get("correct_answer")
        if correct is not None:
            total += 1
            if str(data.answers.get(qid, "")).strip().lower() == str(correct).strip().lower():
                score += 1

    pct_score = round((score / total) * lab.max_score, 1) if total > 0 else None

    submission = LabSubmission(
        lab_id=lab_id,
        user_id=current_user.userId,
        org_id=org_id,
        answers=json.dumps(data.answers),
        score=pct_score,
        max_score=lab.max_score,
        graded_at=datetime.utcnow(),
    )
    db.add(submission)
    db.commit()

    return {
        "success": True,
        "score": pct_score,
        "max_score": lab.max_score,
        "correct": int(score),
        "total": total,
    }


# ─── org progress tracking ────────────────────────────────────────────────────

@router.get("/{lab_id}/progress")
async def lab_progress(
    lab_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Org accounts and admins can view all member submissions for a lab.
    """
    if current_user.role not in (1, 2):
        raise HTTPException(status_code=403, detail="Org or admin access required")

    lab = db.query(Lab).filter(Lab.id == lab_id).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")

    # For org: only show submissions from their members
    if current_user.role == 2:
        member_ids = [
            m.member_id for m in db.query(OrgMembership).filter(
                OrgMembership.org_id == current_user.userId,
                OrgMembership.status == "approved"
            ).all()
        ]
        submissions = db.query(LabSubmission).filter(
            LabSubmission.lab_id == lab_id,
            LabSubmission.user_id.in_(member_ids)
        ).all()
    else:
        submissions = db.query(LabSubmission).filter(
            LabSubmission.lab_id == lab_id
        ).all()

    return {
        "lab_id": lab_id,
        "lab_title": lab.title,
        "submissions": [
            {
                "user_id": s.user_id,
                "username": s.user.username if s.user else None,
                "score": s.score,
                "max_score": s.max_score,
                "submitted_at": s.submitted_at.isoformat() if s.submitted_at else None,
            }
            for s in submissions
        ],
        "total_submitted": len(submissions),
    }


@router.get("/progress/all")
async def all_labs_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Org: summary of all lab progress for their approved members.
    """
    if current_user.role not in (1, 2):
        raise HTTPException(status_code=403, detail="Org or admin access required")

    labs = db.query(Lab).filter(Lab.is_published == True).all()

    if current_user.role == 2:
        member_ids = [
            m.member_id for m in db.query(OrgMembership).filter(
                OrgMembership.org_id == current_user.userId,
                OrgMembership.status == "approved"
            ).all()
        ]
    else:
        member_ids = None  # admin sees all

    result = []
    for lab in labs:
        q = db.query(LabSubmission).filter(LabSubmission.lab_id == lab.id)
        if member_ids is not None:
            q = q.filter(LabSubmission.user_id.in_(member_ids))
        subs = q.all()
        avg_score = (
            sum(s.score for s in subs if s.score is not None) / len(subs)
            if subs else None
        )
        result.append({
            "lab_id": lab.id,
            "lab_title": lab.title,
            "sensor_type": lab.sensor_type,
            "difficulty": lab.difficulty,
            "total_members": len(member_ids) if member_ids is not None else None,
            "submitted_count": len(subs),
            "avg_score": round(avg_score, 1) if avg_score is not None else None,
            "max_score": lab.max_score,
        })

    return {"labs": result}
