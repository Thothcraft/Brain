"""Research Labs endpoints — gated by the ``labs`` plan entitlement.

A Lab is a reproducible computational experiment: the researcher
downloads a notebook template, works through it with thothcraft-sdk,
and submits the completed .ipynb. Grading goes through the LabGrader
abstraction; no untrusted notebook is executed by Brain.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi import File as FormFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import Lab, LabSubmission, User, get_db
from server.entitlements import require_entitlement
from server.lab_grader import (
    NotebookValidationError,
    get_grader,
    validate_notebook,
)
from server.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labs", tags=["labs"])

SUBMISSION_DIR = Path(os.getenv("LAB_SUBMISSION_DIR", "data/lab_submissions"))
TEMPLATE_DIR = Path(os.getenv("LAB_TEMPLATE_DIR", "lab-templates"))

# ── track catalog ─────────────────────────────────────────────────────────────

LAB_TRACKS: Dict[str, Dict[str, Any]] = {
    "sensing_fundamentals": {
        "title": "Sensing Fundamentals",
        "description": "CSI, radar, camera and environmental sensing basics.",
    },
    "dataset_engineering": {
        "title": "Dataset Engineering",
        "description": "Collection, annotation, split design and leakage prevention.",
    },
    "machine_learning": {
        "title": "Machine Learning",
        "description": "Features, baselines, evaluation and deployment.",
    },
    "deep_learning": {
        "title": "Deep Learning",
        "description": "Windowed series, CNN/LSTM/Transformer, fusion, edge deployment.",
    },
    "wifi_csi": {
        "title": "Wi-Fi CSI",
        "description": "Subcarrier processing, occupancy, localization, HAR, domain shift.",
    },
    "radar": {
        "title": "Radar",
        "description": "Range/Doppler processing, detection, occupancy, HAR, fusion.",
    },
    "robotics": {
        "title": "Robotics",
        "description": "Thoth + ROS2: perception, localization, decision, action.",
    },
}


# ── serialization ─────────────────────────────────────────────────────────────

def _lab_dict(lab: Lab) -> Dict[str, Any]:
    return {
        "id": lab.id,
        "slug": lab.slug,
        "title": lab.title,
        "description": lab.description,
        "track": lab.track,
        "track_title": (LAB_TRACKS.get(lab.track) or {}).get("title"),
        "level": lab.level,
        "order_in_track": lab.order_in_track,
        "objectives": json.loads(lab.objectives) if lab.objectives else [],
        "required_artifacts": json.loads(lab.required_artifacts) if lab.required_artifacts else [],
        "has_template": bool(lab.template_path),
        "max_score": lab.max_score,
        "is_published": lab.is_published,
        "created_at": lab.created_at.isoformat() if lab.created_at else None,
    }


def _submission_dict(sub: LabSubmission) -> Dict[str, Any]:
    return {
        "id": sub.id,
        "lab_id": sub.lab_id,
        "status": sub.status,
        "execution_status": sub.execution_status,
        "score": sub.score,
        "max_score": sub.max_score,
        "passed": sub.passed,
        "feedback": json.loads(sub.feedback) if sub.feedback else [],
        "artifacts": json.loads(sub.artifacts) if sub.artifacts else [],
        "notebook_metadata": json.loads(sub.notebook_metadata) if sub.notebook_metadata else None,
        "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
        "graded_at": sub.graded_at.isoformat() if sub.graded_at else None,
    }


def _my_submission(lab_id: int, user_id: int, db: Session) -> Optional[LabSubmission]:
    return db.query(LabSubmission).filter(
        LabSubmission.lab_id == lab_id,
        LabSubmission.user_id == user_id,
    ).first()


# ── tracks ────────────────────────────────────────────────────────────────────

@router.get("/tracks")
async def list_tracks(
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Track catalog with per-track lab counts and user progress."""
    tracks = []
    for key, meta in LAB_TRACKS.items():
        labs = db.query(Lab).filter(Lab.track == key, Lab.is_published == True)\
            .order_by(Lab.order_in_track).all()
        done = 0
        if labs:
            done = db.query(LabSubmission).filter(
                LabSubmission.user_id == current_user.userId,
                LabSubmission.status == "graded",
                LabSubmission.lab_id.in_([l.id for l in labs]),
            ).count()
        tracks.append({
            "id": key,
            **meta,
            "lab_count": len(labs),
            "completed": done,
        })
    return {"tracks": tracks}


# ── list / detail ─────────────────────────────────────────────────────────────

@router.get("")
async def list_labs(
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
    track: Optional[str] = Query(None),
) -> Dict[str, Any]:
    q = db.query(Lab).filter(Lab.is_published == True)
    if track:
        q = q.filter(Lab.track == track)
    labs = q.order_by(Lab.track, Lab.order_in_track).all()

    result = []
    for lab in labs:
        d = _lab_dict(lab)
        sub = _my_submission(lab.id, current_user.userId, db)
        d["my_submission"] = _submission_dict(sub) if sub else None
        result.append(d)
    return {"labs": result}


@router.get("/{lab_id}")
async def get_lab(
    lab_id: int,
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    lab = db.query(Lab).filter(Lab.id == lab_id, Lab.is_published == True).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    d = _lab_dict(lab)
    sub = _my_submission(lab.id, current_user.userId, db)
    d["my_submission"] = _submission_dict(sub) if sub else None
    return d


@router.get("/{lab_id}/template")
async def get_template(
    lab_id: int,
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
):
    """Download the lab's notebook template."""
    lab = db.query(Lab).filter(Lab.id == lab_id, Lab.is_published == True).first()
    if not lab or not lab.template_path:
        raise HTTPException(status_code=404, detail="No template for this lab")
    path = TEMPLATE_DIR / lab.template_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Template file missing")
    return FileResponse(path, filename=path.name, media_type="application/x-ipynb+json")


# ── submission ────────────────────────────────────────────────────────────────

@router.post("/{lab_id}/submit")
async def submit_lab(
    lab_id: int,
    notebook: UploadFile = FormFile(...),
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
    _rate=Depends(rate_limit("lab_submit", 10, 60)),
) -> Dict[str, Any]:
    """Submit a completed .ipynb for grading.

    Validates notebook structure, extracts metadata, persists the file,
    then runs the configured LabGrader (structural checks only — no
    execution). Resubmission replaces the previous attempt.
    """
    lab = db.query(Lab).filter(Lab.id == lab_id, Lab.is_published == True).first()
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")

    filename = notebook.filename or ""
    if not filename.endswith(".ipynb"):
        raise HTTPException(status_code=422, detail="Submission must be a .ipynb file")

    raw = await notebook.read()
    try:
        nb_meta = validate_notebook(raw)
    except NotebookValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Persist notebook
    dest_dir = SUBMISSION_DIR / str(current_user.userId) / str(lab.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "submission.ipynb"
    dest.write_bytes(raw)

    lab_spec = {
        "required_artifacts": json.loads(lab.required_artifacts) if lab.required_artifacts else [],
        "max_score": lab.max_score,
        "track": lab.track,
        "level": lab.level,
    }
    rubric = json.loads(lab.rubric) if lab.rubric else None

    result = get_grader().grade(str(dest), lab_spec, rubric, nb_meta)

    sub = _my_submission(lab.id, current_user.userId, db)
    if sub is None:
        sub = LabSubmission(lab_id=lab.id, user_id=current_user.userId,
                            notebook_path=str(dest))
        db.add(sub)

    sub.notebook_path = str(dest)
    sub.notebook_metadata = json.dumps(nb_meta)
    sub.status = "graded" if result.score is not None else "pending"
    sub.execution_status = result.execution_status
    sub.score = result.score
    sub.max_score = result.max_score or lab.max_score
    sub.passed = result.passed
    sub.feedback = json.dumps(result.feedback)
    sub.artifacts = json.dumps(result.artifacts)
    sub.submitted_at = datetime.utcnow()
    sub.graded_at = datetime.utcnow() if result.score is not None else None
    db.commit()

    return {"success": True, "submission": _submission_dict(sub)}


@router.get("/{lab_id}/submission")
async def get_submission(
    lab_id: int,
    current_user: User = Depends(require_entitlement("labs")),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    sub = _my_submission(lab_id, current_user.userId, db)
    if not sub:
        raise HTTPException(status_code=404, detail="No submission for this lab")
    return {"submission": _submission_dict(sub)}
