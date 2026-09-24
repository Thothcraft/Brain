"""Brain v1 face assets API — the eigenface asset store.

Implements the pipeline from gadm21/Face-recognition-using-PCA-and-SVD
as a service: a per-user PCA basis (mean face + eigenvectors) fitted
from enrolled photos and/or a face dataset, and a ``person_asset``
gallery of enrolled photos with their stored projections.

Edge devices pull ``GET /faces/basis`` (npz) + ``GET /faces/gallery``
and run ``pca-face-recognizer`` (whispy-model-face) locally: crop →
project → nearest gallery projection → ``person:<name>`` when the
distance is under ``max_distance`` ("very close"), else
``person:unknown``.

All rows are user-scoped — tenant isolation everywhere.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import get_current_user
from server.db import FaceBasis, PersonAsset, User, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/faces", tags=["v1", "faces"])

DEFAULT_IMAGE_SIZE = 64
DEFAULT_COMPONENTS = 40
DEFAULT_THRESHOLD_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Eigenface math (server-side; mirrors whispy_model_face.basis)
# ---------------------------------------------------------------------------

def _decode_image_b64(data_b64: str, image_size: int):
    """base64 image → flattened float32 gray vector in [0,1]."""
    from PIL import Image
    try:
        raw = base64.b64decode(data_b64)
        img = Image.open(io.BytesIO(raw)).convert("L")
        img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception as exc:
        raise HTTPException(422, f"undecodable image: {exc}")


def _fit_basis(vectors: List[np.ndarray], n_components: int) -> Dict[str, Any]:
    """SVD on the centered data matrix — same math as cv2.PCACompute."""
    A = np.stack(vectors)
    mean = A.mean(axis=0)
    k = max(1, min(int(n_components), min(A.shape) - 1))
    _, _, vt = np.linalg.svd(A - mean, full_matrices=False)
    return {"mean": mean.astype(np.float32),
            "eigenvectors": vt[:k].astype(np.float32)}


def _serialize_basis(mean, eigenvectors, image_size, max_distance) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, mean=mean, eigenvectors=eigenvectors,
             image_size=np.int64(image_size),
             max_distance=np.float64(max_distance or 0.0))
    return buf.getvalue()


def _load_basis(row: FaceBasis) -> Dict[str, Any]:
    with np.load(io.BytesIO(row.data)) as z:
        return {"mean": z["mean"].astype(np.float32),
                "eigenvectors": z["eigenvectors"].astype(np.float32),
                "image_size": int(z["image_size"]),
                "max_distance": float(z["max_distance"])}


def _project(basis: Dict[str, Any], vec: np.ndarray) -> List[float]:
    centered = vec - basis["mean"]
    return np.dot(centered, basis["eigenvectors"].T).tolist()


def _calibrate(projections: List[List[float]], sigma: float) -> float:
    """mean + sigma·std of each projection's distance to the centroid —
    the "very close" cutoff (the reference repo has no threshold)."""
    if not projections:
        return 0.0
    arr = np.asarray(projections, dtype=np.float32)
    centroid = arr.mean(axis=0)
    dists = [float(np.sqrt(np.sum((p - centroid) ** 2))) for p in arr]
    return float(np.mean(dists) + sigma * np.std(dists))


def _get_basis_row(db: Session, user_id: int,
                   basis_id: Optional[int] = None) -> FaceBasis:
    q = db.query(FaceBasis).filter(FaceBasis.user_id == user_id)
    row = (q.filter(FaceBasis.id == basis_id).first() if basis_id
           else q.order_by(FaceBasis.id.desc()).first())
    if row is None:
        raise HTTPException(404, "no face basis — POST /v1/faces/basis first")
    return row


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class BasisIn(BaseModel):
    name: str = "default"
    image_size: int = Field(default=DEFAULT_IMAGE_SIZE, ge=8, le=512)
    n_components: int = Field(default=DEFAULT_COMPONENTS, ge=1, le=1024)
    # Either pre-fitted basis bytes or raw images to fit from.
    npz_b64: Optional[str] = None
    images: List[str] = Field(default_factory=list)   # base64 photos
    max_distance: float = 0.0                          # 0 → calibrate later


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    photo_b64: str                                     # enrolled face photo
    photo_mime: str = "image/jpeg"
    basis_id: Optional[int] = None                     # default: latest


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

@router.post("/basis", status_code=201)
async def upsert_basis(
    body: BasisIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Store a PCA basis: upload pre-fitted .npz or fit from images."""
    if body.npz_b64:
        try:
            data = base64.b64decode(body.npz_b64)
            with np.load(io.BytesIO(data)) as z:
                image_size = int(z["image_size"])
                n_components = int(z["eigenvectors"].shape[0])
        except Exception as exc:
            raise HTTPException(422, f"invalid npz basis: {exc}")
    elif len(body.images) >= 2:
        vecs = [_decode_image_b64(b, body.image_size) for b in body.images]
        fitted = _fit_basis(vecs, body.n_components)
        image_size, n_components = body.image_size, \
            int(fitted["eigenvectors"].shape[0])
        data = _serialize_basis(fitted["mean"], fitted["eigenvectors"],
                                image_size, body.max_distance)
    else:
        raise HTTPException(422, "provide npz_b64 or >=2 images")

    row = db.query(FaceBasis).filter(
        FaceBasis.user_id == current_user.userId,
        FaceBasis.name == body.name).first()
    if row is None:
        row = FaceBasis(user_id=current_user.userId, name=body.name)
        db.add(row)
    row.image_size = image_size
    row.n_components = n_components
    row.max_distance = body.max_distance
    row.data = data
    db.commit()
    db.refresh(row)
    return row.to_dict()


@router.get("/basis")
async def get_basis(
    name: str = "default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Latest basis as .npz bytes — the recognizer's model artifact."""
    row = db.query(FaceBasis).filter(
        FaceBasis.user_id == current_user.userId,
        FaceBasis.name == name).first()
    if row is None:
        raise HTTPException(404, "basis not found")
    return Response(content=bytes(row.data),
                    media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Persons (enrollment)
# ---------------------------------------------------------------------------

@router.post("/persons", status_code=201)
async def enroll_person(
    body: PersonIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Enroll a photo of a known person → stored PCA projection."""
    basis_row = _get_basis_row(db, current_user.userId, body.basis_id)
    basis = _load_basis(basis_row)
    vec = _decode_image_b64(body.photo_b64, basis["image_size"])
    projection = _project(basis, vec)

    asset = PersonAsset(
        user_id=current_user.userId, name=body.name,
        basis_id=basis_row.id, projection=json.dumps(projection),
        photo=base64.b64decode(body.photo_b64),
        photo_mime=body.photo_mime)
    db.add(asset)

    # Re-calibrate the basis threshold against the updated gallery.
    rows = db.query(PersonAsset).filter(
        PersonAsset.user_id == current_user.userId,
        PersonAsset.basis_id == basis_row.id).all()
    projections = [json.loads(r.projection) for r in rows] + [projection]
    basis_row.max_distance = _calibrate(projections,
                                        DEFAULT_THRESHOLD_SIGMA)
    basis_dict = _load_basis(basis_row)
    basis_row.data = _serialize_basis(
        basis_dict["mean"], basis_dict["eigenvectors"],
        basis_dict["image_size"], basis_row.max_distance)
    db.commit()
    db.refresh(asset)
    return asset.to_dict()


@router.get("/persons")
async def list_persons(
    name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    q = db.query(PersonAsset).filter(PersonAsset.user_id == current_user.userId)
    if name:
        q = q.filter(PersonAsset.name == name)
    return {"persons": [a.to_dict() for a in
                        q.order_by(PersonAsset.name, PersonAsset.id).all()]}


@router.delete("/persons/{asset_id}")
async def delete_person_asset(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    row = db.query(PersonAsset).filter(
        PersonAsset.user_id == current_user.userId,
        PersonAsset.id == asset_id).first()
    if row is None:
        raise HTTPException(404, "person asset not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "id": str(asset_id)}


# ---------------------------------------------------------------------------
# Gallery — what the edge recognizer pulls
# ---------------------------------------------------------------------------

@router.get("/gallery")
async def get_gallery(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Latest basis metadata + all enrolled projections."""
    basis_row = _get_basis_row(db, current_user.userId)
    rows = db.query(PersonAsset).filter(
        PersonAsset.user_id == current_user.userId,
        PersonAsset.basis_id == basis_row.id).all()
    return {
        "basis_id": str(basis_row.id),
        "image_size": basis_row.image_size,
        "n_components": basis_row.n_components,
        "max_distance": basis_row.max_distance or 0.0,
        "persons": [r.to_dict() for r in rows],
    }
