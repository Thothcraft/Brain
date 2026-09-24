"""Face assets API tests — basis fit/upload, enrollment, gallery."""
import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import Base, User, get_db
from server.v1.faces import router as faces_router


def _png_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _face(seed: int, size: int = 64) -> np.ndarray:
    """Deterministic synthetic 'face' — distinct pattern per seed."""
    rng = np.random.RandomState(seed)
    base = rng.rand(size, size) * 40
    yy, xx = np.mgrid[0:size, 0:size]
    base += 120 * np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
                         / (2 * (size / (3 + seed % 3)) ** 2))
    return np.clip(base, 0, 255)


@pytest.fixture
def api():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='test', email='t@x.invalid',
                    hashed_password='unused', role=0, plan='free')
        other = User(userId=2, username='other', email='o@x.invalid',
                     hashed_password='unused', role=0, plan='free')
        session.add_all([user, other])
        session.commit()
        app = FastAPI()
        app.include_router(faces_router, prefix='/v1')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user


def test_basis_fit_from_images(api):
    client, _, _ = api
    images = [_png_b64(_face(s)) for s in range(6)]
    r = client.post("/v1/faces/basis", json={
        "name": "default", "image_size": 64, "n_components": 10,
        "images": images})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["n_components"] == 5  # min(10, 6 imgs - 1)
    assert body["image_size"] == 64

    # basis downloadable as npz
    r = client.get("/v1/faces/basis")
    assert r.status_code == 200
    with np.load(io.BytesIO(r.content)) as z:
        assert z["mean"].shape == (64 * 64,)
        assert z["eigenvectors"].shape[0] == 5


def test_basis_requires_images_or_npz(api):
    client, _, _ = api
    r = client.post("/v1/faces/basis", json={"name": "x", "images": []})
    assert r.status_code == 422


def test_enroll_and_gallery(api):
    client, _, _ = api
    client.post("/v1/faces/basis", json={
        "image_size": 64, "n_components": 8,
        "images": [_png_b64(_face(s)) for s in range(8)]})

    r = client.post("/v1/faces/persons", json={
        "name": "gad", "photo_b64": _png_b64(_face(0))})
    assert r.status_code == 201, r.text
    assert len(r.json()["projection"]) == 7  # min(8, 8-1)

    client.post("/v1/faces/persons", json={
        "name": "gad", "photo_b64": _png_b64(_face(0))})
    client.post("/v1/faces/persons", json={
        "name": "sara", "photo_b64": _png_b64(_face(1))})

    r = client.get("/v1/faces/persons")
    assert len(r.json()["persons"]) == 3
    r = client.get("/v1/faces/persons", params={"name": "gad"})
    assert len(r.json()["persons"]) == 2

    g = client.get("/v1/faces/gallery").json()
    assert g["basis_id"]
    assert g["max_distance"] > 0          # calibrated on enroll
    names = {p["name"] for p in g["persons"]}
    assert names == {"gad", "sara"}
    assert all(len(p["projection"]) == 7 for p in g["persons"])


def test_enroll_without_basis_404(api):
    client, _, _ = api
    r = client.post("/v1/faces/persons", json={
        "name": "gad", "photo_b64": _png_b64(_face(0))})
    assert r.status_code == 404


def test_delete_person_asset(api):
    client, _, _ = api
    client.post("/v1/faces/basis", json={
        "images": [_png_b64(_face(s)) for s in range(4)]})
    r = client.post("/v1/faces/persons", json={
        "name": "gad", "photo_b64": _png_b64(_face(0))})
    asset_id = r.json()["id"]
    assert client.delete(f"/v1/faces/persons/{asset_id}").json()["ok"]
    assert client.get("/v1/faces/persons").json()["persons"] == []
    assert client.delete("/v1/faces/persons/9999").status_code == 404


def test_gallery_tenant_isolation(api):
    client, session, _ = api
    client.post("/v1/faces/basis", json={
        "images": [_png_b64(_face(s)) for s in range(4)]})
    client.post("/v1/faces/persons", json={
        "name": "gad", "photo_b64": _png_b64(_face(0))})
    # other user's view is empty
    other = session.query(User).filter(User.userId == 2).first()
    app = FastAPI()
    app.include_router(faces_router, prefix='/v1')
    app.dependency_overrides[get_current_user] = lambda: other
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as c2:
        assert c2.get("/v1/faces/gallery").status_code == 404
        assert c2.get("/v1/faces/persons").json()["persons"] == []
