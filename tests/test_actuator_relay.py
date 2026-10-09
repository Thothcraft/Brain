"""/v1/devices/{id}/actuators — list + action relay over the node tunnel."""
import json
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from server.auth import get_current_user
from server.db import Base, Device, User, get_db
from server.endpoints import node_ws
from server.v1 import router as v1_router

ACTS = [{"id": "ha-light", "kind": "light", "name": "Desk lamp"}]


@pytest.fixture
def api(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    engine = create_engine(f"sqlite:///{tmp}", poolclass=NullPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    user = User(userId=1, username="t", email="t@example.invalid",
                hashed_password="x", role=0, plan="free")
    dev = Device(userId=1, device_uuid="dev-1", device_name="n",
                 device_type="thoth", approved=True,
                 hardware_info=json.dumps({"actuators": ACTS}))
    with factory() as s:
        s.add_all([user, dev])
        s.commit()

    calls = []

    async def fake_request(device_id, method, path, body=None):
        calls.append((device_id, method, path, body))
        return fake_request.reply

    fake_request.reply = None
    monkeypatch.setattr(node_ws.manager, "request", fake_request)

    app = FastAPI()
    app.include_router(v1_router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: user

    def _db():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db
    with TestClient(app) as c:
        yield c, fake_request, calls


def test_list_falls_back_to_hardware_info_when_offline(api):
    c, fake, _ = api
    r = c.get("/v1/devices/dev-1/actuators")
    assert r.status_code == 200
    assert r.json() == {"device_id": "dev-1", "actuators": ACTS, "live": False}


def test_list_live_from_node(api):
    c, fake, calls = api
    live = [{"id": "buzzer", "kind": "buzzer"}]
    fake.reply = {"status": 200, "body": {"actuators": live}}
    r = c.get("/v1/devices/dev-1/actuators")
    assert r.json()["actuators"] == live and r.json()["live"] is True
    assert calls[-1][:3] == ("dev-1", "GET", "/api/v1/actuators")


def test_action_relays_command_and_status(api):
    c, fake, calls = api
    fake.reply = {"status": 200, "body": {"status": "succeeded"}}
    cmd = {"operation": "turn_on"}
    r = c.post("/v1/devices/dev-1/actuators/ha-light/actions", json=cmd)
    assert r.status_code == 200 and r.json() == {"status": "succeeded"}
    assert calls[-1] == ("dev-1", "POST",
                         "/api/v1/actuators/ha-light/actions", cmd)


def test_action_offline_is_503(api):
    c, _, _ = api
    r = c.post("/v1/devices/dev-1/actuators/ha-light/actions",
               json={"operation": "turn_on"})
    assert r.status_code == 503


def test_unknown_device_404(api):
    c, _, _ = api
    assert c.get("/v1/devices/nope/actuators").status_code == 404
