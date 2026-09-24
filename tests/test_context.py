"""Context model API tests — entities, relationships, evidence, state, events."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import Base, User, get_db
from server.v1.context import router as context_router


@pytest.fixture
def api():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='test', email='test@example.invalid',
                    hashed_password='unused', role=0, plan='free')
        session.add(user)
        session.commit()
        app = FastAPI()
        app.include_router(context_router, prefix='/v1')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user
    engine.dispose()


def test_entity_crud(api):
    client, _, _ = api
    r = client.post("/v1/context/entities", json={
        "id": "person:gad", "kind": "person", "name": "Gad",
        "attributes": {"role": "owner"}})
    assert r.status_code == 201
    assert r.json()["id"] == "person:gad"
    # upsert updates name
    r = client.post("/v1/context/entities", json={
        "id": "person:gad", "kind": "person", "name": "Gad A"})
    assert r.json()["name"] == "Gad A"
    r = client.get("/v1/context/entities")
    assert len(r.json()["entities"]) == 1
    r = client.get("/v1/context/entities", params={"kind": "device"})
    assert r.json()["entities"] == []
    r = client.delete("/v1/context/entities/person:gad")
    assert r.json()["ok"] is True
    assert client.get("/v1/context/entities").json()["entities"] == []


def test_relationship_lifecycle(api):
    client, _, _ = api
    r = client.post("/v1/context/relationships", json={
        "subject": "person:gad", "predicate": "carries",
        "object": "device:phone", "confidence": 0.9})
    assert r.status_code == 201
    rel_id = r.json()["id"]
    r = client.get("/v1/context/relationships",
                   params={"active_only": True})
    assert len(r.json()["relationships"]) == 1
    # end it — history preserved, no longer active
    r = client.delete(f"/v1/context/relationships/{rel_id}")
    assert r.json()["valid_until"]
    assert client.get("/v1/context/relationships",
                      params={"active_only": True}).json()["relationships"] == []
    assert len(client.get("/v1/context/relationships").json()["relationships"]) == 1


def test_evidence_ingest_and_query(api):
    client, _, _ = api
    r = client.post("/v1/context/evidence", json={
        "items": [
            {"key": "spatial.presence/v1", "value": "occupied",
             "source_id": "radar-main", "device_id": "pi2",
             "model_id": "occ-v2", "confidence": 0.95,
             "execution_class": "local"},
            {"key": "digital.foreground_application/v1",
             "value": "code.exe", "source_id": "fgapp-1"},
        ]})
    assert r.status_code == 201
    assert len(r.json()["evidence"]) == 2
    r = client.get("/v1/context/evidence",
                   params={"key": "spatial.presence/v1"})
    ev = r.json()["evidence"]
    assert len(ev) == 1
    assert ev[0]["model_id"] == "occ-v2"
    assert ev[0]["confidence"] == 0.95


def test_state_upsert_emits_events(api):
    client, _, _ = api
    # first state → "entered" event
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "confidence": 0.9,
        "evidence_ids": ["ev-1"], "estimator": "weighted-v1"})
    assert r.json()["key"] == "spatial.occupancy/v1"
    # same value → no event
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab"})
    # change → "exited" (occupied → empty)
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "empty",
        "entity_id": "space:lab"})
    events = client.get("/v1/context/events",
                        params={"key": "spatial.occupancy/v1"}).json()["events"]
    types = [e["event_type"] for e in events]
    assert types == ["exited", "entered"]  # newest first
    assert events[0]["previous_value"] == "occupied"


def test_snapshot(api):
    client, _, _ = api
    client.post("/v1/context/entities", json={
        "id": "space:lab", "kind": "space", "name": "Lab"})
    client.post("/v1/context/entities", json={
        "id": "device:pi2", "kind": "device", "name": "Pi 2"})
    client.post("/v1/context/relationships", json={
        "subject": "device:pi2", "predicate": "located_in",
        "object": "space:lab"})
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab"})
    snap = client.get("/v1/context/snapshot").json()
    assert len(snap["entities"]) == 2
    assert len(snap["relationships"]) == 1
    assert snap["relationships"][0]["predicate"] == "located_in"
    assert len(snap["states"]) == 1
    assert "generated_at" in snap


def test_tenant_isolation(api):
    """Context rows are user-scoped — a second user sees nothing."""
    client, session, _ = api
    client.post("/v1/context/entities", json={
        "id": "person:gad", "kind": "person"})
    other = User(userId=2, username='other', email='o@x.invalid',
                 hashed_password='x', role=0, plan='free')
    session.add(other)
    session.commit()
    app = client.app
    app.dependency_overrides[get_current_user] = lambda: other
    assert client.get("/v1/context/entities").json()["entities"] == []
    assert client.get("/v1/context/snapshot").json()["entities"] == []
