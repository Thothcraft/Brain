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
    assert r.json()["retired_at"]
    assert client.get("/v1/context/entities").json()["entities"] == []
    # soft-deleted entity can be resurrected by a later upsert
    r = client.post("/v1/context/entities", json={
        "id": "person:gad", "kind": "person", "name": "Gad"})
    assert r.status_code == 201
    assert len(client.get("/v1/context/entities").json()["entities"]) == 1


def test_relationship_lifecycle(api):
    client, _, _ = api
    for key, kind in (("person:gad", "person"), ("device:phone", "device")):
        client.post("/v1/context/entities", json={"id": key, "kind": kind})
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


def test_relationship_requires_resolved_endpoints(api):
    client, _, _ = api
    r = client.post("/v1/context/relationships", json={
        "subject": "person:gad", "predicate": "carries",
        "object": "device:nope"})
    assert r.status_code == 422
    assert "device:nope" in r.json()["detail"]["missing"]
    # explicit opt-in for external/unresolved graph objects
    r = client.post("/v1/context/relationships", json={
        "subject": "person:gad", "predicate": "carries",
        "object": "device:nope", "allow_unresolved": True})
    assert r.status_code == 201


def test_relationship_active_window(api):
    """active_only means valid *now*: valid_from<=now AND (until NULL or >now)."""
    import time as _t
    client, _, _ = api
    now = _t.time()
    base = {"subject": "a", "predicate": "p", "allow_unresolved": True}
    client.post("/v1/context/relationships", json={
        **base, "object": "future", "valid_from": now + 3600})
    client.post("/v1/context/relationships", json={
        **base, "object": "current", "valid_from": now - 10,
        "valid_until": now + 3600})
    client.post("/v1/context/relationships", json={
        **base, "object": "expired", "valid_from": now - 7200,
        "valid_until": now - 3600})
    active = client.get("/v1/context/relationships",
                        params={"active_only": True}).json()["relationships"]
    assert [r["object"] for r in active] == ["current"]


def test_relationship_confidence_bounds(api):
    client, _, _ = api
    r = client.post("/v1/context/relationships", json={
        "subject": "a", "predicate": "p", "object": "b",
        "confidence": 15, "allow_unresolved": True})
    assert r.status_code == 422


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


def _evidence_id(client, **kw) -> str:
    r = client.post("/v1/context/evidence", json={
        "key": "spatial.presence/v1", "value": "occupied", **kw})
    return r.json()["evidence"][0]["id"]


def test_state_upsert_emits_events(api):
    client, _, _ = api
    ev_id = _evidence_id(client)
    # first state → "entered" event
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "confidence": 0.9,
        "evidence_ids": [ev_id], "estimator": "weighted-v1"})
    assert r.json()["key"] == "spatial.occupancy/v1"
    # same value → no event
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "estimator": "weighted-v1"})
    # change → generic "changed" (no hardcoded absent/empty semantics)
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "empty",
        "entity_id": "space:lab", "estimator": "weighted-v1"})
    events = client.get("/v1/context/events",
                        params={"key": "spatial.occupancy/v1"}).json()["events"]
    types = [e["event_type"] for e in events]
    assert types == ["changed", "entered"]  # newest first
    assert events[0]["previous_value"] == "occupied"


def test_state_transition_semantics_come_from_estimator(api):
    """The store knows changed/unchanged; entered/exited is estimator input."""
    client, _, _ = api
    client.post("/v1/context/state", json={
        "key": "personal.activity/v1", "value": "walking",
        "estimator": "act-v1"})
    # walking → stationary is a change, not an "exit"
    client.post("/v1/context/state", json={
        "key": "personal.activity/v1", "value": "stationary",
        "estimator": "act-v1"})
    # estimator may supply semantic transition typing
    client.post("/v1/context/state", json={
        "key": "personal.activity/v1", "value": None,
        "estimator": "act-v1", "transition": "exited"})
    events = client.get("/v1/context/events",
                        params={"key": "personal.activity/v1"}).json()["events"]
    assert [e["event_type"] for e in events] == ["exited", "changed", "entered"]


def test_state_since_resets_on_transition(api):
    """occupied since 10:14, not since the 09:00 'empty' observation."""
    client, _, _ = api
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "empty",
        "entity_id": "space:lab", "since": 1000.0, "estimator": "e"})
    # unchanged value → since preserved
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "empty",
        "entity_id": "space:lab", "estimator": "e"})
    assert r.json()["since"] == 1000.0
    # transition → since resets to now (or explicit body.since)
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "since": 2000.0, "estimator": "e"})
    assert r.json()["since"] == 2000.0


def test_state_requires_estimator(api):
    """State is estimator output — unattributed writes are rejected."""
    client, _, _ = api
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied"})
    assert r.status_code == 422


def test_state_evidence_refs_must_exist(api):
    client, _, _ = api
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "evidence_ids": ["99999"], "estimator": "e"})
    assert r.status_code == 422
    ev_id = _evidence_id(client)
    r = client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "evidence_ids": [ev_id], "estimator": "e"})
    assert r.status_code == 200


def test_state_confidence_bounds(api):
    client, _, _ = api
    r = client.post("/v1/context/state", json={
        "key": "k", "value": "v", "confidence": -3, "estimator": "e"})
    assert r.status_code == 422


def test_global_state_upsert_is_unique(api):
    """entity_id '' sentinel: repeated global upserts update one row."""
    client, session, _ = api
    from server.db import ContextState
    for value in ("online", "offline"):
        client.post("/v1/context/state", json={
            "key": "device.network/v1", "value": value, "estimator": "e"})
    rows = session.query(ContextState).filter(
        ContextState.state_key == "device.network/v1").all()
    assert len(rows) == 1
    assert rows[0].entity_id == ""


def test_snapshot_excludes_expired_state(api):
    import time as _t
    client, _, _ = api
    client.post("/v1/context/state", json={
        "key": "spatial.location/v1", "value": "lab",
        "entity_id": "person:gad", "estimator": "e",
        "valid_until": _t.time() - 60})
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "estimator": "e"})
    snap = client.get("/v1/context/snapshot").json()
    assert [s["key"] for s in snap["states"]] == ["spatial.occupancy/v1"]
    # history still queryable via the list endpoint
    all_states = client.get("/v1/context/state").json()["states"]
    assert len(all_states) == 2
    active = client.get("/v1/context/state",
                        params={"active_only": True}).json()["states"]
    assert len(active) == 1


def test_evidence_idempotent_external_id(api):
    """Retry of a buffered batch must not duplicate evidence."""
    client, _, _ = api
    item = {"key": "spatial.presence/v1", "value": "occupied",
            "external_id": "thoth:pi2:2026-09-24T10:00:00Z:0"}
    r1 = client.post("/v1/context/evidence", json=item)
    r2 = client.post("/v1/context/evidence", json=item)
    assert r1.json()["evidence"][0]["id"] == r2.json()["evidence"][0]["id"]
    ev = client.get("/v1/context/evidence").json()["evidence"]
    assert len(ev) == 1


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
        "entity_id": "space:lab", "estimator": "occ-v1"})
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
