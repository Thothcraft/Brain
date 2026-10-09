"""Context inference endpoint — OpenAI function-call → context form."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (
    Base, ContextEntity, ContextEvent, ContextEvidence,
    ContextRelationship, ContextState, User, get_db,
)
from server.v1.context import router as context_router
from server.v1.context_infer import router as infer_router
import server.v1.context_infer as infer


FORM = {
    "summary": "occupied living room, person stationary",
    "entities": [
        {"id": "person:gad", "kind": "person", "name": "Gad"},
        {"id": "space:living-room", "kind": "space"},
    ],
    "relationships": [
        {"subject": "person:gad", "predicate": "located_in",
         "object": "space:living-room", "confidence": 0.8},
    ],
    "evidence": [
        {"ref": "ev1", "key": "occupancy.prob.v1",
         "value": {"occupied": 0.87}, "confidence": 0.87,
         "model_id": "occ-rf-moe", "model_version": "0.1.0"},
        {"ref": "ev2", "key": "imu.variance.v1",
         "value": {"var": 0.004}, "confidence": 0.7},
    ],
    "states": [
        {"key": "occupancy.v1", "value": {"occupied": True},
         "entity_id": "space:living-room", "confidence": 0.87,
         "transition": "entered", "evidence_refs": ["ev1"]},
        {"key": "activity.v1", "value": {"motion": "stationary"},
         "entity_id": "person:gad", "confidence": 0.7,
         "evidence_refs": ["ev2"]},
    ],
}

REQ = {
    "window": {"start_ts": 1000.0, "end_ts": 1005.0,
               "device_id": "dev-1"},
    "calibration": {"occupied": {"mean_p": 0.8, "std": 0.1},
                    "empty": {"mean_p": 0.1, "std": 0.05}},
    "descriptors": {"p_occupied": 0.87, "csi_var": 0.32,
                    "radar_snr": 18.4},
    "entity_hint": "person:gad",
}


@pytest.fixture
def api():
    engine = create_engine('sqlite://', connect_args={
        'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='test', email='t@e.invalid',
                    hashed_password='unused', role=0, plan='free')
        session.add(user)
        session.commit()
        app = FastAPI()
        app.include_router(context_router, prefix='/v1')
        app.include_router(infer_router, prefix='/v1')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user
    engine.dispose()


def test_infer_applies_full_form(api, monkeypatch):
    client, session, _ = api
    monkeypatch.setattr(infer, "_openai_form", lambda req: dict(FORM))
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    body = r.json()
    assert body["form"]["summary"] == FORM["summary"]
    receipt = body["receipt"]
    assert len(receipt["entities"]) == 2
    assert len(receipt["relationships"]) == 1
    assert len(receipt["evidence"]) == 2
    assert len(receipt["states"]) == 2

    assert session.query(ContextEntity).count() == 2
    assert session.query(ContextRelationship).count() == 1
    assert session.query(ContextEvidence).count() == 2
    assert session.query(ContextState).count() == 2
    # each new state emits an 'entered' event; evidence ref resolved
    occ = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    ev1 = session.query(ContextEvidence).filter_by(
        evidence_key="occupancy.prob.v1").one()
    assert json.loads(occ.evidence_ids) == [str(ev1.id)]
    assert session.query(ContextEvent).count() == 2


def test_infer_dry_run_writes_nothing(api, monkeypatch):
    client, session, _ = api
    monkeypatch.setattr(infer, "_openai_form", lambda req: dict(FORM))
    r = client.post("/v1/context/infer", json={**REQ, "dry_run": True})
    assert r.status_code == 200
    assert "receipt" not in r.json()
    assert session.query(ContextState).count() == 0
    assert session.query(ContextEntity).count() == 0


def test_infer_bad_section_does_not_block_valid_ones(api, monkeypatch):
    """Format sensitivity: a malformed state section is reported, valid
    sections still apply — the store is never corrupted."""
    client, session, _ = api
    bad = dict(FORM)
    bad["states"] = [
        {"key": "occupancy.v1", "value": {"occupied": True},
         "confidence": 0.9},
        {"entity_id": "x"},                      # missing key
        {"key": "activity.v1", "confidence": 9},  # confidence > 1
    ]
    monkeypatch.setattr(infer, "_openai_form", lambda req: bad)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    receipt = r.json()["receipt"]
    assert len(receipt["states"]) == 1
    errs = [e for e in receipt["errors"] if e["section"] == "states"]
    assert len(errs) == 2
    assert session.query(ContextState).count() == 1
    assert session.query(ContextEntity).count() == 2


def test_infer_no_tool_call_is_502(api, monkeypatch):
    client, session, _ = api
    def boom(req):
        from fastapi import HTTPException
        raise HTTPException(502, "model did not call submit_context_form")
    monkeypatch.setattr(infer, "_openai_form", boom)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 502
    assert session.query(ContextState).count() == 0


# -- confirmed-fact precedence (FR-SPA-07) ---------------------------------

def _confirmed_state(client, key="occupancy.v1",
                     entity="space:living-room",
                     value=None, **kw):
    body = {"key": key, "value": value if value is not None
            else {"occupied": True},
            "entity_id": entity, "estimator": "user/portal"}
    body.update(kw)
    r = client.post("/v1/context/state", json=body)
    assert r.status_code == 200
    return r


def test_confirmed_state_blocks_llm_form(api, monkeypatch):
    """A user-confirmed state survives the LLM form — the form's write is
    skipped (precedence), not an error."""
    client, session, _ = api
    _confirmed_state(client)
    monkeypatch.setattr(infer, "_openai_form", lambda req: dict(FORM))
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    skipped = r.json()["receipt"]["skipped"]
    assert any(s["section"] == "states"
               and s["reason"] == "confirmed_fact" for s in skipped)
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    assert st.estimator == "user/portal"      # confirmed writer kept
    assert json.loads(st.value) == {"occupied": True}


def test_rest_confirmed_state_409s_inferred_overwrite(api):
    client, session, _ = api
    _confirmed_state(client)
    r = client.post("/v1/context/state", json={
        "key": "occupancy.v1", "entity_id": "space:living-room",
        "value": {"occupied": False},
        "estimator": "openai-context-form/1"})
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "confirmed_fact"
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    assert json.loads(st.value) == {"occupied": True}


def test_confirmed_write_supersedes_confirmed(api):
    """The user can correct a confirmed fact — only with a confirmed
    source."""
    client, session, _ = api
    _confirmed_state(client)
    r = client.post("/v1/context/state", json={
        "key": "occupancy.v1", "entity_id": "space:living-room",
        "value": {"occupied": False}, "estimator": "confirmed/cli"})
    assert r.status_code == 200
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    assert json.loads(st.value) == {"occupied": False}
    assert st.estimator == "confirmed/cli"


def test_expired_confirmed_state_allows_inference(api):
    """Confirmation has a validity window — once valid_until passes,
    inference may write the key again."""
    client, session, _ = api
    _confirmed_state(client, valid_until=1.0)   # long expired
    r = client.post("/v1/context/state", json={
        "key": "occupancy.v1", "entity_id": "space:living-room",
        "value": {"occupied": False},
        "estimator": "openai-context-form/1"})
    assert r.status_code == 200
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    assert json.loads(st.value) == {"occupied": False}


def test_confirmed_entity_survives_form(api, monkeypatch):
    client, session, _ = api
    r = client.post("/v1/context/entities", json={
        "id": "person:gad", "kind": "person",
        "attributes": {"_confirmed": True, "role": "owner"}})
    assert r.status_code == 201
    monkeypatch.setattr(infer, "_openai_form", lambda req: dict(FORM))
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    skipped = r.json()["receipt"]["skipped"]
    assert any(s["section"] == "entities" for s in skipped)
    ent = session.query(ContextEntity).filter_by(
        entity_key="person:gad").one()
    assert json.loads(ent.attributes)["role"] == "owner"


def test_confirmed_relationship_blocks_conflicting_edge(api, monkeypatch):
    client, session, _ = api
    for eid, kind in (("person:gad", "person"),
                      ("space:kitchen", "space"),
                      ("space:bedroom", "space")):
        assert client.post("/v1/context/entities", json={
            "id": eid, "kind": kind}).status_code == 201
    r = client.post("/v1/context/relationships", json={
        "subject": "person:gad", "predicate": "located_in",
        "object": "space:kitchen", "source": "user"})
    assert r.status_code == 201
    rel_id = r.json()["id"]

    # conflicting inferred edge → skipped; re-assertion → deduped
    form = {"summary": "s", "relationships": [
        {"subject": "person:gad", "predicate": "located_in",
         "object": "space:bedroom"},
        {"subject": "person:gad", "predicate": "located_in",
         "object": "space:kitchen"}],
        "states": []}
    monkeypatch.setattr(infer, "_openai_form", lambda req: form)
    r = client.post("/v1/context/infer", json=REQ)
    receipt = r.json()["receipt"]
    skipped = receipt["skipped"]
    assert any(s["section"] == "relationships"
               and s["reason"] == "confirmed_fact" for s in skipped)
    assert [str(i) for i in receipt["relationships"]] == [rel_id]
    assert session.query(ContextRelationship).count() == 1


def test_llm_cannot_self_assert_confirmed(api, monkeypatch):
    """A form that smuggles a confirmed estimator/source gets it
    stripped — it lands as ordinary inference."""
    client, session, _ = api
    form = {"summary": "s",
            "states": [{"key": "occupancy.v1", "entity_id": "e",
                        "value": {"occupied": True},
                        "estimator": "user"}],
            "relationships": []}
    monkeypatch.setattr(infer, "_openai_form", lambda req: form)
    r = client.post("/v1/context/infer", json=REQ)
    receipt = r.json()["receipt"]
    assert any("stripped" in str(e["error"])
               for e in receipt.get("errors", []))
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1").one()
    assert st.estimator == "openai-context-form/1"
