"""Automation tests — engine semantics + /v1/automation API.

Automation evaluation lives in Brain: rules keep firing while control
surfaces are offline. The engine is edge-triggered on state transitions.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.automation import AutomationEngine
from server.db import Base, User, get_db
from server.v1.automation import router as automation_router
from server.v1.context import router as context_router
from server.endpoints.node_ws import router as node_ws_router


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
        app.include_router(automation_router, prefix='/v1')
        app.include_router(context_router, prefix='/v1')
        app.include_router(node_ws_router, prefix='/v1')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user
    engine.dispose()


# --- engine semantics -------------------------------------------------------

def test_engine_edge_triggered_and_cooldown():
    fired = []
    engine = AutomationEngine(
        actuate=lambda aid, cmd: fired.append((aid, cmd)) or {"status": "succeeded"})
    engine.add_rule({
        "name": "lab-occupied",
        "when": {"key": "spatial.occupancy/v1", "entity_id": "space:lab",
                 "equals": "occupied"},
        "then": {"actuator_id": "light-1", "operation": "set",
                 "params": {"on": True}}})
    snap = {"states": [{"key": "spatial.occupancy/v1", "value": "occupied",
                        "entity_id": "space:lab", "confidence": 0.9}]}
    assert len(engine.evaluate(snap)) == 1
    assert fired[0][0] == "light-1"
    # still matched → no re-fire without a false→true transition
    assert engine.evaluate(snap) == []
    # unmatch then re-match → fires again
    engine.evaluate({"states": []})
    assert len(engine.evaluate(snap)) == 1


def test_engine_no_match_no_fire():
    engine = AutomationEngine()
    engine.add_rule({"name": "r", "when": {"key": "k", "equals": "x"},
                     "then": {"actuator_id": "a", "operation": "set"}})
    assert engine.evaluate({"states": []}) == []


def test_engine_rejects_malformed_rule():
    engine = AutomationEngine()
    with pytest.raises(ValueError):
        engine.add_rule({"name": "r"})


def test_engine_extended_comparators():
    """min/max on scalars or {distance_m}-dicts, contains on dicts/
    strings/lists, not_equals, field selector."""
    fired = []
    engine = AutomationEngine(
        actuate=lambda aid, cmd: fired.append((aid, cmd))
        or {"status": "succeeded"})
    engine.add_rule({
        "name": "near", "when": {"key": "proximity.v1",
                                 "field": "distance_m", "max": 10},
        "then": {"actuator_id": "light-1", "operation": "set_color"}})
    engine.add_rule({
        "name": "linked", "when": {"key": "watch.link.v1",
                                   "contains": {"connected": True}},
        "then": {"actuator_id": "light-1", "operation": "set_color",
                 "params": {"rgb": [255, 0, 0]}}})
    engine.add_rule({
        "name": "face", "when": {"key": "prediction",
                                 "not_equals": "no_face"},
        "then": {"actuator_id": "light-1", "operation": "set_color",
                 "params": {"rgb": [255, 255, 0]}}})

    # distance 4.5 < 10 → 'near' matches; watch connected → 'linked';
    # prediction "Gad" != "no_face" → 'face'.
    snap = {"states": [
        {"key": "proximity.v1", "value": {"distance_m": 4.5}},
        {"key": "watch.link.v1",
         "value": {"connected": True, "rssi": -61}},
        {"key": "prediction", "value": "Gad"}]}
    fired_names = {r["rule"] for r in engine.evaluate(snap)}
    assert fired_names == {"near", "linked", "face"}

    engine2 = AutomationEngine()
    engine2.add_rule({"name": "near", "when": {"key": "proximity.v1",
                                               "field": "distance_m",
                                               "max": 10},
                      "then": {"actuator_id": "a", "operation": "o"}})
    # 12 m → beyond the threshold → no fire
    assert engine2.evaluate({"states": [
        {"key": "proximity.v1", "value": {"distance_m": 12.0}}]}) == []


def test_engine_whispy_expr_condition():
    """``when.expr`` runs whispy's rule language over a flattened view
    of live states — ``proximity.v1 {distance_m: 4.2}`` exposes
    ``proximity_v1_distance_m``."""
    fired = []
    engine = AutomationEngine(
        actuate=lambda aid, cmd: fired.append((aid, cmd))
        or {"status": "succeeded"})
    engine.add_rule({
        "name": "watch-near",
        "when": {"expr": "proximity_v1_distance_m < 10"},
        "then": {"actuator_id": "light-1", "operation": "set_color",
                 "params": {"rgb": [0, 255, 0]}}})
    near = {"states": [{"key": "proximity.v1",
                        "value": {"distance_m": 4.2, "rssi": -55}}]}
    assert len(engine.evaluate(near)) == 1
    # edge-reset then out-of-range → no fire
    engine.evaluate({"states": []})
    assert engine.evaluate({"states": [
        {"key": "proximity.v1", "value": {"distance_m": 30}}]}) == []

    # expr composes with the dict matchers — both must hold.
    engine2 = AutomationEngine()
    engine2.add_rule({
        "name": "linked",
        "when": {"key": "watch.link.v1", "contains": {"connected": True},
                 "expr": "watch_link_v1_rssi > -80"},
        "then": {"actuator_id": "a", "operation": "o"}})
    assert len(engine2.evaluate({"states": [
        {"key": "watch.link.v1",
         "value": {"connected": True, "rssi": -61}}]})) == 1


def test_notify_actuator_publishes_event(api):
    """actuator_id 'notify' → ActionRequest succeeds immediately and a
    `notification` NodeEvent is persisted (SSE fan-out + replay)."""
    client, session, _ = api
    client.post("/v1/automation/rules", json={
        "name": "room-occupied-ping",
        "when": {"key": "occupancy.v1", "equals": "occupied"},
        "then": {"actuator_id": "notify", "operation": "notify",
                 "params": {"title": "Occupied",
                            "body": "Someone is in the room"}}})
    client.post("/v1/context/state", json={
        "key": "occupancy.v1", "value": "occupied",
        "entity_id": "place:home", "estimator": "occ-v1"})
    actions = client.get("/v1/automation/actions").json()["actions"]
    assert len(actions) == 1
    assert actions[0]["actuator_id"] == "notify"
    assert actions[0]["status"] == "succeeded"
    events = client.get("/v1/events?kind=notification").json()["events"]
    assert len(events) == 1
    assert events[0]["data"]["title"] == "Occupied"
    # idempotent — a second evaluation doesn't re-notify while matched
    client.post("/v1/automation/evaluate")
    assert len(client.get("/v1/events?kind=notification")
               .json()["events"]) == 1


# --- API --------------------------------------------------------------------

def test_rules_crud(api):
    client, _, _ = api
    r = client.post("/v1/automation/rules", json={
        "name": "lab-light", "when": {"key": "spatial.occupancy/v1",
                                      "equals": "occupied"},
        "then": {"actuator_id": "light-1", "operation": "set"},
        "cooldown_s": 30})
    assert r.status_code == 201
    assert client.get("/v1/automation/rules").json()["rules"][0]["name"] \
        == "lab-light"
    r = client.delete("/v1/automation/rules/lab-light")
    assert r.json()["ok"] is True
    assert client.get("/v1/automation/rules").json()["rules"] == []


def test_rule_validation(api):
    client, _, _ = api
    r = client.post("/v1/automation/rules", json={
        "name": "bad", "when": {}, "then": {"actuator_id": "a"}})
    assert r.status_code == 422
    r = client.post("/v1/automation/rules", json={
        "name": "bad2", "when": {"key": "k"}, "then": {}})
    assert r.status_code == 422


def test_evaluate_fires_against_current_context(api):
    client, _, _ = api
    client.post("/v1/automation/rules", json={
        "name": "lab-light",
        "when": {"key": "spatial.occupancy/v1", "entity_id": "space:lab",
                 "equals": "occupied"},
        "then": {"actuator_id": "light-1", "operation": "set",
                 "params": {"on": True}}})
    # no state → no fire
    assert client.post("/v1/automation/evaluate").json()["fired"] == []
    # estimator writes state → the transition event fires the rule
    # immediately (event-driven — no manual evaluate needed)
    client.post("/v1/context/state", json={
        "key": "spatial.occupancy/v1", "value": "occupied",
        "entity_id": "space:lab", "estimator": "occ-v1"})
    execs = client.get("/v1/automation/executions").json()["executions"]
    assert len(execs) == 1 and execs[0]["rule"] == "lab-light"
    actions = client.get("/v1/automation/actions").json()["actions"]
    assert len(actions) == 1
    assert actions[0]["status"] == "queued"   # no device endpoint → queued
    # edge-triggered + persisted: a manual evaluate does not re-fire
    res = client.post("/v1/automation/evaluate").json()
    assert res["fired"] == []
    # disabled rule doesn't fire
    client.post("/v1/automation/rules", json={
        "name": "lab-light",
        "when": {"key": "spatial.occupancy/v1", "equals": "occupied"},
        "then": {"actuator_id": "light-1", "operation": "set"},
        "enabled": False})
    res = client.post("/v1/automation/evaluate").json()
    assert res["evaluated_rules"] == 0


def test_automation_tenant_isolation(api):
    client, session, _ = api
    client.post("/v1/automation/rules", json={
        "name": "mine", "when": {"key": "k"}, "then": {"actuator_id": "a"}})
    other = User(userId=2, username='other', email='o@x.invalid',
                 hashed_password='x', role=0, plan='free')
    session.add(other)
    session.commit()
    client.app.dependency_overrides[get_current_user] = lambda: other
    assert client.get("/v1/automation/rules").json()["rules"] == []
