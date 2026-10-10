"""Context builder — stable semantic map from descriptor aggregates."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (
    Base, ContextEntity, ContextEvidence, ContextRelationship, ContextState,
    Device, User, get_db,
)
import server.v1.context_builder as cb


T0 = 1_000_000.0


def _proposal(place="place:living-room", activity="activity:sitting",
              conf=0.7, occupied=True):
    return {
        "summary": f"gad in {place}",
        "entities": [
            {"id": "person:gad", "kind": "person", "name": "Gad",
             "confidence": 0.8},
            {"id": place, "kind": "place", "confidence": 0.9},
            {"id": activity, "kind": "activity", "confidence": 0.7},
        ],
        "relationships": [
            {"subject": "person:gad", "predicate": "located_in",
             "object": place, "confidence": conf},
            {"subject": "person:gad", "predicate": "doing",
             "object": activity, "confidence": conf},
            {"subject": "device:dev-1", "predicate": "located_in",
             "object": place, "confidence": 0.9},
        ],
        "states": [
            {"key": "occupancy.v1", "entity_id": place,
             "value": {"occupied": occupied}, "confidence": conf},
        ],
    }


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={
        "check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        session.add(User(userId=1, username="t", email="t@e.invalid",
                         hashed_password="x", role=0, plan="free"))
        session.add(Device(userId=1, device_uuid="dev-1",
                           device_name="thoth-chen", device_type="thoth"))
        session.commit()
        yield session, factory
    engine.dispose()


def _evidence(session, n=5, key="occupancy.prob.v1", ts=T0):
    for i in range(n):
        session.add(ContextEvidence(
            user_id=1, evidence_key=key, device_id="dev-1",
            value=json.dumps({"p": 0.8 + i * 0.01, "label": "occupied"}),
            timestamp=ts - i, confidence=0.8))
    session.commit()


def _active_rel(session, subject, predicate):
    return [r for r in session.query(ContextRelationship).filter_by(
        subject=subject, predicate=predicate).all() if r.valid_until is None]


def test_bundle_aggregates_descriptors_not_raw(db):
    session, _ = db
    _evidence(session, n=10)
    cb.seed_devices(session, 1, T0)
    bundle = cb.build_bundle(session, 1, T0)
    assert bundle["evidence_rows"] == 10
    agg = bundle["descriptors"][0]
    assert agg["key"] == "occupancy.prob.v1" and agg["n"] == 10
    assert agg["fields"]["p"]["n"] == 10
    assert 0.8 <= agg["fields"]["p"]["mean"] <= 0.9
    ids = {e["id"] for e in bundle["map"]["entities"]}
    assert "device:dev-1" in ids
    assert "map:builder" not in ids
    # The registry block tells the model which machine each uuid is —
    # and by absence, which emitters in scans are unknown devices.
    devs = {d["entity"]: d for d in bundle["devices"]}
    assert devs["device:dev-1"]["name"] == "thoth-chen"
    assert devs["device:dev-1"]["uuid"] == "dev-1"


def test_bundle_surfaces_textual_scenes_from_node_uplinks(db):
    session, _ = db
    for ts, scene in ((T0 - 120, "old"), (T0 - 5, "radar: high motion")):
        session.add(ContextEvidence(
            user_id=1, evidence_key="context.descriptors.v1",
            device_id="dev-1", timestamp=ts,
            value=json.dumps({"value": {
                "scene": scene,
                "predictions": {"builtin:occupancy-radar": {
                    "label": "occupied", "confidence": 0.91}},
                "location": {"lat": 43.65, "lon": -79.38,
                             "site": "43 Hillsmount"},
                "estimates": [{"key": "occupancy.v1",
                               "subject": "device:dev-1",
                               "value": {"occupied": False},
                               "confidence": 0.62}],
                "sensors": {
                    "radar-1": {"type": "radar", "n": 20,
                                "rate_hz": 10.0, "age_s": 0.2,
                                "text": "radar: high motion",
                                "cues": {"motion": "high"},
                                "fields": {"snr_db": {"mean": 1.0}}},
                    "mic-1": {"type": "microphone", "n": 1,
                              "text": 'speech: "lights off"',
                              "cues": {"speech": {"text": "lights off"}}}}}})))
    session.commit()
    bundle = cb.build_bundle(session, 1, T0)
    assert len(bundle["scenes"]) == 1
    sc = bundle["scenes"][0]
    assert sc["scene"] == "radar: high motion" and sc["age_s"] == 5.0
    assert sc["sensors"]["mic-1"]["cues"]["speech"]["text"] == "lights off"
    # physical descriptors must reach the model — radar fields are the
    # whole point of the uplink
    assert sc["sensors"]["radar-1"]["fields"]["snr_db"]["mean"] == 1.0
    assert sc["sensors"]["radar-1"]["n"] == 20
    assert sc["location"]["site"] == "43 Hillsmount"
    assert sc["estimates"][0]["key"] == "occupancy.v1"
    assert sc["predictions"]["builtin:occupancy-radar"]["label"] == "occupied"
    assert all(a["key"] != "context.descriptors.v1"
               for a in bundle["descriptors"])


def test_bundle_aggregates_geo_evidence(db):
    """location.geo.v1 rows aggregate like any descriptor — the chat
    agent and builder both see where the reporting node is."""
    session, _ = db
    session.add(ContextEvidence(
        user_id=1, evidence_key="location.geo.v1", device_id="dev-1",
        timestamp=T0,
        value=json.dumps({"lat": 43.65, "lon": -79.38,
                          "site": "43 Hillsmount", "city": "Toronto"})))
    session.commit()
    bundle = cb.build_bundle(session, 1, T0)
    agg = next(a for a in bundle["descriptors"]
               if a["key"] == "location.geo.v1")
    assert agg["latest"]["site"] == "43 Hillsmount"
    assert agg["fields"]["lat"]["mean"] == 43.65


def test_build_applies_map_and_seeds_devices(db):
    session, _ = db
    _evidence(session)
    res = cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    rec = res["receipt"]
    assert "person:gad" in rec["entities"]
    assert len(rec["relationships"]["created"]) == 3
    assert len(rec["states"]["applied"]) == 1
    m = cb.grouped_map(session, 1, T0)
    assert [d["id"] for d in m["devices"]] == ["device:dev-1"]
    assert m["devices"][0]["name"] == "thoth-chen"
    # the address anchor is seeded — the model's place joins it
    assert {p["id"] for p in m["places"]} == {
        "place:living-room", "place:home"}
    assert m["builder"]["builds"] == 1
    # Audit trail — what the model saw and what it produced.
    inp = json.loads(m["builder"]["last_input"])
    assert "scenes" in inp and "descriptors" in inp
    prop = json.loads(m["builder"]["last_proposal"])
    assert prop["summary"] == "gad in place:living-room"


def test_repeated_identical_builds_are_stable(db):
    session, _ = db
    for i in range(4):
        cb.run_build(session, 1, now=T0 + i * 60, llm=lambda b: _proposal())
    assert session.query(ContextRelationship).count() == 3
    assert len(_active_rel(session, "person:gad", "located_in")) == 1
    # dev-1 + person:gad + place:living-room + activity:sitting +
    # the two seeded anchors (person:owner, place:home)
    assert session.query(ContextEntity).filter(
        ContextEntity.retired_at.is_(None),
        ContextEntity.kind != "system").count() == 6


def test_exclusive_relationship_needs_confirmation(db):
    session, _ = db
    cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    # one weak proposal of a new place → pending, old edge kept
    r = cb.run_build(session, 1, now=T0 + 60,
                     llm=lambda b: _proposal(place="place:kitchen"))
    assert ["person:gad", "located_in", "place:kitchen"] in \
        r["receipt"]["relationships"]["pending"]
    active = _active_rel(session, "person:gad", "located_in")
    assert [a.object for a in active] == ["place:living-room"]
    # confirmed on the next build → switch
    cb.run_build(session, 1, now=T0 + 120,
                 llm=lambda b: _proposal(place="place:kitchen"))
    active = _active_rel(session, "person:gad", "located_in")
    assert [a.object for a in active] == ["place:kitchen"]


def test_strong_proposal_switches_immediately(db):
    session, _ = db
    cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    cb.run_build(session, 1, now=T0 + 60,
                 llm=lambda b: _proposal(place="place:kitchen", conf=0.95))
    active = _active_rel(session, "person:gad", "located_in")
    assert [a.object for a in active] == ["place:kitchen"]


def test_state_value_flip_has_hysteresis(db):
    session, _ = db
    cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    cb.run_build(session, 1, now=T0 + 60,
                 llm=lambda b: _proposal(occupied=False))
    st = session.query(ContextState).filter_by(state_key="occupancy.v1").one()
    assert json.loads(st.value) == {"occupied": True}
    cb.run_build(session, 1, now=T0 + 120,
                 llm=lambda b: _proposal(occupied=False))
    session.refresh(st)
    assert json.loads(st.value) == {"occupied": False}


def test_alias_resolution_prevents_forks(db):
    session, _ = db
    p = _proposal()
    p["entities"][0]["aliases"] = ["gad", "aa:bb:cc:dd:ee:ff"]
    cb.run_build(session, 1, now=T0, llm=lambda b: p)
    renamed = _proposal()
    renamed["entities"][0]["id"] = "aa:bb:cc:dd:ee:ff"
    for rel in renamed["relationships"]:
        if rel["subject"] == "person:gad":
            rel["subject"] = "gad"
    cb.run_build(session, 1, now=T0 + 60, llm=lambda b: renamed)
    persons = session.query(ContextEntity).filter_by(kind="person").all()
    assert {e.entity_key for e in persons} == {
        "person:gad", "person:owner"}
    assert len(_active_rel(session, "person:gad", "located_in")) == 1


def test_low_confidence_and_unresolved_are_skipped(db):
    session, _ = db
    p = _proposal(conf=0.3)
    p["relationships"].append({"subject": "person:ghost", "predicate": "near",
                               "object": "place:living-room",
                               "confidence": 0.9})
    r = cb.run_build(session, 1, now=T0, llm=lambda b: p)
    reasons = {s.get("reason") for s in r["receipt"]["skipped"]}
    assert {"low confidence", "unresolved endpoint"} <= reasons
    assert _active_rel(session, "person:gad", "located_in") == []


def test_decay_ends_stale_edges_and_never_retires_devices(db):
    session, _ = db
    cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    later = T0 + 8 * 86400
    cb.run_build(session, 1, now=later, llm=lambda b: {"summary": "quiet"})
    assert _active_rel(session, "person:gad", "located_in") == []
    live = {e.entity_key for e in session.query(ContextEntity).filter(
        ContextEntity.retired_at.is_(None)).all()}
    assert "device:dev-1" in live
    assert "person:gad" not in live
    assert "activity:sitting" not in live


def test_merge_moves_aliases_and_edges(db):
    session, _ = db
    p = _proposal()
    p["entities"].append({"id": "person:g", "kind": "person",
                          "confidence": 0.8})
    p["relationships"].append({"subject": "person:g", "predicate": "uses",
                               "object": "device:dev-1", "confidence": 0.8})
    cb.run_build(session, 1, now=T0, llm=lambda b: p)
    cb.run_build(session, 1, now=T0 + 60, llm=lambda b: {
        "summary": "dup", "merges": [{"source": "person:g",
                                      "target": "person:gad"}]})
    g = session.query(ContextEntity).filter_by(entity_key="person:g").one()
    assert g.retired_at is not None
    assert len(_active_rel(session, "person:gad", "uses")) == 1
    gad = session.query(ContextEntity).filter_by(entity_key="person:gad").one()
    assert "person:g" in json.loads(gad.attributes)["aliases"]


def test_device_fingerprint_resolution_prevents_duplicates(db):
    """A proposal naming the same MAC/hostname as a registered device
    resolves onto device:<uuid> — the model may rename/refine it but can
    never fork a second entity for the same hardware."""
    session, _ = db
    dev = session.query(Device).one()
    dev.mac_address = "d8:3a:dd:11:22:33"
    session.commit()
    p = _proposal()
    p["entities"].append({
        "id": "device:random-slug", "kind": "device",
        "name": "esp32 beacon",
        "attributes": {"mac": "d8:3a:dd:11:22:33"},
        "confidence": 0.9})
    cb.run_build(session, 1, now=T0, llm=lambda b: p)
    devs = {e.entity_key for e in session.query(ContextEntity).filter_by(
        kind="device").all()}
    assert devs == {"device:dev-1"}
    ent = session.query(ContextEntity).filter_by(
        entity_key="device:dev-1").one()
    aliases = json.loads(ent.attributes)["aliases"]
    assert "device:random-slug" in aliases


def test_seeded_owner_person_and_home_place(db):
    session, _ = db
    cb.seed_devices(session, 1, T0)
    owner = session.query(ContextEntity).filter_by(
        entity_key="person:owner").one()
    assert owner.kind == "person"
    home = session.query(ContextEntity).filter_by(
        entity_key="place:home").one()
    assert home.kind == "place"


def test_registry_device_cannot_be_retired_by_llm(db):
    session, _ = db
    cb.run_build(session, 1, now=T0, llm=lambda b: {
        "summary": "x", "retire": [{"id": "device:dev-1"}]})
    dev = session.query(ContextEntity).filter_by(
        entity_key="device:dev-1").one()
    assert dev.retired_at is None


def test_scheduler_builds_only_users_with_fresh_evidence(db, monkeypatch):
    session, factory = db
    monkeypatch.setattr(cb, "_llm_update", lambda b, **_: _proposal())
    assert cb.build_due_users(factory, now=T0) == []
    _evidence(session, ts=T0)
    assert cb.build_due_users(factory, now=T0 + 10) == [1]


@pytest.fixture
def api(db):
    session, _ = db
    user = session.get(User, 1)
    app = FastAPI()
    app.include_router(cb.router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        yield client, session


def test_rebuild_and_map_endpoints(api, monkeypatch):
    client, session = api
    _evidence(session)
    monkeypatch.setattr(cb, "_llm_update", lambda b, **_: _proposal())
    r = client.post("/v1/context/rebuild", json={"window_s": 3600})
    assert r.status_code == 200, r.text
    assert "person:gad" in {p["id"]
                            for p in r.json()["map"]["persons"]}
    m = client.get("/v1/context/map").json()
    assert m["builder"]["builds"] == 1
    assert m["relationships"]


def test_rebuild_dry_run_writes_nothing(api, monkeypatch):
    client, session = api
    monkeypatch.setattr(cb, "_llm_update", lambda b, **_: _proposal())
    r = client.post("/v1/context/rebuild", json={"dry_run": True})
    assert r.status_code == 200
    assert "receipt" not in r.json()
    assert session.query(ContextRelationship).count() == 0


# ---------------------------------------------------------------------------
# Confirmed-fact precedence (FR-SPA-07) — the builder path honors the same
# markers the REST/form paths enforce in context.py.
# ---------------------------------------------------------------------------

def _confirm_entity(session, key, kind, name=None):
    session.add(ContextEntity(
        user_id=1, entity_key=key, kind=kind, name=name,
        attributes=json.dumps({"_confirmed": True})))
    session.commit()


def _confirm_rel(session, subject, predicate, obj):
    rel = ContextRelationship(
        user_id=1, subject=subject, predicate=predicate, object=obj,
        valid_from=T0 - 10, confidence=1.0, source="user")
    session.add(rel)
    session.commit()
    return rel


def _confirm_state(session, key, entity_id, value):
    st = ContextState(
        user_id=1, state_key=key, entity_id=entity_id,
        value=json.dumps(value), confidence=1.0, estimator="user",
        since=T0 - 10)
    session.add(st)
    session.commit()
    return st


def test_confirmed_entity_is_not_rewritten_by_proposal(db):
    session, _ = db
    _confirm_entity(session, "person:gad", "person", name="Gad (confirmed)")
    r = cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    gad = session.query(ContextEntity).filter_by(
        entity_key="person:gad").one()
    assert gad.name == "Gad (confirmed)"          # rename dropped
    attrs = json.loads(gad.attributes)
    assert attrs.get("_confirmed") is True         # marker survives
    assert "person:gad" in r["receipt"]["entities"]


def test_confirmed_edge_blocks_rival_even_when_strong(db):
    session, _ = db
    rel = _confirm_rel(session, "person:gad", "located_in", "place:office")
    r = cb.run_build(session, 1, now=T0,
                     llm=lambda b: _proposal(conf=0.99))
    session.refresh(rel)
    assert rel.valid_until is None                # confirmed edge intact
    assert _active_rel(session, "person:gad", "located_in") == [rel]
    skips = [s for s in r["receipt"]["skipped"]
             if s.get("reason") == "confirmed_fact"]
    assert skips and skips[0]["relationship"] == [
        "person:gad", "located_in", "place:living-room"]


def test_reasserting_confirmed_edge_dedupes_without_mutation(db):
    session, _ = db
    rel = _confirm_rel(session, "person:gad", "located_in",
                       "place:living-room")
    r = cb.run_build(session, 1, now=T0, llm=lambda b: _proposal())
    session.refresh(rel)
    assert rel.id in r["receipt"]["relationships"]["refreshed"]
    assert rel.confidence == 1.0                  # no builder EWMA
    prov = json.loads(rel.provenance) if rel.provenance else {}
    assert "confirmed_at" not in prov             # no builder stamp


def test_confirmed_state_blocks_builder_proposal(db):
    session, _ = db
    st = _confirm_state(session, "occupancy.v1", "place:living-room",
                        {"occupied": True})
    r = cb.run_build(session, 1, now=T0,
                     llm=lambda b: _proposal(occupied=False, conf=0.99))
    session.refresh(st)
    assert json.loads(st.value) == {"occupied": True}
    assert st.estimator == "user"
    assert st.valid_until is None                 # not extended by hysteresis
    assert {"state": "occupancy.v1", "entity_id": "place:living-room",
            "reason": "confirmed_fact"} in r["receipt"]["skipped"]


def test_confirmed_entity_survives_retire_merge_and_decay(db):
    session, _ = db
    _confirm_entity(session, "person:gad", "person")
    _confirm_rel(session, "person:gad", "located_in", "place:office")
    # retire + merge proposals are skipped for confirmed entities/edges
    r = cb.run_build(session, 1, now=T0, llm=lambda b: {
        "summary": "x",
        "retire": [{"id": "person:gad"}],
        "merges": [{"source": "person:gad", "target": "person:x"}],
        "entities": [{"id": "person:x", "kind": "person",
                      "confidence": 0.9}]})
    gad = session.query(ContextEntity).filter_by(
        entity_key="person:gad").one()
    assert gad.retired_at is None
    reasons = {s.get("reason") for s in r["receipt"]["skipped"]}
    assert "confirmed_fact" in reasons
    # decay never ends confirmed edges nor retires confirmed entities
    later = T0 + 8 * 86400
    cb.run_build(session, 1, now=later, llm=lambda b: {"summary": "quiet"})
    session.refresh(gad)
    assert gad.retired_at is None
    assert _active_rel(session, "person:gad", "located_in") != []


def test_confirmed_device_entity_seed_preserves_user_name(db):
    session, _ = db
    _confirm_entity(session, "device:dev-1", "device", name="front-door")
    cb.run_build(session, 1, now=T0, llm=lambda b: {"summary": "x"})
    dev = session.query(ContextEntity).filter_by(
        entity_key="device:dev-1").one()
    assert dev.name == "front-door"               # seed can't clobber it
