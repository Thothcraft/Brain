"""Context inference endpoint — OpenAI function-call → context form."""
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (
    Base, ContextEntity, ContextEvent, ContextEvidence,
    ContextRelationship, ContextState, Device, User, get_db,
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
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: dict(FORM))
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    body = r.json()
    assert body["form"]["summary"] == FORM["summary"]
    receipt = body["receipt"]
    assert len(receipt["entities"]) == 2
    assert len(receipt["relationships"]) == 1
    assert len(receipt["evidence"]) == 2
    assert len(receipt["states"]) == 2

    # +1 = the 'infer:last' audit entity retaining input/output
    assert session.query(ContextEntity).count() == 3
    assert session.query(ContextEntity).filter_by(
        entity_key="infer:last").one().kind == "system"
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
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: dict(FORM))
    r = client.post("/v1/context/infer", json={**REQ, "dry_run": True})
    assert r.status_code == 200
    assert "receipt" not in r.json()
    assert session.query(ContextState).count() == 0
    # only the infer:last audit entity — no form entities land
    assert session.query(ContextEntity).count() == 1
    assert session.query(ContextEntity).one().entity_key == "infer:last"


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
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: bad)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    receipt = r.json()["receipt"]
    assert len(receipt["states"]) == 1
    errs = [e for e in receipt["errors"] if e["section"] == "states"]
    assert len(errs) == 2
    assert session.query(ContextState).count() == 1
    # 2 form entities + the infer:last audit entity
    assert session.query(ContextEntity).count() == 3


def test_infer_no_tool_call_is_502(api, monkeypatch):
    client, session, _ = api
    def boom(req, **_):
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
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: dict(FORM))
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

    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: dict(FORM))

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

    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)

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

    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)

    r = client.post("/v1/context/infer", json=REQ)

    receipt = r.json()["receipt"]

    assert any("stripped" in str(e["error"])

               for e in receipt.get("errors", []))

    st = session.query(ContextState).filter_by(

        state_key="occupancy.v1").one()

    assert st.estimator == "openai-context-form/1"





# -- thinking meter + richer context + curation ------------------------



def test_thinking_tier_routes_to_model(api, monkeypatch):

    """thinking='deep' resolves the deep-tier model; the meter row is

    keyed per-tier so quota/cost stays attributable."""

    client, session, _ = api

    seen = {}

    def fake(req, **_):

        seen["thinking"] = req.thinking

        seen["model"] = infer._model_for_tier(req.thinking)

        return {"summary": "s", "states": []}

    monkeypatch.setattr(infer, "_openai_form", fake)

    monkeypatch.setenv("CONTEXT_INFER_MODEL_DEEP", "gpt-4o-test")

    r = client.post("/v1/context/infer", json={**REQ, "thinking": "deep"})

    assert r.status_code == 200

    assert seen["thinking"] == "deep"

    assert seen["model"] == "gpt-4o-test"

    body = r.json()

    assert body["thinking"] == "deep"

    assert body["model_id"] == "gpt-4o-test"





def test_unknown_thinking_tier_is_422(api):

    client, session, _ = api

    r = client.post("/v1/context/infer",

                    json={**REQ, "thinking": "galaxy"})

    assert r.status_code == 422





def test_curation_sections_persist_as_evidence(api, monkeypatch):

    """uncertainties/questions/notes are the model's memory channel —

    they land as versioned evidence rows, never as facts."""

    client, session, _ = api

    form = dict(FORM)

    form["uncertainties"] = [{"key": "location.v1",

                              "reason": "anchors disagree"}]

    form["questions"] = [{"question": "walk with the watch for 2 min",

                          "target_key": "location.v1"}]

    form["notes"] = [{"subject": "device:watch-1",

                      "text": "belongs on person:gad's left wrist"}]

    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)

    r = client.post("/v1/context/infer", json=REQ)

    assert r.status_code == 200

    body = r.json()

    assert body["questions"][0]["target_key"] == "location.v1"

    curation = body["receipt"]["curation"]

    assert {c["kind"] for c in curation} == {

        "uncertainties", "questions", "notes"}

    keys = {e.evidence_key for e in session.query(ContextEvidence)}

    assert {"context.uncertainty.v1", "context.question.v1",

            "context.note.v1"} <= keys





def test_request_carries_extended_context(api, monkeypatch):

    """context/history/coverage ride the request into the model

    payload — richer steering data stays intact end to end."""

    client, session, _ = api

    seen = {}

    def fake(req, **_):

        seen["context"] = req.context

        seen["history"] = req.history

        seen["coverage"] = req.coverage

        return {"summary": "s", "states": []}

    monkeypatch.setattr(infer, "_openai_form", fake)

    rich = {**REQ,

            "context": {"devices": [{"id": "dev-1", "kind": "pi5",

                                     "meta": {"model": "BGT60TR13C"}}]},

            "history": [{"state": "occupancy.v1",

                         "value": {"occupied": False}, "ts": 999.0}],

            "coverage": ["states", "uncertainties"]}

    r = client.post("/v1/context/infer", json=rich)

    assert r.status_code == 200

    assert seen["context"]["devices"][0]["kind"] == "pi5"

    assert seen["history"][0]["ts"] == 999.0

    assert seen["coverage"] == ["states", "uncertainties"]


# -- model picker, gathered sensor bundle, device_updates, audit ------

def test_user_model_override_routes(api, monkeypatch):
    """The user-set model wins over the tier route and is metered."""
    client, session, _ = api
    seen = {}
    def fake(req, **_):
        seen["resolved"] = infer._model_for_tier(req.thinking, req.model)
        return {"summary": "s", "states": []}
    monkeypatch.setattr(infer, "_openai_form", fake)
    r = client.post("/v1/context/infer",
                    json={**REQ, "model": "gpt-4o"})
    assert r.status_code == 200
    assert seen["resolved"] == "gpt-4o"
    assert r.json()["model_id"] == "gpt-4o"


def test_disallowed_model_is_422(api):
    client, session, _ = api
    r = client.post("/v1/context/infer",
                    json={**REQ, "model": "davinci-003"})
    assert r.status_code == 422


def test_gather_assembles_comprehensive_payload(api, monkeypatch):
    """gather_window_s pulls the device roster (marked registered),
    node scenes and the context map into the model payload — the
    comprehensive sensor view the estimator reads."""
    client, session, _ = api
    from server.db import ContextEvidence, Device
    session.add(Device(userId=1, device_uuid="uuid-radar-pi",
                       device_name="thoth-chen", device_type="thoth",
                       mac_address="d8:3a:dd:00:11:22",
                       hardware_info='{"model": "Pi5"}'))
    session.add(ContextEvidence(
        user_id=1, device_id="uuid-radar-pi",
        evidence_key="context.descriptors.v1",
        timestamp=time.time(),
        source_id="node", confidence=1.0,
        value=json.dumps({"value": {
            "scene": "radar motion",
            "sensors": {"radar-a316": {
                "type": "radar", "n": 40, "rate_hz": 10.0,
                "text": "radar: snr 12dB",
                "fields": {"snr_db": {"mean": 12.1, "min": 4.0,
                                      "max": 18.0}},
                "cues": {"motion": "present"}},
                "radio-24d8": {
                "type": "radio", "n": 3,
                "scan": {"ble": [{"mac": "AA:BB:CC:DD:EE:FF",
                                  "name": "ELK-BLEDOM",
                                  "rssi": -83}]}}}}})))
    session.commit()

    seen = {}
    def fake(req, payload=None, **_):
        seen.update(payload or {})
        return {"summary": "s", "states": []}
    monkeypatch.setattr(infer, "_openai_form", fake)
    r = client.post("/v1/context/infer",
                    json={**REQ, "gather_window_s": 900})
    assert r.status_code == 200
    dev = next(d for d in seen["devices"]
               if d.get("uuid") == "uuid-radar-pi")
    assert dev["association"] == "registered"
    assert dev["name"] == "thoth-chen"
    scene = seen["scenes"][0]
    assert scene["sensors"]["radar-a316"]["fields"]["snr_db"]["max"] == 18.0
    assert "map" in seen and "observations" in seen
    # the response echoes exactly what the model saw
    assert r.json()["seen"]["devices"] == seen["devices"]


def test_device_updates_land_as_entity_proposals(api, monkeypatch):
    """device_updates name/identify devices: new entities for unknown
    emitters, name refresh for registered anchors, never confirmed
    facts."""
    client, session, _ = api
    form = {"summary": "s", "states": [],
            "device_updates": [
                {"device": "AA:BB:CC:DD:EE:FF",
                 "proposed_name": "kitchen LED strip",
                 "proposed_kind": "device", "confidence": 0.7,
                 "rationale": "MAC vendor + advertised name ELK-BLEDOM"},
                {"device": "device:uuid-radar-pi",
                 "proposed_name": "living-room radar Pi"}]}
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    updates = r.json()["receipt"]["device_updates"]
    keys = {u["entity"] for u in updates}
    assert "device:aa-bb-cc-dd-ee-ff" in keys
    assert "device:uuid-radar-pi" in keys
    led = session.query(ContextEntity).filter_by(
        entity_key="device:aa-bb-cc-dd-ee-ff").one()
    attrs = json.loads(led.attributes)
    assert led.name == "kitchen LED strip"
    assert attrs["identified_by"] == "context_infer"
    assert "AA:BB:CC:DD:EE:FF" in attrs["aliases"]


def test_devices_section_resolves_mac_to_registered_anchor(api,
                                                           monkeypatch):
    """devices[] keyed by a registered MAC lands on device:<uuid> —
    one entity per physical device, never a MAC-slug fork."""
    client, session, _ = api
    session.add(Device(userId=1, device_uuid="uuid-radar-pi",
                       device_name="thoth-chen", device_type="thoth",
                       mac_address="d8:3a:dd:11:22:33"))
    session.commit()
    form = {"summary": "s", "states": [],
            "devices": [
                {"device": "D8:3A:DD:11:22:33",
                 "name": "thoth-chen — Pi5 radar rig",
                 "role": "radar-node", "vendor": "Raspberry Pi",
                 "confidence": 0.9},
                {"device": "AA:BB:CC:DD:EE:FF",
                 "name": "kitchen LED strip", "kind": "device",
                 "confidence": 0.7}]}
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    updates = {u["ref"]: u for u in r.json()["receipt"]["device_updates"]}
    assert updates["D8:3A:DD:11:22:33"]["entity"] == \
        "device:uuid-radar-pi"
    assert updates["D8:3A:DD:11:22:33"]["resolved"] == \
        "device:uuid-radar-pi"
    # exactly one entity per piece of hardware — no MAC-slug fork
    keys = {e.entity_key for e in
            session.query(ContextEntity).filter_by(kind="device").all()}
    assert keys == {"device:uuid-radar-pi",
                    "device:aa-bb-cc-dd-ee-ff"}
    ent = session.query(ContextEntity).filter_by(
        entity_key="device:uuid-radar-pi").one()
    attrs = json.loads(ent.attributes)
    assert ent.name == "thoth-chen — Pi5 radar rig"
    assert attrs["role"] == "radar-node"


def test_persons_and_places_sections(api, monkeypatch):
    """The structured form: place building→room hierarchy, person
    associated devices resolved via fingerprints, located_in edge +
    location.v1 state for the person."""
    client, session, _ = api
    session.add(Device(userId=1, device_uuid="uuid-radar-pi",
                       device_name="thoth-chen", device_type="thoth",
                       mac_address="d8:3a:dd:11:22:33"))
    session.commit()
    form = {"summary": "s", "states": [],
            "places": [{
                "id": "place:home", "name": "home",
                "address": "43 Hillsmount", "kind": "building",
                "rooms": [{"id": "place:home-living",
                           "name": "living room",
                           "layout": {"dims_m": [5, 4]},
                           "devices": [{"ref": "device:uuid-radar-pi",
                                        "placement": {"corner": "ne"},
                                        "manual": True}]}],
                "confidence": 0.8}],
            "persons": [{
                "id": "person:owner", "name": "Gad", "is_user": True,
                "address": "43 Hillsmount",
                "associated_devices": ["D8:3A:DD:11:22:33"],
                "location": {"place": "place:home",
                             "room": "place:home-living",
                             "granularity": "indoor_room"},
                "confidence": 0.8}]}
    monkeypatch.setattr(infer, "_openai_form", lambda req, **_: form)
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    rec = r.json()["receipt"]
    assert rec["places"] == ["place:home"]
    assert rec["persons"] == ["person:owner"]

    home = session.query(ContextEntity).filter_by(
        entity_key="place:home").one()
    hattrs = json.loads(home.attributes)
    assert hattrs["address"] == "43 Hillsmount"
    assert hattrs["rooms"][0]["devices"][0]["ref"] == \
        "device:uuid-radar-pi"          # manual placement survives
    assert hattrs["rooms"][0]["devices"][0]["manual"] is True

    room = session.query(ContextEntity).filter_by(
        entity_key="place:home-living").one()
    assert room.kind == "place"
    rel = session.query(ContextRelationship).filter_by(
        subject="place:home-living", predicate="part_of").one()
    assert rel.object == "place:home"

    owner = session.query(ContextEntity).filter_by(
        entity_key="person:owner").one()
    oattrs = json.loads(owner.attributes)
    assert oattrs["address"] == "43 Hillsmount"
    assert oattrs["associated_devices"] == ["device:uuid-radar-pi"]
    loc = session.query(ContextRelationship).filter_by(
        subject="person:owner", predicate="located_in").one()
    assert loc.object == "place:home-living"   # deepest place wins
    st = session.query(ContextState).filter_by(
        state_key="location.v1", entity_id="person:owner").one()
    assert json.loads(st.value)["granularity"] == "indoor_room"


def test_options_and_last_endpoints(api, monkeypatch):
    client, session, _ = api
    assert client.get("/v1/context/infer/last").status_code == 404
    opts = client.get("/v1/context/infer/options").json()
    assert "gpt-4o" in opts["models"]
    assert opts["defaults"]["deep"] == "gpt-4o"
    monkeypatch.setattr(
        infer, "_openai_form",
        lambda req, **_: {"form": {"summary": "audit me",
                                   "states": []},
                          "model_text": "free text"})
    r = client.post("/v1/context/infer", json=REQ)
    assert r.status_code == 200
    last = client.get("/v1/context/infer/last").json()
    assert last["summary"] == "audit me"
    assert last["model_text"] == "free text"
    assert "descriptors" in last["input"]

