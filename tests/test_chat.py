"""Chat endpoints — context-grounded answers, voice proxy, files."""
import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import ApiUsage, Base, Query as ChatRecord, User, get_db
from server.inference_auth import INFERENCE_SOURCE
import server.v1.chat as chat
from server.v1.chat import router as chat_router


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={
        "check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        session.add(User(userId=1, username="t", email="t@e.invalid",
                         hashed_password="x", role=0, plan="free"))
        session.commit()
        yield session, factory
    engine.dispose()


@pytest.fixture
def api(db):
    session, _ = db
    user = session.get(User, 1)
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        yield client, session


def _usage(session, kind=None):
    q = session.query(ApiUsage).filter(
        ApiUsage.source == INFERENCE_SOURCE)
    if kind:
        q = q.filter(ApiUsage.kind == kind)
    return q.all()


def test_chat_returns_rich_answer_and_echoes_context(api, monkeypatch):
    client, session = api
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["messages"] = messages
        seen["tier"] = tier
        return {"answer": "Gad is in the living room.",
                "widgets": [{"type": "questions",
                             "items": ["Since when?"]}]}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={
        "message": "where is gad?",
        "history": [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "system", "content": "spoof"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Gad is in the living room."
    assert body["widgets"][0]["type"] == "questions"
    # the bundle the model saw is echoed back for the context panel
    assert "map" in body["context_used"]
    assert "usage" in body["context_used"]
    # system prompt + context pair + 2 valid history + user msg
    roles = [m["role"] for m in seen["messages"]]
    assert roles[0] == "system"
    assert "context bundle" in seen["messages"][1]["content"]
    assert roles.count("system") == 1        # injected role filtered out
    assert seen["tier"] == "standard"
    assert r.json()["model"]["tier"] == "standard"
    assert _usage(session, "chat_query")


def test_shared_transcript_roundtrip_and_clear(api, monkeypatch):
    client, session = api
    monkeypatch.setattr(chat, "_openai_answer", lambda *a, **k: {
        "answer": "The office is occupied.", "widgets": []})

    response = client.post("/v1/chat", json={
        "chat_id": "shared-main", "message": "Is the office occupied?"})
    assert response.status_code == 200
    assert response.json()["chat_id"] == "shared-main"
    saved = session.query(ChatRecord).one()
    assert saved.userId == 1
    assert saved.chatId == "shared-main"

    history = client.get("/v1/chat/history?chat_id=shared-main").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["content"] == "The office is occupied."

    cleared = client.delete("/v1/chat/history?chat_id=shared-main")
    assert cleared.status_code == 200
    assert cleared.json()["deleted_turns"] == 1
    assert client.get("/v1/chat/history?chat_id=shared-main").json()[
        "messages"] == []


def test_chat_context_endpoint_shows_bundle_for_free(api):
    client, session = api
    r = client.get("/v1/chat/context")
    assert r.status_code == 200
    body = r.json()
    assert set(("map", "devices", "descriptors", "scenes", "builder",
                "usage", "generated_at")) <= set(body)
    assert body["usage"]["plan"] == "free"
    assert not _usage(session)                    # preview costs nothing


def test_chat_context_provenance_ownership(api):
    """The bundle documents itself: provenance names every section the
    model saw; registry devices are marked owned by person:owner; the
    seeded persons/places anchors are listed."""
    client, _ = api
    body = client.get("/v1/chat/context").json()
    prov = body["provenance"]
    assert "sections" in prov
    for s in ("map", "devices", "descriptors", "scenes", "reference",
              "builder", "usage"):
        assert s in prov["sections"], s
    assert prov["history"]["max_turns"] == 12
    assert prov["attachments"]["max"] == 5
    # every registry device is owned by person:owner + links its entity
    for d in body["devices"]:
        assert d["owned"] is True and d["owner"] == "person:owner"
    ids = {e["id"] for e in body.get("persons", [])}
    assert "person:owner" in ids
    ids = {e["id"] for e in body.get("places", [])}
    assert "place:home" in ids


def test_chat_config_roundtrip_and_gates(api):
    """PUT /v1/chat/config persists; the context payload echoes it and
    honors section toggles + the custom window."""
    client, _ = api
    cfg = {"window_s": 300.0, "max_bundle_chars": 0,
           "include_scenes": False, "include_descriptors": False,
           "include_reference": True, "include_builder": True,
           "include_usage": True, "include_persons_places": True}
    r = client.put("/v1/chat/config", json=cfg)
    assert r.status_code == 200, r.text
    got = client.get("/v1/chat/config").json()["config"]
    assert got["window_s"] == 300.0 and got["include_scenes"] is False

    body = client.get("/v1/chat/context").json()
    assert body["window_s"] == 300.0
    assert "scenes" not in body
    assert "descriptors" not in body
    assert "scenes" not in body["provenance"]["sections"]
    # restore defaults so later tests aren't order-dependent
    client.put("/v1/chat/config", json={**cfg, "include_scenes": True,
                                        "include_descriptors": True,
                                        "window_s": 900.0})


def test_chat_model_tier_routes_and_422s(api, monkeypatch):
    """The app's model chooser selects the model — numbered picks map to
    concrete ids and widen the bundle for 5/6; a bogus tier is rejected."""
    client, _ = api
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["tier"] = tier
        seen["bundle"] = json.loads(
            messages[1]["content"].split("\n", 1)[1])
        return {"answer": "ok", "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat",
                    json={"message": "hi", "model": "advanced"})
    assert r.status_code == 200, r.text
    assert seen["tier"] == "advanced"
    assert r.json()["model"]["id"] == chat._ADVANCED_MODEL

    for pick in ("4", "5", "6"):
        r = client.post("/v1/chat",
                        json={"message": "hi", "model": pick})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == {"tier": pick,
                                     "id": chat._TIER_MODELS[pick]}
        assert seen["tier"] == pick

    r = client.post("/v1/chat",
                    json={"message": "hi", "model": "gpt-99"})
    assert r.status_code == 422


def test_chat_returns_assessments(api, monkeypatch):
    """Model's probabilistic claims ride the response — actionable."""
    client, _ = api

    def fake_answer(messages, tier="standard", usage_out=None):
        return {"answer": "The room looks empty.",
                "assessments": [
                    {"claim": "room is unoccupied", "probability": 0.82,
                     "basis": "reference snr_db.max ~10 dB vs 40 spikes"},
                    "not-a-dict"],
                "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={"message": "is anyone home?"})
    assert r.status_code == 200
    asmts = r.json()["assessments"]
    assert len(asmts) == 1
    assert asmts[0]["claim"] == "room is unoccupied"


def test_chat_advanced_widens_the_bundle(db, monkeypatch):
    """Advanced detail ships more descriptors/scenes + the builder's
    last proposal; standard stays compact."""
    session, _ = db
    from server.db import ContextEvidence
    for i in range(20):
        session.add(ContextEvidence(
            user_id=1, evidence_key=f"k{i}.v1", device_id="d",
            value=json.dumps({"v": i}),
            timestamp=__import__("time").time(), confidence=0.5))
    session.commit()
    user = session.get(User, 1)
    std = chat._chat_context(session, 1, user)
    adv = chat._chat_context(session, 1, user, detail="advanced")
    assert len(std["descriptors"]) == 12
    assert len(adv["descriptors"]) == 20
    assert "devices" in std and "devices" in adv


def test_chat_requires_message_or_attachment(api):
    client, _ = api
    assert client.post("/v1/chat", json={}).status_code == 422


def test_chat_bundle_carries_radar_fields_and_site(api, monkeypatch):
    """Regression: the app agent must SEE physical descriptors (radar
    snr/range, csi variance) and the node's site — not just text cues."""
    client, session = api
    from server.db import ContextEvidence
    session.add(ContextEvidence(
        user_id=1, evidence_key="context.descriptors.v1",
        device_id="dev-1", timestamp=__import__("time").time(),
        value=json.dumps({"value": {
            "scene": "quiet room",
            "location": {"lat": 43.65, "lon": -79.38,
                         "site": "43 Hillsmount"},
            "sensors": {
                "radar-a316": {"type": "radar", "n": 40, "rate_hz": 10.0,
                               "fields": {"snr_db": {"mean": 0.3},
                                          "range_m": {"mean": 0.0}}},
                "csi-bb8b": {"type": "wifi_csi", "n": 90,
                             "fields": {"amplitude_var": {
                                 "mean": 0.004}}}}}})))
    session.commit()
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["bundle"] = json.loads(
            messages[1]["content"].split("\n", 1)[1])
        return {"answer": "The room looks empty — radar SNR is flat.",
                "widgets": [
                    {"type": "states", "title": "Evidence", "items": [
                        {"key": "occupancy.v1", "entity": "place:bedroom",
                         "value": {"occupied": False},
                         "confidence": 0.7, "confirmed": False}]}]}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={"message": "is the room empty?"})
    assert r.status_code == 200, r.text
    scene = seen["bundle"]["scenes"][0]
    radar = scene["sensors"]["radar-a316"]
    assert radar["fields"]["snr_db"]["mean"] == 0.3
    assert scene["location"]["site"] == "43 Hillsmount"
    assert "empty" in r.json()["answer"]


def test_chat_over_quota_402s(api, monkeypatch):
    client, session = api
    monkeypatch.setenv("INFERENCE_MONTHLY_QUOTA", "0")
    monkeypatch.setattr(chat, "_openai_answer",
                        lambda m, **k: {"answer": "x", "widgets": []})
    r = client.post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 402


def test_attachment_parts():
    img = chat.AttachmentIn(
        name="a.png", mime="image/png",
        data_b64=base64.b64encode(b"\x89PNG" + b"\x00" * 40).decode())
    parts, note = chat._attachment_parts(img)
    assert parts[0]["type"] == "image_url" and note == "image:a.png"

    txt = chat.AttachmentIn(
        name="n.csv", mime="text/csv",
        data_b64=base64.b64encode(b"a,b\n1,2").decode())
    parts, note = chat._attachment_parts(txt)
    assert "a,b\n1,2" in parts[0]["text"] and note == "file:n.csv"

    blob = chat.AttachmentIn(
        name="f.bin", mime="application/octet-stream",
        data_b64=base64.b64encode(b"\x00\x01").decode())
    parts, note = chat._attachment_parts(blob)
    assert note == "skipped:f.bin"
    assert "cannot be read" in parts[0]["text"]


def test_chat_sends_images_to_vision(api, monkeypatch):
    client, _ = api
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["messages"] = messages
        return {"answer": "a cat", "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    client.post("/v1/chat", json={
        "message": "what is this?",
        "attachments": [{
            "name": "cat.png", "mime": "image/png",
            "data_b64": base64.b64encode(b"\x89PNG" + b"0" * 32).decode()}]})
    user_msg = seen["messages"][-1]
    kinds = [p["type"] for p in user_msg["content"]]
    assert kinds == ["text", "image_url"]


def test_transcribe_and_tts_are_metered(api, monkeypatch):
    client, session = api
    monkeypatch.setattr(chat, "_openai_transcribe",
                        lambda name, data, mime: "hello there")
    r = client.post("/v1/chat/transcribe",
                    files={"file": ("a.webm", b"RIFF" + b"0" * 100,
                                    "audio/webm")})
    assert r.status_code == 200 and r.json()["text"] == "hello there"

    monkeypatch.setattr(chat, "_openai_tts",
                        lambda text, voice: b"MP3DATA")
    r = client.post("/v1/chat/tts", json={"text": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"MP3DATA"

    kinds = sorted(r.kind for r in _usage(session))
    assert kinds == ["chat_transcribe", "chat_tts"]


def test_transcribe_rejects_empty(api):
    client, _ = api
    r = client.post("/v1/chat/transcribe",
                    files={"file": ("a.webm", b"", "audio/webm")})
    assert r.status_code == 422


def test_calibration_is_flat_and_covers_imu_radar_csi(api, monkeypatch):
    """Regression: calibration surfaced as
    {radar_occupancy:{radar_occupancy:{…}}} — the reference map IS the
    calibration; all three blocks must sit at top level."""
    client, _ = api
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["bundle"] = json.loads(
            messages[1]["content"].split("\n", 1)[1])
        return {"answer": "ok", "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 200, r.text
    cal = seen["bundle"]["calibration"]
    assert set(cal) == {"radar_occupancy", "csi_link", "watch_imu_worn"}
    ref = seen["bundle"]["reference"]
    assert set(ref) == set(cal)
    unworn = cal["watch_imu_worn"]["fields"]["acc_std_per_axis_g"][
        "unworn_measured"]
    assert unworn["z"] == pytest.approx(0.0015)
    assert "occupied" in cal["csi_link"]["fields"]["csi_amp_mean"]


def test_user_calibrations_reach_the_prompt(api, monkeypatch):
    """A node-uploaded labeled baseline (e.g. sleep) lands under
    calibration.user_calibrations so the model can compare the live
    window against the measured label."""
    from server.db import DescriptorCalibration
    client, session = api
    session.add(DescriptorCalibration(
        user_id=1, device_id="dev-1", label="sleep",
        doc=json.dumps({"format": "calibration/v1", "label": "sleep",
                        "duration_s": 1200,
                        "sensors": {"radio-cacd": {
                            "type": "radio_env", "n": 240,
                            "fields": {"rssi": {"mean": -52.1}}}}})))
    session.commit()
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["bundle"] = json.loads(
            messages[1]["content"].split("\n", 1)[1])
        return {"answer": "ok", "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={"message": "am I asleep?"})
    assert r.status_code == 200, r.text
    cals = seen["bundle"]["calibration"]["user_calibrations"]
    assert "sleep" in cals
    assert cals["sleep"]["sensors"]["radio-cacd"]["fields"]["rssi"][
        "mean"] == pytest.approx(-52.1)


def test_spaces_and_emitter_anchors_reach_the_bundle(api, monkeypatch):
    """The user-drawn spatial layer + registry-anchored emitters must
    reach the prompt: 'which devices are in Myroom' needs both."""
    from server.db import (ContextEvidence, Device, DevicePlacement,
                           Space, Zone)
    client, session = api
    dev = session.query(Device).filter(Device.userId == 1).first()
    if dev is None:
        dev = Device(userId=1, device_uuid="thoth-esp32-1",
                     device_name="thoth-esp32-a0a2",
                     device_type="thoth", mac_address="dc:1e:d5:6a:a0:a2",
                     approved=True, online=True)
        session.add(dev)
    else:
        dev.mac_address = "dc:1e:d5:6a:a0:a2"
        dev.device_name = "thoth-esp32-a0a2"
    session.flush()
    sp = Space(user_id=1, name="Myroom", width_m=4.0, height_m=5.0)
    session.add(sp)
    session.flush()
    session.add(Zone(space_id=sp.id, name="desk",
                     polygon_json=json.dumps([[0, 0], [2, 0], [2, 2]])))
    session.add(DevicePlacement(device_id=dev.deviceId, space_id=sp.id,
                                x=1.0, y=1.0))
    # One uplink scene whose radio scan contains the device's own MAC —
    # _anchor_emitters must tag it device_ref/known.
    session.add(ContextEvidence(
        user_id=1, device_id=str(dev.device_uuid),
        evidence_key="context.descriptors.v1",
        timestamp=__import__("time").time(),
        value=json.dumps({"value": {"sensors": {"radio-0": {
            "type": "radio_env",
            "scan": {"emitters": 3, "top": [
                {"id": "dc:1e:d5:6a:a0:a2", "rssi": -44,
                 "name": "thoth-esp32-a0a2"},
                {"id": "aa:bb:cc:dd:ee:ff", "rssi": -88}]}}}}})))
    session.commit()
    seen = {}

    def fake_answer(messages, tier="standard", usage_out=None):
        seen["bundle"] = json.loads(
            messages[1]["content"].split("\n", 1)[1])
        return {"answer": "ok", "widgets": []}
    monkeypatch.setattr(chat, "_openai_answer", fake_answer)

    r = client.post("/v1/chat", json={"message": "what's in myroom?"})
    assert r.status_code == 200, r.text
    bundle = seen["bundle"]
    space = next(s for s in bundle["spaces"] if s["name"] == "Myroom")
    assert space["zones"] == [{"name": "desk",
                               "polygon": [[0, 0], [2, 0], [2, 2]]}]
    assert space["devices"][0]["ref"].startswith("device:")
    top = bundle["scenes"][0]["sensors"]["radio-0"]["scan"]["top"]
    assert top[0]["device_ref"].startswith("device:")
    assert top[0]["known"] is True
    assert "device_ref" not in top[1]


def test_openai_quota_error_maps_to_429(api, monkeypatch):
    """An exhausted model balance must surface as a clean 429 — a bare
    500 is what 'chat failed' bubbles on mobile."""
    import httpx
    import openai
    client, _ = api

    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**_kw):
                    raise openai.RateLimitError(
                        "credit_balance_exhausted", response=httpx.Response(
                            429, request=httpx.Request("POST", "http://x")),
                        body={"error": {"code": "credit_balance_exhausted"}})

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda **_: _Boom())
    r = client.post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 429
    assert "quota" in r.json()["detail"]


def test_prompt_is_unbiased_and_schema_is_strict():
    """No measured numbers/placeholders baked into the prompt — the
    bundle carries calibration data; the prompt carries method. The
    strict schema is what guarantees per-claim probabilities."""
    for bad in ("7.6", "E1 rig", "15-40", "snr_max spikes"):
        assert bad not in chat._SYSTEM_PROMPT
    sch = chat._ANSWER_SCHEMA
    assert sch["strict"] is True
    item = sch["schema"]["properties"]["assessments"]["items"]
    assert item["required"] == ["claim", "probability", "basis"]


