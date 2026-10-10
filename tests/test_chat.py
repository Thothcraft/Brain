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
from server.db import ApiUsage, Base, User, get_db
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


def test_chat_context_endpoint_shows_bundle_for_free(api):
    client, session = api
    r = client.get("/v1/chat/context")
    assert r.status_code == 200
    body = r.json()
    assert set(("map", "devices", "descriptors", "scenes", "builder",
                "usage", "generated_at")) <= set(body)
    assert body["usage"]["plan"] == "free"
    assert not _usage(session)                    # preview costs nothing


def test_chat_model_tier_routes_and_422s(api, monkeypatch):
    """The app's Fast/Advanced slider selects the model — advanced also
    widens the context bundle; a bogus tier is rejected."""
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

    r = client.post("/v1/chat",
                    json={"message": "hi", "model": "gpt-99"})
    assert r.status_code == 422


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
