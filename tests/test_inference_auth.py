"""Hosted-inference authorization + metering (server.inference_auth).

Every hosted model call — /v1/context/infer, background context-builder
builds, ai assistant queries — reserves a row in the shared api_usage
ledger before execution and reconciles it after; the plan's monthly
``inference_quota`` caps reserved+ok rows per UTC month.
"""
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import ApiUsage, Base, User, get_db
from server.inference_auth import (
    INFERENCE_SOURCE, inference_allowance, inference_call,
    inference_status, inference_used, period_start,
)
import server.v1.context_builder as cb
import server.v1.context_infer as infer
from server.v1.context_infer import router as infer_router


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={
        "check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        session.add(User(userId=1, username="t", email="t@e.invalid",
                         hashed_password="x", role=0, plan="free"))
        session.add(User(userId=2, username="u2", email="u2@e.invalid",
                         hashed_password="x", role=0, plan="research"))
        session.commit()
        yield session, factory
    engine.dispose()


def _usage_rows(session, user_id=1):
    return session.query(ApiUsage).filter(
        ApiUsage.user_id == user_id,
        ApiUsage.source == INFERENCE_SOURCE).all()


def _status(row):
    return json.loads(row.meta or "{}").get("status")


def test_successful_call_is_metered_ok(db):
    session, _ = db
    with inference_call(session, 1, kind="context_infer") as meter:
        meter["model_id"] = "gpt-4o-mini"
        meter["tokens"] = 123
    rows = _usage_rows(session)
    assert len(rows) == 1
    row = rows[0]
    assert _status(row) == "ok" and row.kind == "context_infer"
    assert row.device_id == "cloud" and row.source == "inference"
    assert row.model_id == "gpt-4o-mini" and row.tokens == 123
    assert row.latency_ms is not None
    assert inference_used(session, 1) == 1


def test_failed_call_counts_and_records_error(db):
    session, _ = db
    with pytest.raises(HTTPException):
        with inference_call(session, 1, kind="context_build"):
            raise HTTPException(502, "model blew up")
    row = _usage_rows(session)[0]
    assert _status(row) == "error"
    assert "model blew up" in row.meta
    assert inference_used(session, 1) == 1       # crashes still consume


def test_quota_exhaustion_rejects_with_402(db, monkeypatch):
    session, _ = db
    monkeypatch.setenv("INFERENCE_MONTHLY_QUOTA", "1")
    calls = []

    def fake_llm():
        calls.append(1)

    with inference_call(session, 1, kind="context_infer"):
        fake_llm()
    with pytest.raises(HTTPException) as caught:
        with inference_call(session, 1, kind="context_infer"):
            fake_llm()
    assert caught.value.status_code == 402
    assert len(calls) == 1                        # model never ran
    statuses = [_status(r) for r in _usage_rows(session)]
    assert statuses == ["ok", "rejected"]
    st = inference_status(session, session.get(User, 1))
    assert st["used"] == 1 and st["remaining"] == 0
    assert st["allowance"] == 1


def test_env_override_unlimited(db, monkeypatch):
    session, _ = db
    monkeypatch.setenv("INFERENCE_MONTHLY_QUOTA", "-1")
    assert inference_allowance(session.get(User, 1)) is None
    for _ in range(3):
        with inference_call(session, 1, kind="ai_query"):
            pass
    assert inference_used(session, 1) == 3


def test_plan_allowances(db):
    session, _ = db
    assert inference_allowance(session.get(User, 1)) == 200      # free
    assert inference_allowance(session.get(User, 2)) == 20000    # research


def test_period_start_is_utc_month_boundary():
    import calendar
    from datetime import datetime, timezone
    ts = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    start = period_start(ts)
    d = datetime.fromtimestamp(start, tz=timezone.utc)
    assert (d.year, d.month, d.day) == (2026, 10, 1)
    assert calendar.monthrange(2026, 10)[0] is not None


# ---------------------------------------------------------------------------
# Endpoint + builder wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def api(db):
    session, _ = db
    user = session.get(User, 1)
    app = FastAPI()
    app.include_router(infer_router, prefix="/v1")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        yield client, session


def test_infer_endpoint_is_metered_and_usage_endpoint_reports(api,
                                                              monkeypatch):
    client, session = api
    monkeypatch.setattr(infer, "_openai_form",
                        lambda req, usage_out=None, **_: {
                            "summary": "s", "states": []})
    r = client.post("/v1/context/infer", json={"descriptors": {}})
    assert r.status_code == 200
    rows = _usage_rows(session)
    assert len(rows) == 1 and rows[0].kind == "context_infer:standard"
    assert _status(rows[0]) == "ok"

    u = client.get("/v1/context/inference/usage")
    assert u.status_code == 200
    body = u.json()
    assert body["plan"] == "free" and body["used"] == 1
    assert body["allowance"] == 200 and body["remaining"] == 199


def test_infer_endpoint_402s_over_quota(api, monkeypatch):
    client, session = api
    monkeypatch.setenv("INFERENCE_MONTHLY_QUOTA", "1")
    monkeypatch.setattr(infer, "_openai_form",
                        lambda req, usage_out=None, **_: {
                            "summary": "s", "states": []})
    assert client.post("/v1/context/infer", json={}).status_code == 200
    r = client.post("/v1/context/infer", json={})
    assert r.status_code == 402
    assert r.json()["detail"]["error"] == "inference_quota_exceeded"
    assert [_status(r) for r in _usage_rows(session)] == ["ok", "rejected"]


def test_builder_real_llm_path_is_gated(db, monkeypatch):
    """Background builds use the same allowance — an injected local llm
    is unmetered, the hosted default path is not."""
    session, _ = db
    monkeypatch.setenv("INFERENCE_MONTHLY_QUOTA", "1")
    ran = []
    monkeypatch.setattr(cb, "_llm_update",
                        lambda bundle, usage_out=None: ran.append(1) or {
                            "summary": "x", "states": []})
    cb.run_build(session, 1, now=1_000_000.0)
    assert ran == [1]
    with pytest.raises(HTTPException) as caught:
        cb.run_build(session, 1, now=1_000_060.0)
    assert caught.value.status_code == 402
    assert ran == [1]                              # model not called
    # injected llm is a local function — not a hosted call, not metered
    cb.run_build(session, 1, now=1_000_120.0,
                 llm=lambda b: {"summary": "y", "states": []})
    assert [_status(r) for r in _usage_rows(session)] == ["ok", "rejected"]
