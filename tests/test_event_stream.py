"""Event-stream + prediction→context projection tests.

The node→Brain ``prediction`` event is what closes the loop end-to-end:
the daemon emits it on a label transition; Brain projects it into
ContextState (key="prediction", entity=device) → ContextEvent → server
automation rules; SSE subscribers see the NodeEvent instantly.
"""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from server import event_bus  # noqa: E402
from server.db import (  # noqa: E402
    AutomationRule, Base, ContextEvent, ContextState, Device, User,
)
from server.endpoints import node_ws  # noqa: E402


@contextmanager
def _isolated_db():
    """In-memory engine patched into node_ws.get_db_session."""
    engine = create_engine("sqlite://",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)

    @contextmanager
    def _session_factory():
        yield session

    with patch.object(node_ws, "get_db_session", _session_factory):
        yield session
    engine.dispose()


def _mk_user(db) -> User:
    u = User(username=f"streamer-{id(db)}", email=None,
             hashed_password="x")
    db.add(u)
    db.commit()
    return u


def _mk_device(db, user: User) -> Device:
    d = Device(userId=user.userId, device_name="pi-test",
               device_uuid=f"dev-{id(db)}", online=True)
    db.add(d)
    db.commit()
    return d


def test_prediction_event_projects_to_context_and_rules():
    with _isolated_db() as db:
        user = _mk_user(db)
        dev = _mk_device(db, user)
        db.add(AutomationRule(
            user_id=user.userId, name="occ-rule", enabled=True,
            when=json.dumps({"key": "prediction",
                             "entity_id": dev.device_uuid,
                             "equals": "occupied"}),
            then=json.dumps({"device_id": dev.device_uuid,
                             "actuator_id": "notification",
                             "operation": "notify",
                             "params": {"text": "occupied!"}})))
        db.commit()

        node_ws._apply_prediction_context(
            user.userId, dev.device_uuid,
            {"label": "occupied", "confidence": 0.9,
             "runtime_model_id": "rm-1"})
        state = db.query(ContextState).filter(
            ContextState.user_id == user.userId,
            ContextState.state_key == "prediction",
            ContextState.entity_id == dev.device_uuid).first()
        assert state is not None
        assert json.loads(state.value) == "occupied"
        evts = db.query(ContextEvent).filter(
            ContextEvent.user_id == user.userId,
            ContextEvent.event_key == "prediction").all()
        assert evts and evts[-1].event_type == "entered"

        # same label → edge-triggered, no new event
        node_ws._apply_prediction_context(
            user.userId, dev.device_uuid,
            {"label": "occupied", "confidence": 0.9})
        same = db.query(ContextEvent).filter(
            ContextEvent.user_id == user.userId,
            ContextEvent.event_key == "prediction").count()
        assert same == len(evts)

        # label change → "changed" event; the automation rule fires and
        # writes a durable ActionRequest (dispatched later by actuation)
        node_ws._apply_prediction_context(
            user.userId, dev.device_uuid,
            {"label": "empty", "confidence": 0.9})
        db.refresh(state)
        assert json.loads(state.value) == "empty"
        all_evts = db.query(ContextEvent).filter(
            ContextEvent.user_id == user.userId).all()
        assert any(e.event_type == "changed" for e in all_evts)


def test_event_bus_fanout():
    """publish() delivers to a subscriber's asyncio.Queue."""
    async def _run():
        q = event_bus.subscribe(7)
        try:
            event_bus.publish(7, {"id": "1", "kind": "prediction",
                                  "data": {"label": "occupied"}})
            evt = await asyncio.wait_for(q.get(), timeout=2.0)
            assert evt["kind"] == "prediction"
        finally:
            event_bus.unsubscribe(7, q)
    asyncio.run(_run())


def test_event_bus_isolated_users():
    """Events don't leak across users."""
    async def _run():
        qa = event_bus.subscribe(1)
        qb = event_bus.subscribe(2)
        try:
            event_bus.publish(1, {"id": "1", "kind": "notification"})
            got = await asyncio.wait_for(qa.get(), timeout=2.0)
            assert got["kind"] == "notification"
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(qb.get(), timeout=0.3)
        finally:
            event_bus.unsubscribe(1, qa)
            event_bus.unsubscribe(2, qb)
    asyncio.run(_run())
