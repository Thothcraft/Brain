"""observation_batch ingest — node frames → ContextEvidence (contract §3)."""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.db import Base, ContextEvidence, User
from server.endpoints.node_ws import ingest_observation_batch


@pytest.fixture
def session():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add(User(userId=1, username='test', email='t@x.invalid',
                   hashed_password='x', role=0, plan='free'))
        s.commit()
        yield s
    engine.dispose()


def _item(**kw):
    d = {"schema": "ble.rssi.v1", "observation_id": "obs-1",
         "timestamp": 1759612345.0, "source_id": "ble:hci0",
         "subject": "device:watch-1",
         "value": {"rssi_dbm": -48, "tx_power_dbm": -59},
         "provenance": {"observer": "device:node-42"}}
    d.update(kw)
    return d


def test_batch_stores_evidence_rows(session):
    frame = {"type": "observation_batch", "id": "b1",
             "items": [_item(), _item(observation_id="obs-2",
                                       subject="device:watch-2")]}
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    assert out == {"stored": 2, "deduplicated": 0, "rejected": 0}
    rows = session.query(ContextEvidence).all()
    assert len(rows) == 2
    r = rows[0]
    assert r.evidence_key == "ble.rssi.v1"
    assert r.external_id == "obs:obs-1"
    assert r.observation_id == "obs-1"
    assert r.device_id == "dev-uuid"
    assert r.source_id == "ble:hci0"
    body = json.loads(r.value)
    assert body["value"]["rssi_dbm"] == -48
    assert body["subject"] == "device:watch-1"
    assert body["batch_id"] == "b1"
    assert json.loads(r.provenance)["observer"] == "device:node-42"


def test_sampled_schema_keeps_one_row_per_subject_per_interval(session):
    frame = {"items": [
        _item(observation_id="a"),
        _item(observation_id="b", timestamp=1759612350.0),
        _item(observation_id="c", subject="person:gad"),
        _item(observation_id="d", timestamp=1759612410.0),
    ]}
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    assert out == {"stored": 3, "deduplicated": 1, "rejected": 0}
    ids = {r.observation_id for r in session.query(ContextEvidence).all()}
    assert ids == {"a", "c", "d"}


def test_redelivery_is_idempotent(session):
    frame = {"type": "observation_batch", "id": "b1", "items": [_item()]}
    ingest_observation_batch(session, 1, "dev-uuid", frame)
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    assert out == {"stored": 0, "deduplicated": 1, "rejected": 0}
    assert session.query(ContextEvidence).count() == 1


def test_malformed_items_are_skipped(session):
    frame = {"type": "observation_batch", "id": "b2", "items": [
        _item(),
        {"schema": "NO_VERSION", "value": 1},      # bad schema
        {"schema": "ble.rssi.v1"},                 # no value
        "not-a-dict",
    ]}
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    assert out == {"stored": 1, "deduplicated": 0, "rejected": 3}
    assert session.query(ContextEvidence).count() == 1


def test_unknown_schemas_are_stored(session):
    """Consumers ignore what they don't read — storage is schema-agnostic."""
    frame = {"items": [_item(schema="radio.csi.v9", observation_id="o9")]}
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    assert out["stored"] == 1
    assert session.query(ContextEvidence).one().evidence_key == \
        "radio.csi.v9"


def test_empty_and_wrong_frames_noop(session):
    assert ingest_observation_batch(session, 1, "d", {})["stored"] == 0
    assert ingest_observation_batch(
        session, 1, "d", {"items": "x"})["rejected"] == 0
