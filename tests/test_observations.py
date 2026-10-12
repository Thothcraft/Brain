"""observation_batch ingest — node frames → ContextEvidence (contract §3)."""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.db import Base, ContextEvidence, ContextState, User
from server.endpoints.node_ws import (ingest_observation_batch,
                                      _apply_prediction_context)


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


def test_item_conflict_rolls_back_only_that_row(tmp_path, monkeypatch):
    """A dup committed by a CONCURRENT connection between the existence
    check and the flush must NOT roll back earlier items — per-item
    savepoints isolate the IntegrityError (was: db.rollback() undid the
    whole batch while ``stored`` still counted the rows).

    WAL mode lets the competing writer commit while the ingester's read
    transaction is open — exactly the production race."""
    import server.v1.context as v1ctx
    from sqlalchemy.orm import sessionmaker

    # Default pool — the two sessions MUST be on separate connections
    # for a genuine cross-connection conflict.
    engine = create_engine(
        f'sqlite:///{tmp_path}/race.db',
        connect_args={'timeout': 30})
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        c.exec_driver_sql('PRAGMA journal_mode=WAL')
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    s1, s2 = sf(), sf()
    try:
        s1.add(User(userId=1, username='test', email='t@x.invalid',
                    hashed_password='x', role=0, plan='free'))
        s1.commit()

        real_gate = v1ctx.evidence_sample_ok

        def sneak(user_id, schema, device, ts, subject):
            # Concurrent writer: commits the same external_id AFTER
            # item B's existence check ran but before its insert.
            if subject == "device:watch-2":
                s2.add(ContextEvidence(
                    user_id=1, external_id="obs:obs-2",
                    evidence_key="ble.rssi.v1", timestamp=ts,
                    device_id="other"))
                s2.commit()
            return real_gate(user_id, schema, device, ts, subject)

        monkeypatch.setattr(v1ctx, "evidence_sample_ok", sneak)
        frame = {"type": "observation_batch", "id": "b9",
                 "items": [_item(observation_id="obs-1"),
                           _item(observation_id="obs-2",
                                 subject="device:watch-2")]}
        out = ingest_observation_batch(s1, 1, "dev-uuid", frame)
        assert out == {"stored": 1, "deduplicated": 1, "rejected": 0}
        # The first item survived; the sneaky row is the only other row
        # (device_id="other", not this batch).
        mine = s1.query(ContextEvidence).filter_by(
            device_id="dev-uuid").all()
        assert [r.observation_id for r in mine] == ["obs-1"]
    finally:
        s1.close()
        s2.close()
        engine.dispose()


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


def test_state_obs_apply_as_states_before_sampling(session):
    """context.state.v1 transitions are first-class ContextStates even
    when the evidence row is sampled out — a fast occupancy flip inside
    the 60 s interval must still land."""
    frame = {"items": [
        _item(schema="context.state.v1", observation_id="s1",
              value={"key": "occupancy.v1", "entity_id": "lap-1",
                     "value": {"occupied": True, "distinct_subjects": 1},
                     "confidence": 0.9, "estimator": "ble-presence/1"}),
        _item(schema="context.state.v1", observation_id="s2",
              timestamp=1759612350.0,
              value={"key": "occupancy.v1", "entity_id": "lap-1",
                     "value": {"occupied": False, "distinct_subjects": 0},
                     "confidence": 0.5, "estimator": "ble-presence/1"}),
    ]}
    out = ingest_observation_batch(session, 1, "dev-uuid", frame)
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1", entity_id="lap-1").one()
    # the *second* transition won — sampled evidence, live state
    assert json.loads(st.value)["occupied"] is False
    assert st.estimator == "ble-presence/1"
    assert out["stored"] == 1 and out["deduplicated"] == 1


def test_occupancy_prediction_mirrors_to_occupancy_state(
        session, monkeypatch):
    """task=occupancy predictions (radar/CSI models) update occupancy.v1
    — the room state stops depending on BLE-enrolled subjects alone."""
    from contextlib import contextmanager
    import server.endpoints.node_ws as nw

    @contextmanager
    def sess():
        yield session

    monkeypatch.setattr(nw, "get_db_session", sess)
    nw._apply_prediction_context(1, "lap-1", {
        "label": "occupied", "confidence": 0.93,
        "task": "occupancy", "runtime_model_id": "occ-rf-moe"})
    st = session.query(ContextState).filter_by(
        state_key="occupancy.v1", entity_id="lap-1").one()
    v = json.loads(st.value)
    assert v == {"occupied": True, "label": "occupied",
                 "model": "occ-rf-moe"}
    assert st.confidence == 0.93
    # the raw prediction state still lands under its own key
    pr = session.query(ContextState).filter_by(
        state_key="prediction", entity_id="lap-1").one()
    assert json.loads(pr.value) == "occupied"


def test_non_occupancy_prediction_does_not_touch_occupancy(
        session, monkeypatch):
    from contextlib import contextmanager
    import server.endpoints.node_ws as nw

    @contextmanager
    def sess():
        yield session

    monkeypatch.setattr(nw, "get_db_session", sess)
    nw._apply_prediction_context(1, "lap-1", {
        "label": "Gad", "confidence": 0.9, "task": "face",
        "runtime_model_id": "face-emb"})
    assert session.query(ContextState).filter_by(
        state_key="occupancy.v1").count() == 0
