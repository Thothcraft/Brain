"""v1 API contract tests - versioned endpoints, tenant isolation, manifest validation."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (
    Base, Device, DeviceCapture, DeviceCaptureChunk, DeviceCommand,
    DeviceDeployment, TrainedModel, User, get_db,
)
from server.v1 import router as v1_router


@pytest.fixture
def v1api():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='alice', email='a@x.invalid',
                    hashed_password='x', role=0, plan='free')
        other = User(userId=2, username='bob', email='b@x.invalid',
                     hashed_password='x', role=0, plan='free')
        session.add_all([user, other])
        session.add(Device(userId=1, device_uuid='pi-a', device_name='thoth-pi-a',
                           device_type='thoth', approved=True, online=True,
                           hardware_info=json.dumps({
                               "platform": "linux",
                               "sensors": [{"id": "radar-0", "type": "radar"},
                                           {"id": "imu-0", "type": "imu"}]})))
        from datetime import datetime, timedelta
        session.add(Device(userId=1, device_uuid='pi-b', device_name='thoth-pi-b',
                           device_type='thoth', approved=True, online=False,
                           last_seen=datetime.utcnow() - timedelta(hours=2)))
        session.add(Device(userId=2, device_uuid='pi-c', device_name='thoth-pi-c',
                           device_type='thoth', approved=True, online=True))
        session.commit()
        app = FastAPI()
        app.include_router(v1_router, prefix='/v1')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user


def test_v1_devices_includes_offline(v1api):
    client, _, _ = v1api
    res = client.get('/v1/devices')
    assert res.status_code == 200
    body = res.json()
    names = {d['name'] for d in body['devices']}
    # Both owned devices visible - offline included (mobile parity).
    assert names == {'thoth-pi-a', 'thoth-pi-b'}
    pi_b = next(d for d in body['devices'] if d['name'] == 'thoth-pi-b')
    assert pi_b['online'] is False
    # v1 contract fields present
    assert pi_b['stable_uuid'] == 'pi-b'
    assert 'sensors' in pi_b


def test_v1_devices_tenant_isolation(v1api):
    client, _, _ = v1api
    res = client.get('/v1/devices')
    names = {d['name'] for d in res.json()['devices']}
    assert 'thoth-pi-c' not in names  # other user's device invisible
    res = client.get('/v1/devices/pi-c')
    assert res.status_code == 404


def test_v1_device_sensors(v1api):
    client, _, _ = v1api
    res = client.get('/v1/devices/pi-a/sensors')
    assert res.status_code == 200
    sensors = res.json()['sensors']
    types = {s['type'] for s in sensors}
    assert types == {'radar', 'imu'}


def test_v1_stream_requires_ownership(v1api):
    client, _, _ = v1api
    res = client.get('/v1/devices/pi-c/streams/radar-0')
    assert res.status_code == 404  # not your device


def test_v1_stream_disconnected_state(v1api):
    client, _, _ = v1api
    res = client.get('/v1/devices/pi-b/streams/radar-0')
    assert res.status_code == 200
    assert res.json()['state'] == 'disconnected'


def test_v1_stream_returns_real_samples(v1api):
    client, session, _ = v1api
    device = session.query(Device).filter_by(device_uuid='pi-a').first()
    session.add(DeviceCaptureChunk(
        device_id=device.deviceId, user_id=1, minute='20260923_0223',
        chunk_index=0, status='stored',
        payload=json.dumps({"features": {"radar-0": {"snr_db": 14.2}}})))
    session.commit()
    res = client.get('/v1/devices/pi-a/streams/radar-0')
    assert res.status_code == 200
    samples = res.json()['samples']
    assert len(samples) == 1
    s = samples[0]
    assert s['sensor_id'] == 'radar-0'
    assert s['payload'] == {"snr_db": 14.2}   # real measurement, not a boolean
    assert s['device_id'] == 'pi-a'


def test_v1_stream_preserves_sample_semantics(v1api):
    """Real SensorSamples keep their own timestamp/sequence/units (#13)."""
    client, session, _ = v1api
    device = session.query(Device).filter_by(device_uuid='pi-a').first()
    session.add(DeviceCaptureChunk(
        device_id=device.deviceId, user_id=1, minute='20260923_0300',
        chunk_index=0, status='stored',
        payload=json.dumps({"samples": [{
            "device_id": "pi-a", "sensor_id": "radar-0",
            "sensor_type": "radar", "timestamp": 1727000000.5,
            "sequence": 42, "payload_type": "ndarray:float32",
            "payload": [1.0, 2.0], "sample_rate": 60.0,
            "units": {"amplitude": "dB"}}]})))
    session.commit()
    res = client.get('/v1/devices/pi-a/streams/radar-0')
    assert res.status_code == 200
    s = res.json()['samples'][0]
    assert s['sequence'] == 42
    assert s['timestamp'] == 1727000000.5
    assert s['sample_rate'] == 60.0
    assert s['units'] == {"amplitude": "dB"}
    assert s['payload_type'] == 'ndarray:float32'


def test_v1_stream_cursor_scans_all_minutes(v1api):
    """A cursor poll must not skip rows in older minutes (#14)."""
    from datetime import datetime, timedelta
    client, session, _ = v1api
    device = session.query(Device).filter_by(device_uuid='pi-a').first()
    base = datetime.utcnow()
    session.add(DeviceCaptureChunk(
        device_id=device.deviceId, user_id=1, minute='20260923_0200',
        chunk_index=0, status='stored', updated_at=base,
        payload=json.dumps({"features": {"radar-0": {"a": 1}}})))
    session.add(DeviceCaptureChunk(
        device_id=device.deviceId, user_id=1, minute='20260923_0201',
        chunk_index=0, status='stored', updated_at=base + timedelta(seconds=1),
        payload=json.dumps({"features": {"radar-0": {"b": 2}}})))
    session.commit()

    # First poll (no cursor) returns only the latest minute.
    r1 = client.get('/v1/devices/pi-a/streams/radar-0').json()
    assert [s['payload'] for s in r1['samples']] == [{"b": 2}]
    cursor = r1['cursor']

    # A late-arriving row lands in the OLDER minute after the cursor.
    session.add(DeviceCaptureChunk(
        device_id=device.deviceId, user_id=1, minute='20260923_0200',
        chunk_index=1, status='stored',
        updated_at=base + timedelta(seconds=2),
        payload=json.dumps({"features": {"radar-0": {"c": 3}}})))
    session.commit()

    r2 = client.get(
        f'/v1/devices/pi-a/streams/radar-0?cursor={cursor}').json()
    # The backfilled older-minute row is delivered, not skipped.
    assert {"c": 3} in [s['payload'] for s in r2['samples']]


def test_v1_model_manifest_validation(v1api):
    client, _, _ = v1api
    bad = client.post('/v1/models', json={"format": "wrong", "processor": "onnx"})
    assert bad.status_code == 422
    good = client.post('/v1/models', json={
        "format": "whispy-model/v1",
        "name": "radar-occupancy-v2",
        "processor": "torchscript",
        "inputs": [{"sensor": "radar", "window_seconds": 2.0}],
        "outputs": ["empty", "occupied"]})
    assert good.status_code == 201
    assert good.json()['model']['name'] == 'radar-occupancy-v2'

    # Legacy pre-rename format is still accepted during the transition.
    legacy = client.post('/v1/models', json={
        "format": "thoth-model/v1",
        "name": "legacy-rule",
        "processor": "rule",
        "inputs": [{"sensor": "radar"}]})
    assert legacy.status_code == 201


def test_v1_deployment_lifecycle(v1api):
    client, session, _ = v1api
    model = TrainedModel(user_id=1, name='m1', processor_type='torchscript')
    session.add(model)
    session.commit()
    res = client.post('/v1/deployments',
                      json={"model_id": str(model.id), "device_id": "pi-a"})
    assert res.status_code == 201
    dep = res.json()
    assert dep['state'] == 'queued'
    assert dep['device_id'] == 'pi-a'

    listed = client.get('/v1/deployments').json()['deployments']
    assert len(listed) == 1
    assert listed[0]['deployment_id'] == dep['deployment_id']


def test_v1_deployment_rejects_foreign_device(v1api):
    client, session, _ = v1api
    model = TrainedModel(user_id=1, name='m1', processor_type='rule')
    session.add(model)
    session.commit()
    res = client.post('/v1/deployments',
                      json={"model_id": str(model.id), "device_id": "pi-c"})
    assert res.status_code == 404


def test_v1_capture_start_stop(v1api):
    client, _, _ = v1api
    res = client.post('/v1/devices/pi-a/captures', json={"sensors": ["radar-0"]})
    assert res.status_code == 201
    cap = res.json()
    # Not 'active' — the node has not confirmed yet (§22 lifecycle).
    assert cap['state'] == 'requested'
    assert cap['device_id'] == 'pi-a'
    stop = client.post(f"/v1/captures/{cap['id']}/stop")
    assert stop.status_code == 200
    assert stop.json()['state'] == 'stopping'


def test_v1_capture_stop_targets_owning_device(v1api):
    """A capture started on pi-b must queue its stop on pi-b, not pi-a."""
    client, session, _ = v1api
    res = client.post('/v1/devices/pi-b/captures', json={"sensors": []})
    assert res.status_code == 201
    cap_id = res.json()['id']

    stop = client.post(f"/v1/captures/{cap_id}/stop")
    assert stop.status_code == 200
    assert stop.json()['device_id'] == 'pi-b'

    # The queued capture_stop command must be addressed to pi-b's PK.
    pi_b = session.query(Device).filter_by(device_uuid='pi-b').first()
    pi_a = session.query(Device).filter_by(device_uuid='pi-a').first()
    cmds = session.query(DeviceCommand).filter_by(command='capture_stop').all()
    assert len(cmds) == 1
    assert cmds[0].device_id == pi_b.deviceId
    assert cmds[0].device_id != pi_a.deviceId


def test_v1_capture_stop_unknown_id_404(v1api):
    client, _, _ = v1api
    res = client.post('/v1/captures/does-not-exist/stop')
    assert res.status_code == 404


def test_v1_capture_persisted_and_listed(v1api):
    """A started capture is a durable row that reconciles with list."""
    client, session, _ = v1api
    res = client.post('/v1/devices/pi-a/captures', json={"sensors": ["radar-0"]})
    cap_id = res.json()['id']
    # Durable record exists with the same ID.
    row = session.query(DeviceCapture).filter_by(capture_id=cap_id).first()
    assert row is not None and row.state == 'requested'
    # The same ID appears in the device capture list.
    listed = client.get('/v1/devices/pi-a/captures').json()['captures']
    assert any(c['id'] == cap_id for c in listed)


# ---------------------------------------------------------------------------
# Scoped automation credentials (#15)
# ---------------------------------------------------------------------------

@pytest.fixture
def v1api_realauth():
    """v1 app with the REAL get_current_user so X-Api-Key resolution runs."""
    from server.db import AutomationKey
    from server.auth import hash_automation_key
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(User(userId=1, username='alice', email='a@x.invalid',
                         hashed_password='x', role=0, plan='free'))
        session.add(Device(userId=1, device_uuid='pi-a', device_name='thoth-pi-a',
                           device_type='thoth', approved=True, online=True,
                           hardware_info=json.dumps({"sensors": [
                               {"id": "radar-0", "type": "radar"}]})))
        # A stream-scoped key and a deploy-scoped key.
        session.add(AutomationKey(
            key_hash=hash_automation_key("tc_stream"), user_id=1,
            name="streamer", scopes=json.dumps(["sensor:stream"])))
        session.add(AutomationKey(
            key_hash=hash_automation_key("tc_deploy"), user_id=1,
            name="deployer", scopes=json.dumps(["model:deploy"])))
        session.commit()
        app = FastAPI()
        app.include_router(v1_router, prefix='/v1')
        # get_current_user resolves its session via server.auth.get_db (a
        # local function that shadows the server.db import), so override both.
        import server.auth as _auth
        app.dependency_overrides[get_db] = lambda: session
        app.dependency_overrides[_auth.get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session


def test_v1_automation_key_scope_enforced(v1api_realauth):
    client, _ = v1api_realauth
    # sensor:stream key may stream…
    ok = client.get('/v1/devices/pi-a/streams/radar-0',
                    headers={"X-Api-Key": "tc_stream"})
    assert ok.status_code == 200
    # …but may NOT deploy models (missing model:deploy scope).
    denied = client.get('/v1/models', headers={"X-Api-Key": "tc_stream"})
    assert denied.status_code == 403
    # The deploy-scoped key can list models but cannot stream.
    assert client.get('/v1/models',
                      headers={"X-Api-Key": "tc_deploy"}).status_code == 200
    assert client.get('/v1/devices/pi-a/streams/radar-0',
                      headers={"X-Api-Key": "tc_deploy"}).status_code == 403


def test_v1_automation_key_invalid_and_revoked(v1api_realauth):
    from server.db import AutomationKey
    from server.auth import hash_automation_key
    client, session = v1api_realauth
    # Unknown key → 401.
    assert client.get('/v1/devices/pi-a/streams/radar-0',
                      headers={"X-Api-Key": "tc_bogus"}).status_code == 401
    # Revoked key → 401.
    key = session.query(AutomationKey).filter_by(
        key_hash=hash_automation_key("tc_stream")).first()
    key.revoked = True
    session.commit()
    assert client.get('/v1/devices/pi-a/streams/radar-0',
                      headers={"X-Api-Key": "tc_stream"}).status_code == 401
