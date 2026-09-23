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
    Base, Device, DeviceCaptureChunk, DeviceDeployment, TrainedModel, User,
    get_db,
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


def test_v1_model_manifest_validation(v1api):
    client, _, _ = v1api
    bad = client.post('/v1/models', json={"format": "wrong", "processor": "onnx"})
    assert bad.status_code == 422
    good = client.post('/v1/models', json={
        "format": "thoth-model/v1",
        "name": "radar-occupancy-v2",
        "processor": "torchscript",
        "inputs": [{"sensor": "radar", "window_seconds": 2.0}],
        "outputs": ["empty", "occupied"]})
    assert good.status_code == 201
    assert good.json()['model']['name'] == 'radar-occupancy-v2'


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
    assert cap['state'] == 'active'
    stop = client.post(f"/v1/captures/{cap['id']}/stop")
    assert stop.status_code == 200
    assert stop.json()['state'] == 'stopped'
