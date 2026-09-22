"""Spatial API tests: spaces, zones, placement, live state, plan limits."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (
    Base, Device, DeviceCaptureChunk, Space, User, get_db,
)
from server.endpoints.spatial_endpoints import router as spatial_router


@pytest.fixture
def api():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='test', email='t@x.invalid',
                    hashed_password='x', role=0, plan='research')
        device = Device(deviceId=1, userId=1, device_uuid='dev-1',
                        device_name='thoth-chen', device_type='thoth')
        session.add_all([user, device])
        session.commit()
        app = FastAPI()
        app.include_router(spatial_router, prefix='/api')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user


def test_space_crud(api):
    client, session, _ = api
    res = client.post('/api/spaces', json={'name': 'office', 'width_m': 6, 'height_m': 5})
    assert res.status_code == 201
    space = res.json()['space']
    assert space['name'] == 'office'

    assert client.get('/api/spaces').json()['spaces'][0]['name'] == 'office'
    assert client.get(f"/api/spaces/{space['id']}").json()['space']['width_m'] == 6

    res = client.put(f"/api/spaces/{space['id']}", json={'name': 'lab'})
    assert res.json()['space']['name'] == 'lab'

    assert client.delete(f"/api/spaces/{space['id']}").json()['success']
    assert client.get('/api/spaces').json()['spaces'] == []


def test_space_limit_enforced(api):
    client, session, user = api
    user.plan = 'free'  # space_limit = 1
    session.commit()
    assert client.post('/api/spaces', json={'name': 'a'}).status_code == 201
    res = client.post('/api/spaces', json={'name': 'b'})
    assert res.status_code == 403


def test_zones_and_placement(api):
    client, session, _ = api
    space = client.post('/api/spaces', json={'name': 'bedroom'}).json()['space']
    zone = client.post(f"/api/spaces/{space['id']}/zones", json={
        'name': 'bed', 'polygon': [[0, 0], [2, 0], [2, 2], [0, 2]],
    }).json()['zone']
    assert zone['name'] == 'bed'

    res = client.put('/api/spaces/devices/dev-1/placement', json={
        'space_id': space['id'], 'x': 1.0, 'y': 1.0,
        'rotation_deg': 0, 'fov_deg': 90, 'range_m': 8,
    })
    assert res.json()['placement']['device_id'] == 'dev-1'

    # placement is unique per device — re-place moves it
    res = client.put('/api/spaces/devices/dev-1/placement', json={
        'space_id': space['id'], 'x': 3.0, 'y': 3.0,
    })
    assert res.json()['placement']['x'] == 3.0

    assert client.delete('/api/spaces/devices/dev-1/placement').json()['success']


def test_space_state_occupied(api):
    client, session, _ = api
    space = client.post('/api/spaces', json={'name': 'office'}).json()['space']
    client.post(f"/api/spaces/{space['id']}/zones", json={
        'name': 'desk', 'polygon': [[0, 0], [4, 0], [4, 4], [0, 4]],
    })
    client.put('/api/spaces/devices/dev-1/placement', json={
        'space_id': space['id'], 'x': 0.0, 'y': 0.0, 'rotation_deg': 0,
    })

    # device reports an occupied prediction with an XY point inside the zone
    chunk = DeviceCaptureChunk(
        device_id=1, user_id=1, minute='20260922_0100', chunk_index=0,
        status='stored', frame_count=10,
        payload=json.dumps({'model_predictions': [
            {'label': 'occupied', 'confidence': 0.9, 'xy': [[1.0, 1.0]]},
        ]}))
    session.add(chunk)
    session.commit()

    state = client.get(f"/api/spaces/{space['id']}/state").json()['state']
    assert state['occupied'] is True
    assert state['zones']['desk']['occupied'] is True
    assert state['zones']['desk']['people_count'] >= 1

    all_states = client.get('/api/spaces/state').json()['spaces']
    assert all_states[0]['name'] == 'office'


def test_space_state_empty(api):
    client, _, _ = api
    space = client.post('/api/spaces', json={'name': 'empty-room'}).json()['space']
    state = client.get(f"/api/spaces/{space['id']}/state").json()['state']
    assert state['occupied'] is False
    assert state['people_count'] == 0


def test_foreign_space_not_found(api):
    client, session, _ = api
    other = User(userId=2, username='other', email='o@x.invalid',
                 hashed_password='x', role=0, plan='research')
    session.add(other)
    session.add(Space(id=99, user_id=2, name='not-yours'))
    session.commit()
    assert client.get('/api/spaces/99').status_code == 404
