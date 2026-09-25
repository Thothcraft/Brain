"""Node↔Brain channel tests (plans/CONTRACT.md §2–§4).

Real routers + real ORM over temp-file SQLite — same pattern as conftest,
plus a patched ``get_db_session`` so the WS handler's short-lived sessions
land on the test engine. The relay round-trip runs the REST call on a
helper thread because the test thread owns the node-side socket.
"""
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from server.auth import create_access_token, get_current_user
from server.db import ApiUsage, Base, Device, NodeEvent, NodeRoom, User, get_db
from server.v1 import router as v1_router
from server.endpoints import node_ws


@pytest.fixture
def api(monkeypatch):
    # File-backed SQLite: the WS task and the REST relay run on different
    # connections — in-memory StaticPool would force them onto ONE shared
    # transaction and corrupt session state across threads.
    tmp = Path(tempfile.mkdtemp()) / 'test.db'
    # NullPool: no checkout ceiling — sessions get a fresh connection each
    # and return it on close, so leaked/late-closed sessions (WS handler
    # to_thread writes, _wait_for probes) can never exhaust the pool.
    engine = create_engine(f'sqlite:///{tmp}',
                           connect_args={'check_same_thread': False,
                                         'timeout': 30},
                           poolclass=NullPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    user = User(userId=1, username='test', email='t@example.invalid',
                hashed_password='unused', role=0, plan='free')
    device = Device(userId=1, device_uuid='dev-1', device_name='thoth-one',
                    device_type='thoth', approved=True, online=False)
    with factory() as s:
        s.add_all([user, device])
        s.commit()

    @contextmanager
    def _session_ctx():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    monkeypatch.setattr(node_ws, 'get_db_session', _session_ctx)

    app = FastAPI()
    app.include_router(v1_router, prefix='/v1')
    app.dependency_overrides[get_current_user] = lambda: user

    def _get_db():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as client:
        yield client, factory, user


def _device_token(device_uuid: str = 'dev-1') -> str:
    return create_access_token(
        data={'sub': '1', 'username': 'test', 'scopes': ['device'],
              'device_id': device_uuid},
        domain='device')


def _user_headers() -> dict:
    return {'Authorization': f'Bearer {create_access_token(data={"sub": "1", "username": "test"})}'}


def _ws_url(device_uuid: str = 'dev-1') -> str:
    return f'/v1/node/ws?device_id={device_uuid}&token={_device_token(device_uuid)}'


def _poll(factory, predicate, timeout=5.0):
    """Poll ``predicate(session)`` with a context-managed session.

    Probes must close their session — a leaked probe transaction pins a
    shared lock on the SQLite file and starves the WS handler's
    ``to_thread`` writers (connect/disconnect bookkeeping) for up to the
    30 s sqlite timeout, which is what flaked CI.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with factory() as s:
            if predicate(s):
                return True
        time.sleep(0.05)
    return False


def test_ws_rejects_bad_token(api):
    client, _factory, _user = api
    with pytest.raises(Exception):
        with client.websocket_connect(
                '/v1/node/ws?device_id=dev-1&token=bogus'):
            pass


def test_ws_relay_roundtrip_and_usage(api):
    client, factory, _user = api
    result = {}

    def do_relay():
        result['resp'] = client.post(
            '/v1/nodes/dev-1/api',
            json={'method': 'GET', 'path': '/api/v1/inference'},
            headers={**_user_headers(), 'x-thoth-source': 'portal'})

    with client.websocket_connect(_ws_url()) as ws:
        t = threading.Thread(target=do_relay, daemon=True)
        t.start()
        req = ws.receive_json()
        assert req['type'] == 'api_request'
        assert req['method'] == 'GET'
        assert req['path'] == '/api/v1/inference'
        ws.send_json({'type': 'api_response', 'id': req['id'],
                      'status': 200, 'body': {'label': 'presence'}})
        t.join(10)
        assert result['resp'].status_code == 200
        assert result['resp'].json() == {'label': 'presence'}

        # A prediction-kind call is metered automatically.
        assert _poll(factory, lambda s: s.query(ApiUsage).count() == 1)
        with factory() as s:
            row = s.query(ApiUsage).first()
            assert row.kind == 'prediction'
            assert row.source == 'portal'
            assert row.device_id == 'dev-1'


def test_ws_event_frames_and_room_cache(api):
    client, factory, _user = api
    room_doc = {'format': 'room/v1', 'room_id': 'living-room',
                'name': 'Living Room', 'dims': {'w': 5, 'd': 4, 'h': 2.6},
                'devices': []}
    with client.websocket_connect(_ws_url()) as ws:
        ws.send_json({'type': 'event', 'kind': 'trigger_fired',
                      'data': {'automation': 'light-on'}, 'id': 'evt-1'})
        ws.send_json({'type': 'room_changed', 'data': room_doc})
        assert _poll(
            factory, lambda s: s.query(NodeEvent)
            .filter(NodeEvent.kind == 'trigger_fired').count() == 1)
        assert _poll(factory, lambda s: s.query(NodeRoom).count() == 1)

    res = client.get('/v1/events?device_id=dev-1', headers=_user_headers())
    kinds = [e['kind'] for e in res.json()['events']]
    assert 'trigger_fired' in kinds and 'room_changed' in kinds

    res = client.get('/v1/nodes/dev-1/room', headers=_user_headers())
    assert res.status_code == 200
    assert res.json()['room']['room_id'] == 'living-room'

    # disconnect unregisters the tunnel (DB online flag is best-effort
    # bookkeeping; the 90s heartbeat staleness is the real offline signal)
    assert _poll(factory, lambda s: not node_ws.manager.online('dev-1'))


def test_relay_offline_node_returns_503(api):
    client, _factory, _user = api
    res = client.post('/v1/nodes/dev-1/api',
                      json={'method': 'GET', 'path': '/api/status'},
                      headers=_user_headers())
    assert res.status_code == 503


def test_room_put_writes_through(api):
    client, factory, _user = api
    result = {}
    room = {'format': 'room/v1', 'room_id': 'office',
            'dims': {'w': 4, 'd': 3, 'h': 2.5}, 'devices': []}

    def do_put():
        result['resp'] = client.put('/v1/nodes/dev-1/room', json=room,
                                    headers=_user_headers())

    with client.websocket_connect(_ws_url()) as ws:
        t = threading.Thread(target=do_put, daemon=True)
        t.start()
        req = ws.receive_json()
        assert req['method'] == 'PUT' and req['path'] == '/api/v1/room'
        ws.send_json({'type': 'api_response', 'id': req['id'],
                      'status': 200, 'body': req['body']})
        t.join(10)
        assert result['resp'].status_code == 200
        assert result['resp'].json()['room']['room_id'] == 'office'
        assert _poll(factory, lambda s: s.query(NodeRoom).count() == 1)


def test_events_and_usage_rest_fallback(api):
    client, factory, _user = api
    device_auth = {'Authorization': f'Bearer {_device_token()}'}

    res = client.post('/v1/events',
                      json={'kind': 'trigger_fired',
                            'data': {'automation': 'a1'}, 'event_id': 'e-1'},
                      headers=device_auth)
    assert res.status_code == 201
    # Idempotent replay: same external_id → single row.
    res = client.post('/v1/events',
                      json={'kind': 'trigger_fired',
                            'data': {'automation': 'a1'}, 'event_id': 'e-1'},
                      headers=device_auth)
    assert res.status_code == 201
    with factory() as s:
        assert s.query(NodeEvent).count() == 1

    res = client.post('/v1/usage',
                      json={'kind': 'capture', 'source': 'dashboard',
                            'latency_ms': 12.5},
                      headers=device_auth)
    assert res.status_code == 201

    res = client.get('/v1/usage?device_id=dev-1', headers=_user_headers())
    assert res.status_code == 200
    rows = res.json()['usage']
    assert len(rows) == 1 and rows[0]['kind'] == 'capture'

    res = client.get('/v1/events?device_id=dev-1', headers=_user_headers())
    assert res.status_code == 200
    assert res.json()['events'][0]['kind'] == 'trigger_fired'

    # Foreign device is isolated.
    res = client.get('/v1/events?device_id=other-device',
                     headers=_user_headers())
    assert res.status_code == 404
