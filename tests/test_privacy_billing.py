"""Privacy-first Brain: Firebase identity, omi-style plans/usage, privacy
gates, E2E relay (thothcraft-omi docs, D-01/D-02/D-09)."""
import json
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from server import firebase_auth, plans
from server.auth import create_access_token, get_current_user
from server.auth import get_db as auth_get_db
from server.db import Base, Device, UsageCounter, User, get_db
from server.endpoints import node_ws
from server.v1 import router as v1_router


@pytest.fixture
def env(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / 'pb.db'
    engine = create_engine(f'sqlite:///{tmp}', connect_args={'check_same_thread': False, 'timeout': 30},
                           poolclass=NullPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        s.add_all([
            User(userId=1, username='legacy', email='legacy@example.invalid', email_verified=True,
                 hashed_password='unused', role=0, plan='free'),
            User(userId=2, username='strict', email='strict@example.invalid', email_verified=True,
                 hashed_password='unused', role=0, plan='free',
                 privacy_json=json.dumps(firebase_auth.default_privacy())),
            Device(userId=1, device_uuid='room1-a', device_name='room1', device_type='thoth',
                   approved=True, online=False),
        ])
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
    from server import event_delivery
    monkeypatch.setattr(event_delivery, 'get_db_session', _session_ctx)
    monkeypatch.setenv('STRIPE_PRICE_ID_PRO_MONTHLY', 'price_pro_m')
    monkeypatch.setenv('STRIPE_PRICE_ID_PRO_ANNUAL', 'price_pro_y')
    monkeypatch.setenv('STRIPE_PRICE_ID_HOME_MONTHLY', 'price_home_m')

    app = FastAPI()
    app.include_router(v1_router, prefix='/v1')

    def _get_db():
        s = factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[auth_get_db] = _get_db
    with TestClient(app) as client:
        yield client, factory


def _hdr(uid=1, username='legacy'):
    return {'Authorization': f'Bearer {create_access_token(data={"sub": str(uid), "username": username})}'}


def _dev_hdr(device='room1-a'):
    tok = create_access_token(data={'sub': '1', 'username': 'legacy', 'scopes': ['device'],
                                    'device_id': device}, domain='device')
    return {'Authorization': f'Bearer {tok}'}


# ── plans ─────────────────────────────────────────────────────────────────────

def test_available_plans_sell_only_pro(env):
    client, _ = env
    r = client.get('/v1/payments/available-plans', headers=_hdr())
    assert r.status_code == 200
    plans = r.json()['plans']
    assert {p['plan_id'] for p in plans} == {'pro'}
    # App wire contract (GeneratedPaymentPricingOption): one option per price,
    # id/interval/price_string/unit_amount required.
    assert {p['id'] for p in plans} == {'price_pro_m', 'price_pro_y'}
    for p in plans:
        assert p['interval'] in ('month', 'year')
        assert 'price_string' in p and 'unit_amount' in p
        assert p['is_active'] is True


def test_legacy_plan_visible_to_its_subscriber(env):
    client, factory = env
    with factory() as s:
        s.query(User).filter(User.userId == 1).update({'plan': 'home'})
        s.commit()
    ids = {p['plan_id'] for p in
           client.get('/v1/payments/available-plans', headers=_hdr()).json()['plans']}
    assert ids == {'pro', 'home'}


def test_subscription_shape_free_and_pro(env):
    client, factory = env
    body = client.get('/v1/users/me/subscription', headers=_hdr()).json()
    assert body['subscription']['plan'] == 'basic'
    assert body['transcription_seconds_limit'] == 18000
    assert body['chat_usage_quota']['limit'] == 30
    with factory() as s:
        s.query(User).filter(User.userId == 1).update({'plan': 'pro'})
        s.commit()
    body = client.get('/v1/users/me/subscription', headers=_hdr()).json()
    assert body['subscription']['plan'] == 'pro'
    assert body['transcription_seconds_limit'] == 0          # omi "unlimited" sentinel
    assert body['chat_usage_quota']['limit'] == 1000


def test_checkout_rejects_unsellable_price(env, monkeypatch):
    client, _ = env
    from server.endpoints import stripe_endpoints
    monkeypatch.setattr(stripe_endpoints, 'STRIPE_AVAILABLE', True)
    monkeypatch.setenv('STRIPE_SECRET_KEY', 'sk_test_x')
    r = client.post('/v1/payments/checkout-session', json={'price_id': 'price_home_m'}, headers=_hdr())
    assert r.status_code == 400


# ── Room1 usage reports ───────────────────────────────────────────────────────

def test_usage_report_idempotent_and_caps(env):
    client, factory = env
    period = plans.current_period()
    r = client.post('/v1/usage/report', headers=_dev_hdr(),
                    json={'period': period, 'seq': 1, 'transcription_seconds': 600, 'chat_questions': 29})
    assert r.status_code == 200 and r.json()['applied'] is True
    again = client.post('/v1/usage/report', headers=_dev_hdr(),
                        json={'period': period, 'seq': 1, 'transcription_seconds': 600, 'chat_questions': 29})
    assert again.json()['applied'] is False
    ent = client.get('/v1/entitlements/me', headers=_dev_hdr()).json()
    assert ent['allocations']['transcription_seconds']['used'] == 600
    assert ent['allocations']['chat_questions']['allowed'] is True
    client.post('/v1/usage/report', headers=_dev_hdr(),
                json={'period': period, 'seq': 2, 'chat_questions': 1})
    ent = client.get('/v1/entitlements/me', headers=_hdr()).json()
    assert ent['allocations']['chat_questions']['allowed'] is False
    with factory() as s:
        assert s.query(UsageCounter).filter_by(user_id=1, metric='chat_questions').one().value == 30


def test_usage_report_requires_device_credential(env):
    client, _ = env
    r = client.post('/v1/usage/report', headers=_hdr(),
                    json={'period': plans.current_period(), 'seq': 1, 'chat_questions': 1})
    assert r.status_code == 403


# ── Firebase identity ─────────────────────────────────────────────────────────

def test_firebase_token_creates_strict_user_and_links_email(env, monkeypatch):
    client, factory = env
    monkeypatch.setattr(firebase_auth, 'looks_like_firebase_token', lambda t: t.startswith('fb.'))
    claims = {
        'fb.new': {'sub': 'uid-new', 'email': 'new@example.invalid', 'email_verified': True},
        'fb.legacy': {'sub': 'uid-legacy', 'email': 'legacy@example.invalid', 'email_verified': True},
    }
    monkeypatch.setattr(firebase_auth, 'verify_firebase_token', lambda t: claims.get(t))

    r = client.get('/v1/privacy/settings', headers={'Authorization': 'Bearer fb.new'})
    assert r.status_code == 200
    assert r.json()['legacy'] is False and r.json()['raw_upload'] is False

    r = client.get('/v1/users/me/subscription', headers={'Authorization': 'Bearer fb.legacy'})
    assert r.status_code == 200
    with factory() as s:
        assert s.query(User).filter_by(userId=1).one().firebase_uid == 'uid-legacy'
        assert s.query(User).filter_by(firebase_uid='uid-new').count() == 1

    assert client.get('/v1/privacy/settings', headers={'Authorization': 'Bearer fb.bogus'}).status_code == 401


# ── privacy gates, export, delete ─────────────────────────────────────────────

def test_cloud_processing_gate(env):
    client, _ = env
    r = client.post('/v1/chat/tts', json={'text': 'hi'}, headers=_hdr(2, 'strict'))
    assert r.status_code == 403
    assert r.json()['detail']['code'] == 'privacy_consent_required'
    r = client.put('/v1/privacy/settings', json={'cloud_processing': True}, headers=_hdr(2, 'strict'))
    assert r.json()['cloud_processing'] is True and r.json()['raw_upload'] is False


def test_export_and_delete(env):
    client, factory = env
    client.post('/v1/usage/report', headers=_dev_hdr(),
                json={'period': plans.current_period(), 'seq': 9, 'chat_questions': 2})
    exp = client.get('/v1/privacy/export', headers=_hdr()).json()
    assert exp['account']['username'] == 'legacy'
    assert 'hashed_password' not in exp['account']
    assert 'usage_counter' in exp['tables'] and 'device' in exp['tables']
    assert client.request('DELETE', '/v1/privacy/data', json={'confirm': 'no'}, headers=_hdr()).status_code == 400
    r = client.request('DELETE', '/v1/privacy/data', json={'confirm': 'DELETE'}, headers=_hdr())
    assert r.status_code == 200
    with factory() as s:
        assert s.query(UsageCounter).filter_by(user_id=1).count() == 0
        assert s.query(User).filter_by(userId=1).count() == 1


# ── E2E relay ─────────────────────────────────────────────────────────────────

def test_e2e_relay_roundtrip_and_offline(env):
    client, _ = env
    assert client.post('/v1/relay/room1-a', json={'cid': 'c1', 'ct': 'AAAA'}, headers=_hdr()).status_code == 409
    result = {}

    def call():
        result['r'] = client.post('/v1/relay/room1-a', json={'cid': 'c1', 'ct': 'c2VhbGVk'}, headers=_hdr())

    tok = _dev_hdr()['Authorization'][7:]
    with client.websocket_connect(f'/v1/node/ws?device_id=room1-a&token={tok}') as ws:
        t = threading.Thread(target=call, daemon=True)
        t.start()
        frame = ws.receive_json()
        assert frame['type'] == 'relay' and frame['ct'] == 'c2VhbGVk' and frame['cid'] == 'c1'
        ws.send_json({'type': 'relay_response', 'id': frame['id'], 'ct': 'cmVwbHk='})
        t.join(10)
    assert result['r'].status_code == 200 and result['r'].json()['ct'] == 'cmVwbHk='


def test_relay_other_users_device_404(env):
    client, _ = env
    assert client.post('/v1/relay/room1-a', json={'cid': 'c', 'ct': 'x'},
                       headers=_hdr(2, 'strict')).status_code == 404


def test_sealed_notify_requires_device(env):
    client, _ = env
    assert client.post('/v1/notify', json={'ct': 'x'}, headers=_hdr()).status_code == 403
    assert client.post('/v1/notify', json={'ct': 'x'}, headers=_dev_hdr()).status_code == 201
