"""App home-tab surface: conversations/memories/action items/language."""
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from server.auth import get_current_user
from server.db import Base, User, get_db
from server.v1 import router as v1_router
from server.v1.app_compat import root_router


@pytest.fixture
def client():
    tmp = Path(tempfile.mkdtemp()) / 'app.db'
    engine = create_engine(f'sqlite:///{tmp}', poolclass=NullPool)
    Base.metadata.create_all(engine)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE app_doc (user_id INTEGER NOT NULL, kind VARCHAR(32) NOT NULL, "
                       "doc_id VARCHAR(64) NOT NULL, body TEXT NOT NULL, "
                       "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                       "PRIMARY KEY (user_id, kind, doc_id))"))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        s.add_all([User(userId=1, username='a', hashed_password='x', role=0, plan='free'),
                   User(userId=2, username='b', hashed_password='x', role=0, plan='free')])
        s.commit()
    state = {'uid': 1}
    app = FastAPI()
    app.include_router(v1_router, prefix='/v1')
    app.include_router(root_router)

    def _db():
        with factory() as s:
            yield s

    def _user():
        with factory() as s:
            return s.get(User, state['uid'])

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = _user
    c = TestClient(app)
    c.state = state
    return c


def test_empty_home_tabs_have_app_shapes(client):
    assert client.get('/v1/conversations?limit=50&offset=0&statuses=').json() == []
    assert client.get('/v3/memories').json() == []
    assert client.get('/v1/folders').json() == []
    assert client.get('/v1/users/people').json() == []
    assert client.get('/v1/action-items').json() == {'action_items': [], 'has_more': False}


def test_conversation_roundtrip_is_per_user(client):
    created = client.post('/v1/conversations', json={'source': 'room1'}).json()
    cid = created['id']
    assert client.get(f'/v1/conversations/{cid}').json()['source'] == 'room1'
    assert [c['id'] for c in client.get('/v1/conversations').json()] == [cid]
    client.state['uid'] = 2
    assert client.get('/v1/conversations').json() == []
    assert client.get(f'/v1/conversations/{cid}').status_code == 404
    client.state['uid'] = 1
    client.delete(f'/v1/conversations/{cid}')
    assert client.get('/v1/conversations').json() == []


def test_language_set_and_read(client):
    assert client.get('/v1/users/language').json() == {'language': None}
    r = client.patch('/v1/users/language', json={'language': 'en'})
    assert r.json() == {'status': 'ok', 'single_language_mode': False}
    assert client.get('/v1/users/language').json() == {'language': 'en'}
    assert client.patch('/v1/users/language', json={'language': 'xx'}).status_code == 422
