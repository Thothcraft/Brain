"""Isolated API tests: real routers and ORM, in-memory SQLite, no server startup."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import Base, User, get_db
from server.endpoints.file_endpoints import router as files_router
from server.endpoints.labs_endpoints import router as labs_router


@pytest.fixture
def api():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        user = User(userId=1, username='test', email='test@example.invalid',
                    hashed_password='unused', role=0, plan='free')
        session.add(user)
        session.commit()
        app = FastAPI()
        app.include_router(files_router, prefix='/api')
        app.include_router(labs_router, prefix='/api')
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, user
    engine.dispose()
