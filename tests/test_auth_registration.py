import asyncio
from unittest import mock

from fastapi import HTTPException

from server.endpoints.auth_endpoints import RegisterRequest, register_user, registration_status


class _Query:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None


class _Database:
    def __init__(self):
        self.user = None

    def query(self, model):
        return _Query()

    def add(self, user):
        self.user = user

    def commit(self):
        self.user.userId = 42

    def rollback(self):
        pass

    def refresh(self, user):
        pass


class _Response:
    status_code = 402

    def json(self):
        return {"code": "overdue_payment", "message": "project is restricted"}


def test_service_restriction_falls_back_to_username_registration():
    database = _Database()
    payload = RegisterRequest(username="new-user", email="new@example.com", password="secret1")
    environment = {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_ANON_KEY": "public-key",
        "ALLOW_LOCAL_REGISTRATION_FALLBACK": "true",
    }

    with mock.patch.dict("os.environ", environment, clear=True), \
         mock.patch("server.endpoints.auth_endpoints.requests.post", return_value=_Response()), \
         mock.patch("server.auth.get_password_hash", return_value="hashed"):
        result = asyncio.run(register_user(payload, database))

    assert result["success"] is True
    assert result["verification_required"] is False
    assert result["email_verification_available"] is False
    assert "sign in with your username" in result["message"]
    assert database.user.supabase_auth_user_id is None
    assert database.user.email_verified is False


def test_service_restriction_is_503_when_fallback_is_disabled():
    payload = RegisterRequest(username="new-user", email="new@example.com", password="secret1")
    environment = {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_ANON_KEY": "public-key",
        "ALLOW_LOCAL_REGISTRATION_FALLBACK": "false",
    }

    with mock.patch.dict("os.environ", environment, clear=True), \
         mock.patch("server.endpoints.auth_endpoints.requests.post", return_value=_Response()):
        try:
            asyncio.run(register_user(payload, _Database()))
        except HTTPException as exc:
            assert exc.status_code == 503
        else:
            raise AssertionError("Expected registration to reject a restricted provider")


def test_registration_status_advertises_local_fallback():
    with mock.patch.dict("os.environ", {"ALLOW_LOCAL_REGISTRATION_FALLBACK": "true"}, clear=True):
        result = asyncio.run(registration_status())

    assert result["account_registration_available"] is True
    assert result["email_registration_configured"] is False
    assert result["local_registration_fallback_enabled"] is True
