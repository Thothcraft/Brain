"""Desktop sign-in broker: authorize → callback → token (PKCE)."""
import base64
import hashlib
import json
import os
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.v1.auth_oauth import router


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "test-secret")
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return TestClient(app)


AUTH_PARAMS = {
    "provider": "google",
    "redirect_uri": "http://127.0.0.1:9999/callback",
    "state": "flow123|nonce456",
    "code_challenge": "A" * 50,
    "code_challenge_method": "S256",
}


def _unsign(blob):
    body, sig = blob.rsplit(".", 1)
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def test_authorize_redirects_to_google(client):
    r = client.get("/v1/auth/authorize", params=AUTH_PARAMS, follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=test-client-id" in loc
    flow = _unsign(loc.split("state=", 1)[1].split("&", 1)[0].replace("%2F", "/").replace("%3D", "="))
    assert flow["r"] == AUTH_PARAMS["redirect_uri"]
    assert flow["s"] == AUTH_PARAMS["state"]
    assert flow["c"] == AUTH_PARAMS["code_challenge"]


def test_authorize_rejects_bad_inputs(client):
    for override, code in (
        ({"provider": "apple"}, 400),
        ({"redirect_uri": "https://evil.example.com/cb"}, 400),
        ({"redirect_uri": "http://10.0.0.5/cb"}, 400),
        ({"code_challenge": "short"}, 400),
        ({"code_challenge_method": "plain"}, 400),
    ):
        p = dict(AUTH_PARAMS, **override)
        r = client.get("/v1/auth/authorize", params=p, follow_redirects=False)
        assert r.status_code == code, override


def test_authorize_503_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID")
    r = client.get("/v1/auth/authorize", params=AUTH_PARAMS, follow_redirects=False)
    assert r.status_code == 503


def _callback_state(client):
    r = client.get("/v1/auth/authorize", params=AUTH_PARAMS, follow_redirects=False)
    loc = r.headers["location"]
    import urllib.parse
    return urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)["state"][0]


def test_callback_no_code(client):
    state = _callback_state(client)
    r = client.get("/v1/auth/callback", params={"state": state})
    assert "didn&#x27;t complete" in r.text or "didn't complete" in r.text


def test_callback_bad_state_404_on_alias(client):
    r = client.get("/v1/integrations/google/callback",
                   params={"state": "garbage", "code": "x"})
    assert r.status_code == 404


def test_full_flow_with_mocked_google(client):
    state = _callback_state(client)
    google_id_token = "header." + base64.urlsafe_b64encode(
        json.dumps({"email": "u@x.com"}).encode()).decode() + ".sig"
    with mock.patch("server.v1.auth_oauth.requests.post") as post, \
         mock.patch("google.oauth2.id_token.verify_oauth2_token") as verify:
        post.return_value = mock.Mock(status_code=200, json=lambda: {"id_token": google_id_token})
        r = client.get("/v1/auth/callback", params={"state": state, "code": "gcode"})
        assert r.status_code == 200
        verify.assert_called_once()
    # The page navigates the browser to the loopback with ?code&state.
    assert "127.0.0.1:9999/callback" in r.text

    import urllib.parse, re
    m = re.search(r'window\.location\.replace\("(.*?)"\)', r.text)
    target = m.group(1)
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(target).query)
    auth_code = qs["code"][0]
    assert qs["state"][0] == AUTH_PARAMS["state"]

    # Token exchange with the right verifier succeeds.
    verifier = "v" * 60
    grant = _unsign(auth_code)
    # The stored challenge is AUTH_PARAMS' fixed challenge; recompute the
    # verifier binding by signing check — use the real digest relation:
    digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    # Force challenge to match verifier for this exchange test.
    with mock.patch.dict(os.environ, {}):
        pass
    # Re-run authorize with a challenge derived from the verifier.
    params = dict(AUTH_PARAMS, code_challenge=digest)
    r = client.get("/v1/auth/authorize", params=params, follow_redirects=False)
    loc = r.headers["location"]
    state2 = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)["state"][0]
    with mock.patch("server.v1.auth_oauth.requests.post") as post, \
         mock.patch("google.oauth2.id_token.verify_oauth2_token"):
        post.return_value = mock.Mock(status_code=200, json=lambda: {"id_token": google_id_token})
        r = client.get("/v1/auth/callback", params={"state": state2, "code": "gcode"})
    m = re.search(r'window\.location\.replace\("(.*?)"\)', r.text)
    auth_code = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(1)).query)["code"][0]

    r = client.post("/v1/auth/token", data={
        "grant_type": "authorization_code", "code": auth_code,
        "redirect_uri": params["redirect_uri"], "code_verifier": verifier})
    assert r.status_code == 200
    body = r.json()
    assert body["id_token"] == google_id_token

    # Wrong verifier → invalid grant.
    r = client.post("/v1/auth/token", data={
        "grant_type": "authorization_code", "code": auth_code,
        "redirect_uri": params["redirect_uri"], "code_verifier": "x" * 60})
    assert r.status_code == 400
