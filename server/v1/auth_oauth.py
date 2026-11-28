"""Desktop sign-in broker — the omi-contract endpoints the desktop app uses.

The desktop (Electron) runs a system-browser OAuth flow:

1. ``GET /v1/auth/authorize`` — validates the loopback redirect + PKCE
   challenge, then 302s to Google's consent screen. Brain's own callback
   URI is the one registered on the thothcraft OAuth client.
2. ``GET /v1/auth/callback`` (and the already-registered alias
   ``GET /v1/integrations/google/callback``) — exchanges the Google code
   server-side (client_secret stays on Brain), verifies the Google
   ``id_token``, then serves an HTML page that navigates the browser to
   the desktop's loopback ``{redirect_uri}?code=<auth code>&state=…``.
   The auth code is a short-lived HMAC-signed blob carrying the Google
   id_token and the flow's PKCE challenge — self-contained, no storage.
3. ``POST /v1/auth/token`` (form) — verifies the auth code signature and
   the PKCE verifier, then returns ``{id_token, access_token, …}``.
   The renderer completes Firebase sign-in client-side via
   ``GoogleAuthProvider.credential(id_token)``.

Required env: ``GOOGLE_OAUTH_CLIENT_ID`` + ``GOOGLE_OAUTH_CLIENT_SECRET``
(the thothcraft web OAuth client). ``GOOGLE_OAUTH_REDIRECT_URI`` defaults
to the registered ``/v1/integrations/google/callback`` path.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import requests
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from server.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

GOOGLE_AUTHORIZE = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
DEFAULT_CALLBACK = "https://api.thothcraft.com/v1/integrations/google/callback"

AUTH_STATE_TTL_S = 600    # authorize → callback window
AUTH_CODE_TTL_S = 120     # callback → token-exchange window

_PKCE_RE = re.compile(r"^[A-Za-z0-9\-._~]{43,128}$")
_LOOPBACK_RE = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d+)?/")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1].strip()
    return v


def _client_id() -> str:
    return _unquote(os.getenv("GOOGLE_OAUTH_CLIENT_ID") or "")


def _client_secret() -> str:
    return _unquote(os.getenv("GOOGLE_OAUTH_CLIENT_SECRET") or "")


def _callback_uri() -> str:
    return (os.getenv("GOOGLE_OAUTH_REDIRECT_URI") or DEFAULT_CALLBACK).strip()


def _sign(payload: Dict[str, Any]) -> str:
    """b64(json).b64(hmac) — self-contained signed blob for OAuth state/code."""
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(sig)}"


def _unsign(blob: str) -> Optional[Dict[str, Any]]:
    try:
        body, sig = blob.rsplit(".", 1)
        expected = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _unb64(sig)):
            return None
        return json.loads(_unb64(body))
    except Exception:
        return None


def _callback_page(target: str, title: str, body: str) -> str:
    """HTML handed to the browser after the Brain-side callback. JS navigates
    to the desktop's loopback listener; the text is the no-JS fallback."""
    safe_t, safe_b = (x.replace("&", "&amp;").replace("<", "&lt;") for x in (title, body))
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>thothcraft</title></head>"
        "<body style=\"margin:0;display:flex;align-items:center;justify-content:center;"
        "min-height:100vh;background:#121212;color:#fff;"
        "font-family:'Segoe UI',system-ui,sans-serif\">"
        "<div style=\"text-align:center;max-width:420px;padding:2rem\">"
        "<div style=\"font-size:28px;font-weight:600;letter-spacing:-0.02em;"
        "margin-bottom:16px\">thothcraft</div>"
        f"<div style=\"font-size:17px;margin-bottom:8px\">{safe_t}</div>"
        f"<div style=\"font-size:14px;color:rgba(255,255,255,0.6)\">{safe_b}</div>"
        "</div>"
        f"<script>window.location.replace({json.dumps(target)})</script>"
        "</body></html>"
    )


def _error_response(message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=400)


@router.get("/auth/authorize")
def authorize(
    provider: str = Query(...),
    redirect_uri: str = Query(...),
    state: str = Query(...),
    code_challenge: str = Query(...),
    code_challenge_method: str = Query("S256"),
):
    """Desktop → Brain → Google. Validates the flow then 302s to consent."""
    if provider != "google":
        raise HTTPException(400, "provider not supported")
    if not _LOOPBACK_RE.match(redirect_uri or ""):
        raise HTTPException(400, "redirect_uri must be a loopback http URL")
    if not (0 < len(state) <= 512):
        raise HTTPException(400, "invalid state")
    if code_challenge_method != "S256" or not _PKCE_RE.match(code_challenge or ""):
        raise HTTPException(400, "invalid code_challenge")
    if not _client_id():
        raise HTTPException(503, "sign-in is not configured on this server")

    flow = {
        "r": redirect_uri,
        "s": state,
        "c": code_challenge,
        "n": secrets.token_urlsafe(8),
        "exp": int(time.time()) + AUTH_STATE_TTL_S,
    }
    params = {
        "client_id": _client_id(),
        "redirect_uri": _callback_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": _sign(flow),
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTHORIZE}?{urlencode(params)}", status_code=302)


def _finish_callback(code: str, flow: Dict[str, Any]) -> str:
    """Exchange the Google code → verify id_token → mint the desktop-facing
    auth code. Returns the loopback target URL the browser should open."""
    try:
        resp = requests.post(GOOGLE_TOKEN, data={
            "code": code,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "redirect_uri": _callback_uri(),
            "grant_type": "authorization_code",
        }, timeout=15)
    except Exception as exc:
        raise HTTPException(502, f"provider exchange failed: {type(exc).__name__}")
    if resp.status_code != 200:
        logger.info(
            "[auth] google token exchange rejected: %s %s",
            resp.status_code, (resp.text or "")[:300])
        raise HTTPException(400, "provider rejected the authorization code")
    google_id_token = str((resp.json() or {}).get("id_token") or "")
    if not google_id_token:
        raise HTTPException(502, "provider returned no id_token")
    try:
        from google.auth.transport import requests as ga_requests
        from google.oauth2 import id_token as ga_id_token
        ga_id_token.verify_oauth2_token(
            google_id_token, ga_requests.Request(),
            audience=_client_id(), clock_skew_in_seconds=10)
    except Exception as exc:
        logger.info("[auth] google id_token rejected: %s", type(exc).__name__)
        raise HTTPException(400, "provider id_token failed verification")

    auth_code = _sign({
        "idt": google_id_token,
        "c": flow["c"],
        "n": secrets.token_urlsafe(8),
        "exp": int(time.time()) + AUTH_CODE_TTL_S,
    })
    sep = "&" if urlparse(flow["r"]).query else "?"
    return f"{flow['r']}{sep}code={auth_code}&state={flow['s']}"


def _callback_impl(code: Optional[str], state: Optional[str], error: Optional[str]) -> HTMLResponse:
    flow = _unsign(state or "") if state else None
    if flow is None or int(flow.get("exp") or 0) < time.time():
        return HTMLResponse(_callback_page(
            "about:blank", "Sign-in didn't complete", "The sign-in link expired. Close this tab and retry."))
    if error:
        target = f"{flow['r']}?error={error}&state={flow['s']}"
        return HTMLResponse(_callback_page(target, "Authorization failed", str(error)))
    if not code:
        return HTMLResponse(_callback_page(
            "about:blank", "Sign-in didn't complete", "No authorization code was returned."))
    try:
        target = _finish_callback(code, flow)
    except HTTPException as exc:
        target = f"{flow['r']}?error={exc.detail}&state={flow['s']}"
        return HTMLResponse(_callback_page(target, "Sign-in didn't complete", str(exc.detail)))
    return HTMLResponse(_callback_page(target, "You're signed in", "Return to thothcraft — you can close this tab."))


@router.get("/auth/callback", include_in_schema=False)
def auth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    return _callback_impl(code, state, error)


@router.get("/integrations/google/callback", include_in_schema=False)
def integrations_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """Alias on the URI already registered with the OAuth client. Auth-flow
    states are signed blobs; anything else isn't ours → 404."""
    flow = _unsign(state or "") if state else None
    if flow is None:
        raise HTTPException(404, "not found")
    return _callback_impl(code, state, error)


@router.post("/auth/token")
def token_exchange(
    grant_type: str = Form(...),
    code: str = Form(...),
    redirect_uri: str = Form(""),
    code_verifier: str = Form(""),
    use_custom_token: str = Form(""),
):
    """Verify the auth code + PKCE verifier, hand back the Google id_token.
    The desktop completes Firebase sign-in client-side via
    ``GoogleAuthProvider.credential(id_token)``."""
    if grant_type != "authorization_code":
        raise HTTPException(400, "unsupported grant_type")
    grant = _unsign(code)
    if grant is None or int(grant.get("exp") or 0) < time.time():
        raise HTTPException(400, "invalid or expired code")
    digest = _b64(hashlib.sha256(code_verifier.encode()).digest())
    if not hmac.compare_digest(digest, str(grant.get("c") or "")):
        raise HTTPException(400, "PKCE verification failed")
    google_id_token = str(grant.get("idt") or "")
    if not google_id_token:
        raise HTTPException(400, "grant carries no id_token")
    return {
        "access_token": google_id_token,
        "id_token": google_id_token,
        "token_type": "Bearer",
        "expires_in": AUTH_CODE_TTL_S,
    }
