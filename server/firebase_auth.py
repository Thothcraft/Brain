"""Firebase ID-token verification for the omi-based Thothcraft apps.

The phone/desktop/web apps (forks of omi) sign in with Firebase Auth and
send ``Authorization: Bearer <Firebase ID token>``. Brain verifies the
token against Google's public certificates for ``FIREBASE_PROJECT_ID`` and
links it to a ``user_account`` row:

1. ``firebase_uid`` match, else
2. verified-email match (existing Brain/Supabase account adopts the uid), else
3. a new account is created with privacy-first defaults.

Verification is disabled (returns ``None``) until ``FIREBASE_PROJECT_ID``
is configured, so existing Brain JWT logins are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from server.db import User

logger = logging.getLogger(__name__)

_request = None


def firebase_project_id() -> str:
    return (os.getenv("FIREBASE_PROJECT_ID") or "").strip()


def looks_like_firebase_token(token: str) -> bool:
    """Cheap pre-check: RS256 JWT with a ``kid`` header (Brain JWTs are HS256)."""
    try:
        from jose import jwt as _jwt
        header = _jwt.get_unverified_header(token)
    except Exception:
        return False
    return header.get("alg") == "RS256" and bool(header.get("kid"))


def _google_request():
    """Shared transport so Google's cert response is reused per process."""
    global _request
    if _request is None:
        import requests
        from google.auth.transport import requests as ga_requests
        _request = ga_requests.Request(session=requests.Session())
    return _request


def verify_firebase_token(token: str) -> Optional[Dict[str, Any]]:
    """Return verified claims, or ``None`` when not configured / invalid."""
    project = firebase_project_id()
    if not project or not token or not looks_like_firebase_token(token):
        return None
    try:
        from google.oauth2 import id_token as ga_id_token
        claims = ga_id_token.verify_firebase_token(
            token, _google_request(), audience=project)
    except Exception as exc:
        logger.info("[firebase] token rejected: %s", type(exc).__name__)
        return None
    if not claims or claims.get("iss") != f"https://securetoken.google.com/{project}":
        return None
    if not claims.get("sub"):
        return None
    return claims


def default_privacy() -> Dict[str, bool]:
    """Privacy-first defaults for accounts created through Firebase."""
    return {"cloud_backup": False, "raw_upload": False,
            "cloud_processing": False, "analytics": False}


def _unique_username(db: Session, email: Optional[str], uid: str) -> str:
    base = (email or "").split("@")[0] or "user"
    base = re.sub(r"[^a-zA-Z0-9_.-]", "", base)[:40] or "user"
    candidate = base
    while db.query(User).filter(User.username == candidate).first():
        candidate = f"{base}-{secrets.token_hex(3)}"
    return candidate


def resolve_firebase_user(db: Session, claims: Dict[str, Any]) -> User:
    """Find or create the Brain user for verified Firebase ``claims``."""
    uid = str(claims["sub"])
    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        return user
    email = (claims.get("email") or "").strip() or None
    email_verified = bool(claims.get("email_verified"))
    if email and email_verified:
        user = db.query(User).filter(
            func.lower(User.email) == email.lower()).first()
        if user and not user.firebase_uid:
            user.firebase_uid = uid
            user.email_verified = True
            db.commit()
            logger.info("[firebase] linked uid to existing user %s", user.userId)
            return user
    user = User(
        username=_unique_username(db, email, uid),
        email=email if email_verified else None,
        email_verified=email_verified,
        hashed_password=None,
        role=0,
        plan="free",
        firebase_uid=uid,
        privacy_json=json.dumps(default_privacy()),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("[firebase] created user %s", user.userId)
    return user
