"""Audit event recording for security-relevant actions."""

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from server.db import AuditEvent

logger = logging.getLogger(__name__)


def audit(
    db: Session,
    action: str,
    *,
    user_id: Optional[int] = None,
    device_id: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    commit: bool = True,
) -> None:
    """Append an audit event. Never raises into the request path."""
    try:
        db.add(AuditEvent(
            user_id=user_id,
            device_id=device_id,
            action=action,
            detail=json.dumps(detail) if detail else None,
            ip_address=ip_address,
        ))
        if commit:
            db.commit()
    except Exception:
        logger.exception("Failed to record audit event %s", action)
        try:
            db.rollback()
        except Exception:
            pass
