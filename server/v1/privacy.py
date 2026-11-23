"""Privacy settings, export and deletion (NFR-PRV-01/03).

    GET    /v1/privacy/settings
    PUT    /v1/privacy/settings     {raw_upload?, cloud_processing?, cloud_backup?, analytics?}
    GET    /v1/privacy/export       everything Brain stores for the caller (JSON)
    DELETE /v1/privacy/data         {"confirm": "DELETE"} — erase the caller's cloud data
"""

from __future__ import annotations

import base64
from datetime import date, datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import LargeBinary, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from server import privacy
from server.auth import get_current_user
from server.db import Base, User, get_db

router = APIRouter(prefix="/privacy", tags=["v1", "privacy"])

# Kept on deletion: financial/legal records and the account row itself
# (the account can be closed separately). Stripe remains the billing system.
_RETAINED_TABLES = {"user_account", "payment"}
_SECRET_COLUMNS = {"hashed_password", "key_hash", "token_hash", "secret", "pairing_secret"}


class PrivacyIn(BaseModel):
    raw_upload: bool | None = None
    cloud_processing: bool | None = None
    cloud_backup: bool | None = None
    analytics: bool | None = None


@router.get("/settings")
async def get_settings(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return privacy.settings(current_user)


@router.put("/settings")
async def put_settings(
    body: PrivacyIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    out = privacy.update(current_user, body.model_dump(exclude_none=True))
    db.commit()
    return out


def _user_tables() -> List[Any]:
    """Mapped tables owning per-user rows (``user_id`` column), FK-safe order."""
    return [t for t in Base.metadata.sorted_tables
            if "user_id" in t.c and t.name not in _RETAINED_TABLES]


def _jsonable(value: Any, column: Any) -> Any:
    if value is None:
        return None
    if isinstance(column.type, LargeBinary) or isinstance(value, (bytes, bytearray)):
        return {"bytes": len(value), "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


@router.get("/export")
async def export_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    uid = current_user.userId
    account = {c.name: _jsonable(getattr(current_user, c.key, None), c)
               for c in User.__table__.c if c.name not in _SECRET_COLUMNS}
    tables: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[str] = []
    for table in _user_tables() + [Base.metadata.tables["payment"]]:
        try:
            with db.begin_nested():
                rows = db.execute(select(table).where(table.c.user_id == uid)).mappings().all()
        except SQLAlchemyError:
            skipped.append(table.name)
            continue
        if rows:
            tables[table.name] = [
                {k: _jsonable(v, table.c[k]) for k, v in row.items() if k not in _SECRET_COLUMNS}
                for row in rows]
    return {"format": "thothcraft.brain-export/v1", "exported_at": datetime.utcnow().isoformat(),
            "account": account, "privacy": privacy.settings(current_user), "tables": tables,
            "unavailable_tables": skipped,
            "note": "Raw audio, transcripts and sensor data live on your Room1; export them there."}


class DeleteIn(BaseModel):
    confirm: str


@router.delete("/data")
async def delete_data(
    body: DeleteIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    if body.confirm != "DELETE":
        raise HTTPException(status_code=400, detail='Send {"confirm": "DELETE"} to erase your cloud data')
    uid = current_user.userId
    deleted: Dict[str, int] = {}
    failed: List[str] = []
    # Children before parents; a pass may unblock a parent whose rows are
    # referenced by another user table, so retry failures once.
    pending = list(reversed(_user_tables()))
    for _attempt in range(2):
        still: List[Any] = []
        for table in pending:
            try:
                with db.begin_nested():
                    result = db.execute(delete(table).where(table.c.user_id == uid))
            except SQLAlchemyError:
                still.append(table)
                continue
            if result.rowcount:
                deleted[table.name] = deleted.get(table.name, 0) + int(result.rowcount)
        pending = still
    failed = [t.name for t in pending]
    db.commit()
    return {"deleted": deleted, "failed": failed, "retained": sorted(_RETAINED_TABLES),
            "note": "Account and payment records are retained; Room1 data is deleted on the device."}
