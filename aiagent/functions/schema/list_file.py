from typing import Dict, Any, List, Optional
from server.utils.functions_metadata import function_schema

try:
    from server.db import SessionLocal, File as DBFile  # type: ignore
    from server.utils.logging_utils import logger
except Exception:  # pragma: no cover
    SessionLocal = None  # type: ignore
    DBFile = None  # type: ignore
    from logging import getLogger
    logger = getLogger(__name__)


def _coerce_user_id(uid) -> int:
    """Coerce a user identifier to an integer.
    - Accepts int directly.
    - Converts pure digit strings (e.g., "1234").
    - Extracts trailing digits from mixed strings (e.g., "user-1234" -> 1234).
    """
    if isinstance(uid, int):
        return uid
    if isinstance(uid, str):
        s = uid.strip()
        if s.isdigit():
            return int(s)
        import re
        m = re.search(r"(\d+)$", s)
        if m:
            coerced = int(m.group(1))
            logger.info(f"[list_file] Coerced string user_id='{uid}' to integer {coerced}")
            return coerced
    raise ValueError("user_id must be an integer or contain a trailing numeric ID (e.g., 'user-1234')")


@function_schema(
    name="list_file",
    description="List files for a user directly from the database.",
    required_params=["user_id"],
    optional_params=[]
)
def list_file(user_id: int) -> Dict[str, Any]:
    """List files for the given user from the DB, including timestamps.

    Args:
        user_id: The user's ID (int or coercible string like 'user-1234').

    Returns:
        Dict with keys: status, files (list), count.
    """
    user_id_int = _coerce_user_id(user_id)

    if not (SessionLocal and DBFile):
        return {"status": "error", "message": "Database layer unavailable; cannot list files."}

    db = SessionLocal()
    try:
        records: List[DBFile] = db.query(DBFile).filter(DBFile.userId == user_id_int).all()
        files = [
            {
                "fileId": r.fileId,
                "filename": r.filename,
                "size": r.size,
                "uploaded_at": r.uploaded_at.isoformat() if getattr(r, 'uploaded_at', None) else None,
                "file_hash": getattr(r, 'file_hash', None),
                "last_modified": r.last_modified.isoformat() if getattr(r, 'last_modified', None) else None,
            }
            for r in records
        ]
        return {"status": "success", "files": files, "count": len(files)}
    finally:
        db.close()
