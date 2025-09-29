from typing import Dict, Any
from server.utils.functions_metadata import function_schema

from datetime import datetime

try:
    from server.db import SessionLocal, File as DBFile, DATABASE_URL  # type: ignore
    from server.utils import compute_sha256
    from server.utils.logging_utils import logger
except Exception:  # pragma: no cover
    SessionLocal = None  # type: ignore
    DBFile = None  # type: ignore
    compute_sha256 = None  # type: ignore
    DATABASE_URL = None  # type: ignore
    from logging import getLogger
    logger = getLogger(__name__)


@function_schema(
    name="write_file",
    description="Create or update a user file, storing its content on the database.",
    required_params=["filename", "content", "user_id"],
    optional_params=["mode"]
)
def write_file(filename: str, content: str, user_id: int, mode: str = "overwrite") -> Dict[str, Any]:
    """Write content to a file, create if it doesn't exist.

    Args:
        filename (str): Logical filename to create or update for the given user.
        content (str): The content to write.
        mode (str, optional): Either "overwrite" (default) or "append". Determines the write behaviour.
        user_id (int): The owning user's ID.

    Returns:
        dict: Details about the written file.
    """

    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be either 'overwrite' or 'append'")

    # Normalize user_id to int (handle cases like "user-1234")
    def _coerce_user_id(uid) -> int:
        """Attempt to coerce uid to int.
        - If uid is an int, return as-is.
        - If uid is a numeric string, convert directly.
        - If uid is a mixed string like 'user-1234', extract the last digit sequence.
        - Otherwise, raise ValueError with guidance.
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
                logger.info(f"[write_file] Coerced string user_id='{uid}' to integer {coerced}")
                return coerced
        raise ValueError("user_id must be an integer or contain a trailing numeric ID (e.g., 'user-1234')")

    user_id_int = _coerce_user_id(user_id)

    bytes_content = content.encode("utf-8")
    bytes_written = len(bytes_content)

    if not (SessionLocal and DBFile):
        raise RuntimeError("Database layer unavailable; cannot store files.")

    logger.info(f"[write_file] Begin - user_id={user_id_int}, filename={filename}, mode={mode}, bytes={bytes_written}")
    if DATABASE_URL:
        logger.info(f"[write_file] Using DATABASE_URL: {DATABASE_URL}")

    db = SessionLocal()
    try:
        record = db.query(DBFile).filter(
            DBFile.userId == user_id_int,
            DBFile.filename == filename
        ).first()
        now = datetime.utcnow()

        if record:
            logger.info(f"[write_file] Existing record found: fileId={record.fileId}, size={record.size}")
            if mode == "append":
                combined = (record.content or b"") + bytes_content
                record.content = combined
                record.size = len(combined)
            else:  # overwrite
                record.content = bytes_content
                record.size = bytes_written
            record.file_hash = compute_sha256(record.content) if compute_sha256 else None
            record.last_modified = now
        else:
            logger.info("[write_file] No existing record; creating new one")
            record = DBFile(
                userId=user_id_int,
                filename=filename,
                size=bytes_written,
                content=bytes_content,
                content_type="text/plain",
                uploaded_at=now,
                file_hash=compute_sha256(bytes_content) if compute_sha256 else None,
            )
            db.add(record)
        try:
            db.commit()
            db.refresh(record)
            verify_count = db.query(DBFile).filter(DBFile.userId == user_id_int, DBFile.filename == filename).count()
            logger.info(f"[write_file] Commit OK: fileId={record.fileId}, total_size={record.size}, verify_count={verify_count}")
        except Exception as e:
            db.rollback()
            logger.error(f"[write_file] Commit failed: {str(e)}", exc_info=True)
            raise
    finally:
        db.close()

    return {
        "status": "success",
        "fileId": record.fileId,
        "filename": filename,
        "mode": mode,
        "bytes_written": bytes_written,
        "total_size": record.size,
    }