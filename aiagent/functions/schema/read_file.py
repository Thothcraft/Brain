from server.utils.functions_metadata import function_schema

# Optional heavy imports guarded to prevent circular/import errors in limited environments
try:
    from server.db import SessionLocal, File as DBFile
except Exception:  # pragma: no cover
    SessionLocal = None  # type: ignore
    DBFile = None  # type: ignore


@function_schema(
    name="read_file",
    description="Read the content of a user file stored in the database and return it as text.",
    required_params=["filename", "user_id"],
    optional_params=[]
)
def read_file(filename: str, user_id: int):
    """Retrieve the content of a file belonging to the specified user.

    :param filename: Logical filename (e.g. `notes.md`).
    :param user_id: Owning user's database ID.
    :returns: A dictionary containing the file metadata and its UTF-8 decoded content.
    """

    if not (SessionLocal and DBFile):
        raise RuntimeError("Database layer unavailable; cannot read files.")


    db = SessionLocal()
    try:
        record = db.query(DBFile).filter(
            DBFile.userId == user_id,
            DBFile.filename == filename
        ).first()
        if not record:
            return {
                "status": "error",
                "message": f"File '{filename}' not found for user {user_id}."
            }

        try:
            content_text = (record.content or b"").decode("utf-8", errors="ignore")
        except Exception:
            # binary / undecodable file
            content_text = "[BINARY OR UNDECODABLE CONTENT]"

        return {
            "status": "success",
            "fileId": record.fileId,
            "filename": filename,
            "size": record.size,
            "content": content_text,
            "content_type": record.content_type,
        }
    finally:
        db.close()
