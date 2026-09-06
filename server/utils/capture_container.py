"""Safe readers for Thoth synchronized NPZ capture containers."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np


CONTAINER_SCHEMA = "thoth-capture-npz/v1"


def _blob_at(data: np.ndarray, offsets: np.ndarray, index: int) -> bytes:
    if index < 0 or index + 1 >= len(offsets):
        return b""
    return data[int(offsets[index]):int(offsets[index + 1])].tobytes()


def open_capture(content: bytes):
    """Open an NPZ with pickle disabled; callers must close the result."""
    return np.load(io.BytesIO(content), allow_pickle=False)


def metadata(content: bytes) -> dict[str, Any]:
    with open_capture(content) as archive:
        raw = archive["metadata_json"].astype(np.uint8, copy=False).tobytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or value.get("schema") != CONTAINER_SCHEMA:
            raise ValueError("Unsupported capture container schema")
        return value


def camera_frame(content: bytes, second_index: int) -> bytes | None:
    with open_capture(content) as archive:
        present = archive["camera_present"]
        if second_index < 0 or second_index >= len(present) or not bool(present[second_index]):
            return None
        value = _blob_at(archive["camera_jpeg_bytes"], archive["camera_jpeg_offsets"], second_index)
        return value or None


def csi_payload(content: bytes, second_index: int | None = None, limit: int = 2400) -> dict[str, Any]:
    with open_capture(content) as archive:
        seconds = archive["csi_sample_second_index"]
        receivers = archive["csi_sample_receiver_index"]
        unix_ns = archive["csi_sample_unix_ns"]
        payload = archive["csi_sample_bytes"]
        offsets = archive["csi_sample_offsets"]
        indexes = range(len(seconds)) if second_index is None else np.flatnonzero(seconds == second_index)
        rows = []
        for raw_index in indexes:
            index = int(raw_index)
            rows.append({
                "second_index": int(seconds[index]),
                "receiver_index": int(receivers[index]),
                "unix_ns": int(unix_ns[index]),
                "raw_csi_line": _blob_at(payload, offsets, index).decode("utf-8", errors="replace"),
            })
            if len(rows) >= limit:
                break
        return {"samples": rows, "count": len(rows)}

