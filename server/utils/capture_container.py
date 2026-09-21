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


def _window_slice(grid_ts: np.ndarray, t0_ns: int | None, t1_ns: int | None) -> slice:
    count = len(grid_ts)
    if count == 0:
        return slice(0, 0)
    start = 0 if t0_ns is None else int(np.searchsorted(grid_ts, t0_ns, side="left"))
    stop = count if t1_ns is None else int(np.searchsorted(grid_ts, t1_ns, side="right"))
    return slice(max(0, start), max(0, min(count, stop)))


def sensor_window(
    content: bytes,
    sensor: str,
    t0_ns: int | None = None,
    t1_ns: int | None = None,
) -> dict[str, Any]:
    """Slice a sensor's resampled fixed-Hz grid to a time window.

    Returns the uniform grid timestamps, the ``real`` mask (measured vs held /
    interpolated), and either interpolated values (csi/sense) or held source
    indices (radar/camera). Raises ``KeyError`` for unknown sensors and
    ``ValueError`` when the container predates the resampled-grid schema.
    """
    sensor = str(sensor or "").strip().lower()
    with open_capture(content) as archive:
        files = set(archive.files)
        meta = metadata(content)
        grids = meta.get("grids") if isinstance(meta.get("grids"), dict) else {}

        if sensor == "radar":
            if "radar_grid_ts" not in files:
                raise ValueError("container has no resampled radar grid")
            grid_ts = archive["radar_grid_ts"]
            w = _window_slice(grid_ts, t0_ns, t1_ns)
            return {
                "sensor": "radar", "hz": (grids.get("radar") or {}).get("hz"),
                "t_ns": grid_ts[w].astype(np.int64).tolist(),
                "real": archive["radar_real"][w].astype(bool).tolist(),
                "source_index": archive["radar_grid_source"][w].astype(int).tolist(),
            }
        if sensor in ("csi", "wifi_csi"):
            if "csi_grid_ts" not in files:
                raise ValueError("container has no resampled csi grid")
            grid_ts = archive["csi_grid_ts"]
            w = _window_slice(grid_ts, t0_ns, t1_ns)
            return {
                "sensor": "csi", "hz": (grids.get("csi") or {}).get("hz"),
                "t_ns": grid_ts[w].astype(np.int64).tolist(),
                "amp": archive["csi_grid_amp"][w].tolist(),
                "real": archive["csi_real"][w].astype(bool).tolist(),
                "channels": (grids.get("csi") or {}).get("subcarriers"),
            }
        if sensor == "camera":
            if "camera_grid_ts" not in files:
                raise ValueError("container has no resampled camera grid")
            grid_ts = archive["camera_grid_ts"]
            w = _window_slice(grid_ts, t0_ns, t1_ns)
            return {
                "sensor": "camera", "hz": (grids.get("camera") or {}).get("hz"),
                "t_ns": grid_ts[w].astype(np.int64).tolist(),
                "real": archive["camera_real"][w].astype(bool).tolist(),
                "source_index": archive["camera_grid_source"][w].astype(int).tolist(),
            }
        if sensor in ("sense", "sense_hat", "sensehat"):
            if "sense_grid_ts" not in files:
                raise ValueError("container has no resampled sense grid")
            grid_ts = archive["sense_grid_ts"]
            w = _window_slice(grid_ts, t0_ns, t1_ns)
            return {
                "sensor": "sense", "hz": (grids.get("sense") or {}).get("hz"),
                "t_ns": grid_ts[w].astype(np.int64).tolist(),
                "values": archive["sense_grid"][w].tolist(),
                "real": archive["sense_real"][w].astype(bool).tolist(),
                "channels": (grids.get("sense") or {}).get("channels"),
            }
    raise KeyError(f"unknown sensor '{sensor}'")


def sense_payload(content: bytes, second_index: int | None = None, limit: int = 3600) -> dict[str, Any]:
    """Parse Sense HAT JSON lines from the container into per-sample dicts."""
    with open_capture(content) as archive:
        if "sense_sample_bytes" not in archive.files:
            return {"samples": [], "count": 0}
        seconds = archive["sense_sample_second_index"]
        unix_ns = archive["sense_sample_unix_ns"]
        payload = archive["sense_sample_bytes"]
        offsets = archive["sense_sample_offsets"]
        indexes = range(len(seconds)) if second_index is None else np.flatnonzero(seconds == second_index)
        rows = []
        for raw_index in indexes:
            index = int(raw_index)
            line = _blob_at(payload, offsets, index).decode("utf-8", errors="replace")
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            row["second_index"] = int(seconds[index])
            row["unix_ns"] = int(unix_ns[index])
            rows.append(row)
            if len(rows) >= limit:
                break
        return {"samples": rows, "count": len(rows)}

