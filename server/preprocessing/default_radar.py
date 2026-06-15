"""Default preprocessing for radar data."""

from __future__ import annotations

from typing import Any, Dict, List


def get_default_pipeline() -> List[Dict[str, Any]]:
    return [
        {
            "type": "fmcw_loader",
            "enabled": True,
            "params": {},
        },
        {
            "type": "radar_basic_preprocess",
            "enabled": True,
            "params": {},
        },
    ]
