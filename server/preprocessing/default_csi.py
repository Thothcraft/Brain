"""Default preprocessing for CSI data."""

from __future__ import annotations

from typing import Any, Dict, List


def get_default_pipeline() -> List[Dict[str, Any]]:
    return [
        {
            "type": "csi_loader",
            "enabled": True,
            "params": {"window_size": 1000},
        },
        {
            "type": "amplitude_extractor",
            "enabled": True,
            "params": {},
        },
        {
            "type": "subcarrier_filter",
            "enabled": True,
            "params": {
                "subcarrier_start": 5,
                "subcarrier_end": 56,
            },
        },
    ]
