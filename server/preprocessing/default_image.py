"""Default preprocessing for image data."""

from __future__ import annotations

from typing import Any, Dict, List


def get_default_pipeline() -> List[Dict[str, Any]]:
    return [
        {
            "type": "image_loader",
            "enabled": True,
            "params": {"width": 224, "height": 224},
        },
        {
            "type": "image_resize",
            "enabled": True,
            "params": {"width": 224, "height": 224},
        },
        {
            "type": "canny_edge",
            "enabled": True,
            "params": {"low_threshold": 50, "high_threshold": 150, "to_grayscale": True},
        },
    ]
