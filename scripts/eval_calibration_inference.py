"""Calibration-inference eval — does the chat agent infer the label?

Feeds the REAL chat pipeline: the exact ``_SYSTEM_PROMPT`` + a context
bundle carrying ``calibration.user_calibrations`` (your measured
baseline) + live-looking window descriptors, then calls the real
``_openai_answer``. If the model can't infer the labeled state from
the numbers, this prints the answer so the prompt can be iterated —
nothing is hardcoded into the bundle beyond measured stats.

    python scripts/eval_calibration_inference.py \
        --calibration path/to/sleep.json \
        --question "is the user asleep?" \
        [--window path/to/window.json] [--tier standard]

``--window`` (optional) overrides the synthesized current-window
descriptors with a JSON file — without it, the window is drawn from the
calibration's own sensor stats (what a matching window looks like).
Requires OPENAI_API_KEY (env or repo .env).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.v1 import chat  # noqa: E402
from server.v1.context_builder import (  # noqa: E402
    CSI_REFERENCE, IMU_REFERENCE, RADAR_REFERENCE)


def _load_env_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()


def _window_from_calibration(doc: dict) -> dict:
    """A 'current window' synthesized from the baseline's own stats —
    what the context builder would emit if the room looked exactly
    like the labeled state right now."""
    scenes = []
    for sid, desc in (doc.get("sensors") or {}).items():
        if not isinstance(desc, dict):
            continue
        scenes.append({
            "device_id": doc.get("device_id"),
            "sensor_id": sid,
            "type": desc.get("type"),
            "n": desc.get("n"),
            "rate_hz": desc.get("rate_hz"),
            "age_s": 0.5,
            "fields": desc.get("fields") or {},
            "scan": desc.get("scan") or {},
        })
    return {"scenes": scenes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True,
                    help="calibration/v1 doc JSON (label -> baseline)")
    ap.add_argument("--question", default="is the user asleep?")
    ap.add_argument("--window", default=None,
                    help="optional JSON overriding the live window")
    ap.add_argument("--tier", default="standard")
    args = ap.parse_args()

    _load_env_key()
    doc = json.loads(Path(args.calibration).read_text())
    label = doc.get("label") or Path(args.calibration).stem

    window = (json.loads(Path(args.window).read_text())
              if args.window else _window_from_calibration(doc))

    reference = {
        "radar_occupancy": RADAR_REFERENCE,
        "csi_link": CSI_REFERENCE,
        "watch_imu_worn": IMU_REFERENCE,
        "user_calibrations": {label: {
            "format": doc.get("format"),
            "device_id": doc.get("device_id"),
            "duration_s": doc.get("duration_s"),
            "sensors": doc.get("sensors") or {},
        }},
    }
    context_used = {
        "now": time.time(),
        "window_s": 60,
        "devices": [{"uuid": doc.get("device_id"), "name": "thoth-node",
                     "type": "thoth", "online": True,
                     "entity_key": f"device:{doc.get('device_id')}"}],
        "scenes": window.get("scenes") or [],
        "descriptors": window.get("descriptors") or [],
        "map": window.get("map") or {},
        "calibration": reference,
        "config": {"window_s": 60},
        "provenance": {"sections": ["devices", "scenes", "descriptors",
                                    "map", "calibration"]},
    }

    messages = [
        {"role": "system", "content": chat._SYSTEM_PROMPT},
        {"role": "user",
         "content": "[context bundle]\n"
                    + json.dumps(context_used, default=str)[:60_000]},
        {"role": "assistant",
         "content": "Understood — I will answer only from that bundle "
                    "plus what you send."},
        {"role": "user", "content": args.question},
    ]
    parsed = chat._openai_answer(messages, tier=args.tier)
    print(json.dumps({
        "question": args.question,
        "answer": parsed.get("answer"),
        "assessments": parsed.get("assessments") or [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
if __name__ == "__main__":
    raise SystemExit(main())
