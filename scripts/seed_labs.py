"""Seed the 8 platform labs into the `lab` table + write notebook
templates to ``lab-templates/``.

Run:  ``python scripts/seed_labs.py`` (from Brain/, with the server venv).
Idempotent: upserts by slug.

Each lab exercises the real platform loop — collect (captures), model
(whispy processors), context/events (Brain /v1), automate (rules) —
graded on notebook artifacts, not canned datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.db import Lab, get_db_session  # noqa: E402

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "lab-templates"

RUBRIC = [
    {"name": "setup", "points": 20,
     "hint": "node/sdk connection established"},
    {"name": "data", "points": 30,
     "hint": "real samples collected from a device or fixture"},
    {"name": "analysis", "points": 30,
     "hint": "metrics/figures produced"},
    {"name": "conclusions", "points": 20,
     "hint": "markdown interpretation of results"},
]

LABS = [
    {
        "slug": "sensing-radar-presence",
        "title": "Radar Presence Detection",
        "track": "radar", "level": "beginner", "order_in_track": 1,
        "description": (
            "Collect radar frames from your node, compute per-frame SNR, "
            "and implement the occupancy threshold rule (SNR >= 3 dB). "
            "Then register it as a rule model and watch prediction events "
            "arrive over the Brain event stream."
        ),
        "objectives": [
            "Fetch the latest radar observation via the SDK/node API",
            "Compute snr_db statistics over a capture",
            "Implement occupied = snr_mean >= 3 and evaluate on held-out frames",
            "Observe prediction events on /v1/events",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "figures", "conclusions"],
    },
    {
        "slug": "sensing-radar-har",
        "title": "Radar Human Activity Recognition",
        "track": "radar", "level": "advanced", "order_in_track": 2,
        "description": (
            "Go beyond presence: label captures by activity, build range/"
            "Doppler features from xy_map/range_profile, and train a small "
            "classifier. Deploy it to the node and compare edge predictions "
            "to your offline metrics."
        ),
        "objectives": [
            "Capture labeled radar sessions (>=2 classes)",
            "Build windowed features from range profile + energy map",
            "Train a baseline classifier; report confusion matrix",
            "Register the model on the node and compare live vs offline",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "figures", "conclusions"],
    },
    {
        "slug": "sensing-csi-posture",
        "title": "Wi-Fi CSI Posture Sensing",
        "track": "wifi_csi", "level": "intermediate", "order_in_track": 1,
        "description": (
            "Use CSI amplitude/phase streams to classify posture "
            "(standing / sitting / lying). Segment synchronized windows, "
            "extract subcarrier statistics, train a baseline, and explain "
            "domain shift between rooms."
        ),
        "objectives": [
            "Capture labeled CSI windows for 3 postures",
            "Extract per-subcarrier amplitude statistics",
            "Train + evaluate a posture classifier",
            "Discuss cross-room generalization (domain shift)",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "figures", "conclusions"],
    },
    {
        "slug": "sensing-camera-objects",
        "title": "Camera Object Detection Pipeline",
        "track": "sensing_fundamentals", "level": "intermediate",
        "order_in_track": 1,
        "description": (
            "Pipeline camera frames through a detection model, store "
            "detections as context states, and correlate object presence "
            "with radar occupancy to reduce false positives."
        ),
        "objectives": [
            "Tail camera observations via the node API",
            "Run a detection model over collected frames",
            "Write detections into /v1/context/state",
            "Fuse camera + radar occupancy and measure precision gain",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "figures", "conclusions"],
    },
    {
        "slug": "ml-audio-wakeword",
        "title": "Audio Wake-Word Detector",
        "track": "machine_learning", "level": "intermediate",
        "order_in_track": 1,
        "description": (
            "Collect microphone windows, build a spectrogram feature "
            "pipeline, train a keyword spotter, and wire detection to a "
            "notification action on the node."
        ),
        "objectives": [
            "Capture labeled audio windows (wake word + negatives)",
            "Compute log-mel/spectrogram features",
            "Train a small spotter; report ROC/FAR",
            "Trigger a notification event on detection",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "figures", "conclusions"],
    },
    {
        "slug": "sensing-imu-bump",
        "title": "IMU Impact / Bump Detection",
        "track": "sensing_fundamentals", "level": "beginner",
        "order_in_track": 1,
        "description": (
            "Detect sharp impacts from IMU vectors using RMS magnitude "
            "windows (rms_accel). Tune the threshold on a labeled capture "
            "and deploy as a rule model."
        ),
        "objectives": [
            "Capture IMU windows around bump events",
            "Implement rms_accel >= threshold detection",
            "Evaluate latency + false alarms on held-out windows",
            "Deploy the rule model on the node",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "conclusions"],
    },
    {
        "slug": "dataset-synchronized-collection",
        "title": "Synchronized Multi-Device Collection",
        "track": "dataset_engineering", "level": "intermediate",
        "order_in_track": 1,
        "description": (
            "Start captures on two devices covering the same event, then "
            "verify second-level alignment using manifest second indexes "
            "while per-sample timestamps keep full precision."
        ),
        "objectives": [
            "Run simultaneous captures on 2 devices (or a fixture pair)",
            "Merge manifests' second indexes — aligned epochs match",
            "Compute per-sensor sample counts per second",
            "Label the collection and export a zip",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "conclusions"],
    },
    {
        "slug": "ml-event-driven-agent",
        "title": "Event-Driven Agent",
        "track": "machine_learning", "level": "advanced",
        "order_in_track": 2,
        "description": (
            "Build an agent that subscribes to /v1/events/stream (SSE) "
            "with Last-Event-ID resume, queries /v1/context on prediction "
            "events, and creates an automation rule — no polling."
        ),
        "objectives": [
            "Subscribe to the event stream with a resume cursor",
            "React to prediction edges by querying context",
            "Create an automation rule via /v1/automation/rules",
            "Demonstrate a full occupancy -> action loop",
        ],
        "required_artifacts": ["notebook", "outputs", "metrics",
                               "conclusions"],
    },
]


def _template_nb(lab: dict) -> dict:
    """Minimal notebook template — scaffold cells referencing real SDK calls."""
    title_md = (
        f"# {lab['title']}\n\n{lab['description']}\n\n"
        "## Objectives\n" + "\n".join(f"- {o}" for o in lab["objectives"]))
    setup = (
        "# Setup — SDK client (whispy) talks to Brain with your credentials\n"
        "import whispy\n"
        "client = whispy.Client()  # ~/.whispy/credentials.json\n"
        "client.account()          # sanity check\n")
    scaffold = "# TODO: implement this lab's core analysis\n"
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [line + "\n" for line in title_md.split("\n")]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [line + "\n" for line in setup.split("\n")]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": scaffold},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_session() as db:
        for lab in LABS:
            rel = f"{lab['slug']}.ipynb"
            path = TEMPLATE_DIR / rel
            if not path.exists():
                path.write_text(json.dumps(_template_nb(lab), indent=1))
            row = db.query(Lab).filter(Lab.slug == lab["slug"]).first()
            if row is None:
                row = Lab(slug=lab["slug"])
                db.add(row)
            row.title = lab["title"]
            row.description = lab["description"]
            row.track = lab["track"]
            row.level = lab["level"]
            row.order_in_track = lab["order_in_track"]
            row.objectives = json.dumps(lab["objectives"])
            row.required_artifacts = json.dumps(lab["required_artifacts"])
            row.rubric = json.dumps(RUBRIC)
            row.template_path = str(rel)
            row.is_published = True
        db.commit()
        n = db.query(Lab).count()
        print(f"seeded {len(LABS)} labs ({n} total), "
              f"templates in {TEMPLATE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
