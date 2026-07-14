"""Validation and midpoint derivation for guided occupancy calibration."""

from __future__ import annotations

import statistics

REGIONS = ("red", "yellow", "green")


def derive_thresholds(samples_by_region: dict, *, max_spread: float = 35.0) -> dict:
    summaries = {}
    medians = []
    for region in REGIONS:
        raw = samples_by_region.get(region)
        values = [float(value) for value in raw] if isinstance(raw, list) else []
        if not values or any(value < 0.0 or value > 100.0 for value in values):
            raise ValueError(f"{region} requires ratio samples from 0 to 100")
        spread = max(values) - min(values)
        if spread > max_spread:
            raise ValueError(f"{region} calibration is unstable")
        median = float(statistics.median(values))
        medians.append(median)
        summaries[region] = {
            "sample_count": len(values), "median_percent": median,
            "minimum_percent": min(values), "maximum_percent": max(values),
            "spread_percent": spread,
        }
    if not medians[0] < medians[1] < medians[2]:
        raise ValueError("calibration medians must be ordered red < yellow < green")
    return {
        "yellow_threshold_percent": (medians[0] + medians[1]) / 2.0,
        "green_threshold_percent": (medians[1] + medians[2]) / 2.0,
        "sample_summaries": summaries,
    }
