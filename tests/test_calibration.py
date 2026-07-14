import unittest

from server.calibration import derive_thresholds
from server.inventory import is_newer_snapshot


class CalibrationTests(unittest.TestCase):
    def test_midpoints_are_derived_from_adjacent_medians(self):
        result = derive_thresholds({
            "red": [4, 6, 8], "yellow": [38, 40, 42], "green": [84, 86, 88],
        })
        self.assertEqual(result["yellow_threshold_percent"], 23.0)
        self.assertEqual(result["green_threshold_percent"], 63.0)

    def test_incomplete_crossed_and_unstable_samples_are_rejected(self):
        invalid = (
            {"red": [], "yellow": [40], "green": [80]},
            {"red": [30], "yellow": [20], "green": [80]},
            {"red": [0, 40], "yellow": [50], "green": [90]},
        )
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                derive_thresholds(samples)


if __name__ == "__main__":
    unittest.main()


class InventoryOrderingTests(unittest.TestCase):
    def test_revision_and_timestamp_must_not_regress(self):
        current = {"revision": 10, "timestamp": "2026-07-13T12:00:00+00:00"}
        self.assertTrue(is_newer_snapshot(11, "2026-07-13T12:01:00+00:00", current))
        self.assertFalse(is_newer_snapshot(10, "2026-07-13T12:02:00+00:00", current))
        self.assertFalse(is_newer_snapshot(11, "2026-07-13T11:59:00+00:00", current))
