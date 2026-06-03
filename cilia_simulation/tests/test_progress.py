"""
SweepProgressTracker validation tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.progress import (
    SweepProgressTracker,
    _effective_eta_min_cases,
    _format_eta_duration,
    _smooth_eta_display,
    _throughput_eta_seconds,
)


class TestThroughputEta(unittest.TestCase):
    """スループット ETA の基本性質。"""

    def test_parallel_like_progress_has_nonzero_eta(self) -> None:
        eta = _throughput_eta_seconds(done=1700, total=6840, elapsed_seconds=56.0)
        self.assertGreater(eta, 60.0)
        self.assertLess(eta, 400.0)

    def test_serial_like_progress_eta(self) -> None:
        eta = _throughput_eta_seconds(done=10, total=100, elapsed_seconds=100.0)
        self.assertAlmostEqual(eta, 900.0, delta=1e-9)

    def test_completed_sweep_eta_zero(self) -> None:
        eta = _throughput_eta_seconds(done=100, total=100, elapsed_seconds=50.0)
        self.assertAlmostEqual(eta, 0.0, delta=1e-9)


class TestEffectiveEtaMinCases(unittest.TestCase):
    def test_explicit_min_cases(self) -> None:
        self.assertEqual(
            _effective_eta_min_cases(total_cases=6840, eta_min_cases=100),
            100,
        )

    def test_default_min_cases(self) -> None:
        self.assertEqual(
            _effective_eta_min_cases(total_cases=6840, eta_min_cases=0),
            min(60, max(15, 6840 // 300)),
        )
        self.assertEqual(
            _effective_eta_min_cases(total_cases=152000, eta_min_cases=0),
            60,
        )


class TestFormatEtaDuration(unittest.TestCase):
    def test_short_eta_keeps_seconds(self) -> None:
        self.assertEqual(_format_eta_duration(45.0), "45s")

    def test_medium_eta_rounds_to_minutes(self) -> None:
        self.assertEqual(_format_eta_duration(1250.0), "21m")

    def test_long_eta_rounds_to_five_minutes(self) -> None:
        self.assertEqual(_format_eta_duration(7500.0), "2h5m")


class TestSmoothEtaDisplay(unittest.TestCase):
    def test_first_sample_uses_raw(self) -> None:
        new_ema, smoothed = _smooth_eta_display(
            1000.0,
            eta_display_ema=None,
            eta_alpha=0.08,
        )
        self.assertAlmostEqual(new_ema, 1000.0)
        self.assertAlmostEqual(smoothed, 1000.0)

    def test_ema_dampens_step_change(self) -> None:
        _, first = _smooth_eta_display(1000.0, eta_display_ema=None, eta_alpha=0.08)
        new_ema, second = _smooth_eta_display(2000.0, eta_display_ema=first, eta_alpha=0.08)
        self.assertAlmostEqual(new_ema, 0.08 * 2000.0 + 0.92 * 1000.0)
        self.assertLess(second, 1200.0)
        self.assertGreater(second, 1000.0)


class TestSweepProgressTrackerEtaDisplay(unittest.TestCase):
    def test_time_left_estimating_before_min_cases(self) -> None:
        tracker = SweepProgressTracker(total_cases=1000, eta_min_cases=50)
        label = tracker._time_left_label(eta_raw=3600.0)
        self.assertEqual(label, "estimating")

    def test_time_left_smoothed_after_min_cases(self) -> None:
        tracker = SweepProgressTracker(
            total_cases=1000,
            eta_min_cases=10,
            eta_alpha=0.08,
        )
        tracker._done_cases = 10
        first = tracker._time_left_label(eta_raw=3600.0)
        second = tracker._time_left_label(eta_raw=7200.0)
        self.assertNotEqual(first, "estimating")
        self.assertIn("h", first)
        self.assertNotEqual(second, first)


if __name__ == "__main__":
    unittest.main()
