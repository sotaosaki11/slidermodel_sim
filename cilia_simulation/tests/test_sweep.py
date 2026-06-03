"""
core/sweep.py and core/two_slider.py validation tests.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import resolve_exp03_config
from core.sweep import default_worker_count, sweep_with_progress
from core.two_slider import compute_two_slider_Q


class TestDefaultWorkerCount(unittest.TestCase):
    def test_at_least_one(self) -> None:
        self.assertGreaterEqual(default_worker_count(), 1)


class TestComputeTwoSliderQ(unittest.TestCase):
    def test_returns_finite_scalar(self) -> None:
        sweep_defaults, solver_config = resolve_exp03_config("fast")
        Q = compute_two_slider_Q(
            mu=float(sweep_defaults["mu"]),
            a=float(sweep_defaults["a"]),
            k=float(sweep_defaults["k"]),
            F_0=float(sweep_defaults["F_0"]),
            omega=float(sweep_defaults["omega"]),
            phi=float(sweep_defaults["phi"]),
            h=float(sweep_defaults["h"]),
            l=2.0,
            delta=0.0,
            s1_0=float(sweep_defaults["s1_0"]),
            s2_0=float(sweep_defaults["s2_0"]),
            solver_config=solver_config,
        )
        self.assertTrue(math.isfinite(Q))


class TestSweepWithProgress(unittest.TestCase):
    def test_serial_accumulates_results(self) -> None:
        results: list[int] = []

        def worker(x: int) -> int:
            return x * 2

        sweep_with_progress(
            cases=[1, 2, 3],
            worker_fn=worker,
            workers=1,
            on_result=results.append,
            progress_kwargs={"eta_min_cases": 1, "update_every_cases": 1},
        )
        self.assertEqual(results, [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
