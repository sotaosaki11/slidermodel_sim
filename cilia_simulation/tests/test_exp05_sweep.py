"""
exp05 Delta-l sweep (Blakelet) validation tests.

Checks:
1) resolve_exp05_config returns expected fast/fine presets.
2) Serial and parallel sweeps produce identical q_map on a small grid.
3) Q(Delta, l) is not flat on the small grid (phase/distance dependence).
4) All Q values are finite on the small grid.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import resolve_exp05_config
from core.solver import SolverConfig
from experiments.exp05_sweep_delta_l_wall import _build_cases, _sweep_q_map


def _small_sweep_q_map(*, workers: int) -> np.ndarray:
    """l=2 固定・Delta 36 点の縮小グリッドで q_map を返す（Blakelet）。"""
    sweep_defaults, _ = resolve_exp05_config("fast")
    # 本番 fast プリセット（Euler 8 周期×40000 点）は重いため、検証用に縮小。
    solver_config = SolverConfig(
        method="EULER",
        n_periods=4,
        n_eval_per_period=400,
        rtol=1e-8,
        atol=1e-10,
    )

    l_values = np.array([2.0], dtype=np.float64)
    delta_values = np.linspace(
        -math.pi,
        math.pi,
        36,
        endpoint=False,
        dtype=np.float64,
    )
    cases = _build_cases(
        l_values=l_values,
        delta_values=delta_values,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        phi=float(sweep_defaults["phi"]),
        h=float(sweep_defaults["h"]),
        s1_0=float(sweep_defaults["s1_0"]),
        s2_0=float(sweep_defaults["s2_0"]),
        solver_config=solver_config,
    )
    return _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(l_values.size, delta_values.size),
    )


class TestExp05Config(unittest.TestCase):
    """Tests for fast/fine configuration presets."""

    def test_fast_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp05_config("fast")
        self.assertEqual(int(sweep_defaults["delta_points"]), 360)
        self.assertEqual(solver_config.method.upper(), "EULER")
        self.assertEqual(solver_config.n_periods, 8)
        self.assertEqual(solver_config.n_eval_per_period, 40000)

    def test_fine_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp05_config("fine")
        self.assertEqual(int(sweep_defaults["delta_points"]), 8000)
        self.assertEqual(solver_config.method.upper(), "RK45")
        self.assertEqual(solver_config.n_periods, 10)
        self.assertEqual(solver_config.n_eval_per_period, 10000)


class TestExp05Sweep(unittest.TestCase):
    """Tests for Blakelet sweep execution and physical expectations."""

    def test_serial_parallel_q_map_match(self) -> None:
        q_serial = _small_sweep_q_map(workers=1)
        cpu_count = os.cpu_count() or 1
        if cpu_count < 2:
            self.skipTest("Need at least 2 CPUs for parallel consistency check.")
        q_parallel = _small_sweep_q_map(workers=2)
        np.testing.assert_allclose(
            q_serial,
            q_parallel,
            rtol=0.0,
            atol=0.0,
            err_msg="serial and parallel q_map differ",
        )

    def test_q_map_varies_with_delta_at_l2(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        row = q_map[0, :]
        self.assertGreater(float(np.std(row)), 1e-10)
        self.assertGreater(float(np.max(row) - np.min(row)), 1e-10)

    def test_all_q_finite(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        self.assertTrue(np.all(np.isfinite(q_map)))


if __name__ == "__main__":
    unittest.main()
