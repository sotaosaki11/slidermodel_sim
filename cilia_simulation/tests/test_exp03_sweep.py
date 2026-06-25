"""
exp03 Delta-l sweep validation tests.

Checks:
1) Serial and parallel sweeps produce identical q_map on a small grid.
2) Fast-mode Q(Delta) at l=2 has a peak near -90 deg (antiplectic branch).
3) All Q values are finite on the small grid.
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

from config.default_params import resolve_exp03_config
from experiments.exp03_sweep_delta_l import _build_cases, _sweep_q_map


def _small_sweep_q_map(*, workers: int) -> np.ndarray:
    """l=2 固定・Delta 36 点の縮小グリッドで q_map を返す。"""
    sweep_defaults, solver_config = resolve_exp03_config("fast")

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


class TestExp03Sweep(unittest.TestCase):
    """Tests for sweep execution and physical expectations."""

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

    def test_fast_q_peak_near_minus_90_deg_at_l2(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        delta_values = np.linspace(-math.pi, math.pi, 36, endpoint=False)
        i_peak = int(np.argmax(q_map[0, :]))
        delta_opt_deg = float(np.degrees(delta_values[i_peak]))
        # 36 点グリッド（10 deg 刻み）なので ±10 deg 程度の余裕
        self.assertLess(abs(delta_opt_deg - (-90.0)), 12.0)

    def test_all_q_finite(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        self.assertTrue(np.all(np.isfinite(q_map)))


if __name__ == "__main__":
    unittest.main()
