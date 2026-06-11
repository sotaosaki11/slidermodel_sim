"""
exp06 Delta-theta sweep (Blakelet, l fixed) validation tests.

Checks:
1) resolve_exp06_config returns expected fast/fine presets.
2) Serial and parallel sweeps produce identical q_map on a small grid.
3) layout_theta=0 regression: Q matches exp05-style sweep at l=2.0.
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

from config.default_params import EXP06_L_STAR, EXP06_THETA_VALUES, resolve_exp06_config
from core.solver import SolverConfig
from core.two_slider import compute_two_slider_Q_blakelet
from experiments.exp06_sweep_delta_theta import _build_cases, _sweep_q_map


def _small_solver_config() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=4,
        n_eval_per_period=400,
        rtol=1e-8,
        atol=1e-10,
    )


def _small_sweep_q_map(*, workers: int, theta_values: np.ndarray | None = None) -> np.ndarray:
    sweep_defaults, _ = resolve_exp06_config("fast")
    if theta_values is None:
        theta_values = np.asarray(EXP06_THETA_VALUES, dtype=np.float64)
    delta_values = np.linspace(
        -math.pi,
        math.pi,
        36,
        endpoint=False,
        dtype=np.float64,
    )
    cases = _build_cases(
        theta_values=theta_values,
        delta_values=delta_values,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        phi=float(sweep_defaults["phi"]),
        h=float(sweep_defaults["h"]),
        l=float(sweep_defaults["l"]),
        s1_0=float(sweep_defaults["s1_0"]),
        s2_0=float(sweep_defaults["s2_0"]),
        solver_config=_small_solver_config(),
    )
    return _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(theta_values.size, delta_values.size),
    )


class TestExp06Config(unittest.TestCase):
    def test_fast_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp06_config("fast")
        self.assertEqual(int(sweep_defaults["delta_points"]), 360)
        self.assertAlmostEqual(float(sweep_defaults["l"]), EXP06_L_STAR)
        self.assertEqual(solver_config.method.upper(), "EULER")
        self.assertEqual(solver_config.n_periods, 10)

    def test_fine_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp06_config("fine")
        self.assertEqual(int(sweep_defaults["delta_points"]), 8000)
        self.assertEqual(solver_config.method.upper(), "RK45")


class TestExp06Sweep(unittest.TestCase):
    def test_case_count_two_theta(self) -> None:
        sweep_defaults, _ = resolve_exp06_config("fast")
        theta_values = np.asarray(EXP06_THETA_VALUES, dtype=np.float64)
        delta_values = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        cases = _build_cases(
            theta_values=theta_values,
            delta_values=delta_values,
            mu=float(sweep_defaults["mu"]),
            a=float(sweep_defaults["a"]),
            k=float(sweep_defaults["k"]),
            F_0=float(sweep_defaults["F_0"]),
            omega=float(sweep_defaults["omega"]),
            phi=float(sweep_defaults["phi"]),
            h=float(sweep_defaults["h"]),
            l=float(sweep_defaults["l"]),
            s1_0=float(sweep_defaults["s1_0"]),
            s2_0=float(sweep_defaults["s2_0"]),
            solver_config=_small_solver_config(),
        )
        self.assertEqual(len(cases), 2 * 360)

    def test_serial_parallel_q_map_match(self) -> None:
        q_serial = _small_sweep_q_map(workers=1)
        cpu_count = os.cpu_count() or 1
        if cpu_count < 2:
            self.skipTest("Need at least 2 CPUs for parallel consistency check.")
        q_parallel = _small_sweep_q_map(workers=2)
        np.testing.assert_allclose(q_serial, q_parallel, rtol=0.0, atol=0.0)

    def test_all_q_finite(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        self.assertTrue(np.all(np.isfinite(q_map)))

    def test_theta0_regression_matches_no_layout(self) -> None:
        """layout_theta=0 の掃引は layout_theta 省略（exp05 互換）と一致する。"""
        sweep_defaults, _ = resolve_exp06_config("fast")
        solver_config = _small_solver_config()
        delta_values = np.linspace(-math.pi, math.pi, 24, endpoint=False, dtype=np.float64)
        mu = float(sweep_defaults["mu"])
        a = float(sweep_defaults["a"])
        k = float(sweep_defaults["k"])
        F_0 = float(sweep_defaults["F_0"])
        omega = float(sweep_defaults["omega"])
        phi = float(sweep_defaults["phi"])
        h = float(sweep_defaults["h"])
        l = float(sweep_defaults["l"])
        s1_0 = float(sweep_defaults["s1_0"])
        s2_0 = float(sweep_defaults["s2_0"])

        q_explicit_zero: list[float] = []
        q_default: list[float] = []
        for delta in delta_values:
            q_explicit_zero.append(
                compute_two_slider_Q_blakelet(
                    mu=mu,
                    a=a,
                    k=k,
                    F_0=F_0,
                    omega=omega,
                    phi=phi,
                    h=h,
                    l=l,
                    delta=float(delta),
                    s1_0=s1_0,
                    s2_0=s2_0,
                    solver_config=solver_config,
                    layout_theta=0.0,
                )
            )
            q_default.append(
                compute_two_slider_Q_blakelet(
                    mu=mu,
                    a=a,
                    k=k,
                    F_0=F_0,
                    omega=omega,
                    phi=phi,
                    h=h,
                    l=l,
                    delta=float(delta),
                    s1_0=s1_0,
                    s2_0=s2_0,
                    solver_config=solver_config,
                )
            )
        np.testing.assert_allclose(q_explicit_zero, q_default, rtol=0.0, atol=0.0)

        cases = _build_cases(
            theta_values=np.array([0.0], dtype=np.float64),
            delta_values=delta_values,
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
            phi=phi,
            h=h,
            l=l,
            s1_0=s1_0,
            s2_0=s2_0,
            solver_config=solver_config,
        )
        q_map = _sweep_q_map(
            cases=cases,
            workers=1,
            q_map_shape=(1, delta_values.size),
        )
        np.testing.assert_allclose(q_map[0, :], q_explicit_zero, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
