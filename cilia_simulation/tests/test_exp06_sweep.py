"""
exp06 Delta-l sweep (Blakelet, layout_theta fixed) validation tests.

Checks:
1) resolve_exp06_config returns expected fast/fine presets.
2) Serial and parallel sweeps produce identical q_map on a small grid.
3) layout_theta=0 regression: Q matches exp05-style sweep (no layout_theta).
4) layout_theta=0 regression: delta_opt(l) matches exp05 on a small multi-l grid.
5) All Q values are finite on the small grid.
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

from config.default_params import EXP06_LAYOUT_THETA, resolve_exp06_config
from core.solver import SolverConfig
from core.two_slider import compute_two_slider_Q_blakelet
from experiments.exp05_sweep_delta_l_wall import _build_cases as _build_exp05_cases
from experiments.exp05_sweep_delta_l_wall import _sweep_q_map as _sweep_exp05_q_map
from experiments.exp06_sweep_delta_theta import _build_cases, _sweep_q_map


def _small_solver_config() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=4,
        n_eval_per_period=400,
        rtol=1e-8,
        atol=1e-10,
    )


def _small_sweep_q_map(*, workers: int, layout_theta: float | None = None) -> np.ndarray:
    sweep_defaults, _ = resolve_exp06_config("fast")
    if layout_theta is None:
        layout_theta = float(sweep_defaults["layout_theta"])
    l_values = np.linspace(1.5, 6.0, 5, endpoint=True, dtype=np.float64)
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
        layout_theta=layout_theta,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        phi=float(sweep_defaults["phi"]),
        h=float(sweep_defaults["h"]),
        s1_0=float(sweep_defaults["s1_0"]),
        s2_0=float(sweep_defaults["s2_0"]),
        solver_config=_small_solver_config(),
    )
    return _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(l_values.size, delta_values.size),
    )


class TestExp06Config(unittest.TestCase):
    def test_fast_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp06_config("fast")
        self.assertEqual(int(sweep_defaults["delta_points"]), 360)
        self.assertEqual(int(sweep_defaults["l_points"]), 19)
        self.assertAlmostEqual(float(sweep_defaults["layout_theta"]), EXP06_LAYOUT_THETA)
        self.assertEqual(solver_config.method.upper(), "EULER")
        self.assertEqual(solver_config.n_periods, 10)

    def test_fine_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp06_config("fine")
        self.assertEqual(int(sweep_defaults["delta_points"]), 8000)
        self.assertEqual(solver_config.method.upper(), "RK45")


class TestExp06Sweep(unittest.TestCase):
    def test_case_count_matches_grid(self) -> None:
        sweep_defaults, _ = resolve_exp06_config("fast")
        l_values = np.linspace(
            float(sweep_defaults["l_min"]),
            float(sweep_defaults["l_max"]),
            int(sweep_defaults["l_points"]),
            endpoint=True,
        )
        delta_values = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        cases = _build_cases(
            l_values=l_values,
            delta_values=delta_values,
            layout_theta=float(sweep_defaults["layout_theta"]),
            mu=float(sweep_defaults["mu"]),
            a=float(sweep_defaults["a"]),
            k=float(sweep_defaults["k"]),
            F_0=float(sweep_defaults["F_0"]),
            omega=float(sweep_defaults["omega"]),
            phi=float(sweep_defaults["phi"]),
            h=float(sweep_defaults["h"]),
            s1_0=float(sweep_defaults["s1_0"]),
            s2_0=float(sweep_defaults["s2_0"]),
            solver_config=_small_solver_config(),
        )
        self.assertEqual(len(cases), 19 * 360)

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
        l_values = np.array([2.0], dtype=np.float64)
        delta_values = np.linspace(-math.pi, math.pi, 24, endpoint=False, dtype=np.float64)
        mu = float(sweep_defaults["mu"])
        a = float(sweep_defaults["a"])
        k = float(sweep_defaults["k"])
        F_0 = float(sweep_defaults["F_0"])
        omega = float(sweep_defaults["omega"])
        phi = float(sweep_defaults["phi"])
        h = float(sweep_defaults["h"])
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
                    l=float(l_values[0]),
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
                    l=float(l_values[0]),
                    delta=float(delta),
                    s1_0=s1_0,
                    s2_0=s2_0,
                    solver_config=solver_config,
                )
            )
        np.testing.assert_allclose(q_explicit_zero, q_default, rtol=0.0, atol=0.0)

        cases = _build_cases(
            l_values=l_values,
            delta_values=delta_values,
            layout_theta=0.0,
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
            phi=phi,
            h=h,
            s1_0=s1_0,
            s2_0=s2_0,
            solver_config=solver_config,
        )
        q_map = _sweep_q_map(
            cases=cases,
            workers=1,
            q_map_shape=(l_values.size, delta_values.size),
        )
        np.testing.assert_allclose(q_map[0, :], q_explicit_zero, rtol=0.0, atol=0.0)

    def test_theta0_delta_opt_matches_exp05(self) -> None:
        """layout_theta=0 の Q(Delta,l) と Delta_opt(l) が exp05 掃引と一致する。"""
        sweep_defaults, _ = resolve_exp06_config("fast")
        solver_config = _small_solver_config()
        l_values = np.linspace(1.5, 6.0, 3, endpoint=True, dtype=np.float64)
        delta_values = np.linspace(
            -math.pi,
            math.pi,
            24,
            endpoint=False,
            dtype=np.float64,
        )
        common_kwargs = dict(
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
        q_map_shape = (l_values.size, delta_values.size)

        q_exp05 = _sweep_exp05_q_map(
            cases=_build_exp05_cases(**common_kwargs),
            workers=1,
            q_map_shape=q_map_shape,
        )
        q_exp06_theta0 = _sweep_q_map(
            cases=_build_cases(**common_kwargs, layout_theta=0.0),
            workers=1,
            q_map_shape=q_map_shape,
        )
        np.testing.assert_allclose(q_exp05, q_exp06_theta0, rtol=0.0, atol=0.0)

        delta_opt_idx_exp05 = np.argmax(q_exp05, axis=1)
        delta_opt_idx_exp06 = np.argmax(q_exp06_theta0, axis=1)
        np.testing.assert_array_equal(delta_opt_idx_exp05, delta_opt_idx_exp06)
        np.testing.assert_allclose(
            delta_values[delta_opt_idx_exp05],
            delta_values[delta_opt_idx_exp06],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            q_exp05[np.arange(l_values.size), delta_opt_idx_exp05],
            q_exp06_theta0[np.arange(l_values.size), delta_opt_idx_exp06],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
