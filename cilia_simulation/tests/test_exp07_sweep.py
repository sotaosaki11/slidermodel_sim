"""
exp07 phi-l sweep (Blakelet, layout_theta fixed) validation tests.

Checks:
1) resolve_exp07_config returns expected fast/fine presets.
2) Serial and parallel sweeps produce identical q_map on a small grid.
3) Post-processed maps have expected shapes and Q_max / coordination consistency.
4) End-to-end run_experiment smoke test produces required output files.
5) All Q values are finite on the small grid.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import (
    EXP07_LAYOUT_THETA,
    EXP07_L_VALUES,
    EXP07_PHI_MAX_DEG,
    EXP07_PHI_MIN_DEG,
    EXP07_PHI_STEP_DEG,
    build_exp07_phi_values,
    resolve_exp07_config,
)
from core.solver import SolverConfig
from experiments.exp07_sweep_phi_l import (
    _build_cases,
    _postprocess_maps,
    _resolve_l_values,
    _resolve_phi_values,
    _sweep_q_map,
    run_experiment,
)


def _tiny_solver_config() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=2,
        n_eval_per_period=80,
        rtol=1e-8,
        atol=1e-10,
    )


def _small_grid_defaults() -> dict[str, float | int | tuple[float, ...]]:
    sweep_defaults, _ = resolve_exp07_config("fast")
    return {
        **sweep_defaults,
        "phi_min_deg": 0.0,
        "phi_step_deg": 30.0,
        "phi_max_deg": 90.0,
        "l_values": (1.5, 3.0, 6.0),
        "delta_points": 8,
    }


def _small_sweep_q_map(*, workers: int, layout_theta: float | None = None) -> np.ndarray:
    sweep_defaults = _small_grid_defaults()
    if layout_theta is None:
        layout_theta = float(sweep_defaults["layout_theta"])
    phi_values = _resolve_phi_values(sweep_defaults)
    l_values = _resolve_l_values(sweep_defaults)
    delta_values = np.linspace(
        -math.pi,
        math.pi,
        int(sweep_defaults["delta_points"]),
        endpoint=False,
        dtype=np.float64,
    )
    cases = _build_cases(
        phi_values=phi_values,
        l_values=l_values,
        delta_values=delta_values,
        layout_theta=layout_theta,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        h=float(sweep_defaults["h"]),
        s1_0=float(sweep_defaults["s1_0"]),
        s2_0=float(sweep_defaults["s2_0"]),
        solver_config=_tiny_solver_config(),
    )
    return _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(phi_values.size, l_values.size, delta_values.size),
    )


class TestExp07Config(unittest.TestCase):
    def test_build_phi_values_excludes_90_deg(self) -> None:
        phi_values = build_exp07_phi_values(0.0, 30.0, 90.0)
        self.assertEqual(len(phi_values), 3)
        self.assertAlmostEqual(math.degrees(phi_values[-1]), 60.0)
        self.assertTrue(all(math.degrees(phi) < 90.0 for phi in phi_values))

    def test_fast_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp07_config("fast")
        phi_values = build_exp07_phi_values(
            float(sweep_defaults["phi_min_deg"]),
            float(sweep_defaults["phi_step_deg"]),
            float(sweep_defaults["phi_max_deg"]),
        )
        self.assertEqual(len(phi_values), 19)
        self.assertEqual(len(sweep_defaults["l_values"]), 19)
        self.assertEqual(sweep_defaults["l_values"], EXP07_L_VALUES)
        self.assertAlmostEqual(float(sweep_defaults["phi_min_deg"]), EXP07_PHI_MIN_DEG)
        self.assertAlmostEqual(float(sweep_defaults["phi_step_deg"]), EXP07_PHI_STEP_DEG)
        self.assertAlmostEqual(float(sweep_defaults["phi_max_deg"]), EXP07_PHI_MAX_DEG)
        self.assertEqual(int(sweep_defaults["delta_points"]), 360)
        self.assertAlmostEqual(float(sweep_defaults["layout_theta"]), EXP07_LAYOUT_THETA)
        self.assertAlmostEqual(float(sweep_defaults["k"]), 2.0)
        self.assertAlmostEqual(float(sweep_defaults["omega"]), math.pi)
        self.assertEqual(solver_config.method.upper(), "EULER")
        self.assertEqual(solver_config.n_periods, 10)

    def test_fine_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp07_config("fine")
        self.assertEqual(int(sweep_defaults["delta_points"]), 8000)
        self.assertEqual(solver_config.method.upper(), "RK45")


class TestExp07Sweep(unittest.TestCase):
    def test_case_count_matches_grid(self) -> None:
        sweep_defaults = _small_grid_defaults()
        phi_values = _resolve_phi_values(sweep_defaults)
        l_values = _resolve_l_values(sweep_defaults)
        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        cases = _build_cases(
            phi_values=phi_values,
            l_values=l_values,
            delta_values=delta_values,
            layout_theta=float(sweep_defaults["layout_theta"]),
            mu=float(sweep_defaults["mu"]),
            a=float(sweep_defaults["a"]),
            k=float(sweep_defaults["k"]),
            F_0=float(sweep_defaults["F_0"]),
            omega=float(sweep_defaults["omega"]),
            h=float(sweep_defaults["h"]),
            s1_0=float(sweep_defaults["s1_0"]),
            s2_0=float(sweep_defaults["s2_0"]),
            solver_config=_tiny_solver_config(),
        )
        self.assertEqual(len(cases), 3 * 3 * 8)

    def test_serial_parallel_q_map_match(self) -> None:
        if os.cpu_count() is not None and os.cpu_count() < 2:
            self.skipTest("Need at least 2 CPUs for parallel test.")
        q_serial = _small_sweep_q_map(workers=1)
        q_parallel = _small_sweep_q_map(workers=2)
        np.testing.assert_allclose(q_serial, q_parallel, rtol=0.0, atol=0.0)

    def test_postprocess_shapes_and_consistency(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        delta_opt_map, q_max_map, coordination = _postprocess_maps(q_map, delta_values)

        self.assertEqual(delta_opt_map.shape, (3, 3))
        self.assertEqual(q_max_map.shape, (3, 3))
        self.assertEqual(coordination.shape, (3, 3))
        np.testing.assert_allclose(q_max_map, np.max(q_map, axis=2))
        np.testing.assert_allclose(coordination, np.sign(delta_opt_map))

    def test_all_q_finite(self) -> None:
        q_map = _small_sweep_q_map(workers=1)
        self.assertTrue(np.all(np.isfinite(q_map)))


class TestExp07Run(unittest.TestCase):
    def test_run_experiment_smoke(self) -> None:
        sweep_defaults = _small_grid_defaults()
        _, solver_config = resolve_exp07_config("fast")
        phi_values = _resolve_phi_values(sweep_defaults)
        l_values = _resolve_l_values(sweep_defaults)
        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        fake_q_map = np.random.default_rng(0).random(
            (phi_values.size, l_values.size, delta_values.size),
        )

        def _fake_resolve(mode: str = "fast"):
            return sweep_defaults, solver_config

        def _fake_sweep_q_map(**_kwargs):
            return fake_q_map

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "exp07_run"
            run_dir.mkdir()
            with (
                patch(
                    "experiments.exp07_sweep_phi_l.resolve_exp07_config",
                    side_effect=_fake_resolve,
                ),
                patch(
                    "experiments.exp07_sweep_phi_l.make_run_directory",
                    return_value=run_dir,
                ),
                patch(
                    "experiments.exp07_sweep_phi_l._sweep_q_map",
                    side_effect=_fake_sweep_q_map,
                ),
            ):
                output_dir = run_experiment(
                    mode="fast",
                    workers=1,
                    layout_theta=0.0,
                )

            self.assertEqual(output_dir, run_dir)
            for filename in (
                "fig6_phi_l_combined.png",
                "delta_opt_map_phi_l.png",
                "qmax_map_phi_l.png",
                "delta_opt_map.csv",
                "parameters.json",
                "summary.txt",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
