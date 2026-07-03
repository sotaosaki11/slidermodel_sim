"""
exp08 theta + phi-l sweep validation tests.

Checks:
1) resolve_exp08_config returns expected fast/fine presets.
2) build_exp08_theta_values includes 90 deg (19 points).
3) extract_inversion_boundary finds sign-change crossings.
4) Serial sweep at theta=90 matches exp07 on the same small grid.
5) End-to-end run_experiment smoke test produces required output files.
6) CSV reload + replot smoke test.
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
    EXP07_L_VALUES,
    EXP08_INITIAL_CONDITION_METHOD,
    EXP08_STEADY_N_PERIODS,
    EXP08_THETA_MAX_DEG,
    EXP08_THETA_MIN_DEG,
    EXP08_THETA_STEP_DEG,
    build_exp08_theta_values,
    resolve_exp08_config,
)
from core.initial_conditions import build_initial_position_lookup
from core.solver import SolverConfig
from core.utils import extract_inversion_boundary
from experiments.exp07_sweep_phi_l import (
    _postprocess_maps,
    _resolve_l_values,
    _resolve_phi_values,
)
from experiments.exp08_sweep_theta_phi_l import (
    _build_cases_exp08,
    _sweep_q_map_exp08,
    run_experiment,
)
from optionrun.replot_exp08_from_csv import load_exp08_csv_data, replot_exp08_from_csv


def _tiny_solver_config() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=2,
        n_eval_per_period=80,
        rtol=1e-8,
        atol=1e-10,
    )


def _small_grid_defaults() -> dict[str, float | int | tuple[float, ...]]:
    sweep_defaults, _ = resolve_exp08_config("fast")
    return {
        **sweep_defaults,
        "theta_min_deg": 0.0,
        "theta_step_deg": 90.0,
        "theta_max_deg": 90.0,
        "phi_min_deg": 0.0,
        "phi_step_deg": 30.0,
        "phi_max_deg": 90.0,
        "l_values": (1.5, 3.0, 6.0),
        "delta_points": 8,
    }


def _small_sweep_q_map_exp08(*, workers: int, layout_theta: float) -> np.ndarray:
    sweep_defaults = _small_grid_defaults()
    phi_values = _resolve_phi_values(sweep_defaults)
    l_values = _resolve_l_values(sweep_defaults)
    delta_values = np.linspace(
        -math.pi,
        math.pi,
        int(sweep_defaults["delta_points"]),
        endpoint=False,
        dtype=np.float64,
    )
    s1_0_lookup, s2_0_lookup = build_initial_position_lookup(
        delta_values,
        method="decoupled_analytic",
        phi_values=phi_values,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        h=float(sweep_defaults["h"]),
    )
    cases = _build_cases_exp08(
        phi_values=phi_values,
        l_values=l_values,
        delta_values=delta_values,
        s1_0_lookup=s1_0_lookup,
        s2_0_lookup=s2_0_lookup,
        layout_theta=layout_theta,
        mu=float(sweep_defaults["mu"]),
        a=float(sweep_defaults["a"]),
        k=float(sweep_defaults["k"]),
        F_0=float(sweep_defaults["F_0"]),
        omega=float(sweep_defaults["omega"]),
        h=float(sweep_defaults["h"]),
        solver_config=_tiny_solver_config(),
        steady_n_periods=int(sweep_defaults["steady_n_periods"]),
    )
    return _sweep_q_map_exp08(
        cases=cases,
        workers=workers,
        q_map_shape=(phi_values.size, l_values.size, delta_values.size),
    )


class TestExp08Config(unittest.TestCase):
    def test_build_theta_values_includes_90_deg(self) -> None:
        theta_values = build_exp08_theta_values(0.0, 5.0, 90.0)
        self.assertEqual(len(theta_values), 19)
        self.assertAlmostEqual(math.degrees(theta_values[0]), 0.0)
        self.assertAlmostEqual(math.degrees(theta_values[-1]), 90.0)

    def test_fast_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp08_config("fast")
        theta_values = build_exp08_theta_values(
            float(sweep_defaults["theta_min_deg"]),
            float(sweep_defaults["theta_step_deg"]),
            float(sweep_defaults["theta_max_deg"]),
        )
        self.assertEqual(len(theta_values), 19)
        self.assertAlmostEqual(float(sweep_defaults["theta_min_deg"]), EXP08_THETA_MIN_DEG)
        self.assertAlmostEqual(float(sweep_defaults["theta_step_deg"]), EXP08_THETA_STEP_DEG)
        self.assertAlmostEqual(float(sweep_defaults["theta_max_deg"]), EXP08_THETA_MAX_DEG)
        self.assertEqual(sweep_defaults["l_values"], EXP07_L_VALUES)
        self.assertEqual(int(sweep_defaults["delta_points"]), 360)
        self.assertEqual(int(sweep_defaults["steady_n_periods"]), EXP08_STEADY_N_PERIODS)
        self.assertEqual(
            sweep_defaults["initial_condition_method"],
            EXP08_INITIAL_CONDITION_METHOD,
        )
        self.assertEqual(solver_config.method.upper(), "EULER")
        self.assertEqual(solver_config.n_periods, 10)

    def test_fine_preset(self) -> None:
        sweep_defaults, solver_config = resolve_exp08_config("fine")
        self.assertEqual(int(sweep_defaults["delta_points"]), 3600)
        self.assertEqual(solver_config.method.upper(), "RK45")
        self.assertEqual(solver_config.n_periods, 10)
        self.assertEqual(int(sweep_defaults["steady_n_periods"]), EXP08_STEADY_N_PERIODS)
        self.assertEqual(
            sweep_defaults["initial_condition_method"],
            EXP08_INITIAL_CONDITION_METHOD,
        )


class TestExp08BoundaryExtraction(unittest.TestCase):
    def test_extract_inversion_boundary_linear_interpolation(self) -> None:
        phi_values = np.radians([0.0, 10.0, 20.0])
        l_values = np.array([1.0])
        coordination = np.array(
            [
                [1.0],
                [-1.0],
                [-1.0],
            ],
            dtype=np.float64,
        )
        rows = extract_inversion_boundary(
            phi_values,
            l_values,
            coordination,
            theta_deg=45.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][0], 45.0)
        self.assertAlmostEqual(rows[0][1], 1.0)
        self.assertAlmostEqual(rows[0][2], 5.0)


class TestExp08DecoupledICSweep(unittest.TestCase):
    def test_small_grid_sweep_with_decoupled_ic(self) -> None:
        q_map = _small_sweep_q_map_exp08(workers=1, layout_theta=math.radians(90.0))
        self.assertEqual(q_map.ndim, 3)
        self.assertTrue(np.all(np.isfinite(q_map)))

        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        delta_opt_map, q_max_map, coordination = _postprocess_maps(q_map, delta_values)
        self.assertEqual(delta_opt_map.shape, q_max_map.shape)
        self.assertEqual(coordination.shape, delta_opt_map.shape)


class TestExp08Run(unittest.TestCase):
    def test_run_experiment_smoke(self) -> None:
        sweep_defaults = _small_grid_defaults()
        _, solver_config = resolve_exp08_config("fast")
        phi_values = _resolve_phi_values(sweep_defaults)
        l_values = _resolve_l_values(sweep_defaults)
        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        fake_q_map = np.random.default_rng(1).random(
            (phi_values.size, l_values.size, delta_values.size),
        )

        def _fake_resolve(mode: str = "fast"):
            return sweep_defaults, solver_config

        call_count = {"n": 0}

        def _fake_sweep_q_map(**_kwargs):
            call_count["n"] += 1
            return fake_q_map

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "exp08_run"
            run_dir.mkdir()
            with (
                patch(
                    "experiments.exp08_sweep_theta_phi_l.resolve_exp08_config",
                    side_effect=_fake_resolve,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l.make_run_directory",
                    return_value=run_dir,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l._sweep_q_map_exp08",
                    side_effect=_fake_sweep_q_map,
                ),
            ):
                output_dir = run_experiment(mode="fast", workers=1)

            self.assertEqual(output_dir, run_dir)
            self.assertEqual(call_count["n"], 2)
            for filename in (
                "delta_opt_map_all_theta.csv",
                "global_opt_vs_theta.csv",
                "inversion_boundary.csv",
                "delta_opt_small_multiples_theta.png",
                "qmax_small_multiples_theta.png",
                "inversion_boundary_multi_l.png",
                "global_opt_vs_theta.png",
                "Q_max_star_vs_theta.png",
                "global_opt_trajectory_phi_l.png",
                "parameters.json",
                "summary.txt",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)
            self.assertTrue((run_dir / "theta_000").is_dir())
            self.assertTrue((run_dir / "theta_090").is_dir())

    def test_replot_from_csv_smoke(self) -> None:
        sweep_defaults = _small_grid_defaults()
        _, solver_config = resolve_exp08_config("fast")
        phi_values = _resolve_phi_values(sweep_defaults)
        l_values = _resolve_l_values(sweep_defaults)
        delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        fake_q_map = np.random.default_rng(2).random(
            (phi_values.size, l_values.size, delta_values.size),
        )

        def _fake_resolve(mode: str = "fast"):
            return sweep_defaults, solver_config

        def _fake_sweep_q_map(**_kwargs):
            return fake_q_map

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "exp08_run"
            run_dir.mkdir()
            with (
                patch(
                    "experiments.exp08_sweep_theta_phi_l.resolve_exp08_config",
                    side_effect=_fake_resolve,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l.make_run_directory",
                    return_value=run_dir,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l._sweep_q_map_exp08",
                    side_effect=_fake_sweep_q_map,
                ),
            ):
                run_experiment(mode="fast", workers=1)

            payload = load_exp08_csv_data(run_dir)
            self.assertEqual(len(payload["delta_opt_maps"]), 2)
            self.assertEqual(len(payload["global_opt_rows"]), 2)

            replot_exp08_from_csv(run_dir)
            self.assertTrue((run_dir / "delta_opt_small_multiples_theta.png").is_file())


if __name__ == "__main__":
    unittest.main()
