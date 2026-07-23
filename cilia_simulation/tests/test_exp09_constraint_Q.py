"""
exp09: constraint-force-in-Q validation tests.

Checks:
1) resolve_exp09_config sets include_constraint_force_in_Q=True.
2) Fx with constraints differs from f_total*cos(phi) when lambda is nonzero.
3) Q with include_constraint_force_in_Q True/False can differ on a small case.
4) End-to-end run_experiment smoke (mocked sweep) writes required files.
"""

from __future__ import annotations

import json
import math
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
    EXP08_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    resolve_exp08_config,
    resolve_exp09_config,
)
from core.hydrodynamics import BlakeletTwoSliderMobility
from core.solver import SolverConfig, TwoSliderTimeStepper
from core.two_slider import compute_two_slider_Q_blakelet
from experiments.exp09_sweep_theta_phi_l_constraint_Q import EXP_NAME, run_experiment


def _tiny_solver() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=2,
        n_eval_per_period=40,
        rtol=1e-8,
        atol=1e-10,
    )


class TestExp09Config(unittest.TestCase):
    def test_flags(self) -> None:
        self.assertFalse(EXP08_INCLUDE_CONSTRAINT_FORCE_IN_Q)
        self.assertTrue(EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q)

    def test_resolve_exp09_sets_true(self) -> None:
        sweep_defaults, solver_config = resolve_exp09_config("fast")
        self.assertTrue(bool(sweep_defaults["include_constraint_force_in_Q"]))
        self.assertEqual(solver_config.method.upper(), "EULER")

    def test_resolve_exp08_sets_false(self) -> None:
        sweep_defaults, _ = resolve_exp08_config("fast")
        self.assertFalse(bool(sweep_defaults["include_constraint_force_in_Q"]))


class TestConstraintForceInQ(unittest.TestCase):
    def test_Fx_differs_from_rail_projection(self) -> None:
        mobility = BlakeletTwoSliderMobility(mu=1.0, a=0.05)
        phi = math.pi / 4.0
        stepper = TwoSliderTimeStepper(
            mobility=mobility,
            omega=math.pi,
            k=2.0,
            F_0=1.0,
            phi=phi,
            l=1.0,
            h=1.0,
            delta=math.pi / 2.0,
            s1_0=0.0,
            s2_0=0.0,
            layout_theta=math.pi / 4.0,
            config=_tiny_solver(),
        )
        result = stepper.run()
        Fx1, Fx2 = stepper.compute_Fx_series_with_constraints(result)
        Fx1_rail = result.f1_total * math.cos(phi)
        Fx2_rail = result.f2_total * math.cos(phi)
        self.assertGreater(float(np.max(np.abs(Fx1 - Fx1_rail))), 1e-10)
        self.assertGreater(float(np.max(np.abs(Fx2 - Fx2_rail))), 1e-10)

    def test_Q_with_and_without_constraint_can_differ(self) -> None:
        kwargs = dict(
            mu=1.0,
            a=0.05,
            k=2.0,
            F_0=1.0,
            omega=math.pi,
            phi=math.pi / 4.0,
            h=1.0,
            l=1.0,
            delta=math.pi / 2.0,
            s1_0=0.0,
            s2_0=0.0,
            solver_config=_tiny_solver(),
            layout_theta=math.pi / 4.0,
            steady_n_periods=1,
        )
        Q_without = compute_two_slider_Q_blakelet(
            include_constraint_force_in_Q=False,
            **kwargs,
        )
        Q_with = compute_two_slider_Q_blakelet(
            include_constraint_force_in_Q=True,
            **kwargs,
        )
        self.assertTrue(math.isfinite(Q_without))
        self.assertTrue(math.isfinite(Q_with))
        self.assertGreater(abs(Q_with - Q_without), 1e-12)


class TestExp09Run(unittest.TestCase):
    def test_run_experiment_smoke(self) -> None:
        sweep_defaults, solver_config = resolve_exp09_config("fast")
        sweep_defaults = {
            **sweep_defaults,
            "theta_min_deg": 0.0,
            "theta_step_deg": 90.0,
            "theta_max_deg": 90.0,
            "phi_min_deg": 0.0,
            "phi_step_deg": 45.0,
            "phi_max_deg": 90.0,
            "l_values": (1.0, 2.0),
            "delta_points": 4,
            "include_constraint_force_in_Q": True,
        }
        n_phi, n_l, n_d = 2, 2, 4
        fake_q_map = np.random.default_rng(3).random((n_phi, n_l, n_d))

        def _fake_resolve(mode: str = "fast"):
            return sweep_defaults, solver_config

        def _fake_sweep(**_kwargs):
            return fake_q_map

        def _fake_ic_lookup(*_args, **_kwargs):
            return (
                np.zeros((n_phi, n_d), dtype=np.float64),
                np.zeros((n_phi,), dtype=np.float64),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "exp09_run"
            run_dir.mkdir()
            with (
                patch(
                    "experiments.exp09_sweep_theta_phi_l_constraint_Q.resolve_exp09_config",
                    side_effect=_fake_resolve,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l.make_run_directory",
                    return_value=run_dir,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l.build_initial_position_lookup",
                    side_effect=_fake_ic_lookup,
                ),
                patch(
                    "experiments.exp08_sweep_theta_phi_l._sweep_q_map_exp08",
                    side_effect=_fake_sweep,
                ),
            ):
                output_dir = run_experiment(mode="fast", workers=1)

            self.assertEqual(output_dir, run_dir)
            for filename in (
                "delta_opt_map_all_theta.csv",
                "global_opt_vs_theta.csv",
                "inversion_boundary.csv",
                "parameters.json",
                "summary.txt",
                "global_opt_vs_theta.png",
                "Q_max_star_vs_theta.png",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)

            params = json.loads((run_dir / "parameters.json").read_text())
            self.assertEqual(params["experiment"], EXP_NAME)
            self.assertTrue(params["include_constraint_force_in_Q"])


if __name__ == "__main__":
    unittest.main()
