"""
initial_conditions モジュールのテスト。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import resolve_exp08_config
from core.initial_conditions import (
    build_initial_position_lookup,
    compute_blakelet_single_slider_steady_s_at_t0,
    compute_decoupled_analytic_initial_positions,
    compute_single_slider_steady_s_at_t0_stokes,
    decoupled_steady_amplitude_and_lag,
)
from core.solver import SolverConfig


class TestDecoupledInitialPositions(unittest.TestCase):
  def setUp(self) -> None:
    sweep_defaults, _ = resolve_exp08_config("fast")
    self.mu = float(sweep_defaults["mu"])
    self.a = float(sweep_defaults["a"])
    self.k = float(sweep_defaults["k"])
    self.F_0 = float(sweep_defaults["F_0"])
    self.omega = float(sweep_defaults["omega"])
    self.h = float(sweep_defaults["h"])
    self.phi_values = np.asarray([0.0, math.radians(45.0)], dtype=np.float64)
    self.delta_values = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    self.ic_solver_config = SolverConfig(
      method="RK45",
      n_periods=10,
      n_eval_per_period=400,
    )

  def test_amplitude_positive(self) -> None:
    A, delta_lag = decoupled_steady_amplitude_and_lag(
      mu=self.mu,
      a=self.a,
      k=self.k,
      F_0=self.F_0,
      omega=self.omega,
    )
    self.assertGreater(A, 0.0)
    self.assertGreater(delta_lag, 0.0)
    self.assertLess(delta_lag, math.pi / 2.0)

  def test_analytic_lookup_shapes(self) -> None:
    s1_lookup, s2_lookup = build_initial_position_lookup(
      self.delta_values,
      method="decoupled_analytic",
      phi_values=self.phi_values,
      mu=self.mu,
      a=self.a,
      k=self.k,
      F_0=self.F_0,
      omega=self.omega,
      h=self.h,
    )
    self.assertEqual(s1_lookup.shape, (self.phi_values.size, self.delta_values.size))
    self.assertEqual(s2_lookup.shape, (self.phi_values.size,))
    np.testing.assert_allclose(s1_lookup[0], s1_lookup[1])

  def test_blakelet_lookup_shapes(self) -> None:
    s1_lookup, s2_lookup = build_initial_position_lookup(
      self.delta_values,
      method="decoupled_blakelet",
      phi_values=self.phi_values,
      mu=self.mu,
      a=self.a,
      k=self.k,
      F_0=self.F_0,
      omega=self.omega,
      h=self.h,
      ic_solver_config=self.ic_solver_config,
    )
    self.assertEqual(s1_lookup.shape, (self.phi_values.size, self.delta_values.size))
    self.assertEqual(s2_lookup.shape, (self.phi_values.size,))
    self.assertTrue(np.all(np.isfinite(s1_lookup)))
    self.assertTrue(np.all(np.isfinite(s2_lookup)))

  def test_stokes_analytic_matches_stokes_numerical(self) -> None:
    for delta in (0.0, math.pi / 4.0, -math.pi / 3.0):
      s1_analytic, _ = compute_decoupled_analytic_initial_positions(
        delta=delta,
        mu=self.mu,
        a=self.a,
        k=self.k,
        F_0=self.F_0,
        omega=self.omega,
      )
      s1_numeric = compute_single_slider_steady_s_at_t0_stokes(
        drive_phase=delta,
        mu=self.mu,
        a=self.a,
        k=self.k,
        F_0=self.F_0,
        omega=self.omega,
        solver_config=self.ic_solver_config,
      )
      self.assertAlmostEqual(s1_analytic, s1_numeric, places=5)

  def test_blakelet_phi_zero_drive_reproducible(self) -> None:
    s_a = compute_blakelet_single_slider_steady_s_at_t0(
      drive_phase=0.0,
      phi=math.radians(30.0),
      h=self.h,
      mu=self.mu,
      a=self.a,
      k=self.k,
      F_0=self.F_0,
      omega=self.omega,
      solver_config=self.ic_solver_config,
    )
    s_b = compute_blakelet_single_slider_steady_s_at_t0(
      drive_phase=0.0,
      phi=math.radians(30.0),
      h=self.h,
      mu=self.mu,
      a=self.a,
      k=self.k,
      F_0=self.F_0,
      omega=self.omega,
      solver_config=self.ic_solver_config,
    )
    self.assertAlmostEqual(s_a, s_b, places=10)


if __name__ == "__main__":
  unittest.main()
