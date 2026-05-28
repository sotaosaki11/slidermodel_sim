"""
exp01 single-slider validation tests.

Checks:
1) Steady amplitude is close to linear theory.
2) Phase lag is close to linear theory.
3) Mean flow over the last period is near zero.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from config.default_params import EXP01_DEFAULTS
from core.flow_rate import FlowCalculator
from core.hydrodynamics import Mobility
from core.slider import Slider, SliderParameters
from core.solver import SolverConfig, TimeStepper


def _wrap_angle_diff(a: float, b: float) -> float:
    """Return wrapped angle difference in [-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class TestSingleSlider(unittest.TestCase):
    """Validation tests for exp01 dynamics and flow metric."""

    @classmethod
    def setUpClass(cls) -> None:
        params = SliderParameters(**EXP01_DEFAULTS)
        slider = Slider(params)
        mobility = Mobility.from_slider_parameters(params)
        flow = FlowCalculator.from_slider_parameters(params)
        solver_config = SolverConfig(
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            n_periods=8,
            n_eval_per_period=300,
        )
        result = TimeStepper(slider, mobility, solver_config).run()

        cls.params = params
        cls.slider = slider
        cls.mobility = mobility
        cls.flow = flow
        cls.result = result

    def test_steady_amplitude_matches_theory(self) -> None:
        p = self.params
        gamma_0 = self.mobility.gamma_0
        amplitude_theory = p.F_0 / math.sqrt(p.k**2 + (p.omega / gamma_0) ** 2)
        s_steady = np.asarray(self.result.s_steady, dtype=np.float64)
        amplitude_numeric = 0.5 * float(s_steady.max() - s_steady.min())

        rel_err = abs(amplitude_numeric - amplitude_theory) / max(abs(amplitude_theory), 1e-14)
        self.assertLess(rel_err, 0.02, msg=f"amplitude rel_err too large: {rel_err:.3e}")

    def test_phase_lag_matches_theory(self) -> None:
        p = self.params
        t = np.asarray(self.result.t_steady, dtype=np.float64)
        s = np.asarray(self.result.s_steady, dtype=np.float64)
        theta = p.omega * t
        basis = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
        coeffs, *_ = np.linalg.lstsq(basis, s, rcond=None)
        c_cos, c_sin, _offset = coeffs
        phase_lag_numeric = math.atan2(float(c_sin), float(c_cos))

        phase_lag_theory = math.atan(p.omega / (p.k * self.mobility.gamma_0))
        phase_error = abs(_wrap_angle_diff(phase_lag_numeric, phase_lag_theory))
        self.assertLess(
            phase_error,
            0.08,
            msg=f"phase error too large: {phase_error:.3e} rad",
        )

    def test_mean_flow_near_zero(self) -> None:
        t_used, q_x, q_mean = self.flow.compute_from_simulation(self.slider, self.result)
        self.assertGreater(len(t_used), 2)
        self.assertEqual(len(t_used), len(q_x))
        check = self.flow.check_Q_near_zero(q_mean, self.params.F_0, tol_factor=2e-4)
        self.assertTrue(
            check.passed,
            msg=(
                f"Q check failed: Q={check.Q:.3e}, "
                f"scale={check.reference_scale:.3e}, tol={check.tolerance:.3e}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
