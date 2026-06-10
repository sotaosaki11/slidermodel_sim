"""
Blakelet mobility and exp04 Q pipeline validation tests.

Checks:
1) Wall self-mobility M_hat structure (Eq. 2.51).
2) Blakelet cross-mobility differs from Oseen at finite separation.
3) Constrained velocities are finite.
4) compute_two_slider_Q_blakelet returns a finite scalar.
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

from config.default_params import EXP02_DEFAULTS
from core.hydrodynamics import (
    BlakeletTwoSliderMobility,
    TwoSliderMobility,
    blakelet_mobility_tensor,
    wall_reflection_mobility,
)
from core.solver import SolverConfig
from core.two_slider import compute_two_slider_Q_blakelet


def _as_float(name: str) -> float:
    return float(EXP02_DEFAULTS[name])


class TestWallReflectionMobility(unittest.TestCase):
    """Eq.(2.51) wall self-mobility correction."""

    def test_diagonal_structure(self) -> None:
        mu, a, z = 1.0, 0.05, 1.0
        M_hat = wall_reflection_mobility(z, mu=mu, a=a)

        self.assertEqual(M_hat.shape, (3, 3))
        self.assertAlmostEqual(M_hat[0, 1], 0.0)
        self.assertAlmostEqual(M_hat[1, 0], 0.0)
        self.assertAlmostEqual(M_hat[0, 2], 0.0)
        self.assertAlmostEqual(M_hat[2, 0], 0.0)
        self.assertAlmostEqual(M_hat[1, 2], 0.0)
        self.assertAlmostEqual(M_hat[2, 1], 0.0)
        self.assertAlmostEqual(M_hat[0, 0], M_hat[1, 1])

    def test_normal_component_stronger_than_tangent(self) -> None:
        mu, a, z = 1.0, 0.05, 1.0
        M_hat = wall_reflection_mobility(z, mu=mu, a=a)
        self.assertLess(M_hat[2, 2], M_hat[0, 0])
        self.assertLess(0.0, abs(M_hat[0, 0]))

    def test_decays_with_height(self) -> None:
        mu, a = 1.0, 0.05
        M_near = wall_reflection_mobility(0.5, mu=mu, a=a)
        M_far = wall_reflection_mobility(2.0, mu=mu, a=a)
        self.assertGreater(np.max(np.abs(M_near)), np.max(np.abs(M_far)))

    def test_requires_positive_z(self) -> None:
        with self.assertRaises(ValueError):
            wall_reflection_mobility(0.0, mu=1.0, a=0.05)


class TestBlakeletMobilityTensor(unittest.TestCase):
    """Eq.(2.44)(4.17) Blakelet cross-mobility."""

    def setUp(self) -> None:
        self.mu = 1.0
        self.r1 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        self.r2 = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self.oseen = TwoSliderMobility(mu=self.mu, a=0.05)
        self.blake = BlakeletTwoSliderMobility(mu=self.mu, a=0.05)

    def test_differs_from_oseen(self) -> None:
        G = blakelet_mobility_tensor(self.r1, self.r2, mu=self.mu)
        O = self.oseen.oseen_tensor(self.r1, self.r2)
        ratio = G[0, 0] / O[0, 0]
        self.assertTrue(math.isfinite(ratio))
        self.assertNotAlmostEqual(ratio, 1.0, places=2)

    def test_self_mobility_includes_wall_correction(self) -> None:
        M_aa = self.blake.self_mobility(self.r1)
        gamma_0 = self.oseen.gamma_0
        m_hat_00 = wall_reflection_mobility(
            float(self.r1[2]), mu=self.mu, a=self.blake.a
        )[0, 0]
        # 論文 Eq.(2.52): M_aa = gamma_0 (I + M_hat)（gamma_0 が両項に掛かる）
        self.assertAlmostEqual(M_aa[0, 0], gamma_0 * (1.0 + m_hat_00))

    def test_source_must_be_above_wall(self) -> None:
        r_below = np.array([0.0, 0.0, -0.1], dtype=np.float64)
        with self.assertRaises(ValueError):
            blakelet_mobility_tensor(self.r1, r_below, mu=self.mu)

    def test_no_slip_on_wall(self) -> None:
        """壁面 z=0 上で G=0（滑りなし条件, 論文 式(2.44)(4.17)）。

        Blakelet の定義そのものである壁面ゼロ条件を検証する。これが破れると
        壁ありモデル（exp04/exp05）の交差移動度が物理的に誤りになる。
        """
        sources = (
            np.array([0.0, 0.0, 1.0]),
            np.array([0.3, -0.2, 0.7]),
            np.array([0.1, 0.4, 2.0]),
        )
        for source in sources:
            for x in (-1.0, 0.0, 0.5, 2.0):
                for y in (-0.5, 0.0, 1.3):
                    obs = np.array([x, y, 0.0])  # 壁面上の観測点
                    G = blakelet_mobility_tensor(obs, source, mu=self.mu)
                    self.assertLess(
                        float(np.max(np.abs(G))),
                        1e-9,
                        msg=f"no-slip broken at obs={obs}, source={source}",
                    )


class TestBlakeletComputeVelocities(unittest.TestCase):
    def test_returns_finite_velocities(self) -> None:
        mu, a = 1.0, 0.05
        phi = math.pi / 4.0
        blake = BlakeletTwoSliderMobility(mu=mu, a=a)
        es = np.array([math.cos(phi), 0.0, -math.sin(phi)], dtype=np.float64)
        esp = np.array([math.sin(phi), 0.0, math.cos(phi)], dtype=np.float64)
        r1 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        r2 = np.array([1.0, 0.0, 1.0], dtype=np.float64)

        ds1, ds2 = blake.compute_velocities(
            s1=0.0,
            s2=0.0,
            r1=r1,
            r2=r2,
            e_s=es,
            e_s_perp=esp,
            f_active1=1.0,
            f_active2=0.5,
            k=1.0,
        )
        self.assertTrue(math.isfinite(ds1))
        self.assertTrue(math.isfinite(ds2))


class TestComputeTwoSliderQBlakelet(unittest.TestCase):
    def test_returns_finite_scalar(self) -> None:
        solver_config = SolverConfig(
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            n_periods=6,
            n_eval_per_period=200,
        )
        Q = compute_two_slider_Q_blakelet(
            mu=_as_float("mu"),
            a=_as_float("a"),
            k=_as_float("k"),
            F_0=_as_float("F_0"),
            omega=_as_float("omega"),
            phi=_as_float("phi"),
            h=_as_float("h"),
            l=_as_float("l"),
            delta=_as_float("delta"),
            s1_0=_as_float("s1_0"),
            s2_0=_as_float("s2_0"),
            solver_config=solver_config,
        )
        self.assertTrue(math.isfinite(Q))


if __name__ == "__main__":
    unittest.main()
