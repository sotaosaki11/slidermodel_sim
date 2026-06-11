"""
exp06 拘束力（4×4 Lagrange 系）の検証テスト。

Checks:
1) layout_theta=45° で rdot·e_y ≈ 0 かつ rdot·e_s⊥ ≈ 0。
2) layout_theta=0 では 2×2 系と 4×4 系が同じ ds/dt を返す。
3) use_y_constraint=False かつ y≠0 では y 拘束が破れる（4×4 が必要であることの確認）。
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

from config.default_params import EXP04_DEFAULTS, EXP06_LAYOUT_THETA
from core.hydrodynamics import (
    BlakeletTwoSliderMobility,
    _E_Y,
    _solve_two_slider_constrained_velocities,
)


class TestExp06YConstraint(unittest.TestCase):
    """y 方向拘束を含む 4×4 系の検証。"""

    def setUp(self) -> None:
        self.mu = 1.0
        self.a = 0.05
        self.phi = math.pi / 4.0
        self.l = float(EXP04_DEFAULTS["l"])
        self.h = 1.0
        self.es = np.array(
            [math.cos(self.phi), 0.0, -math.sin(self.phi)],
            dtype=np.float64,
        )
        self.esp = np.array(
            [math.sin(self.phi), 0.0, math.cos(self.phi)],
            dtype=np.float64,
        )
        self.blake = BlakeletTwoSliderMobility(mu=self.mu, a=self.a)

    def _bases(self, layout_theta: float) -> tuple[np.ndarray, np.ndarray]:
        ct = math.cos(layout_theta)
        st = math.sin(layout_theta)
        half_l = 0.5 * self.l
        r1_base = np.array([-half_l * ct, -half_l * st, self.h], dtype=np.float64)
        r2_base = np.array([half_l * ct, half_l * st, self.h], dtype=np.float64)
        return r1_base, r2_base

    def _solve_at(
        self,
        *,
        layout_theta: float,
        use_y_constraint: bool,
        s1: float = 0.1,
        s2: float = -0.05,
        f1: float = 1.0,
        f2: float = 0.5,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        r1_base, r2_base = self._bases(layout_theta)
        r1 = r1_base + s1 * self.es
        r2 = r2_base + s2 * self.es
        M_aa_1 = self.blake.self_mobility(r1)
        M_aa_2 = self.blake.self_mobility(r2)
        M_ab_12 = self.blake.cross_mobility(r1, r2)
        M_ab_21 = self.blake.cross_mobility(r2, r1)
        f_vec1 = f1 * self.es
        f_vec2 = f2 * self.es
        return _solve_two_slider_constrained_velocities(
            M_aa_1=M_aa_1,
            M_aa_2=M_aa_2,
            M_ab_12=M_ab_12,
            M_ab_21=M_ab_21,
            f_vec1=f_vec1,
            f_vec2=f_vec2,
            es=self.es,
            esp=self.esp,
            use_y_constraint=use_y_constraint,
        )

    def test_theta45_satisfies_y_and_perp_constraints(self) -> None:
        _, _, rdot1, rdot2 = self._solve_at(
            layout_theta=math.pi / 4.0,
            use_y_constraint=True,
        )
        tol = 1e-12
        self.assertLess(abs(float(rdot1 @ self.esp)), tol)
        self.assertLess(abs(float(rdot2 @ self.esp)), tol)
        self.assertLess(abs(float(rdot1 @ _E_Y)), tol)
        self.assertLess(abs(float(rdot2 @ _E_Y)), tol)

    def test_theta0_two_by_two_matches_four_by_four(self) -> None:
        result_2x2 = self._solve_at(layout_theta=0.0, use_y_constraint=False)
        result_4x4 = self._solve_at(layout_theta=0.0, use_y_constraint=True)
        self.assertAlmostEqual(result_2x2[0], result_4x4[0], places=12)
        self.assertAlmostEqual(result_2x2[1], result_4x4[1], places=12)

    def test_theta45_without_y_constraint_violates_y(self) -> None:
        _, _, rdot1, rdot2 = self._solve_at(
            layout_theta=math.pi / 4.0,
            use_y_constraint=False,
        )
        y_violation = max(abs(float(rdot1 @ _E_Y)), abs(float(rdot2 @ _E_Y)))
        self.assertGreater(y_violation, 1e-10)


if __name__ == "__main__":
    unittest.main()
