"""
exp06: θ=0 境界での拘束系切り替え（2×2 ↔ 4×4）と Δ_opt 反転の診断テスト。

仮説:
  layout_theta=0  → use_y_constraint=False（exp05 互換 2×2）
  layout_theta≠0  → use_y_constraint=True（4×4）
  θ=1° は幾何的には θ=0 に近いが力学系が別 → Q(Δ) の argmax が跳びうる。
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

from core.flow_rate import FlowCalculator
from core.hydrodynamics import BlakeletTwoSliderMobility
from core.solver import SolverConfig, TwoSliderTimeStepper


def _small_solver_config() -> SolverConfig:
    return SolverConfig(
        method="EULER",
        n_periods=4,
        n_eval_per_period=400,
        rtol=1e-8,
        atol=1e-10,
    )


def _compute_q(
    *,
    layout_theta: float,
    delta: float,
    l: float,
    solver_config: SolverConfig,
    use_y_constraint: bool | None = None,
) -> float:
    """1 ケースの Q。use_y_constraint を上書きすると拘束系を診断できる。"""
    mobility = BlakeletTwoSliderMobility(mu=1.0, a=0.05)
    stepper = TwoSliderTimeStepper(
        mobility=mobility,
        omega=2.0 * math.pi,
        k=1.0,
        F_0=1.0,
        phi=math.pi / 4.0,
        h=1.0,
        l=l,
        delta=delta,
        s1_0=0.0,
        s2_0=0.0,
        layout_theta=layout_theta,
        config=solver_config,
    )
    if use_y_constraint is not None:
        stepper._use_y_constraint = use_y_constraint
    result = stepper.run()
    flow = FlowCalculator(mu=1.0, h=1.0)
    _, _, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=math.pi / 4.0,
        use_steady_window=True,
    )
    return float(Q)


def _q_vs_delta(
    *,
    layout_theta: float,
    l: float,
    delta_values: np.ndarray,
    solver_config: SolverConfig,
    use_y_constraint: bool | None = None,
) -> np.ndarray:
    return np.array(
        [
            _compute_q(
                layout_theta=layout_theta,
                delta=float(delta),
                l=l,
                solver_config=solver_config,
                use_y_constraint=use_y_constraint,
            )
            for delta in delta_values
        ],
        dtype=np.float64,
    )


def _delta_opt_deg(delta_values: np.ndarray, q_values: np.ndarray) -> float:
    return float(np.degrees(delta_values[int(np.argmax(q_values))]))


class TestExp06ThetaBoundary(unittest.TestCase):
    """θ=0 境界の不連続性と Δ_opt 反転の診断。"""

    def setUp(self) -> None:
        self.solver_config = _small_solver_config()
        self.delta_values = np.linspace(
            -math.pi,
            math.pi,
            36,
            endpoint=False,
            dtype=np.float64,
        )
        self.l = 2.0

    def test_theta0_two_by_two_matches_four_by_four_q_map(self) -> None:
        """θ=0 では 2×2 と 4×4 で Q(Δ) が一致（exp05 互換）。"""
        q_2x2 = _q_vs_delta(
            layout_theta=0.0,
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
            use_y_constraint=False,
        )
        q_4x4 = _q_vs_delta(
            layout_theta=0.0,
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
            use_y_constraint=True,
        )
        np.testing.assert_allclose(q_2x2, q_4x4, rtol=0.0, atol=0.0)

    def test_theta1_default_four_by_four_differs_from_two_by_two(self) -> None:
        """θ=1° では 2×2 と 4×4 の Q(Δ) が一致しない（y 拘束が効く）。"""
        theta_1 = math.radians(1.0)
        q_2x2 = _q_vs_delta(
            layout_theta=theta_1,
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
            use_y_constraint=False,
        )
        q_4x4 = _q_vs_delta(
            layout_theta=theta_1,
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
            use_y_constraint=True,
        )
        self.assertFalse(np.allclose(q_2x2, q_4x4, rtol=0.0, atol=1e-12))

    def test_use_y_constraint_flag_at_theta_boundary(self) -> None:
        """layout_theta=0 → 2×2、θ≠0 → 4×4（solver 初期化時の切り替え）。"""
        mobility = BlakeletTwoSliderMobility(mu=1.0, a=0.05)
        stepper_0 = TwoSliderTimeStepper(
            mobility=mobility,
            omega=2.0 * math.pi,
            k=1.0,
            F_0=1.0,
            phi=math.pi / 4.0,
            h=1.0,
            l=self.l,
            delta=0.0,
            layout_theta=0.0,
            config=self.solver_config,
        )
        stepper_1 = TwoSliderTimeStepper(
            mobility=mobility,
            omega=2.0 * math.pi,
            k=1.0,
            F_0=1.0,
            phi=math.pi / 4.0,
            h=1.0,
            l=self.l,
            delta=0.0,
            layout_theta=math.radians(1.0),
            config=self.solver_config,
        )
        self.assertFalse(stepper_0._use_y_constraint)
        self.assertTrue(stepper_1._use_y_constraint)

    def test_theta0_vs_theta1_q_maps_differ(self) -> None:
        """θ=0（2×2）と θ=1°（4×4）で Q(Δ) が完全一致しない。"""
        q_theta0 = _q_vs_delta(
            layout_theta=0.0,
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
        )
        q_theta1 = _q_vs_delta(
            layout_theta=math.radians(1.0),
            l=self.l,
            delta_values=self.delta_values,
            solver_config=self.solver_config,
        )
        self.assertFalse(np.allclose(q_theta0, q_theta1, rtol=0.0, atol=0.0))

    def test_theta1_argmax_can_differ_between_two_and_four_by_four(self) -> None:
        """
        θ=1° で 2×2 と 4×4 の argmax が一致しないケースがある（拘束系の影響）。
        粗いグリッドでは同じ bin に落ちることもあるため、±90° 付近の細かいサブグリッドを使う。
        """
        delta_lobes = np.linspace(
            math.radians(75.0),
            math.radians(105.0),
            16,
            endpoint=True,
            dtype=np.float64,
        )
        delta_lobes = np.concatenate(
            [-delta_lobes[::-1], delta_lobes],
            dtype=np.float64,
        )
        theta_1 = math.radians(1.0)
        q_2x2 = _q_vs_delta(
            layout_theta=theta_1,
            l=self.l,
            delta_values=delta_lobes,
            solver_config=self.solver_config,
            use_y_constraint=False,
        )
        q_4x4 = _q_vs_delta(
            layout_theta=theta_1,
            l=self.l,
            delta_values=delta_lobes,
            solver_config=self.solver_config,
            use_y_constraint=True,
        )
        self.assertFalse(np.allclose(q_2x2, q_4x4, rtol=0.0, atol=1e-12))
        dopt_2x2 = _delta_opt_deg(delta_lobes, q_2x2)
        dopt_4x4 = _delta_opt_deg(delta_lobes, q_4x4)
        # 診断ログ（失敗時の手がかり）
        print(
            f"  theta=1° lobe diagnostic: Delta_opt 2x2={dopt_2x2:.1f} deg, "
            f"4x4={dopt_4x4:.1f} deg"
        )

    def test_exp05_equivalent_theta0_matches_exp06_path(self) -> None:
        """layout_theta=0 の exp06 経路は layout_theta 省略（exp05）と一致。"""
        from core.two_slider import compute_two_slider_Q_blakelet

        delta = math.radians(80.0)
        q_exp05 = compute_two_slider_Q_blakelet(
            mu=1.0,
            a=0.05,
            k=1.0,
            F_0=1.0,
            omega=2.0 * math.pi,
            phi=math.pi / 4.0,
            h=1.0,
            l=self.l,
            delta=delta,
            s1_0=0.0,
            s2_0=0.0,
            solver_config=self.solver_config,
        )
        q_exp06 = _compute_q(
            layout_theta=0.0,
            delta=delta,
            l=self.l,
            solver_config=self.solver_config,
        )
        self.assertAlmostEqual(q_exp05, q_exp06, places=12)

    def test_delta_opt_l_pattern_matches_exp05_at_theta0(self) -> None:
        """
        θ=0（exp05 相当）で l ごとの Δ_opt が exp05 掃引と一致する。
        負側（≈-90°）と正側（≈+90°）のどちらが argmax かは l とグリッド解像度に依存。
        """
        from experiments.exp05_sweep_delta_l_wall import (
            _build_cases as _build_exp05_cases,
            _sweep_q_map as _sweep_exp05_q_map,
        )
        from experiments.exp06_sweep_delta_theta import _build_cases, _sweep_q_map

        l_values = np.linspace(1.5, 6.0, 5, endpoint=True, dtype=np.float64)
        common_kwargs = dict(
            l_values=l_values,
            delta_values=self.delta_values,
            mu=1.0,
            a=0.05,
            k=1.0,
            F_0=1.0,
            omega=2.0 * math.pi,
            phi=math.pi / 4.0,
            h=1.0,
            s1_0=0.0,
            s2_0=0.0,
            solver_config=self.solver_config,
        )
        q_map_shape = (l_values.size, self.delta_values.size)

        q_exp05 = _sweep_exp05_q_map(
            cases=_build_exp05_cases(**common_kwargs),
            workers=1,
            q_map_shape=q_map_shape,
        )
        q_exp06 = _sweep_q_map(
            cases=_build_cases(**common_kwargs, layout_theta=0.0),
            workers=1,
            q_map_shape=q_map_shape,
        )
        np.testing.assert_allclose(q_exp05, q_exp06, rtol=0.0, atol=0.0)

        idx_exp05 = np.argmax(q_exp05, axis=1)
        idx_exp06 = np.argmax(q_exp06, axis=1)
        np.testing.assert_array_equal(idx_exp05, idx_exp06)

        for l_value, idx in zip(l_values, idx_exp05, strict=True):
            dopt = float(np.degrees(self.delta_values[idx]))
            lobe = "negative" if dopt < -45.0 else "positive"
            print(
                f"  theta=0 exp05/exp06 match: l*={l_value:g} -> "
                f"Delta_opt={dopt:.1f} deg ({lobe} lobe)"
            )


if __name__ == "__main__":
    unittest.main()
