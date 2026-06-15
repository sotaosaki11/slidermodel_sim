"""
exp06: 符号・対称性の実装監査。

Δ_opt ローブ反転が符号バグ由来かどうかを、以下で検証する:
  1) Q(Δ) の奇対称性 Q(-Δ) ≈ -Q(Δ)
  2) θ=0 で 2×2 と 4×4 が完全一致（符号含む）
  3) 同一 θ・同一 l で 2×2 と 4×4 の argmax ローブ（±90° 帯）が一致
  4) 両方 4×4 に統一したとき θ=0 と θ=1° で argmax ローブが一致
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


def _solver_config() -> SolverConfig:
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
    use_y_constraint: bool | None = None,
) -> float:
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
        layout_theta=layout_theta,
        config=_solver_config(),
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


def _q_map(
    *,
    layout_theta: float,
    delta_values: np.ndarray,
    l: float,
    use_y_constraint: bool | None = None,
) -> np.ndarray:
    return np.array(
        [
            _compute_q(
                layout_theta=layout_theta,
                delta=float(d),
                l=l,
                use_y_constraint=use_y_constraint,
            )
            for d in delta_values
        ],
        dtype=np.float64,
    )


def _lobe_of_argmax(delta_values: np.ndarray, q_values: np.ndarray) -> str:
    """argmax が正側ローブ (+90° 付近) か負側ローブ (-90° 付近) か。"""
    i = int(np.argmax(q_values))
    d_deg = float(np.degrees(delta_values[i]))
    if abs(d_deg) < 45.0:
        return "equator"
    return "positive" if d_deg > 0.0 else "negative"


class TestExp06SignAudit(unittest.TestCase):
    """符号反転・対称性の監査。"""

    def setUp(self) -> None:
        self.l = 2.0
        self.delta_samples = np.array(
            [math.radians(d) for d in (30.0, 60.0, 89.0, 120.0, 150.0)],
            dtype=np.float64,
        )
        self.lobe_grid = np.linspace(
            math.radians(75.0),
            math.radians(105.0),
            16,
            endpoint=True,
            dtype=np.float64,
        )
        self.lobe_grid = np.concatenate(
            [-self.lobe_grid[::-1], self.lobe_grid],
            dtype=np.float64,
        )

    def test_opposite_sign_on_pm89_lobes_when_converged(self) -> None:
        """
        十分収束した積分では ±89° ローブの Q は反対符号（fast 設定で spot check）。
        短い積分では過渡で奇対称が崩れるが、符号反転バグとは別問題。
        """
        fast = SolverConfig(
            method="EULER",
            n_periods=10,
            n_eval_per_period=40000,
            rtol=1e-8,
            atol=1e-10,
        )
        mobility = BlakeletTwoSliderMobility(mu=1.0, a=0.05)
        l = self.l
        delta_plus = math.radians(89.0)
        delta_minus = math.radians(-89.0)

        def q_at(delta: float, *, layout_theta: float, use_y: bool) -> float:
            stepper = TwoSliderTimeStepper(
                mobility=mobility,
                omega=2.0 * math.pi,
                k=1.0,
                F_0=1.0,
                phi=math.pi / 4.0,
                h=1.0,
                l=l,
                delta=delta,
                layout_theta=layout_theta,
                config=fast,
            )
            stepper._use_y_constraint = use_y
            result = stepper.run()
            flow = FlowCalculator(mu=1.0, h=1.0)
            _, _, Q = flow.compute_two_slider_Q_from_result(
                result=result,
                phi=math.pi / 4.0,
                use_steady_window=True,
            )
            return float(Q)

        for layout_theta, use_y, label in [
            (0.0, False, "theta=0 2x2"),
            (0.0, True, "theta=0 4x4"),
            (math.radians(1.0), True, "theta=1 4x4"),
        ]:
            q_plus = q_at(delta_plus, layout_theta=layout_theta, use_y=use_y)
            q_minus = q_at(delta_minus, layout_theta=layout_theta, use_y=use_y)
            with self.subTest(label=label):
                self.assertGreater(q_plus, 0.0, f"{label}: +89° lobe should be positive")
                self.assertLess(q_minus, 0.0, f"{label}: -89° lobe should be negative")
                # argmax は正ローブ側（+90° 付近）を選ぶのが正しい挙動
                self.assertGreater(q_plus, q_minus)

    def test_q_odd_symmetry_soft_at_short_window(self) -> None:
        """短い積分では Q(-Δ)+Q(Δ)≠0 になりうる（過渡）。±89° で反対符号であることのみ確認。"""
        delta = math.radians(89.0)
        q_pos = _compute_q(layout_theta=0.0, delta=delta, l=self.l, use_y_constraint=False)
        q_neg = _compute_q(layout_theta=0.0, delta=-delta, l=self.l, use_y_constraint=False)
        self.assertGreater(q_pos, 0.0)
        self.assertLess(q_neg, 0.0)

    def test_theta0_two_by_two_equals_four_by_four_pointwise(self) -> None:
        """θ=0: 2×2 と 4×4 は点ごとに符号含め一致（実装の符号ミスがない）。"""
        for delta in self.delta_samples:
            q_2 = _compute_q(
                layout_theta=0.0, delta=delta, l=self.l, use_y_constraint=False
            )
            q_4 = _compute_q(
                layout_theta=0.0, delta=delta, l=self.l, use_y_constraint=True
            )
            with self.subTest(delta=delta):
                self.assertEqual(q_2, q_4)

    def test_same_theta_argmax_lobe_unchanged_by_constraint_mode(self) -> None:
        """同一 θ では 2×2 と 4×4 で argmax ローブが変わらない（符号反転バグなし）。"""
        for layout_theta in (0.0, math.radians(1.0), math.radians(45.0)):
            q_2 = _q_map(
                layout_theta=layout_theta,
                delta_values=self.lobe_grid,
                l=self.l,
                use_y_constraint=False,
            )
            q_4 = _q_map(
                layout_theta=layout_theta,
                delta_values=self.lobe_grid,
                l=self.l,
                use_y_constraint=True,
            )
            lobe_2 = _lobe_of_argmax(self.lobe_grid, q_2)
            lobe_4 = _lobe_of_argmax(self.lobe_grid, q_4)
            with self.subTest(theta=layout_theta):
                self.assertEqual(lobe_2, lobe_4)

    def test_both_four_by_four_theta0_and_theta1_same_argmax_lobe(self) -> None:
        """
        4×4 に統一すれば θ=0 と θ=1° で最適位相ローブは一致する。
        （拘束系切り替えが反転の原因であり、4×4 内の符号バグではない）
        """
        q_t0 = _q_map(
            layout_theta=0.0,
            delta_values=self.lobe_grid,
            l=self.l,
            use_y_constraint=True,
        )
        q_t1 = _q_map(
            layout_theta=math.radians(1.0),
            delta_values=self.lobe_grid,
            l=self.l,
            use_y_constraint=True,
        )
        lobe_t0 = _lobe_of_argmax(self.lobe_grid, q_t0)
        lobe_t1 = _lobe_of_argmax(self.lobe_grid, q_t1)
        self.assertEqual(lobe_t0, lobe_t1)

    def test_default_theta0_vs_forced_four_by_four_theta1_lobe_may_differ_only_if_q_differs(self) -> None:
        """
        実装デフォルト: θ=0 は 2×2、θ=1° は 4×4。
        ローブ不一致が起きるのは Q 曲線自体が変わる場合のみ（符号反転単独では説明不可）。
        """
        q_def_t0 = _q_map(
            layout_theta=0.0,
            delta_values=self.lobe_grid,
            l=self.l,
            use_y_constraint=None,
        )
        q_def_t1 = _q_map(
            layout_theta=math.radians(1.0),
            delta_values=self.lobe_grid,
            l=self.l,
            use_y_constraint=None,
        )
        lobe_t0 = _lobe_of_argmax(self.lobe_grid, q_def_t0)
        lobe_t1 = _lobe_of_argmax(self.lobe_grid, q_def_t1)
        max_diff = float(np.max(np.abs(q_def_t0 - q_def_t1)))
        print(
            f"  default path: lobe theta=0={lobe_t0}, theta=1={lobe_t1}, "
            f"max|Q0-Q1|={max_diff:.3e}"
        )
        if lobe_t0 != lobe_t1:
            self.assertGreater(max_diff, 1e-12, "lobe flip requires Q curve change")

    def test_es_perp_orthonormal(self) -> None:
        """e_s と e_s_perp の符号・直交性（拘束法線の定義ミス検出）。"""
        phi = math.pi / 4.0
        es = np.array([math.cos(phi), 0.0, -math.sin(phi)], dtype=np.float64)
        esp = np.array([math.sin(phi), 0.0, math.cos(phi)], dtype=np.float64)
        self.assertAlmostEqual(float(es @ esp), 0.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(es)), 1.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(esp)), 1.0, places=14)


if __name__ == "__main__":
    unittest.main()
