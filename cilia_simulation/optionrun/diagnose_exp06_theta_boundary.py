"""
exp06 θ=0 境界の診断: Q*(Delta) を θ=0° と θ=1° で重ね描きする CLI.

【目的】
    layout_theta=0（2×2 拘束）と θ=1°（4×4 拘束）で Q(Δ) がどう変わるかを可視化し、
    Δ_opt の ±90° 反転の原因（拘束系の不連続切り替え）を確認する。

【実行例】
    cd cilia_simulation
    python optionrun/diagnose_exp06_theta_boundary.py
    python optionrun/diagnose_exp06_theta_boundary.py --l 2.0 --output output/diag_theta_boundary.png
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flow_rate import FlowCalculator
from core.hydrodynamics import BlakeletTwoSliderMobility
from core.solver import SolverConfig, TwoSliderTimeStepper

# 診断用: fast より軽い積分（テストと同型）
_DIAG_SOLVER = SolverConfig(
    method="EULER",
    n_periods=4,
    n_eval_per_period=400,
    rtol=1e-8,
    atol=1e-10,
)
_L_DEFAULT = 2.0
_DELTA_POINTS = 36


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
        config=_DIAG_SOLVER,
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


def _q_curve(layout_theta: float, l: float, delta_values: np.ndarray) -> np.ndarray:
    return np.array(
        [_compute_q(layout_theta=layout_theta, delta=float(d), l=l) for d in delta_values],
        dtype=np.float64,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Q*(Delta) at theta=0 vs theta=1 deg (constraint switch diagnostic).",
    )
    parser.add_argument("--l", type=float, default=_L_DEFAULT, help="Slider separation l*.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG (default: output/diag_theta_boundary_l{l}.png).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    args = _parse_args()

    delta_values = np.linspace(-math.pi, math.pi, _DELTA_POINTS, endpoint=False)
    delta_deg = np.degrees(delta_values)

    q_theta0 = _q_curve(0.0, args.l, delta_values)
    q_theta1 = _q_curve(math.radians(1.0), args.l, delta_values)
    q_theta1_2x2 = np.array(
        [
            _compute_q(
                layout_theta=math.radians(1.0),
                delta=float(d),
                l=args.l,
                use_y_constraint=False,
            )
            for d in delta_values
        ],
        dtype=np.float64,
    )
    q_theta0_4x4 = np.array(
        [
            _compute_q(
                layout_theta=0.0,
                delta=float(d),
                l=args.l,
                use_y_constraint=True,
            )
            for d in delta_values
        ],
        dtype=np.float64,
    )

    i0 = int(np.argmax(q_theta0))
    i1 = int(np.argmax(q_theta1))
    i1_2x2 = int(np.argmax(q_theta1_2x2))
    log.info(
        "l*=%g: Delta_opt theta=0 (2x2)=%.1f deg, theta=1 (4x4)=%.1f deg, "
        "theta=1 forced 2x2=%.1f deg",
        args.l,
        delta_deg[i0],
        delta_deg[i1],
        delta_deg[i1_2x2],
    )
    log.info(
        "theta=0: 2x2 vs 4x4 max abs diff = %.3e (should be 0)",
        float(np.max(np.abs(q_theta0 - q_theta0_4x4))),
    )
    log.info(
        "theta=1: 2x2 vs 4x4 max abs diff = %.3e",
        float(np.max(np.abs(q_theta1_2x2 - q_theta1))),
    )

    out_path = (
        Path(args.output)
        if args.output is not None
        else PROJECT_ROOT / "output" / f"diag_theta_boundary_l{args.l:g}.png"
    )
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    axis.plot(
        delta_deg,
        q_theta0,
        color="C0",
        linewidth=1.2,
        label=rf"$\theta=0°$ (2$\times$2, $\Delta_{{\mathrm{{opt}}}}$={delta_deg[i0]:.0f}°)",
    )
    axis.plot(
        delta_deg,
        q_theta1,
        color="C3",
        linewidth=1.5,
        label=rf"$\theta=1°$ (4$\times$4, $\Delta_{{\mathrm{{opt}}}}$={delta_deg[i1]:.0f}°)",
    )
    axis.plot(
        delta_deg,
        q_theta1_2x2,
        color="C2",
        linewidth=1.0,
        linestyle="-.",
        label=rf"$\theta=1°$ forced 2$\times$2 ($\Delta_{{\mathrm{{opt}}}}$={delta_deg[i1_2x2]:.0f}°)",
    )
    axis.plot(
        delta_deg,
        q_theta0_4x4,
        color="C0",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label=r"$\theta=0°$ forced 4$\times$4 (overlaps 2$\times$2)",
    )
    axis.axhline(0.0, color="0.5", linewidth=0.5, linestyle=":")
    axis.set_xlabel(r"$\Delta$ [deg]")
    axis.set_ylabel(r"$Q^{*}$")
    axis.set_title(rf"$Q^{{*}}(\Delta)$ at $l^{{*}}={args.l:g}$: constraint switch diagnostic")
    axis.legend(fontsize=9, loc="best")
    axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Diagnostic plot saved: %s", out_path)


if __name__ == "__main__":
    main()
