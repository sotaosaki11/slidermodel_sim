"""
第4段階実験: 2スライダー（壁 z=0 あり, Blakelet + 拘束）/ Experiment 04.

目的:
    BlakeletTwoSliderMobility と TwoSliderTimeStepper を用いて
    2本スライダーの時系列 s1(t), s2(t) と周期平均流量 Q を計算する。
    exp02 単点と同一パラメータで Oseen 版 Q も参考値として記録する。
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

# cilia_simulation ルートを import できるようにする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import EXP04_DEFAULTS, resolve_exp04_solver_config
from core.flow_rate import FlowCalculator
from core.hydrodynamics import BlakeletTwoSliderMobility
from core.solver import TwoSliderTimeStepper
from core.two_slider import compute_two_slider_Q
from core.utils import (
    PlotStyle,
    make_run_directory,
    plot_phase_portrait_s1_s2,
    plot_two_slider_forces,
    plot_two_slider_trajectories,
    save_parameters,
    save_summary,
)

# ==========================================
# 1. パラメータ設定
# ==========================================

EXP_NAME = "exp04_two_sliders_wall"
OUTPUT_BASE = PROJECT_ROOT / "output"
PLOT_LAST_N_PERIODS = 2
PLOT_STYLE = PlotStyle()


def _as_float(name: str) -> float:
    """EXP04_DEFAULTS から数値を float として取り出す。"""
    return float(EXP04_DEFAULTS[name])


def run_experiment() -> Path:
    """
    exp04 単点パイプラインを実行して summary を保存する。

    Returns
    -------
    Path
        実行ディレクトリ（output/<exp>/<timestamp>/）。
    """
    mu = _as_float("mu")
    a = _as_float("a")
    k = _as_float("k")
    F_0 = _as_float("F_0")
    omega = _as_float("omega")
    phi = _as_float("phi")
    l = _as_float("l")
    h = _as_float("h")
    delta = _as_float("delta")
    s1_0 = _as_float("s1_0")
    s2_0 = _as_float("s2_0")

    mobility = BlakeletTwoSliderMobility(mu=mu, a=a)
    flow = FlowCalculator(mu=mu, h=h)
    solver_config = resolve_exp04_solver_config()
    stepper = TwoSliderTimeStepper(
        mobility=mobility,
        omega=omega,
        k=k,
        F_0=F_0,
        phi=phi,
        l=l,
        h=h,
        delta=delta,
        s1_0=s1_0,
        s2_0=s2_0,
        config=solver_config,
    )
    result = stepper.run()
    t_used, q_x, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
    )

    # exp02 同条件の Oseen 参考値（比較用）
    Q_oseen_ref = compute_two_slider_Q(
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        phi=phi,
        h=h,
        l=l,
        delta=delta,
        s1_0=s1_0,
        s2_0=s2_0,
        solver_config=solver_config,
    )

    sin_phi = math.sin(phi)
    z1 = h - np.asarray(result.s1, dtype=np.float64) * sin_phi
    z2 = h - np.asarray(result.s2, dtype=np.float64) * sin_phi
    min_z = float(min(z1.min(), z2.min()))

    run_dir = make_run_directory(EXP_NAME, base=OUTPUT_BASE)
    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": EXP_NAME,
            "mobility": "BlakeletTwoSliderMobility",
            "wall": "no-slip at z=0",
            "defaults": EXP04_DEFAULTS,
            "solver": solver_config,
        },
    )

    amp1 = 0.5 * float(np.max(result.s1_steady) - np.min(result.s1_steady))
    amp2 = 0.5 * float(np.max(result.s2_steady) - np.min(result.s2_steady))
    corr = float(np.corrcoef(result.s1_steady, result.s2_steady)[0, 1])
    finite_ok = bool(
        np.all(np.isfinite(result.s1))
        and np.all(np.isfinite(result.s2))
        and np.all(np.isfinite(result.f1_total))
        and np.all(np.isfinite(result.f2_total))
    )
    above_wall_ok = min_z > 0.0

    summary_lines = [
        f"experiment: {EXP_NAME}",
        f"mobility: Blakelet (wall z=0, Eq. 4.4)",
        f"integration_method: {solver_config.method}",
        f"n_periods: {solver_config.n_periods}",
        f"period_T: {result.period:.8e}",
        "",
        "--- Inputs ---",
        f"a: {a:.8e}",
        f"mu: {mu:.8e}",
        f"k: {k:.8e}",
        f"F_0: {F_0:.8e}",
        f"omega: {omega:.8e}",
        f"phi [rad]: {phi:.8e}",
        f"phi [deg]: {math.degrees(phi):.4f}",
        f"l: {l:.8e}",
        f"h: {h:.8e}",
        f"delta [rad]: {delta:.8e}",
        f"delta [deg]: {math.degrees(delta):.4f}",
        "",
        "--- Stability checks ---",
        f"finite_values_passed: {finite_ok}",
        f"min_bead_z_above_wall: {min_z:.8e}",
        f"beads_above_wall_passed: {above_wall_ok}",
        f"s1_steady_amplitude: {amp1:.8e}",
        f"s2_steady_amplitude: {amp2:.8e}",
        f"corr(s1_steady, s2_steady): {corr:.8e}",
        "",
        "--- Flow (paper-aligned) ---",
        "Definition: Q = (1/T)∫ q_x(t)dt, "
        "q_x=(z1*Fx1+z2*Fx2)/(pi*mu), z_i=h-s_i*sin(phi), Fx_i=f_i_total*cos(phi)",
        f"Q_blakelet: {Q:.8e}",
        f"Q_oseen_reference (exp02 same params): {Q_oseen_ref:.8e}",
        f"Q_blakelet_minus_Q_oseen: {(Q - Q_oseen_ref):.8e}",
        f"Q_sign: {'positive(+x)' if Q > 0 else ('negative(-x)' if Q < 0 else 'zero')}",
        f"Q_window_start: {float(t_used[0]):.8e}",
        f"Q_window_end: {float(t_used[-1]):.8e}",
        f"Q_window_duration: {float(t_used[-1] - t_used[0]):.8e}",
        "",
        "Note: exp04 uses Blakelet mobility (Eq. 2.44, 2.51, 4.4). "
        "Flow metric matches exp02/exp03 (Eq. 2.61, 4.65).",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)

    plot_two_slider_trajectories(
        run_dir / "trajectory_s1s2.png",
        result.t,
        result.s1,
        result.s2,
        result.period,
        n_periods=PLOT_LAST_N_PERIODS,
        style=PLOT_STYLE,
    )
    plot_two_slider_forces(
        run_dir / "forces_f1f2.png",
        result.t,
        result.f1_total,
        result.f2_total,
        result.period,
        n_periods=PLOT_LAST_N_PERIODS,
        style=PLOT_STYLE,
    )
    plot_phase_portrait_s1_s2(
        run_dir / "phase_portrait_s1_vs_s2.png",
        result.t,
        result.s1,
        result.s2,
        result.period,
        n_periods=PLOT_LAST_N_PERIODS,
        style=PLOT_STYLE,
    )
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_dir = run_experiment()
    logging.getLogger(__name__).info("Finished. Results: %s", run_dir)


if __name__ == "__main__":
    main()
