"""
第2段階実験: 2スライダー（壁なし, Oseen + 拘束）/ Experiment 02.

目的:
    拘束付き TwoSliderMobility と TwoSliderTimeStepper を用いて
    2本スライダーの時系列 s1(t), s2(t) を計算する。
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

from config.default_params import EXP02_DEFAULTS
from core.hydrodynamics import TwoSliderMobility
from core.solver import SolverConfig, TwoSliderTimeStepper
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

EXP_NAME = "exp02_two_sliders_nowall"
OUTPUT_BASE = PROJECT_ROOT / "output"

SOLVER_METHOD = "RK45"
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-10
N_PERIODS = 10
N_EVAL_PER_PERIOD = 300
PLOT_LAST_N_PERIODS = 2
PLOT_STYLE = PlotStyle()


def _as_float(name: str) -> float:
    """EXP02_DEFAULTS から数値を float として取り出す。"""
    value = EXP02_DEFAULTS[name]
    return float(value)


def run_experiment() -> Path:
    """
    exp02 の最小パイプラインを実行して summary を保存する。

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

    mobility = TwoSliderMobility(mu=mu, a=a)
    solver_config = SolverConfig(
        method=SOLVER_METHOD,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        n_periods=N_PERIODS,
        n_eval_per_period=N_EVAL_PER_PERIOD,
    )
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

    run_dir = make_run_directory(EXP_NAME, base=OUTPUT_BASE)
    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": EXP_NAME,
            "defaults": EXP02_DEFAULTS,
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

    summary_lines = [
        f"experiment: {EXP_NAME}",
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
        f"s1_steady_amplitude: {amp1:.8e}",
        f"s2_steady_amplitude: {amp2:.8e}",
        f"corr(s1_steady, s2_steady): {corr:.8e}",
        "",
        "Note: exp02 uses free-space Oseen tensor with line constraint.",
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
