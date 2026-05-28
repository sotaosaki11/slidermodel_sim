"""
第3段階実験: Delta-l 掃引（壁なし）/ Experiment 03 sweep.

目的:
    壁なし2スライダーモデルで位相差 Delta と距離 l を掃引し、
    平均流量 Q(Delta, l) を計算して可視化する。
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

# cilia_simulation ルートを import できるようにする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import EXP02_DEFAULTS, EXP03_SWEEP_DEFAULTS
from core.flow_rate import FlowCalculator
from core.hydrodynamics import TwoSliderMobility
from core.progress import SweepProgressTracker
from core.solver import SolverConfig, TwoSliderTimeStepper
from core.utils import (
    PlotStyle,
    make_run_directory,
    plot_delta_opt_vs_l,
    plot_q_vs_delta_fixed_l,
    plot_q_heatmap_delta_l,
    save_parameters,
    save_summary,
)

# ==========================================
# 1. パラメータ設定
# ==========================================

EXP_NAME = "exp03_sweep_delta_l"
OUTPUT_BASE = PROJECT_ROOT / "output"

SOLVER_METHOD = "RK45"
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-10
N_PERIODS = 10
N_EVAL_PER_PERIOD = 300
PLOT_STYLE = PlotStyle()
PROGRESS_EMA_ALPHA = 0.2


def _as_float(name: str) -> float:
    return float(EXP03_SWEEP_DEFAULTS[name])


def _as_int(name: str) -> int:
    return int(EXP03_SWEEP_DEFAULTS[name])


def _compute_Q_for_one_case(
    *,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    phi: float,
    h: float,
    l: float,
    delta: float,
    s1_0: float,
    s2_0: float,
    solver_config: SolverConfig,
) -> float:
    mobility = TwoSliderMobility(mu=mu, a=a)
    flow = FlowCalculator(mu=mu, h=h)
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
    _, _, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
    )
    return float(Q)


def run_experiment() -> Path:
    """
    Delta-l 掃引を実行し、CSV と図を出力する。
    """
    start_time = time.perf_counter()
    mu = _as_float("mu")
    a = _as_float("a")
    k = _as_float("k")
    F_0 = _as_float("F_0")
    omega = _as_float("omega")
    phi = _as_float("phi")
    h = _as_float("h")
    s1_0 = _as_float("s1_0")
    s2_0 = _as_float("s2_0")

    delta_min = _as_float("delta_min")
    delta_max = _as_float("delta_max")
    delta_points = _as_int("delta_points")
    l_min = _as_float("l_min")
    l_max = _as_float("l_max")
    l_points = _as_int("l_points")

    delta_values = np.linspace(
        delta_min,
        delta_max,
        delta_points,
        endpoint=False,
        dtype=np.float64,
    )
    l_values = np.linspace(
        l_min,
        l_max,
        l_points,
        endpoint=True,
        dtype=np.float64,
    )

    solver_config = SolverConfig(
        method=SOLVER_METHOD,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        n_periods=N_PERIODS,
        n_eval_per_period=N_EVAL_PER_PERIOD,
    )

    q_map = np.empty((l_values.size, delta_values.size), dtype=np.float64)
    total_cases = int(l_values.size * delta_values.size)
    with SweepProgressTracker(
        total_cases=total_cases,
        alpha=PROGRESS_EMA_ALPHA,
        desc="exp03 sweep",
    ) as progress:
        for i_l, l in enumerate(l_values):
            for i_d, delta in enumerate(delta_values):
                case_start = time.perf_counter()
                q_map[i_l, i_d] = _compute_Q_for_one_case(
                    mu=mu,
                    a=a,
                    k=k,
                    F_0=F_0,
                    omega=omega,
                    phi=phi,
                    h=h,
                    l=float(l),
                    delta=float(delta),
                    s1_0=s1_0,
                    s2_0=s2_0,
                    solver_config=solver_config,
                )
                case_seconds = time.perf_counter() - case_start
                progress.update(
                    l_value=float(l),
                    delta_rad=float(delta),
                    case_seconds=case_seconds,
                )

    delta_opt_idx = np.argmax(q_map, axis=1)
    delta_opt_values = delta_values[delta_opt_idx]
    q_max_values = q_map[np.arange(l_values.size), delta_opt_idx]
    l_fixed = float(EXP02_DEFAULTS["l"])
    i_l_fixed = int(np.argmin(np.abs(l_values - l_fixed)))
    l_used = float(l_values[i_l_fixed])
    q_vs_delta_fixed_l = q_map[i_l_fixed, :]

    run_dir = make_run_directory(EXP_NAME, base=OUTPUT_BASE)
    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": EXP_NAME,
            "defaults": EXP03_SWEEP_DEFAULTS,
            "solver": solver_config,
        },
    )

    delta_values_deg = np.degrees(delta_values)
    delta_opt_values_deg = np.degrees(delta_opt_values)

    rows: list[tuple[float, float, float, float]] = []
    for i_l, l in enumerate(l_values):
        for i_d, (delta_rad, delta_deg) in enumerate(
            zip(delta_values, delta_values_deg, strict=True)
        ):
            rows.append((float(l), float(delta_rad), float(delta_deg), float(q_map[i_l, i_d])))
    np.savetxt(
        run_dir / "q_delta_l.csv",
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="l,delta_rad,delta_deg,Q",
        comments="",
        encoding="utf-8",
    )

    np.savetxt(
        run_dir / "delta_opt_vs_l.csv",
        np.column_stack([l_values, delta_opt_values, delta_opt_values_deg, q_max_values]),
        delimiter=",",
        header="l,delta_opt_rad,delta_opt_deg,Q_max",
        comments="",
        encoding="utf-8",
    )
    np.savetxt(
        run_dir / "q_vs_delta_fixed_l.csv",
        np.column_stack([delta_values, delta_values_deg, q_vs_delta_fixed_l]),
        delimiter=",",
        header="delta_rad,delta_deg,Q",
        comments="",
        encoding="utf-8",
    )

    plot_q_heatmap_delta_l(
        run_dir / "Q_heatmap_delta_l.png",
        delta_values,
        l_values,
        q_map,
        style=PLOT_STYLE,
    )
    plot_delta_opt_vs_l(
        run_dir / "delta_opt_vs_l.png",
        l_values,
        delta_opt_values,
        style=PLOT_STYLE,
    )
    plot_q_vs_delta_fixed_l(
        run_dir / "Q_vs_delta_fixed_l.png",
        delta_values,
        q_vs_delta_fixed_l,
        l_fixed=l_used,
        style=PLOT_STYLE,
    )

    elapsed_seconds = time.perf_counter() - start_time
    summary_lines = [
        f"experiment: {EXP_NAME}",
        f"integration_method: {solver_config.method}",
        f"n_periods: {solver_config.n_periods}",
        f"delta_range [rad]: [{delta_min:.8e}, {delta_max:.8e})",
        f"delta_range [deg]: [{math.degrees(delta_min):.6f}, {math.degrees(delta_max):.6f})",
        f"delta_points: {delta_points}",
        f"l_range: [{l_min:.8e}, {l_max:.8e}]",
        f"l_points: {l_points}",
        f"Q_min: {float(np.min(q_map)):.8e}",
        f"Q_max: {float(np.max(q_map)):.8e}",
        f"l_fixed_requested_from_exp02: {l_fixed:.8e}",
        f"l_fixed_used: {l_used:.8e}",
        f"global_opt_l: {float(l_values[np.unravel_index(np.argmax(q_map), q_map.shape)[0]]):.8e}",
        f"global_opt_delta [rad]: {float(delta_values[np.unravel_index(np.argmax(q_map), q_map.shape)[1]]):.8e}",
        f"global_opt_delta [deg]: {float(delta_values_deg[np.unravel_index(np.argmax(q_map), q_map.shape)[1]]):.4f}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"elapsed_minutes: {elapsed_seconds / 60.0:.3f}",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)
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

