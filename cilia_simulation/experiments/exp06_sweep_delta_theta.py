"""
第6段階実験: Delta-l 掃引（壁 z=0 あり, Blakelet, 配置角 theta 固定）/ Experiment 06 sweep.

目的:
    exp05 と同型の Delta-l 掃引を、x-y 平面内の相対配置角 theta を
    default_params で指定した値に固定して実行する。
    出力グラフ・CSV は exp05 と同一。

実行例:
    python experiments/exp06_sweep_delta_theta.py
    python experiments/exp06_sweep_delta_theta.py --mode fine --workers 4

IDE ▷ 実行:
    config/default_params.py の EXP06_DEFAULT_MODE を "fast" または "fine" に設定する。
    配置角は EXP06_LAYOUT_THETA（既定: 45°）で指定する。
    CLI の --mode はこの値を上書きする。
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import (
    EXP04_DEFAULTS,
    EXP06_DEFAULT_MODE,
    EXP06_LAYOUT_THETA,
    Exp06Mode,
    resolve_exp06_config,
)
from core.solver import SolverConfig
from core.sweep import default_worker_count, sweep_with_progress
from core.two_slider import compute_two_slider_Q_blakelet
from core.utils import (
    PlotStyle,
    make_run_directory,
    plot_delta_opt_vs_l,
    plot_q_vs_delta_fixed_l,
    plot_q_vs_delta_multi_l_from_q_map,
    plot_q_heatmap_delta_l,
    save_parameters,
    save_summary,
)

# ==========================================
# 1. パラメータ設定
# ==========================================

EXP_NAME = "exp06_sweep_delta_theta"
OUTPUT_BASE = PROJECT_ROOT / "output"
PLOT_STYLE = PlotStyle()
PROGRESS_EMA_ALPHA = 0.2


@dataclass(frozen=True)
class _Exp06Case:
    """
    1 ケース分の入力。ProcessPoolExecutor で pickle するためモジュール直下に置く。
    """

    i_l: int
    i_d: int
    l: float
    delta: float
    layout_theta: float
    mu: float
    a: float
    k: float
    F_0: float
    omega: float
    phi: float
    h: float
    s1_0: float
    s2_0: float
    solver_config: SolverConfig


def _as_float(sweep_defaults: dict[str, float | int], name: str) -> float:
    return float(sweep_defaults[name])


def _as_int(sweep_defaults: dict[str, float | int], name: str) -> int:
    return int(sweep_defaults[name])


def _run_one_case(case: _Exp06Case) -> tuple[int, int, float]:
    """ワーカープロセスから呼ばれる 1 ケース実行（Blakelet）。"""
    Q = compute_two_slider_Q_blakelet(
        mu=case.mu,
        a=case.a,
        k=case.k,
        F_0=case.F_0,
        omega=case.omega,
        phi=case.phi,
        h=case.h,
        l=case.l,
        delta=case.delta,
        s1_0=case.s1_0,
        s2_0=case.s2_0,
        solver_config=case.solver_config,
        layout_theta=case.layout_theta,
    )
    return case.i_l, case.i_d, Q


def _build_cases(
    *,
    l_values: np.ndarray,
    delta_values: np.ndarray,
    layout_theta: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    phi: float,
    h: float,
    s1_0: float,
    s2_0: float,
    solver_config: SolverConfig,
) -> list[_Exp06Case]:
    cases: list[_Exp06Case] = []
    for i_l, l in enumerate(l_values):
        for i_d, delta in enumerate(delta_values):
            cases.append(
                _Exp06Case(
                    i_l=i_l,
                    i_d=i_d,
                    l=float(l),
                    delta=float(delta),
                    layout_theta=float(layout_theta),
                    mu=mu,
                    a=a,
                    k=k,
                    F_0=F_0,
                    omega=omega,
                    phi=phi,
                    h=h,
                    s1_0=s1_0,
                    s2_0=s2_0,
                    solver_config=solver_config,
                )
            )
    return cases


def _progress_kwargs_for_workers(workers: int) -> dict:
    if workers > 1:
        return {
            "update_every_cases": 8,
            "mininterval": 0.5,
            "eta_alpha": 0.08,
            "eta_min_cases": 20,
        }
    return {
        "update_every_cases": 10,
        "mininterval": 0.5,
        "eta_alpha": 0.08,
        "eta_min_cases": 15,
    }


def _sweep_q_map(
    *,
    cases: list[_Exp06Case],
    workers: int,
    q_map_shape: tuple[int, int],
) -> np.ndarray:
    """全ケースを実行し q_map を返す。"""
    q_map = np.empty(q_map_shape, dtype=np.float64)

    def _store_result(result: tuple[int, int, float]) -> None:
        i_l, i_d, Q = result
        q_map[i_l, i_d] = Q

    sweep_with_progress(
        cases=cases,
        worker_fn=_run_one_case,
        workers=workers,
        on_result=_store_result,
        desc="exp06 sweep",
        alpha=PROGRESS_EMA_ALPHA,
        progress_kwargs=_progress_kwargs_for_workers(workers),
    )
    return q_map


def run_experiment(
    *,
    mode: Exp06Mode = "fast",
    workers: int | None = None,
    layout_theta: float | None = None,
) -> Path:
    """
    Delta-l 掃引（Blakelet, layout_theta 固定）を実行し、CSV と図を出力する。

    Parameters
    ----------
    mode : {"fast", "fine"}
        掃引解像度と積分設定のプリセット。
    workers : int, optional
        並列ワーカー数。None のとき cpu_count - 1。1 で直列。
    layout_theta : float, optional
        配置角 [rad]。省略時は EXP06_LAYOUT_THETA。
    """
    if workers is None:
        workers = default_worker_count()
    if layout_theta is None:
        layout_theta = EXP06_LAYOUT_THETA

    start_time = time.perf_counter()
    sweep_defaults, solver_config = resolve_exp06_config(mode)

    mu = _as_float(sweep_defaults, "mu")
    a = _as_float(sweep_defaults, "a")
    k = _as_float(sweep_defaults, "k")
    F_0 = _as_float(sweep_defaults, "F_0")
    omega = _as_float(sweep_defaults, "omega")
    phi = _as_float(sweep_defaults, "phi")
    h = _as_float(sweep_defaults, "h")
    s1_0 = _as_float(sweep_defaults, "s1_0")
    s2_0 = _as_float(sweep_defaults, "s2_0")

    delta_min = _as_float(sweep_defaults, "delta_min")
    delta_max = _as_float(sweep_defaults, "delta_max")
    delta_points = _as_int(sweep_defaults, "delta_points")
    l_min = _as_float(sweep_defaults, "l_min")
    l_max = _as_float(sweep_defaults, "l_max")
    l_points = _as_int(sweep_defaults, "l_points")

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

    cases = _build_cases(
        l_values=l_values,
        delta_values=delta_values,
        layout_theta=layout_theta,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        phi=phi,
        h=h,
        s1_0=s1_0,
        s2_0=s2_0,
        solver_config=solver_config,
    )
    q_map = _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(l_values.size, delta_values.size),
    )

    delta_opt_idx = np.argmax(q_map, axis=1)
    delta_opt_values = delta_values[delta_opt_idx]
    q_max_values = q_map[np.arange(l_values.size), delta_opt_idx]
    l_fixed = float(EXP04_DEFAULTS["l"])
    i_l_fixed = int(np.argmin(np.abs(l_values - l_fixed)))
    l_used = float(l_values[i_l_fixed])
    q_vs_delta_fixed_l = q_map[i_l_fixed, :]

    run_dir = make_run_directory(EXP_NAME, base=OUTPUT_BASE)
    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": EXP_NAME,
            "mobility": "BlakeletTwoSliderMobility",
            "wall": "no-slip at z=0",
            "mode": mode,
            "workers": workers,
            "total_cases": len(cases),
            "layout_theta_rad": layout_theta,
            "layout_theta_deg": math.degrees(layout_theta),
            "defaults": sweep_defaults,
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
    plot_q_vs_delta_multi_l_from_q_map(
        run_dir / "Q_vs_delta_multi_l.png",
        l_values=l_values,
        delta_values=delta_values,
        q_map=q_map,
        style=PLOT_STYLE,
    )

    elapsed_seconds = time.perf_counter() - start_time
    summary_lines = [
        f"experiment: {EXP_NAME}",
        f"mobility: Blakelet (wall z=0, Eq. 4.4)",
        f"mode: {mode}",
        f"workers: {workers}",
        f"total_cases: {len(cases)}",
        f"integration_method: {solver_config.method}",
        f"n_periods: {solver_config.n_periods}",
        f"n_eval_per_period: {solver_config.n_eval_per_period}",
        f"layout_theta [rad]: {layout_theta:.8e}",
        f"layout_theta [deg]: {math.degrees(layout_theta):.6f}",
        f"delta_range [rad]: [{delta_min:.8e}, {delta_max:.8e})",
        f"delta_range [deg]: [{math.degrees(delta_min):.6f}, {math.degrees(delta_max):.6f})",
        f"delta_points: {delta_points}",
        f"l_range: [{l_min:.8e}, {l_max:.8e}]",
        f"l_points: {l_points}",
        f"Q_min: {float(np.min(q_map)):.8e}",
        f"Q_max: {float(np.max(q_map)):.8e}",
        f"l_fixed_requested: {l_fixed:.8e}",
        f"l_fixed_used: {l_used:.8e}",
        f"global_opt_l: {float(l_values[np.unravel_index(np.argmax(q_map), q_map.shape)[0]]):.8e}",
        f"global_opt_delta [rad]: {float(delta_values[np.unravel_index(np.argmax(q_map), q_map.shape)[1]]):.8e}",
        f"global_opt_delta [deg]: {float(delta_values_deg[np.unravel_index(np.argmax(q_map), q_map.shape)[1]]):.4f}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"elapsed_minutes: {elapsed_seconds / 60.0:.3f}",
        "",
        "Note: layout_theta rotates base positions in x-y plane.",
        "Note: use_y_constraint=True when layout_theta != 0 (4x4 Lagrange system).",
        "Note: Flow metric Eq. (2.61)(4.65). Mobility Blakelet Eq. (2.44)(2.51)(4.4).",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "exp06: Delta-l sweep for two-slider model "
            "(wall, Blakelet, layout_theta from config)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "fine"),
        default=EXP06_DEFAULT_MODE,
        help=(
            "fast: coarse Delta grid + Euler. fine: dense grid + RK45. "
            f"Default from config: EXP06_DEFAULT_MODE={EXP06_DEFAULT_MODE!r}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Default: cpu_count - 1. Use 1 for serial.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    workers = args.workers if args.workers is not None else default_worker_count()
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    logging.getLogger(__name__).info(
        "Starting exp06 sweep: mode=%s, workers=%d, layout_theta=%.1f deg",
        args.mode,
        workers,
        math.degrees(EXP06_LAYOUT_THETA),
    )
    run_dir = run_experiment(mode=args.mode, workers=workers)
    logging.getLogger(__name__).info("exp06 sweep finished. Output: %s", run_dir)


if __name__ == "__main__":
    main()
