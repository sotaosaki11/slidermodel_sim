"""
第7段階実験: phi-l 掃引（壁 z=0 あり, Blakelet, 配置角 theta 固定）/ Experiment 07 sweep.

目的:
    論文 Fig.6 と同形式の数値マップを出力する。
    各 (phi, l*) 格子点で位相差 Delta を掃引し Q を最大化する Delta_opt を求める。
    摂動理論のコンターは描かず、数値シミュレーション結果のみを背景とする。

    phi (傾き角): 掃引軸（縦軸）
    layout_theta (配置角 theta): 実行中固定。theta=0 で論文 Fig.6 配置。

実行例:
    python experiments/exp07_sweep_phi_l.py --mode fast --workers 4
    python experiments/exp07_sweep_phi_l.py --mode fast --layout-theta 0

IDE ▷ 実行:
    config/default_params.py の EXP07_DEFAULT_MODE と EXP07_LAYOUT_THETA を変更する。
    CLI の --mode / --layout-theta はこれらを上書きする。
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
    EXP07_DEFAULT_MODE,
    EXP07_LAYOUT_THETA,
    Exp07Mode,
    resolve_exp07_config,
)
from core.solver import SolverConfig
from core.sweep import default_worker_count, sweep_with_progress
from core.two_slider import compute_two_slider_Q_blakelet
from core.utils import (
    PlotStyle,
    make_run_directory,
    plot_delta_opt_map_phi_l,
    plot_fig6_style_phi_l,
    plot_qmax_map_phi_l,
    save_parameters,
    save_summary,
)

# ==========================================
# 1. パラメータ設定
# ==========================================

EXP_NAME = "exp07_sweep_phi_l"
OUTPUT_BASE = PROJECT_ROOT / "output"
PLOT_STYLE = PlotStyle()
PROGRESS_EMA_ALPHA = 0.2


@dataclass(frozen=True)
class _Exp07Case:
    """
    1 ケース分の入力。ProcessPoolExecutor で pickle するためモジュール直下に置く。
    """

    i_phi: int
    i_l: int
    i_d: int
    phi: float
    l: float
    delta: float
    layout_theta: float
    mu: float
    a: float
    k: float
    F_0: float
    omega: float
    h: float
    s1_0: float
    s2_0: float
    solver_config: SolverConfig


def _as_float(sweep_defaults: dict[str, float | int], name: str) -> float:
    return float(sweep_defaults[name])


def _as_int(sweep_defaults: dict[str, float | int], name: str) -> int:
    return int(sweep_defaults[name])


def _run_one_case(case: _Exp07Case) -> tuple[int, int, int, float]:
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
    return case.i_phi, case.i_l, case.i_d, Q


def _build_cases(
    *,
    phi_values: np.ndarray,
    l_values: np.ndarray,
    delta_values: np.ndarray,
    layout_theta: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    h: float,
    s1_0: float,
    s2_0: float,
    solver_config: SolverConfig,
) -> list[_Exp07Case]:
    cases: list[_Exp07Case] = []
    for i_phi, phi in enumerate(phi_values):
        for i_l, l in enumerate(l_values):
            for i_d, delta in enumerate(delta_values):
                cases.append(
                    _Exp07Case(
                        i_phi=i_phi,
                        i_l=i_l,
                        i_d=i_d,
                        phi=float(phi),
                        l=float(l),
                        delta=float(delta),
                        layout_theta=float(layout_theta),
                        mu=mu,
                        a=a,
                        k=k,
                        F_0=F_0,
                        omega=omega,
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
    cases: list[_Exp07Case],
    workers: int,
    q_map_shape: tuple[int, int, int],
) -> np.ndarray:
    """全ケースを実行し q_map を返す。"""
    q_map = np.empty(q_map_shape, dtype=np.float64)

    def _store_result(result: tuple[int, int, int, float]) -> None:
        i_phi, i_l, i_d, Q = result
        q_map[i_phi, i_l, i_d] = Q

    sweep_with_progress(
        cases=cases,
        worker_fn=_run_one_case,
        workers=workers,
        on_result=_store_result,
        desc="exp07 sweep",
        alpha=PROGRESS_EMA_ALPHA,
        progress_kwargs=_progress_kwargs_for_workers(workers),
    )
    return q_map


def _postprocess_maps(
    q_map: np.ndarray,
    delta_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_opt_idx = np.argmax(q_map, axis=2)
    delta_opt_map = delta_values[delta_opt_idx]
    n_phi, n_l = q_map.shape[:2]
    q_max_map = q_map[np.arange(n_phi)[:, None], np.arange(n_l), delta_opt_idx]
    coordination = np.sign(delta_opt_map)
    return delta_opt_map, q_max_map, coordination


def run_experiment(
    *,
    mode: Exp07Mode = "fast",
    workers: int | None = None,
    layout_theta: float | None = None,
) -> Path:
    """
    phi-l 掃引（Blakelet, layout_theta 固定）を実行し、Fig.6 形式の図と CSV を出力する。

    Parameters
    ----------
    mode : {"fast", "fine"}
        掃引解像度と積分設定のプリセット。
    workers : int, optional
        並列ワーカー数。None のとき cpu_count - 1。1 で直列。
    layout_theta : float, optional
        配置角 [rad]。省略時は EXP07_LAYOUT_THETA。
    """
    if workers is None:
        workers = default_worker_count()
    if layout_theta is None:
        layout_theta = EXP07_LAYOUT_THETA

    start_time = time.perf_counter()
    sweep_defaults, solver_config = resolve_exp07_config(mode)

    mu = _as_float(sweep_defaults, "mu")
    a = _as_float(sweep_defaults, "a")
    k = _as_float(sweep_defaults, "k")
    F_0 = _as_float(sweep_defaults, "F_0")
    omega = _as_float(sweep_defaults, "omega")
    h = _as_float(sweep_defaults, "h")
    s1_0 = _as_float(sweep_defaults, "s1_0")
    s2_0 = _as_float(sweep_defaults, "s2_0")

    phi_min = _as_float(sweep_defaults, "phi_min")
    phi_max = _as_float(sweep_defaults, "phi_max")
    phi_points = _as_int(sweep_defaults, "phi_points")
    delta_min = _as_float(sweep_defaults, "delta_min")
    delta_max = _as_float(sweep_defaults, "delta_max")
    delta_points = _as_int(sweep_defaults, "delta_points")
    l_min = _as_float(sweep_defaults, "l_min")
    l_max = _as_float(sweep_defaults, "l_max")
    l_points = _as_int(sweep_defaults, "l_points")

    phi_values = np.linspace(
        phi_min,
        phi_max,
        phi_points,
        endpoint=False,
        dtype=np.float64,
    )
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
        phi_values=phi_values,
        l_values=l_values,
        delta_values=delta_values,
        layout_theta=layout_theta,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        h=h,
        s1_0=s1_0,
        s2_0=s2_0,
        solver_config=solver_config,
    )
    q_map = _sweep_q_map(
        cases=cases,
        workers=workers,
        q_map_shape=(phi_values.size, l_values.size, delta_values.size),
    )

    delta_opt_map, q_max_map, coordination = _postprocess_maps(q_map, delta_values)
    global_idx = np.unravel_index(int(np.argmax(q_map)), q_map.shape)
    i_phi_opt, i_l_opt, i_d_opt = global_idx

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

    phi_values_deg = np.degrees(phi_values)
    delta_opt_values_deg = np.degrees(delta_opt_map)

    rows: list[tuple[float, float, float, float, float, float, float]] = []
    for i_phi, (phi_rad, phi_deg) in enumerate(
        zip(phi_values, phi_values_deg, strict=True)
    ):
        for i_l, l in enumerate(l_values):
            rows.append(
                (
                    float(phi_rad),
                    float(phi_deg),
                    float(l),
                    float(delta_opt_map[i_phi, i_l]),
                    float(delta_opt_values_deg[i_phi, i_l]),
                    float(q_max_map[i_phi, i_l]),
                    float(coordination[i_phi, i_l]),
                )
            )
    np.savetxt(
        run_dir / "delta_opt_map.csv",
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="phi_rad,phi_deg,l,delta_opt_rad,delta_opt_deg,Q_max,coordination",
        comments="",
        encoding="utf-8",
    )

    if mode == "fine":
        np.save(run_dir / "q_map.npy", q_map)

    plot_fig6_style_phi_l(
        run_dir / "fig6_phi_l_combined.png",
        l_values,
        phi_values,
        delta_opt_map,
        q_max_map,
        style=PLOT_STYLE,
    )
    plot_delta_opt_map_phi_l(
        run_dir / "delta_opt_map_phi_l.png",
        l_values,
        phi_values,
        delta_opt_map,
        style=PLOT_STYLE,
    )
    plot_qmax_map_phi_l(
        run_dir / "qmax_map_phi_l.png",
        l_values,
        phi_values,
        q_max_map,
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
        f"phi_range [rad]: [{phi_min:.8e}, {phi_max:.8e})",
        f"phi_range [deg]: [{math.degrees(phi_min):.6f}, {math.degrees(phi_max):.6f})",
        f"phi_points: {phi_points}",
        f"delta_range [rad]: [{delta_min:.8e}, {delta_max:.8e})",
        f"delta_range [deg]: [{math.degrees(delta_min):.6f}, {math.degrees(delta_max):.6f})",
        f"delta_points: {delta_points}",
        f"l_range: [{l_min:.8e}, {l_max:.8e}]",
        f"l_points: {l_points}",
        f"Q_min: {float(np.min(q_map)):.8e}",
        f"Q_max: {float(np.max(q_map)):.8e}",
        f"global_opt_phi [rad]: {float(phi_values[i_phi_opt]):.8e}",
        f"global_opt_phi [deg]: {float(phi_values_deg[i_phi_opt]):.4f}",
        f"global_opt_l: {float(l_values[i_l_opt]):.8e}",
        f"global_opt_delta [rad]: {float(delta_values[i_d_opt]):.8e}",
        f"global_opt_delta [deg]: {float(np.degrees(delta_values[i_d_opt])):.4f}",
        f"global_opt_Q_max: {float(q_map[global_idx]):.8e}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"elapsed_minutes: {elapsed_seconds / 60.0:.3f}",
        "",
        "Note: phi is slider tilt angle (sweep axis). layout_theta is x-y layout angle.",
        "Note: layout_theta=0 reproduces paper Fig.6 geometry.",
        "Note: default k=1, omega=2*pi. Paper Fig.6 uses k*=2, omega*=pi.",
        "Note: use_y_constraint=True when layout_theta != 0 (4x4 Lagrange system).",
        "Note: Flow metric Eq. (2.61)(4.65). Mobility Blakelet Eq. (2.44)(2.51)(4.4).",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "exp07: phi-l sweep for two-slider model "
            "(wall, Blakelet, layout_theta fixed, Delta optimized per point)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "fine"),
        default=EXP07_DEFAULT_MODE,
        help=(
            "fast: coarse Delta grid + Euler. fine: dense grid + RK45. "
            f"Default from config: EXP07_DEFAULT_MODE={EXP07_DEFAULT_MODE!r}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Default: cpu_count - 1. Use 1 for serial.",
    )
    parser.add_argument(
        "--layout-theta",
        type=float,
        default=None,
        help=(
            "Layout angle theta in degrees. "
            f"Default from config: EXP07_LAYOUT_THETA={math.degrees(EXP07_LAYOUT_THETA):.1f} deg. "
            "Use 0 for paper Fig.6 geometry."
        ),
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

    layout_theta = (
        math.radians(args.layout_theta)
        if args.layout_theta is not None
        else EXP07_LAYOUT_THETA
    )

    logging.getLogger(__name__).info(
        "Starting exp07 sweep: mode=%s, workers=%d, layout_theta=%.1f deg",
        args.mode,
        workers,
        math.degrees(layout_theta),
    )
    run_dir = run_experiment(
        mode=args.mode,
        workers=workers,
        layout_theta=layout_theta,
    )
    logging.getLogger(__name__).info("exp07 sweep finished. Output: %s", run_dir)


if __name__ == "__main__":
    main()
