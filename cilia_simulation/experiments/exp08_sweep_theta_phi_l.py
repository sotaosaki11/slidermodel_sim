"""
第8段階実験: theta 掃引 + phi-l 掃引（壁 z=0 あり, Blakelet）/ Experiment 08 sweep.

目的:
    exp07 の phi×l 掃引を layout_theta = 0,5,...,90° で繰り返す。
    各 (theta, phi, l*) 格子点で位相差 Delta を掃引し Q を最大化する Delta_opt を求める。
    小倍数マップ、反転境界、グローバル最適軌跡など論文向け図を出力する。

実行例:
    python experiments/exp08_sweep_theta_phi_l.py --mode fast --workers 4
    python experiments/exp08_sweep_theta_phi_l.py --mode fine --workers 4

IDE ▷ 実行:
    config/default_params.py の EXP08_DEFAULT_MODE、EXP08_THETA_*、
    EXP07_L_VALUES、EXP07_PHI_* を変更する。
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

from typing import Callable

from config.default_params import (
    EXP08_BOUNDARY_L_VALUES,
    EXP08_DEFAULT_MODE,
    EXP08_IC_SOLVER_PRESET,
    EXP08_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    EXP08_THETA_MAX_DEG,
    EXP08_THETA_MIN_DEG,
    EXP08_THETA_STEP_DEG,
    Exp08Mode,
    build_exp08_theta_values,
    resolve_exp08_config,
)
from core.initial_conditions import build_initial_position_lookup
from core.solver import SolverConfig
from core.sweep import default_worker_count, sweep_with_progress
from core.two_slider import compute_two_slider_Q_blakelet
from core.utils import (
    PlotStyle,
    extract_inversion_boundary,
    make_run_directory,
    plot_coordination_map_theta_phi,
    plot_delta_opt_map_phi_l,
    plot_delta_opt_small_multiples_theta,
    plot_global_opt_trajectory_phi_l,
    plot_global_opt_vs_theta,
    plot_inversion_boundary_multi_l,
    plot_inversion_boundary_theta_phi,
    plot_q_max_star_vs_theta,
    plot_qmax_map_phi_l,
    plot_qmax_small_multiples_theta,
    save_parameters,
    save_summary,
)
from experiments.exp07_sweep_phi_l import (
    _as_float,
    _as_int,
    _postprocess_maps,
    _progress_kwargs_for_workers,
    _resolve_l_values,
    _resolve_phi_values,
)

# ==========================================
# 1. パラメータ設定
# ==========================================

EXP_NAME = "exp08_sweep_theta_phi_l"
OUTPUT_BASE = PROJECT_ROOT / "output"
PLOT_STYLE = PlotStyle()
PROGRESS_EMA_ALPHA = 0.2
DELTA_OPT_MAP_HEADER = (
    "phi_rad,phi_deg,l,delta_opt_rad,delta_opt_deg,Q_max,coordination"
)
ALL_THETA_MAP_HEADER = f"theta_deg,{DELTA_OPT_MAP_HEADER}"
GLOBAL_OPT_HEADER = "theta_deg,phi_star_deg,l_star,delta_star_deg,Q_max"
INVERSION_BOUNDARY_HEADER = "theta_deg,l,phi_crit_deg"


@dataclass(frozen=True)
class _Exp08Case:
    """exp08/exp09 1 ケース分の入力。ProcessPoolExecutor で pickle する。"""

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
    steady_n_periods: int
    include_constraint_force_in_Q: bool


def _run_one_case_exp08(case: _Exp08Case) -> tuple[int, int, int, float]:
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
        steady_n_periods=case.steady_n_periods,
        include_constraint_force_in_Q=case.include_constraint_force_in_Q,
    )
    return case.i_phi, case.i_l, case.i_d, Q


def _build_cases_exp08(
    *,
    phi_values: np.ndarray,
    l_values: np.ndarray,
    delta_values: np.ndarray,
    s1_0_lookup: np.ndarray,
    s2_0_lookup: np.ndarray,
    layout_theta: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    h: float,
    solver_config: SolverConfig,
    steady_n_periods: int,
    include_constraint_force_in_Q: bool = False,
) -> list[_Exp08Case]:
    cases: list[_Exp08Case] = []
    for i_phi, phi in enumerate(phi_values):
        for i_l, l in enumerate(l_values):
            for i_d, delta in enumerate(delta_values):
                cases.append(
                    _Exp08Case(
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
                        s1_0=float(s1_0_lookup[i_phi, i_d]),
                        s2_0=float(s2_0_lookup[i_phi]),
                        solver_config=solver_config,
                        steady_n_periods=steady_n_periods,
                        include_constraint_force_in_Q=include_constraint_force_in_Q,
                    )
                )
    return cases


def _sweep_q_map_exp08(
    *,
    cases: list[_Exp08Case],
    workers: int,
    q_map_shape: tuple[int, int, int],
    progress_desc: str = "exp08 sweep",
) -> np.ndarray:
    """全ケースを実行し q_map を返す。"""
    q_map = np.empty(q_map_shape, dtype=np.float64)

    def _store_result(result: tuple[int, int, int, float]) -> None:
        i_phi, i_l, i_d, Q = result
        q_map[i_phi, i_l, i_d] = Q

    sweep_with_progress(
        cases=cases,
        worker_fn=_run_one_case_exp08,
        workers=workers,
        on_result=_store_result,
        desc=progress_desc,
        alpha=PROGRESS_EMA_ALPHA,
        progress_kwargs=_progress_kwargs_for_workers(workers),
    )
    return q_map


def _resolve_theta_values(
    sweep_defaults: dict[str, float | int | tuple[float, ...]],
) -> np.ndarray:
    theta_values = build_exp08_theta_values(
        _as_float(sweep_defaults, "theta_min_deg"),
        _as_float(sweep_defaults, "theta_step_deg"),
        _as_float(sweep_defaults, "theta_max_deg"),
    )
    return np.asarray(theta_values, dtype=np.float64)


def _theta_subdir_name(theta_deg: float) -> str:
    return f"theta_{int(round(theta_deg)):03d}"


def _format_l_tag(l_value: float) -> str:
    return f"{l_value:g}".replace(".", "p")


def _save_delta_opt_map_csv(
    path: Path,
    *,
    phi_values: np.ndarray,
    l_values: np.ndarray,
    delta_opt_map: np.ndarray,
    q_max_map: np.ndarray,
    coordination: np.ndarray,
) -> None:
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
        path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header=DELTA_OPT_MAP_HEADER,
        comments="",
        encoding="utf-8",
    )


def _save_per_theta_outputs(
    theta_dir: Path,
    *,
    phi_values: np.ndarray,
    l_values: np.ndarray,
    delta_opt_map: np.ndarray,
    q_max_map: np.ndarray,
    coordination: np.ndarray,
) -> None:
    theta_dir.mkdir(parents=True, exist_ok=True)
    _save_delta_opt_map_csv(
        theta_dir / "delta_opt_map.csv",
        phi_values=phi_values,
        l_values=l_values,
        delta_opt_map=delta_opt_map,
        q_max_map=q_max_map,
        coordination=coordination,
    )
    plot_delta_opt_map_phi_l(
        theta_dir / "delta_opt_map_phi_l.png",
        l_values,
        phi_values,
        delta_opt_map,
        style=PLOT_STYLE,
    )
    plot_qmax_map_phi_l(
        theta_dir / "qmax_map_phi_l.png",
        l_values,
        phi_values,
        q_max_map,
        style=PLOT_STYLE,
    )


def render_exp08_figures(
    run_dir: Path,
    *,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
    l_values: np.ndarray,
    delta_opt_maps: list[np.ndarray],
    q_max_maps: list[np.ndarray],
    coordination_maps: list[np.ndarray],
    global_opt_rows: list[tuple[float, float, float, float, float]],
    boundary_rows: list[tuple[float, float, float]],
    boundary_l_values: tuple[float, ...] = EXP08_BOUNDARY_L_VALUES,
) -> None:
    """exp08 実行後または CSV 再読込後に全図を生成する。"""
    theta_values_deg = np.degrees(theta_values)

    plot_delta_opt_small_multiples_theta(
        run_dir / "delta_opt_small_multiples_theta.png",
        l_values,
        phi_values,
        theta_values_deg,
        delta_opt_maps,
        style=PLOT_STYLE,
    )
    plot_qmax_small_multiples_theta(
        run_dir / "qmax_small_multiples_theta.png",
        l_values,
        phi_values,
        theta_values_deg,
        q_max_maps,
        style=PLOT_STYLE,
    )

    for l_fixed in boundary_l_values:
        if any(abs(row[1] - l_fixed) < 1e-9 for row in boundary_rows):
            plot_inversion_boundary_theta_phi(
                run_dir / f"inversion_boundary_theta_phi_l{_format_l_tag(l_fixed)}.png",
                boundary_rows,
                l_fixed=l_fixed,
                style=PLOT_STYLE,
            )
            plot_coordination_map_theta_phi(
                run_dir / f"coordination_map_theta_phi_l{_format_l_tag(l_fixed)}.png",
                theta_values_deg,
                phi_values,
                coordination_maps,
                l_fixed=l_fixed,
                l_values=l_values,
                style=PLOT_STYLE,
            )

    plot_inversion_boundary_multi_l(
        run_dir / "inversion_boundary_multi_l.png",
        boundary_rows,
        l_values=boundary_l_values,
        style=PLOT_STYLE,
    )

    theta_star = [row[0] for row in global_opt_rows]
    phi_star = [row[1] for row in global_opt_rows]
    l_star = [row[2] for row in global_opt_rows]
    q_star = [row[4] for row in global_opt_rows]

    plot_global_opt_vs_theta(
        run_dir / "global_opt_vs_theta.png",
        theta_star,
        phi_star,
        l_star,
        q_star,
        style=PLOT_STYLE,
    )
    plot_q_max_star_vs_theta(
        run_dir / "Q_max_star_vs_theta.png",
        theta_star,
        q_star,
        style=PLOT_STYLE,
    )
    plot_global_opt_trajectory_phi_l(
        run_dir / "global_opt_trajectory_phi_l.png",
        phi_star,
        l_star,
        theta_star,
        style=PLOT_STYLE,
    )


def run_experiment(
    *,
    mode: Exp08Mode = "fast",
    workers: int | None = None,
    exp_name: str = EXP_NAME,
    resolve_config: Callable[[str], tuple[dict, SolverConfig]] = resolve_exp08_config,
    include_constraint_force_in_Q: bool = EXP08_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    ic_solver_preset: dict[str, float | int | str] = EXP08_IC_SOLVER_PRESET,
    boundary_l_values: tuple[float, ...] = EXP08_BOUNDARY_L_VALUES,
    log_prefix: str = "exp08",
) -> Path:
    """
    theta 外側ループ + phi-l 掃引（Blakelet）を実行し、CSV と論文向け図を出力する。

    exp09 は include_constraint_force_in_Q=True と別 exp_name で本関数を再利用する。
    """
    if workers is None:
        workers = default_worker_count()

    start_time = time.perf_counter()
    sweep_defaults, solver_config = resolve_config(mode)

    mu = _as_float(sweep_defaults, "mu")
    a = _as_float(sweep_defaults, "a")
    k = _as_float(sweep_defaults, "k")
    F_0 = _as_float(sweep_defaults, "F_0")
    omega = _as_float(sweep_defaults, "omega")
    h = _as_float(sweep_defaults, "h")
    steady_n_periods = _as_int(sweep_defaults, "steady_n_periods")
    initial_condition_method = str(sweep_defaults["initial_condition_method"])
    if "include_constraint_force_in_Q" in sweep_defaults:
        include_constraint_force_in_Q = bool(
            sweep_defaults["include_constraint_force_in_Q"]
        )

    theta_min_deg = _as_float(sweep_defaults, "theta_min_deg")
    theta_step_deg = _as_float(sweep_defaults, "theta_step_deg")
    theta_max_deg = _as_float(sweep_defaults, "theta_max_deg")
    phi_min_deg = _as_float(sweep_defaults, "phi_min_deg")
    phi_step_deg = _as_float(sweep_defaults, "phi_step_deg")
    phi_max_deg = _as_float(sweep_defaults, "phi_max_deg")
    delta_min = _as_float(sweep_defaults, "delta_min")
    delta_max = _as_float(sweep_defaults, "delta_max")
    delta_points = _as_int(sweep_defaults, "delta_points")

    theta_values = _resolve_theta_values(sweep_defaults)
    phi_values = _resolve_phi_values(sweep_defaults)
    l_values = _resolve_l_values(sweep_defaults)
    delta_values = np.linspace(
        delta_min,
        delta_max,
        delta_points,
        endpoint=False,
        dtype=np.float64,
    )

    logging.getLogger(__name__).info(
        "%s: building initial position lookup (%s, workers=%d)",
        log_prefix,
        initial_condition_method,
        workers,
    )

    s1_0_lookup, s2_0_lookup = build_initial_position_lookup(
        delta_values,
        method=initial_condition_method,  # type: ignore[arg-type]
        phi_values=phi_values,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        h=h,
        ic_solver_config=SolverConfig(
            method=str(ic_solver_preset["method"]),
            rtol=float(ic_solver_preset["rtol"]),
            atol=float(ic_solver_preset["atol"]),
            n_periods=int(ic_solver_preset["n_periods"]),
            n_eval_per_period=int(ic_solver_preset["n_eval_per_period"]),
        ),
        ic_workers=workers,
        progress_desc=f"{log_prefix} IC lookup",
    )
    logging.getLogger(__name__).info(
        "%s: initial position lookup ready (%s, shape s1=%s, s2=%s)",
        log_prefix,
        initial_condition_method,
        s1_0_lookup.shape,
        s2_0_lookup.shape,
    )

    run_dir = make_run_directory(exp_name, base=OUTPUT_BASE)

    delta_opt_maps: list[np.ndarray] = []
    q_max_maps: list[np.ndarray] = []
    coordination_maps: list[np.ndarray] = []
    all_theta_rows: list[tuple[float, float, float, float, float, float, float, float]] = []
    global_opt_rows: list[tuple[float, float, float, float, float]] = []
    boundary_rows: list[tuple[float, float, float]] = []
    total_cases = 0

    for i_theta, layout_theta in enumerate(theta_values):
        theta_deg = float(np.degrees(layout_theta))
        logging.getLogger(__name__).info(
            "%s: sweeping theta=%.1f deg (%d/%d)",
            log_prefix,
            theta_deg,
            i_theta + 1,
            theta_values.size,
        )

        cases = _build_cases_exp08(
            phi_values=phi_values,
            l_values=l_values,
            delta_values=delta_values,
            s1_0_lookup=s1_0_lookup,
            s2_0_lookup=s2_0_lookup,
            layout_theta=float(layout_theta),
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
            h=h,
            solver_config=solver_config,
            steady_n_periods=steady_n_periods,
            include_constraint_force_in_Q=include_constraint_force_in_Q,
        )
        total_cases += len(cases)

        q_map = _sweep_q_map_exp08(
            cases=cases,
            workers=workers,
            q_map_shape=(phi_values.size, l_values.size, delta_values.size),
            progress_desc=f"{log_prefix} sweep",
        )
        delta_opt_map, q_max_map, coordination = _postprocess_maps(q_map, delta_values)

        delta_opt_maps.append(delta_opt_map)
        q_max_maps.append(q_max_map)
        coordination_maps.append(coordination)

        global_idx = np.unravel_index(int(np.argmax(q_map)), q_map.shape)
        i_phi_opt, i_l_opt, i_d_opt = global_idx
        global_opt_rows.append(
            (
                theta_deg,
                float(np.degrees(phi_values[i_phi_opt])),
                float(l_values[i_l_opt]),
                float(np.degrees(delta_values[i_d_opt])),
                float(q_map[global_idx]),
            )
        )

        boundary_rows.extend(
            extract_inversion_boundary(
                phi_values,
                l_values,
                coordination,
                theta_deg=theta_deg,
            )
        )

        phi_values_deg = np.degrees(phi_values)
        delta_opt_values_deg = np.degrees(delta_opt_map)
        for i_phi, (phi_rad, phi_deg) in enumerate(
            zip(phi_values, phi_values_deg, strict=True)
        ):
            for i_l, l in enumerate(l_values):
                all_theta_rows.append(
                    (
                        theta_deg,
                        float(phi_rad),
                        float(phi_deg),
                        float(l),
                        float(delta_opt_map[i_phi, i_l]),
                        float(delta_opt_values_deg[i_phi, i_l]),
                        float(q_max_map[i_phi, i_l]),
                        float(coordination[i_phi, i_l]),
                    )
                )

        _save_per_theta_outputs(
            run_dir / _theta_subdir_name(theta_deg),
            phi_values=phi_values,
            l_values=l_values,
            delta_opt_map=delta_opt_map,
            q_max_map=q_max_map,
            coordination=coordination,
        )

    np.savetxt(
        run_dir / "delta_opt_map_all_theta.csv",
        np.asarray(all_theta_rows, dtype=np.float64),
        delimiter=",",
        header=ALL_THETA_MAP_HEADER,
        comments="",
        encoding="utf-8",
    )
    np.savetxt(
        run_dir / "global_opt_vs_theta.csv",
        np.asarray(global_opt_rows, dtype=np.float64),
        delimiter=",",
        header=GLOBAL_OPT_HEADER,
        comments="",
        encoding="utf-8",
    )
    np.savetxt(
        run_dir / "inversion_boundary.csv",
        np.asarray(boundary_rows, dtype=np.float64),
        delimiter=",",
        header=INVERSION_BOUNDARY_HEADER,
        comments="",
        encoding="utf-8",
    )

    render_exp08_figures(
        run_dir,
        theta_values=theta_values,
        phi_values=phi_values,
        l_values=l_values,
        delta_opt_maps=delta_opt_maps,
        q_max_maps=q_max_maps,
        coordination_maps=coordination_maps,
        global_opt_rows=global_opt_rows,
        boundary_rows=boundary_rows,
        boundary_l_values=boundary_l_values,
    )

    elapsed_seconds = time.perf_counter() - start_time
    q_all = np.asarray([row[4] for row in global_opt_rows], dtype=np.float64)
    i_best_theta = int(np.argmax(q_all))

    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": exp_name,
            "mobility": "BlakeletTwoSliderMobility",
            "wall": "no-slip at z=0",
            "mode": mode,
            "workers": workers,
            "total_cases": total_cases,
            "include_constraint_force_in_Q": include_constraint_force_in_Q,
            "theta_sweep_deg": {
                "min": theta_min_deg,
                "max": theta_max_deg,
                "step": theta_step_deg,
                "points": theta_values.size,
            },
            "defaults": sweep_defaults,
            "solver": solver_config,
            "steady_n_periods": steady_n_periods,
            "initial_condition_method": initial_condition_method,
        },
    )

    summary_lines = [
        f"experiment: {exp_name}",
        f"mobility: Blakelet (wall z=0, Eq. 4.4)",
        f"mode: {mode}",
        f"workers: {workers}",
        f"total_cases: {total_cases}",
        f"include_constraint_force_in_Q: {include_constraint_force_in_Q}",
        f"integration_method: {solver_config.method}",
        f"n_periods: {solver_config.n_periods}",
        f"steady_n_periods: {steady_n_periods}",
        f"initial_condition_method: {initial_condition_method}",
        f"n_eval_per_period: {solver_config.n_eval_per_period}",
        f"theta_sweep [deg]: [{theta_min_deg:.6f}, {theta_max_deg:.6f}] "
        f"step={theta_step_deg:.6f}",
        f"theta_points: {theta_values.size}",
        f"phi_sweep [deg]: [{phi_min_deg:.6f}, {phi_max_deg:.6f}) step={phi_step_deg:.6f}",
        f"phi_points: {phi_values.size} (excludes {phi_max_deg:.6f} deg)",
        f"delta_range [rad]: [{delta_min:.8e}, {delta_max:.8e})",
        f"delta_points: {delta_points}",
        f"l_values: {', '.join(f'{float(l):.8e}' for l in l_values)}",
        f"l_points: {l_values.size}",
        f"global_best_theta [deg]: {global_opt_rows[i_best_theta][0]:.4f}",
        f"global_best_phi_star [deg]: {global_opt_rows[i_best_theta][1]:.4f}",
        f"global_best_l_star: {global_opt_rows[i_best_theta][2]:.8e}",
        f"global_best_delta_star [deg]: {global_opt_rows[i_best_theta][3]:.4f}",
        f"global_best_Q_max: {global_opt_rows[i_best_theta][4]:.8e}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"elapsed_minutes: {elapsed_seconds / 60.0:.3f}",
        "",
        "Note: phi is slider tilt angle. layout_theta is x-y layout angle (sweep axis).",
        "Note: per-theta outputs are under theta_XXX/ subdirectories.",
        "Note: include_constraint_force_in_Q=False uses F_x=f_total*cos(phi); "
        "True includes constraint force lambda in F_x (exp09).",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "exp08: theta sweep with phi-l maps "
            "(wall, Blakelet, Delta optimized per point)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "fine"),
        default=EXP08_DEFAULT_MODE,
        help=(
            "fast: coarse Delta grid + Euler. fine: dense grid + RK45. "
            f"Default from config: EXP08_DEFAULT_MODE={EXP08_DEFAULT_MODE!r}."
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
        "Starting exp08 sweep: mode=%s, workers=%d, theta=[%.1f, %.1f] step %.1f deg",
        args.mode,
        workers,
        EXP08_THETA_MIN_DEG,
        EXP08_THETA_MAX_DEG,
        EXP08_THETA_STEP_DEG,
    )
    run_dir = run_experiment(mode=args.mode, workers=workers)
    logging.getLogger(__name__).info("exp08 sweep finished. Output: %s", run_dir)


if __name__ == "__main__":
    main()
