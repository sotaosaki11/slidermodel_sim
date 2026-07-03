"""
非結合スライダーから 2 スライダー初期位置を導出する / Decoupled initial conditions.

exp08 では壁あり Blakelet 自己移動度で単独スライダーを積分し定常周期解の
t=0 位置を warm-start に使う。検証・比較用に自由空間スカラー gamma_0 の
解析式も残す。

駆動位相規約（論文式 4.3）:
    F_1 = F_0 cos(omega t + Delta),  F_2 = F_0 cos(omega t)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from core.hydrodynamics import compute_single_slider_blakelet_ds_dt
from core.solver import SolverConfig
from core.sweep import sweep_with_progress

logger = logging.getLogger(__name__)

InitialConditionMethod = Literal["decoupled_analytic", "decoupled_blakelet"]
IC_LOOKUP_PROGRESS_ALPHA = 0.2


@dataclass(frozen=True)
class _IcLookupCase:
    """Blakelet IC ルックアップ 1 ケース。ProcessPoolExecutor で pickle する。"""

    kind: Literal["s1", "s2"]
    i_phi: int
    i_d: int
    drive_phase: float
    phi: float
    h: float
    mu: float
    a: float
    k: float
    F_0: float
    omega: float
    solver_config: SolverConfig


def _run_one_ic_lookup_case(case: _IcLookupCase) -> tuple[str, int, int, float]:
    """ワーカープロセスから呼ばれる 1 ケース Blakelet 単独スライダー積分。"""
    s_value = compute_blakelet_single_slider_steady_s_at_t0(
        drive_phase=case.drive_phase,
        phi=case.phi,
        h=case.h,
        mu=case.mu,
        a=case.a,
        k=case.k,
        F_0=case.F_0,
        omega=case.omega,
        solver_config=case.solver_config,
    )
    return case.kind, case.i_phi, case.i_d, s_value


def _ic_progress_kwargs_for_workers(workers: int) -> dict:
    if workers > 1:
        return {
            "update_every_cases": 4,
            "mininterval": 0.5,
            "eta_alpha": 0.08,
            "eta_min_cases": 10,
        }
    return {
        "update_every_cases": 1,
        "mininterval": 0.5,
        "eta_alpha": 0.08,
        "eta_min_cases": 5,
    }


def scalar_gamma_0(*, mu: float, a: float) -> float:
    """非結合スライダーの自由空間スカラー移動度 gamma_0 = 1/(6 pi mu a)。"""
    if mu <= 0.0 or a <= 0.0:
        raise ValueError("mu and a must be positive.")
    return 1.0 / (6.0 * math.pi * mu * a)


def decoupled_steady_amplitude_and_lag(
    *,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
) -> tuple[float, float]:
    """
    自由空間・線形定常解の振幅 A と位相遅れ delta_lag を返す。

    A = F_0 / sqrt(k^2 + (omega/gamma_0)^2)
    delta_lag = arctan(omega / (k * gamma_0))
    """
    gamma_0 = scalar_gamma_0(mu=mu, a=a)
    A = F_0 / math.sqrt(k**2 + (omega / gamma_0) ** 2)
    delta_lag = math.atan(omega / (k * gamma_0))
    return A, delta_lag


def compute_decoupled_analytic_initial_positions(
    *,
    delta: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
) -> tuple[float, float]:
    """
    自由空間スカラー gamma_0 の解析式から (s1_0, s2_0) を返す（比較用）。

    s1_0 = A cos(Delta - delta_lag)   # 駆動 cos(omega t + Delta)
    s2_0 = A cos(delta_lag)             # 駆動 cos(omega t)
    """
    A, delta_lag = decoupled_steady_amplitude_and_lag(
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
    )
    s1_0 = A * math.cos(delta - delta_lag)
    s2_0 = A * math.cos(delta_lag)
    return s1_0, s2_0


def compute_single_slider_steady_s_at_t0_stokes(
    *,
    drive_phase: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    solver_config: SolverConfig | None = None,
) -> float:
    """
    自由空間スカラー gamma_0 の非結合単独スライダーを数値積分し、
    定常窓先頭の s を返す（解析式との比較検証用）。
    """
    if solver_config is None:
        solver_config = SolverConfig(
            method="RK45",
            n_periods=20,
            n_eval_per_period=1000,
        )
    gamma_0 = scalar_gamma_0(mu=mu, a=a)
    period = 2.0 * math.pi / omega
    t_end = solver_config.t_start + solver_config.n_periods * period
    n_eval = solver_config.n_periods * solver_config.n_eval_per_period + 1
    t_eval = np.linspace(solver_config.t_start, t_end, n_eval)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        s = float(y[0])
        f_active = F_0 * math.cos(omega * t + drive_phase)
        return np.array([gamma_0 * (f_active - k * s)], dtype=np.float64)

    sol = solve_ivp(
        rhs,
        (solver_config.t_start, t_end),
        np.array([0.0], dtype=np.float64),
        t_eval=t_eval,
        method="RK45",
        rtol=solver_config.rtol,
        atol=solver_config.atol,
    )
    if not sol.success:
        raise RuntimeError(f"single-slider Stokes integration failed: {sol.message}")

    t_steady = solver_config.t_start + (solver_config.n_periods - 1) * period
    steady_index = int(np.searchsorted(sol.t, t_steady, side="left"))
    return float(sol.y[0, steady_index])


def compute_blakelet_single_slider_steady_s_at_t0(
    *,
    drive_phase: float,
    phi: float,
    h: float,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    solver_config: SolverConfig | None = None,
) -> float:
    """
    壁あり Blakelet 自己移動度の非結合単独スライダーを積分し、
    定常窓先頭（周期境界 t=0 相当）の s を返す。
    """
    if solver_config is None:
        solver_config = SolverConfig(
            method="RK45",
            n_periods=20,
            n_eval_per_period=1000,
        )
    period = 2.0 * math.pi / omega
    t_end = solver_config.t_start + solver_config.n_periods * period
    n_eval = solver_config.n_periods * solver_config.n_eval_per_period + 1
    t_eval = np.linspace(solver_config.t_start, t_end, n_eval)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        s = float(y[0])
        ds_dt = compute_single_slider_blakelet_ds_dt(
            s=s,
            t=t,
            drive_phase=drive_phase,
            phi=phi,
            h=h,
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
        )
        return np.array([ds_dt], dtype=np.float64)

    sol = solve_ivp(
        rhs,
        (solver_config.t_start, t_end),
        np.array([0.0], dtype=np.float64),
        t_eval=t_eval,
        method="RK45",
        rtol=solver_config.rtol,
        atol=solver_config.atol,
    )
    if not sol.success:
        raise RuntimeError(f"single-slider Blakelet integration failed: {sol.message}")

    t_steady = solver_config.t_start + (solver_config.n_periods - 1) * period
    steady_index = int(np.searchsorted(sol.t, t_steady, side="left"))
    return float(sol.y[0, steady_index])


def _build_analytic_lookup(
    delta_values: NDArray[np.float64],
    *,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    phi_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """自由空間解析式ルックアップ。s1: (n_phi, n_delta), s2: (n_phi,)。"""
    s1_1d = np.empty(delta_values.size, dtype=np.float64)
    s2_at_zero = compute_decoupled_analytic_initial_positions(
        delta=0.0,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
    )[1]
    for i, delta in enumerate(delta_values):
        s1_1d[i], _ = compute_decoupled_analytic_initial_positions(
            delta=float(delta),
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
        )
    s1_lookup = np.broadcast_to(s1_1d, (phi_values.size, delta_values.size)).copy()
    s2_lookup = np.full(phi_values.size, s2_at_zero, dtype=np.float64)
    return s1_lookup, s2_lookup


def _build_ic_lookup_cases(
    delta_values: NDArray[np.float64],
    *,
    phi_values: NDArray[np.float64],
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    h: float,
    ic_solver_config: SolverConfig,
) -> list[_IcLookupCase]:
    cases: list[_IcLookupCase] = []
    for i_phi, phi in enumerate(phi_values):
        cases.append(
            _IcLookupCase(
                kind="s2",
                i_phi=i_phi,
                i_d=-1,
                drive_phase=0.0,
                phi=float(phi),
                h=h,
                mu=mu,
                a=a,
                k=k,
                F_0=F_0,
                omega=omega,
                solver_config=ic_solver_config,
            )
        )
        for i_d, delta in enumerate(delta_values):
            cases.append(
                _IcLookupCase(
                    kind="s1",
                    i_phi=i_phi,
                    i_d=i_d,
                    drive_phase=float(delta),
                    phi=float(phi),
                    h=h,
                    mu=mu,
                    a=a,
                    k=k,
                    F_0=F_0,
                    omega=omega,
                    solver_config=ic_solver_config,
                )
            )
    return cases


def _build_blakelet_lookup(
    delta_values: NDArray[np.float64],
    *,
    phi_values: NDArray[np.float64],
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    h: float,
    ic_solver_config: SolverConfig,
    ic_workers: int = 1,
    progress_desc: str = "IC lookup",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Blakelet 単独スライダー積分ルックアップ。s1: (n_phi, n_delta), s2: (n_phi,)。"""
    s1_lookup = np.empty((phi_values.size, delta_values.size), dtype=np.float64)
    s2_lookup = np.empty(phi_values.size, dtype=np.float64)
    cases = _build_ic_lookup_cases(
        delta_values,
        phi_values=phi_values,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        h=h,
        ic_solver_config=ic_solver_config,
    )

    def _store_result(result: tuple[str, int, int, float]) -> None:
        kind, i_phi, i_d, s_value = result
        if kind == "s2":
            s2_lookup[i_phi] = s_value
        else:
            s1_lookup[i_phi, i_d] = s_value

    sweep_with_progress(
        cases=cases,
        worker_fn=_run_one_ic_lookup_case,
        workers=ic_workers,
        on_result=_store_result,
        desc=progress_desc,
        alpha=IC_LOOKUP_PROGRESS_ALPHA,
        progress_kwargs=_ic_progress_kwargs_for_workers(ic_workers),
    )
    logger.info(
        "IC lookup (Blakelet) finished: %d integrations (%s)",
        len(cases),
        progress_desc,
    )
    return s1_lookup, s2_lookup


def build_initial_position_lookup(
    delta_values: NDArray[np.float64] | np.ndarray,
    *,
    method: InitialConditionMethod,
    phi_values: NDArray[np.float64] | np.ndarray,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    h: float,
    ic_solver_config: SolverConfig | None = None,
    ic_workers: int = 1,
    progress_desc: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Delta×phi グリッドの (s1_0, s2_0) ルックアップを生成する。

    Parameters
    ----------
    ic_workers : int
        Blakelet 数値ルックアップの並列ワーカー数。1 で直列。
    progress_desc : str, optional
        進捗表示の説明。Blakelet ルックアップ時のみ使用。

    Returns
    -------
    s1_lookup : ndarray, shape (n_phi, n_delta)
    s2_lookup : ndarray, shape (n_phi,)
    """
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    phi_arr = np.asarray(phi_values, dtype=np.float64)
    if delta_arr.ndim != 1 or phi_arr.ndim != 1:
        raise ValueError("delta_values and phi_values must be 1-D arrays.")
    if delta_arr.size == 0 or phi_arr.size == 0:
        raise ValueError("delta_values and phi_values must be non-empty.")

    if method == "decoupled_analytic":
        return _build_analytic_lookup(
            delta_arr,
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
            phi_values=phi_arr,
        )
    if method == "decoupled_blakelet":
        if ic_solver_config is None:
            ic_solver_config = SolverConfig(
                method="RK45",
                n_periods=20,
                n_eval_per_period=1000,
            )
        lookup_desc = progress_desc if progress_desc is not None else "IC lookup"
        if ic_workers < 1:
            raise ValueError("ic_workers must be at least 1.")
        return _build_blakelet_lookup(
            delta_arr,
            phi_values=phi_arr,
            mu=mu,
            a=a,
            k=k,
            F_0=F_0,
            omega=omega,
            h=h,
            ic_solver_config=ic_solver_config,
            ic_workers=ic_workers,
            progress_desc=lookup_desc,
        )
    raise ValueError(
        f"Unknown initial condition method: {method!r}. "
        "Use 'decoupled_analytic' or 'decoupled_blakelet'."
    )


# 後方互換エイリアス
compute_decoupled_initial_positions = compute_decoupled_analytic_initial_positions
compute_single_slider_steady_s_at_t0_numerical = compute_single_slider_steady_s_at_t0_stokes
