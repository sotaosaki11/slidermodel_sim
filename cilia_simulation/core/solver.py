"""
時間積分：スライダー運動方程式 / Time integration for the slider ODE.

【目的 / Purpose】
    ds_1/dt = gamma_0 * f_total(t, s_1) を数値積分し、軌道 s_1(t) を求める。
    過渡を捨て、最後の1周期を定常解析（流量平均など）に使えるようにする。

【構成 / Contents】
    1. SolverConfig — 積分手法・周期数・許容誤差など
    2. SimulationResult — 時系列と定常窓のインデックス
    3. TimeStepper — solve_ivp (RK45) または前進 Euler

【理論対応 / Theory mapping】
    - 運動方程式: ds_1/dt = gamma_0 * (F_0 cos(omega t) - k s_1)
    - 定常評価: 最低 n_periods 周期（デフォルト5）、最後1周期を steady window とする
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from core.hydrodynamics import Mobility, TwoSliderMobility
from core.slider import Slider

logger = logging.getLogger(__name__)

# ==========================================
# 1. 設定と結果のデータクラス
# ==========================================


@dataclass(frozen=True)
class SolverConfig:
    """
    時間積分の設定 / Time integration settings.

    Attributes
    ----------
    method : str
        積分手法。"RK45" は scipy solve_ivp の Runge-Kutta。
        "euler" は前進オイラー（デバッグ・理解用）。
    rtol, atol : float
        solve_ivp の相対・絶対誤差許容（euler では未使用）。
    n_periods : int
        積分する総周期数。過渡除去のため >= 5 を推奨。
    n_eval_per_period : int
        1周期あたりの出力点数（Euler のステップ数にも使用）。
    t_start : float
        積分開始時刻（無次元）。
    """

    method: str = "RK45"
    rtol: float = 1e-8
    atol: float = 1e-10
    n_periods: int = 5
    n_eval_per_period: int = 200
    t_start: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    """
    積分結果 / Integration output.

    Attributes
    ----------
    t, s, f_total : ndarray
        時刻、位置 s_1、直線方向合力の時系列。
    omega, period : float
        角振動数と周期 T = 2 pi / omega。
    n_periods : int
        積分した総周期数。
    steady_start_index : int
        最後の1周期の開始インデックス（このインデックス以降を定常窓とする）。
    """

    t: NDArray[np.float64]
    s: NDArray[np.float64]
    f_total: NDArray[np.float64]
    omega: float
    period: float
    n_periods: int
    steady_start_index: int

    @property
    def t_steady(self) -> NDArray[np.float64]:
        """最後の1周期分の時刻 / Time samples in the last period."""
        return self.t[self.steady_start_index :]

    @property
    def s_steady(self) -> NDArray[np.float64]:
        """最後の1周期分の位置 / Position in the last period."""
        return self.s[self.steady_start_index :]

    @property
    def f_total_steady(self) -> NDArray[np.float64]:
        """最後の1周期分の合力 / Total force in the last period."""
        return self.f_total[self.steady_start_index :]

    @property
    def steady_start_time(self) -> float:
        """定常窓の開始時刻 / Start time of the steady window."""
        return float(self.t[self.steady_start_index])


# ==========================================
# 2. TimeStepper
# ==========================================


class TimeStepper:
    """
    単一スライダーの運動方程式を積分する / Integrate one slider ODE.

    Parameters
    ----------
    slider : Slider
        力 f_total を提供するスライダー。
    mobility : Mobility
        gamma_0 と velocity_from_force を提供する。
    config : SolverConfig
        積分設定。
    """

    def __init__(
        self,
        slider: Slider,
        mobility: Mobility,
        config: SolverConfig | None = None,
    ) -> None:
        self.slider = slider
        self.mobility = mobility
        self.config = config if config is not None else SolverConfig()
        self._omega = float(slider.params.omega)
        if self._omega <= 0.0:
            raise ValueError("omega must be positive.")
        self._period = 2.0 * math.pi / self._omega

    @property
    def period(self) -> float:
        """周期 T = 2 pi / omega / Oscillation period."""
        return self._period

    def run(self) -> SimulationResult:
        """
        積分を実行し結果を返す / Run integration and return trajectories.

        Returns
        -------
        SimulationResult
            全期間の時系列と、最後1周期の開始インデックス。

        Raises
        ------
        ValueError
            未知の積分手法、または n_periods < 1。
        RuntimeError
            solve_ivp が失敗したとき。
        """
        cfg = self.config
        if cfg.n_periods < 1:
            raise ValueError("n_periods must be at least 1.")

        t_end = cfg.t_start + cfg.n_periods * self._period
        n_eval = cfg.n_periods * cfg.n_eval_per_period + 1
        t_eval = np.linspace(cfg.t_start, t_end, n_eval)

        method = cfg.method.upper()
        if method == "EULER":
            t, s = self._integrate_euler(t_eval)
        elif method == "RK45":
            t, s = self._integrate_rk45(t_eval, cfg.rtol, cfg.atol)
        else:
            raise ValueError(f"Unknown integration method: {cfg.method}")

        self.slider.set_state(float(s[-1]))
        f_total = self._compute_f_total_series(t, s)
        steady_start_index = self._steady_start_index(t, cfg.n_periods)

        logger.info(
            "Integration finished: method=%s, periods=%d, t_end=%.4f, steady_from t=%.4f",
            cfg.method,
            cfg.n_periods,
            float(t[-1]),
            float(t[steady_start_index]),
        )

        return SimulationResult(
            t=t,
            s=s,
            f_total=f_total,
            omega=self._omega,
            period=self._period,
            n_periods=cfg.n_periods,
            steady_start_index=steady_start_index,
        )

    def _rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        運動方程式の右辺 / RHS of ds_1/dt = gamma_0 * f_total.

        Parameters
        ----------
        t : float
            時刻。
        y : ndarray, shape (1,)
            状態 [s_1]。

        Returns
        -------
        ndarray, shape (1,)
            [ds_1/dt]。
        """
        self.slider.set_state(float(y[0]))
        f = self.slider.f_total(t)
        if isinstance(f, float):
            ds_dt = self.mobility.velocity_from_force(f)
        else:
            ds_dt = float(self.mobility.velocity_from_force(float(f)))
        return np.array([ds_dt], dtype=np.float64)

    def _integrate_rk45(
        self,
        t_eval: NDArray[np.float64],
        rtol: float,
        atol: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        scipy solve_ivp (RK45) で積分する / Integrate with adaptive RK45.

        RK45 を第一候補とする理由:
        線形＋周期駆動で滑らかな ODE のため、適応刻みで精度とコストのバランスが良い。
        """
        y0 = np.array([self.slider.s], dtype=np.float64)
        sol = solve_ivp(
            self._rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.t, sol.y[0]

    def _integrate_euler(
        self,
        t_eval: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        前進オイラー法で積分する / Forward Euler (fixed steps on t_eval grid).

        理解・デバッグ用。精度は RK45 より劣る。
        """
        n = len(t_eval)
        s = np.empty(n, dtype=np.float64)
        s[0] = self.slider.s
        self.slider.set_state(s[0])

        for i in range(n - 1):
            t_i = float(t_eval[i])
            dt = float(t_eval[i + 1] - t_eval[i])
            f = self.slider.f_total(t_i)
            ds_dt = self.mobility.velocity_from_force(f)
            if not isinstance(ds_dt, float):
                ds_dt = float(ds_dt)
            s[i + 1] = s[i] + dt * ds_dt
            self.slider.set_state(s[i + 1])

        return t_eval, s

    def _compute_f_total_series(
        self,
        t: NDArray[np.float64],
        s: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        軌道に対応する f_total 時系列をベクトル化で計算する / Total force along trajectory.

        f_total = F_0 cos(omega t) - k s は t,s から直接評価できる。
        """
        p = self.slider.params
        f_active = p.F_0 * np.cos(p.omega * t)
        f_spring = -p.k * s
        return f_active + f_spring

    def _steady_start_index(
        self,
        t: NDArray[np.float64],
        n_periods: int,
    ) -> int:
        """
        最後の1周期の開始インデックスを返す / Index where the last period begins.

        開始時刻は t_start + (n_periods - 1) * T。
        """
        cfg = self.config
        t_steady = cfg.t_start + (n_periods - 1) * self._period
        index = int(np.searchsorted(t, t_steady, side="left"))
        if index >= len(t):
            index = max(0, len(t) - 1)
        return index


# ==========================================
# 3. TwoSliderTimeStepper（exp02）
# ==========================================


@dataclass(frozen=True)
class TwoSliderResult:
    """
    2スライダー積分結果 / Integration output for exp02.

    Attributes
    ----------
    t : ndarray
        時刻列。
    s1, s2 : ndarray
        各スライダーの線方向座標。
    f1_total, f2_total : ndarray
        各スライダーの直線方向合力。
    period : float
        駆動周期 T=2*pi/omega。
    n_periods : int
        積分した周期数。
    steady_start_index : int
        最後の1周期開始インデックス。
    """

    t: NDArray[np.float64]
    s1: NDArray[np.float64]
    s2: NDArray[np.float64]
    f1_total: NDArray[np.float64]
    f2_total: NDArray[np.float64]
    period: float
    n_periods: int
    steady_start_index: int

    @property
    def t_steady(self) -> NDArray[np.float64]:
        return self.t[self.steady_start_index :]

    @property
    def s1_steady(self) -> NDArray[np.float64]:
        return self.s1[self.steady_start_index :]

    @property
    def s2_steady(self) -> NDArray[np.float64]:
        return self.s2[self.steady_start_index :]


class TwoSliderTimeStepper:
    """
    拘束付き2スライダー ODE を積分する（exp02）。

    Notes
    -----
    状態は y=[s1, s2]。速度は TwoSliderMobility.compute_velocities(...) で評価する。
    """

    def __init__(
        self,
        *,
        mobility: TwoSliderMobility,
        omega: float,
        k: float,
        F_0: float,
        phi: float,
        l: float,
        h: float,
        delta: float,
        s1_0: float = 0.0,
        s2_0: float = 0.0,
        config: SolverConfig | None = None,
    ) -> None:
        if omega <= 0.0:
            raise ValueError("omega must be positive.")
        if l <= 0.0:
            raise ValueError("l must be positive.")
        if h < 0.0:
            raise ValueError("h must be non-negative.")
        self.mobility = mobility
        self.omega = float(omega)
        self.k = float(k)
        self.F_0 = float(F_0)
        self.phi = float(phi)
        self.l = float(l)
        self.h = float(h)
        self.delta = float(delta)
        self.s1_0 = float(s1_0)
        self.s2_0 = float(s2_0)
        self.config = config if config is not None else SolverConfig()
        self._period = 2.0 * math.pi / self.omega

        self._e_s = np.array(
            [math.cos(self.phi), 0.0, -math.sin(self.phi)],
            dtype=np.float64,
        )
        self._e_s_perp = np.array(
            [math.sin(self.phi), 0.0, math.cos(self.phi)],
            dtype=np.float64,
        )
        # 論文式(4.1)(4.2)に合わせる: x 基準は ±l/2, z 基準は h。
        self._r1_base = np.array([-0.5 * self.l, 0.0, self.h], dtype=np.float64)
        self._r2_base = np.array([0.5 * self.l, 0.0, self.h], dtype=np.float64)

    @property
    def period(self) -> float:
        """駆動周期 T=2*pi/omega."""
        return self._period

    def _positions_from_state(
        self,
        s1: float,
        s2: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r1 = self._r1_base + s1 * self._e_s
        r2 = self._r2_base + s2 * self._e_s
        return r1, r2

    def _active_forces(self, t: float) -> tuple[float, float]:
        # 論文式(4.3)の位相規約: F1 = F0 cos(omega t + Delta), F2 = F0 cos(omega t)
        f_active1 = self.F_0 * math.cos(self.omega * t + self.delta)
        f_active2 = self.F_0 * math.cos(self.omega * t)
        return f_active1, f_active2

    def _rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        s1 = float(y[0])
        s2 = float(y[1])
        r1, r2 = self._positions_from_state(s1, s2)
        f_active1, f_active2 = self._active_forces(t)
        ds1_dt, ds2_dt = self.mobility.compute_velocities(
            s1=s1,
            s2=s2,
            r1=r1,
            r2=r2,
            e_s=self._e_s,
            e_s_perp=self._e_s_perp,
            f_active1=f_active1,
            f_active2=f_active2,
            k=self.k,
        )
        return np.array([ds1_dt, ds2_dt], dtype=np.float64)

    def run(self) -> TwoSliderResult:
        cfg = self.config
        if cfg.n_periods < 1:
            raise ValueError("n_periods must be at least 1.")

        t_end = cfg.t_start + cfg.n_periods * self._period
        n_eval = cfg.n_periods * cfg.n_eval_per_period + 1
        t_eval = np.linspace(cfg.t_start, t_end, n_eval)

        y0 = np.array([self.s1_0, self.s2_0], dtype=np.float64)
        method = cfg.method.upper()
        if method == "RK45":
            sol = solve_ivp(
                self._rhs,
                (float(t_eval[0]), float(t_eval[-1])),
                y0,
                method="RK45",
                t_eval=t_eval,
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            t = np.asarray(sol.t, dtype=np.float64)
            s1 = np.asarray(sol.y[0], dtype=np.float64)
            s2 = np.asarray(sol.y[1], dtype=np.float64)
        elif method == "EULER":
            t = t_eval
            s1 = np.empty_like(t)
            s2 = np.empty_like(t)
            s1[0], s2[0] = self.s1_0, self.s2_0
            for i in range(len(t) - 1):
                dt = float(t[i + 1] - t[i])
                dydt = self._rhs(float(t[i]), np.array([s1[i], s2[i]], dtype=np.float64))
                s1[i + 1] = s1[i] + dt * float(dydt[0])
                s2[i + 1] = s2[i] + dt * float(dydt[1])
        else:
            raise ValueError(f"Unknown integration method: {cfg.method}")

        f_active1 = self.F_0 * np.cos(self.omega * t + self.delta)
        f_active2 = self.F_0 * np.cos(self.omega * t)
        f1_total = f_active1 - self.k * s1
        f2_total = f_active2 - self.k * s2

        t_steady = cfg.t_start + (cfg.n_periods - 1) * self._period
        steady_start_index = int(np.searchsorted(t, t_steady, side="left"))
        if steady_start_index >= len(t):
            steady_start_index = max(0, len(t) - 1)

        return TwoSliderResult(
            t=t,
            s1=s1,
            s2=s2,
            f1_total=np.asarray(f1_total, dtype=np.float64),
            f2_total=np.asarray(f2_total, dtype=np.float64),
            period=self._period,
            n_periods=cfg.n_periods,
            steady_start_index=steady_start_index,
        )
