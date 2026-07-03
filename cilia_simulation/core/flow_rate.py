"""
瞬時流量と周期平均流量 / Instantaneous and period-averaged flow rate.

【目的 / Purpose】
    スライダーが流体に与える x 方向力 F_x から、瞬時流量 q_x(t) と
    1 周期平均の正味流量 Q を評価する。

【構成 / Contents】
    1. FlowCalculator — 式(2.61) に基づく q_x, Q の計算
    2. QZeroCheckResult — 単一スライダーで Q≈0 の判定用

【理論対応 / Theory mapping】
    - 論文 式(2.61): q_x(t) = (h / (pi mu)) * F_x(t)
    - F_x(t) = f_total(t) * cos(phi)  … slider 側で定義
    - 周期平均: Q = (1/T) integral_{1 period} q_x(t) dt

【物理上の注意 / Physical note】
    式(2.61) は滑りなし壁 z=0 に対する Blakelet 遠方近似から導かれる。
    第1段階（自由空間・壁なし）では厳密には成立しないが、
    論文と同じ流量指標で第2・第3段階へ接続するため本式を用いる。
    第4段階で壁を導入すれば物理的にも正当化される。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.slider import Slider, SliderParameters
from core.solver import SimulationResult, TwoSliderResult

logger = logging.getLogger(__name__)

# ==========================================
# 1. 判定結果
# ==========================================


@dataclass(frozen=True)
class QZeroCheckResult:
    """
    Q≈0 検証の結果 / Result of net flow check for a single slider.

    Attributes
    ----------
    Q : float
        最後の1周期で評価した平均流量。
    reference_scale : float
        判定用のスケール F_0 * h / (pi mu)。
    tolerance : float
        許容幅 tol_factor * reference_scale。
    passed : bool
        |Q| < tolerance なら True。
  """

    Q: float
    reference_scale: float
    tolerance: float
    passed: bool


# ==========================================
# 2. FlowCalculator
# ==========================================


class FlowCalculator:
    """
    瞬時流量と周期平均流量を計算する / Flow rate from F_x.

    Parameters
    ----------
    mu : float
        粘性係数。
    h : float
        流量式の基準高さ（第1段階では幾何パラメータ兼スケール）。
    """

    def __init__(self, mu: float, h: float) -> None:
        if mu <= 0.0:
            raise ValueError("mu must be positive for flow rate calculation.")
        if h <= 0.0:
            raise ValueError("h must be positive for flow rate calculation.")
        self._mu = float(mu)
        self._h = float(h)
        self._prefactor = self._h / (math.pi * self._mu)

    @classmethod
    def from_slider_parameters(cls, params: SliderParameters) -> FlowCalculator:
        """
        SliderParameters から構築する / Build from SliderParameters.

        Parameters
        ----------
        params : SliderParameters
            mu, h を使用する。

        Returns
        -------
        FlowCalculator
        """
        return cls(mu=params.mu, h=params.h)

    @property
    def mu(self) -> float:
        """粘性係数 / dynamic viscosity."""
        return self._mu

    @property
    def h(self) -> float:
        """基準高さ / reference height."""
        return self._h

    @property
    def prefactor(self) -> float:
        """
        q_x = prefactor * F_x の係数 h/(pi mu) / Prefactor in Eq. (2.61).
        """
        return self._prefactor

    def instantaneous_q_x(
        self, F_x: ArrayLike
    ) -> NDArray[np.floating] | float:
        """
        瞬時流量を返す / Instantaneous flow rate in the x direction.

        論文 式(2.61): q_x(t) = (h / (pi mu)) * F_x(t)。

        第1段階では壁なしのため、本式は厳密な Blakelet 遠方場ではなく
        「論文と同一の流量定義」として用いる（第4段階で壁を入れて正当化）。

        Parameters
        ----------
        F_x : float or array_like
            x 方向力（slider.F_x または f_total * cos(phi)）。

        Returns
        -------
        float or ndarray
            瞬時流量 q_x(t)。
        """
        fx_arr = np.asarray(F_x, dtype=np.float64)
        q_x = self._prefactor * fx_arr
        if q_x.ndim == 0:
            return float(q_x)
        return q_x

    @staticmethod
    def F_x_from_f_total(
        f_total: ArrayLike, cos_phi: float
    ) -> NDArray[np.float64]:
        """
        直線方向合力から x 方向力を求める / F_x = f_total * cos(phi).

        Parameters
        ----------
        f_total : array_like
            直線方向合力の時系列。
        cos_phi : float
            傾き角の余弦（slider.cos_phi）。

        Returns
        -------
        ndarray
            F_x の時系列。
        """
        return np.asarray(f_total, dtype=np.float64) * float(cos_phi)

    def period_average_Q(
        self,
        t: ArrayLike,
        q_x: ArrayLike,
        period: float | None = None,
    ) -> float:
        """
        1 周期の時間平均流量 Q を返す / Period-averaged net flow Q.

        Q = (1/T) * integral q_x(t) dt。
        t, q_x は通常は SimulationResult の定常窓（最後の1周期）を渡す。

        Parameters
        ----------
        t : array_like
            時刻（単調増加、おおよそ1周期分）。
        q_x : array_like
            瞬時流量。
        period : float, optional
            周期 T。省略時は t[-1] - t[0] を使用する。

        Returns
        -------
        float
            平均流量 Q。
        """
        t_arr = np.asarray(t, dtype=np.float64)
        q_arr = np.asarray(q_x, dtype=np.float64)
        if t_arr.size < 2:
            raise ValueError("t must contain at least two samples for integration.")
        duration = float(t_arr[-1] - t_arr[0])
        if duration <= 0.0:
            raise ValueError("t must span a positive duration.")
        T = float(period) if period is not None else duration
        integral = float(np.trapezoid(q_arr, t_arr))
        return integral / T

    def compute_from_simulation(
        self,
        slider: Slider,
        result: SimulationResult,
        *,
        use_steady_window: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        積分結果から q_x 時系列と平均 Q を一括計算する / q_x series and Q from simulation.

        Parameters
        ----------
        slider : Slider
            cos(phi) の取得に使用。
        result : SimulationResult
            積分結果（t, f_total, period, steady 窓）。
        use_steady_window : bool
            True なら最後の1周期のみで Q を評価する。

        Returns
        -------
        t_used, q_x, Q
            使用した時刻、瞬時流量、周期平均流量。
        """
        if use_steady_window:
            t_used = np.asarray(result.t_steady, dtype=np.float64)
            f_used = np.asarray(result.f_total_steady, dtype=np.float64)
        else:
            t_used = np.asarray(result.t, dtype=np.float64)
            f_used = np.asarray(result.f_total, dtype=np.float64)

        F_x = self.F_x_from_f_total(f_used, slider.cos_phi)
        q_x = np.asarray(self.instantaneous_q_x(F_x), dtype=np.float64)
        Q = self.period_average_Q(t_used, q_x, period=result.period)
        return t_used, q_x, Q

    def check_Q_near_zero(
        self,
        Q: float,
        F_0: float,
        *,
        tol_factor: float = 1e-4,
    ) -> QZeroCheckResult:
        """
        単一スライダーで Q≈0 かどうかを判定する / Check |Q| < tol * F_0 h/(pi mu).

        成功基準（第1段階）: |Q| < 1e-4 * F_0 * h / (pi mu)。

        Parameters
        ----------
        Q : float
            評価した平均流量。
        F_0 : float
            駆動力振幅。
        tol_factor : float
            許容倍率（デフォルト 1e-4）。

        Returns
        -------
        QZeroCheckResult
        """
        reference_scale = float(F_0) * self._prefactor
        tolerance = float(tol_factor) * reference_scale
        passed = abs(Q) < tolerance
        logger.info(
            "Q check: Q=%.6e, scale=%.6e, tol=%.6e, passed=%s",
            Q,
            reference_scale,
            tolerance,
            passed,
        )
        return QZeroCheckResult(
            Q=Q,
            reference_scale=reference_scale,
            tolerance=tolerance,
            passed=passed,
        )

    def instantaneous_q_x_two_slider(
        self,
        *,
        s1: ArrayLike,
        s2: ArrayLike,
        f1_total: ArrayLike,
        f2_total: ArrayLike,
        phi: float,
    ) -> NDArray[np.float64]:
        """
        2スライダーの瞬時流量 q_x(t) を返す。

        Theory
        ------
        論文式(2.61),(2.62) と 4章の式(4.65)に対応させて、
        壁高さを z_i = h - s_i sin(phi) とし、

            q_x(t) = (1/(pi*mu)) * [ z_1*F_{x,1} + z_2*F_{x,2} ]
            F_{x,i} = f_{i,total} * cos(phi)

        で評価する。
        """
        s1_arr = np.asarray(s1, dtype=np.float64)
        s2_arr = np.asarray(s2, dtype=np.float64)
        f1_arr = np.asarray(f1_total, dtype=np.float64)
        f2_arr = np.asarray(f2_total, dtype=np.float64)
        cos_phi = float(np.cos(phi))
        sin_phi = float(np.sin(phi))

        z1 = self._h - s1_arr * sin_phi
        z2 = self._h - s2_arr * sin_phi
        Fx1 = f1_arr * cos_phi
        Fx2 = f2_arr * cos_phi
        return (z1 * Fx1 + z2 * Fx2) / (math.pi * self._mu)

    def compute_two_slider_Q_from_result(
        self,
        *,
        result: TwoSliderResult,
        phi: float,
        use_steady_window: bool = True,
        steady_n_periods: int = 1,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        2スライダー積分結果から q_x(t) と周期平均 Q を返す。

        Parameters
        ----------
        result : TwoSliderResult
            2スライダー積分結果。
        phi : float
            スライダー傾き角。
        use_steady_window : bool
            True の場合は定常窓で評価する。
        steady_n_periods : int
            定常窓として使う周期数。1 のとき最後の1周期、
            n > 1 のとき最後 n 周期の時間平均 Q = integral / (n T)。
        """
        if steady_n_periods < 1:
            raise ValueError("steady_n_periods must be at least 1.")

        if use_steady_window and steady_n_periods == 1:
            t_used = np.asarray(result.t_steady, dtype=np.float64)
            s1_used = np.asarray(result.s1_steady, dtype=np.float64)
            s2_used = np.asarray(result.s2_steady, dtype=np.float64)
            f1_used = np.asarray(result.f1_total[result.steady_start_index :], dtype=np.float64)
            f2_used = np.asarray(result.f2_total[result.steady_start_index :], dtype=np.float64)
            averaging_period = result.period
        elif use_steady_window:
            t_end = float(result.t[-1])
            t_start = t_end - steady_n_periods * result.period
            mask = result.t >= t_start
            t_used = np.asarray(result.t[mask], dtype=np.float64)
            s1_used = np.asarray(result.s1[mask], dtype=np.float64)
            s2_used = np.asarray(result.s2[mask], dtype=np.float64)
            f1_used = np.asarray(result.f1_total[mask], dtype=np.float64)
            f2_used = np.asarray(result.f2_total[mask], dtype=np.float64)
            averaging_period = steady_n_periods * result.period
        else:
            t_used = np.asarray(result.t, dtype=np.float64)
            s1_used = np.asarray(result.s1, dtype=np.float64)
            s2_used = np.asarray(result.s2, dtype=np.float64)
            f1_used = np.asarray(result.f1_total, dtype=np.float64)
            f2_used = np.asarray(result.f2_total, dtype=np.float64)
            averaging_period = result.period

        q_x = self.instantaneous_q_x_two_slider(
            s1=s1_used,
            s2=s2_used,
            f1_total=f1_used,
            f2_total=f2_used,
            phi=phi,
        )
        Q = self.period_average_Q(t_used, q_x, period=averaging_period)
        return t_used, np.asarray(q_x, dtype=np.float64), Q