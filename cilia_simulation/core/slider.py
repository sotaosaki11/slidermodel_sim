"""
スライダー（ビーズ）の状態と力 / Slider state and forces on a bead.

【目的 / Purpose】
    1次元スライダーモデルにおける単一ビーズの幾何（傾き角 phi）、
    スカラー位置 s_1(t)、および直線方向・x 方向の力を定義する。
    第2段階以降は複数 Slider インスタンスを並べ、流体力は hydrodynamics 側で扱う。

【構成 / Contents】
    1. SliderParameters — パラメータの型定義（実数は config/default_params.py）
    2. Slider — 位置状態と各力の計算

【主要クラス / Main classes】
    SliderParameters
        無次元パラメータ一式（論文2.4節に準拠した無次元化を想定）。
    Slider
        f_active, f_spring, f_total, F_x を提供する。

【理論対応 / Theory mapping】
    - 駆動力: 論文 式(4.3)  … f_active = F_0 cos(omega t)
    - 復元力: 線形ばね      … f_spring = -k s_1
    - x 方向力: F_x = f_total cos(phi)（流量式(2.61)の入力）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ==========================================
# 1. パラメータ型定義
# ==========================================


@dataclass(frozen=True)
class SliderParameters:
    """
    単一スライダーの無次元パラメータ / Dimensionless parameters for one slider.

    実数のデフォルト値は config/default_params.py の EXP01_DEFAULTS 等で管理する。
    本 dataclass はフィールド定義のみを担う。

    Attributes
    ----------
    a : float
        ビーズ半径 / bead radius（移動度 gamma_0 の計算に使用）。
    mu : float
        粘性係数 / dynamic viscosity。
    k : float
        ばね定数 / spring constant（復元力 -k s_1）。
    F_0 : float
        能動力の振幅 / active force amplitude（論文 式(4.3)）。
    omega : float
        角振動数 / angular frequency [rad/time]（無次元時間あたり）。
    phi : float
        スライダー直線の傾き角 [rad]。e_s = (cos phi, 0, -sin phi)。
    h : float
        流量計算の基準高さ / reference height（flow_rate モジュールで使用）。
    s0 : float
        初期位置 s_1(0) / initial position along the slider line。
    """

    a: float
    mu: float
    k: float
    F_0: float
    omega: float
    phi: float
    h: float
    s0: float = 0.0


# ==========================================
# 2. Slider クラス
# ==========================================


class Slider:
    """
    直線軌道上的に運動する1つのスライダー（ビーズ） / One bead on a tilted line.

    直線方向の単位ベクトルは e_s = (cos phi, 0, -sin phi)。
    スカラー座標 s_1 は直線中心からの符号付き距離を表す。
    第1段階では hydrodynamics が ds_1/dt = gamma_0 * f_total を担う。

    Parameters
    ----------
    params : SliderParameters
        ビーズ半径・粘性・ばね・駆動などのパラメータ。
    """

    def __init__(self, params: SliderParameters) -> None:
        self.params = params
        self._s: float = float(params.s0)
        # 傾き角から x 方向余弦を先に計算する（毎ステップの cos 呼び出しを避ける）
        # Precompute cos(phi) for F_x = f_total * cos(phi).
        self._cos_phi: float = float(np.cos(params.phi))
        self._sin_phi: float = float(np.sin(params.phi))

    # ------------------------------------------
    # 状態 / State
    # ------------------------------------------

    @property
    def s(self) -> float:
        """
        現在のスカラー位置 s_1(t) / Current position along the slider line.

        Returns
        -------
        float
            直線中心からの符号付き距離（無次元）。
        """
        return self._s

    def set_state(self, s: float) -> None:
        """
        積分器から位置を更新する / Update position from the time integrator.

        Parameters
        ----------
        s : float
            新しい s_1 の値。
        """
        self._s = float(s)

    # ------------------------------------------
    # 幾何 / Geometry
    # ------------------------------------------

    @property
    def e_s(self) -> NDArray[np.float64]:
        """
        スライダー直線の単位ベクトル / Unit vector along the slider line.

        e_s = (cos phi, 0, -sin phi)。
        x 成分が cos phi、z 成分が -sin phi となる（phi>0 で z 負方向に傾く）。

        Returns
        -------
        ndarray, shape (3,)
            単位ベクトル（無次元）。
        """
        return np.array([self._cos_phi, 0.0, -self._sin_phi], dtype=np.float64)

    @property
    def cos_phi(self) -> float:
        """傾き角の x 方向余弦 / Cosine of tilt angle (x component of e_s)."""
        return self._cos_phi

    # ------------------------------------------
    # 力 / Forces
    # ------------------------------------------

    def f_active(self, t: ArrayLike) -> NDArray[np.floating] | float:
        """
        能動駆動力（直線方向スカラー） / Active driving force along the line.

        論文 式(4.3) に準拠: f_active(t) = F_0 cos(omega t)。
        第2段階で位相差 Delta を入れる際も、この形を基準に拡張する。

        Parameters
        ----------
        t : float or array_like
            時刻（無次元） / time。

        Returns
        -------
        float or ndarray
            直線方向の能動力。
        """
        t_arr = np.asarray(t, dtype=np.float64)
        force = self.params.F_0 * np.cos(self.params.omega * t_arr)
        if force.ndim == 0:
            return float(force)
        return force

    def f_spring(self) -> float:
        """
        線形復元力 / Linear restoring force toward the line center.

        f_spring = -k * s_1。
        直線中心（s_1=0）が平衡点である。

        Returns
        -------
        float
            直線方向の復元力。
        """
        return -self.params.k * self._s

    def f_total(self, t: ArrayLike) -> NDArray[np.floating] | float:
        """
        直線方向の合力 / Total force along the slider line.

        f_total = f_active + f_spring。
        運動方程式 ds_1/dt = gamma_0 * f_total の右辺（移動度は hydrodynamics 側）。

        Parameters
        ----------
        t : float or array_like
            時刻。

        Returns
        -------
        float or ndarray
            合力の直線方向成分。
        """
        f_act = self.f_active(t)
        f_spr = self.f_spring()
        if isinstance(f_act, float):
            return f_act + f_spr
        return f_act + f_spr

    def F_x(self, t: ArrayLike) -> NDArray[np.floating] | float:
        """
        流体に与える x 方向の力 / Force on the fluid in the x direction.

        F_x(t) = f_total(t) * cos(phi)。
        瞬時流量 q_x の計算（論文 式(2.61)）に入力する。
        第1段階では壁なしのため、式(2.61) 自体は指標として用いる（flow_rate 参照）。

        Parameters
        ----------
        t : float or array_like
            時刻。

        Returns
        -------
        float or ndarray
            x 方向力。
        """
        f_tot = self.f_total(t)
        if isinstance(f_tot, float):
            return f_tot * self._cos_phi
        return f_tot * self._cos_phi
