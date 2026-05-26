"""
低レイノルズ流体力学：移動度と速度 / Low-Reynolds hydrodynamics: mobility and velocity.

【目的 / Purpose】
    Stokes 流れにおける「力 → 速度」の関係（移動度）を担当する。
    slider モジュールが計算した直線方向の力 f に対し、スカラー速度 ds_1/dt を返す。
    第2段階ではスカラー gamma_0 をモビリティ行列 Gamma に拡張する。

【構成 / Contents】
    1. stokes_translation_mobility — 自由空間並進移動度 gamma_0 の計算
    2. Mobility — 第1段階用のスカラー移動度ラッパ

【理論対応 / Theory mapping】
    - Stokes drag: 6 pi mu a v = F  →  v = gamma_0 F,  gamma_0 = 1/(6 pi mu a)
    - 運動方程式（第1段階）: ds_1/dt = gamma_0 * f_total(t)

【第2・第3段階での拡張 / Future extension】
    - stokeslet_mobility_matrix (2 sliders, no wall)
    - blakelet_mobility_matrix (2 sliders, wall at z=0)
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.slider import SliderParameters

# ==========================================
# 1. 移動度の計算
# ==========================================


def stokes_translation_mobility(mu: float, a: float) -> float:
    """
    自由空間における球の並進移動度を返す / Translational mobility in free space.

    Stokes の粘性抵抗 law 6 pi mu a v = F より、
    速度 v と力 F の比例定数（移動度）を

        gamma_0 = 1 / (6 pi mu a)

    と定義する。低レイノルズ数では慣性を無視し、
    この関係が瞬時に成立するとみなす（過阻尼系）。

    Parameters
    ----------
    mu : float
        粘性係数 / dynamic viscosity（無次元）。
    a : float
        ビーズ半径 / bead radius（無次元）。

    Returns
    -------
    float
        並進移動度 gamma_0。

    Raises
    ------
    ValueError
        mu または a が非正のとき。
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive for Stokes mobility.")
    if a <= 0.0:
        raise ValueError("a must be positive for Stokes mobility.")

    # 6 pi mu a v = F  →  v = F / (6 pi mu a)
    # Stokes drag balance: linear velocity proportional to force.
    return 1.0 / (6.0 * math.pi * mu * a)


# ==========================================
# 2. Mobility クラス（第1段階: スカラー）
# ==========================================


class Mobility:
    """
    スカラー移動度による力―速度関係 / Scalar mobility (single slider, free space).

    第1段階では 1x1 の移動度（gamma_0）のみを保持する。
    velocity_from_force(f) は ds_1/dt = gamma_0 * f に対応する。

    Parameters
    ----------
    mu : float
        粘性係数。
    a : float
        ビーズ半径。
    """

    def __init__(self, mu: float, a: float) -> None:
        self._mu = float(mu)
        self._a = float(a)
        self._gamma_0 = stokes_translation_mobility(self._mu, self._a)

    @classmethod
    def from_slider_parameters(cls, params: SliderParameters) -> Mobility:
        """
        SliderParameters から Mobility を構築する / Build from SliderParameters.

        Parameters
        ----------
        params : SliderParameters
            a, mu を使用する。

        Returns
        -------
        Mobility
        """
        return cls(mu=params.mu, a=params.a)

    @property
    def mu(self) -> float:
        """粘性係数 / dynamic viscosity."""
        return self._mu

    @property
    def a(self) -> float:
        """ビーズ半径 / bead radius."""
        return self._a

    @property
    def gamma_0(self) -> float:
        """
        並進移動度 gamma_0 = 1/(6 pi mu a) / Translational mobility.

        Returns
        -------
        float
            スカラー移動度（第2段階以降は行列の対角成分に相当）。
        """
        return self._gamma_0

    def velocity_from_force(
        self, force: ArrayLike
    ) -> NDArray[np.floating] | float:
        """
        直線方向の力から速度 ds_1/dt を返す / Velocity from force along the slider.

        第1段階の運動方程式 ds_1/dt = gamma_0 * f_total の右辺を評価する。
        力が配列のときはベクトル化して一括計算する。

        Parameters
        ----------
        force : float or array_like
            直線方向スカラー力 f（例: slider.f_total(t)）。

        Returns
        -------
        float or ndarray
            ds_1/dt [length/time]（無次元時間の場合は無次元）。
        """
        f_arr = np.asarray(force, dtype=np.float64)
        velocity = self._gamma_0 * f_arr
        if velocity.ndim == 0:
            return float(velocity)
        return velocity
