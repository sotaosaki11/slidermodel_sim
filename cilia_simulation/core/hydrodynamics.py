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
from dataclasses import dataclass

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


# ==========================================
# 3. 第2段階: 2スライダー移動度（壁なし, 拘束付き）
# ==========================================


@dataclass(frozen=True)
class TwoSliderMobility:
    """
    2スライダー（壁なし）を Oseen テンソルと拘束条件で結ぶ移動度モデル。

    Notes
    -----
    論文 Eq.(4.4),(4.5) を意識した最小実装:
    - 自己モビリティ: M_aa = gamma0 * I, gamma0 = 1/(6*pi*mu*a)
    - 交差モビリティ: M_ab = Oseen tensor J(r_i, r_j)
    - 拘束: rdot_i · e_s_perp = 0
      （未知の拘束力 lambda_1, lambda_2 を 2x2 線形方程式で解く）
    """

    mu: float
    a: float

    def __post_init__(self) -> None:
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")
        if self.a <= 0.0:
            raise ValueError("a must be positive.")

    @property
    def gamma_0(self) -> float:
        """自己移動度 gamma_0 = 1/(6*pi*mu*a)。"""
        return 1.0 / (6.0 * math.pi * self.mu * self.a)

    @property
    def M_aa(self) -> NDArray[np.float64]:
        """自己モビリティ M_aa = gamma_0 * I."""
        return self.gamma_0 * np.eye(3, dtype=np.float64)

    def stokeslet_kernel_tensor(
        self,
        r_i: ArrayLike,
        r_j: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        論文 Eq.(2.40) の核テンソル J(r_i, r_j) を返す。

        J = I + (R outer R) / R^2,  R = r_i - r_j,  R = |R|
        （ここでは係数 1/(8*pi*mu*R) は含めない）
        """
        ri = np.asarray(r_i, dtype=np.float64)
        rj = np.asarray(r_j, dtype=np.float64)
        if ri.shape != (3,) or rj.shape != (3,):
            raise ValueError("r_i and r_j must be 3D vectors with shape (3,).")

        R = ri - rj
        r = float(np.linalg.norm(R))
        if r <= 0.0:
            raise ValueError("Distance between r_i and r_j must be positive.")

        I = np.eye(3, dtype=np.float64)
        RR = np.outer(R, R)
        return I + RR / (r**2)

    def oseen_tensor(
        self,
        r_i: ArrayLike,
        r_j: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        論文 Eq.(2.39) と同じ物理次元の Oseen 移動度テンソルを返す。

        M_ab = (1/(8*pi*mu*R)) * J
            = (1/(8*pi*mu)) * (I/R + (R outer R)/R^3)
        """
        ri = np.asarray(r_i, dtype=np.float64)
        rj = np.asarray(r_j, dtype=np.float64)
        if ri.shape != (3,) or rj.shape != (3,):
            raise ValueError("r_i and r_j must be 3D vectors with shape (3,).")

        R_vec = ri - rj
        R_norm = float(np.linalg.norm(R_vec))
        if R_norm <= 0.0:
            raise ValueError("Distance between r_i and r_j must be positive.")

        J = self.stokeslet_kernel_tensor(ri, rj)
        return (1.0 / (8.0 * math.pi * self.mu * R_norm)) * J

    def compute_velocities(
        self,
        *,
        s1: float,
        s2: float,
        r1: ArrayLike,
        r2: ArrayLike,
        e_s: ArrayLike,
        e_s_perp: ArrayLike,
        f_active1: float,
        f_active2: float,
        k: float,
    ) -> tuple[float, float]:
        """
        拘束付き2スライダー速度を返す。

        Parameters
        ----------
        s1, s2 : float
            現在のスカラー座標。
        r1, r2 : array_like, shape (3,)
            各ビーズの現在位置ベクトル。
        e_s, e_s_perp : array_like, shape (3,)
            直線方向とその垂直方向の単位ベクトル。
        f_active1, f_active2 : float
            各スライダーの能動力（スカラー）。
        k : float
            ばね定数。

        Returns
        -------
        tuple[float, float]
            (ds1/dt, ds2/dt)
        """
        if k < 0.0:
            raise ValueError("k must be non-negative.")

        es = np.asarray(e_s, dtype=np.float64)
        esp = np.asarray(e_s_perp, dtype=np.float64)
        if es.shape != (3,) or esp.shape != (3,):
            raise ValueError("e_s and e_s_perp must be vectors with shape (3,).")

        M_aa = self.M_aa
        M_ab_12 = self.oseen_tensor(r1, r2)
        M_ab_21 = self.oseen_tensor(r2, r1)

        f_total1 = float(f_active1 - k * s1)
        f_total2 = float(f_active2 - k * s2)
        f_vec1 = f_total1 * es
        f_vec2 = f_total2 * es

        A = np.array(
            [
                [esp @ (M_aa @ esp), esp @ (M_ab_12 @ esp)],
                [esp @ (M_ab_21 @ esp), esp @ (M_aa @ esp)],
            ],
            dtype=np.float64,
        )
        b = np.array(
            [
                -(esp @ (M_aa @ f_vec1 + M_ab_12 @ f_vec2)),
                -(esp @ (M_ab_21 @ f_vec1 + M_aa @ f_vec2)),
            ],
            dtype=np.float64,
        )
        lambda1, lambda2 = np.linalg.solve(A, b)

        F1 = f_vec1 + lambda1 * esp
        F2 = f_vec2 + lambda2 * esp
        rdot1 = M_aa @ F1 + M_ab_12 @ F2
        rdot2 = M_ab_21 @ F1 + M_aa @ F2

        ds1_dt = float(rdot1 @ es)
        ds2_dt = float(rdot2 @ es)
        return ds1_dt, ds2_dt
