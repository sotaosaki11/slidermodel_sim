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

【第2・第4段階での拡張 / Extension】
    - TwoSliderMobility (2 sliders, no wall, Oseen)
    - BlakeletTwoSliderMobility (2 sliders, wall at z=0, Blakelet)
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
        use_y_constraint: bool = False,
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
        use_y_constraint : bool
            True のとき y 方向拘束（e_y）を追加し 4×4 系を解く（exp06）。

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

        ds1_dt, ds2_dt, _, _ = _solve_two_slider_constrained_velocities(
            M_aa_1=M_aa,
            M_aa_2=M_aa,
            M_ab_12=M_ab_12,
            M_ab_21=M_ab_21,
            f_vec1=f_vec1,
            f_vec2=f_vec2,
            es=es,
            esp=esp,
            use_y_constraint=use_y_constraint,
        )
        return ds1_dt, ds2_dt


# ==========================================
# 4. 第4段階: 2スライダー移動度（壁 z=0, Blakelet, 拘束付き）
# ==========================================


def _as_position3(r: ArrayLike, name: str) -> NDArray[np.float64]:
    """3次元位置ベクトルを (3,) ndarray に正規化する。"""
    arr = np.asarray(r, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector with shape (3,).")
    return arr


_E_Y = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _mobility_constraint_coeff(
    n: NDArray[np.float64],
    M: NDArray[np.float64],
    v: NDArray[np.float64],
) -> float:
    """拘束法線 n に対する n·(M·v) の係数。"""
    return float(n @ M @ v)


def _solve_two_slider_constrained_velocities(
    *,
    M_aa_1: NDArray[np.float64],
    M_aa_2: NDArray[np.float64],
    M_ab_12: NDArray[np.float64],
    M_ab_21: NDArray[np.float64],
    f_vec1: NDArray[np.float64],
    f_vec2: NDArray[np.float64],
    es: NDArray[np.float64],
    esp: NDArray[np.float64],
    use_y_constraint: bool,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """
    拘束付き 2 スライダーのレール方向速度と 3D 速度ベクトルを返す。

    use_y_constraint=False: e_s⊥ 方向のみ（2×2、exp01–05）。
    use_y_constraint=True : e_s⊥ と e_y の両方（4×4、exp06 layout_theta≠0）。

    Returns
    -------
    ds1_dt, ds2_dt, rdot1, rdot2
    """
    rdot_free_1 = M_aa_1 @ f_vec1 + M_ab_12 @ f_vec2
    rdot_free_2 = M_ab_21 @ f_vec1 + M_aa_2 @ f_vec2

    if not use_y_constraint:
        A = np.array(
            [
                [_mobility_constraint_coeff(esp, M_aa_1, esp), _mobility_constraint_coeff(esp, M_ab_12, esp)],
                [_mobility_constraint_coeff(esp, M_ab_21, esp), _mobility_constraint_coeff(esp, M_aa_2, esp)],
            ],
            dtype=np.float64,
        )
        b = np.array(
            [
                -float(esp @ rdot_free_1),
                -float(esp @ rdot_free_2),
            ],
            dtype=np.float64,
        )
        lambda_perp_1, lambda_perp_2 = np.linalg.solve(A, b)
        F1 = f_vec1 + lambda_perp_1 * esp
        F2 = f_vec2 + lambda_perp_2 * esp
    else:
        ey = _E_Y
        A = np.array(
            [
                [
                    _mobility_constraint_coeff(esp, M_aa_1, esp),
                    _mobility_constraint_coeff(esp, M_aa_1, ey),
                    _mobility_constraint_coeff(esp, M_ab_12, esp),
                    _mobility_constraint_coeff(esp, M_ab_12, ey),
                ],
                [
                    _mobility_constraint_coeff(ey, M_aa_1, esp),
                    _mobility_constraint_coeff(ey, M_aa_1, ey),
                    _mobility_constraint_coeff(ey, M_ab_12, esp),
                    _mobility_constraint_coeff(ey, M_ab_12, ey),
                ],
                [
                    _mobility_constraint_coeff(esp, M_ab_21, esp),
                    _mobility_constraint_coeff(esp, M_ab_21, ey),
                    _mobility_constraint_coeff(esp, M_aa_2, esp),
                    _mobility_constraint_coeff(esp, M_aa_2, ey),
                ],
                [
                    _mobility_constraint_coeff(ey, M_ab_21, esp),
                    _mobility_constraint_coeff(ey, M_ab_21, ey),
                    _mobility_constraint_coeff(ey, M_aa_2, esp),
                    _mobility_constraint_coeff(ey, M_aa_2, ey),
                ],
            ],
            dtype=np.float64,
        )
        b = np.array(
            [
                -float(esp @ rdot_free_1),
                -float(ey @ rdot_free_1),
                -float(esp @ rdot_free_2),
                -float(ey @ rdot_free_2),
            ],
            dtype=np.float64,
        )
        lambda_perp_1, lambda_y_1, lambda_perp_2, lambda_y_2 = np.linalg.solve(A, b)
        F1 = f_vec1 + lambda_perp_1 * esp + lambda_y_1 * ey
        F2 = f_vec2 + lambda_perp_2 * esp + lambda_y_2 * ey

    rdot1 = M_aa_1 @ F1 + M_ab_12 @ F2
    rdot2 = M_ab_21 @ F1 + M_aa_2 @ F2
    return float(rdot1 @ es), float(rdot2 @ es), rdot1, rdot2


def wall_reflection_mobility(
    z: float,
    *,
    mu: float,
    a: float,
) -> NDArray[np.float64]:
    """
    壁 z=0 に対する自己移動度の反射寄与を返す / Wall self-mobility correction.

    論文 Eq.(2.51)(2.52): 正味の自己モビリティは

        M_aa = (1/(6 pi mu a)) (I + M_hat)

    （M_hat は 6 pi mu a で無次元化された反射寄与なので gamma_0 が両項に掛かる）。
    ここで h = z/a とし、本関数は M_hat（3x3）を返す。
    z はビーズ中心の壁法線座標（z > 0 を想定）。

    Notes
    -----
    Faxén 演算子 F_x F_y は自己項では Jw の評価に含まれるが、
    式(2.51) は既にその結果を成分で与えている。
    """
    if z <= 0.0:
        raise ValueError("z must be positive for wall reflection mobility.")
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    if a <= 0.0:
        raise ValueError("a must be positive.")

    h = z / a
    h_inv = 1.0 / h
    h_inv3 = h_inv**3
    h_inv5 = h_inv**5

    # 式(2.51) の係数（6 pi mu a で無次元化された反射寄与）
    coeff_tangent = -(1.0 / 16.0) * (9.0 * h_inv - 2.0 * h_inv3 + h_inv5)
    coeff_normal = -(1.0 / 8.0) * (9.0 * h_inv - 4.0 * h_inv3 + h_inv5)

    M_hat = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:
                M_hat[i, j] = coeff_normal
            elif i != 2 and j != 2:
                M_hat[i, j] = coeff_tangent if i == j else 0.0
            # i=2,j!=2 および i!=2,j=2 は 0 のまま
    return M_hat


def blakelet_mobility_tensor(
    r_obs: ArrayLike,
    r_source: ArrayLike,
    *,
    mu: float,
) -> NDArray[np.float64]:
    """
    Blakelet 移動度テンソル G(r_obs, r_source) を返す。

    論文 Eq.(2.44)(4.17): 観測点 r、力の作用点 r_0 (= r_source) に対し

        8 pi mu G_ij = delta_ij/rho + rho_i rho_j/rho^3
                       - delta_ij/R - R_i R_j/R^3
                       + 2 z_0^2 (P_ij/R^3 - 3 R_i Rtilde_j/R^5)
                       - 2 z_0 ((P_ij R_3 + P_j3 R_i - delta_i3 Rtilde_j)/R^3
                                 - 3 R_i R_3 Rtilde_j/R^5)

    ここで rho = r - r_0, R = r - r_I, r_I = (x_0, y_0, -z_0),
    P = diag(1,1,-1), Rtilde = P @ R, R_3 は R の z 成分。
    Stokes dipole 項（最後の括弧）は論文 式(4.17) に厳密に一致させてあり、
    壁面 z=0 上で G = 0（滑りなし）を満たす（tests/test_blakelet_mobility.py で検証）。

    Parameters
    ----------
    r_obs : array_like, shape (3,)
        速度を評価する点（観測点）。
    r_source : array_like, shape (3,)
        力が作用する点（z_0 > 0 を想定）。
    mu : float
        粘性係数。

    Returns
    -------
    ndarray, shape (3, 3)
        移動度テンソル G。速度は rdot = G @ F。

    Notes
    -----
    exp02/exp03 の Oseen 相互項と同様、相互項では Faxén 演算子 F_x F_y を
    省略し G を直接使用する（遠距離・点粒子近似）。論文 Eq.(2.50) との差分は
    コメントで明記する。
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    r = _as_position3(r_obs, "r_obs")
    r0 = _as_position3(r_source, "r_source")
    z0 = float(r0[2])
    if z0 <= 0.0:
        raise ValueError("r_source z-coordinate must be positive (above the wall).")

    rho_vec = r - r0
    r_image = np.array([r0[0], r0[1], -r0[2]], dtype=np.float64)
    R_vec = r - r_image
    R_tilde = np.array([R_vec[0], R_vec[1], -R_vec[2]], dtype=np.float64)

    rho = float(np.linalg.norm(rho_vec))
    R_norm = float(np.linalg.norm(R_vec))
    if rho <= 0.0:
        raise ValueError("Observation point must not coincide with force location.")
    if R_norm <= 0.0:
        raise ValueError("Observation point must not coincide with image location.")

    rho3 = rho**3
    rho5 = rho**5
    R3 = R_norm**3
    R5 = R_norm**5

    G = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            stokes = (1.0 if i == j else 0.0) / rho + rho_vec[i] * rho_vec[j] / rho3
            image = (1.0 if i == j else 0.0) / R_norm + R_vec[i] * R_vec[j] / R3

            # P = diag(1,1,-1)。P_ij は対角成分、P_j3 は (j,3) 成分。
            P_ij = 1.0 if (i == j and i != 2) else (-1.0 if i == j == 2 else 0.0)
            P_j3 = -1.0 if j == 2 else 0.0
            delta_i3 = 1.0 if i == 2 else 0.0
            R_z = R_vec[2]  # R の z 成分（論文 式(4.17) の R_3）

            # source doublet（論文 式(4.17) 第3項）
            sd_term = 2.0 * z0 * z0 * (P_ij / R3 - 3.0 * R_vec[i] * R_tilde[j] / R5)
            # Stokes dipole（論文 式(4.17) 第4項）。壁面 no-slip を満たすために
            # (P_ij R_z + P_j3 R_i - delta_i3 Rtilde_j)/R^3 の3項すべてが必要。
            dip_term = 2.0 * z0 * (
                (P_ij * R_z + P_j3 * R_vec[i] - delta_i3 * R_tilde[j]) / R3
                - 3.0 * R_vec[i] * R_z * R_tilde[j] / R5
            )

            G[i, j] = (stokes - image + sd_term - dip_term) / (8.0 * math.pi * mu)

    return G


@dataclass(frozen=True)
class BlakeletTwoSliderMobility:
    """
    2スライダー（壁 z=0 あり）を Blakelet と拘束条件で結ぶ移動度モデル。

    Notes
    -----
    論文 Eq.(4.4)(4.5) の数値実装（論文最終形）:
    - 自己モビリティ: M_aa = gamma0 (I + M_hat(z/a))  … Eq.(2.51)(2.52)
    - 交差モビリティ: M_ab = G(r_i, r_j)            … Eq.(2.44)(4.17)
    - 拘束: rdot_i · e_s_perp = 0（Eq.(4.5) の 2x2 連立）

    exp02 の TwoSliderMobility と API を揃え、TwoSliderTimeStepper から
    そのまま差し替え可能とする。
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
        """自由空間の自己移動度 gamma_0 = 1/(6*pi*mu*a)。"""
        return 1.0 / (6.0 * math.pi * self.mu * self.a)

    def self_mobility(self, r: ArrayLike) -> NDArray[np.float64]:
        """
        位置 r における自己移動度 M_aa(r) を返す。

        M_aa = gamma_0 (I + M_hat(z/a)),  z = r[2]（論文 Eq.(2.52)）。
        M_hat は 6 pi mu a で無次元化された反射寄与なので、gamma_0 が
        I と M_hat の両方に掛かる（次元整合）。
        """
        ri = _as_position3(r, "r")
        z = float(ri[2])
        return self.gamma_0 * (
            np.eye(3, dtype=np.float64)
            + wall_reflection_mobility(z, mu=self.mu, a=self.a)
        )

    def cross_mobility(
        self,
        r_i: ArrayLike,
        r_j: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        粒子 i の速度に対する粒子 j の力の寄与 M_ij = G(r_i, r_j)。
        """
        return blakelet_mobility_tensor(r_i, r_j, mu=self.mu)

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
        use_y_constraint: bool = False,
    ) -> tuple[float, float]:
        """
        拘束付き2スライダー速度を返す（Blakelet 版）。

        TwoSliderMobility.compute_velocities と同一の拘束解法（Eq.(4.5)）。
        use_y_constraint=True のとき e_y 拘束を追加（exp06）。
        """
        if k < 0.0:
            raise ValueError("k must be non-negative.")

        es = np.asarray(e_s, dtype=np.float64)
        esp = np.asarray(e_s_perp, dtype=np.float64)
        if es.shape != (3,) or esp.shape != (3,):
            raise ValueError("e_s and e_s_perp must be vectors with shape (3,).")

        r1_arr = _as_position3(r1, "r1")
        r2_arr = _as_position3(r2, "r2")

        M_aa_1 = self.self_mobility(r1_arr)
        M_aa_2 = self.self_mobility(r2_arr)
        M_ab_12 = self.cross_mobility(r1_arr, r2_arr)
        M_ab_21 = self.cross_mobility(r2_arr, r1_arr)

        f_total1 = float(f_active1 - k * s1)
        f_total2 = float(f_active2 - k * s2)
        f_vec1 = f_total1 * es
        f_vec2 = f_total2 * es

        ds1_dt, ds2_dt, _, _ = _solve_two_slider_constrained_velocities(
            M_aa_1=M_aa_1,
            M_aa_2=M_aa_2,
            M_ab_12=M_ab_12,
            M_ab_21=M_ab_21,
            f_vec1=f_vec1,
            f_vec2=f_vec2,
            es=es,
            esp=esp,
            use_y_constraint=use_y_constraint,
        )
        return ds1_dt, ds2_dt
