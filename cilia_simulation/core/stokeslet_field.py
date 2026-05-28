"""
自由空間 Stokeslet 速度場 / Free-space Stokeslet velocity field.

【目的 / Purpose】
    点力源（ビーズ）による低レイノルズ流れの速度場を、xz 断面（y=0）上で評価する。
    第1段階は壁なし。第3段階で Blakelet に差し替え可能なよう API を分離する。

【理論 / Theory】
    u_i(x) = (1 / (8 pi mu)) * ( F_i/r + (F_j r_j) r_i / r^3 )
    ここで r は特異点からの相対位置ベクトル。

【注意】
    第1段階では厳密な壁付き問題ではない（可視化用の遠方近似）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bead_position_3d(s1: float, cos_phi: float, sin_phi: float, h: float) -> NDArray[np.float64]:
    """
    ビーズの3次元位置を返す / Bead center position from scalar s_1.

    X = (0, 0, h) + s_1 * e_s,  e_s = (cos phi, 0, -sin phi).

    Parameters
    ----------
    s1 : float
        直線上のスカラー座標。
    cos_phi, sin_phi : float
        傾き角の三角関数。
    h : float
        直線中心の z 座標（高さ）。

    Returns
    -------
    ndarray, shape (3,)
        (x, y, z)。
    """
    return np.array(
        [s1 * cos_phi, 0.0, h - s1 * sin_phi],
        dtype=np.float64,
    )


def bead_force_3d(f_total: float, cos_phi: float, sin_phi: float) -> NDArray[np.float64]:
    """
    流体に与える力の3次元ベクトルを返す / Force vector along the slider line.

    F = f_total * e_s.

    Parameters
    ----------
    f_total : float
        直線方向スカラー力。
    cos_phi, sin_phi : float
        傾き角の三角関数。

    Returns
    -------
    ndarray, shape (3,)
    """
    return np.array(
        [f_total * cos_phi, 0.0, -f_total * sin_phi],
        dtype=np.float64,
    )


def stokeslet_velocity(
    field_xyz: NDArray[np.float64],
    singularity_xyz: NDArray[np.float64],
    force_xyz: NDArray[np.float64],
    mu: float,
    min_radius: float,
) -> NDArray[np.float64]:
    """
    自由空間 Stokeslet による速度場を返す / Velocity at field points.

    Parameters
    ----------
    field_xyz : ndarray, shape (N, 3)
        評価点の座標。
    singularity_xyz : ndarray, shape (3,)
        力の作用点（ビーズ位置）。
    force_xyz : ndarray, shape (3,)
        力ベクトル。
    mu : float
        粘性係数。
    min_radius : float
        r この値未満では速度を NaN にする（特異点回避）。

    Returns
    -------
    ndarray, shape (N, 3)
        各点での速度 (u_x, u_y, u_z)。
    """
    rel = field_xyz - singularity_xyz[None, :]
    radius = np.linalg.norm(rel, axis=1)
    safe_radius = np.maximum(radius, min_radius)
    radius_cubed = safe_radius**3

    # u_i = (1/(8 pi mu)) * (F_i/r + (F_j r_j) r_i/r^3)
    force_dot_rel = np.einsum("j,nj->n", force_xyz, rel)
    term_one = force_xyz[None, :] / safe_radius[:, None]
    term_two = force_dot_rel[:, None] * rel / radius_cubed[:, None]
    velocity = (term_one + term_two) / (8.0 * np.pi * mu)

    mask = radius < min_radius
    velocity[mask, :] = np.nan
    return velocity


def velocity_on_xz_grid(
    x_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    singularity_xyz: NDArray[np.float64],
    force_xyz: NDArray[np.float64],
    mu: float,
    min_radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    xz 平面（y=0）上の (u_x, u_z) を格子状に返す / u_x and u_z on a 2D mesh.

    Parameters
    ----------
    x_grid, z_grid : ndarray, shape (n_z, n_x)
        np.meshgrid(..., indexing="xy") の出力。
    singularity_xyz, force_xyz : ndarray
        ビーズ位置と力。
    mu, min_radius : float
        粘性と特異点マスク。

    Returns
    -------
    u_x, u_z : ndarray, shape (n_z, n_x)
    """
    x_flat = x_grid.ravel()
    z_flat = z_grid.ravel()
    y_flat = np.zeros_like(x_flat)
    field_xyz = np.column_stack([x_flat, y_flat, z_flat])
    velocity = stokeslet_velocity(
        field_xyz, singularity_xyz, force_xyz, mu, min_radius
    )
    u_x = velocity[:, 0].reshape(x_grid.shape)
    u_z = velocity[:, 2].reshape(z_grid.shape)
    return u_x, u_z
