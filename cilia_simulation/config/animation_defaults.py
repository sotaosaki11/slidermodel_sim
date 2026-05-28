"""
アニメーション用デフォルト設定 / Default settings for visualization animations.

【目的 / Purpose】
    Stokeslet 速度ベクトル（quiver）アニメーションの格子・表示範囲を一括管理する。
    物理パラメータは parameters.json から読み込む（ここには書かない）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimationConfig:
    """
    アニメーション描画の設定 / Plot and export settings.

    Attributes
    ----------
    x_min, x_max, z_min, z_max : float
        xz 平面の表示範囲（無次元）。
    n_x, n_z : int
        速度場グリッドの分割数（大きいほど重い）。
    n_frames : int
        全ループ合計のフレーム数（loop_count 周期分）。
    loop_count : int
        CSV の1周期をループ再生する回数。
    n_quiver_x, n_quiver_z : int
        xz 平面に並べる速度ベクトルの本数（等間隔）。
    quiver_arrow_scale : float
        最大速度に対する矢印長さ（表示域の短辺に対する比率、例 0.15）。
    singularity_mask_factor : float
        特異点マスク半径 = factor * a。
    fps : int
        動画のフレームレート。
    fig_width, fig_height : float
        図サイズ [inch]。
    dpi : int
        出力解像度。
    bead_marker_size : float
        ビーズのマーカーサイズ。
    """

    x_min: float = -1.5
    x_max: float = 1.5
    z_min: float = -0.5
    z_max: float = 2.0
    n_x: int = 40
    n_z: int = 30
    n_frames: int = 60
    loop_count: int = 2
    n_quiver_x: int = 14
    n_quiver_z: int = 10
    quiver_arrow_scale: float = 0.15
    singularity_mask_factor: float = 1.5
    fps: int = 5
    fig_width: float = 8.0
    fig_height: float = 5.0
    dpi: int = 120
    bead_marker_size: float = 90.0
