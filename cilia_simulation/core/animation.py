"""
Stokeslet 速度ベクトル（quiver）アニメーション / Quiver animation from saved experiment runs.

【目的 / Purpose】
    保存済み exp01 出力（parameters.json, flow_rate.csv）から、
    スライダー運動と xz 平面の流れ場をオンデマンドで動画化する。

【構成 / Contents】
    1. RunData — 読み込んだ時系列とパラメータ
    2. load_run_directory — 既存 run フォルダの読み込み
    3. render_animation — MP4/GIF/preview の生成
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.typing import NDArray

from config.animation_defaults import AnimationConfig
from core.slider import SliderParameters
from core.stokeslet_field import (
    bead_force_3d,
    bead_position_3d,
    velocity_on_xz_grid,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunData:
    """
    保存済み実験 run の時系列 / Time series loaded from a run folder.

    Attributes
    ----------
    source_run_dir : Path
        読み込み元フォルダ。
    params : SliderParameters
        スライダーパラメータ。
    t : ndarray
        時刻（定常窓1周期分）。
    s1 : ndarray
        位置 s_1(t)。
    f_total : ndarray
        合力 f_total(t)。
    period : float
        周期 T = 2 pi / omega。
    """

    source_run_dir: Path
    params: SliderParameters
    t: NDArray[np.float64]
    s1: NDArray[np.float64]
    f_total: NDArray[np.float64]
    period: float


def load_run_directory(run_dir: Path | str) -> RunData:
    """
    exp01 の出力フォルダを読み込む / Load parameters.json and flow_rate.csv.

    Parameters
    ----------
    run_dir : Path or str
        output/exp01_single_slider/<timestamp>/ など。

    Returns
    -------
    RunData

    Raises
    ------
    FileNotFoundError
        必須ファイルが無いとき。
    """
    directory = Path(run_dir)
    params_path = directory / "parameters.json"
    csv_path = directory / "flow_rate.csv"

    if not params_path.is_file():
        raise FileNotFoundError(f"parameters.json not found: {params_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"flow_rate.csv not found: {csv_path}")

    with params_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    slider_dict = payload["slider"]
    params = SliderParameters(
        a=float(slider_dict["a"]),
        mu=float(slider_dict["mu"]),
        k=float(slider_dict["k"]),
        F_0=float(slider_dict["F_0"]),
        omega=float(slider_dict["omega"]),
        phi=float(slider_dict["phi"]),
        h=float(slider_dict["h"]),
        s0=float(slider_dict.get("s0", 0.0)),
    )

    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if table.ndim == 1:
        table = table.reshape(1, -1)

    t = table[:, 0]
    s1 = table[:, 2]
    f_total = table[:, 3]
    period = 2.0 * math.pi / params.omega

    logger.info(
        "Loaded run: %s (%d samples, period=%.4f)",
        directory,
        len(t),
        period,
    )
    return RunData(
        source_run_dir=directory.resolve(),
        params=params,
        t=t,
        s1=s1,
        f_total=f_total,
        period=period,
    )


def _interpolate_periodic(
    t_query: NDArray[np.float64],
    t_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    period: float,
) -> NDArray[np.float64]:
    """
    1周期データを周期的に線形補間する / Periodic linear interpolation.

    t_query はアニメーション時間（0 から始まる）。
    t_data は CSV の時刻（例: 4.0〜5.0）。位相だけ揃えてから補間する。
    """
    t0 = float(t_data[0])
    phase = np.mod(t_query, period)
    t_mapped = t0 + phase
    order = np.argsort(t_data)
    t_sorted = t_data[order]
    y_sorted = y_data[order]
    t_end = float(t_sorted[-1])
    if t_mapped.max() > t_end + 1e-12:
        t_ext = np.concatenate([t_sorted, [t_sorted[0] + period]])
        y_ext = np.concatenate([y_sorted, [y_sorted[0]]])
        return np.interp(t_mapped, t_ext, y_ext)
    return np.interp(t_mapped, t_sorted, y_sorted)


def _slider_line_xz(
    s_values: NDArray[np.float64],
    cos_phi: float,
    sin_phi: float,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """スライダー直線の xz 座標列を返す / Slider line in the xz plane."""
    x_line = s_values * cos_phi
    z_line = h - s_values * sin_phi
    return x_line, z_line


def render_animation(
    run_data: RunData,
    output_dir: Path | str,
    config: AnimationConfig | None = None,
    *,
    save_gif: bool = False,
) -> Path:
    """
    アニメーションを生成して保存する / Build and save MP4 (and optional GIF).

    Parameters
    ----------
    run_data : RunData
        load_run_directory の戻り値。
    output_dir : Path or str
        保存先ディレクトリ。
    config : AnimationConfig, optional
        描画設定。
    save_gif : bool
        True なら animation.gif も保存する。

    Returns
    -------
    Path
        主出力（animation.mp4 または animation.gif）のパス。
    """
    anim_config = config if config is not None else AnimationConfig()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    params = run_data.params
    cos_phi = math.cos(params.phi)
    sin_phi = math.sin(params.phi)
    min_radius = anim_config.singularity_mask_factor * params.a

    x_1d = np.linspace(anim_config.x_min, anim_config.x_max, anim_config.n_x)
    z_1d = np.linspace(anim_config.z_min, anim_config.z_max, anim_config.n_z)
    x_grid, z_grid = np.meshgrid(x_1d, z_1d, indexing="xy")

    duration = run_data.period * anim_config.loop_count
    t_frames = np.linspace(0.0, duration, anim_config.n_frames, endpoint=False)
    s_frames = _interpolate_periodic(t_frames, run_data.t, run_data.s1, run_data.period)
    f_frames = _interpolate_periodic(
        t_frames, run_data.t, run_data.f_total, run_data.period
    )

    s_line = np.linspace(float(run_data.s1.min()), float(run_data.s1.max()), 80)
    x_line, z_line = _slider_line_xz(s_line, cos_phi, sin_phi, params.h)

    index_x = np.linspace(0, anim_config.n_x - 1, anim_config.n_quiver_x, dtype=int)
    index_z = np.linspace(0, anim_config.n_z - 1, anim_config.n_quiver_z, dtype=int)
    domain_short_side = min(
        anim_config.x_max - anim_config.x_min,
        anim_config.z_max - anim_config.z_min,
    )
    max_arrow_length = anim_config.quiver_arrow_scale * domain_short_side

    # Keep one quiver scale for all frames to avoid visual jumps caused by
    # frame-by-frame autoscaling.
    global_max_speed = 0.0
    for frame_index in range(anim_config.n_frames):
        s1_tmp = float(s_frames[frame_index])
        f_tot_tmp = float(f_frames[frame_index])
        pos_tmp = bead_position_3d(s1_tmp, cos_phi, sin_phi, params.h)
        force_tmp = bead_force_3d(f_tot_tmp, cos_phi, sin_phi)
        u_x_tmp, u_z_tmp = velocity_on_xz_grid(
            x_grid, z_grid, pos_tmp, force_tmp, params.mu, min_radius
        )
        u_sub_tmp = u_x_tmp[np.ix_(index_z, index_x)]
        w_sub_tmp = u_z_tmp[np.ix_(index_z, index_x)]
        speed_sub_tmp = np.hypot(u_sub_tmp, w_sub_tmp)
        valid_tmp = np.isfinite(speed_sub_tmp) & (speed_sub_tmp > 0.0)
        if np.any(valid_tmp):
            frame_max_speed = float(np.nanmax(speed_sub_tmp[valid_tmp]))
            global_max_speed = max(global_max_speed, frame_max_speed)
    quiver_scale = global_max_speed / max_arrow_length if global_max_speed > 0.0 else 1.0

    fig, axis = plt.subplots(
        figsize=(anim_config.fig_width, anim_config.fig_height),
        dpi=anim_config.dpi,
    )

    def _draw_frame(frame_index: int) -> None:
        axis.clear()
        s1 = float(s_frames[frame_index])
        f_tot = float(f_frames[frame_index])
        pos = bead_position_3d(s1, cos_phi, sin_phi, params.h)
        force = bead_force_3d(f_tot, cos_phi, sin_phi)

        u_x, u_z = velocity_on_xz_grid(
            x_grid, z_grid, pos, force, params.mu, min_radius
        )
        x_sub, z_sub = np.meshgrid(
            x_1d[index_x], z_1d[index_z], indexing="xy"
        )
        u_sub = u_x[np.ix_(index_z, index_x)]
        w_sub = u_z[np.ix_(index_z, index_x)]
        speed_sub = np.hypot(u_sub, w_sub)
        valid = np.isfinite(speed_sub) & (speed_sub > 0.0)

        if np.any(valid):
            axis.quiver(
                x_sub[valid],
                z_sub[valid],
                u_sub[valid],
                w_sub[valid],
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                color="steelblue",
                width=0.004,
                headwidth=4.0,
                headlength=5.0,
                zorder=2,
            )
        axis.plot(x_line, z_line, color="gray", linestyle="--", linewidth=1.0, label="slider line")
        axis.scatter(
            [pos[0]],
            [pos[2]],
            s=anim_config.bead_marker_size,
            c="crimson",
            zorder=5,
            label="bead",
        )
        axis.set_xlim(anim_config.x_min, anim_config.x_max)
        axis.set_ylim(anim_config.z_min, anim_config.z_max)
        axis.set_xlabel(r"$x^{*}$")
        axis.set_ylabel(r"$z^{*}$")
        axis.set_title(
            r"Stokeslet velocity (xz), $t^{*}=%.2f$" % float(t_frames[frame_index])
        )
        axis.legend(loc="upper right", fontsize=9)
        axis.grid(True, alpha=0.35)
        axis.set_aspect("equal", adjustable="box")

    _draw_frame(0)
    preview_path = out_path / "preview.png"
    fig.savefig(preview_path)
    logger.info("Preview saved: %s", preview_path)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=anim_config.n_frames,
        interval=1000.0 / anim_config.fps,
        repeat=True,
    )

    mp4_path = out_path / "animation.mp4"
    gif_path = out_path / "animation.gif"
    primary_path = mp4_path

    if "ffmpeg" in animation.writers.list():
        writer = animation.writers["ffmpeg"](fps=anim_config.fps)
        anim.save(str(mp4_path), writer=writer, dpi=anim_config.dpi)
        logger.info("MP4 saved: %s", mp4_path)
    else:
        logger.warning("ffmpeg writer not available; saving GIF instead of MP4.")
        primary_path = gif_path
        writer = animation.writers["pillow"](fps=anim_config.fps)
        anim.save(str(gif_path), writer=writer, dpi=anim_config.dpi)
        logger.info("GIF saved: %s", gif_path)

    if save_gif and primary_path != gif_path:
        writer = animation.writers["pillow"](fps=anim_config.fps)
        anim.save(str(gif_path), writer=writer, dpi=anim_config.dpi)
        logger.info("GIF saved: %s", gif_path)

    plt.close(fig)

    anim_config_path = out_path / "anim_config.json"
    with anim_config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_run": str(run_data.source_run_dir),
                "animation_config": anim_config.__dict__,
                "period": run_data.period,
                "n_samples_csv": int(len(run_data.t)),
                "note": "Stokeslet free-space visualization; not Blakelet (exp01).",
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    return primary_path


def make_animation_output_directory(
    exp_name: str,
    source_run_dir: Path,
    base: Path | str = "output/animations",
) -> Path:
    """
    アニメーション出力フォルダを作成する / output/animations/<exp>/<source>_<time>/.

    Parameters
    ----------
    exp_name : str
        実験名。
    source_run_dir : Path
        元の run フォルダ。
    base : Path or str
        animations のルート。

    Returns
    -------
    Path
    """
    source_id = source_run_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base) / exp_name / f"{source_id}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
