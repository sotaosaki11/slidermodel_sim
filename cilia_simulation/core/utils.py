"""
入出力・プロットの補助 / I/O helpers and matplotlib figures.

【目的 / Purpose】
    実験実行ごとの出力ディレクトリ作成、parameters.json 保存、
    要約テキスト・CSV・図（s_1, q_x）の保存を行う。

【構成 / Contents】
    1. PlotStyle — 図の見た目（実験スクリプトから上書き可能）
    2. ディレクトリ・JSON・テキスト・CSV
    3. プロット（最後 n 周期分を切り出して描画）

【主要関数 / Main functions】
    make_run_directory, save_parameters, save_summary, save_flow_rate_csv
    plot_trajectory, plot_flow_rate, slice_last_n_periods
    plot_q_heatmap_delta_l, plot_delta_opt_vs_l
    plot_q_vs_delta_fixed_l, plot_q_vs_delta_multi_l_from_q_map
    dimless_label
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

# ==========================================
# 1. プロット設定
# ==========================================


@dataclass(frozen=True)
class PlotStyle:
    """
    Matplotlib の体裁 / Figure style for publication-quality plots.

    物理パラメータは含めない。実験スクリプトで必要ならコピーして上書きする。
    """

    figure_width: float = 8.0
    figure_height: float = 4.5
    dpi: int = 150
    line_width: float = 1.5
    color_primary: str = "C0"
    color_secondary: str = "C1"
    grid_alpha: float = 0.35
    font_size: int = 11


def dimless_label(latex_symbol: str) -> str:
    """
    無次元量の軸・凡例ラベル用 LaTeX 文字列（上付き *）を返す。

    Parameters
    ----------
    latex_symbol : str
        本体記号（例: ``Q``, ``l``, ``s_1``）。

    Returns
    -------
    str
        例: ``$Q^{*}$``
    """
    return rf"${latex_symbol}^{{*}}$"


# ==========================================
# 2. ディレクトリとファイル I/O
# ==========================================


def make_run_directory(
    exp_name: str,
    base: Path | str = "output",
) -> Path:
    """
    実行ごとの出力フォルダを作成する / Create output/<exp>/<YYYYMMDD_HHMMSS>/.

    Parameters
    ----------
    exp_name : str
        実験名（例: exp01_single_slider）。
    base : Path or str
        出力のルート（デフォルト output）。

    Returns
    -------
    Path
        作成した実行ディレクトリのパス。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory created: %s", run_dir)
    return run_dir


def _to_json_serializable(obj: Any) -> Any:
    """
    dataclass や numpy 型を JSON 化可能な形に変換する / Convert to JSON-safe types.
    """
    if is_dataclass(obj):
        return {key: _to_json_serializable(val) for key, val in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): _to_json_serializable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(val) for val in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_parameters(path: Path | str, parameters: dict[str, Any] | Any) -> None:
    """
    parameters.json を保存する / Save run parameters for reproducibility.

    Parameters
    ----------
    path : Path or str
        保存先（例: run_dir / parameters.json）。
    parameters : dict or dataclass
        記録するパラメータ一式。
    """
    target = Path(path)
    payload = _to_json_serializable(parameters)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    logger.info("Parameters saved: %s", target)


def save_summary(path: Path | str, lines: list[str]) -> None:
    """
    summary.txt を保存する / Save text summary (Q check, etc.).

    Parameters
    ----------
    path : Path or str
        保存先。
    lines : list of str
        行ごとのテキスト。
    """
    target = Path(path)
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    target.write_text(text, encoding="utf-8")
    logger.info("Summary saved: %s", target)


def save_flow_rate_csv(
    path: Path | str,
    t: ArrayLike,
    q_x: ArrayLike,
    *,
    s: ArrayLike | None = None,
    f_total: ArrayLike | None = None,
) -> None:
    """
    流量時系列を CSV に保存する / Save flow-rate time series to CSV.

    Parameters
    ----------
    path : Path or str
        保存先（flow_rate.csv）。
    t, q_x : array_like
        時刻と瞬時流量。
    s, f_total : array_like, optional
        あれば同じ行に s_1, f_total も書く。
    """
    target = Path(path)
    t_arr = np.asarray(t, dtype=np.float64)
    q_arr = np.asarray(q_x, dtype=np.float64)
    columns: list[NDArray[np.float64]] = [t_arr, q_arr]
    header = "t,q_x"

    if s is not None:
        columns.append(np.asarray(s, dtype=np.float64))
        header += ",s_1"
    if f_total is not None:
        columns.append(np.asarray(f_total, dtype=np.float64))
        header += ",f_total"

    data = np.column_stack(columns)
    np.savetxt(
        target,
        data,
        delimiter=",",
        header=header,
        comments="",
        encoding="utf-8",
    )
    logger.info("Flow-rate CSV saved: %s", target)


# ==========================================
# 3. 時系列の切り出しとプロット
# ==========================================


def slice_last_n_periods(
    t: ArrayLike,
    y: ArrayLike,
    period: float,
    n_periods: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    最後 n 周期分のデータを切り出す / Extract the last n periods for plotting.

    Parameters
    ----------
    t, y : array_like
        時系列。
    period : float
        周期 T。
    n_periods : int
        切り出す周期数（デフォルト2）。

    Returns
    -------
    t_slice, y_slice
    """
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1.")
    if period <= 0.0:
        raise ValueError("period must be positive.")

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    t_end = float(t_arr[-1])
    t_start = t_end - n_periods * period
    mask = t_arr >= t_start
    return t_arr[mask], y_arr[mask]


def plot_trajectory(
    path: Path | str,
    t: ArrayLike,
    s: ArrayLike,
    period: float,
    *,
    n_periods: int = 2,
    style: PlotStyle | None = None,
) -> None:
    """
    s_1(t) の時系列を保存する / Plot slider position vs time.

    Parameters
    ----------
    path : Path or str
        保存先 PNG（trajectory.png）。
    t, s : array_like
        時刻と s_1。
    period : float
        周期 T（切り出し用）。
    n_periods : int
        プロットに使う末尾周期数。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    t_plot, s_plot = slice_last_n_periods(t, s, period, n_periods)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        t_plot,
        s_plot,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        label=r"$s_1^{*}(t^{*})$",
    )
    axis.set_xlabel(dimless_label("t"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("s_1"), fontsize=plot_style.font_size)
    axis.set_title(
        r"Slider position (last {:d} periods)".format(n_periods),
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Trajectory plot saved: %s", path)


def plot_flow_rate(
    path: Path | str,
    t: ArrayLike,
    q_x: ArrayLike,
    period: float,
    *,
    n_periods: int = 2,
    style: PlotStyle | None = None,
) -> None:
    """
    q_x(t) の時系列を保存する / Plot instantaneous flow rate vs time.

    q_x は論文 式(2.61) に基づく指標（第1段階では壁なし）。

    Parameters
    ----------
    path : Path or str
        保存先 PNG（flow_rate.png）。
    t, q_x : array_like
        時刻と瞬時流量。
    period : float
        周期 T。
    n_periods : int
        プロットに使う末尾周期数。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    t_plot, q_plot = slice_last_n_periods(t, q_x, period, n_periods)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        t_plot,
        q_plot,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        label=r"$q_x^{*}(t^{*})$",
    )
    axis.set_xlabel(dimless_label("t"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("q_x"), fontsize=plot_style.font_size)
    axis.set_title(
        r"Instantaneous flow rate (last {:d} periods)".format(n_periods),
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Flow-rate plot saved: %s", path)


def plot_two_slider_trajectories(
    path: Path | str,
    t: ArrayLike,
    s1: ArrayLike,
    s2: ArrayLike,
    period: float,
    *,
    n_periods: int = 2,
    style: PlotStyle | None = None,
) -> None:
    """
    2スライダーの軌道 s1(t), s2(t) を重ね描きで保存する。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    t : array_like
        時刻列。
    s1, s2 : array_like
        各スライダー座標。
    period : float
        駆動周期 T。
    n_periods : int
        末尾から描画する周期数。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    t_plot, s1_plot = slice_last_n_periods(t, s1, period, n_periods)
    _, s2_plot = slice_last_n_periods(t, s2, period, n_periods)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        t_plot,
        s1_plot,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        label=r"$s_1^{*}(t^{*})$",
    )
    axis.plot(
        t_plot,
        s2_plot,
        color=plot_style.color_secondary,
        linewidth=plot_style.line_width,
        label=r"$s_2^{*}(t^{*})$",
    )
    axis.set_xlabel(dimless_label("t"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("s_i"), fontsize=plot_style.font_size)
    axis.set_title(
        r"Two-slider trajectories (last {:d} periods)".format(n_periods),
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Two-slider trajectory plot saved: %s", path)


def plot_two_slider_forces(
    path: Path | str,
    t: ArrayLike,
    f1_total: ArrayLike,
    f2_total: ArrayLike,
    period: float,
    *,
    n_periods: int = 2,
    style: PlotStyle | None = None,
) -> None:
    """
    2スライダーの合力 f1(t), f2(t) を重ね描きで保存する。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    t : array_like
        時刻列。
    f1_total, f2_total : array_like
        各スライダー直線方向の合力。
    period : float
        駆動周期 T。
    n_periods : int
        末尾から描画する周期数。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    t_plot, f1_plot = slice_last_n_periods(t, f1_total, period, n_periods)
    _, f2_plot = slice_last_n_periods(t, f2_total, period, n_periods)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        t_plot,
        f1_plot,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        label=r"$f_1^{*}(t^{*})$",
    )
    axis.plot(
        t_plot,
        f2_plot,
        color=plot_style.color_secondary,
        linewidth=plot_style.line_width,
        label=r"$f_2^{*}(t^{*})$",
    )
    axis.set_xlabel(dimless_label("t"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("f_i"), fontsize=plot_style.font_size)
    axis.set_title(
        r"Two-slider total forces (last {:d} periods)".format(n_periods),
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Two-slider force plot saved: %s", path)


def plot_phase_portrait_s1_s2(
    path: Path | str,
    t: ArrayLike,
    s1: ArrayLike,
    s2: ArrayLike,
    period: float,
    *,
    n_periods: int = 2,
    style: PlotStyle | None = None,
) -> None:
    """
    相図 s2 vs s1 を保存する（末尾 n 周期）。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    t : array_like
        時刻列（切り出し用）。
    s1, s2 : array_like
        各スライダー座標。
    period : float
        駆動周期 T。
    n_periods : int
        末尾から描画する周期数。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    _, s1_plot = slice_last_n_periods(t, s1, period, n_periods)
    _, s2_plot = slice_last_n_periods(t, s2, period, n_periods)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        s1_plot,
        s2_plot,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
    )
    axis.set_xlabel(dimless_label("s_1"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("s_2"), fontsize=plot_style.font_size)
    axis.set_title(
        r"Phase portrait $s_2^{{*}}$ vs $s_1^{{*}}$ (last {:d} periods)".format(n_periods),
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Phase-portrait plot saved: %s", path)


def plot_q_heatmap_delta_l(
    path: Path | str,
    delta_values: ArrayLike,
    l_values: ArrayLike,
    q_map: ArrayLike,
    *,
    style: PlotStyle | None = None,
) -> None:
    """
    Q(Delta, l) ヒートマップを保存する。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    delta_values : array_like, shape (n_delta,)
        位相差 Delta [rad] の配列（プロット時に [deg] へ変換）。
    l_values : array_like, shape (n_l,)
        スライダー間距離 l の配列。
    q_map : array_like, shape (n_l, n_delta)
        各 (l, Delta) に対応する平均流量 Q。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    delta_deg = np.degrees(delta_arr)
    l_arr = np.asarray(l_values, dtype=np.float64)
    q_arr = np.asarray(q_map, dtype=np.float64)
    if q_arr.shape != (l_arr.size, delta_arr.size):
        raise ValueError("q_map shape must be (len(l_values), len(delta_values)).")

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    image = axis.pcolormesh(
        delta_deg,
        l_arr,
        q_arr,
        shading="auto",
        cmap="coolwarm",
    )
    axis.set_xlabel(r"$\Delta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("l"), fontsize=plot_style.font_size)
    axis.set_title(
        r"$Q^{*}(\Delta,l^{*})$ heatmap",
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label(dimless_label("Q"), fontsize=plot_style.font_size)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Q heatmap saved: %s", path)


def plot_delta_opt_vs_l(
    path: Path | str,
    l_values: ArrayLike,
    delta_opt_values: ArrayLike,
    *,
    style: PlotStyle | None = None,
) -> None:
    """
    Delta_opt(l) の線図を保存する。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    l_values : array_like
        スライダー間距離 l。
    delta_opt_values : array_like
        各 l で Q を最大化する位相差 Delta_opt [rad]
        （プロット時に [deg] へ変換）。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    l_arr = np.asarray(l_values, dtype=np.float64)
    dopt_arr = np.asarray(delta_opt_values, dtype=np.float64)
    if l_arr.shape != dopt_arr.shape:
        raise ValueError("l_values and delta_opt_values must have the same shape.")
    dopt_deg = np.degrees(dopt_arr)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        l_arr,
        dopt_deg,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        marker="o",
        label=r"$\Delta_{\mathrm{opt}}(l^{*})$",
    )
    axis.set_xlabel(dimless_label("l"), fontsize=plot_style.font_size)
    axis.set_ylabel(r"$\Delta_{\mathrm{opt}}$ [deg]", fontsize=plot_style.font_size)
    axis.set_title(
        r"Optimal phase difference vs slider spacing",
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Delta-opt plot saved: %s", path)


def plot_q_vs_delta_fixed_l(
    path: Path | str,
    delta_values: ArrayLike,
    q_values: ArrayLike,
    *,
    l_fixed: float,
    style: PlotStyle | None = None,
) -> None:
    """
    固定 l における Q(Delta) を保存する（論文 Fig.4.2 相当）。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    delta_values : array_like
        位相差 Delta [rad]。
    q_values : array_like
        対応する平均流量 Q。
    l_fixed : float
        固定したスライダー間距離。
    style : PlotStyle, optional
        図の体裁。
    """
    plot_style = style if style is not None else PlotStyle()
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    q_arr = np.asarray(q_values, dtype=np.float64)
    if delta_arr.shape != q_arr.shape:
        raise ValueError("delta_values and q_values must have the same shape.")

    delta_deg = np.degrees(delta_arr)
    i_opt = int(np.argmax(q_arr))
    delta_opt_deg = float(delta_deg[i_opt])
    q_opt = float(q_arr[i_opt])

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        delta_deg,
        q_arr,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        linestyle="-",
        marker="o",
        markersize=4.0,
        label=r"$Q^{*}(\Delta)$",
    )
    axis.scatter(
        [delta_opt_deg],
        [q_opt],
        color=plot_style.color_secondary,
        s=40,
        label=r"$\Delta_{\mathrm{opt}}$",
        zorder=3,
    )
    axis.set_xlabel(r"$\Delta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("Q"), fontsize=plot_style.font_size)
    axis.set_title(
        rf"$Q^{{*}}(\Delta)$ at fixed $l^{{*}}={l_fixed:.3f}$",
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Q-vs-Delta plot saved: %s", path)


# 掃引結果の複数 l* 重ね描き用（optionrun / exp03 / exp05 共通）
SWEEP_MULTI_L_OVERLAY_DEFAULTS: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
_L_MATCH_ATOL = 1e-9


def resolve_overlay_l_values(
    l_values: ArrayLike,
    requested: Sequence[float] | None = None,
) -> list[float]:
    """
  掃引グリッド上に存在する l* のうち、重ね描き用の値を返す。

  Parameters
  ----------
  l_values : array_like
      掃引で使った l* の配列。
  requested : sequence of float, optional
      重ねたい l*。省略時は SWEEP_MULTI_L_OVERLAY_DEFAULTS。
  """
    l_arr = np.asarray(l_values, dtype=np.float64)
    req = list(requested) if requested is not None else list(SWEEP_MULTI_L_OVERLAY_DEFAULTS)
    resolved: list[float] = []
    for l_req in req:
        matches = l_arr[np.isclose(l_arr, l_req, rtol=0.0, atol=_L_MATCH_ATOL)]
        if matches.size == 0:
            available_str = ", ".join(f"{v:g}" for v in l_arr)
            raise ValueError(
                f"l*={l_req:g} not found in sweep grid. Available: {available_str}"
            )
        resolved.append(float(matches[0]))
    return resolved


def plot_q_vs_delta_multi_l(
    path: Path | str,
    *,
    l_values: list[float],
    delta_by_l: dict[float, np.ndarray],
    q_by_l: dict[float, np.ndarray],
    style: PlotStyle | None = None,
) -> None:
    """
    複数 l* における Q*(Delta) を1枚の図に重ねて保存する。

    各曲線の Q 最大点に、曲線と同色の ★ マーカーで Δ_opt を示す。
    """
    plot_style = style if style is not None else PlotStyle()
    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    opt_marker_size = 90.0
    opt_marker_edgewidth = 0.9

    for index, l_value in enumerate(l_values):
        delta_rad = delta_by_l[l_value]
        q_values = q_by_l[l_value]
        color = colors[index % len(colors)]
        delta_deg = np.degrees(delta_rad)
        axis.plot(
            delta_deg,
            q_values,
            color=color,
            linewidth=plot_style.line_width,
            linestyle="-",
            marker="o",
            markersize=3.0,
            label=rf"$l^{{*}}={l_value:g}$",
        )
        i_opt = int(np.argmax(q_values))
        delta_opt_deg = float(delta_deg[i_opt])
        q_opt = float(q_values[i_opt])
        axis.scatter(
            [delta_opt_deg],
            [q_opt],
            color=color,
            marker="*",
            s=opt_marker_size,
            edgecolors="white",
            linewidths=opt_marker_edgewidth,
            zorder=5,
        )

    opt_legend_handle = Line2D(
        [],
        [],
        linestyle="None",
        marker="*",
        markersize=10,
        markerfacecolor="0.35",
        markeredgecolor="white",
        markeredgewidth=opt_marker_edgewidth,
        label=r"$\Delta_{\mathrm{opt}}$ (peak)",
    )
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(
        legend_handles + [opt_legend_handle],
        legend_labels + [opt_legend_handle.get_label()],
        fontsize=plot_style.font_size,
        loc="best",
    )

    axis.set_xlabel(r"$\Delta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("Q"), fontsize=plot_style.font_size)
    axis.set_title(
        r"$Q^{*}(\Delta)$ at multiple $l^{*}$",
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Multi-l Q-vs-Delta plot saved: %s", path)


def plot_q_vs_delta_multi_l_from_q_map(
    path: Path | str,
    *,
    l_values: ArrayLike,
    delta_values: ArrayLike,
    q_map: ArrayLike,
    l_overlay: Sequence[float] | None = None,
    style: PlotStyle | None = None,
) -> None:
    """
    掃引 q_map から複数 l* の Q*(Delta) 重ね描きを保存する。
    """
    l_arr = np.asarray(l_values, dtype=np.float64)
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    q_arr = np.asarray(q_map, dtype=np.float64)
    if q_arr.shape != (l_arr.size, delta_arr.size):
        raise ValueError("q_map shape must be (len(l_values), len(delta_values)).")

    l_selected = resolve_overlay_l_values(l_arr, l_overlay)
    delta_by_l: dict[float, np.ndarray] = {}
    q_by_l: dict[float, np.ndarray] = {}
    for l_value in l_selected:
        i_l = int(np.where(np.isclose(l_arr, l_value, rtol=0.0, atol=_L_MATCH_ATOL))[0][0])
        delta_by_l[l_value] = delta_arr
        q_by_l[l_value] = q_arr[i_l, :]

    plot_q_vs_delta_multi_l(
        path,
        l_values=l_selected,
        delta_by_l=delta_by_l,
        q_by_l=q_by_l,
        style=style,
    )


# exp06: 複数 θ の Q(Delta) 重ね描き
def plot_q_heatmap_delta_theta(
    path: Path | str,
    delta_values: ArrayLike,
    theta_values: ArrayLike,
    q_map: ArrayLike,
    *,
    l_fixed: float,
    style: PlotStyle | None = None,
) -> None:
    """Q(Delta, theta) のヒートマップを保存する（exp06）。"""
    plot_style = style if style is not None else PlotStyle()
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    theta_arr = np.asarray(theta_values, dtype=np.float64)
    q_arr = np.asarray(q_map, dtype=np.float64)
    if q_arr.shape != (theta_arr.size, delta_arr.size):
        raise ValueError("q_map shape must be (len(theta_values), len(delta_values)).")

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    delta_deg = np.degrees(delta_arr)
    theta_deg = np.degrees(theta_arr)
    mesh = axis.pcolormesh(
        delta_deg,
        theta_deg,
        q_arr,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=axis, label=dimless_label("Q"))
    axis.set_xlabel(r"$\Delta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(r"$\theta$ [deg]", fontsize=plot_style.font_size)
    axis.set_title(
        rf"$Q^{{*}}(\Delta,\theta)$ at $l^{{*}}={l_fixed:g}$",
        fontsize=plot_style.font_size,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Q heatmap (Delta-theta) saved: %s", path)


def plot_delta_opt_vs_theta(
    path: Path | str,
    theta_values: ArrayLike,
    delta_opt_values: ArrayLike,
    *,
    l_fixed: float,
    style: PlotStyle | None = None,
) -> None:
    """Delta_opt(theta) の線図を保存する（exp06）。"""
    plot_style = style if style is not None else PlotStyle()
    theta_arr = np.asarray(theta_values, dtype=np.float64)
    dopt_arr = np.asarray(delta_opt_values, dtype=np.float64)
    if theta_arr.shape != dopt_arr.shape:
        raise ValueError("theta_values and delta_opt_values must have the same shape.")
    dopt_deg = np.degrees(dopt_arr)
    theta_deg = np.degrees(theta_arr)

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        theta_deg,
        dopt_deg,
        color=plot_style.color_primary,
        linewidth=plot_style.line_width,
        marker="o",
        label=r"$\Delta_{\mathrm{opt}}(\theta)$",
    )
    axis.set_xlabel(r"$\theta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(r"$\Delta_{\mathrm{opt}}$ [deg]", fontsize=plot_style.font_size)
    axis.set_title(
        rf"Optimal phase difference vs layout angle ($l^{{*}}={l_fixed:g}$)",
        fontsize=plot_style.font_size,
    )
    axis.legend(fontsize=plot_style.font_size)
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Delta-opt vs theta plot saved: %s", path)


def plot_q_vs_delta_multi_theta(
    path: Path | str,
    *,
    theta_values: list[float],
    delta_by_theta: dict[float, np.ndarray],
    q_by_theta: dict[float, np.ndarray],
    l_fixed: float,
    style: PlotStyle | None = None,
) -> None:
    """複数 θ における Q*(Delta) を1枚の図に重ねて保存する（exp06）。"""
    plot_style = style if style is not None else PlotStyle()
    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    opt_marker_size = 90.0
    opt_marker_edgewidth = 0.9

    for index, theta_value in enumerate(theta_values):
        delta_rad = delta_by_theta[theta_value]
        q_values = q_by_theta[theta_value]
        color = colors[index % len(colors)]
        delta_deg = np.degrees(delta_rad)
        theta_deg = float(np.degrees(theta_value))
        axis.plot(
            delta_deg,
            q_values,
            color=color,
            linewidth=plot_style.line_width,
            linestyle="-",
            marker="o",
            markersize=3.0,
            label=rf"$\theta={theta_deg:g}^\circ$",
        )
        i_opt = int(np.argmax(q_values))
        axis.scatter(
            [float(delta_deg[i_opt])],
            [float(q_values[i_opt])],
            color=color,
            marker="*",
            s=opt_marker_size,
            edgecolors="white",
            linewidths=opt_marker_edgewidth,
            zorder=5,
        )

    opt_legend_handle = Line2D(
        [],
        [],
        linestyle="None",
        marker="*",
        markersize=10,
        markerfacecolor="0.35",
        markeredgecolor="white",
        markeredgewidth=opt_marker_edgewidth,
        label=r"$\Delta_{\mathrm{opt}}$ (peak)",
    )
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(
        legend_handles + [opt_legend_handle],
        legend_labels + [opt_legend_handle.get_label()],
        fontsize=plot_style.font_size,
        loc="best",
    )
    axis.set_xlabel(r"$\Delta$ [deg]", fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("Q"), fontsize=plot_style.font_size)
    axis.set_title(
        rf"$Q^{{*}}(\Delta)$ at multiple $\theta$ ($l^{{*}}={l_fixed:g}$)",
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Multi-theta Q-vs-Delta plot saved: %s", path)


def plot_q_vs_delta_multi_theta_from_q_map(
    path: Path | str,
    *,
    theta_values: ArrayLike,
    delta_values: ArrayLike,
    q_map: ArrayLike,
    l_fixed: float,
    style: PlotStyle | None = None,
) -> None:
    """掃引 q_map から複数 θ の Q*(Delta) 重ね描きを保存する（exp06）。"""
    theta_arr = np.asarray(theta_values, dtype=np.float64)
    delta_arr = np.asarray(delta_values, dtype=np.float64)
    q_arr = np.asarray(q_map, dtype=np.float64)
    if q_arr.shape != (theta_arr.size, delta_arr.size):
        raise ValueError("q_map shape must be (len(theta_values), len(delta_values)).")

    theta_selected = [float(t) for t in theta_arr]
    delta_by_theta: dict[float, np.ndarray] = {}
    q_by_theta: dict[float, np.ndarray] = {}
    for i_theta, theta_value in enumerate(theta_selected):
        delta_by_theta[theta_value] = delta_arr
        q_by_theta[theta_value] = q_arr[i_theta, :]

    plot_q_vs_delta_multi_theta(
        path,
        theta_values=theta_selected,
        delta_by_theta=delta_by_theta,
        q_by_theta=q_by_theta,
        l_fixed=l_fixed,
        style=style,
    )
