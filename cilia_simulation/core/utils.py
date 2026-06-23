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

from core.solver import SolverConfig

logger = logging.getLogger(__name__)

SWEEP_GRID_MATCH_ATOL = 1e-9

_EXP03_NAME = "exp03_sweep_delta_l"
_EXP05_NAME = "exp05_sweep_delta_l_wall"
_EXP06_NAME = "exp06_sweep_delta_theta"
_EXP07_NAME = "exp07_sweep_phi_l"
_SUPPORTED_SWEEP_EXPERIMENTS = frozenset(
    {_EXP03_NAME, _EXP05_NAME, _EXP06_NAME, _EXP07_NAME}
)
_BLAKELET_SWEEP_EXPERIMENTS = frozenset({_EXP05_NAME, _EXP06_NAME, _EXP07_NAME})

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


@dataclass(frozen=True)
class SweepRunContext:
    """
    exp03 / exp05 sweep run の再現用コンテキスト / Context loaded from a sweep run folder.

    Attributes
    ----------
    run_dir : Path
        読み込み元 run フォルダ。
    experiment : str
        parameters.json の experiment 名。
    use_blakelet : bool
        True なら Blakelet（exp05）、False なら Oseen（exp03）。
    defaults : dict
        物理パラメータ（mu, a, k, F_0, omega, phi, h, s1_0, s2_0 など）。
    solver_config : SolverConfig
        積分器設定。
    """


    run_dir: Path
    experiment: str
    use_blakelet: bool
    defaults: dict[str, float | int]
    solver_config: SolverConfig


def solver_config_from_dict(data: dict[str, Any]) -> SolverConfig:
    """parameters.json の solver フィールドから SolverConfig を復元する。"""
    return SolverConfig(
        method=str(data["method"]),
        rtol=float(data["rtol"]),
        atol=float(data["atol"]),
        n_periods=int(data["n_periods"]),
        n_eval_per_period=int(data["n_eval_per_period"]),
        t_start=float(data.get("t_start", 0.0)),
    )


def load_sweep_run_parameters(run_dir: Path | str) -> SweepRunContext:
    """
    exp03 / exp05 / exp06 の parameters.json を読み、再積分用コンテキストを返す。

    Parameters
    ----------
    run_dir : Path or str
        output/exp03_sweep_delta_l/<timestamp>/ など。

    Returns
    -------
    SweepRunContext

    Raises
    ------
    FileNotFoundError
        parameters.json が無い場合。
    ValueError
        未対応の experiment 名の場合。
    """
    directory = Path(run_dir)
    params_path = directory / "parameters.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"parameters.json not found: {params_path}")

    with params_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    experiment = str(payload["experiment"])
    if experiment not in _SUPPORTED_SWEEP_EXPERIMENTS:
        raise ValueError(
            f"Unsupported experiment {experiment!r}. "
            f"Expected one of: {sorted(_SUPPORTED_SWEEP_EXPERIMENTS)}."
        )

    defaults_raw = payload["defaults"]
    defaults: dict[str, float | int] = dict(defaults_raw)
    for key in ("delta_points", "l_points"):
        if key in defaults:
            defaults[key] = int(defaults[key])

    return SweepRunContext(
        run_dir=directory,
        experiment=experiment,
        use_blakelet=(experiment in _BLAKELET_SWEEP_EXPERIMENTS),
        defaults=defaults,
        solver_config=solver_config_from_dict(payload["solver"]),
    )


def load_q_delta_l_csv(run_dir: Path | str) -> NDArray[np.float64]:
    """
    q_delta_l.csv を読み込む。

    Returns
    -------
    ndarray
        shape (N, 4): l, delta_rad, delta_deg, Q
    """
    csv_path = Path(run_dir) / "q_delta_l.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing q_delta_l.csv in run directory: {run_dir}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def lookup_q_delta_l_case(
    data: NDArray[np.float64],
    *,
    l_value: float,
    delta_rad: float,
    atol: float = SWEEP_GRID_MATCH_ATOL,
) -> tuple[float, float, float]:
    """
    q_delta_l.csv から (l*, Delta) に一致する行を返す。

    Returns
    -------
    l_matched, delta_matched, Q
        グリッド上の一致値と保存済み Q。

    Raises
    ------
    ValueError
        グリッド上に一致行が無い場合。
    """
    l_mask = np.isclose(data[:, 0], l_value, rtol=0.0, atol=atol)
    if not np.any(l_mask):
        raise ValueError(f"No rows for l*={l_value:g} in q_delta_l.csv.")
    subset = data[l_mask]
    delta_mask = np.isclose(subset[:, 1], delta_rad, rtol=0.0, atol=atol)
    if not np.any(delta_mask):
        raise ValueError(
            f"No rows for l*={l_value:g}, Delta={delta_rad:g} rad in q_delta_l.csv."
        )
    row = subset[delta_mask][0]
    return float(row[0]), float(row[1]), float(row[3])


def load_delta_opt_vs_l_csv(run_dir: Path | str) -> NDArray[np.float64]:
    """
    delta_opt_vs_l.csv を読み込む。

    Returns
    -------
    ndarray
        shape (N, 4): l, delta_opt_rad, delta_opt_deg, Q_max
    """
    csv_path = Path(run_dir) / "delta_opt_vs_l.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing delta_opt_vs_l.csv in run directory: {run_dir}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def lookup_delta_opt_at_l(
    *,
    run_dir: Path | str,
    l_value: float,
    atol: float = SWEEP_GRID_MATCH_ATOL,
) -> tuple[float, float, float, float]:
    """
    指定 l* における最適位相差 Delta_opt と Q_max を返す。

    delta_opt_vs_l.csv を優先し、無い場合は q_delta_l.csv から argmax で再計算する。

    Returns
    -------
    l_matched, delta_opt_rad, delta_opt_deg, Q_max
    """
    directory = Path(run_dir)
    dopt_path = directory / "delta_opt_vs_l.csv"
    if dopt_path.is_file():
        data = load_delta_opt_vs_l_csv(directory)
        l_mask = np.isclose(data[:, 0], l_value, rtol=0.0, atol=atol)
        if not np.any(l_mask):
            raise ValueError(f"No rows for l*={l_value:g} in delta_opt_vs_l.csv.")
        row = data[l_mask][0]
        return float(row[0]), float(row[1]), float(row[2]), float(row[3])

    q_data = load_q_delta_l_csv(directory)
    l_mask = np.isclose(q_data[:, 0], l_value, rtol=0.0, atol=atol)
    if not np.any(l_mask):
        raise ValueError(f"No rows for l*={l_value:g} in q_delta_l.csv.")
    subset = q_data[l_mask]
    i_peak = int(np.argmax(subset[:, 3]))
    row = subset[i_peak]
    return float(row[0]), float(row[1]), float(row[2]), float(row[3])


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


def plot_phase_portrait_convergence_s1_s2(
    path: Path | str,
    s1: ArrayLike,
    s2: ArrayLike,
    steady_start_index: int,
    *,
    l_value: float,
    delta_deg: float,
    n_periods: int,
    Q: float,
    delta_label: str | None = None,
    style: PlotStyle | None = None,
    transient_linewidth: float = 0.8,
    steady_linewidth: float = 1.5,
) -> None:
    """
    過渡収束確認用の相図 s2 vs s1 を保存する。

    全軌道を細い青、Q 評価窓（最終1周期）を赤で重ね描きする。

    Parameters
    ----------
    path : Path or str
        保存先 PNG。
    s1, s2 : array_like
        全積分区間の各スライダー座標。
    steady_start_index : int
        定常窓（Q 計算窓）の開始インデックス。
    l_value : float
        スライダー間距離 l*（タイトル用）。
    delta_deg : float
        位相差 Delta [deg]（タイトル用）。
    delta_label : str, optional
        タイトル用の Delta 表示。None なら ``$\\Delta={delta_deg:.2f}$°``。
        ``r\"$\\Delta_{\\mathrm{opt}}={delta_deg:.2f}$°\"`` などを渡す。
    n_periods : int
        積分周期数（タイトル用）。
    Q : float
        定常窓の周期平均流量 Q*（タイトル用）。
    style : PlotStyle, optional
        図の体裁。
    transient_linewidth, steady_linewidth : float
        過渡軌道・定常窓の線幅。
    """
    plot_style = style if style is not None else PlotStyle()
    s1_arr = np.asarray(s1, dtype=np.float64)
    s2_arr = np.asarray(s2, dtype=np.float64)
    idx = int(steady_start_index)
    if idx < 0 or idx >= s1_arr.size:
        raise ValueError("steady_start_index out of range for trajectory arrays.")

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    axis.plot(
        s1_arr,
        s2_arr,
        color="C0",
        linewidth=transient_linewidth,
        label="transient + approach",
    )
    axis.plot(
        s1_arr[idx:],
        s2_arr[idx:],
        color="C3",
        linewidth=steady_linewidth,
        label="Q window (last period)",
    )
    axis.set_xlabel(dimless_label("s_1"), fontsize=plot_style.font_size)
    axis.set_ylabel(dimless_label("s_2"), fontsize=plot_style.font_size)
    delta_title = (
        delta_label
        if delta_label is not None
        else rf"$\Delta={delta_deg:.2f}$°"
    )
    axis.set_title(
        (
            r"Phase portrait $s_2^{{*}}$ vs $s_1^{{*}}$: "
            rf"$l^{{*}}={l_value:g}$, {delta_title}, "
            rf"$n_{{\mathrm{{periods}}}}={n_periods}$, $Q^{{*}}={Q:.4e}$"
        ),
        fontsize=plot_style.font_size,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(
        fontsize=plot_style.font_size,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Convergence phase-portrait plot saved: %s", path)


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


def _validate_phi_l_map(
    l_values: ArrayLike,
    phi_values: ArrayLike,
    field_map: ArrayLike,
    *,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l_arr = np.asarray(l_values, dtype=np.float64)
    phi_arr = np.asarray(phi_values, dtype=np.float64)
    field_arr = np.asarray(field_map, dtype=np.float64)
    if field_arr.shape != (phi_arr.size, l_arr.size):
        raise ValueError(
            f"{field_name} shape must be (len(phi_values), len(l_values))."
        )
    if np.any(l_arr <= 0.0):
        raise ValueError("l_values must be positive for log-scale x axis.")
    return l_arr, phi_arr, field_arr


def _draw_delta_opt_map_phi_l_axis(
    axis: plt.Axes,
    l_arr: np.ndarray,
    phi_arr: np.ndarray,
    delta_opt_map: np.ndarray,
    *,
    plot_style: PlotStyle,
    show_colorbar: bool,
    fig: plt.Figure | None = None,
) -> None:
    phi_deg = np.degrees(phi_arr)
    c_norm = delta_opt_map / np.pi
    l_centers, phi_centers = np.meshgrid(l_arr, phi_deg)

    image = axis.pcolormesh(
        l_arr,
        phi_deg,
        c_norm,
        shading="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    pos_mask = delta_opt_map > 0.0
    neg_mask = delta_opt_map < 0.0
    if np.any(pos_mask):
        axis.scatter(
            l_centers[pos_mask],
            phi_centers[pos_mask],
            marker="^",
            c=c_norm[pos_mask],
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=22.0,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
    if np.any(neg_mask):
        axis.scatter(
            l_centers[neg_mask],
            phi_centers[neg_mask],
            marker="v",
            c=c_norm[neg_mask],
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=22.0,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
    axis.contour(
        l_centers,
        phi_centers,
        c_norm,
        levels=[-0.5, 0.5],
        colors="black",
        linestyles="--",
        linewidths=0.8,
    )
    axis.set_xscale("log")
    axis.set_xlabel(dimless_label("l"), fontsize=plot_style.font_size)
    axis.set_ylabel(r"$\varphi$ [deg]", fontsize=plot_style.font_size)
    axis.set_title(
        r"Optimal $\Delta/\pi$ (numerical)",
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    if show_colorbar:
        if fig is None:
            fig = axis.figure
        cbar = fig.colorbar(image, ax=axis)
        cbar.set_label(r"$\Delta/\pi$", fontsize=plot_style.font_size)


def plot_delta_opt_map_phi_l(
    path: Path | str,
    l_values: ArrayLike,
    phi_values: ArrayLike,
    delta_opt_map: ArrayLike,
    *,
    style: PlotStyle | None = None,
) -> None:
    """最適 Delta/pi の phi-l マップ（Fig.6 パネル a 形式）を保存する。"""
    plot_style = style if style is not None else PlotStyle()
    l_arr, phi_arr, delta_arr = _validate_phi_l_map(
        l_values,
        phi_values,
        delta_opt_map,
        field_name="delta_opt_map",
    )

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    _draw_delta_opt_map_phi_l_axis(
        axis,
        l_arr,
        phi_arr,
        delta_arr,
        plot_style=plot_style,
        show_colorbar=True,
        fig=fig,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Delta-opt phi-l map saved: %s", path)


def _draw_qmax_map_phi_l_axis(
    axis: plt.Axes,
    l_arr: np.ndarray,
    phi_arr: np.ndarray,
    qmax_map: np.ndarray,
    *,
    plot_style: PlotStyle,
    show_colorbar: bool,
    fig: plt.Figure | None = None,
) -> None:
    phi_deg = np.degrees(phi_arr)
    log_q = np.log10(qmax_map)
    l_centers, phi_centers = np.meshgrid(l_arr, phi_deg)

    image = axis.pcolormesh(
        l_arr,
        phi_deg,
        log_q,
        shading="auto",
        cmap="viridis",
    )
    axis.contour(
        l_centers,
        phi_centers,
        log_q,
        levels=[-4.0, -3.0, -2.0],
        colors="white",
        linestyles="--",
        linewidths=0.8,
    )
    axis.set_xscale("log")
    axis.set_xlabel(dimless_label("l"), fontsize=plot_style.font_size)
    axis.set_ylabel(r"$\varphi$ [deg]", fontsize=plot_style.font_size)
    axis.set_title(
        r"$\log_{10} Q^{*}_{\max}$ (numerical)",
        fontsize=plot_style.font_size,
    )
    axis.grid(True, alpha=plot_style.grid_alpha)
    if show_colorbar:
        if fig is None:
            fig = axis.figure
        cbar = fig.colorbar(image, ax=axis)
        cbar.set_label(
            r"$\log_{10} Q^{*}_{\max}$",
            fontsize=plot_style.font_size,
        )


def plot_qmax_map_phi_l(
    path: Path | str,
    l_values: ArrayLike,
    phi_values: ArrayLike,
    qmax_map: ArrayLike,
    *,
    style: PlotStyle | None = None,
) -> None:
    """最大流量 log10(Q_max) の phi-l マップ（Fig.6 パネル b 形式）を保存する。"""
    plot_style = style if style is not None else PlotStyle()
    l_arr, phi_arr, qmax_arr = _validate_phi_l_map(
        l_values,
        phi_values,
        qmax_map,
        field_name="qmax_map",
    )

    fig, axis = plt.subplots(
        figsize=(plot_style.figure_width, plot_style.figure_height),
        dpi=plot_style.dpi,
    )
    _draw_qmax_map_phi_l_axis(
        axis,
        l_arr,
        phi_arr,
        qmax_arr,
        plot_style=plot_style,
        show_colorbar=True,
        fig=fig,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Q-max phi-l map saved: %s", path)


def plot_fig6_style_phi_l(
    path: Path | str,
    l_values: ArrayLike,
    phi_values: ArrayLike,
    delta_opt_map: ArrayLike,
    qmax_map: ArrayLike,
    *,
    style: PlotStyle | None = None,
) -> None:
    """Fig.6 形式（Delta/pi + log10 Q_max）の縦2段マップを1枚に保存する。"""
    plot_style = style if style is not None else PlotStyle()
    l_arr, phi_arr, delta_arr = _validate_phi_l_map(
        l_values,
        phi_values,
        delta_opt_map,
        field_name="delta_opt_map",
    )
    _, _, qmax_arr = _validate_phi_l_map(
        l_values,
        phi_values,
        qmax_map,
        field_name="qmax_map",
    )

    fig, (axis_a, axis_b) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(plot_style.figure_width, plot_style.figure_height * 2.0),
        dpi=plot_style.dpi,
    )
    _draw_delta_opt_map_phi_l_axis(
        axis_a,
        l_arr,
        phi_arr,
        delta_arr,
        plot_style=plot_style,
        show_colorbar=True,
        fig=fig,
    )
    _draw_qmax_map_phi_l_axis(
        axis_b,
        l_arr,
        phi_arr,
        qmax_arr,
        plot_style=plot_style,
        show_colorbar=True,
        fig=fig,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Fig.6-style phi-l map saved: %s", path)
