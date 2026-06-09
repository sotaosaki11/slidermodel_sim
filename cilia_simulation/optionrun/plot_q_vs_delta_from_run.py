"""
保存済み exp03 結果から Q(Delta) を複数 l で重ね描きする / Multi-l Q(Delta) overlay CLI.

【目的 / Purpose】
    数値実験（exp03）を再実行せず、q_delta_l.csv から
    複数の l における Q*(Delta) 曲線を1枚の図に重ねて保存する。

【実行方法 / How to run】
    (A) IDE の▷（Run Python File）: 下の RUN_DIR_FOR_PLAY と L_VALUES_FOR_PLAY を設定。
    (B) ターミナル:
        cd cilia_simulation
        python optionrun/plot_q_vs_delta_from_run.py
        python optionrun/plot_q_vs_delta_from_run.py --run-dir output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS>
        python optionrun/plot_q_vs_delta_from_run.py --l-values 1.5,2.0,3.0,4.0,5.0,6.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# プロジェクトルートを import パスに追加する / Allow imports from cilia_simulation root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import PlotStyle, dimless_label

# ==========================================
# パラメータ設定セクション（▷実行・CLI 既定値）
# ==========================================

DEFAULT_EXP_NAME = "exp03_sweep_delta_l"
EXP03_OUTPUT_BASE = Path("output") / DEFAULT_EXP_NAME
Q_DELTA_L_CSV = "q_delta_l.csv"
DEFAULT_OUTPUT_NAME = "Q_vs_delta_multi_l.png"

# ▷ 実行用: 固定の run を使う場合はパスを書く。None なら最新 timestamp を自動選択。
RUN_DIR_FOR_PLAY: str | None = None
RUN_DIR_FOR_PLAY = "output/exp03_sweep_delta_l/20260608_180938"

# ▷ 実行用: 重ねる l* のリスト。None なら下の DEFAULT_L_VALUES を使う。
L_VALUES_FOR_PLAY: list[float] | None = None

# 引数なし（▷）のときの既定 l*（0.25 刻みの掃引点から代表的な値を選ぶ）
DEFAULT_L_VALUES: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)

_L_MATCH_ATOL = 1e-9


def _find_latest_run_directory() -> Path:
    """exp03 出力のうち最新の timestamp フォルダを返す / Newest run folder by name."""
    base = PROJECT_ROOT / EXP03_OUTPUT_BASE
    if not base.is_dir():
        raise FileNotFoundError(f"exp03 output base not found: {base}")

    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and (path / Q_DELTA_L_CSV).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid run folders under {base}. Run exp03_sweep_delta_l.py first."
        )
    return max(candidates, key=lambda path: path.name)


def _resolve_run_directory(run_dir_arg: str | None) -> Path:
    """読み込む run フォルダを決める / Resolve run directory from CLI or play defaults."""
    if run_dir_arg is not None:
        chosen = run_dir_arg
    elif RUN_DIR_FOR_PLAY is not None:
        chosen = RUN_DIR_FOR_PLAY
    else:
        return _find_latest_run_directory()

    path = Path(chosen)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _parse_l_values(l_values_arg: str | None) -> list[float]:
    """カンマ区切り文字列または既定値から l* リストを返す / Parse l* list."""
    if l_values_arg is not None:
        parts = [part.strip() for part in l_values_arg.split(",") if part.strip()]
        if not parts:
            raise ValueError("--l-values must contain at least one value.")
        return [float(part) for part in parts]
    if L_VALUES_FOR_PLAY is not None:
        return list(L_VALUES_FOR_PLAY)
    return list(DEFAULT_L_VALUES)


def _load_q_delta_l_csv(run_dir: Path) -> np.ndarray:
    """
    q_delta_l.csv を読み込む。

    Returns
    -------
    ndarray
        shape (N, 4): l, delta_rad, delta_deg, Q
    """
    csv_path = run_dir / Q_DELTA_L_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {Q_DELTA_L_CSV} in run directory: {run_dir}")
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)


def _unique_l_values_in_csv(data: np.ndarray) -> np.ndarray:
    """CSV に含まれる l* の一意値（昇順）を返す / Sorted unique l* in data."""
    return np.unique(data[:, 0])


def _resolve_l_values_against_csv(
    requested: list[float],
    available: np.ndarray,
) -> list[float]:
    """
    要求された l* を CSV 上の実値に対応付ける。見つからない値はエラー。
    """
    resolved: list[float] = []
    for l_req in requested:
        matches = available[np.isclose(available, l_req, rtol=0.0, atol=_L_MATCH_ATOL)]
        if matches.size == 0:
            available_str = ", ".join(f"{v:g}" for v in available)
            raise ValueError(
                f"l*={l_req:g} not found in q_delta_l.csv. Available: {available_str}"
            )
        resolved.append(float(matches[0]))
    return resolved


def _extract_q_vs_delta(
    data: np.ndarray,
    *,
    l_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """1つの l* に対応する (delta_rad, Q) を返す / Slice Q(Delta) at fixed l*."""
    mask = np.isclose(data[:, 0], l_value, rtol=0.0, atol=_L_MATCH_ATOL)
    if not np.any(mask):
        raise ValueError(f"No rows for l*={l_value:g} in q_delta_l.csv.")
    subset = data[mask]
    order = np.argsort(subset[:, 1])
    subset = subset[order]
    return subset[:, 1], subset[:, 3]


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


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する / Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Overlay Q*(Delta) curves for multiple l* from a saved exp03 run "
            "(reads q_delta_l.csv; no re-simulation)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to exp03 output folder. "
            "If omitted, uses RUN_DIR_FOR_PLAY or the latest run under "
            "output/exp03_sweep_delta_l/."
        ),
    )
    parser.add_argument(
        "--l-values",
        type=str,
        default=None,
        help=(
            "Comma-separated l* values to overlay "
            f"(default: {','.join(f'{v:g}' for v in DEFAULT_L_VALUES)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            f"Output PNG path (default: <run-dir>/{DEFAULT_OUTPUT_NAME})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Q(Delta) 重ね描きを1回実行する / Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    args = _parse_args()

    run_dir = _resolve_run_directory(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    log.info("Using run directory: %s", run_dir)

    data = _load_q_delta_l_csv(run_dir)
    available_l = _unique_l_values_in_csv(data)
    requested_l = _parse_l_values(args.l_values)
    l_values = _resolve_l_values_against_csv(requested_l, available_l)
    log.info("Overlay l* values: %s", ", ".join(f"{v:g}" for v in l_values))

    delta_by_l: dict[float, np.ndarray] = {}
    q_by_l: dict[float, np.ndarray] = {}
    for l_value in l_values:
        delta_rad, q_values = _extract_q_vs_delta(data, l_value=l_value)
        delta_by_l[l_value] = delta_rad
        q_by_l[l_value] = q_values

    if args.output is not None:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = run_dir / DEFAULT_OUTPUT_NAME

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_q_vs_delta_multi_l(
        out_path,
        l_values=l_values,
        delta_by_l=delta_by_l,
        q_by_l=q_by_l,
    )
    log.info("Multi-l Q-vs-Delta plot saved: %s", out_path)


if __name__ == "__main__":
    main()
