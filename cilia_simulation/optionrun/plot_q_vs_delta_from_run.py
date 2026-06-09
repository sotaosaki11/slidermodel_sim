"""
保存済み exp03 / exp05 結果から Q(Delta) を複数 l で重ね描きする / Multi-l Q(Delta) overlay CLI.

【目的 / Purpose】
    数値実験を再実行せず、q_delta_l.csv から
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

import numpy as np

# プロジェクトルートを import パスに追加する / Allow imports from cilia_simulation root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import (
    SWEEP_MULTI_L_OVERLAY_DEFAULTS,
    plot_q_vs_delta_multi_l,
    resolve_overlay_l_values,
)

_L_MATCH_ATOL = 1e-9

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

# ▷ 実行用: 重ねる l* のリスト。None なら SWEEP_MULTI_L_OVERLAY_DEFAULTS を使う。
L_VALUES_FOR_PLAY: list[float] | None = None


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


def _parse_l_values(l_values_arg: str | None) -> list[float] | None:
    """カンマ区切り文字列または既定値から l* リストを返す / Parse l* list."""
    if l_values_arg is not None:
        parts = [part.strip() for part in l_values_arg.split(",") if part.strip()]
        if not parts:
            raise ValueError("--l-values must contain at least one value.")
        return [float(part) for part in parts]
    if L_VALUES_FOR_PLAY is not None:
        return list(L_VALUES_FOR_PLAY)
    return None


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


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する / Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Overlay Q*(Delta) curves for multiple l* from a saved sweep run "
            "(reads q_delta_l.csv; no re-simulation)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to sweep output folder. "
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
            f"(default: {','.join(f'{v:g}' for v in SWEEP_MULTI_L_OVERLAY_DEFAULTS)})."
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
    l_values = resolve_overlay_l_values(available_l, requested_l)
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
