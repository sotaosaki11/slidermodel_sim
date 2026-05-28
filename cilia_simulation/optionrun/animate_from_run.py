"""
保存済み exp01 結果から Stokeslet アニメーションを生成する / Animation CLI.

【目的 / Purpose】
    数値実験（exp01）を再実行せず、parameters.json と flow_rate.csv から
    スライダーと流体（xz 流線）のアニメーションをオンデマンドで作成する。

【実行方法 / How to run】
    (A) IDE の▷（Run Python File）: 下の RUN_DIR_FOR_PLAY を設定するか、
        None のままなら output/exp01_single_slider/ の最新 run を自動選択。
    (B) ターミナル:
        cd cilia_simulation
        python optionrun/animate_from_run.py
        python optionrun/animate_from_run.py --run-dir output/exp01_single_slider/<YYYYMMDD_HHMMSS>
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# プロジェクトルートを import パスに追加する / Allow imports from cilia_simulation root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.animation_defaults import AnimationConfig
from core.animation import (
    load_run_directory,
    make_animation_output_directory,
    render_animation,
)

# ==========================================
# パラメータ設定セクション（▷実行・CLI 既定値）
# ==========================================

DEFAULT_EXP_NAME = "exp01_single_slider"
EXP01_OUTPUT_BASE = Path("output") / DEFAULT_EXP_NAME
DEFAULT_OUTPUT_BASE = "output/animations"

# ▷ 実行用: 固定の run を使う場合はパスを書く。None なら最新 timestamp を自動選択。
RUN_DIR_FOR_PLAY: str | None = None
# 例: RUN_DIR_FOR_PLAY = "output/exp01_single_slider/20260526_184029"

# 引数なし（▷）のとき GIF も保存するか
SAVE_GIF_WHEN_NO_ARGS = True


def _find_latest_run_directory() -> Path:
    """
    exp01 出力のうち最新の timestamp フォルダを返す / Newest run folder by name.
    """
    base = PROJECT_ROOT / EXP01_OUTPUT_BASE
    if not base.is_dir():
        raise FileNotFoundError(f"exp01 output base not found: {base}")

    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir()
        and (path / "parameters.json").is_file()
        and (path / "flow_rate.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No valid run folders under {base}. Run exp01_single_slider.py first."
        )
    return max(candidates, key=lambda path: path.name)


def _resolve_run_directory(run_dir_arg: str | None) -> Path:
    """
    読み込む run フォルダを決める / Resolve run directory from CLI or play defaults.
    """
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


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する / Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Stokeslet animation from a saved exp01 run folder.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to exp01 output folder. "
            "If omitted, uses RUN_DIR_FOR_PLAY or the latest run under output/exp01_single_slider/."
        ),
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also save animation.gif (in addition to MP4 when ffmpeg is available).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override number of animation frames.",
    )
    parser.add_argument(
        "--loop-count",
        type=int,
        default=None,
        help="How many periods to loop in the animation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: output/animations/...).",
    )
    return parser.parse_args()


def main() -> None:
    """アニメーションを1回生成する / Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    log = logging.getLogger(__name__)

    run_dir = _resolve_run_directory(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    log.info("Using run directory: %s", run_dir)

    run_data = load_run_directory(run_dir)

    anim_config = AnimationConfig()
    if args.frames is not None:
        anim_config = replace(anim_config, n_frames=args.frames)
    if args.loop_count is not None:
        anim_config = replace(anim_config, loop_count=args.loop_count)

    save_gif = args.gif
    if len(sys.argv) == 1 and SAVE_GIF_WHEN_NO_ARGS:
        save_gif = True

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = make_animation_output_directory(
            DEFAULT_EXP_NAME,
            run_data.source_run_dir,
            base=PROJECT_ROOT / DEFAULT_OUTPUT_BASE,
        )

    primary = render_animation(
        run_data,
        out_dir,
        config=anim_config,
        save_gif=save_gif,
    )
    log.info("Animation finished: %s", primary)


if __name__ == "__main__":
    main()
