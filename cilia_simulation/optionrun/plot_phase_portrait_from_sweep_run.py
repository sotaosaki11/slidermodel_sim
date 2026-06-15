"""
保存済み exp03 / exp05 / exp06 sweep 結果から過渡収束確認用相図を描く CLI.

【目的 / Purpose】
    parameters.json と sweep CSV を読み、指定 l* で1ケース再積分し、
    s2* vs s1* 相図を「全軌道（細い青）＋ Q 評価窓（赤）」で保存する。
    Delta 未指定時は delta_opt_vs_l.csv からその l* の Delta_opt を自動選択する。

【実行方法 / How to run】
    (A) IDE の▷（Run Python File）: 下の RUN_DIR_FOR_PLAY, L_FOR_PLAY を設定。
    (B) ターミナル:
        cd cilia_simulation
        python optionrun/plot_phase_portrait_from_sweep_run.py \\
            --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS> \\
            --l 2.0
        python optionrun/plot_phase_portrait_from_sweep_run.py \\
            --run-dir output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS> \\
            --l 2.0 --delta-deg 90
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

# プロジェクトルートを import パスに追加する / Allow imports from cilia_simulation root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flow_rate import FlowCalculator
from core.two_slider import integrate_two_slider_case
from core.utils import (
    load_q_delta_l_csv,
    load_sweep_run_parameters,
    lookup_delta_opt_at_l,
    lookup_q_delta_l_case,
    plot_phase_portrait_convergence_s1_s2,
)

Q_VERIFY_RTOL = 1e-6

# ==========================================
# パラメータ設定セクション（▷実行・CLI 既定値）
# ==========================================

DEFAULT_EXP_NAMES = (
    "exp03_sweep_delta_l",
    "exp05_sweep_delta_l_wall",
    "exp06_sweep_delta_theta",
)
PARAMETERS_JSON = "parameters.json"
Q_DELTA_L_CSV = "q_delta_l.csv"

# ▷ 実行用: 固定の run を使う場合はパスを書く。None なら最新 timestamp を自動選択。
RUN_DIR_FOR_PLAY: str | None = None

# ▷ 実行用: l*（必須）。Delta 未指定ならその l* の Delta_opt を使用。
L_FOR_PLAY: float | None = None
DELTA_FOR_PLAY: float | None = None
DELTA_DEG_FOR_PLAY: bool = False


def _find_latest_run_directory() -> Path:
    """exp03 / exp05 / exp06 出力のうち最新の有効 run フォルダを返す。"""
    candidates: list[Path] = []
    for exp_name in DEFAULT_EXP_NAMES:
        base = PROJECT_ROOT / "output" / exp_name
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if not path.is_dir():
                continue
            if (path / PARAMETERS_JSON).is_file() and (path / Q_DELTA_L_CSV).is_file():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No valid exp03/exp05/exp06 run folders found under output/. "
            "Run a sweep experiment first."
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


def _default_output_path(
    run_dir: Path,
    l_value: float,
    delta_deg: float,
    *,
    at_delta_opt: bool,
) -> Path:
    delta_label = f"deltaOpt{delta_deg:.2f}".replace(".", "p") if at_delta_opt else (
        f"delta{delta_deg:.2f}".replace(".", "p")
    )
    return run_dir / f"phase_portrait_s1_vs_s2_l{l_value:g}_{delta_label}deg.png"


def _verify_q_match(Q_recomputed: float, Q_csv: float, *, rtol: float = Q_VERIFY_RTOL) -> None:
    """再積分 Q と CSV Q の一致をログ用に検証する。"""
    if not math.isfinite(Q_recomputed) or not math.isfinite(Q_csv):
        raise ValueError(f"Non-finite Q values: recomputed={Q_recomputed}, csv={Q_csv}")
    denom = max(abs(Q_csv), 1e-30)
    rel_err = abs(Q_recomputed - Q_csv) / denom
    if rel_err > rtol:
        raise ValueError(
            f"Q mismatch: recomputed={Q_recomputed:.8e}, csv={Q_csv:.8e}, "
            f"relative error={rel_err:.3e} > rtol={rtol:g}"
        )


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する / Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot s2* vs s1* phase portrait with full trajectory (blue) and "
            "Q evaluation window (red) from a saved sweep run. "
            "If Delta is omitted, uses Delta_opt at the given l*."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to sweep output folder. "
            "If omitted, uses RUN_DIR_FOR_PLAY or the latest sweep run."
        ),
    )
    parser.add_argument(
        "--l",
        type=float,
        default=None,
        help="Slider separation l* (required unless L_FOR_PLAY is set for IDE run).",
    )
    delta_group = parser.add_mutually_exclusive_group()
    delta_group.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Phase difference Delta [rad]. Omit to use Delta_opt at l*.",
    )
    delta_group.add_argument(
        "--delta-deg",
        type=float,
        default=None,
        help="Phase difference Delta [deg]. Omit to use Delta_opt at l*.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: <run-dir>/phase_portrait_s1_vs_s2_l{l}_....png).",
    )
    return parser.parse_args()


def _resolve_l_value(args: argparse.Namespace) -> float:
    l_value = args.l if args.l is not None else L_FOR_PLAY
    if l_value is None:
        raise ValueError("--l is required (or set L_FOR_PLAY for IDE run).")
    return float(l_value)


def _resolve_delta_rad(args: argparse.Namespace) -> float | None:
    """CLI / IDE から明示 Delta [rad]。未指定なら None（Delta_opt モード）。"""
    if args.delta_deg is not None:
        return math.radians(args.delta_deg)
    if args.delta is not None:
        return float(args.delta)
    if DELTA_FOR_PLAY is not None:
        return math.radians(DELTA_FOR_PLAY) if DELTA_DEG_FOR_PLAY else float(DELTA_FOR_PLAY)
    return None


def main() -> None:
    """過渡収束確認用相図を1回生成する / Entry point."""
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

    l_value = _resolve_l_value(args)
    delta_rad_explicit = _resolve_delta_rad(args)
    at_delta_opt = delta_rad_explicit is None

    if at_delta_opt:
        l_matched, delta_rad, delta_deg, Q_csv = lookup_delta_opt_at_l(
            run_dir=run_dir,
            l_value=l_value,
        )
        l_value = l_matched
        log.info(
            "Delta_opt at l*=%g: Delta=%g rad (%.4f deg), Q_max=%.8e",
            l_value,
            delta_rad,
            delta_deg,
            Q_csv,
        )
    else:
        assert delta_rad_explicit is not None
        delta_rad = delta_rad_explicit
        delta_deg = math.degrees(delta_rad)
        csv_data = load_q_delta_l_csv(run_dir)
        _, delta_rad, Q_csv = lookup_q_delta_l_case(
            csv_data,
            l_value=l_value,
            delta_rad=delta_rad,
        )
        log.info(
            "Case: l*=%g, Delta=%g rad (%.4f deg); Q from CSV=%.8e",
            l_value,
            delta_rad,
            delta_deg,
            Q_csv,
        )

    sweep_ctx = load_sweep_run_parameters(run_dir)
    defaults = sweep_ctx.defaults
    mu = float(defaults["mu"])
    h = float(defaults["h"])
    phi = float(defaults["phi"])
    layout_theta = float(defaults.get("layout_theta", 0.0))

    result = integrate_two_slider_case(
        mu=mu,
        a=float(defaults["a"]),
        k=float(defaults["k"]),
        F_0=float(defaults["F_0"]),
        omega=float(defaults["omega"]),
        phi=phi,
        h=h,
        l=l_value,
        delta=delta_rad,
        s1_0=float(defaults["s1_0"]),
        s2_0=float(defaults["s2_0"]),
        solver_config=sweep_ctx.solver_config,
        use_blakelet=sweep_ctx.use_blakelet,
        layout_theta=layout_theta,
    )

    flow = FlowCalculator(mu=mu, h=h)
    _, _, Q_recomputed = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
    )
    rel_err = abs(Q_recomputed - Q_csv) / max(abs(Q_csv), 1e-30)
    log.info(
        "Q recomputed=%.8e, csv=%.8e, relative error=%.3e",
        Q_recomputed,
        Q_csv,
        rel_err,
    )
    _verify_q_match(Q_recomputed, Q_csv)

    if args.output is not None:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = _default_output_path(
            run_dir,
            l_value,
            delta_deg,
            at_delta_opt=at_delta_opt,
        )

    delta_label = (
        rf"$\Delta_{{\mathrm{{opt}}}}={delta_deg:.2f}$°"
        if at_delta_opt
        else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_phase_portrait_convergence_s1_s2(
        out_path,
        result.s1,
        result.s2,
        result.steady_start_index,
        l_value=l_value,
        delta_deg=delta_deg,
        n_periods=sweep_ctx.solver_config.n_periods,
        Q=Q_recomputed,
        delta_label=delta_label,
    )
    log.info("Phase portrait saved: %s", out_path)


if __name__ == "__main__":
    main()
