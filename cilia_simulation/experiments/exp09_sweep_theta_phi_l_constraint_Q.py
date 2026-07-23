"""
第9段階実験: exp08 と同型の theta×phi×l 掃引（流量に拘束力を含める）/ Experiment 09.

目的:
    exp08 と同一の掃引グリッド・積分設定で、流量評価のみを変更する。
    F_x に拘束力 λ（e_s⊥ 方向）の x 成分を含め、定義の違いが結果にどう効くかを比較する。

    include_constraint_force_in_Q = True（exp08 は False）。

実行例:
    python experiments/exp09_sweep_theta_phi_l_constraint_Q.py --mode fast --workers 4
    python experiments/exp09_sweep_theta_phi_l_constraint_Q.py --mode fine --workers 4

IDE ▷ 実行:
    config/default_params.py の EXP09_DEFAULT_MODE などを変更する。
    CLI の --mode はこの値を上書きする。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import (
    EXP09_BOUNDARY_L_VALUES,
    EXP09_DEFAULT_MODE,
    EXP09_IC_SOLVER_PRESET,
    EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    EXP09_THETA_MAX_DEG,
    EXP09_THETA_MIN_DEG,
    EXP09_THETA_STEP_DEG,
    Exp09Mode,
    resolve_exp09_config,
)
from core.sweep import default_worker_count
from experiments.exp08_sweep_theta_phi_l import run_experiment as _run_theta_phi_l_sweep

EXP_NAME = "exp09_sweep_theta_phi_l_constraint_Q"


def run_experiment(
    *,
    mode: Exp09Mode = "fast",
    workers: int | None = None,
) -> Path:
    """
    exp08 と同型スイープ。流量 Q に拘束力を含める（exp09）。
    """
    return _run_theta_phi_l_sweep(
        mode=mode,
        workers=workers,
        exp_name=EXP_NAME,
        resolve_config=resolve_exp09_config,
        include_constraint_force_in_Q=EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q,
        ic_solver_preset=EXP09_IC_SOLVER_PRESET,
        boundary_l_values=EXP09_BOUNDARY_L_VALUES,
        log_prefix="exp09",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "exp09: same as exp08 but Q includes constraint force in F_x "
            "(include_constraint_force_in_Q=True)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "fine"),
        default=EXP09_DEFAULT_MODE,
        help=(
            "fast: coarse Delta grid + Euler. fine: dense grid + RK45. "
            f"Default from config: EXP09_DEFAULT_MODE={EXP09_DEFAULT_MODE!r}."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Default: cpu_count - 1. Use 1 for serial.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    workers = args.workers if args.workers is not None else default_worker_count()
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    logging.getLogger(__name__).info(
        "Starting exp09 sweep: mode=%s, workers=%d, theta=[%.1f, %.1f] step %.1f deg, "
        "include_constraint_force_in_Q=%s",
        args.mode,
        workers,
        EXP09_THETA_MIN_DEG,
        EXP09_THETA_MAX_DEG,
        EXP09_THETA_STEP_DEG,
        EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q,
    )
    run_dir = run_experiment(mode=args.mode, workers=workers)
    logging.getLogger(__name__).info("exp09 sweep finished. Output: %s", run_dir)


if __name__ == "__main__":
    main()
