"""
非結合解析 IC と Blakelet 数値積分の比較検証 / Validate initial conditions.

壁あり Blakelet 単独スライダー積分（exp08 本番 IC）と、
自由空間スカラー gamma_0 解析式（旧近似）を Delta×phi グリッド上で比較する。

実行例:
    cd cilia_simulation
    python optionrun/validate_decoupled_initial_positions.py
    python optionrun/validate_decoupled_initial_positions.py \\
        --output-dir output/optionrun/ic_validation
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import EXP08_IC_SOLVER_PRESET, resolve_exp08_config
from core.initial_conditions import build_initial_position_lookup
from core.solver import SolverConfig
from core.utils import PlotStyle, save_summary

DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "output" / "optionrun" / "ic_validation"
DEFAULT_DELTA_POINTS = 37
DEFAULT_PHI_POINTS = 5
REPRESENTATIVE_PHI_DEG = 45.0
PLOT_STYLE = PlotStyle()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stokes analytic IC with Blakelet single-slider "
            "numerical steady-state initial positions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--delta-points",
        type=int,
        default=DEFAULT_DELTA_POINTS,
        help=f"Number of Delta samples (default: {DEFAULT_DELTA_POINTS})",
    )
    parser.add_argument(
        "--phi-points",
        type=int,
        default=DEFAULT_PHI_POINTS,
        help=f"Number of phi samples (default: {DEFAULT_PHI_POINTS})",
    )
    return parser.parse_args()


def validate_decoupled_initial_positions(
    *,
    output_dir: Path,
    delta_points: int = DEFAULT_DELTA_POINTS,
    phi_points: int = DEFAULT_PHI_POINTS,
) -> Path:
    sweep_defaults, _ = resolve_exp08_config("fast")
    mu = float(sweep_defaults["mu"])
    a = float(sweep_defaults["a"])
    k = float(sweep_defaults["k"])
    F_0 = float(sweep_defaults["F_0"])
    omega = float(sweep_defaults["omega"])
    h = float(sweep_defaults["h"])

    delta_values = np.linspace(-math.pi, math.pi, delta_points, endpoint=False)
    phi_deg_grid = np.linspace(0.0, 85.0, phi_points, endpoint=True)
    phi_deg_grid = np.unique(np.append(phi_deg_grid, REPRESENTATIVE_PHI_DEG))
    phi_values = np.radians(phi_deg_grid)

    ic_solver_config = SolverConfig(
        method=str(EXP08_IC_SOLVER_PRESET["method"]),
        rtol=float(EXP08_IC_SOLVER_PRESET["rtol"]),
        atol=float(EXP08_IC_SOLVER_PRESET["atol"]),
        n_periods=int(EXP08_IC_SOLVER_PRESET["n_periods"]),
        n_eval_per_period=int(EXP08_IC_SOLVER_PRESET["n_eval_per_period"]),
    )

    s1_stokes, s2_stokes = build_initial_position_lookup(
        delta_values,
        method="decoupled_analytic",
        phi_values=phi_values,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        h=h,
    )
    s1_blakelet, s2_blakelet = build_initial_position_lookup(
        delta_values,
        method="decoupled_blakelet",
        phi_values=phi_values,
        mu=mu,
        a=a,
        k=k,
        F_0=F_0,
        omega=omega,
        h=h,
        ic_solver_config=ic_solver_config,
    )

    s1_err = np.abs(s1_stokes - s1_blakelet)
    s2_err = np.abs(s2_stokes - s2_blakelet)
    max_s1_err = float(np.max(s1_err))
    max_s2_err = float(np.max(s2_err))

    output_dir.mkdir(parents=True, exist_ok=True)
    delta_deg = np.degrees(delta_values)
    phi_deg = np.degrees(phi_values)

    representative_phi_idx = int(np.argmin(np.abs(phi_deg - REPRESENTATIVE_PHI_DEG)))
    representative_phi_deg = float(phi_deg[representative_phi_idx])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(
        delta_deg,
        s1_stokes[representative_phi_idx],
        label=rf"$s_{{1,0}}$ Stokes analytic ($\phi={representative_phi_deg:.1f}°$)",
        color="C0",
    )
    axes[0].plot(
        delta_deg,
        s1_blakelet[representative_phi_idx],
        linestyle="--",
        label=rf"$s_{{1,0}}$ Blakelet numerical",
        color="C1",
    )
    axes[0].set_ylabel(r"$s_{1,0}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        delta_deg,
        np.full_like(delta_deg, s2_stokes[representative_phi_idx]),
        label=rf"$s_{{2,0}}$ Stokes analytic ($\phi={representative_phi_deg:.1f}°$)",
        color="C0",
    )
    axes[1].plot(
        delta_deg,
        np.full_like(delta_deg, s2_blakelet[representative_phi_idx]),
        linestyle="--",
        label=rf"$s_{{2,0}}$ Blakelet numerical",
        color="C1",
    )
    axes[1].set_xlabel(r"$\Delta$ [deg]")
    axes[1].set_ylabel(r"$s_{2,0}$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    y_vals = np.concatenate(
        [
            s1_stokes[representative_phi_idx],
            s1_blakelet[representative_phi_idx],
            np.full(delta_values.size, s2_stokes[representative_phi_idx]),
            np.full(delta_values.size, s2_blakelet[representative_phi_idx]),
        ]
    )
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.01
    shared_ylim = (y_min - y_margin, y_max + y_margin)
    for axis in axes:
        axis.set_ylim(shared_ylim)

    fig.suptitle(
        "Initial positions: Stokes analytic vs Blakelet single-slider (decoupled)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "initial_positions_vs_delta.png", dpi=PLOT_STYLE.dpi)
    plt.close(fig)

    phi_grid, delta_grid = np.meshgrid(phi_deg, delta_deg, indexing="ij")
    comparison = np.column_stack(
        [
            phi_grid.ravel(),
            delta_grid.ravel(),
            s1_stokes.ravel(),
            s1_blakelet.ravel(),
            s1_err.ravel(),
            np.repeat(s2_stokes, delta_values.size),
            np.repeat(s2_blakelet, delta_values.size),
            np.repeat(s2_err, delta_values.size),
        ]
    )
    np.savetxt(
        output_dir / "initial_positions_comparison.csv",
        comparison,
        delimiter=",",
        header=(
            "phi_deg,delta_deg,s1_stokes_analytic,s1_blakelet_numerical,s1_abs_err,"
            "s2_stokes_analytic,s2_blakelet_numerical,s2_abs_err"
        ),
        comments="",
        encoding="utf-8",
    )

    summary_lines = [
        "validation: Stokes analytic vs Blakelet single-slider IC",
        f"delta_points: {delta_points}",
        f"phi_points: {phi_points}",
        f"mu: {mu}",
        f"a: {a}",
        f"k: {k}",
        f"F_0: {F_0}",
        f"omega: {omega}",
        f"h: {h}",
        f"representative_phi_deg: {representative_phi_deg:.1f}",
        f"max_abs_err_s1: {max_s1_err:.6e}",
        f"max_abs_err_s2: {max_s2_err:.6e}",
        "",
        "Blakelet IC uses wall self-mobility only (decoupled, no cross terms).",
        "Differences from Stokes analytic reflect wall correction and phi dependence.",
    ]
    save_summary(output_dir / "summary.txt", summary_lines)
    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    output_dir = validate_decoupled_initial_positions(
        output_dir=args.output_dir,
        delta_points=args.delta_points,
        phi_points=args.phi_points,
    )
    logging.getLogger(__name__).info("Validation finished. Output: %s", output_dir)


if __name__ == "__main__":
    main()
