"""
exp08 保存済み CSV から論文向け図を再生成する CLI.

【目的】
    delta_opt_map_all_theta.csv / global_opt_vs_theta.csv /
    inversion_boundary.csv を読み、長時間計算なしで全図を再描画する。

【実行例】
    cd cilia_simulation
    python optionrun/replot_exp08_from_csv.py \\
        --run-dir output/exp08_sweep_theta_phi_l/<YYYYMMDD_HHMMSS>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import EXP08_BOUNDARY_L_VALUES
from experiments.exp08_sweep_theta_phi_l import render_exp08_figures

ALL_THETA_CSV = "delta_opt_map_all_theta.csv"
GLOBAL_OPT_CSV = "global_opt_vs_theta.csv"
INVERSION_BOUNDARY_CSV = "inversion_boundary.csv"


def load_exp08_csv_data(run_dir: Path) -> dict:
    """exp08 run ディレクトリから再プロット用データを読み込む。"""
    all_theta_path = run_dir / ALL_THETA_CSV
    if not all_theta_path.is_file():
        raise FileNotFoundError(f"Missing {ALL_THETA_CSV} in {run_dir}")

    data = np.genfromtxt(
        all_theta_path,
        delimiter=",",
        names=True,
        dtype=np.float64,
    )
    theta_deg_unique = np.unique(data["theta_deg"])
    phi_deg_unique = np.unique(data["phi_deg"])
    l_unique = np.unique(data["l"])

    theta_values = np.radians(theta_deg_unique)
    phi_values = np.radians(phi_deg_unique)
    l_values = l_unique

    n_theta = theta_deg_unique.size
    n_phi = phi_deg_unique.size
    n_l = l_values.size

    delta_opt_maps: list[np.ndarray] = []
    q_max_maps: list[np.ndarray] = []
    coordination_maps: list[np.ndarray] = []

    for theta_deg in theta_deg_unique:
        mask = data["theta_deg"] == theta_deg
        block = data[mask]
        delta_map = np.empty((n_phi, n_l), dtype=np.float64)
        q_map = np.empty((n_phi, n_l), dtype=np.float64)
        coord_map = np.empty((n_phi, n_l), dtype=np.float64)
        for row in block:
            i_phi = int(np.where(phi_deg_unique == row["phi_deg"])[0][0])
            i_l = int(np.where(l_values == row["l"])[0][0])
            delta_map[i_phi, i_l] = row["delta_opt_rad"]
            q_map[i_phi, i_l] = row["Q_max"]
            coord_map[i_phi, i_l] = row["coordination"]
        delta_opt_maps.append(delta_map)
        q_max_maps.append(q_map)
        coordination_maps.append(coord_map)

    global_opt_path = run_dir / GLOBAL_OPT_CSV
    if global_opt_path.is_file():
        global_data = np.genfromtxt(
            global_opt_path,
            delimiter=",",
            names=True,
            dtype=np.float64,
        )
        global_opt_rows = [
            (
                float(row["theta_deg"]),
                float(row["phi_star_deg"]),
                float(row["l_star"]),
                float(row["delta_star_deg"]),
                float(row["Q_max"]),
            )
            for row in global_data
        ]
    else:
        global_opt_rows = []
        for theta_deg, q_map in zip(theta_deg_unique, q_max_maps, strict=True):
            i_phi, i_l = np.unravel_index(int(np.argmax(q_map)), q_map.shape)
            global_opt_rows.append(
                (
                    float(theta_deg),
                    float(phi_deg_unique[i_phi]),
                    float(l_values[i_l]),
                    float(np.degrees(delta_opt_maps[int(np.where(theta_deg_unique == theta_deg)[0][0])][i_phi, i_l])),
                    float(q_map[i_phi, i_l]),
                )
            )

    boundary_path = run_dir / INVERSION_BOUNDARY_CSV
    if boundary_path.is_file():
        boundary_data = np.genfromtxt(
            boundary_path,
            delimiter=",",
            names=True,
            dtype=np.float64,
        )
        if boundary_data.ndim == 0:
            boundary_rows = [
                (
                    float(boundary_data["theta_deg"]),
                    float(boundary_data["l"]),
                    float(boundary_data["phi_crit_deg"]),
                )
            ]
        else:
            boundary_rows = [
                (float(row["theta_deg"]), float(row["l"]), float(row["phi_crit_deg"]))
                for row in boundary_data
            ]
    else:
        from core.utils import extract_inversion_boundary

        boundary_rows = []
        for theta_deg, coord_map in zip(theta_deg_unique, coordination_maps, strict=True):
            boundary_rows.extend(
                extract_inversion_boundary(
                    phi_values,
                    l_values,
                    coord_map,
                    theta_deg=float(theta_deg),
                )
            )

    return {
        "theta_values": theta_values,
        "phi_values": phi_values,
        "l_values": l_values,
        "delta_opt_maps": delta_opt_maps,
        "q_max_maps": q_max_maps,
        "coordination_maps": coordination_maps,
        "global_opt_rows": global_opt_rows,
        "boundary_rows": boundary_rows,
    }


def replot_exp08_from_csv(
    run_dir: Path,
    *,
    boundary_l_values: tuple[float, ...] = EXP08_BOUNDARY_L_VALUES,
) -> None:
    """CSV から exp08 の全図を再生成する。"""
    payload = load_exp08_csv_data(run_dir)
    render_exp08_figures(
        run_dir,
        theta_values=payload["theta_values"],
        phi_values=payload["phi_values"],
        l_values=payload["l_values"],
        delta_opt_maps=payload["delta_opt_maps"],
        q_max_maps=payload["q_max_maps"],
        coordination_maps=payload["coordination_maps"],
        global_opt_rows=payload["global_opt_rows"],
        boundary_rows=payload["boundary_rows"],
        boundary_l_values=boundary_l_values,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate exp08 figures from saved CSV files.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to exp08 run directory containing delta_opt_map_all_theta.csv.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logging.getLogger(__name__).info("Replotting exp08 figures from %s", run_dir)
    replot_exp08_from_csv(run_dir)
    logging.getLogger(__name__).info("Replot finished.")


if __name__ == "__main__":
    main()
