"""
第1段階実験: 単一スライダー（壁なし）/ Experiment 01: single slider, no wall.

【目的 / Purpose】
    1次元スライダーモデル（第1段階）を実行し、定常状態の s_1(t)、q_x(t) を得る。
    単一スライダーでは周期平均流量 Q が 0 に近いことを数値的に確認する（論文 4.3.6節）。

【構成 / Contents】
    1. パラメータ設定セクション
    2. run_experiment — 積分・流量・保存の一連処理
    3. main — ログ設定と実行入口

【実行方法 / How to run】
    cd cilia_simulation
    python experiments/exp01_single_slider.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

# プロジェクトルートを import パスに追加する / Allow imports from cilia_simulation root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.default_params import EXP01_DEFAULTS
from core.flow_rate import FlowCalculator
from core.hydrodynamics import Mobility
from core.slider import Slider, SliderParameters
from core.solver import SimulationResult, SolverConfig, TimeStepper
from core.utils import (
    PlotStyle,
    make_run_directory,
    plot_flow_rate,
    plot_trajectory,
    save_flow_rate_csv,
    save_parameters,
    save_summary,
)

# ==========================================
# 1. パラメータ設定セクション
# ==========================================

EXP_NAME = "exp01_single_slider"
OUTPUT_BASE = PROJECT_ROOT / "output"
# 物理パラメータ（実数は config/default_params.py）
SLIDER_PARAMETER_DICT: dict[str, float] = dict(EXP01_DEFAULTS)

# 時間積分
SOLVER_METHOD = "RK45"
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-10
N_PERIODS = 8
N_EVAL_PER_PERIOD = 200

# Q≈0 判定
Q_TOL_FACTOR = 1e-4

# プロット
PLOT_LAST_N_PERIODS = 2
PLOT_STYLE = PlotStyle()

# ==========================================
# 2. 実験本体
# ==========================================


def run_experiment() -> Path:
    """
    単一スライダー実験を実行し、output 以下に結果を保存する / Run exp01 pipeline.

    Returns
    -------
    Path
        実行ディレクトリ（output/<exp>/<timestamp>/）。
    """
    params = SliderParameters(**SLIDER_PARAMETER_DICT)
    slider = Slider(params)
    mobility = Mobility.from_slider_parameters(params)
    flow = FlowCalculator.from_slider_parameters(params)

    solver_config = SolverConfig(
        method=SOLVER_METHOD,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
        n_periods=N_PERIODS,
        n_eval_per_period=N_EVAL_PER_PERIOD,
    )
    stepper = TimeStepper(slider, mobility, solver_config)
    result = stepper.run()

    gamma_0 = mobility.gamma_0
    t_steady, q_x_steady, Q = flow.compute_from_simulation(slider, result)
    q_check = flow.check_Q_near_zero(Q, params.F_0, tol_factor=Q_TOL_FACTOR)

    amplitude_theory, phase_lag_theory = _theory_steady_response(
        params, gamma_0
    )
    amplitude_numeric = _peak_to_peak_amplitude(result.s_steady) / 2.0

    run_dir = make_run_directory(EXP_NAME, base=OUTPUT_BASE)
    _save_all_outputs(
        run_dir=run_dir,
        params=params,
        solver_config=solver_config,
        result=result,
        slider=slider,
        flow=flow,
        t_steady=t_steady,
        q_x_steady=q_x_steady,
        Q=Q,
        q_check=q_check,
        gamma_0=gamma_0,
        amplitude_theory=amplitude_theory,
        amplitude_numeric=amplitude_numeric,
        phase_lag_theory=phase_lag_theory,
    )

    return run_dir


def _theory_steady_response(
    params: SliderParameters,
    gamma_0: float,
) -> tuple[float, float]:
    """
    線形モデルの定常振幅と位相遅れを返す / Analytic steady amplitude and phase lag.

    s_1 ~ A cos(omega t - phi_lag),
    A = F_0 / sqrt(k^2 + (omega/gamma_0)^2),
    phi_lag = arctan(omega / (k * gamma_0)).
    """
    omega_over_gamma = params.omega / gamma_0
    amplitude = params.F_0 / math.sqrt(
        params.k**2 + omega_over_gamma**2
    )
    phase_lag = math.atan(params.omega / (params.k * gamma_0))
    return amplitude, phase_lag


def _peak_to_peak_amplitude(s: object) -> float:
    """ピークtoピーク幅を返す / Peak-to-peak span of a sequence."""
    import numpy as np

    s_arr = np.asarray(s, dtype=np.float64)
    return float(s_arr.max() - s_arr.min())


def _save_all_outputs(
    *,
    run_dir: Path,
    params: SliderParameters,
    solver_config: SolverConfig,
    result: SimulationResult,
    slider: Slider,
    flow: FlowCalculator,
    t_steady: object,
    q_x_steady: object,
    Q: float,
    q_check: object,
    gamma_0: float,
    amplitude_theory: float,
    amplitude_numeric: float,
    phase_lag_theory: float,
) -> None:
    """成果物を run_dir に保存する / Write JSON, CSV, plots, and summary."""
    import numpy as np

    save_parameters(
        run_dir / "parameters.json",
        {
            "experiment": EXP_NAME,
            "slider": params,
            "solver": solver_config,
            "plot_last_n_periods": PLOT_LAST_N_PERIODS,
            "q_tol_factor": Q_TOL_FACTOR,
        },
    )

    save_flow_rate_csv(
        run_dir / "flow_rate.csv",
        t_steady,
        q_x_steady,
        s=result.s_steady,
        f_total=result.f_total_steady,
    )

    f_total_all = np.asarray(result.f_total, dtype=np.float64)
    F_x_all = flow.F_x_from_f_total(f_total_all, slider.cos_phi)
    q_x_all = np.asarray(flow.instantaneous_q_x(F_x_all), dtype=np.float64)

    plot_trajectory(
        run_dir / "trajectory.png",
        result.t,
        result.s,
        result.period,
        n_periods=PLOT_LAST_N_PERIODS,
        style=PLOT_STYLE,
    )
    plot_flow_rate(
        run_dir / "flow_rate.png",
        result.t,
        q_x_all,
        result.period,
        n_periods=PLOT_LAST_N_PERIODS,
        style=PLOT_STYLE,
    )

    summary_lines = [
        f"experiment: {EXP_NAME}",
        f"integration_method: {solver_config.method}",
        f"n_periods: {solver_config.n_periods}",
        f"gamma_0: {gamma_0:.8e}",
        f"period_T: {result.period:.8e}",
        "",
        "--- Net flow (last period) ---",
        f"Q: {Q:.8e}",
        f"reference_scale F_0*h/(pi*mu): {q_check.reference_scale:.8e}",
        f"tolerance ({Q_TOL_FACTOR} * scale): {q_check.tolerance:.8e}",
        f"|Q|/scale: {abs(Q) / q_check.reference_scale:.8e}",
        f"Q_near_zero_passed: {q_check.passed}",
        "",
        "--- Steady oscillation check ---",
        f"amplitude_theory: {amplitude_theory:.8e}",
        f"amplitude_numeric (half peak-to-peak): {amplitude_numeric:.8e}",
        f"phase_lag_theory [rad]: {phase_lag_theory:.8e}",
        f"phase_lag_theory [deg]: {math.degrees(phase_lag_theory):.4f}",
        "",
        "Note: q_x uses Eq. (2.61) as a metric (no wall in exp01).",
    ]
    save_summary(run_dir / "summary.txt", summary_lines)


# ==========================================
# 3. エントリポイント
# ==========================================


def main() -> None:
    """ログを設定し実験を1回実行する / Configure logging and run once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # 実験を実行し、出力先パスをログに残す / Run pipeline and log output path.
    run_dir = run_experiment()
    logging.getLogger(__name__).info("Finished. Results: %s", run_dir)


if __name__ == "__main__":
    main()
