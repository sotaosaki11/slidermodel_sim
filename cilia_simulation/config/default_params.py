"""
デフォルト実数パラメータの一括管理 / Central default numeric parameters.

【目的 / Purpose】
    全実験段階（exp01〜exp06）で使う物理・数値パラメータの実数値を
    1か所に集約し、マジックナンバーの散在を防ぐ。

【構成 / Contents】
    - EXP01_DEFAULTS: 第1段階（単一スライダー、壁なし）
    - EXP02_DEFAULTS: 第2段階（2スライダー、Stokeslet）
    - EXP03_DEFAULT_MODE: exp03 の IDE ▷ 実行時デフォルト（fast / fine）
    - EXP03_SOLVER_PRESETS: exp03 用の積分設定（fast / fine）
    - resolve_exp03_config: mode から掃引辞書と SolverConfig を返す
    - EXP04_DEFAULTS / EXP04_SOLVER_PRESET: 第4段階単点（Blakelet）
    - EXP05_DEFAULT_MODE / resolve_exp05_config: 第5段階 Δ×l 掃引（fast / fine）
    - EXP06_LAYOUT_THETA / resolve_exp06_config: 第6段階 Δ×l 掃引（配置角 θ 固定、fast / fine）

【使い方 / Usage】
    experiments/exp01_single_slider.py などから import し、
    core.slider.SliderParameters に渡す。

    from config.default_params import EXP01_DEFAULTS
    from core.slider import SliderParameters
    params = SliderParameters(**EXP01_DEFAULTS)

    exp03 では resolve_exp03_config("fast") などを使う。
    IDE ▷ 実行時は EXP03_DEFAULT_MODE を変更する（--mode で CLI 上書き可）。
"""

from __future__ import annotations

import math
from typing import Literal

from core.solver import SolverConfig

Exp03Mode = Literal["fast", "fine"]
Exp05Mode = Literal["fast", "fine"]
Exp06Mode = Literal["fast", "fine"]
Exp07Mode = Literal["fast", "fine"]
Exp08Mode = Literal["fast", "fine"]
Exp09Mode = Literal["fast", "fine"]

# ==========================================
# 第1段階 exp01: 単一スライダー（壁なし）
# 論文・指示書の初期値（無次元）
# ==========================================

EXP01_DEFAULTS: dict[str, float] = {
  "a": 0.05,       # ビーズ半径 / bead radius
  "mu": 1.0,       # 粘性係数 / dynamic viscosity
  "k": 1.0,        # ばね定数 / spring constant
  "F_0": 1.0,      # 駆動力振幅 / active force amplitude（論文 式(4.3)）
  "omega": 2.0 * math.pi,  # 角振動数 / angular frequency（周期 T=1.0）
  "phi": math.pi / 4.0,    # 傾き角 [rad] / tilt angle（45°）
  "h": 1.0,        # 流量式の基準高さ / reference height for flow formula
  "s0": 0.0,       # 初期位置 s_1(0) / initial slider position
}

# ==========================================
# 第2段階 exp02: 2スライダー（壁なし）
# ==========================================

EXP02_DEFAULTS: dict[str, float | int] = {
  # 単体パラメータ（exp01 と共通）
  "a": 0.05,               # ビーズ半径 / bead radius
  "mu": 1.0,               # 粘性係数 / dynamic viscosity
  "k": 1.0,                # ばね定数 / spring constant
  "F_0": 1.0,              # 駆動力振幅 / active force amplitude
  "omega": 2.0 * math.pi,  # 角振動数 / angular frequency
  "phi": math.pi / 4.0,    # 傾き角 [rad] / tilt angle
  "h": 1.0,                # 流量式の基準高さ / reference height
  "s1_0": 0.0,             # スライダー1 初期位置 / initial position of slider 1
  "s2_0": 0.0,             # スライダー2 初期位置 / initial position of slider 2

  # 2スライダー固有パラメータ
  "l": 2.0,                # スライダー間距離 / center-to-center distance
  "delta": math.pi / 2.0,  # 位相差 Delta [rad]（まず単一点）

  # 将来の掃引用（Step 1 では未使用）
  "delta_min": -math.pi,   # Delta 掃引の最小値
  "delta_max": math.pi,    # Delta 掃引の最大値
  "delta_points": 13,      # Delta 掃引点数（例: 粗い掃引）
}

# ==========================================
# 第3段階 exp03: Delta-l 掃引（壁なし）
# ==========================================

# IDE ▷ 実行（CLI で --mode 未指定）時に使う mode。
# "fast" = Delta 360 点 + Euler / "fine" = Delta 8000 点 + RK45
EXP03_DEFAULT_MODE: Exp03Mode = "fast"

# 物理量と l 掃引範囲は fast / fine で共通。Delta 解像度は mode で変える。
_EXP03_SWEEP_PHYSICAL: dict[str, float] = {
  "a": 0.05,
  "mu": 1.0,
  "k": 1.0,
  "F_0": 1.0,
  "omega": 2.0 * math.pi,
  "phi": math.pi / 4.0,
  "h": 1.0,
  "s1_0": 0.0,
  "s2_0": 0.0,
  "delta_min": -math.pi,
  "delta_max": math.pi,
  "l_min": 1.5,
  "l_max": 6.0,
}

_EXP03_SWEEP_GRID: dict[str, dict[str, int]] = {
  "fast": {
    "delta_points": 360,   # 1 deg 刻み（-180 <= Delta < 180）
    "l_points": 19,
  },
  "fine": {
    "delta_points": 8000,
    "l_points": 19,
  },
}

EXP03_SWEEP_FAST_DEFAULTS: dict[str, float | int] = {
  **_EXP03_SWEEP_PHYSICAL,
  **_EXP03_SWEEP_GRID["fast"],
}

EXP03_SWEEP_FINE_DEFAULTS: dict[str, float | int] = {
  **_EXP03_SWEEP_PHYSICAL,
  **_EXP03_SWEEP_GRID["fine"],
}

# 既存 import との互換（fine = 従来の 8000 点掃引）
EXP03_SWEEP_DEFAULTS: dict[str, float | int] = EXP03_SWEEP_FINE_DEFAULTS

# exp03 専用: 掃引用積分設定（論文 4.2 は Euler、最終比較は RK45）
EXP03_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = {
  "fast": {
    "method": "EULER",
    "n_periods": 10,
    "n_eval_per_period": 40000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
  "fine": {
    "method": "RK45",
    "n_periods": 10,
    "n_eval_per_period": 10000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
}


def resolve_exp03_config(
    mode: Exp03Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp03 の掃引パラメータ辞書と SolverConfig を mode から返す。

    Parameters
    ----------
    mode : {"fast", "fine"}
        fast: 粗い Delta グリッド + 前進 Euler（探索用）。
        fine: 密な Delta グリッド + RK45（論文比較・高精度用）。

    Returns
    -------
    sweep_defaults : dict
        EXP03_SWEEP_FAST_DEFAULTS または EXP03_SWEEP_FINE_DEFAULTS。
    solver_config : SolverConfig
        EXP03_SOLVER_PRESETS に対応する積分設定。
    """
    if mode == "fast":
        sweep_defaults = EXP03_SWEEP_FAST_DEFAULTS
    elif mode == "fine":
        sweep_defaults = EXP03_SWEEP_FINE_DEFAULTS
    else:
        raise ValueError(f"Unknown exp03 mode: {mode!r}. Use 'fast' or 'fine'.")

    preset = EXP03_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config



# ==========================================
# 第4段階 exp04: 2スライダー（壁 z=0 あり, Blakelet）— 単点検証
# ==========================================

# 物理パラメータは exp02 単点と同一（Oseen vs Blakelet の比較用）。
# 移動度のみ BlakeletTwoSliderMobility（論文 式(4.4)）に差し替える。
EXP04_DEFAULTS: dict[str, float | int] = {
  "a": 0.05,
  "mu": 1.0,
  "k": 1.0,
  "F_0": 1.0,
  "omega": 2.0 * math.pi,
  "phi": math.pi / 4.0,
  "h": 1.0,
  "s1_0": 0.0,
  "s2_0": 0.0,
  "l": 2.0,
  "delta": math.pi / 2.0,
}

# exp04 単点: exp02 と同型（RK45, 10 周期, 300 点/周期）
EXP04_SOLVER_PRESET: dict[str, float | int | str] = {
  "method": "RK45",
  "n_periods": 10,
  "n_eval_per_period": 300,
  "rtol": 1e-8,
  "atol": 1e-10,
}


def resolve_exp04_solver_config() -> SolverConfig:
    """
    exp04 単点実験用の SolverConfig を返す。

    Returns
    -------
    SolverConfig
        EXP04_SOLVER_PRESET に対応する積分設定。
    """
    preset = EXP04_SOLVER_PRESET
    return SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )


# exp05 Δ×l 掃引（Blakelet）。グリッド範囲は exp03 と共通、積分は Blakelet 上で実行。
EXP05_DEFAULT_MODE: Exp05Mode = "fine"

_EXP05_SWEEP_PHYSICAL: dict[str, float] = {
  "a": 0.05,
  "mu": 1.0,
  "k": 2.0,
  "F_0": 1.0,
  "omega": math.pi,
  "phi": 0,
  "h": 1.0,
  "s1_0": 0.0,
  "s2_0": 0.0,
  "delta_min": -math.pi,
  "delta_max": math.pi,
  "l_min": 1.5,
  "l_max": 6.0,
}

_EXP05_SWEEP_GRID: dict[str, dict[str, int]] = {
  "fast": {
    "delta_points": 360,
    "l_points": 19,
  },
  "fine": {
    "delta_points": 8000,
    "l_points": 19,
  },
}

EXP05_SWEEP_FAST_DEFAULTS: dict[str, float | int] = {
  **_EXP05_SWEEP_PHYSICAL,
  **_EXP05_SWEEP_GRID["fast"],
}

EXP05_SWEEP_FINE_DEFAULTS: dict[str, float | int] = {
  **_EXP05_SWEEP_PHYSICAL,
  **_EXP05_SWEEP_GRID["fine"],
}

EXP05_SWEEP_DEFAULTS: dict[str, float | int] = EXP05_SWEEP_FINE_DEFAULTS

EXP05_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = {
  "fast": {
    "method": "EULER",
    "n_periods": 10,
    "n_eval_per_period": 40000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
  "fine": {
    "method": "RK45",
    "n_periods": 10,
    "n_eval_per_period": 10000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
}


def resolve_exp05_config(
    mode: Exp05Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp05 Δ×l 掃引のパラメータ辞書と SolverConfig を mode から返す。

    Parameters
    ----------
    mode : {"fast", "fine"}
        fast: 粗い Delta グリッド + 前進 Euler（探索用）。
        fine: 密な Delta グリッド + RK45（論文比較・高精度用）。

    Returns
    -------
    sweep_defaults : dict
        EXP05_SWEEP_FAST_DEFAULTS または EXP05_SWEEP_FINE_DEFAULTS。
    solver_config : SolverConfig
        EXP05_SOLVER_PRESETS に対応する積分設定。
    """
    if mode == "fast":
        sweep_defaults = EXP05_SWEEP_FAST_DEFAULTS
    elif mode == "fine":
        sweep_defaults = EXP05_SWEEP_FINE_DEFAULTS
    else:
        raise ValueError(f"Unknown exp05 mode: {mode!r}. Use 'fast' or 'fine'.")

    preset = EXP05_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config


# ==========================================
# 第6段階 exp06: Delta-l 掃引（exp05 同型、配置角 theta 固定）
# ==========================================

# x-y 平面内の相対配置角 [rad]。掃引軸は exp05 と同じ Delta × l。
EXP06_LAYOUT_THETA: float = math.pi / 4.0  # 45°

_EXP06_SWEEP_PHYSICAL: dict[str, float] = {
  "a": 0.05,
  "mu": 1.0,
  "k": 2.0,
  "F_0": 1.0,
  "omega": math.pi,
  "phi": math.pi / 4.0,
  "h": 1.0,
  "s1_0": 0.0,
  "s2_0": 0.0,
  "layout_theta": EXP06_LAYOUT_THETA,
  "delta_min": -math.pi,
  "delta_max": math.pi,
  "l_min": 1.5,
  "l_max": 6.0,
}

_EXP06_SWEEP_GRID: dict[str, dict[str, int]] = {
  "fast": {
    "delta_points": 360,
    "l_points": 19,
  },
  "fine": {
    "delta_points": 8000,
    "l_points": 19,
  },
}

EXP06_SWEEP_FAST_DEFAULTS: dict[str, float | int] = {
  **_EXP06_SWEEP_PHYSICAL,
  **_EXP06_SWEEP_GRID["fast"],
}

EXP06_SWEEP_FINE_DEFAULTS: dict[str, float | int] = {
  **_EXP06_SWEEP_PHYSICAL,
  **_EXP06_SWEEP_GRID["fine"],
}

EXP06_SWEEP_DEFAULTS: dict[str, float | int] = EXP06_SWEEP_FINE_DEFAULTS

EXP06_DEFAULT_MODE: Exp06Mode = "fast"

EXP06_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = {
  "fast": {
    "method": "EULER",
    "n_periods": 10,
    "n_eval_per_period": 40000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
  "fine": {
    "method": "RK45",
    "n_periods": 10,
    "n_eval_per_period": 10000,
    "rtol": 1e-8,
    "atol": 1e-10,
  },
}


def resolve_exp06_config(
    mode: Exp06Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp06 Δ×l 掃引（layout_theta 固定）のパラメータ辞書と SolverConfig を mode から返す。

    Parameters
    ----------
    mode : {"fast", "fine"}
        fast: 粗い Delta グリッド + 前進 Euler（探索用）。
        fine: 密な Delta グリッド + RK45（高精度用）。

    Returns
    -------
    sweep_defaults : dict
        EXP06_SWEEP_FAST_DEFAULTS または EXP06_SWEEP_FINE_DEFAULTS。
    solver_config : SolverConfig
        EXP06_SOLVER_PRESETS に対応する積分設定。
    """
    if mode == "fast":
        sweep_defaults = EXP06_SWEEP_FAST_DEFAULTS
    elif mode == "fine":
        sweep_defaults = EXP06_SWEEP_FINE_DEFAULTS
    else:
        raise ValueError(f"Unknown exp06 mode: {mode!r}. Use 'fast' or 'fine'.")

    preset = EXP06_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config


# ==========================================
# 第7段階 exp07: phi-l 掃引（配置角 theta 固定、各点で Delta 最適化）
# ==========================================

# x-y 平面内の相対配置角 [rad]。掃引軸は phi × l。theta=0 で論文 Fig.6 配置。
EXP07_LAYOUT_THETA: float = math.pi / 2.0

# phi 掃引: [phi_min_deg, phi_max_deg) を phi_step_deg 刻み（90° は含めない）。
EXP07_PHI_MIN_DEG: float = 0.0
EXP07_PHI_STEP_DEG: float = 5.0
EXP07_PHI_MAX_DEG: float = 90.0

# l* 掃引: 値を直接列挙（旧 linspace(1.5, 6.0, 19) と同等）。
EXP07_L_VALUES: tuple[float, ...] = (
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    8.0,
)

# 現行既定: k=1, omega=2π。論文 Fig.6 は k*=2, omega*=π。
# Fig.6 と直接比較する場合は k=2.0, omega=math.pi に変更。
_EXP07_SWEEP_PHYSICAL: dict[str, float | tuple[float, ...]] = {
    "a": 0.05,
    "mu": 1.0,
    "k": 2.0,
    "F_0": 1.0,
    "omega": math.pi,
    "h": 1.0,
    "s1_0": 0.0,
    "s2_0": 0.0,
    "layout_theta": EXP07_LAYOUT_THETA,
    "phi_min_deg": EXP07_PHI_MIN_DEG,
    "phi_step_deg": EXP07_PHI_STEP_DEG,
    "phi_max_deg": EXP07_PHI_MAX_DEG,
    "l_values": EXP07_L_VALUES,
    "delta_min": -math.pi,
    "delta_max": math.pi,
}

_EXP07_SWEEP_GRID: dict[str, dict[str, int]] = {
    "fast": {
        "delta_points": 360,
    },
    "fine": {
        "delta_points": 10000,
    },
}

EXP07_SWEEP_FAST_DEFAULTS: dict[str, float | int | tuple[float, ...]] = {
    **_EXP07_SWEEP_PHYSICAL,
    **_EXP07_SWEEP_GRID["fast"],
}

EXP07_SWEEP_FINE_DEFAULTS: dict[str, float | int | tuple[float, ...]] = {
    **_EXP07_SWEEP_PHYSICAL,
    **_EXP07_SWEEP_GRID["fine"],
}

EXP07_SWEEP_DEFAULTS: dict[str, float | int | tuple[float, ...]] = (
    EXP07_SWEEP_FINE_DEFAULTS
)


def build_exp07_phi_values(
    phi_min_deg: float,
    phi_step_deg: float,
    phi_max_deg: float = EXP07_PHI_MAX_DEG,
) -> tuple[float, ...]:
    """
    exp07 用 phi 掃引値 [rad] を生成する。

    [phi_min_deg, phi_max_deg) を phi_step_deg 刻みで列挙し、
    phi_max_deg（既定 90°）ちょうどは含めない。
    """
    if phi_step_deg <= 0.0:
        raise ValueError("phi_step_deg must be positive.")
    values: list[float] = []
    phi_deg = phi_min_deg
    while phi_deg < phi_max_deg - 1e-12:
        values.append(math.radians(phi_deg))
        phi_deg += phi_step_deg
    if not values:
        raise ValueError(
            "phi sweep produced no points; check phi_min_deg, phi_step_deg, phi_max_deg."
        )
    return tuple(values)

EXP07_DEFAULT_MODE: Exp07Mode = "fast"

EXP07_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = {
    "fast": {
        "method": "EULER",
        "n_periods": 10,
        "n_eval_per_period": 40000,
        "rtol": 1e-8,
        "atol": 1e-10,
    },
    "fine": {
        "method": "RK45",
        "n_periods": 20,
        "n_eval_per_period": 10000,
        "rtol": 1e-8,
        "atol": 1e-10,
    },
}


def resolve_exp07_config(
    mode: Exp07Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp07 phi×l 掃引（layout_theta 固定、各点で Delta 最適化）の設定を mode から返す。

    Parameters
    ----------
    mode : {"fast", "fine"}
        fast: 粗い Delta グリッド + 前進 Euler（探索用）。
        fine: 密な Delta グリッド + RK45（高精度用）。

    Returns
    -------
    sweep_defaults : dict
        EXP07_SWEEP_FAST_DEFAULTS または EXP07_SWEEP_FINE_DEFAULTS。
    solver_config : SolverConfig
        EXP07_SOLVER_PRESETS に対応する積分設定。
    """
    if mode == "fast":
        sweep_defaults = EXP07_SWEEP_FAST_DEFAULTS
    elif mode == "fine":
        sweep_defaults = EXP07_SWEEP_FINE_DEFAULTS
    else:
        raise ValueError(f"Unknown exp07 mode: {mode!r}. Use 'fast' or 'fine'.")

    preset = EXP07_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config


# ==========================================
# 第8段階 exp08: theta 掃引（phi×l 掃引を各 theta で実行、各点で Delta 最適化）
# ==========================================

# layout_theta 掃引: [theta_min_deg, theta_max_deg] を theta_step_deg 刻み（90° を含む）。
EXP08_THETA_MIN_DEG: float = 0.0
EXP08_THETA_STEP_DEG: float = 5.0
EXP08_THETA_MAX_DEG: float = 90.0

# phi 掃引: [phi_min_deg, phi_max_deg) を phi_step_deg 刻み（90° は含めない）。
EXP08_PHI_MIN_DEG: float = 0.0
EXP08_PHI_STEP_DEG: float = 5.0
EXP08_PHI_MAX_DEG: float = 90.0

# l* 掃引: 値を直接列挙（旧 linspace(1.5, 6.0, 19) と同等）。
EXP08_L_VALUES: tuple[float, ...] = (
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    8.0,
)

# phi / l / Delta グリッドは exp07 と同一。
_EXP08_SWEEP_PHYSICAL: dict[str, float | int | tuple[float, ...]] = {
    "a": 0.05,
    "mu": 1.0,
    "k": 2.0,
    "F_0": 1.0,
    "omega": math.pi,
    "h": 1.0,
    "s1_0": 0.0,
    "s2_0": 0.0,
    "theta_min_deg": EXP08_THETA_MIN_DEG,
    "theta_step_deg": EXP08_THETA_STEP_DEG,
    "theta_max_deg": EXP08_THETA_MAX_DEG,
    "phi_min_deg": EXP08_PHI_MIN_DEG,
    "phi_step_deg": EXP08_PHI_STEP_DEG,
    "phi_max_deg": EXP08_PHI_MAX_DEG,
    "l_values": EXP08_L_VALUES,
    "delta_min": -math.pi,
    "delta_max": math.pi,
}

_EXP08_SWEEP_GRID: dict[str, dict[str, int]] = {
    "fast": {
        "delta_points": 360,
    },
    "fine": {
        "delta_points": 3600,
    },
}

EXP08_SWEEP_FAST_DEFAULTS: dict[str, float | int | tuple[float, ...]] = {
    **_EXP08_SWEEP_PHYSICAL,
    **_EXP08_SWEEP_GRID["fast"],
}

EXP08_SWEEP_FINE_DEFAULTS: dict[str, float | int | tuple[float, ...]] = {
    **_EXP08_SWEEP_PHYSICAL,
    **_EXP08_SWEEP_GRID["fine"],
}

EXP08_SWEEP_DEFAULTS: dict[str, float | int | tuple[float, ...]] = (
    EXP08_SWEEP_FINE_DEFAULTS
)

# 境界重ね描き・協調マップの既定 l*。
EXP08_BOUNDARY_L_VALUES: tuple[float, ...] = (0.8, 1.0, 1.5, 2.0)

EXP08_DEFAULT_MODE: Exp08Mode = "fast"

EXP08_STEADY_N_PERIODS: int = 5
EXP08_INITIAL_CONDITION_METHOD: str = "decoupled_blakelet"

EXP08_IC_SOLVER_PRESET: dict[str, float | int | str] = {
    "method": "RK45",
    "n_periods": 20,
    "n_eval_per_period": 1000,
    "rtol": 1e-8,
    "atol": 1e-10,
}

EXP08_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = {
    "fast": {
        "method": "EULER",
        "n_periods": 10,
        "n_eval_per_period": 40000,
        "rtol": 1e-8,
        "atol": 1e-10,
    },
    "fine": {
        "method": "RK45",
        "n_periods": 10,
        "n_eval_per_period": 10000,
        "rtol": 1e-8,
        "atol": 1e-10,
    },
}


def build_exp08_theta_values(
    theta_min_deg: float,
    theta_step_deg: float,
    theta_max_deg: float = EXP08_THETA_MAX_DEG,
) -> tuple[float, ...]:
    """
    exp08 用 layout_theta 掃引値 [rad] を生成する。

    [theta_min_deg, theta_max_deg] を theta_step_deg 刻みで列挙し、
    theta_max_deg（既定 90°）を含める。
    """
    if theta_step_deg <= 0.0:
        raise ValueError("theta_step_deg must be positive.")
    if theta_max_deg < theta_min_deg - 1e-12:
        raise ValueError("theta_max_deg must be >= theta_min_deg.")
    values: list[float] = []
    theta_deg = theta_min_deg
    while theta_deg <= theta_max_deg + 1e-12:
        values.append(math.radians(theta_deg))
        theta_deg += theta_step_deg
    if not values:
        raise ValueError(
            "theta sweep produced no points; check theta_min_deg, "
            "theta_step_deg, theta_max_deg."
        )
    return tuple(values)


def resolve_exp08_config(
    mode: Exp08Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp08 theta×phi×l 掃引（各 (theta, phi, l) で Delta 最適化）の設定を mode から返す。

    Parameters
    ----------
    mode : {"fast", "fine"}
        fast: 粗い Delta グリッド + 前進 Euler（探索用）。
        fine: 密な Delta グリッド + RK45（高精度用）。

    Returns
    -------
    sweep_defaults : dict
        EXP08_SWEEP_FAST_DEFAULTS または EXP08_SWEEP_FINE_DEFAULTS。
    solver_config : SolverConfig
        EXP08_SOLVER_PRESETS に対応する積分設定。
    """
    if mode == "fast":
        sweep_defaults = dict(EXP08_SWEEP_FAST_DEFAULTS)
    elif mode == "fine":
        sweep_defaults = dict(EXP08_SWEEP_FINE_DEFAULTS)
    else:
        raise ValueError(f"Unknown exp08 mode: {mode!r}. Use 'fast' or 'fine'.")

    sweep_defaults["steady_n_periods"] = EXP08_STEADY_N_PERIODS
    sweep_defaults["initial_condition_method"] = EXP08_INITIAL_CONDITION_METHOD
    sweep_defaults["include_constraint_force_in_Q"] = False

    preset = EXP08_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config


# ==========================================
# 第9段階 exp09: exp08 と同型スイープ（流量に拘束力を含める）
# ==========================================

# 流量定義スイッチ（コア経路 include_constraint_force_in_Q に対応）。
# exp08 = False（直線方向力のみ）、exp09 = True（拘束力込みの F_x）。
EXP08_INCLUDE_CONSTRAINT_FORCE_IN_Q: bool = False
EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q: bool = True

# θ / φ / l / Δ グリッドは exp08 と同一。
EXP09_THETA_MIN_DEG: float = EXP08_THETA_MIN_DEG
EXP09_THETA_STEP_DEG: float = EXP08_THETA_STEP_DEG
EXP09_THETA_MAX_DEG: float = EXP08_THETA_MAX_DEG
EXP09_BOUNDARY_L_VALUES: tuple[float, ...] = EXP08_BOUNDARY_L_VALUES

EXP09_SWEEP_FAST_DEFAULTS: dict[str, float | int | tuple[float, ...]] = dict(
    EXP08_SWEEP_FAST_DEFAULTS
)
EXP09_SWEEP_FINE_DEFAULTS: dict[str, float | int | tuple[float, ...]] = dict(
    EXP08_SWEEP_FINE_DEFAULTS
)
EXP09_SWEEP_DEFAULTS: dict[str, float | int | tuple[float, ...]] = (
    EXP09_SWEEP_FINE_DEFAULTS
)

EXP09_DEFAULT_MODE: Exp09Mode = "fast"
EXP09_STEADY_N_PERIODS: int = EXP08_STEADY_N_PERIODS
EXP09_INITIAL_CONDITION_METHOD: str = EXP08_INITIAL_CONDITION_METHOD
EXP09_IC_SOLVER_PRESET: dict[str, float | int | str] = dict(EXP08_IC_SOLVER_PRESET)
EXP09_SOLVER_PRESETS: dict[str, dict[str, float | int | str]] = EXP08_SOLVER_PRESETS


def build_exp09_theta_values(
    theta_min_deg: float,
    theta_step_deg: float,
    theta_max_deg: float = EXP09_THETA_MAX_DEG,
) -> tuple[float, ...]:
    """exp09 用 layout_theta 掃引値 [rad]（exp08 と同一生成規則）。"""
    return build_exp08_theta_values(theta_min_deg, theta_step_deg, theta_max_deg)


def resolve_exp09_config(
    mode: Exp09Mode = "fast",
) -> tuple[dict[str, float | int], SolverConfig]:
    """
    exp09 theta×phi×l 掃引の設定を mode から返す。

    exp08 と同一グリッド・積分設定。流量のみ拘束力込み（
    include_constraint_force_in_Q=True）。
    """
    if mode == "fast":
        sweep_defaults = dict(EXP09_SWEEP_FAST_DEFAULTS)
    elif mode == "fine":
        sweep_defaults = dict(EXP09_SWEEP_FINE_DEFAULTS)
    else:
        raise ValueError(f"Unknown exp09 mode: {mode!r}. Use 'fast' or 'fine'.")

    sweep_defaults["steady_n_periods"] = EXP09_STEADY_N_PERIODS
    sweep_defaults["initial_condition_method"] = EXP09_INITIAL_CONDITION_METHOD
    sweep_defaults["include_constraint_force_in_Q"] = (
        EXP09_INCLUDE_CONSTRAINT_FORCE_IN_Q
    )

    preset = EXP09_SOLVER_PRESETS[mode]
    solver_config = SolverConfig(
        method=str(preset["method"]),
        rtol=float(preset["rtol"]),
        atol=float(preset["atol"]),
        n_periods=int(preset["n_periods"]),
        n_eval_per_period=int(preset["n_eval_per_period"]),
    )
    return sweep_defaults, solver_config
