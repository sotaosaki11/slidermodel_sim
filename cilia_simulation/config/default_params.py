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
    - 第6段階 exp06: 実験スクリプトのみプレースホルダ（exp05 後に段階実装）

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
# 第6段階 exp06: Delta-theta 掃引（l 固定、x-y 配置角）— プレースホルダ
# exp05 完了後に段階実装（core 幾何・config・掃引スクリプトは未追加）
# ==========================================

# EXP06_DEFAULTS / EXP06_SWEEP_* / resolve_exp06_config: 後日追加