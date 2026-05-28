"""
デフォルト実数パラメータの一括管理 / Central default numeric parameters.

【目的 / Purpose】
    全実験段階（exp01〜exp03）で使う物理・数値パラメータの実数値を
    1か所に集約し、マジックナンバーの散在を防ぐ。

【構成 / Contents】
    - EXP01_DEFAULTS: 第1段階（単一スライダー、壁なし）
    - EXP02_DEFAULTS: 第2段階（2スライダー、Stokeslet）
    - EXP03_DEFAULTS: 第3段階（2スライダー、Blakelet）— 未使用・後日追加

【使い方 / Usage】
    experiments/exp01_single_slider.py などから import し、
    core.slider.SliderParameters に渡す。

    from config.default_params import EXP01_DEFAULTS
    from core.slider import SliderParameters
    params = SliderParameters(**EXP01_DEFAULTS)
"""

from __future__ import annotations

import math

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
# 第3段階 exp03: 2スライダー（壁あり）— プレースホルダ
# ==========================================

# EXP03_DEFAULTS: dict[str, float] = { ... }
