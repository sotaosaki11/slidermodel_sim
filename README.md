# slidermodel_sim

修士論文第4章の **1次元スライダーモデル**（繊毛簡略化モデル）を Python で段階的に再実装する。
論文再現の目標は **正味流量 Q** および **最適位相差 Δ_opt** の再現（exp05 まで）。
その先の研究目標は、**l\* 固定**のもと x–y 平面内で相対配置角 θ を変化させ、θ に応じた流量最大化戦略（Δ_opt(θ)）の相違を検証すること（exp06、exp05 完了後）。

ソースコードは [`cilia_simulation/`](cilia_simulation/) 配下に配置。

## 現段階の進捗

| 段階 | スクリプト | 状態 | 内容 |
|------|-----------|------|------|
| exp01 | `experiments/exp01_single_slider.py` | 完了 | 単一スライダー、Q≈0 検証 |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | 完了 | 2本スライダー、Oseen 相互作用 |
| exp03 | `experiments/exp03_sweep_delta_l.py` | 完了 | Δ×l 掃引、Q(Δ,l)、Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | 完了 | 壁あり Blakelet（単点検証） |
| exp05 | `experiments/exp05_sweep_delta_l_wall.py` | 完了 | 壁あり Blakelet Δ×l 掃引 |
| exp06 | `experiments/exp06_sweep_delta_theta.py` | 未実装 | Δ×θ 掃引（l 固定）— **exp05 後に実装** |

### 現モデルの限界

| 段階 | 移動度（力→速度） | 流量 $Q$ |
|------|-------------------|----------|
| exp01〜03 | Oseen（壁なし） | 式(2.61)（壁 z=0 遠方近似） |
| **exp04** | **Blakelet（壁 z=0）** | 式(2.61)(4.65)（移動度と整合） |

論文 Fig との **定量比較** は exp05 の **fine 掃引**（`--mode fine`）で評価する。

物理・数学の詳細は [`cilia_simulation/PHYSICS_REPORT.md`](cilia_simulation/PHYSICS_REPORT.md) を参照。

## セットアップ

```bash
cd cilia_simulation
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## 実行

いずれも `cilia_simulation/` 内で実行する。

### exp01 — 単一スライダー

```bash
python experiments/exp01_single_slider.py
```

**出力先**: `output/exp01_single_slider/<YYYYMMDD_HHMMSS>/`

### exp02 — 2本スライダー（壁なし）

```bash
python experiments/exp02_two_sliders_nowall.py
```

**出力先**: `output/exp02_two_sliders_nowall/<YYYYMMDD_HHMMSS>/`
**出力物**: 時系列・相図・Q を含む summary / PNG

### exp03 — Δ×l 掃引

```bash
# config/default_params.py の EXP03_DEFAULT_MODE が CLI 未指定時の既定
# （現在: "fine"）。探索用は --mode fast を明示する
python experiments/exp03_sweep_delta_l.py --mode fast

# 論文比較用（Delta 8000 点 + RK45）
python experiments/exp03_sweep_delta_l.py --mode fine --workers 4

# デバッグ（直列）
python experiments/exp03_sweep_delta_l.py --workers 1
```

IDE ▷ 実行時は `config/default_params.py` の `EXP03_DEFAULT_MODE` を `"fast"` または `"fine"` に設定する。

| `--mode` | Delta 点数 | 積分 | 用途 |
|----------|------------|------|------|
| `fast` | 360 | Euler | 日常の探索・プロット |
| `fine` | 8000 | RK45 | 論文 Fig との定量比較 |

**出力先**: `output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS>/`
**出力物**: CSV、ヒートマップ、Δ_opt 曲線

### exp04 — 2本スライダー（壁あり Blakelet、単点）

```bash
python experiments/exp04_two_sliders_wall.py
```

パラメータ: `config/default_params.py` の `EXP04_DEFAULTS`（exp02 単点と同一）。積分: `EXP04_SOLVER_PRESET`（RK45, 10 周期）。

**出力先**: `output/exp04_two_sliders_wall/<YYYYMMDD_HHMMSS>/`

**出力物**: `parameters.json`, `summary.txt`, 時系列・力・相図の PNG。`summary.txt` に `Q_blakelet` と Oseen 参考値 `Q_oseen_reference` を記録。

### exp05 — Δ×l 掃引（壁あり Blakelet）

```bash
python experiments/exp05_sweep_delta_l_wall.py --mode fast
python experiments/exp05_sweep_delta_l_wall.py --mode fine --workers 4
```

IDE ▷ 実行時は `config/default_params.py` の `EXP05_DEFAULT_MODE` を変更する。

**出力先**: `output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>/`

## テスト

```bash
cd cilia_simulation
python -m unittest discover tests -v
```

## ディレクトリ構成

```
slidermodel_sim/
├── README.md                 # 本ファイル（GitHub トップ用）
└── cilia_simulation/
    ├── README.md             # cilia_simulation 内作業向け（本ファイルと同期）
    ├── PHYSICS_REPORT.md     # 物理・数学解説（exp01〜exp05）
    ├── config/               # デフォルトパラメータ
    ├── core/                 # 共通ライブラリ（力・移動度・積分・流量・掃引）
    ├── experiments/          # exp01〜exp06（exp05 まで完了、exp06 はプレースホルダ）
    ├── tests/
    ├── optionrun/            # オンデマンド可視化
    └── output/               # 実行結果（Git 管理外）
```

## exp01 物理モデル（概要）

- 半径 `a` のビーズを、傾き角 `phi` の直線上でスカラー座標 `s_1(t)` により記述する
- 運動方程式: `ds_1/dt = γ₀ (F₀ cos(ωt) − k s_1)`，`γ₀ = 1/(6πμa)`
- 瞬時流量（式(2.61) 形式）: `q_x = (h/(πμ)) F_x`，`F_x = f_total cos(phi)`
- 単一スライダーでは 1周期平均 **Q ≈ 0**

## exp02/exp03 — 2体モデル（要点・壁なし）

- **駆動力**: `f_active,1 = F₀ cos(ωt + Δ)`，`f_active,2 = F₀ cos(ωt)`（スライダー1に +Δ）
- **配置**: `r1_base = (−l/2, 0, h)`，`r2_base = (+l/2, 0, h)`
- **相互作用**: Oseen テンソル + 直線拘束 `ṙ_i · e_s⊥ = 0`
- **2体流量**: `q_x = [z₁ F_x,1 + z₂ F_x,2] / (πμ)`，`z_i = h − s_i sin(phi)`

## exp04 — 2体モデル（壁あり Blakelet、単点）

- **移動度**: `BlakeletTwoSliderMobility`（式(2.44)(2.51)、式(4.4)）
- **流量・拘束**: exp02 と同型。`compute_two_slider_Q_blakelet` / `exp04_two_sliders_wall.py`

## アニメーション（オンデマンド）

```bash
cd cilia_simulation
python optionrun/animate_from_run.py
python optionrun/animate_from_run.py --run-dir output/exp01_single_slider/<YYYYMMDD_HHMMSS>
```

MP4 生成には [ffmpeg](https://ffmpeg.org/) を要する場合あり。

## 参照

- 論文第4章: スライダーモデル、位相差・流量
- 式(2.39)(2.40): Oseen テンソル
- 式(2.61)(2.62)(4.65): 瞬時・平均流量
- 式(4.3): 駆動力

## License

Copyright (c) 2025–2026 Sota Osaki.

Research and educational use is permitted. Contact the author for other uses.

---

# slidermodel_sim (English)

A staged Python reimplementation of the **1D slider model** (a simplified cilia model) from Chapter 4 of the master's thesis.
The paper-reproduction goal is to reproduce the **net flow rate Q** and the **optimal phase difference Δ_opt** (through exp05).
The subsequent research goal is, with **l\* fixed**, to vary the relative placement angle θ in the x–y plane and verify how the flow-maximization strategy differs with θ (Δ_opt(θ)) — exp06, after exp05 is complete.

The source code is located under [`cilia_simulation/`](cilia_simulation/).

## Current Progress

| Stage | Script | Status | Description |
|-------|--------|--------|-------------|
| exp01 | `experiments/exp01_single_slider.py` | Done | Single slider, Q≈0 verification |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | Done | Two sliders, Oseen interaction |
| exp03 | `experiments/exp03_sweep_delta_l.py` | Done | Δ×l sweep, Q(Δ,l), Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | Done | With-wall Blakelet (single-point validation) |
| exp05 | `experiments/exp05_sweep_delta_l_wall.py` | Done | With-wall Blakelet Δ×l sweep |
| exp06 | `experiments/exp06_sweep_delta_theta.py` | Not implemented | Δ×θ sweep (l fixed) — **implement after exp05** |

### Limitations of the Current Model

| Stage | Mobility (force→velocity) | Flow rate $Q$ |
|-------|---------------------------|---------------|
| exp01–03 | Oseen (no wall) | Eq. (2.61) far-field at wall z=0 |
| **exp04** | **Blakelet (wall z=0)** | Eq. (2.61)(4.65) (consistent with mobility) |

**Quantitative comparison** with the paper's figures uses the exp05 **fine sweep** (`--mode fine`).

See [`cilia_simulation/PHYSICS_REPORT.md`](cilia_simulation/PHYSICS_REPORT.md) for the physical and mathematical details.

## Setup

```bash
cd cilia_simulation
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run all of the following inside `cilia_simulation/`.

### exp01 — Single Slider

```bash
python experiments/exp01_single_slider.py
```

**Output**: `output/exp01_single_slider/<YYYYMMDD_HHMMSS>/`

### exp02 — Two Sliders (No Wall)

```bash
python experiments/exp02_two_sliders_nowall.py
```

**Output**: `output/exp02_two_sliders_nowall/<YYYYMMDD_HHMMSS>/`
**Artifacts**: summary / PNG including time series, phase diagram, and Q

### exp03 — Δ×l Sweep

```bash
# EXP03_DEFAULT_MODE in config/default_params.py is the default when no CLI
# option is given (currently: "fine"). For exploration, specify --mode fast explicitly
python experiments/exp03_sweep_delta_l.py --mode fast

# For paper comparison (Delta 8000 points + RK45)
python experiments/exp03_sweep_delta_l.py --mode fine --workers 4

# Debug (serial)
python experiments/exp03_sweep_delta_l.py --workers 1
```

When running via the IDE ▷ button, set `EXP03_DEFAULT_MODE` in `config/default_params.py` to `"fast"` or `"fine"`.

| `--mode` | Delta points | Integration | Use |
|----------|--------------|-------------|-----|
| `fast` | 360 | Euler | Everyday exploration and plotting |
| `fine` | 8000 | RK45 | Quantitative comparison with paper figures |

**Output**: `output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS>/`
**Artifacts**: CSV, heatmap, Δ_opt curve

### exp04 — Two Sliders (With-Wall Blakelet, Single Point)

```bash
python experiments/exp04_two_sliders_wall.py
```

Parameters: `EXP04_DEFAULTS` in `config/default_params.py` (same single point as exp02). Integration: `EXP04_SOLVER_PRESET` (RK45, 10 periods).

**Output**: `output/exp04_two_sliders_wall/<YYYYMMDD_HHMMSS>/`

**Artifacts**: `parameters.json`, `summary.txt`, trajectory/force/phase PNG. `summary.txt` records `Q_blakelet` and Oseen reference `Q_oseen_reference`.

### exp05 — Δ×l Sweep (With-Wall Blakelet)

```bash
python experiments/exp05_sweep_delta_l_wall.py --mode fast
python experiments/exp05_sweep_delta_l_wall.py --mode fine --workers 4
```

When running via the IDE ▷ button, set `EXP05_DEFAULT_MODE` in `config/default_params.py`.

**Output**: `output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>/`

## Tests

```bash
cd cilia_simulation
python -m unittest discover tests -v
```

## Directory Structure

```
slidermodel_sim/
├── README.md                 # This file (for the GitHub top page)
└── cilia_simulation/
    ├── README.md             # For work inside cilia_simulation (kept in sync with this file)
    ├── PHYSICS_REPORT.md     # Physics and math explanation (exp01–exp05)
    ├── config/               # Default parameters
    ├── core/                 # Shared libraries (force, mobility, integration, flow, sweep)
    ├── experiments/          # exp01–exp06 (through exp05 done; exp06 placeholder)
    ├── tests/
    ├── optionrun/            # On-demand visualization
    └── output/               # Run results (not under Git management)
```

## exp01 Physical Model (Overview)

- A bead of radius `a` is described by a scalar coordinate `s_1(t)` on a straight line with tilt angle `phi`
- Equation of motion: `ds_1/dt = γ₀ (F₀ cos(ωt) − k s_1)`, `γ₀ = 1/(6πμa)`
- Instantaneous flow (Eq. (2.61) form): `q_x = (h/(πμ)) F_x`, `F_x = f_total cos(phi)`
- For a single slider, the one-period average is **Q ≈ 0**

## exp02/exp03 — Two-Body Model (Key Points)

- **Driving force**: `f_active,1 = F₀ cos(ωt + Δ)`, `f_active,2 = F₀ cos(ωt)` (+Δ on slider 1)
- **Placement**: `r1_base = (−l/2, 0, h)`, `r2_base = (+l/2, 0, h)`
- **Interaction**: Oseen tensor + straight-line constraint `ṙ_i · e_s⊥ = 0`
- **Two-body flow**: `q_x = [z₁ F_x,1 + z₂ F_x,2] / (πμ)`, `z_i = h − s_i sin(phi)`

## Animation (On-Demand)

```bash
cd cilia_simulation
python optionrun/animate_from_run.py
python optionrun/animate_from_run.py --run-dir output/exp01_single_slider/<YYYYMMDD_HHMMSS>
```

MP4 generation may require [ffmpeg](https://ffmpeg.org/).

## References

- Thesis Chapter 4: slider model, phase difference and flow rate
- Eqs. (2.39)(2.40): Oseen tensor
- Eqs. (2.61)(2.62)(4.65): instantaneous and average flow
- Eq. (4.3): driving force

## License

Copyright (c) 2025–2026 Sota Osaki.

Research and educational use is permitted. Contact the author for other uses.
