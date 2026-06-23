# slidermodel_sim

修士論文第4章の **1次元スライダーモデル**（繊毛簡略化モデル）を Python で段階的に再実装する。
論文再現の目標は **正味流量 Q** および **最適位相差 Δ_opt** の再現（exp05 まで）。
exp06 では **l\* 固定**のもと x–y 平面内の相対配置角 θ を `EXP06_LAYOUT_THETA` で指定し、θ に応じた流量最大化戦略（Δ_opt, Q_max）の相違を検証する。

ソースコードは [`cilia_simulation/`](cilia_simulation/) 配下に配置。

## 現段階の進捗

| 段階 | スクリプト | 状態 | 内容 |
|------|-----------|------|------|
| exp01 | `experiments/exp01_single_slider.py` | 完了 | 単一スライダー、Q≈0 検証 |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | 完了 | 2本スライダー、Oseen 相互作用 |
| exp03 | `experiments/exp03_sweep_delta_l.py` | 完了 | Δ×l 掃引、Q(Δ,l)、Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | 完了 | 壁あり Blakelet（単点検証） |
| exp05 | `experiments/exp05_sweep_delta_l_wall.py` | 完了 | 壁あり Blakelet Δ×l 掃引（θ=0） |
| exp06 | `experiments/exp06_sweep_delta_theta.py` | 完了 | 壁あり Blakelet Δ×l 掃引（配置角 θ 固定、既定 45°） |

### 現モデルの限界

| 段階 | 移動度（力→速度） | 流量 $Q$ |
|------|-------------------|----------|
| exp01〜03 | Oseen（壁なし） | 式(2.61)（壁 z=0 遠方近似） |
| **exp04〜06** | **Blakelet（壁 z=0）** | 式(2.61)(4.65)（移動度と整合） |

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

### exp05 — Δ×l 掃引（壁あり Blakelet、θ=0）

```bash
python experiments/exp05_sweep_delta_l_wall.py --mode fast
python experiments/exp05_sweep_delta_l_wall.py --mode fine --workers 4
```

IDE ▷ 実行時は `config/default_params.py` の `EXP05_DEFAULT_MODE` を変更する。

**出力先**: `output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>/`

**出力物**: exp03 と同型（`q_delta_l.csv`, `delta_opt_vs_l.csv`, ヒートマップ等）

### exp06 — Δ×l 掃引（壁あり Blakelet、配置角 θ 固定）

exp05 と**同一の Δ×l 掃引・同一の出力グラフ**。2 スライダーの x-y 平面内配置角 θ を `EXP06_LAYOUT_THETA` で固定する（既定: **45°**）。

```bash
python experiments/exp06_sweep_delta_theta.py --mode fast
python experiments/exp06_sweep_delta_theta.py --mode fine --workers 4
```

IDE ▷ 実行時は `config/default_params.py` の `EXP06_DEFAULT_MODE` と `EXP06_LAYOUT_THETA` を変更する。

**出力先**: `output/exp06_sweep_delta_theta/<YYYYMMDD_HHMMSS>/`

**出力物**: exp05 と同型（`q_delta_l.csv`, `delta_opt_vs_l.csv`, ヒートマップ等）

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
    ├── PHYSICS_REPORT.md     # 物理・数学解説（exp01〜exp06）
    ├── config/               # デフォルトパラメータ
    ├── core/                 # 共通ライブラリ（力・移動度・積分・流量・掃引）
    ├── experiments/          # exp01〜exp06
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

## 拘束力 λ と流量 F（2体モデル）

2体モデル（exp02 以降）では、力の扱いが **ダイナミクス** と **流量計算** で意図的に分かれている。

- **ダイナミクス（ODE）**: 3D 合力 `F_i = f_i,total e_s + λ_i e_s⊥`（θ≠0 では y 方向拘束も追加）。`core/hydrodynamics.py` が λ を解き、ビーズをレール上に保つ。
- **流量（式(2.61)(4.65)）**: `F_x,i = f_i,total cos(phi)` のみ。λ の x 成分は **意図的に除外**（論文式(4.65) の指標定義）。

`f_i,total = F₀ cos(ωt + Δ_i) − k s_i` は能動力＋復元力のみ。

exp05（θ=0, 2×2 拘束）と exp06（θ≠0, 4×4 拘束）で Q_max が異なりうるのは、流量式に λ を入れないことではなく、**幾何・水力結合・拘束ダイナミクスの違い**による。

## exp04 — 2体モデル（壁あり Blakelet、単点）

- **移動度**: `BlakeletTwoSliderMobility`（式(2.44)(2.51)、式(4.4)）
- **拘束・駆動力・流量**: exp02 と同型。`compute_two_slider_Q_blakelet` / `exp04_two_sliders_wall.py`

## exp06 — 配置角 θ 固定（物理モデル）

- **配置**: `r_i,base = (±l/2 cos θ, ±l/2 sin θ, h)`（θ=0 で exp05 と一致）
- **拘束**: θ=0 → `e_s⊥` のみ（2×2）; θ≠0 → `e_s⊥` + `e_y`（4×4）
- **比較**: exp05（θ=0）と exp06（例: θ=45°）の `delta_opt_vs_l.csv` の Q_max を比較可能

## オンデマンド可視化（optionrun）

```bash
cd cilia_simulation
python optionrun/animate_from_run.py
python optionrun/plot_q_vs_delta_from_run.py --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>
python optionrun/plot_phase_portrait_from_sweep_run.py --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS> --l 2.0
python optionrun/diagnose_exp06_theta_boundary.py
```

MP4 生成には [ffmpeg](https://ffmpeg.org/) を要する場合あり。詳細は [`cilia_simulation/README.md`](cilia_simulation/README.md) を参照。

## 参照

- 論文第4章: スライダーモデル、位相差・流量
- 式(2.39)(2.40): Oseen テンソル
- 式(2.44)(2.51): Blakelet / 壁自己項（exp04）
- 式(2.61)(2.62)(4.65): 瞬時・平均流量
- 式(4.3)(4.4)(4.5): 駆動力・2体運動（拘束）

## License

Copyright (c) 2025–2026 Sota Osaki.

Research and educational use is permitted. Contact the author for other uses.

---

# slidermodel_sim (English)

A staged Python reimplementation of the **1D slider model** (a simplified cilia model) from Chapter 4 of the master's thesis.
The paper-reproduction goal is to reproduce the **net flow rate Q** and the **optimal phase difference Δ_opt** (through exp05).
In **exp06**, with **l\* fixed**, the relative placement angle θ in the x–y plane is set via `EXP06_LAYOUT_THETA` to study how the flow-maximization strategy (Δ_opt, Q_max) differs with θ.

The source code is located under [`cilia_simulation/`](cilia_simulation/).

## Current Progress

| Stage | Script | Status | Description |
|-------|--------|--------|-------------|
| exp01 | `experiments/exp01_single_slider.py` | Done | Single slider, Q≈0 verification |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | Done | Two sliders, Oseen interaction |
| exp03 | `experiments/exp03_sweep_delta_l.py` | Done | Δ×l sweep, Q(Δ,l), Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | Done | With-wall Blakelet (single-point validation) |
| exp05 | `experiments/exp05_sweep_delta_l_wall.py` | Done | With-wall Blakelet Δ×l sweep (θ=0) |
| exp06 | `experiments/exp06_sweep_delta_theta.py` | Done | With-wall Blakelet Δ×l sweep (layout angle θ fixed, default 45°) |

### Limitations of the Current Model

| Stage | Mobility (force→velocity) | Flow rate $Q$ |
|-------|---------------------------|---------------|
| exp01–03 | Oseen (no wall) | Eq. (2.61) far-field at wall z=0 |
| **exp04–06** | **Blakelet (wall z=0)** | Eq. (2.61)(4.65) (consistent with mobility) |

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

### exp05 — Δ×l Sweep (With-Wall Blakelet, θ=0)

```bash
python experiments/exp05_sweep_delta_l_wall.py --mode fast
python experiments/exp05_sweep_delta_l_wall.py --mode fine --workers 4
```

When running via the IDE ▷ button, set `EXP05_DEFAULT_MODE` in `config/default_params.py`.

**Output**: `output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>/`

**Artifacts**: same as exp03 (`q_delta_l.csv`, `delta_opt_vs_l.csv`, heatmaps, etc.)

### exp06 — Δ×l Sweep (With-Wall Blakelet, Layout Angle θ Fixed)

Same Δ×l grid and output format as exp05. The x–y placement angle θ is fixed by `EXP06_LAYOUT_THETA` (default: **45°**).

```bash
python experiments/exp06_sweep_delta_theta.py --mode fast
python experiments/exp06_sweep_delta_theta.py --mode fine --workers 4
```

When running via the IDE ▷ button, set `EXP06_DEFAULT_MODE` and `EXP06_LAYOUT_THETA` in `config/default_params.py`.

**Output**: `output/exp06_sweep_delta_theta/<YYYYMMDD_HHMMSS>/`

**Artifacts**: same as exp05

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
    ├── PHYSICS_REPORT.md     # Physics and math explanation (exp01–exp06)
    ├── config/               # Default parameters
    ├── core/                 # Shared libraries (force, mobility, integration, flow, sweep)
    ├── experiments/          # exp01–exp06
    ├── tests/
    ├── optionrun/            # On-demand visualization
    └── output/               # Run results (not under Git management)
```

## exp01 Physical Model (Overview)

- A bead of radius `a` is described by a scalar coordinate `s_1(t)` on a straight line with tilt angle `phi`
- Equation of motion: `ds_1/dt = γ₀ (F₀ cos(ωt) − k s_1)`, `γ₀ = 1/(6πμa)`
- Instantaneous flow (Eq. (2.61) form): `q_x = (h/(πμ)) F_x`, `F_x = f_total cos(phi)`
- For a single slider, the one-period average is **Q ≈ 0**

## exp02/exp03 — Two-Body Model (Key Points, No Wall)

- **Driving force**: `f_active,1 = F₀ cos(ωt + Δ)`, `f_active,2 = F₀ cos(ωt)` (+Δ on slider 1)
- **Placement**: `r1_base = (−l/2, 0, h)`, `r2_base = (+l/2, 0, h)`
- **Interaction**: Oseen tensor + straight-line constraint `ṙ_i · e_s⊥ = 0`
- **Two-body flow**: `q_x = [z₁ F_x,1 + z₂ F_x,2] / (πμ)`, `z_i = h − s_i sin(phi)`

## Constraint Force λ and Flow Force F (Two-Body Model)

In two-body models (exp02 onward), force handling is **intentionally split** between dynamics and flow evaluation.

- **Dynamics (ODE)**: 3D force `F_i = f_i,total e_s + λ_i e_s⊥` (plus y-constraint when θ≠0). `core/hydrodynamics.py` solves for λ to keep beads on the rail.
- **Flow (Eqs. (2.61)(4.65))**: `F_x,i = f_i,total cos(phi)` only. The x-component of λ is **deliberately excluded** (same flow metric as the thesis, Eq. (4.65)).

`f_i,total = F₀ cos(ωt + Δ_i) − k s_i` is active + spring force only (no λ).

Differences in Q_max between exp05 (θ=0, 2×2 constraints) and exp06 (θ≠0, 4×4 constraints) arise from **geometry, hydrodynamic coupling, and constrained dynamics**, not from omitting λ in the flow formula.

## exp04 — Two-Body Model (With-Wall Blakelet, Single Point)

- **Mobility**: `BlakeletTwoSliderMobility` (Eqs. (2.44)(2.51), (4.4))
- **Constraints, driving, flow**: same structure as exp02. `compute_two_slider_Q_blakelet` / `exp04_two_sliders_wall.py`

## exp06 — Fixed Layout Angle θ (Physical Model)

- **Placement**: `r_i,base = (±l/2 cos θ, ±l/2 sin θ, h)` (matches exp05 when θ=0)
- **Constraints**: θ=0 → `e_s⊥` only (2×2); θ≠0 → `e_s⊥` + `e_y` (4×4)
- **Comparison**: compare Q_max from `delta_opt_vs_l.csv` between exp05 (θ=0) and exp06 (e.g. θ=45°)

## On-Demand Visualization (optionrun)

```bash
cd cilia_simulation
python optionrun/animate_from_run.py
python optionrun/plot_q_vs_delta_from_run.py --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>
python optionrun/plot_phase_portrait_from_sweep_run.py --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS> --l 2.0
python optionrun/diagnose_exp06_theta_boundary.py
```

MP4 generation may require [ffmpeg](https://ffmpeg.org/). See [`cilia_simulation/README.md`](cilia_simulation/README.md) for details.

## References

- Thesis Chapter 4: slider model, phase difference and flow rate
- Eqs. (2.39)(2.40): Oseen tensor
- Eqs. (2.44)(2.51): Blakelet / wall self-term (exp04)
- Eqs. (2.61)(2.62)(4.65): instantaneous and average flow
- Eqs. (4.3)(4.4)(4.5): driving force and two-body motion (constraints)

## License

Copyright (c) 2025–2026 Sota Osaki.

Research and educational use is permitted. Contact the author for other uses.
