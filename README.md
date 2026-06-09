# slidermodel_sim

修士論文第4章の **1次元スライダーモデル**（繊毛簡略化モデル）を Python で段階的に再実装する。
論文再現の目標は **正味流量 Q** および **最適位相差 Δ_opt** の再現（exp04 まで）。
その先の研究目標は、**l\* 固定**のもと x–y 平面内で相対配置角 θ を変化させ、θ に応じた流量最大化戦略（Δ_opt(θ)）の相違を検証すること（exp05、exp04 完了後）。

ソースコードは [`cilia_simulation/`](cilia_simulation/) 配下に配置。

## 現段階の進捗

| 段階 | スクリプト | 状態 | 内容 |
|------|-----------|------|------|
| exp01 | `experiments/exp01_single_slider.py` | 完了 | 単一スライダー、Q≈0 検証 |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | 完了 | 2本スライダー、Oseen 相互作用 |
| exp03 | `experiments/exp03_sweep_delta_l.py` | 完了 | Δ×l 掃引、Q(Δ,l)、Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | 未実装 | 壁あり Blakelet（論文最終形） |
| exp05 | `experiments/exp05_sweep_delta_theta.py` | 未実装 | Δ×θ 掃引（l 固定）— **exp04 後に実装** |

### 現モデルの限界

exp03 までの結果を解釈する際は、次の2点を区別する。

- **移動度**: 壁なし **Oseen / Stokeslet** を採用（論文最終形の Blakelet ではない）
- **流量 Q**: 論文 式(2.61) 系の **壁 z=0 遠方近似** を先行適用（exp04 で物理的整合を図る）

exp03 の出力は **定性的トレンド** および **数値パイプラインの検証** に有効。論文 Fig との **厳密な定量一致** は exp04 完了後に評価する。

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
    ├── PHYSICS_REPORT.md     # 物理・数学解説（exp01〜exp03）
    ├── config/               # デフォルトパラメータ
    ├── core/                 # 共通ライブラリ（力・移動度・積分・流量・掃引）
    ├── experiments/          # exp01〜exp05 実行スクリプト（exp04・exp05 はプレースホルダ）
    ├── tests/
    ├── optionrun/            # オンデマンド可視化
    └── output/               # 実行結果（Git 管理外）
```

## exp01 物理モデル（概要）

- 半径 `a` のビーズを、傾き角 `phi` の直線上でスカラー座標 `s_1(t)` により記述する
- 運動方程式: `ds_1/dt = γ₀ (F₀ cos(ωt) − k s_1)`，`γ₀ = 1/(6πμa)`
- 瞬時流量（式(2.61) 形式）: `q_x = (h/(πμ)) F_x`，`F_x = f_total cos(phi)`
- 単一スライダーでは 1周期平均 **Q ≈ 0**

## exp02/exp03 — 2体モデル（要点）

- **駆動力**: `f_active,1 = F₀ cos(ωt + Δ)`，`f_active,2 = F₀ cos(ωt)`（スライダー1に +Δ）
- **配置**: `r1_base = (−l/2, 0, h)`，`r2_base = (+l/2, 0, h)`
- **相互作用**: Oseen テンソル + 直線拘束 `ṙ_i · e_s⊥ = 0`
- **2体流量**: `q_x = [z₁ F_x,1 + z₂ F_x,2] / (πμ)`，`z_i = h − s_i sin(phi)`

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
The paper-reproduction goal is to reproduce the **net flow rate Q** and the **optimal phase difference Δ_opt** (through exp04).
The subsequent research goal is, with **l\* fixed**, to vary the relative placement angle θ in the x–y plane and verify how the flow-maximization strategy differs with θ (Δ_opt(θ)) — exp05, after exp04 is complete.

The source code is located under [`cilia_simulation/`](cilia_simulation/).

## Current Progress

| Stage | Script | Status | Description |
|-------|--------|--------|-------------|
| exp01 | `experiments/exp01_single_slider.py` | Done | Single slider, Q≈0 verification |
| exp02 | `experiments/exp02_two_sliders_nowall.py` | Done | Two sliders, Oseen interaction |
| exp03 | `experiments/exp03_sweep_delta_l.py` | Done | Δ×l sweep, Q(Δ,l), Δ_opt(l) |
| exp04 | `experiments/exp04_two_sliders_wall.py` | Not implemented | With-wall Blakelet (final paper form) |
| exp05 | `experiments/exp05_sweep_delta_theta.py` | Not implemented | Δ×θ sweep (l fixed) — **implement after exp04** |

### Limitations of the Current Model

When interpreting the results through exp03, distinguish the following two points.

- **Mobility**: Uses the wall-free **Oseen / Stokeslet** (not the Blakelet of the final paper form)
- **Flow rate Q**: Applies the **far-field approximation at wall z=0** of the paper's Eq. (2.61) family in advance (physical consistency to be addressed in exp04)

The exp03 output is useful for **qualitative trends** and for **validating the numerical pipeline**. **Strict quantitative agreement** with the paper's figures is to be evaluated after exp04 is complete.

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
    ├── PHYSICS_REPORT.md     # Physics and math explanation (exp01–exp03)
    ├── config/               # Default parameters
    ├── core/                 # Shared libraries (force, mobility, integration, flow, sweep)
    ├── experiments/          # exp01–exp05 run scripts (exp04 and exp05 are placeholders)
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
