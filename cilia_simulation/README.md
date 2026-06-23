# cilia_simulation

修士論文第4章の **1次元スライダーモデル**（繊毛簡略化モデル）を Python で段階的に再実装する。  
論文再現の目標は **正味流量 Q** および **最適位相差 Δ_opt** の再現（exp05 まで）。  
exp06 では **l\* 固定**のもと x–y 平面内の相対配置角 θ を `EXP06_LAYOUT_THETA` で指定し、θ に応じた流量最大化戦略（Δ_opt, Q_max）の相違を検証する。

GitHub リポジトリトップの README は [`../README.md`](../README.md) に配置（内容同期）。

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

exp03 は Oseen 上の定性的トレンド検証。論文 Fig との **定量比較** は exp05 の **fine 掃引**（`--mode fine`）で評価する。

物理・数学の詳細は [`PHYSICS_REPORT.md`](PHYSICS_REPORT.md) を参照。

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

**出力物**: `parameters.json`, `summary.txt`, `trajectory_s1s2.png`, `forces_f1f2.png`, `phase_portrait_s1_vs_s2.png`

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

IDE ▷ 実行時は `config/default_params.py` の `EXP03_DEFAULT_MODE` を変更する。

| `--mode` | Delta 点数 | 積分 | 用途 |
|----------|------------|------|------|
| `fast` | 360 | Euler | 日常の探索・プロット |
| `fine` | 8000 | RK45 | 論文 Fig との定量比較 |

**出力先**: `output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS>/`

**出力物**:

- `q_delta_l.csv`, `delta_opt_vs_l.csv`, `q_vs_delta_fixed_l.csv`
- `Q_heatmap_delta_l.png`, `delta_opt_vs_l.png`, `Q_vs_delta_fixed_l.png`

### exp04 — 2本スライダー（壁あり Blakelet、単点）

```bash
python experiments/exp04_two_sliders_wall.py
```

パラメータは `config/default_params.py` の `EXP04_DEFAULTS`（exp02 単点と同一の $(l, \Delta)$）。積分は `EXP04_SOLVER_PRESET`（RK45, 10 周期）。

**出力先**: `output/exp04_two_sliders_wall/<YYYYMMDD_HHMMSS>/`

**出力物**: `parameters.json`, `summary.txt`, `trajectory_s1s2.png`, `forces_f1f2.png`, `phase_portrait_s1_vs_s2.png`

**summary の要点**:

- `Q_blakelet` … Blakelet 移動度での周期平均流量
- `Q_oseen_reference` … 同条件の Oseen（exp02 相当）参考値
- `beads_above_wall_passed` … 全時刻で $z_i > 0$ か

1ケースの $Q$ のみ必要なときは `core/two_slider.py` の `compute_two_slider_Q_blakelet` も利用可。

### exp05 — Δ×l 掃引（壁あり Blakelet、θ=0）

```bash
# 探索（Delta 360 点 + Euler）
python experiments/exp05_sweep_delta_l_wall.py --mode fast

# 論文比較（Delta 8000 点 + RK45）
python experiments/exp05_sweep_delta_l_wall.py --mode fine --workers 4

python experiments/exp05_sweep_delta_l_wall.py --workers 1
```

IDE ▷ 実行時は `config/default_params.py` の `EXP05_DEFAULT_MODE` を変更する。

**出力先**: `output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS>/`

**出力物**: exp03 と同型（`q_delta_l.csv`, `delta_opt_vs_l.csv`, ヒートマップ等）

### exp06 — Δ×l 掃引（壁あり Blakelet、配置角 θ 固定）

exp05 と**同一の Δ×l 掃引・同一の出力グラフ**。2 スライダーの x-y 平面内配置角 θ を `EXP06_LAYOUT_THETA` で固定する（既定: **45°**）。θ を変えたいときは `default_params.py` の値を書き換えて再実行する。

```bash
# 探索（Delta 360 点 + Euler）
python experiments/exp06_sweep_delta_theta.py --mode fast

# 高精度（Delta 8000 点 + RK45）
python experiments/exp06_sweep_delta_theta.py --mode fine --workers 4

python experiments/exp06_sweep_delta_theta.py --workers 1
```

IDE ▷ 実行時は `config/default_params.py` の `EXP06_DEFAULT_MODE` と `EXP06_LAYOUT_THETA` を変更する。

**出力先**: `output/exp06_sweep_delta_theta/<YYYYMMDD_HHMMSS>/`

**出力物**: exp05 と同型（`q_delta_l.csv`, `delta_opt_vs_l.csv`, ヒートマップ等）

## テスト

```bash
python -m unittest discover tests -v
```

## ディレクトリ構成

```
cilia_simulation/
├── README.md              # 本ファイル
├── PHYSICS_REPORT.md      # 物理・数学解説（exp01〜exp06）
├── .cursorrules           # コーディング規約
├── requirements.txt
├── config/
│   └── default_params.py
├── core/
│   ├── slider.py
│   ├── hydrodynamics.py
│   ├── solver.py
│   ├── flow_rate.py
│   ├── two_slider.py      # 1ケース Q（Oseen / Blakelet）
│   ├── sweep.py           # 並列パラメータ掃引
│   ├── progress.py
│   ├── stokeslet_field.py
│   ├── animation.py
│   └── utils.py
├── experiments/
│   ├── exp01_single_slider.py
│   ├── exp02_two_sliders_nowall.py
│   ├── exp03_sweep_delta_l.py
│   ├── exp04_two_sliders_wall.py       # Blakelet 単点
│   ├── exp05_sweep_delta_l_wall.py     # Blakelet Δ×l 掃引（θ=0）
│   └── exp06_sweep_delta_theta.py      # Blakelet Δ×l 掃引（θ 固定）
├── tests/
│   ├── test_single_slider.py
│   ├── test_blakelet_mobility.py
│   ├── test_exp03_sweep.py
│   ├── test_exp05_sweep.py
│   ├── test_exp06_sweep.py
│   ├── test_exp06_constraint.py
│   ├── test_exp06_theta_boundary.py
│   ├── test_exp06_sign_audit.py
│   ├── test_progress.py
│   └── test_sweep.py
├── optionrun/
│   ├── animate_from_run.py
│   ├── plot_q_vs_delta_from_run.py
│   ├── plot_phase_portrait_from_sweep_run.py
│   └── diagnose_exp06_theta_boundary.py
└── output/                # 実行結果（Git 管理外）
```

## exp01 物理モデル（概要）

- 半径 `a` のビーズを、傾き角 `phi` の直線上でスカラー座標 `s_1(t)` により記述する
- 運動方程式: `ds_1/dt = γ₀ (F₀ cos(ωt) − k s_1)`，`γ₀ = 1/(6πμa)`
- 瞬時流量（式(2.61) 形式、壁なしのため **指標として** 適用）:  
  `q_x = (h/(πμ)) F_x`，`F_x = f_total cos(phi)`
- **成功基準**: 1周期平均 `Q ≈ 0`、定常振動の位相遅れが理論値と一致

## exp02/exp03 — 2体モデル（要点・壁なし）

- **駆動力**: `f_active,1 = F₀ cos(ωt + Δ)`，`f_active,2 = F₀ cos(ωt)`（スライダー1に +Δ）
- **配置**: `r1_base = (−l/2, 0, h)`，`r2_base = (+l/2, 0, h)`
- **移動度**: 自己項 `M_aa = γ₀ I`、交差項 Oseen `M_ab = J/(8πμR)`，`J = I + (R⊗R)/R²`
- **拘束**: `ṙ_i · e_s⊥ = 0` より Lagrange 乗数 λ₁, λ₂ を決定する
- **2体流量**: `q_x = [z₁ F_x,1 + z₂ F_x,2] / (πμ)`，`z_i = h − s_i sin(phi)`

## 拘束力 λ と流量 F（2体モデル）

2体モデル（exp02 以降）では、力の扱いが **ダイナミクス** と **流量計算** で意図的に分かれている。

- **ダイナミクス（ODE）**: 3D 合力 `F_i = f_i,total e_s + λ_i e_s⊥`（θ≠0 では y 方向拘束も追加）。`core/hydrodynamics.py` の `_solve_two_slider_constrained_velocities` が λ を解き、ビーズをレール上に保つ。
- **流量（式(2.61)(4.65)）**: `F_x,i = f_i,total cos(phi)` のみ。λ の x 成分 `λ sin(phi)` は **意図的に除外**（論文と同一の流量指標定義）。

`f_i,total = F₀ cos(ωt + Δ_i) − k s_i` は能動力＋復元力のみ（λ を含まない）。

exp05（θ=0, 2×2 拘束）と exp06（θ≠0, 4×4 拘束）で Q(Δ) や Q_max が異なりうる。これは流量式に λ を入れないことによる直接効果ではなく、**幾何・水力結合・拘束ダイナミクスの違い**による。

## exp04 — 2体モデル（壁あり Blakelet、単点）

- **移動度**: `BlakeletTwoSliderMobility`（`core/hydrodynamics.py`）
  - 自己項: `M_aa = γ₀ I + M̂(z/a)`（式(2.51)(2.52)）
  - 交差項: Blakelet `G(r_i, r_j)`（式(2.44)(4.17)）
- **拘束・駆動力・流量**: exp02 と同型（式(4.4)(4.5)、式(2.61)(4.65)）
- **実装**: `compute_two_slider_Q_blakelet` / `experiments/exp04_two_sliders_wall.py`

## exp06 — 配置角 θ 固定（物理モデル）

- **配置**: `r_i,base = (±l/2 cos θ, ±l/2 sin θ, h)`（θ=0 で exp05 と一致）
- **拘束**: θ=0 → `e_s⊥` のみ（2×2）; θ≠0 → `e_s⊥` + `e_y`（4×4）
- **比較**: exp05（θ=0）と exp06（例: θ=45°）の `delta_opt_vs_l.csv` の Q_max を比較可能

## オンデマンド可視化（optionrun）

### アニメーション（exp01）

```bash
python optionrun/animate_from_run.py
python optionrun/animate_from_run.py --run-dir output/exp01_single_slider/<YYYYMMDD_HHMMSS>
```

**オプション**: `--gif`, `--frames 60`, `--loop-count 2`  
MP4 生成には [ffmpeg](https://ffmpeg.org/) を要する場合あり。

### Q(Δ) 重ね描き（exp03 / exp05 / exp06）

```bash
python optionrun/plot_q_vs_delta_from_run.py --run-dir output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS>
```

### 過渡収束確認用相図（exp03 / exp05 / exp06）

指定 `l*` で再積分し、全軌道（細い青）と Q 評価窓・最終1周期（赤）の s₂* vs s₁* 相図を保存する。
**Delta を省略**すると、その run の `delta_opt_vs_l.csv` から `l*` における **Δ_opt** を自動選択する。

```bash
# l* のみ → その l* で Q を最大化する Δ の相図
python optionrun/plot_phase_portrait_from_sweep_run.py \
  --run-dir output/exp05_sweep_delta_l_wall/<YYYYMMDD_HHMMSS> \
  --l 2.0

# 位相差を手動指定する場合（従来どおり）
python optionrun/plot_phase_portrait_from_sweep_run.py \
  --run-dir output/exp03_sweep_delta_l/<YYYYMMDD_HHMMSS> \
  --l 2.0 --delta-deg 90
```

### θ=0 境界診断（exp06）

θ=0°（2×2 拘束）と θ=1°（4×4 拘束）で Q(Δ) を重ね描きし、拘束系切り替えの影響を確認する。

```bash
python optionrun/diagnose_exp06_theta_boundary.py
python optionrun/diagnose_exp06_theta_boundary.py --l 2.0 --output output/diag_theta_boundary_l2.png
```

## 参照

- 論文第4章: スライダーモデル、位相差・流量
- 式(2.39)(2.40): Oseen テンソル
- 式(2.44)(2.51): Blakelet / 壁自己項（exp04）
- 式(2.61)(2.62)(4.65): 瞬時・平均流量
- 式(4.3)(4.4)(4.5): 駆動力・2体運動（拘束）
- 4.3.6 節: 単一スライダーで Q₀=0

## License

Copyright (c) 2025–2026 Sota Osaki.

Research and educational use is permitted. Contact the author for other uses.
