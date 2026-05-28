# cilia_simulation

修士論文で扱った **1次元スライダーモデル**（繊毛の簡略化モデル）を Python で段階的に再実装するプロジェクトです。  
最終目標は、論文第4章の流体力学的相互作用による **正味流量 Q** と **最適位相差 Δ_opt** の再現です。

## 段階的開発ロードマップ

| 段階 | ディレクトリ / スクリプト | 内容 |
|------|---------------------------|------|
| **第1段階** | `experiments/exp01_single_slider.py` | 単一スライダー、壁なし、自由空間。Q≈0 の検証 |
| **第2段階** | `experiments/exp02_two_sliders_nowall.py` | 2本スライダー、Stokeslet 相互作用 |
| **第3段階** | `experiments/exp03_sweep_delta_l.py` | 壁なしで `Delta × l` 掃引、`Q(Delta,l)` 解析 |
| **第4段階** | `experiments/exp04_two_sliders_wall.py` | 2本スライダー、壁あり、Blakelet（論文の最終形） |

各段階は **前段階の `core/` を拡張** し、実験スクリプトは履歴として残します。

## 現在の進捗メモ（exp02 準備）

- `config/default_params.py` に `EXP02_DEFAULTS` を追加済み。
  - 初期値: `l=2.0`, `delta=pi/2`
  - 将来の掃引用に `delta_min`, `delta_max`, `delta_points` も定義済み
- `core/hydrodynamics.py` に拘束付き `TwoSliderMobility` を追加済み。
  - 自己項: `M_aa = gamma_0 I`, `gamma_0 = 1/(6*pi*mu*a)`
  - 交差項: Oseen テンソルを使用
  - 拘束条件: `rdot_i · e_s_perp = 0` から `lambda_1, lambda_2` を連立で解く
- `core/solver.py` に `TwoSliderTimeStepper` と `TwoSliderResult` を追加済み。
  - 状態: `y = [s1, s2]`
  - 駆動: `f_active1 = F0 cos(omega t)`, `f_active2 = F0 cos(omega t - delta)`
  - 幾何: `r1 = r1_base + s1 e_s`, `r2 = r2_base + s2 e_s`
    - 論文式に合わせて `r1_base=(-l/2, 0, h)`, `r2_base=(+l/2, 0, h)` を採用
  - 速度: `TwoSliderMobility.compute_velocities(...)` から `(ds1/dt, ds2/dt)` を取得
  - 積分: `RK45` / `Euler` 両対応、最後の1周期を steady window として切り出し
- `experiments/exp02_two_sliders_nowall.py` を新規作成済み。
  - `EXP02_DEFAULTS` を読み込み、`TwoSliderMobility` + `TwoSliderTimeStepper` で時系列を計算
  - 出力: `output/exp02_two_sliders_nowall/<timestamp>/`
  - `Q` 定義は論文式(2.62)と4章式(4.65)に合わせ、
    `Q=(1/T)∫q_x(t)dt`, `q_x=(z1*Fx1+z2*Fx2)/(pi*mu)`,
    `z_i=h-s_i sin(phi)`, `Fx_i=f_i_total cos(phi)` で評価
  - 保存物:
    - `parameters.json`
    - `summary.txt`（有限値チェック、定常振幅、`corr(s1,s2)`、`Q` と評価窓を含む）
    - `trajectory_s1s2.png`（`s1(t), s2(t)`）
    - `forces_f1f2.png`（`f1(t), f2(t)`）
    - `phase_portrait_s1_vs_s2.png`（相図 `s2` vs `s1`）
- `config/default_params.py` に `EXP03_SWEEP_DEFAULTS` を追加済み。
  - `Delta` 掃引: `delta_min`, `delta_max`, `delta_points`
  - `l` 掃引: `l_min`, `l_max`, `l_points`
  - 固定パラメータ（`a, mu, k, F_0, omega, phi, h`）は exp02 と同じ基準値
- `core/utils.py` に掃引用プロット関数を追加済み。
  - `plot_q_heatmap_delta_l(...)`
  - `plot_delta_opt_vs_l(...)`
  - `plot_q_vs_delta_fixed_l(...)`（論文 Fig.4.2 相当）
  - これにより `Q(Delta,l)` と `Delta_opt(l)` を実験スクリプト側から共通呼び出し可能
- `core/progress.py` を追加済み。
  - `SweepProgressTracker` で sweep 実行時の進捗表示を共通化
  - `tqdm` の1行更新表示と EMA 残り時間推定 (`eta_ema_s`) を再利用可能
- `experiments/exp03_sweep_delta_l.py` を実装済み。
  - `EXP03_SWEEP_DEFAULTS` を読み込み、`Delta × l` 掃引で `Q` を評価
  - `FlowCalculator.compute_two_slider_Q_from_result(...)` を用いて各ケースの `Q` を統一定義で計算
  - 実行中は `core.progress.SweepProgressTracker` を利用して進捗表示
  - postfix に `l`, `delta_deg`, `eta_ema_s`（EMAベース残り時間推定）を表示
  - 出力: `output/exp03_sweep_delta_l/<timestamp>/`
  - 保存物:
    - `parameters.json`, `summary.txt`（`elapsed_seconds`, `elapsed_minutes` を含む）
    - `q_delta_l.csv`（`l,delta_rad,delta_deg,Q`）
    - `delta_opt_vs_l.csv`（`l,delta_opt_rad,delta_opt_deg,Q_max`）
    - `q_vs_delta_fixed_l.csv`（`delta_rad,delta_deg,Q` at fixed `l`）
    - `Q_heatmap_delta_l.png`
    - `delta_opt_vs_l.png`
    - `Q_vs_delta_fixed_l.png`（`l` は `EXP02_DEFAULTS["l"]` を固定値として使用）
- グラフの無次元量ラベルは `$Q^{*}$`, `$l^{*}$` のように上付き `*` を付ける（`.cursorrules` セクション6、`core/utils.py` の `dimless_label`）。
- Stokeslet の表記は論文に合わせて整理済み。
  - 核テンソル: `J = I + (R⊗R)/R^2`（Eq.2.40 形式）
  - 物理次元付き相互移動度: `M_ab = (1/(8*pi*mu*R)) * J`（Eq.2.39 と同値）

## ディレクトリ構成

```
cilia_simulation/
├── .cursorrules           # コーディング規約
├── README.md              # 本ファイル
├── requirements.txt
├── config/                # デフォルト実数パラメータ（マジックナンバー禁止）
│   └── default_params.py  # EXP01_DEFAULTS など
├── core/                  # 共通ライブラリ
│   ├── slider.py
│   ├── hydrodynamics.py
│   ├── solver.py
│   ├── flow_rate.py
│   ├── stokeslet_field.py
│   ├── animation.py
│   └── utils.py
├── optionrun/             # オンデマンド可視化（実験とは別実行）
│   └── animate_from_run.py
├── experiments/           # 実行スクリプト
│   ├── exp01_single_slider.py
│   ├── exp02_two_sliders_nowall.py
│   ├── exp03_sweep_delta_l.py
│   └── exp04_two_sliders_wall.py
├── tests/
│   └── test_single_slider.py
└── output/                # 実行結果（タイムスタンプ付き）
    ├── exp01_single_slider/
    ├── animations/          # アニメーション（オンデマンド）
    ├── exp02_two_sliders_nowall/
    ├── exp03_sweep_delta_l/
    └── exp04_two_sliders_wall/
```

## 第1段階 (exp01) の物理モデル（概要）

- 半径 `a` のビーズが、傾き角 `phi` の直線上をスカラー座標 `s_1(t)` で運動。
- 直線方向の運動方程式: `ds_1/dt = γ₀ (f_active + f_spring)`  
  - `γ₀ = 1/(6πμa)`（Stokes 並進移動度）  
  - `f_active = F₀ cos(ωt)`（論文 式(4.3) 準拠）  
  - `f_spring = -k s_1`
- 瞬時流量（論文 式(2.61) 形式、第1段階では壁なしのため **指標として** 使用）:  
  `q_x = (h/(πμ)) F_x`, `F_x = f_total cos(phi)`
- **成功基準**: 1周期平均流量 `Q ≈ 0`、定常振動の位相遅れが理論と一致。

⚠️ 式(2.61) は本来 Blakelet / 滑りなし壁の遠方近似に基づく。第4段階で壁を入れるまで、流量は「論文と同じ定義での準備段階」として扱う。

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

## 実行（第1段階・実装後）

```bash
python experiments/exp01_single_slider.py
```

結果は `output/exp01_single_slider/<YYYYMMDD_HHMMSS>/` に保存されます。

## アニメーション（オンデマンド）

数値実験を再実行せず、保存済み run からスライダーと流体（xz 流線・Stokeslet 近似）の動画を生成します。

**IDE の▷（Run Python File）**: `optionrun/animate_from_run.py` を開いて実行。  
`RUN_DIR_FOR_PLAY = None` なら `output/exp01_single_slider/` の **最新** run を自動選択。

```bash
# 最新 run を自動選択
python optionrun/animate_from_run.py

# 特定の run を指定
python optionrun/animate_from_run.py --run-dir output/exp01_single_slider/<YYYYMMDD_HHMMSS>
```

オプション: `--gif`（GIF も保存）, `--frames 60`, `--loop-count 2`

出力: `output/animations/exp01_single_slider/<source_timestamp>_<anim_timestamp>/`

- `animation.mp4`（ffmpeg がある場合）または `animation.gif`
- `preview.png`, `anim_config.json`

`flow_rate.csv` は定常窓1周期分をループ再生します。MP4 には [ffmpeg](https://ffmpeg.org/) が必要な場合があります（無い場合は GIF のみ）。

## 出力物

各実行フォルダに以下を保存します。

- `parameters.json` — 全パラメータ（再現用）
- `trajectory.png` — `s_1(t)`（最後の2周期）
- `flow_rate.png` — `q_x(t)`（最後の2周期）
- `flow_rate.csv` — 時系列データ
- `summary.txt` — 平均流量 Q、理論比較、収束チェック

## 参照

- 論文第4章: スライダーモデル、位相差・流量
- 論文 式(4.3): 駆動力
- 論文 式(2.61): 瞬時流量（第3段階で物理的正当化）
- 論文 4.3.6節: 単一スライダーで Q₀=0

## ライセンス・引用

（必要に応じて追記）
