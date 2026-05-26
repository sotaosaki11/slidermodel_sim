# cilia_simulation

修士論文で扱った **1次元スライダーモデル**（繊毛の簡略化モデル）を Python で段階的に再実装するプロジェクトです。  
最終目標は、論文第4章の流体力学的相互作用による **正味流量 Q** と **最適位相差 Δ_opt** の再現です。

## 段階的開発ロードマップ

| 段階 | ディレクトリ / スクリプト | 内容 |
|------|---------------------------|------|
| **第1段階** | `experiments/exp01_single_slider.py` | 単一スライダー、壁なし、自由空間。Q≈0 の検証 |
| **第2段階** | `experiments/exp02_two_sliders_nowall.py` | 2本スライダー、Stokeslet 相互作用 |
| **第3段階** | `experiments/exp03_two_sliders_wall.py` | 2本スライダー、壁あり、Blakelet（論文の最終形） |

各段階は **前段階の `core/` を拡張** し、実験スクリプトは履歴として残します。

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
│   └── utils.py
├── experiments/           # 実行スクリプト
│   ├── exp01_single_slider.py
│   ├── exp02_two_sliders_nowall.py
│   └── exp03_two_sliders_wall.py
├── tests/
│   └── test_single_slider.py
└── output/                # 実行結果（タイムスタンプ付き）
    ├── exp01_single_slider/
    ├── exp02_two_sliders_nowall/
    └── exp03_two_sliders_wall/
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

⚠️ 式(2.61) は本来 Blakelet / 滑りなし壁の遠方近似に基づく。第3段階で壁を入れるまで、流量は「論文と同じ定義での準備段階」として扱う。

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
