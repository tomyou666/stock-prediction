# 株価指数予測：PSO-LSTM（5分足・終値）

本リポジトリは、論文「Enhancing stock index prediction: A hybrid LSTM-PSO model for improved forecasting accuracy」を参考に、**PSOでLSTMのハイパーパラメータを最適化**して、**5分足の終値**を予測するJupyter実装です。
学習は **5分 / 30分 / 60分** の3粒度で行い、yfinance のデータを使用します。

## ノートブック

- `pso_lstm_5m.ipynb`
  5分足データの取得、特徴量生成、Wavelet DWT、PSO最適化、学習・評価までを実行します。

## 予測対象

- **出力（正解値）**: 次時点の終値（`close`）
- **入力**:
  - OHLCV（5分足）
  - テクニカル指標
  - マクロ指標（USD/JPY, 金利）

## 主要仕様

- データ取得: `yfinance`
- Wavelet DWT: Haar / 3レベル
- ルックバック: 5分 / 30分 / 60分
- 学習評価: RMSE / MAE / MAPE / R2
- PSO:
  - 粒子数 `N=20`
  - 反復 `K=50`
  - `w=0.8, c1=1.5, c2=1.5`

## セットアップ

### Dev Container で起動

VS Code / Cursor でこのリポジトリを開き、**Dev Containers** 拡張機能を使ってコンテナで起動できます。

1. リポジトリをクローンして開く
2. 「Reopen in Container」でコンテナ内で開く（またはコマンドパレットで `Dev Containers: Reopen in Container`）
3. 依存関係インストール

```bash
uv sync
```

4. Jupyter起動

```bash
uv run jupyter lab
```

Python や uv はコンテナ内に用意されているため、ローカルに Python を入れなくても開発できます。

### ローカルで起動する場合

#### 前提

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー

#### 依存関係インストール

```bash
uv sync
```

#### Jupyter起動

```bash
uv run jupyter lab
```

## 注意事項

- yfinanceの**5分足は直近数十日分のみ**取得可能です。データ不足時は期間を短く調整してください。
- マクロ指標（USD/JPY, 金利）は日次データのため、5分足へは前方補完で揃えます。
- GPUがない環境でも動きますが、GPUがある前提のコード構成です。

## プロジェクト構造

```
.
├── README.md
├── pyproject.toml
├── uv.lock
├── sample.ipynb
└── pso_lstm_5m.ipynb
```
