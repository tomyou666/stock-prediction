# Jupyter Notebook セットアップテンプレート

このプロジェクトは、Jupyter Notebookでデータ分析を素早く始めるためのテンプレートです。Pythonの主要なデータ分析ライブラリが事前に設定されており、日本語フォントの設定も含まれています。

## 含まれるライブラリ

- **pandas**: データ操作・分析
- **numpy**: 数値計算
- **matplotlib**: グラフ作成
- **seaborn**: 統計的データ可視化
- **japanize-matplotlib**: 日本語フォント対応
- **openpyxl**: Excelファイル読み書き
- **jupyter**: Jupyter Notebook環境
- **ruff**: コードフォーマッター

## セットアップ手順

### 1. 前提条件

- Python 3.11以上
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー

### 2. プロジェクトのクローン

```bash
git clone <このリポジトリのURL>
cd <プロジェクト名>
```

### 3. 依存関係のインストール

```bash
uv sync
```

#### ※依存関係を佐伸する場合
```bash
uv add japanize-matplotlib jupyter matplotlib numpy openpyxl pandas ruff seaborn
uv lock --upgrade
```

### 4. Jupyter Notebookの起動

```bash
uv run jupyter notebook
```

または

```bash
uv run jupyter lab
```

### 5. サンプルノートブックの実行

`sample.ipynb` を開いて、セットアップが正しく動作することを確認してください。このサンプルでは：

- ライブラリのインポート
- 日本語フォントの設定
- 幾何ブラウン運動のシミュレーション例

が含まれています。

## 使用方法

### 新しいノートブックの作成

1. Jupyter Notebookを起動
2. 「New」→「Python 3」を選択
3. 以下のコードを最初のセルに貼り付けて実行：

```python
# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

# 日本語フォントの設定
try:
    import japanize_matplotlib
    print("japanize_matplotlib を使用して日本語フォントを設定しました")
except ImportError:
    # japanize_matplotlibが利用できない場合の代替設定
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    print("代替フォント設定を使用しました")
sns.set(font="IPAexGothic")

print("ライブラリのインポート完了")
```

### 追加ライブラリのインストール

新しいライブラリが必要な場合：

```bash
uv add <ライブラリ名>
```

例：
```bash
uv add scikit-learn
uv add plotly
```

## トラブルシューティング

### 日本語フォントが表示されない場合

1. システムに日本語フォントがインストールされていることを確認
2. `japanize_matplotlib` が正しくインストールされていることを確認
3. 代替フォント設定が適用されていることを確認

### Jupyter Notebookが起動しない場合

1. `uv sync` で依存関係が正しくインストールされていることを確認
2. 仮想環境が正しくアクティベートされていることを確認

## プロジェクト構造

```
.
├── README.md           # このファイル
├── pyproject.toml      # プロジェクト設定と依存関係
├── uv.lock            # 依存関係のロックファイル
└── sample.ipynb       # サンプルノートブック
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要望は、GitHubのIssuesページでお知らせください。
