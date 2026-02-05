"""
PSO-LSTM 株価予測の共通関数・定数。
pso_lstm_5m.ipynb と pso_lstm_5m_visualize.ipynb から利用する。
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import tensorflow as tf
from tensorflow import keras
import pyswarms as ps
from tensorflow.keras import layers

# L2正則化の係数（build_lstm_model のデフォルト）
L2_LAMBDA = 1e-4

# PSO・学習のデフォルト（pso_optimize の引数で上書き可能）
BATCH_SIZE = 32
NEURON_BOUNDS = (50, 300)  # ニューロン数 論文は(0, 300)
EPOCH_BOUNDS = (50, 300)  # 反復回数（エポック）: 論文は(50, 300)
LAYER_BOUNDS = (1, 3)  # 評価する隠れ層: 論文は (1, 3)
PSO_W = 0.8  # 慣性重み: 論文は 0.8
PSO_C1 = 1.5  # 加速定数: 論文は 1.5
PSO_C2 = 1.5  # 加速定数: 論文は 1.5
# PSO_PARTICLES × PSO_ITERS の計算が行われるため小さいほうがいい
PSO_PARTICLES = 20  # グループサイズ: 論文は 20
PSO_ITERS = 1  # PSO の最大反復回数: 論文は 50


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのインデックスをdatetimeに変換し、タイムゾーンを除去する。"""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def download_price_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """yfinanceで指定銘柄・足・期間の価格データ（OHLCV）をダウンロードし、正規化して返す。"""
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"データ取得に失敗: {ticker}")
    df = _normalize_index(df)
    df = df.rename(columns=str.lower)
    return df


def download_macro_daily(ticker: str, period_years: int = 5) -> pd.Series:
    """日足のマクロデータ（為替・金利など）を指定年数分ダウンロードし、終値のSeriesで返す。"""
    df = yf.download(ticker, interval="1d", period=f"{period_years}y", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"マクロデータ取得に失敗: {ticker}")
    df = _normalize_index(df)
    series = df["Close"].copy()
    series.name = ticker
    return series


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """OHLCVを指定分足（5/30/60分など）にリサンプルする。5分の場合はそのままコピーを返す。"""
    if minutes == 5:
        return df.copy()
    rule = f"{minutes}min"
    ohlc = df["open"].resample(rule).first()
    high = df["high"].resample(rule).max()
    low = df["low"].resample(rule).min()
    close = df["close"].resample(rule).last()
    volume = df["volume"].resample(rule).sum()
    out = pd.concat([ohlc, high, low, close, volume], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()


def merge_macro_features(price_df: pd.DataFrame, fx: pd.Series, rate: pd.Series) -> pd.DataFrame:
    """価格DataFrameに為替（fx）と金利（rate）の列を前方補完で結合する。"""
    df = price_df.copy()
    fx = fx.reindex(df.index, method="ffill")
    rate = rate.reindex(df.index, method="ffill")
    df["usd_jpy"] = fx
    df["interest_rate"] = rate
    return df


def resample_series(series: pd.Series, minutes: int) -> pd.Series:
    """時系列を指定分足にリサンプルする（各区間の最後の値を採用）。"""
    if minutes == 5:
        return series.copy()
    rule = f"{minutes}min"
    return series.resample(rule).last().dropna()


def wavelet_denoise(series: pd.Series, wavelet: str = "haar", level: int = 3) -> pd.Series:
    """ウェーブレット変換で時系列のノイズを除去する。"""
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    arr = np.asarray(series, dtype=np.float64).ravel().copy()
    coeffs = pywt.wavedec(arr, wavelet, level=level)
    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745 if len(detail_coeffs) > 0 else 0
    uthresh = sigma * np.sqrt(2 * np.log(len(arr))) if sigma > 0 else 0
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs, wavelet)
    reconstructed = reconstructed[: len(arr)]
    name = getattr(series, "name", None)
    index = series.index if hasattr(series, "index") else pd.RangeIndex(len(reconstructed))
    return pd.Series(reconstructed, index=index, name=name)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVにテクニカル指標（MACD, CCI, ATR, ボリンジャー, EMA, MA, モメンタム, ROC, SMI, WVAD）を追加する。"""
    out = df.copy()
    idx = out.index
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.Series(np.asarray(out[col], dtype=float).ravel(), index=idx)

    macd = ta.trend.MACD(close=out["close"].squeeze())
    out["macd"] = macd.macd_diff().values.reshape(-1, 1)
    out["cci"] = ta.trend.cci(out["high"].squeeze(), out["low"].squeeze(), out["close"].squeeze()).values.reshape(
        -1, 1
    )
    out["atr"] = ta.volatility.average_true_range(
        out["high"].squeeze(), out["low"].squeeze(), out["close"].squeeze()
    ).values.reshape(-1, 1)
    boll = ta.volatility.BollingerBands(close=out["close"].squeeze())
    out["boll"] = boll.bollinger_mavg().values.reshape(-1, 1)
    out["ema20"] = ta.trend.EMAIndicator(out["close"].squeeze(), window=20).ema_indicator().values.reshape(-1, 1)
    out["ma5"] = out["close"].rolling(5).mean()
    out["ma10"] = out["close"].rolling(10).mean()
    out["mtm6"] = out["close"] - out["close"].shift(6)
    out["mtm12"] = out["close"] - out["close"].shift(12)
    out["roc"] = ta.momentum.ROCIndicator(out["close"].squeeze(), window=10).roc().values.reshape(-1, 1)
    hl = (out["high"] + out["low"]) / 2
    diff = out["close"] - hl
    hl_range = out["high"] - out["low"]
    ema1 = diff.ewm(span=14, adjust=False).mean()
    ema2 = ema1.ewm(span=14, adjust=False).mean()
    range_ema1 = hl_range.ewm(span=14, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=14, adjust=False).mean()
    out["smi"] = 100 * (ema2 / (0.5 * range_ema2.replace(0, np.nan)))
    denom = (out["high"] - out["low"]).replace(0, np.nan)
    out["wvad"] = ((out["close"] - out["open"]) / denom) * out["volume"]
    return out


def _flatten_column_index(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex 列を 1 段階の文字列列名に変換する（呼び出し側が文字列で列を指定できるようにする）。"""
    if df.columns.nlevels <= 1:
        return df.copy()
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            part = [str(x).strip() for x in c if x]
            new_cols.append("_".join(part) if part else str(c[0]))
        else:
            new_cols.append(str(c))
    out = df.copy()
    out.columns = new_cols
    return out


def remove_high_corr_features(df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    """目的変数との絶対相関がthresholdを超える特徴量を削除し、削除した列名のリストも返す。
    MultiIndex 列の場合は先に平坦化し、列名は常に文字列のリストで返す。"""
    df = _flatten_column_index(df)
    corr = df.corr(numeric_only=True)
    if target_col not in corr.columns:
        raise ValueError(f"target_col '{target_col}' not found in the DataFrame correlation matrix.")
    corr_series = corr[target_col].abs()
    if isinstance(corr_series, pd.DataFrame):
        corr_series = corr_series.squeeze()
    vals = np.asarray(corr_series).ravel()
    names = list(corr_series.index)
    is_high = (vals > threshold).astype(bool)
    not_self = np.array([n != target_col for n in names], dtype=bool)
    mask = is_high & not_self
    drop_cols = [names[i] for i in range(len(names)) if mask[i]]
    drop_cols = [col for col in drop_cols if col != target_col]
    return df.drop(columns=drop_cols), drop_cols


def build_target_close(df: pd.DataFrame) -> pd.Series:
    """終値の次時点の値（翌バーの終値）を予測対象の目的変数として返す。"""
    close = pd.Series(np.asarray(df["close"]).ravel(), index=df.index)
    next_close = close.shift(-1)
    next_close.name = "target_close"
    return next_close


def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """LSTM用に、過去lookbackステップの特徴量を入力・その時点の目的変数を出力とするシーケンスの組 (X, y) を作成する。"""
    xs, ys = [], []
    for i in range(lookback, len(features)):
        xs.append(features[i - lookback : i])
        ys.append(target[i])
    return np.array(xs), np.array(ys)


def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.2):
    """時系列順を保ったまま、先頭から train_ratio を訓練、その中の val_ratio を検証、残りをテストに分割する。"""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(train_end * (1 - val_ratio))
    X_train = X[:val_end]
    y_train = y[:val_end]
    X_val = X[val_end:train_end]
    y_val = y[val_end:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_train_val_test(X_train, X_val, X_test, y_train, y_val, y_test):
    """訓練データでMinMaxScaler(-1,1)をfitし、訓練・検証・テストの特徴量と目的変数をスケーリングする。スケーラーも返す。"""
    n_features = X_train.shape[2]
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = x_scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_s = x_scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1))
    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler


def build_lstm_model(input_shape, num_layers: int, num_units: int, l2_lambda: float | None = None):
    """指定した層数・ユニット数でスタックLSTMモデルを構築する。各層にDropout(0.2)、L2正則化、最終層はDense(1)、損失はMSE。"""
    if l2_lambda is None:
        l2_lambda = L2_LAMBDA
    l2_reg = keras.regularizers.l2(l2_lambda)
    model = keras.Sequential()
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        lstm_kw = dict(
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            bias_regularizer=l2_reg,
        )
        if i == 0:
            model.add(
                layers.LSTM(
                    num_units,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                    **lstm_kw,
                )
            )
        else:
            model.add(layers.LSTM(num_units, return_sequences=return_sequences, **lstm_kw))
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, kernel_regularizer=l2_reg, bias_regularizer=l2_reg))
    model.compile(optimizer="adam", loss="mse")
    return model


def pso_optimize(
    X_train,
    y_train,
    X_val,
    y_val,
    input_shape,
    csv_log_path="pso_training.csv",
    *,
    batch_size=None,
    neuron_bounds=None,
    epoch_bounds=None,
    layer_bounds=None,
    pso_w=None,
    pso_c1=None,
    pso_c2=None,
    pso_particles=None,
    pso_iters=None,
    strategy=None,
):
    """PSOでLSTMのハイパーパラメータを探索し、検証RMSEが最小の解とそのコストを返す。
    strategy: tf.distribute.Strategy (複数GPU時は MirroredStrategy)。None の場合は単一デバイス。
    """
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    neuron_bounds = neuron_bounds if neuron_bounds is not None else NEURON_BOUNDS
    epoch_bounds = epoch_bounds if epoch_bounds is not None else EPOCH_BOUNDS
    layer_bounds = layer_bounds if layer_bounds is not None else LAYER_BOUNDS
    pso_w = pso_w if pso_w is not None else PSO_W
    pso_c1 = pso_c1 if pso_c1 is not None else PSO_C1
    pso_c2 = pso_c2 if pso_c2 is not None else PSO_C2
    pso_particles = pso_particles if pso_particles is not None else PSO_PARTICLES
    pso_iters = pso_iters if pso_iters is not None else PSO_ITERS

    def _objective(particles):
        costs = []
        for particle in particles:
            units = int(np.clip(round(particle[0]), *neuron_bounds))
            epochs = int(np.clip(round(particle[1]), *epoch_bounds))
            n_layers = int(np.clip(round(particle[2]), *layer_bounds))
            tf.keras.backend.clear_session()
            if strategy is not None:
                with strategy.scope():
                    model = build_lstm_model(input_shape, n_layers, units)
            else:
                model = build_lstm_model(input_shape, n_layers, units)
            es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            csv_log = keras.callbacks.CSVLogger(csv_log_path)
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[es, csv_log],
            )
            preds = model.predict(X_val, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            costs.append(rmse)
        return np.array(costs)

    options = {"c1": pso_c1, "c2": pso_c2, "w": pso_w}
    lower_bounds = np.array([neuron_bounds[0], epoch_bounds[0], layer_bounds[0]])
    upper_bounds = np.array([neuron_bounds[1], epoch_bounds[1], layer_bounds[1]])
    bounds = (lower_bounds, upper_bounds)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=pso_particles,
        dimensions=3,
        options=options,
        bounds=bounds,
    )
    best_cost, best_pos = optimizer.optimize(_objective, iters=pso_iters, verbose=False)
    return best_pos, best_cost


def compute_metrics(y_true, y_pred):
    """予測と実測から RMSE, MAE, MAPE(%), R2 を計算し、タプルで返す。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2
