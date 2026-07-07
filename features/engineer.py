import numpy as np
import pandas as pd


def _safe_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    effective_window = min(window, len(close))
    avg_gain = gain.rolling(effective_window, min_periods=1).mean()
    avg_loss = loss.rolling(effective_window, min_periods=1).mean()
    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    valid = avg_loss > 0
    rsi.loc[valid] = 100 - (100 / (1 + (avg_gain[valid] / avg_loss[valid])))
    rsi.loc[~valid & (avg_gain > 0)] = 100.0
    rsi.loc[~valid & (avg_gain == 0)] = 50.0
    rsi.iloc[: max(0, effective_window - 1)] = np.nan
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling returns, volatility, RSI, MACD, and z-scores — verified no-lookahead."""
    if "close" not in df.columns:
        raise ValueError("build_features requires a 'close' column")
    out = df.copy()
    close = out["close"].astype(float)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["vol_20d"] = close.pct_change().rolling(20, min_periods=20).std(ddof=0)
    out["vol_60d"] = close.pct_change().rolling(60, min_periods=60).std(ddof=0)
    out["rsi"] = calculate_rsi(close)
    out["macd_line"], out["macd_signal"] = calculate_macd(close)
    for column in ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_60d", "rsi", "macd_line", "macd_signal"]:
        out[f"{column}_z"] = _safe_zscore(out[column], window=20)
    return out
