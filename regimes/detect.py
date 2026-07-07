import numpy as np
import pandas as pd

from features.engineer import build_features


def _regime_from_signal(trend_signal: str, vol_label: str) -> str:
    return f"{trend_signal}_{vol_label}"


def label_regime(df: pd.DataFrame, hysteresis: int = 3) -> pd.Series:
    """Classify into 4 regimes (bullish/bearish x high/low vol) with hysteresis smoothing."""
    if "close" not in df.columns:
        raise ValueError("label_regime requires a 'close' column")
    if "vol_60d" not in df.columns:
        df = build_features(df.copy())

    trend_ma = df["close"].rolling(200, min_periods=200).mean()
    trend_signal = np.where(df["close"] > trend_ma, "bullish", "bearish")

    vol_percentile = df["vol_60d"].rolling(60, min_periods=60).rank(pct=True) * 100
    vol_label = np.where(vol_percentile > 75, "high", "low")

    regimes = pd.Series(
        [_regime_from_signal(ts, vl) for ts, vl in zip(trend_signal, vol_label)],
        index=df.index, dtype="object",
    )

    if hysteresis <= 0:
        return regimes

    smoothed = regimes.copy()
    for i in range(1, len(regimes)):
        prev = smoothed.iloc[i - 1]
        current = regimes.iloc[i]
        if prev != current:
            if i < hysteresis:
                smoothed.iloc[i] = prev
            else:
                smoothed.iloc[i] = current if current == regimes.iloc[i - hysteresis] else prev
    return smoothed
