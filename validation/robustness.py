from typing import Dict, List

import numpy as np
import pandas as pd

from signals.apply import apply_signal_lagged


def _safe_returns(prices: pd.DataFrame) -> pd.Series:
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    if "close" in prices.columns:
        close = pd.to_numeric(prices["close"], errors="coerce")
    else:
        numeric_cols = [c for c in prices.columns if pd.api.types.is_numeric_dtype(prices[c])]
        close = pd.to_numeric(prices[numeric_cols[0]], errors="coerce") if numeric_cols else pd.Series(dtype=float)
    close = pd.Series(close, index=prices.index, dtype=float)
    return close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def walk_forward_split(prices: pd.DataFrame, signal: str, window_size: int, n_splits: int) -> List[float]:
    if prices is None or prices.empty or window_size <= 0 or n_splits <= 0:
        return []
    n_rows = len(prices)
    if n_rows < window_size * 2 + 1:
        return []

    results = []
    for i in range(min(n_splits, max(0, n_rows - window_size * 2))):
        train_end = (i + 1) * window_size
        if train_end >= n_rows:
            break
        in_sample = prices.iloc[i * window_size:train_end]
        out_sample = prices.iloc[train_end: min(train_end + window_size, n_rows)]
        if len(out_sample) < 2 or len(in_sample) < window_size:
            continue
        returns = _safe_returns(out_sample)
        # Lagged within this window only (apply_signal_lagged shifts using
        # out_sample's own length/index, so day 0 of out_sample can never be a
        # trade day — no leakage across the train/test boundary from in_sample).
        mask = apply_signal_lagged(out_sample, signal).to_numpy()
        if not mask.any():
            continue
        signal_returns = returns[mask]
        if len(signal_returns) < 2:
            continue
        std = signal_returns.std(ddof=0)
        if not np.isfinite(std) or np.isclose(std, 0.0):
            continue
        results.append(float(np.sqrt(252) * (signal_returns.mean() / std)))
    return results


def monte_carlo_test(signal: str, prices: pd.DataFrame, n_shuffles: int = 500) -> Dict[str, float]:
    returns = _safe_returns(prices)
    if returns.empty or len(returns) < 2:
        return {"signal": signal, "actual_sharpe": float("nan"), "p_value": 1.0}

    mask = apply_signal_lagged(prices, signal).to_numpy()
    n_trades = int(mask.sum())
    if n_trades < 2:
        return {"signal": signal, "actual_sharpe": float("nan"), "p_value": 1.0}

    returns_arr = returns.to_numpy()
    signal_returns = returns_arr[mask]
    std = signal_returns.std(ddof=0)
    actual_sharpe = float(np.sqrt(252) * (signal_returns.mean() / std)) if np.isfinite(std) and std > 0 else float("nan")
    if not np.isfinite(actual_sharpe):
        return {"signal": signal, "actual_sharpe": actual_sharpe, "p_value": 1.0}

    # Null distribution: keep the same number of "in trade" days, but randomize
    # which days count by permuting the mask itself (not the return values) so
    # each shuffle still draws from the real, unshuffled return series.
    null_dist = []
    for _ in range(max(1, int(n_shuffles))):
        shuffled_mask = np.random.permutation(mask)
        shuffled_returns = returns_arr[shuffled_mask]
        s_std = shuffled_returns.std(ddof=0)
        s_sharpe = float(np.sqrt(252) * (shuffled_returns.mean() / s_std)) if np.isfinite(s_std) and s_std > 0 else float("nan")
        null_dist.append(s_sharpe)
    null_dist = np.array(null_dist, dtype=float)
    p_value = float(np.mean(np.isfinite(null_dist) & (null_dist >= actual_sharpe)))
    return {"signal": signal, "actual_sharpe": actual_sharpe, "p_value": p_value}


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []
    valid = [float(p) for p in p_values if pd.notna(p)]
    if not valid:
        return [False for _ in p_values]

    ranked = sorted(enumerate(valid), key=lambda item: item[1])
    n = len(ranked)
    threshold = None
    for rank, (_, p) in enumerate(ranked, start=1):
        candidate = (rank / n) * alpha
        if p <= candidate:
            threshold = candidate
        else:
            break

    if threshold is None:
        return [False for _ in p_values]

    significant = [p <= threshold for p in valid]
    result, idx = [], 0
    for p in p_values:
        if pd.isna(p):
            result.append(False)
        else:
            result.append(significant[idx])
            idx += 1
    return result


def regime_consistency_check(signal: str, prices: pd.DataFrame, regimes: List[str]) -> bool:
    if prices is None or prices.empty or not regimes:
        return False
    close = pd.to_numeric(prices["close"] if "close" in prices.columns else prices.iloc[:, 0], errors="coerce").astype(float)
    returns = close.pct_change().fillna(0.0)
    if returns.empty:
        return False

    signal_mask = apply_signal_lagged(prices, signal).to_numpy()

    regime_returns = []
    for regime in regimes:
        if regime == "bull":
            regime_mask = close > close.rolling(5, min_periods=5).mean()
        elif regime == "bear":
            regime_mask = close < close.rolling(5, min_periods=5).mean()
        else:
            regime_mask = pd.Series(np.ones(len(prices), dtype=bool), index=close.index)
        combined_mask = regime_mask.to_numpy() & signal_mask & np.isfinite(returns).to_numpy()
        regime_slice = returns[combined_mask]
        if len(regime_slice) >= 5:
            regime_returns.append(regime_slice.mean())
    return sum(v > 0 for v in regime_returns) >= 2


if __name__ == "__main__":
    sample = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "ret_5d_z": np.random.randn(100),
        "rsi": np.random.randint(0, 100, size=100),
    })
    print(walk_forward_split(sample, "ret_5d_z > 0.5 AND rsi < 30", 50, 5))
    print(monte_carlo_test("ret_5d_z > 0.5 AND rsi < 30", sample))
    print(fdr_correction([0.05, 0.02, 0.07, 0.01]))
    print(regime_consistency_check("ret_5d_z > 0.5", sample, ["bull", "bear"]))
