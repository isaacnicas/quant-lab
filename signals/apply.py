import re
from typing import List, Tuple

import numpy as np
import pandas as pd

_CONDITION_RE = re.compile(r"^([A-Za-z0-9_.]+)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)$")


def _parse_signal(signal: str) -> List[Tuple[str, str, float]]:
    parsed = []
    for condition in re.split(r"\s+AND\s+", signal.strip(), flags=re.IGNORECASE):
        match = _CONDITION_RE.match(condition.strip())
        if not match:
            return []
        feature, op, threshold = match.groups()
        parsed.append((feature, op, float(threshold)))
    return parsed


def compute_signal_mask(prices: pd.DataFrame, signal: str) -> np.ndarray:
    """Raw, same-bar boolean mask of which rows the signal's conditions are true on."""
    n = len(prices) if prices is not None else 0
    if not signal or prices is None or prices.empty:
        return np.zeros(n, dtype=bool)

    conditions = _parse_signal(signal)
    if not conditions:
        return np.zeros(n, dtype=bool)

    mask = np.ones(n, dtype=bool)
    for feature, op, threshold in conditions:
        if feature not in prices.columns:
            return np.zeros(n, dtype=bool)
        values = pd.to_numeric(prices[feature], errors="coerce").astype(float).to_numpy()
        finite = np.isfinite(values)
        if op == "<":
            mask &= finite & (values < threshold)
        elif op == ">":
            mask &= finite & (values > threshold)
        elif op == "<=":
            mask &= finite & (values <= threshold)
        elif op == ">=":
            mask &= finite & (values >= threshold)
    return mask


def lag_mask(mask: np.ndarray, index: pd.Index) -> pd.Series:
    """Shift a boolean mask by one bar: True on day t means a trade opened on day t+1."""
    return pd.Series(mask, index=index, dtype=bool).shift(1, fill_value=False)


def apply_signal_lagged(prices: pd.DataFrame, signal: str) -> pd.Series:
    """
    Signal mask shifted by one bar into a trade mask: a signal computed from day
    t's features only opens a trade on day t+1. Without this lag, signals built
    from same-day return features (e.g. ret_1d_z) are partly a restatement of
    the very return being measured against, which inflates backtest Sharpe and
    Monte Carlo significance for that class of signal.
    """
    mask = compute_signal_mask(prices, signal)
    index = prices.index if prices is not None else pd.RangeIndex(len(mask))
    return lag_mask(mask, index)
