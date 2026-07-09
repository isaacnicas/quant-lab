import hashlib
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from signals.apply import apply_signal_lagged

_EULER_MASCHERONI = 0.5772156649015329


def deterministic_seed(ticker: str, signal: str) -> int:
    """Python's built-in hash() randomizes str hashing per-process (PYTHONHASHSEED),
    so hash((ticker, signal)) gives a DIFFERENT value on every run -- defeating
    the point of a reproducible seed. SHA-256 is stable across processes.

    This is the single canonical implementation -- do not duplicate it
    elsewhere (see signals/apply.py for the analogous single-path principle
    applied to signal masking/lagging)."""
    digest = hashlib.sha256(f"{ticker}|{signal}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


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


def monte_carlo_test(
    signal: str,
    prices: pd.DataFrame,
    n_shuffles: int = 2000,
    ticker: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Seed resolution is structural, not caller-remembered (this is what bug #14
    was: seed=None silently fell back to fresh entropy). Order:
      1. explicit `seed` wins (back-compat / advanced use)
      2. else derive from `ticker` via deterministic_seed(ticker, signal)
      3. else raise -- there is no silent non-reproducible fallback.
    """
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

    if seed is None:
        if ticker is None:
            raise ValueError(
                "monte_carlo_test requires either a ticker (to derive a "
                "reproducible seed via deterministic_seed) or an explicit "
                "seed. Reproducibility must not silently fall back to entropy."
            )
        seed = deterministic_seed(ticker, signal)

    # A local Generator (rather than the global np.random state) keeps the
    # seed fully isolated to this call.
    rng = np.random.default_rng(seed)

    # Null distribution: keep the same number of "in trade" days, but randomize
    # which days count by permuting the mask itself (not the return values) so
    # each shuffle still draws from the real, unshuffled return series.
    null_dist = []
    for _ in range(max(1, int(n_shuffles))):
        shuffled_mask = rng.permutation(mask)
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


def expected_max_sharpe(n_trials: int, mean_sharpe: float, std_sharpe: float) -> float:
    """Expected maximum of n_trials iid Sharpe-ratio draws under the null
    (Bailey & Lopez de Prado 2014, "The Deflated Sharpe Ratio", eq. 8):

        E[max] ~= mean + std * ( (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) )

    where gamma is the Euler-Mascheroni constant and Phi^-1 is the inverse
    standard normal CDF. Unit-agnostic: mean_sharpe/std_sharpe must simply be
    in the same units as the Sharpe values being maximized over."""
    if n_trials <= 1:
        return float(mean_sharpe)
    n = float(n_trials)
    term1 = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
    term2 = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(mean_sharpe + std_sharpe * (term1 + term2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    returns: pd.Series,
    n_trials: int,
    mean_sharpe_trials: float,
    std_sharpe_trials: float,
) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    UNITS: `observed_sharpe`, `mean_sharpe_trials`, and `std_sharpe_trials`
    are all expected ANNUALIZED (sqrt(252)-scaled) -- matching every other
    Sharpe value in this codebase (run_backtest, monte_carlo_test, the
    `sharpe` column in the experiments table). The DSR formula itself is
    derived on the NON-annualized per-period Sharpe (the sqrt(T-1) term and
    the skew/kurtosis correction come from the per-period return
    distribution), so this function de-annualizes all three inputs by
    dividing by sqrt(252) before applying the formula. `sr0` in the return
    dict is re-annualized (multiplied back by sqrt(252)) purely for
    human-readable side-by-side comparison against `observed_sharpe`.

    `returns` must be the actual per-period (daily) strategy return series
    realized (NOT annualized) -- skew, kurtosis, and T (sample size) are
    computed directly from it.
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    T = len(r)
    if T < 3 or not np.isfinite(observed_sharpe):
        return {
            "dsr": float("nan"), "sr0": float("nan"), "skew": float("nan"),
            "kurtosis": float("nan"), "n_trials": int(n_trials), "T": int(T),
        }

    sqrt_252 = np.sqrt(252.0)
    sr_daily = observed_sharpe / sqrt_252
    mean_daily = mean_sharpe_trials / sqrt_252
    std_daily = std_sharpe_trials / sqrt_252

    sr0_daily = expected_max_sharpe(n_trials, mean_daily, std_daily)

    g3 = float(_skew(r))
    g4 = float(_kurtosis(r, fisher=False))  # Pearson's kurtosis (normal = 3), per the DSR formula

    denom = 1.0 - g3 * sr_daily + ((g4 - 1.0) / 4.0) * sr_daily ** 2
    if denom <= 0 or not np.isfinite(denom):
        dsr = float("nan")
    else:
        z = ((sr_daily - sr0_daily) * np.sqrt(T - 1)) / np.sqrt(denom)
        dsr = float(norm.cdf(z))

    return {
        "dsr": dsr,
        "sr0": float(sr0_daily * sqrt_252),
        "skew": g3,
        "kurtosis": g4,
        "n_trials": int(n_trials),
        "T": int(T),
    }


def probability_of_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 16) -> Dict[str, object]:
    """
    Combinatorially Symmetric Cross-Validation (CSCV) estimate of the
    Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado &
    Zhu 2017; Lopez de Prado 2018, "Advances in Financial Machine Learning",
    ch. 11).

    returns_matrix: shape (T, N) -- T periods (rows), N candidate strategies
    (columns), per-period (daily) returns.

    Splits T into n_splits contiguous blocks. For every combination of
    n_splits/2 blocks as in-sample (IS) and the complementary blocks as
    out-of-sample (OOS) -- C(n_splits, n_splits/2) combinations -- finds the
    IS-best strategy and computes its relative OOS rank among all N
    strategies. PBO is the fraction of combinations where that IS-best
    strategy's logit-transformed OOS rank is <= 0 (i.e. it landed in the
    bottom half OOS): a selection process no better than chance.

    With n_splits=16, C(16,8)=12,870 combinations -- do not raise n_splits
    without measuring runtime, since combinations grow combinatorially.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D: (T periods, N strategies)")
    T, N = returns_matrix.shape
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if T < n_splits:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed T ({T})")

    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2

    logits = []
    for is_blocks in combinations(range(n_splits), half):
        is_set = set(is_blocks)
        oos_block_ids = [b for b in range(n_splits) if b not in is_set]
        is_idx = np.concatenate([blocks[b] for b in sorted(is_set)])
        oos_idx = np.concatenate([blocks[b] for b in oos_block_ids])

        is_returns = returns_matrix[is_idx]
        oos_returns = returns_matrix[oos_idx]

        with np.errstate(invalid="ignore", divide="ignore"):
            is_std = is_returns.std(axis=0, ddof=0)
            is_sharpe = np.where(is_std > 0, is_returns.mean(axis=0) / is_std, -np.inf)
            oos_std = oos_returns.std(axis=0, ddof=0)
            oos_sharpe = np.where(oos_std > 0, oos_returns.mean(axis=0) / oos_std, -np.inf)

        n_star = int(np.argmax(is_sharpe))

        # Relative rank of the IS-best strategy's OOS performance among all N
        # strategies, scaled into (0, 1) via rank/(N+1) to avoid logit(0)/logit(1).
        rank_position = int(np.sum(oos_sharpe <= oos_sharpe[n_star]))  # 1..N
        omega = rank_position / (N + 1)
        logits.append(float(np.log(omega / (1.0 - omega))))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr <= 0))
    return {"pbo": pbo, "n_combinations": len(logits), "logits": logits_arr.tolist()}


if __name__ == "__main__":
    sample = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "ret_5d_z": np.random.randn(100),
        "rsi": np.random.randint(0, 100, size=100),
    })
    print(walk_forward_split(sample, "ret_5d_z > 0.5 AND rsi < 30", 50, 5))
    print(monte_carlo_test("ret_5d_z > 0.5 AND rsi < 30", sample, ticker="DEMO"))
    print(fdr_correction([0.05, 0.02, 0.07, 0.01]))
    print(regime_consistency_check("ret_5d_z > 0.5", sample, ["bull", "bear"]))
