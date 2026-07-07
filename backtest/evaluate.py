from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from signals.apply import compute_signal_mask, lag_mask


def _coerce_price_series(prices: pd.DataFrame) -> pd.Series:
    if prices is None or prices.empty or "close" not in prices.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(prices["close"], errors="coerce").astype(float)
    series.index = prices.index
    return series.replace([np.inf, -np.inf], np.nan)


def apply_stop_loss(trade_mask: np.ndarray, prices: pd.Series, stop_loss: float) -> np.ndarray:
    if prices is None or len(prices) < 2:
        return np.zeros_like(trade_mask, dtype=bool)
    stop_mask = np.zeros_like(trade_mask, dtype=bool)
    for i in range(1, len(prices)):
        if trade_mask[i - 1] and prices.iloc[i] < prices.iloc[i - 1] * (1 - stop_loss):
            stop_mask[i] = True
    return np.logical_and(trade_mask, ~stop_mask)


def calculate_returns(trade_mask: np.ndarray, prices: pd.Series, position_size: float) -> pd.Series:
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float, index=prices.index if prices is not None else pd.Index([]))
    price_returns = prices.pct_change().fillna(0.0)
    mask = np.asarray(trade_mask, dtype=bool) & np.isfinite(price_returns.to_numpy())
    return pd.Series(np.where(mask, price_returns.to_numpy() * position_size, 0.0), index=prices.index)


def apply_transaction_costs(
    returns: pd.Series, trade_mask: np.ndarray, cost_bps: float, position_size: float
) -> pd.Series:
    """Charge cost_bps/10000 of notional only on days the (lagged) trade mask
    flips — entry or exit — not as a blanket drag on every day a position is
    held. Scaled by position_size to match the units of `returns`, which is
    itself price-return * position_size: a cost expressed as bps of notional
    traded should only hit the portfolio at the same fraction as the position."""
    if returns is None or returns.empty:
        return returns
    mask = np.asarray(trade_mask, dtype=bool)
    prev = np.concatenate(([False], mask[:-1]))
    transitions = mask != prev
    cost = (cost_bps / 10000.0) * position_size
    net = returns.to_numpy(dtype=float).copy()
    net[transitions] -= cost
    return pd.Series(net, index=returns.index)


def calculate_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return float("nan")
    std = r.std(ddof=0)
    if np.isnan(std) or np.isclose(std, 0.0):
        return float("nan")
    return float(np.sqrt(252) * (r.mean() / std))


def calculate_max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r.empty:
        return float("nan")
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak.replace(0, np.nan)
    result = dd.min()
    return float(result) if not np.isnan(result) else float("nan")


def group_trades(trade_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end) index pairs for each maximal run of True in trade_mask —
    i.e. one entry per discrete trade rather than per day held."""
    mask = np.asarray(trade_mask, dtype=bool)
    trades = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            trades.append((i, j))
            i = j
        else:
            i += 1
    return trades


def calculate_win_rate(returns: pd.Series, trade_mask: np.ndarray) -> float:
    """Fraction of discrete trades (consecutive runs in trade_mask) with a
    positive compounded return, not fraction of all days (which would count
    every non-trade day as a loss and understate win rate)."""
    trades = group_trades(trade_mask)
    if not trades:
        return float("nan")
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    wins = 0
    for start, end in trades:
        cumulative = np.prod(1.0 + r[start:end]) - 1.0
        if cumulative > 0:
            wins += 1
    return float(wins / len(trades))


def calculate_cagr(returns: pd.Series, dates: Optional[pd.Series] = None) -> float:
    """Annualized compound return. Prefers the actual elapsed calendar span
    (dates.max() - dates.min()) when dates are supplied, since not every
    period has exactly 252 trading days (holidays, partial years, gaps);
    falls back to a 252-trading-day-per-year approximation otherwise."""
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if r.empty:
        return float("nan")
    cum = (1 + r).cumprod()
    final = cum.iloc[-1] if len(cum) else np.nan
    if np.isnan(final) or final <= 0:
        return float("nan")

    years = None
    if dates is not None and len(dates) == len(r):
        dates_dt = pd.to_datetime(pd.Series(dates).to_numpy())
        span_days = (dates_dt.max() - dates_dt.min()).days
        if span_days > 0:
            years = span_days / 365.25
    if years is None:
        years = len(r) / 252.0
    if years <= 0:
        return float("nan")
    return float(final ** (1 / years) - 1)


def run_backtest(
    signal: str,
    prices: pd.DataFrame,
    stop_loss: float = 0.05,
    position_size: float = 0.01,
    cost_bps: float = 5.0,
) -> Dict[str, float]:
    price_series = _coerce_price_series(prices)
    if price_series.empty or len(price_series) < 2:
        return {
            "Sharpe": float("nan"), "Max Drawdown": float("nan"), "Win Rate": float("nan"),
            "Trades": 0, "CAGR": float("nan"),
            "Gross Sharpe": float("nan"), "Net Sharpe": float("nan"),
            "Gross Max Drawdown": float("nan"), "Net Max Drawdown": float("nan"),
            "Gross CAGR": float("nan"), "Net CAGR": float("nan"),
            "Gross CAGR (Full Allocation)": float("nan"), "Net CAGR (Full Allocation)": float("nan"),
        }

    dates = prices["date"] if "date" in prices.columns else None

    # Stop-loss is evaluated against the raw (same-bar) signal, same as before this
    # refactor — only the final entry into calculate_returns is lagged by one bar,
    # via the same lag_mask() primitive validation/robustness.py now uses too.
    raw_mask = compute_signal_mask(prices, signal)
    raw_mask = apply_stop_loss(raw_mask, price_series, stop_loss)
    trade_mask = lag_mask(raw_mask, price_series.index).to_numpy()
    gross_returns = calculate_returns(trade_mask, price_series, position_size)
    net_returns = apply_transaction_costs(gross_returns, trade_mask, cost_bps, position_size)

    gross_sharpe, net_sharpe = calculate_sharpe(gross_returns), calculate_sharpe(net_returns)
    gross_dd, net_dd = calculate_max_drawdown(gross_returns), calculate_max_drawdown(net_returns)
    gross_cagr, net_cagr = calculate_cagr(gross_returns, dates), calculate_cagr(net_returns, dates)

    # Full-allocation CAGR: divide position_size back out before compounding, so
    # it reads as "if 100% of capital were deployed on every trade" rather than
    # the arbitrarily conservative position_size fraction used for the Sharpe/
    # drawdown-scaled numbers above. Dividing net_returns (not gross) by
    # position_size correctly recovers per-day price return minus per-transition
    # cost_bps, since both were scaled by position_size in the first place.
    gross_cagr_full = calculate_cagr(gross_returns / position_size, dates)
    net_cagr_full = calculate_cagr(net_returns / position_size, dates)

    # "Sharpe"/"Max Drawdown"/"CAGR" default to net-of-cost — the economically
    # real numbers — with Gross/Net variants both exposed for side-by-side
    # comparison; existing callers (e.g. main.py -> journal) keep working
    # unchanged, just against more realistic figures.
    return {
        "Sharpe": net_sharpe,
        "Max Drawdown": net_dd,
        "Win Rate": calculate_win_rate(net_returns, trade_mask),
        "Trades": int(np.sum(raw_mask)),
        "CAGR": net_cagr,
        "Gross Sharpe": gross_sharpe,
        "Net Sharpe": net_sharpe,
        "Gross Max Drawdown": gross_dd,
        "Net Max Drawdown": net_dd,
        "Gross CAGR": gross_cagr,
        "Net CAGR": net_cagr,
        "Gross CAGR (Full Allocation)": gross_cagr_full,
        "Net CAGR (Full Allocation)": net_cagr_full,
        "cost_bps": cost_bps,
    }


if __name__ == "__main__":
    sample = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "ret_5d_z": np.random.randn(100),
        "rsi": np.random.randint(0, 100, size=100),
    })
    results = run_backtest("ret_5d_z > 0.5 AND rsi < 30", sample)
    print(results)
