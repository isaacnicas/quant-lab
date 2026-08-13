"""
Genuine holdout test for QQQ's "ret_1d_z < 1.0 AND rsi < 30" on 2024-2025 data
that no other signal in the 2,000-experiment sweep has ever touched. Not part
of the main pipeline loop (main.py); run directly with:

    PYTHONPATH=. python validate_qqq_signal_holdout.py

Fetches with a lookback buffer before 2024-01-01 so rolling-window features
(vol_60d, rsi, etc.) are fully warmed up before the reported holdout window
starts, then restricts the reported metrics to 2024-01-01 through 2025-12-31.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from data.ingest import fetch_and_cache, load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.apply import compute_signal_mask, lag_mask
from backtest.evaluate import (
    _coerce_price_series,
    apply_stop_loss,
    apply_transaction_costs,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe,
    group_trades,
)

TICKER = "QQQ"
SIGNAL = "ret_1d_z < 1.0 AND rsi < 30"
STOP_LOSS = 0.05
POSITION_SIZE = 0.01
COST_BPS = 5.0
DUCKDB_PATH = "data/store.duckdb"

FETCH_START = datetime(2023, 9, 1)   # lookback buffer for rolling features
HOLDOUT_START = datetime(2024, 1, 1)
HOLDOUT_END = datetime(2025, 12, 31)


def main() -> None:
    fetch_and_cache([TICKER], FETCH_START, HOLDOUT_END, DUCKDB_PATH)
    prices = load_prices([TICKER], FETCH_START, HOLDOUT_END, DUCKDB_PATH)

    featured = build_features(prices)
    featured["regime"] = label_regime(featured)

    price_series = _coerce_price_series(featured)
    raw_mask = compute_signal_mask(featured, SIGNAL)
    raw_mask = apply_stop_loss(raw_mask, price_series, STOP_LOSS)
    trade_mask = lag_mask(raw_mask, price_series.index).to_numpy()
    gross_returns = calculate_returns(trade_mask, price_series, POSITION_SIZE)
    net_returns = apply_transaction_costs(gross_returns, trade_mask, COST_BPS, POSITION_SIZE)

    dates = pd.to_datetime(featured["date"])
    holdout_mask = (dates >= HOLDOUT_START).to_numpy() & (dates <= HOLDOUT_END).to_numpy()
    net_arr = net_returns.to_numpy()

    holdout_returns = pd.Series(net_arr[holdout_mask])
    holdout_dates = dates[holdout_mask].reset_index(drop=True)
    holdout_days_in_position = int(trade_mask[holdout_mask].sum())

    # Trade-level win rate + discrete trade count: group into discrete trades on
    # the full (lookback + holdout) series so a trade isn't fragmented at the
    # 2024-01-01 boundary, then keep only trades that actually entered within
    # the holdout window.
    wins = []
    for start, end in group_trades(trade_mask):
        entry_date = dates.iloc[start]
        if HOLDOUT_START <= entry_date <= HOLDOUT_END:
            cumulative = float(np.prod(1.0 + net_arr[start:end]) - 1.0)
            wins.append(cumulative > 0)
    win_rate = (sum(wins) / len(wins)) if wins else float("nan")
    holdout_trades = len(wins)

    print(f"Signal: {SIGNAL!r} on {TICKER}, holdout {HOLDOUT_START.date()} - {HOLDOUT_END.date()}, "
          f"net of {COST_BPS}bps transaction costs\n")
    print(f"{'Trades (discrete)':20s}: {holdout_trades}")
    print(f"{'Days In Position':20s}: {holdout_days_in_position}")
    print(f"{'Sharpe':20s}: {calculate_sharpe(holdout_returns):.4f}")
    print(f"{'Win Rate':20s}: {win_rate:.4f}")
    print(f"{'Max Drawdown (Full Alloc)':20s}: {calculate_max_drawdown(holdout_returns / POSITION_SIZE):.4f}")
    print(f"{'Net CAGR':20s}: {calculate_cagr(holdout_returns, holdout_dates):.8f}")
    print(f"{'Net CAGR (Full Alloc)':20s}: {calculate_cagr(holdout_returns / POSITION_SIZE, holdout_dates):.8f}")


if __name__ == "__main__":
    main()
