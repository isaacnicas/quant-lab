"""
Standalone validation script for one specific surviving signal — QQQ's
"ret_1d_z < 1.0 AND rsi < 30" — broken out by calendar year. Not part of the
main pipeline loop (main.py); run directly with:

    PYTHONPATH=. python validate_qqq_signal.py
"""
import numpy as np
import pandas as pd

from config import Config
from data.ingest import load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.apply import compute_signal_mask, lag_mask
from backtest.evaluate import (
    _coerce_price_series,
    apply_stop_loss,
    apply_transaction_costs,
    calculate_cagr,
    calculate_returns,
    calculate_sharpe,
    group_trades,
)

TICKER = "QQQ"
SIGNAL = "ret_1d_z < 1.0 AND rsi < 30"
STOP_LOSS = 0.05
POSITION_SIZE = 0.01
COST_BPS = 5.0


def main() -> None:
    cfg = Config()
    prices = load_prices([TICKER], cfg.start_date, cfg.end_date, cfg.duckdb_path)
    featured = build_features(prices)
    featured["regime"] = label_regime(featured)

    # Build the full-history trade mask and net-of-cost returns ONCE over the
    # continuous series (not per-year) so rolling-window features stay correct
    # and trades/costs that straddle a year boundary aren't double-counted or lost.
    price_series = _coerce_price_series(featured)
    raw_mask = compute_signal_mask(featured, SIGNAL)
    raw_mask = apply_stop_loss(raw_mask, price_series, STOP_LOSS)
    trade_mask = lag_mask(raw_mask, price_series.index).to_numpy()
    gross_returns = calculate_returns(trade_mask, price_series, POSITION_SIZE)
    net_returns = apply_transaction_costs(gross_returns, trade_mask, COST_BPS, POSITION_SIZE)
    net_arr = net_returns.to_numpy()

    dates = pd.to_datetime(featured["date"])
    years = sorted(dates.dt.year.unique())

    # Win rate is trade-level: group into discrete trades on the FULL series
    # first (so a trade straddling Dec 31 -> Jan 1 is judged as one trade, not
    # split into two misleading fragments), then bucket each trade's outcome by
    # its entry year.
    trades_by_year = {}
    for start, end in group_trades(trade_mask):
        cumulative = float(np.prod(1.0 + net_arr[start:end]) - 1.0)
        entry_year = int(dates.iloc[start].year)
        trades_by_year.setdefault(entry_year, []).append(cumulative > 0)

    rows = []
    for year in years:
        year_day_mask = (dates.dt.year == year).to_numpy()
        year_net = pd.Series(net_arr[year_day_mask])
        year_dates = dates[year_day_mask].reset_index(drop=True)
        wins = trades_by_year.get(year, [])
        rows.append({
            "Year": year,
            "Trades": len(wins),
            "Days In Position": int(trade_mask[year_day_mask].sum()),
            "Sharpe": calculate_sharpe(year_net),
            "Win Rate": (sum(wins) / len(wins)) if wins else float("nan"),
            "Net CAGR": calculate_cagr(year_net, year_dates),
            "Net CAGR (Full Alloc)": calculate_cagr(year_net / POSITION_SIZE, year_dates),
        })

    table = pd.DataFrame(rows)
    print(f"Signal: {SIGNAL!r} on {TICKER}, net of {COST_BPS}bps transaction costs\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
