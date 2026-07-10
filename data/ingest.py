import yfinance as yf
import duckdb
import pandas as pd
from typing import List
from datetime import datetime


class HoldoutViolationError(Exception):
    """Raised when a discovery-path read would expose data past
    IN_SAMPLE_END. Discovery (signal generation, Monte Carlo, FDR, backtest,
    log_experiment) must never see the 2024-2025 holdout window -- it is
    this project's only clean holdout, and once discovery touches it, it is
    permanently spent. There is no un-seeing it."""


def _reshape_to_long(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Reshape yfinance output into long format: one row per (ticker, date).
    Uses .stack() on the actual MultiIndex structure instead of guessing/joining
    column name strings — that approach breaks whenever yfinance changes its
    column shape between versions (confirmed twice in this project already).
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = pd.MultiIndex.from_product([tickers, raw.columns])

    long_df = (
        raw.stack(level=0, future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )
    long_df.columns = [str(c).lower() for c in long_df.columns]

    if "adj close" in long_df.columns:
        long_df = long_df.drop(columns=["close"], errors="ignore")
        long_df = long_df.rename(columns={"adj close": "close"})

    long_df = long_df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    long_df = long_df.dropna(subset=["open", "high", "low", "close", "volume"])
    long_df["volume"] = long_df["volume"].astype("int64")
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.date
    # yfinance has silently changed the float precision of OHLC columns across
    # versions before (float64 -> float32) and broke this pipeline's schema
    # assumptions each time. Cast explicitly so the schema is stable regardless
    # of what a given yfinance version happens to return.
    long_df[["open", "high", "low", "close"]] = long_df[["open", "high", "low", "close"]].astype("float64")
    return long_df


def fetch_and_cache(tickers: List[str], start: datetime, end: datetime, duckdb_path: str) -> None:
    conn = duckdb.connect(duckdb_path)
    # OHLC columns are DOUBLE (not FLOAT/float32): DuckDB truncates inserted
    # values to the column's declared width regardless of the source pandas
    # dtype, so a FLOAT column would silently re-introduce float32 precision
    # loss even after casting long_df to float64 above.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume BIGINT, PRIMARY KEY (ticker, date)
        )
    """)
    raw = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
    if raw.empty:
        print(f"No data returned for {tickers} in range {start}-{end}")
        conn.close()
        return
    long_df = _reshape_to_long(raw, tickers)
    conn.execute("INSERT OR REPLACE INTO prices SELECT ticker, date, open, high, low, close, volume FROM long_df")
    conn.close()


def load_prices(tickers: List[str], start: datetime, end: datetime, duckdb_path: str) -> pd.DataFrame:
    conn = duckdb.connect(duckdb_path)
    placeholders = ",".join(["?"] * len(tickers))
    query = f"""
        SELECT * FROM prices
        WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ?
        ORDER BY ticker, date
    """
    params = [*tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]
    df = conn.execute(query, params).fetchdf()
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def assert_discovery_in_sample(prices: pd.DataFrame, boundary: datetime) -> None:
    """Structural backstop for the discovery path: raise HoldoutViolationError
    if any row in `prices` is dated after `boundary` (normally
    config.IN_SAMPLE_END). Call this immediately after every load_prices()
    call used for discovery, regardless of what bound was requested at the
    call site -- this is the check that makes a wrong call-site argument
    impossible to violate silently, the same way signals/apply.py's shared
    lag_mask() made double-lagging impossible rather than just documented
    against. fetch_and_cache() is NOT guarded -- caching data through today
    is harmless and useful for later evaluation; only the discovery READ is
    bounded."""
    if prices is None or prices.empty:
        return
    dates = pd.to_datetime(prices["date"])
    max_date = dates.max()
    boundary_ts = pd.Timestamp(boundary)
    if max_date > boundary_ts:
        raise HoldoutViolationError(
            f"Discovery attempted to read data through {max_date.date()}, past "
            f"the IN_SAMPLE_END boundary of {boundary_ts.date()}. This would "
            f"burn the project's only clean 2024-2025 holdout window -- "
            f"discovery (signal generation, Monte Carlo, FDR, backtest, "
            f"log_experiment) must never see post-holdout data. Fix the "
            f"load_prices() call on the discovery path to bound by "
            f"config.IN_SAMPLE_END, not config.end_date."
        )


if __name__ == "__main__":
    fetch_and_cache(["SPY", "AAPL"], datetime(2023, 1, 1), datetime(2023, 12, 31), "data/store.duckdb")
    fetch_and_cache(["SPY", "AAPL"], datetime(2023, 1, 1), datetime(2023, 12, 31), "data/store.duckdb")
    print(load_prices(["SPY"], datetime(2023, 1, 1), datetime(2023, 1, 10), "data/store.duckdb"))
