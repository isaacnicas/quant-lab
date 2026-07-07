import yfinance as yf
import duckdb
import pandas as pd
from typing import List
from datetime import datetime


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


if __name__ == "__main__":
    fetch_and_cache(["SPY", "AAPL"], datetime(2023, 1, 1), datetime(2023, 12, 31), "data/store.duckdb")
    fetch_and_cache(["SPY", "AAPL"], datetime(2023, 1, 1), datetime(2023, 12, 31), "data/store.duckdb")
    print(load_prices(["SPY"], datetime(2023, 1, 1), datetime(2023, 1, 10), "data/store.duckdb"))
