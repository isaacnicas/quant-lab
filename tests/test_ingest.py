import pytest
from datetime import datetime
from data.ingest import fetch_and_cache, load_prices


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_store.duckdb")


def test_fetch_and_cache(db_path):
    tickers = ["SPY", "AAPL"]
    start, end = datetime(2023, 1, 1), datetime(2023, 1, 31)
    fetch_and_cache(tickers, start, end, db_path)
    df = load_prices(tickers, start, end, db_path)
    assert not df.empty
    assert set(df.columns) == {"ticker", "date", "open", "high", "low", "close", "volume"}
    assert df["volume"].dtype == "int64"
    assert df["close"].dtype == "float64"


def test_upsert_no_duplicates(db_path):
    tickers = ["SPY"]
    start, end = datetime(2023, 1, 1), datetime(2023, 1, 10)
    fetch_and_cache(tickers, start, end, db_path)
    fetch_and_cache(tickers, start, end, db_path)
    df = load_prices(tickers, start, end, db_path)
    assert df.shape[0] > 0
    assert df.duplicated(subset=["ticker", "date"]).sum() == 0


def test_single_ticker(db_path):
    tickers = ["AAPL"]
    start, end = datetime(2023, 1, 1), datetime(2023, 1, 10)
    fetch_and_cache(tickers, start, end, db_path)
    df = load_prices(tickers, start, end, db_path)
    assert df.shape[0] > 0
    assert (df["ticker"] == "AAPL").all()


def test_multiple_tickers_both_present(db_path):
    tickers = ["SPY", "AAPL"]
    start, end = datetime(2023, 1, 1), datetime(2023, 1, 10)
    fetch_and_cache(tickers, start, end, db_path)
    df = load_prices(tickers, start, end, db_path)
    assert set(df["ticker"].unique()) == {"SPY", "AAPL"}
