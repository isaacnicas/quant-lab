"""
Offline tests for the PEAD data ingest layer -- synthetic DataFrames only,
no SimFin API calls, no real data files. Must pass with no SIMFIN_API_KEY set.
"""
import duckdb
import numpy as np
import pandas as pd
import pytest

from data.ingest import HoldoutViolationError
from data.ingest_pead import (
    DataIntegrityError,
    PEADIngestor,
    _assert_publish_after_report,
    assert_holdout_integrity,
)
from features.engineer_pead import compute_forward_returns, compute_sue_srw


def _make_income_13q(ticker="AAA", pub_lag_days=45):
    """13 consecutive quarters for one ticker -- enough for one event (the
    last row) to have exactly 8 PIT-eligible prior surprises."""
    report_dates = pd.date_range("2019-03-31", periods=13, freq="QE")
    publish_dates = report_dates + pd.Timedelta(days=pub_lag_days)
    # Explicit, all-distinct values (not a formula) -- a periodic formula like
    # 1.0 + 0.1*i + 0.05*(i%3) collides (eps[11] == eps[12]), which breaks
    # tests that identify a specific row by its eps_t value.
    eps = [1.00, 1.12, 0.98, 1.25, 1.30, 1.08, 1.42, 1.15, 1.55, 1.20, 1.65, 1.33, 1.78]
    return pd.DataFrame({
        "ticker": ticker,
        "report_date": report_dates,
        "publish_date": publish_dates,
        "eps_basic": eps,
    })


def test_publish_date_integrity():
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "report_date": [pd.Timestamp("2022-12-31")],
        "publish_date": [pd.Timestamp("2022-12-01")],  # before report_date -- impossible
        "eps_basic": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        _assert_publish_after_report(df)


def test_holdout_guard():
    df = pd.DataFrame({
        "ticker": ["AAA"],
        "publish_date": [pd.Timestamp("2024-06-01")],  # past IN_SAMPLE_END (2023-12-31)
        "eps_basic": [1.0],
    })
    with pytest.raises(HoldoutViolationError):
        assert_holdout_integrity(df, pit_col="publish_date")


def test_sue_srw_pit_discipline():
    income_df = _make_income_13q()
    # Row 8 (positionally a valid prior surprise for row 12) is made NOT
    # PIT-available at row 12's announcement time -- its publish_date is
    # moved up to equal row 12's publish_date instead of preceding it.
    event_publish_date = income_df.loc[12, "publish_date"]
    income_df.loc[8, "publish_date"] = event_publish_date

    result = compute_sue_srw(income_df)
    # Row 8's own event now shares the same (moved) publish_date, so filter
    # on eps_t (unique to row 12's event) rather than publish_date alone.
    row12 = result[result["eps_t"] == income_df.loc[12, "eps_basic"]]
    assert len(row12) == 1
    assert row12["n_quarters_used"].iloc[0] == 7  # row 8 excluded -> only 7 valid PIT priors
    assert pd.isna(row12["sue_srw"].iloc[0])


def test_sue_srw_insufficient_history():
    income_df = _make_income_13q().iloc[:6].reset_index(drop=True)  # far fewer than needed
    result = compute_sue_srw(income_df)
    assert result["sue_srw"].isna().all()
    assert (result["n_quarters_used"] < 8).all()


def test_sue_srw_n_quarters_audit():
    income_df = _make_income_13q()
    result = compute_sue_srw(income_df)
    last_publish_date = income_df.loc[12, "publish_date"]
    row12 = result[result["publish_date"] == last_publish_date]
    assert len(row12) == 1
    assert row12["n_quarters_used"].iloc[0] == 8
    assert pd.notna(row12["sue_srw"].iloc[0])


def _make_prices(ticker, dates, opens, closes):
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": opens,
        "high": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
        "low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [1_000_000] * len(dates),
        "shares_outstanding": [10_000_000] * len(dates),
    })


def test_forward_returns_t1_entry():
    dates = pd.bdate_range("2023-01-02", periods=15)
    opens = [100.0 + i for i in range(15)]
    closes = [100.5 + i for i in range(15)]  # deliberately distinct from opens
    prices_df = _make_prices("AAA", dates, opens, closes)

    publish_date = dates[3]  # a weekday, itself a trading day
    events_df = pd.DataFrame({"ticker": ["AAA"], "publish_date": [publish_date]})

    result = compute_forward_returns(prices_df, events_df, hold_days=5)
    assert len(result) == 1
    expected_entry_date = dates[4]  # T+1 from publish_date
    assert result["entry_date"].iloc[0] == expected_entry_date
    assert result["entry_price"].iloc[0] == pytest.approx(opens[4])
    assert result["entry_price"].iloc[0] != pytest.approx(closes[3])
    assert result["entry_price"].iloc[0] != pytest.approx(closes[4])


def test_forward_returns_stop_loss():
    dates = pd.bdate_range("2023-01-02", periods=30)
    opens = [100.0] * 30
    closes = [100.0] * 30
    prices_df = _make_prices("AAA", dates, opens, closes)

    publish_date = dates[0]
    events_df = pd.DataFrame({"ticker": ["AAA"], "publish_date": [publish_date]})
    # entry_idx = 1 (T+1). Day 5 after entry (index 6) closes -12% from entry_price (100).
    prices_df.loc[prices_df["date"] == dates[6], "close"] = 88.0

    result = compute_forward_returns(prices_df, events_df, hold_days=20, stop_loss_pct=0.10)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["stopped"] == True
    assert row["exit_date"] == dates[6]
    assert row["hold_days_actual"] == 5
    assert row["days_to_stop"] == 5


def test_forward_returns_weekend_publish():
    # 2023-01-07 is a Saturday.
    saturday = pd.Timestamp("2023-01-07")
    assert saturday.day_name() == "Saturday"
    dates = pd.bdate_range("2023-01-02", periods=10)  # Mon 2023-01-02 .. Fri 2023-01-13
    opens = [100.0 + i for i in range(10)]
    closes = [100.5 + i for i in range(10)]
    prices_df = _make_prices("AAA", dates, opens, closes)

    events_df = pd.DataFrame({"ticker": ["AAA"], "publish_date": [saturday]})
    result = compute_forward_returns(prices_df, events_df, hold_days=5)
    assert len(result) == 1
    # Effective announcement date (first trading day at/after Saturday) is
    # Monday 2023-01-09; entry is T+1 from THAT (Tuesday 2023-01-10), per the
    # pre-registration's explicit "add one more day" rule for weekend/holiday
    # publish dates -- not the naive "next trading day is entry" reading
    # (which would give Monday). Documented as a judgment call in the PR
    # writeup since the task's one-line test description said "Monday".
    monday = pd.Timestamp("2023-01-09")
    tuesday = pd.Timestamp("2023-01-10")
    assert result["entry_date"].iloc[0] == tuesday
    assert result["entry_date"].iloc[0] != monday
    assert result["entry_date"].iloc[0].day_name() != "Sunday"
    assert result["entry_date"].iloc[0].day_name() != "Saturday"


def test_named_column_inserts(tmp_path):
    # DuckDB's ":memory:" databases are isolated per-connection -- a second
    # connect(":memory:") call would open a different, empty database, so a
    # temp FILE path (pytest's tmp_path fixture) is used here to genuinely
    # read back what was written, while still being fully throwaway/isolated.
    db_path = str(tmp_path / "pead_test.duckdb")

    income_df = pd.DataFrame({
        "ticker": ["AAA"], "report_date": [pd.Timestamp("2022-12-31")],
        "publish_date": [pd.Timestamp("2023-02-15")], "eps_basic": [1.23],
    })
    prices_df = pd.DataFrame({
        "ticker": ["AAA"], "date": [pd.Timestamp("2023-02-16")],
        "open": [50.0], "high": [51.0], "low": [49.0], "close": [50.5],
        "volume": [1_000_000], "shares_outstanding": [10_000_000.0],
    })
    universe_df = pd.DataFrame({
        "ticker": ["AAA"], "publish_date": [pd.Timestamp("2023-02-15")],
        "market_cap_at_announce": [505_000_000.0], "passes_all_filters": [True],
        "filter_reason": [None], "sic_filter_applied": [False],
    })

    PEADIngestor().store_pead_data(income_df, prices_df, universe_df, db_path=db_path)

    conn = duckdb.connect(db_path)
    row = conn.execute(
        "SELECT ticker, publish_date, eps_basic FROM pead_income"
    ).fetchone()
    assert row[0] == "AAA"
    assert row[1] == pd.Timestamp("2023-02-15").date()
    assert row[2] == pytest.approx(1.23)

    urow = conn.execute(
        "SELECT ticker, publish_date, passes_all_filters FROM pead_universe"
    ).fetchone()
    assert urow[0] == "AAA"
    assert urow[1] == pd.Timestamp("2023-02-15").date()
    assert urow[2] is True
    conn.close()
