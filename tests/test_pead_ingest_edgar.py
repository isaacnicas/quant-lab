"""
Offline tests for the edgar PEAD data ingest layer. Core-logic tests use
synthetic DataFrames only, no SEC EDGAR calls, no EDGAR_IDENTITY required --
mirrors tests/test_pead_ingest.py's structure and offline-only philosophy.

The universe-filter-correctness tests against the 5 real sample tickers
from Step 1 read from the already-populated data/edgar_cache/ parquet
cache (offline -- no live network call) and are skipped if that cache is
not present, so a fresh checkout without the cache doesn't force a live
SEC EDGAR fetch during a normal test run.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from data.ingest import HoldoutViolationError
from data.ingest_pead_edgar import (
    EDGAR_CACHE_DIR,
    QUARTER_DURATION_MAX_DAYS,
    QUARTER_DURATION_MIN_DAYS,
    EdgarPEADIngestor,
    _select_quarterly_eps,
    migrate_source_column,
)
from data.ingest_pead import assert_holdout_integrity

SAMPLE_TICKERS = ["ACLS", "ADUS", "ADMA", "ABUS", "ACRS"]


# ── PIT correctness: filing date used, not period-end date ──────────────────

def test_select_quarterly_eps_uses_filing_date_not_period_end():
    """publish_date in the output must come from the fact's filing_date
    (PIT-correct), never from report_date (period_end) -- the two are
    deliberately set to different, unambiguous values here so a bug that
    swapped them would fail this test."""
    fact_df = pd.DataFrame([
        {
            "ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
            "report_date": pd.Timestamp("2022-12-31"),  # period end
            "publish_date": pd.Timestamp("2023-02-15"),  # filing acceptance date
            "eps_value": 1.23, "duration_days": 90,
        },
    ])
    result = _select_quarterly_eps(fact_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["publish_date"] == pd.Timestamp("2023-02-15")
    assert row["report_date"] == pd.Timestamp("2022-12-31")
    assert row["publish_date"] != row["report_date"]


def test_select_quarterly_eps_discards_ytd_cumulative_duplicates():
    """Confirmed empirically against real AAPL XBRL data: the same
    fiscal_period label can carry both a ~90-day quarterly fact and a
    longer YTD-cumulative fact (e.g. 272 days). Only the ~90-day fact
    should survive -- this is exactly the AAPL Q3 2025 case (quarterly
    EPS=1.57 vs YTD EPS=5.64), reproduced synthetically here."""
    fact_df = pd.DataFrame([
        {
            "ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
            "report_date": pd.Timestamp("2025-06-28"), "publish_date": pd.Timestamp("2025-08-01"),
            "eps_value": 5.64, "duration_days": 272,  # YTD-cumulative -- must be discarded
        },
        {
            "ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
            "report_date": pd.Timestamp("2025-06-28"), "publish_date": pd.Timestamp("2025-08-01"),
            "eps_value": 1.57, "duration_days": 90,  # quarterly -- must survive
        },
    ])
    result = _select_quarterly_eps(fact_df)
    assert len(result) == 1
    assert result.iloc[0]["eps_basic"] == pytest.approx(1.57)


def test_select_quarterly_eps_boundary_durations():
    fact_df = pd.DataFrame([
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
         "report_date": pd.Timestamp("2022-03-31"), "publish_date": pd.Timestamp("2022-05-01"),
         "eps_value": 1.0, "duration_days": QUARTER_DURATION_MIN_DAYS - 1},  # too short -- excluded
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
         "report_date": pd.Timestamp("2022-06-30"), "publish_date": pd.Timestamp("2022-08-01"),
         "eps_value": 2.0, "duration_days": QUARTER_DURATION_MAX_DAYS + 1},  # too long -- excluded
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
         "report_date": pd.Timestamp("2022-09-30"), "publish_date": pd.Timestamp("2022-11-01"),
         "eps_value": 3.0, "duration_days": QUARTER_DURATION_MIN_DAYS},  # in range -- included
    ])
    result = _select_quarterly_eps(fact_df)
    assert len(result) == 1
    assert result.iloc[0]["eps_basic"] == pytest.approx(3.0)


def test_select_quarterly_eps_diluted_fallback():
    """A report_date with only a diluted fact (no basic at all) should
    still be included, using diluted as the fallback -- matching
    ingest_pead.py's SimFin fallback convention."""
    fact_df = pd.DataFrame([
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareDiluted",
         "report_date": pd.Timestamp("2022-03-31"), "publish_date": pd.Timestamp("2022-05-01"),
         "eps_value": 0.95, "duration_days": 90},
    ])
    result = _select_quarterly_eps(fact_df)
    assert len(result) == 1
    assert result.iloc[0]["eps_basic"] == pytest.approx(0.95)


def test_select_quarterly_eps_basic_preferred_over_diluted():
    fact_df = pd.DataFrame([
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareBasic",
         "report_date": pd.Timestamp("2022-03-31"), "publish_date": pd.Timestamp("2022-05-01"),
         "eps_value": 1.10, "duration_days": 90},
        {"ticker": "AAA", "concept": "us-gaap:EarningsPerShareDiluted",
         "report_date": pd.Timestamp("2022-03-31"), "publish_date": pd.Timestamp("2022-05-01"),
         "eps_value": 1.05, "duration_days": 90},
    ])
    result = _select_quarterly_eps(fact_df)
    assert len(result) == 1
    assert result.iloc[0]["eps_basic"] == pytest.approx(1.10)


def test_holdout_guard_applies_to_edgar_publish_dates():
    """assert_holdout_integrity (shared with the SimFin path) must reject
    edgar-sourced rows with publish_date past IN_SAMPLE_END exactly the
    same way it rejects SimFin rows -- the PEAD holdout boundary is
    source-agnostic."""
    df = pd.DataFrame({
        "ticker": ["AAA"], "publish_date": [pd.Timestamp("2024-06-01")], "eps_basic": [1.0],
    })
    with pytest.raises(HoldoutViolationError):
        assert_holdout_integrity(df, pit_col="publish_date")


# ── No-lookahead: perturbation test (mirrors test_sue_srw_pit_discipline) ───

def test_universe_filter_pit_discipline_no_lookahead():
    """Perturbation test: a prior quarter that is positionally earlier but
    whose publish_date is moved to NOT precede the event's publish_date
    must not count toward the >= MIN_PRIOR_QUARTERS filter -- position in
    the report_date-sorted frame is not sufficient on its own, mirroring
    ingest_pead.py's filter_universe PIT discipline exactly.

    Market-cap lookup is keyed purely on (ticker, publish_date) against
    the ticker's unified price timeline -- independent of which income
    row triggered it -- so once row 0's publish_date collides with row
    8's, both events necessarily resolve to the same market cap and are
    no longer individually distinguishable in the output. The robust
    assertion is therefore: baseline has exactly one PASSING event at
    publish_dates[-1]; after the perturbation, no event at that date
    passes (row 8's own eligibility flipped, and row 0's is a permanent
    "insufficient_eps_history" case either way, positionally first).
    """
    from data.ingest_pead import MIN_PRIOR_QUARTERS

    report_dates = pd.date_range("2019-03-31", periods=MIN_PRIOR_QUARTERS + 1, freq="QE")
    publish_dates = report_dates + pd.Timedelta(days=45)
    income_df = pd.DataFrame({
        "ticker": "AAA", "report_date": report_dates,
        "publish_date": publish_dates, "eps_basic": np.linspace(1.0, 2.0, len(report_dates)),
    })

    # Baseline: exactly MIN_PRIOR_QUARTERS prior, PIT-eligible -> passes_c True.
    prices_df = pd.DataFrame({
        "ticker": ["AAA"] * len(report_dates),
        "date": publish_dates - pd.Timedelta(days=1),
        "close": [50.0] * len(report_dates),
        "shares_outstanding": [20_000_000.0] * len(report_dates),  # $1B mkt cap -- in range
    })
    baseline = EdgarPEADIngestor().filter_universe_edgar(income_df, prices_df)
    last_event = baseline[baseline["publish_date"] == publish_dates[-1]]
    assert len(last_event) == 1
    assert bool(last_event["passes_all_filters"].iloc[0]) is True
    assert pd.isna(last_event["filter_reason"].iloc[0])

    # Perturbation: move the earliest prior quarter's publish_date to equal
    # the last event's publish_date (no longer strictly before it) -- this
    # must drop it from the PIT-eligible count, flipping passes_c to False
    # for row 8.
    perturbed_income = income_df.copy()
    perturbed_income.loc[0, "publish_date"] = publish_dates[-1]

    perturbed = EdgarPEADIngestor().filter_universe_edgar(perturbed_income, prices_df)
    at_shared_date = perturbed[perturbed["publish_date"] == publish_dates[-1]]
    assert len(at_shared_date) >= 1
    assert not at_shared_date["passes_all_filters"].any(), (
        "row 8 should have lost one PIT-eligible prior quarter (down to "
        f"{MIN_PRIOR_QUARTERS - 1}) once row 0's publish_date no longer strictly "
        "precedes it -- position in the sorted frame alone must not count as PIT-eligible, "
        "so no event at this date should still pass all filters"
    )
    assert (at_shared_date["filter_reason"] == "insufficient_eps_history").all()


# ── Universe filter correctness on the 5 sample tickers from Step 1 ────────

_CACHE_PRESENT = all(
    os.path.exists(os.path.join(EDGAR_CACHE_DIR, f"{t}_facts.parquet")) for t in SAMPLE_TICKERS
)


@pytest.mark.skipif(not _CACHE_PRESENT, reason="edgar_cache/ not populated for sample tickers -- run data/ingest_pead_edgar.py first")
def test_sample_tickers_income_loads_from_cache_offline():
    """Reads the 5 sample tickers' already-cached facts (no live SEC EDGAR
    call) and confirms every row has a real ticker, a plausible EPS value,
    and publish_date strictly after report_date (PIT sanity)."""
    ingestor = EdgarPEADIngestor()
    for ticker in SAMPLE_TICKERS:
        fact_df = pd.read_parquet(os.path.join(EDGAR_CACHE_DIR, f"{ticker}_facts.parquet"))
        assert not fact_df.empty, f"{ticker}: cache is empty"
        selected = _select_quarterly_eps(fact_df)
        assert not selected.empty, f"{ticker}: no quarterly EPS facts selected"
        assert (selected["publish_date"] > selected["report_date"]).all(), (
            f"{ticker}: found publish_date <= report_date -- impossible PIT violation"
        )


@pytest.mark.skipif(not _CACHE_PRESENT, reason="edgar_cache/ not populated for sample tickers -- run data/ingest_pead_edgar.py first")
def test_sample_tickers_earliest_publish_date_before_2020():
    """The whole motivation for adding edgar as a source is extending
    usable history back before SimFin's ~2020 free-tier floor -- confirm
    at least one sample ticker actually has real filing history that far
    back (does not assert ALL do, since IPO dates vary)."""
    earliest_dates = []
    for ticker in SAMPLE_TICKERS:
        fact_df = pd.read_parquet(os.path.join(EDGAR_CACHE_DIR, f"{ticker}_facts.parquet"))
        selected = _select_quarterly_eps(fact_df)
        if not selected.empty:
            earliest_dates.append(selected["publish_date"].min())
    assert any(d < pd.Timestamp("2020-01-01") for d in earliest_dates), (
        f"expected at least one sample ticker with publish_date before 2020-01-01, "
        f"got earliest dates: {earliest_dates}"
    )


# ── Schema compatibility with existing pead_store.duckdb tables ────────────

def test_migrate_source_column_preserves_existing_rows_and_is_idempotent(tmp_path):
    db_path = str(tmp_path / "pead_test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE pead_income (
            ticker VARCHAR, report_date DATE, publish_date DATE, eps_basic DOUBLE,
            PRIMARY KEY (ticker, report_date)
        )
    """)
    conn.execute("""
        CREATE TABLE pead_universe (
            ticker VARCHAR, publish_date DATE, market_cap_at_announce DOUBLE,
            passes_all_filters BOOLEAN, filter_reason VARCHAR, sic_filter_applied BOOLEAN,
            PRIMARY KEY (ticker, publish_date)
        )
    """)
    conn.execute("INSERT INTO pead_income VALUES ('AAA', '2022-12-31', '2023-02-15', 1.23)")
    conn.execute("INSERT INTO pead_universe VALUES ('AAA', '2023-02-15', 5.0e8, True, NULL, False)")
    conn.close()

    migrate_source_column(db_path=db_path)

    conn = duckdb.connect(db_path, read_only=True)
    row = conn.execute("SELECT ticker, report_date, publish_date, eps_basic, source FROM pead_income").fetchone()
    assert row == ("AAA", pd.Timestamp("2022-12-31").date(), pd.Timestamp("2023-02-15").date(), pytest.approx(1.23), "simfin")
    conn.close()

    # Idempotent: running again on an already-migrated db is a no-op, not an error.
    migrate_source_column(db_path=db_path)
    conn = duckdb.connect(db_path, read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM pead_income").fetchone()[0] == 1
    conn.close()


def test_simfin_and_edgar_rows_coexist_same_ticker_quarter(tmp_path):
    """The whole point of the source column + expanded PK: a SimFin row
    and an edgar row for the SAME (ticker, report_date) must both survive
    -- the second INSERT OR REPLACE must not silently overwrite the first,
    which is exactly what the old (ticker, report_date)-only PK would do."""
    db_path = str(tmp_path / "pead_test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE pead_income (
            ticker VARCHAR, report_date DATE, publish_date DATE, eps_basic DOUBLE,
            PRIMARY KEY (ticker, report_date)
        )
    """)
    conn.execute("""
        CREATE TABLE pead_universe (
            ticker VARCHAR, publish_date DATE, market_cap_at_announce DOUBLE,
            passes_all_filters BOOLEAN, filter_reason VARCHAR, sic_filter_applied BOOLEAN,
            PRIMARY KEY (ticker, publish_date)
        )
    """)
    conn.execute("INSERT INTO pead_income VALUES ('AAA', '2022-12-31', '2023-02-15', 1.23)")
    conn.close()

    migrate_source_column(db_path=db_path)

    edgar_income = pd.DataFrame({
        "ticker": ["AAA"], "report_date": [pd.Timestamp("2022-12-31")],
        "publish_date": [pd.Timestamp("2023-02-14")],  # edgar's filing date differs slightly
        "eps_basic": [1.19], "source": ["edgar"],
    })
    edgar_universe = pd.DataFrame({
        "ticker": ["AAA"], "publish_date": [pd.Timestamp("2023-02-14")],
        "market_cap_at_announce": [4.9e8], "passes_all_filters": [True],
        "filter_reason": [None], "sic_filter_applied": [False], "source": ["edgar"],
    })
    EdgarPEADIngestor().store_pead_data_edgar(edgar_income, edgar_universe, db_path=db_path)

    conn = duckdb.connect(db_path, read_only=True)
    rows = conn.execute(
        "SELECT source, eps_basic FROM pead_income WHERE ticker='AAA' AND report_date='2022-12-31' ORDER BY source"
    ).fetchall()
    conn.close()
    assert rows == [("edgar", pytest.approx(1.19)), ("simfin", pytest.approx(1.23))]


def test_store_pead_data_edgar_requires_migration_first(tmp_path):
    db_path = str(tmp_path / "pead_test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE pead_income (
            ticker VARCHAR, report_date DATE, publish_date DATE, eps_basic DOUBLE,
            PRIMARY KEY (ticker, report_date)
        )
    """)
    conn.close()

    edgar_income = pd.DataFrame({
        "ticker": ["AAA"], "report_date": [pd.Timestamp("2022-12-31")],
        "publish_date": [pd.Timestamp("2023-02-14")], "eps_basic": [1.19], "source": ["edgar"],
    })
    with pytest.raises(RuntimeError, match="run migrate_source_column"):
        EdgarPEADIngestor().store_pead_data_edgar(edgar_income, pd.DataFrame(), db_path=db_path)
