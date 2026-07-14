"""
PEAD data ingest layer -- SEC EDGAR (edgartools) as a SECOND, INDEPENDENT
data source alongside the existing SimFin ingest (data/ingest_pead.py,
UNTOUCHED by this module). Writes into the SAME pead_store.duckdb tables,
tagged source='edgar', so both sources coexist and can be compared
(see docs/edgar_vs_simfin_data_quality.md).

Does NOT replace SimFin. Does NOT amend the earnings_drift_pead
pre-registration (mechanism_id in data/store.duckdb's mechanisms table) --
that mechanism's locked test_plan still declares SimFin as the sole data
source; amending it to add edgar as a source is a decision for a later
session, after this ingest is confirmed working. See docs/
edgar_vs_simfin_data_quality.md for the explicit reminder.

Price data gap: edgartools is XBRL/filings-only -- it does not provide
share prices. The market-cap universe filter here reuses the SAME
SimFin-cached pead_prices data the existing ingest already populates
(price data is objective shared market data, not vendor-differentiated
financial-statement data -- the actual point of a second source is
comparing EPS actuals and PIT filing dates, which Step 4's data-quality
check is scoped to). No new price rows are written by this module.

Analyst estimate count filter ("<5 analyst estimates"): NOT APPLIED here,
matching the existing SimFin ingest's current state (also not implemented
there -- see ingest_pead.py's filter_universe, which only applies market
cap and prior-quarters filters). This is a pre-existing gap in both
paths, not something introduced by edgar.

SIC exclusion (financials/utilities): NOT APPLIED, matching SimFin's
documented gap in ingest_pead.py.

SEC EDGAR identity: edgar.set_identity() must be called with a real,
descriptive contact string (SEC fair-use requirement) before any request.
Read from the EDGAR_IDENTITY environment variable -- never hardcoded here.
Rate limiting is handled automatically by edgartools (pyrate_limiter,
wired into its http client).
"""
import os
import time
from datetime import date, datetime, timezone
from typing import List

import duckdb
import pandas as pd

from config import IN_SAMPLE_END
from data.ingest import HoldoutViolationError
from data.ingest_pead import (
    MAX_MARKET_CAP,
    MIN_MARKET_CAP,
    MIN_PRIOR_QUARTERS,
    DataIntegrityError,
    assert_holdout_integrity,
)

EDGAR_CACHE_DIR = "data/edgar_cache/"

# Single-quarter (not YTD-cumulative) duration facts are the ones we want --
# XBRL duration facts for the same fiscal_period label can span either just
# the quarter (~90 days) or year-to-date-through-that-quarter (up to ~365
# days), and both appear under the same fiscal_period/fiscal_year label.
# Confirmed empirically against AAPL: Q3 2025 carries both a 272-day
# YTD-cumulative EPS (5.64) and a 90-day quarter-only EPS (1.57) -- only
# the latter is comparable to SimFin's quarterly eps_basic.
QUARTER_DURATION_MIN_DAYS = 80
QUARTER_DURATION_MAX_DAYS = 100

_EPS_CONCEPTS = ("us-gaap:EarningsPerShareBasic", "us-gaap:EarningsPerShareDiluted")


def _set_edgar_identity() -> None:
    """Reads EDGAR_IDENTITY from the environment (never hardcoded) and
    calls edgar.set_identity(). Raises if unset -- SEC fair-use requires a
    real, descriptive User-Agent with contact info; there is no safe
    default to silently fall back to."""
    import edgar

    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise ValueError(
            "EDGAR_IDENTITY environment variable is not set. SEC EDGAR fair-use "
            "policy requires a descriptive User-Agent with real contact info "
            "(e.g. 'Company Name contact@example.com') -- see "
            "https://www.sec.gov/os/accessing-edgar-data. There is no safe "
            "default; set EDGAR_IDENTITY before running this ingest."
        )
    edgar.set_identity(identity)


def _select_quarterly_eps(fact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure, offline-testable core of the EPS selection logic -- factored out
    of load_income_pit_edgar (mirroring _assert_publish_after_report's
    factoring-out in ingest_pead.py) so it's testable against a synthetic
    frame, without requiring a live/cached edgar fetch.

    Input: raw fact rows with columns ticker, concept (one of
    us-gaap:EarningsPerShareBasic / us-gaap:EarningsPerShareDiluted),
    report_date (period_end), publish_date (filing_date), eps_value,
    duration_days (period_end - period_start, in days).

    Keeps only single-quarter duration facts (QUARTER_DURATION_MIN_DAYS to
    QUARTER_DURATION_MAX_DAYS), discarding YTD-cumulative facts that share
    the same fiscal_period label -- confirmed empirically against AAPL:
    Q3 2025 carries both a 272-day YTD EPS (5.64) and a 90-day quarterly
    EPS (1.57) under the same fiscal_period label; only the latter is
    comparable to SimFin's quarterly eps_basic.

    Prefers EarningsPerShareBasic per (ticker, report_date); falls back to
    EarningsPerShareDiluted only for periods with no basic fact at all.

    Returns columns: ticker, report_date, publish_date, eps_basic.
    """
    if fact_df.empty:
        return pd.DataFrame(columns=["ticker", "report_date", "publish_date", "eps_basic"])

    quarterly = fact_df[
        (fact_df["duration_days"] >= QUARTER_DURATION_MIN_DAYS)
        & (fact_df["duration_days"] <= QUARTER_DURATION_MAX_DAYS)
    ]

    rows = []
    for ticker, group in quarterly.groupby("ticker", sort=False):
        basic = group[group["concept"] == "us-gaap:EarningsPerShareBasic"]
        diluted = group[group["concept"] == "us-gaap:EarningsPerShareDiluted"]

        basic_by_period = basic.groupby("report_date").first()
        diluted_by_period = diluted.groupby("report_date").first()

        for report_date, brow in basic_by_period.iterrows():
            rows.append({
                "ticker": ticker, "report_date": report_date,
                "publish_date": brow["publish_date"], "eps_basic": brow["eps_value"],
            })
        missing_periods = set(diluted_by_period.index) - set(basic_by_period.index)
        for report_date in missing_periods:
            drow = diluted_by_period.loc[report_date]
            rows.append({
                "ticker": ticker, "report_date": report_date,
                "publish_date": drow["publish_date"], "eps_basic": drow["eps_value"],
            })

    return pd.DataFrame(rows)


class EdgarPEADIngestor:
    def load_income_pit_edgar(self, tickers: List[str], start_date, end_date) -> pd.DataFrame:
        """
        Loads quarterly EPS (basic, falling back to diluted where basic is
        unavailable -- same fallback convention as ingest_pead.py) via
        edgartools' Company(ticker).get_facts(), for each ticker in
        `tickers`. Caches each ticker's raw fact list (unfiltered by
        duration -- filtering happens in _select_quarterly_eps) to
        EDGAR_CACHE_DIR as parquet so repeated runs don't re-hit SEC EDGAR.

        PIT discipline: publish_date is the SEC filing's acceptance date
        (edgar Filing.acceptance_datetime, matched here via the fact-level
        `filing_date`, which is date-equal to acceptance_datetime's date
        component -- confirmed empirically against AAPL's Q2 2026 10-Q:
        acceptance_datetime=2026-05-01 10:01:00+00:00, fact filing_date=
        2026-05-01). report_date is the fiscal period_end. This mirrors
        SimFin's Publish Date vs Report Date distinction exactly -- never
        key PIT logic off report_date.

        Returns columns: ticker, report_date, publish_date, eps_basic,
        source ('edgar'), filtered to publish_date within
        [start_date, end_date].
        """
        _set_edgar_identity()
        import edgar

        os.makedirs(EDGAR_CACHE_DIR, exist_ok=True)
        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

        rows = []
        for ticker in tickers:
            cache_path = os.path.join(EDGAR_CACHE_DIR, f"{ticker}_facts.parquet")
            if os.path.exists(cache_path):
                fact_df = pd.read_parquet(cache_path)
            else:
                try:
                    company = edgar.Company(ticker)
                    all_facts = company.get_facts().get_all_facts()
                except Exception as exc:
                    print(f"load_income_pit_edgar: {ticker} failed ({exc}) -- skipping")
                    continue

                fact_rows = []
                for f in all_facts:
                    if f.concept not in _EPS_CONCEPTS:
                        continue
                    if f.period_start is None or f.period_end is None:
                        continue
                    duration_days = (pd.Timestamp(f.period_end) - pd.Timestamp(f.period_start)).days
                    fact_rows.append({
                        "ticker": ticker,
                        "concept": f.concept,
                        "report_date": pd.Timestamp(f.period_end),
                        "publish_date": pd.Timestamp(f.filing_date),
                        "eps_value": float(f.value) if f.value is not None else None,
                        "duration_days": duration_days,
                    })
                fact_df = pd.DataFrame(fact_rows)
                fact_df.to_parquet(cache_path, index=False)
                # Fair-use pacing beyond edgartools' own internal rate limiter --
                # a small explicit gap between distinct companies.
                time.sleep(0.2)

            if fact_df.empty:
                continue

            selected = _select_quarterly_eps(fact_df)
            rows.extend(selected.to_dict("records"))

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["ticker", "report_date", "publish_date", "eps_basic", "source"])

        out = out[(out["publish_date"] >= start_ts) & (out["publish_date"] <= end_ts)]
        out = out.sort_values(["ticker", "report_date"]).reset_index(drop=True)
        out["source"] = "edgar"
        return out

    def filter_universe_edgar(
        self, income_df: pd.DataFrame, prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Same filter logic as PEADIngestor.filter_universe (market cap via
        prices_df, >= MIN_PRIOR_QUARTERS PIT-eligible prior quarters), reused
        against edgar-sourced income rows. `prices_df` is the EXISTING
        SimFin-sourced pead_prices data (see module docstring) -- not a
        second price source.
        """
        income_df = income_df.copy()
        income_df["report_date"] = pd.to_datetime(income_df["report_date"])
        income_df["publish_date"] = pd.to_datetime(income_df["publish_date"])
        income_df = income_df.sort_values(["ticker", "report_date"]).reset_index(drop=True)

        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        events = []
        for ticker, group in income_df.groupby("ticker", sort=False):
            group = group.reset_index(drop=True)
            eps_valid = group["eps_basic"].notna().to_numpy()
            publish_dates = group["publish_date"].to_numpy()

            price_group = prices_df[prices_df["ticker"] == ticker].sort_values("date")
            price_dates = price_group["date"].to_numpy()
            price_closes = price_group["close"].to_numpy(dtype=float)
            price_shares = price_group["shares_outstanding"].to_numpy(dtype=float)

            for i in range(len(group)):
                publish_date = publish_dates[i]

                n_prior_valid = 0
                for j in range(i - 1, -1, -1):
                    if eps_valid[j] and publish_dates[j] < publish_date:
                        n_prior_valid += 1
                passes_c = n_prior_valid >= MIN_PRIOR_QUARTERS

                import numpy as np
                if len(price_dates) == 0:
                    market_cap = np.nan
                    passes_a = False
                else:
                    idx = np.searchsorted(price_dates, publish_date, side="right") - 1
                    if idx >= 0:
                        market_cap = float(price_closes[idx] * price_shares[idx])
                        passes_a = MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP
                    else:
                        market_cap = np.nan
                        passes_a = False

                passes_b = True  # SIC filter not applied, matching ingest_pead.py

                if not passes_a:
                    reason = "no_price_data" if pd.isna(market_cap) else "market_cap_out_of_range"
                elif not passes_c:
                    reason = "insufficient_eps_history"
                else:
                    reason = None

                events.append({
                    "ticker": ticker,
                    "publish_date": pd.Timestamp(publish_date),
                    "market_cap_at_announce": market_cap,
                    "passes_all_filters": bool(passes_a and passes_b and passes_c),
                    "filter_reason": reason,
                    "sic_filter_applied": False,
                    "source": "edgar",
                })

        return pd.DataFrame(events)

    def store_pead_data_edgar(
        self,
        income_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        db_path: str = "data/pead_store.duckdb",
    ) -> None:
        """Named-column INSERT OR REPLACE into pead_income/pead_universe,
        source='edgar'. Requires migrate_source_column() to have already
        run (adds the `source` column and expands the primary key to
        (ticker, report_date/publish_date, source) so edgar rows coexist
        with, rather than overwrite, existing SimFin rows for the same
        ticker/quarter)."""
        conn = duckdb.connect(db_path)

        cols = conn.execute("PRAGMA table_info('pead_income')").fetchdf()["name"].tolist()
        if "source" not in cols:
            raise RuntimeError(
                "pead_income has no `source` column -- run migrate_source_column() first "
                "(schema migration is additive and preserves all existing SimFin rows)."
            )

        income_insert = income_df[["ticker", "report_date", "publish_date", "eps_basic", "source"]]
        conn.execute(
            "INSERT OR REPLACE INTO pead_income "
            "SELECT ticker, report_date, publish_date, eps_basic, source FROM income_insert"
        )

        universe_insert = universe_df[
            ["ticker", "publish_date", "market_cap_at_announce", "passes_all_filters",
             "filter_reason", "sic_filter_applied", "source"]
        ]
        conn.execute(
            "INSERT OR REPLACE INTO pead_universe "
            "SELECT ticker, publish_date, market_cap_at_announce, passes_all_filters, "
            "filter_reason, sic_filter_applied, source FROM universe_insert"
        )

        n_income_edgar = conn.execute(
            "SELECT COUNT(*) FROM pead_income WHERE source='edgar'"
        ).fetchone()[0]
        n_universe_edgar = conn.execute(
            "SELECT COUNT(*) FROM pead_universe WHERE source='edgar'"
        ).fetchone()[0]
        n_income_simfin = conn.execute(
            "SELECT COUNT(*) FROM pead_income WHERE source='simfin'"
        ).fetchone()[0]

        print(f"pead_income:   {n_income_edgar} edgar rows written "
              f"({n_income_simfin} pre-existing simfin rows untouched)")
        print(f"pead_universe: {n_universe_edgar} edgar rows written")
        conn.close()


def migrate_source_column(db_path: str = "data/pead_store.duckdb") -> None:
    """
    Additive, non-destructive schema migration: adds a `source` column to
    pead_income and pead_universe and expands their PRIMARY KEY to include
    it, so SimFin and edgar rows can coexist for the same ticker/quarter
    instead of the second INSERT OR REPLACE silently overwriting the first.
    All pre-existing rows are backfilled source='simfin' -- their data is
    otherwise byte-identical, verified by row-count check below.

    pead_prices is NOT migrated -- it has no source-specific rows (see
    module docstring; edgar does not provide price data).

    DuckDB does not support altering a PRIMARY KEY in place, so this uses
    the standard rename-recreate-copy-drop pattern, wrapped in a single
    transaction so it's all-or-nothing. Idempotent: a no-op if `source`
    already exists on pead_income.
    """
    conn = duckdb.connect(db_path)
    cols = conn.execute("PRAGMA table_info('pead_income')").fetchdf()["name"].tolist()
    if "source" in cols:
        print("migrate_source_column: pead_income already has a `source` column -- no-op")
        conn.close()
        return

    n_income_before = conn.execute("SELECT COUNT(*) FROM pead_income").fetchone()[0]
    n_universe_before = conn.execute("SELECT COUNT(*) FROM pead_universe").fetchone()[0]

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("ALTER TABLE pead_income RENAME TO pead_income_pre_edgar_migration")
        conn.execute("""
            CREATE TABLE pead_income (
                ticker VARCHAR, report_date DATE, publish_date DATE, eps_basic DOUBLE,
                source VARCHAR DEFAULT 'simfin',
                PRIMARY KEY (ticker, report_date, source)
            )
        """)
        conn.execute(
            "INSERT INTO pead_income "
            "SELECT ticker, report_date, publish_date, eps_basic, 'simfin' "
            "FROM pead_income_pre_edgar_migration"
        )
        conn.execute("DROP TABLE pead_income_pre_edgar_migration")

        conn.execute("ALTER TABLE pead_universe RENAME TO pead_universe_pre_edgar_migration")
        conn.execute("""
            CREATE TABLE pead_universe (
                ticker VARCHAR, publish_date DATE, market_cap_at_announce DOUBLE,
                passes_all_filters BOOLEAN, filter_reason VARCHAR, sic_filter_applied BOOLEAN,
                source VARCHAR DEFAULT 'simfin',
                PRIMARY KEY (ticker, publish_date, source)
            )
        """)
        conn.execute(
            "INSERT INTO pead_universe "
            "SELECT ticker, publish_date, market_cap_at_announce, passes_all_filters, "
            "filter_reason, sic_filter_applied, 'simfin' FROM pead_universe_pre_edgar_migration"
        )
        conn.execute("DROP TABLE pead_universe_pre_edgar_migration")

        n_income_after = conn.execute(
            "SELECT COUNT(*) FROM pead_income WHERE source='simfin'"
        ).fetchone()[0]
        n_universe_after = conn.execute(
            "SELECT COUNT(*) FROM pead_universe WHERE source='simfin'"
        ).fetchone()[0]
        if n_income_after != n_income_before or n_universe_after != n_universe_before:
            raise RuntimeError(
                f"Migration row-count mismatch: pead_income {n_income_before} -> {n_income_after}, "
                f"pead_universe {n_universe_before} -> {n_universe_after}. Rolling back."
            )

        conn.execute("COMMIT")
        print(f"migrate_source_column: pead_income {n_income_after}/{n_income_before} rows preserved, "
              f"pead_universe {n_universe_after}/{n_universe_before} rows preserved. "
              f"source='simfin' backfilled on all pre-existing rows.")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def run_pead_ingest_edgar(
    tickers: List[str],
    history_start,
    in_sample_end=None,
    db_path: str = "data/pead_store.duckdb",
):
    """Ties the edgar ingest layer together, mirroring run_pead_ingest()
    in ingest_pead.py. Does NOT run experiments, does NOT amend the
    pre-registration -- ingest and universe-filter only."""
    if in_sample_end is None:
        in_sample_end = pd.Timestamp(IN_SAMPLE_END)

    migrate_source_column(db_path=db_path)

    ingestor = EdgarPEADIngestor()

    income_df = ingestor.load_income_pit_edgar(tickers, history_start, in_sample_end)
    assert_holdout_integrity(income_df, pit_col="publish_date")
    print(f"Loaded {len(income_df)} edgar PIT income rows for {len(tickers)} tickers, "
          f"publish_date in [{pd.Timestamp(history_start).date()}, {pd.Timestamp(in_sample_end).date()}]")

    # Reuse existing SimFin-sourced prices for the market-cap filter -- see
    # module docstring (edgar provides no price data).
    conn = duckdb.connect(db_path, read_only=True)
    prices_df = conn.execute(
        "SELECT ticker, date, open, high, low, close, volume, shares_outstanding "
        "FROM pead_prices WHERE ticker IN ?", [tickers]
    ).fetchdf() if income_df.empty else conn.execute(
        f"SELECT ticker, date, open, high, low, close, volume, shares_outstanding "
        f"FROM pead_prices WHERE ticker IN ({','.join('?' * len(tickers))})", tickers
    ).fetchdf()
    conn.close()

    universe_df = ingestor.filter_universe_edgar(income_df, prices_df)
    print(f"Universe events (edgar income x simfin prices): {len(universe_df)} total, "
          f"{int(universe_df['passes_all_filters'].sum()) if not universe_df.empty else 0} passing all filters")

    ingestor.store_pead_data_edgar(income_df, universe_df, db_path=db_path)

    return income_df, universe_df


if __name__ == "__main__":
    print("data/ingest_pead_edgar.py is a library module for this session's verification "
          "run only -- see tests/test_pead_ingest_edgar.py and docs/"
          "edgar_vs_simfin_data_quality.md. Not wired into any production routine yet.")
