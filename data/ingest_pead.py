"""
PEAD data ingest layer -- a separate research track alongside the existing
yfinance/store.duckdb pipeline (data/ingest.py), not a replacement for it.
Writes to its own database (data/pead_store.duckdb) so the audited
3,600-experiment pipeline's tables are never touched.

Pre-registration: earnings_drift_pead, hash a3e32b91... (locked).
In-sample event window: 2022-07-01 through 2023-12-31 (the data-floor window
set by the second amendment -- SimFin free tier coverage plus the 8-quarter
SUE warmup produces no usable event before 2022-06-22). Holdout: 2024-2025,
untouched until the in-sample result is known.

SIMFIN_API_KEY is read only from the environment, never written to any file.
If the key is unset, these methods still work as long as data/simfin_cache/
already has the datasets cached on disk (sf.load_income /
sf.load_shareprices read from cache automatically and do not require a live
API call when the cache is present).
"""
import importlib.metadata
import os
from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from config import IN_SAMPLE_END
from data.ingest import HoldoutViolationError

MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 2_000_000_000
MIN_PRIOR_QUARTERS = 8

# Wide enough to cover SimFin's actual data floor (Report Date coverage
# starts 2020-08-31) plus buffer -- NOT the same as the pre-registered
# in-sample EVENT window below. History this far back is needed purely to
# validate the "8 prior quarters" filter and compute SUE for events near the
# start of the event window; those older quarters are not usable events
# themselves.
PEAD_HISTORY_START = pd.Timestamp("2020-07-01")

# The actual pre-registered in-sample window (2nd amendment, hash a3e32b91...).
PEAD_IN_SAMPLE_START = pd.Timestamp("2022-07-01")
PEAD_IN_SAMPLE_END = pd.Timestamp(IN_SAMPLE_END)  # 2023-12-31, same boundary as the main pipeline
PEAD_HOLDOUT_START = pd.Timestamp("2024-01-01")
PEAD_HOLDOUT_END = pd.Timestamp("2025-12-31")


class DataIntegrityError(Exception):
    """Raised when ingested PEAD data violates a structural invariant -- e.g.
    a filing whose Publish Date does not strictly follow its own Report
    Date, which is impossible (a filing cannot be available at or before
    the period it covers ends) and signals corrupted or misparsed source
    data rather than something to filter around."""


def _assert_publish_after_report(df: pd.DataFrame) -> None:
    """Raise DataIntegrityError if any row has publish_date <= report_date.
    Factored out of load_income_pit so it's testable offline against a
    synthetic frame, without requiring a live/cached SimFin fetch."""
    violations = df[df["publish_date"] <= df["report_date"]]
    if not violations.empty:
        raise DataIntegrityError(
            f"{len(violations)} row(s) have publish_date <= report_date, which is "
            f"impossible -- a filing cannot be available at or before the period it "
            f"covers ends. Violating rows:\n{violations.to_string(index=False)}"
        )


def _quarantine_bad_pit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows with publish_date <= report_date rather than hard-failing
    the whole ingest over them. Confirmed against real cached SimFin data
    (2026-07-10) at 17 of 32,609 rows (0.052%) across 14 tickers -- including
    established non-SPAC names (ADBE, HRB, UA), so this is a vendor-side PIT
    data-quality issue, not a bug in this check and not isolated to
    thinly-traded names. User-confirmed quarantine-and-continue decision;
    excluded rows are always printed in full, never silently dropped. Use
    _assert_publish_after_report directly if a hard-fail check is wanted
    instead (e.g. in tests)."""
    bad_mask = df["publish_date"] <= df["report_date"]
    n_bad = int(bad_mask.sum())
    if n_bad:
        print(f"DATA QUALITY: excluding {n_bad} row(s) with publish_date <= report_date "
              f"(vendor PIT data-quality issue, confirmed not a bug in this check):")
        print(df[bad_mask].to_string(index=False))
    return df[~bad_mask].reset_index(drop=True)


def assert_holdout_integrity(df: pd.DataFrame, pit_col: str = "publish_date") -> None:
    """Structural backstop mirroring data/ingest.py's assert_discovery_in_sample:
    raise HoldoutViolationError if any row in `df` has a PIT date past
    IN_SAMPLE_END. The PEAD track shares the same holdout boundary as the
    existing pipeline -- the 2024-2025 window must stay untouched until the
    in-sample result is known. Call this on every loaded frame before
    storing, regardless of what range was requested at the call site."""
    if df is None or df.empty:
        return
    dates = pd.to_datetime(df[pit_col])
    max_date = dates.max()
    boundary = pd.Timestamp(IN_SAMPLE_END)
    if max_date > boundary:
        raise HoldoutViolationError(
            f"PEAD pipeline attempted to read {pit_col!r} through {max_date.date()}, "
            f"past IN_SAMPLE_END ({boundary.date()}). The PEAD track shares the same "
            f"holdout boundary as the existing pipeline -- the 2024-2025 window must "
            f"stay untouched until the in-sample result is known."
        )


class PEADIngestor:
    def load_income_pit(self, start_date, end_date) -> pd.DataFrame:
        """
        Loads quarterly US income statements via SimFin
        (sf.load_income(variant='quarterly', market='us')) from the local
        cache at data/simfin_cache/ (set via sf.set_data_dir before this
        call). SimFin reads from disk automatically and does not re-fetch
        when the cache is already present.

        PIT discipline: uses SimFin's 'Publish Date' column as the
        availability timestamp, NOT 'Report Date' (the fiscal period end,
        exposed as an index level). A filing for the quarter ending
        2022-12-31 published on 2023-02-15 is not knowable until
        2023-02-15 -- all downstream filtering and joins key off Publish
        Date, never Report Date.

        EPS: SimFin's quarterly income statement has no direct EPS column.
        eps_basic is derived as `Net Income (Common) / Shares (Basic)`, the
        same convention used in data/verify_simfin_pead.py's feasibility
        check. Where eps_basic is unavailable but Shares (Diluted) is,
        eps_diluted (`Net Income (Common) / Shares (Diluted)`) is used as a
        fallback -- the fallback count is printed so a silent data-quality
        shift doesn't go unnoticed.

        Returns columns: ticker, report_date, publish_date, eps_basic,
        filtered to Publish Date within [start_date, end_date].
        """
        import simfin as sf

        api_key = os.environ.get("SIMFIN_API_KEY")
        if api_key:
            sf.set_api_key(api_key)
        sf.set_data_dir("data/simfin_cache/")

        income = sf.load_income(variant="quarterly", market="us")
        df = income.reset_index()
        df["Report Date"] = pd.to_datetime(df["Report Date"])
        df["Publish Date"] = pd.to_datetime(df["Publish Date"])

        df["eps_basic"] = df["Net Income (Common)"] / df["Shares (Basic)"]
        eps_diluted = df["Net Income (Common)"] / df["Shares (Diluted)"]
        fallback_mask = df["eps_basic"].isna() & eps_diluted.notna()
        n_fallback = int(fallback_mask.sum())
        if n_fallback:
            print(f"load_income_pit: {n_fallback} row(s) fall back to eps_diluted "
                  f"(eps_basic unavailable, Shares (Diluted) present).")
        df.loc[fallback_mask, "eps_basic"] = eps_diluted[fallback_mask]

        out = df.rename(columns={
            "Ticker": "ticker", "Report Date": "report_date", "Publish Date": "publish_date",
        })[["ticker", "report_date", "publish_date", "eps_basic"]]

        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        out = out[(out["publish_date"] >= start_ts) & (out["publish_date"] <= end_ts)]
        out = out.reset_index(drop=True)

        out = _quarantine_bad_pit_rows(out)

        return out

    def load_prices_pit(self, tickers, start_date, end_date) -> pd.DataFrame:
        """
        Loads daily US share prices via SimFin
        (sf.load_shareprices(variant='daily', market='us')) from the local
        cache. SimFin's daily shareprices dataset returns split/dividend-
        adjusted OHLC directly -- there is no separate unadjusted/adjusted
        variant flag for this endpoint (confirmed during Stage 1
        feasibility). 'Shares Outstanding' is included in the same dataset,
        giving market cap at any date as Close * Shares Outstanding without
        a second join.

        Returns columns: ticker, date, open, high, low, close, volume,
        shares_outstanding, filtered to the given tickers and
        [start_date, end_date] on date.
        """
        import simfin as sf

        api_key = os.environ.get("SIMFIN_API_KEY")
        if api_key:
            sf.set_api_key(api_key)
        sf.set_data_dir("data/simfin_cache/")

        prices = sf.load_shareprices(variant="daily", market="us")
        df = prices.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Ticker"].isin(set(tickers))]

        start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
        df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)]

        out = df.rename(columns={
            "Ticker": "ticker", "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Shares Outstanding": "shares_outstanding",
        })[["ticker", "date", "open", "high", "low", "close", "volume", "shares_outstanding"]]

        return out.reset_index(drop=True)

    def filter_universe(self, income_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pre-registered universe filters and returns one row per
        eligible (ticker, publish_date) ANNOUNCEMENT EVENT -- not just
        eligible tickers, since the same ticker can pass some quarters and
        fail others (e.g. market cap drifts out of range).

        Filter (a) market cap $300M-$2B at announcement: shares_outstanding
        * close on publish_date, or the nearest PRIOR trading day if
        publish_date is not itself a trading day -- the last price *known*
        at announcement time, never a later/future price (this is a PIT
        eligibility filter, unlike compute_forward_returns's T+1 entry rule
        which is about execution, not eligibility).

        Filter (b) SIC 6000-6999 / 4900-4999 exclusion: NOT APPLIED.
        SimFin's load_companies(market='us') has no SIC code field (only
        IndustryId, which maps to sector/industry names) -- confirmed
        during Stage 1 feasibility. This is a documented known gap, not a
        silent skip: sic_filter_applied=False on every row.

        Filter (c) >= 8 prior quarters of non-null EPS, PIT: "prior" means
        publish_date of those earlier filings is strictly less than this
        event's publish_date -- position in the sorted-by-report_date frame
        is not sufficient on its own (a restated or out-of-order filing
        could make a positionally-prior quarter not actually knowable yet).
        """
        income_df = income_df.copy()
        income_df["report_date"] = pd.to_datetime(income_df["report_date"])
        income_df["publish_date"] = pd.to_datetime(income_df["publish_date"])
        income_df = income_df.sort_values(["ticker", "report_date"]).reset_index(drop=True)

        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        sic_filter_applied = False  # see docstring -- SimFin has no SIC codes

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

                passes_b = True  # SIC filter not applied -- see docstring

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
                    "sic_filter_applied": sic_filter_applied,
                })

        return pd.DataFrame(events)

    def store_pead_data(
        self,
        income_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        db_path: str = "data/pead_store.duckdb",
    ) -> None:
        """Writes pead_income, pead_prices, pead_universe (named-column
        INSERT OR REPLACE -- idempotent, safe to run twice) plus one audit
        row per run in pead_ingest_log (plain INSERT -- an append-only log,
        not idempotent by design)."""
        conn = duckdb.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pead_income (
                ticker VARCHAR, report_date DATE, publish_date DATE, eps_basic DOUBLE,
                PRIMARY KEY (ticker, report_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pead_prices (
                ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
                close DOUBLE, volume BIGINT, shares_outstanding DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pead_universe (
                ticker VARCHAR, publish_date DATE, market_cap_at_announce DOUBLE,
                passes_all_filters BOOLEAN, filter_reason VARCHAR, sic_filter_applied BOOLEAN,
                PRIMARY KEY (ticker, publish_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pead_ingest_log (
                run_timestamp TIMESTAMP, simfin_version VARCHAR, n_income_rows INTEGER,
                n_price_rows INTEGER, n_universe_events INTEGER, n_events_passing_filters INTEGER,
                in_sample_start DATE, in_sample_end DATE, holdout_start DATE, holdout_end DATE
            )
        """)

        income_insert = income_df[["ticker", "report_date", "publish_date", "eps_basic"]]
        conn.execute(
            "INSERT OR REPLACE INTO pead_income "
            "SELECT ticker, report_date, publish_date, eps_basic FROM income_insert"
        )

        prices_insert = prices_df[
            ["ticker", "date", "open", "high", "low", "close", "volume", "shares_outstanding"]
        ]
        conn.execute(
            "INSERT OR REPLACE INTO pead_prices "
            "SELECT ticker, date, open, high, low, close, volume, shares_outstanding FROM prices_insert"
        )

        universe_insert = universe_df[
            ["ticker", "publish_date", "market_cap_at_announce", "passes_all_filters",
             "filter_reason", "sic_filter_applied"]
        ]
        conn.execute(
            "INSERT OR REPLACE INTO pead_universe "
            "SELECT ticker, publish_date, market_cap_at_announce, passes_all_filters, "
            "filter_reason, sic_filter_applied FROM universe_insert"
        )

        n_income = conn.execute("SELECT COUNT(*) FROM pead_income").fetchone()[0]
        n_prices = conn.execute("SELECT COUNT(*) FROM pead_prices").fetchone()[0]
        n_universe = conn.execute("SELECT COUNT(*) FROM pead_universe").fetchone()[0]
        n_passing = conn.execute(
            "SELECT COUNT(*) FROM pead_universe WHERE passes_all_filters"
        ).fetchone()[0]

        print(f"pead_income:   {n_income} rows")
        print(f"pead_prices:   {n_prices} rows")
        print(f"pead_universe: {n_universe} rows ({n_passing} passing all filters)")

        try:
            simfin_version = importlib.metadata.version("simfin")
        except importlib.metadata.PackageNotFoundError:
            simfin_version = "unknown"

        conn.execute(
            """
            INSERT INTO pead_ingest_log (
                run_timestamp, simfin_version, n_income_rows, n_price_rows,
                n_universe_events, n_events_passing_filters, in_sample_start,
                in_sample_end, holdout_start, holdout_end
            ) VALUES (
                $run_timestamp, $simfin_version, $n_income_rows, $n_price_rows,
                $n_universe_events, $n_events_passing_filters, $in_sample_start,
                $in_sample_end, $holdout_start, $holdout_end
            )
            """,
            {
                "run_timestamp": datetime.now(timezone.utc),
                "simfin_version": simfin_version,
                "n_income_rows": n_income,
                "n_price_rows": n_prices,
                "n_universe_events": n_universe,
                "n_events_passing_filters": n_passing,
                "in_sample_start": PEAD_IN_SAMPLE_START.date(),
                "in_sample_end": PEAD_IN_SAMPLE_END.date(),
                "holdout_start": PEAD_HOLDOUT_START.date(),
                "holdout_end": PEAD_HOLDOUT_END.date(),
            },
        )
        conn.close()


def run_pead_ingest(
    history_start=PEAD_HISTORY_START,
    in_sample_end=PEAD_IN_SAMPLE_END,
    db_path: str = "data/pead_store.duckdb",
):
    """Ties the ingest layer together: load PIT income/prices from
    history_start through in_sample_end (a WIDE range -- needed so the
    8-prior-quarters filter and later SUE computation have runway before
    the pre-registered event window even starts), filter the universe,
    store all three tables, then check stopping rule 5 by counting only
    events whose publish_date falls in the pre-registered in-sample window
    (PEAD_IN_SAMPLE_START through PEAD_IN_SAMPLE_END)."""
    ingestor = PEADIngestor()

    income_df = ingestor.load_income_pit(history_start, in_sample_end)
    assert_holdout_integrity(income_df, pit_col="publish_date")
    print(f"Loaded {len(income_df)} PIT income rows, publish_date in "
          f"[{pd.Timestamp(history_start).date()}, {pd.Timestamp(in_sample_end).date()}]")

    tickers = income_df["ticker"].unique().tolist()
    prices_df = ingestor.load_prices_pit(tickers, history_start, in_sample_end)
    assert_holdout_integrity(prices_df, pit_col="date")
    print(f"Loaded {len(prices_df)} PIT price rows for {len(tickers)} tickers")

    universe_df = ingestor.filter_universe(income_df, prices_df)
    print(f"Universe events (all history, not just in-sample window): {len(universe_df)} total, "
          f"{int(universe_df['passes_all_filters'].sum())} passing all filters")

    ingestor.store_pead_data(income_df, prices_df, universe_df, db_path=db_path)

    in_window = universe_df[
        universe_df["passes_all_filters"]
        & (universe_df["publish_date"] >= PEAD_IN_SAMPLE_START)
        & (universe_df["publish_date"] <= PEAD_IN_SAMPLE_END)
    ]
    n_events = len(in_window)
    print(f"\nUsable in-sample events ({PEAD_IN_SAMPLE_START.date()} to "
          f"{PEAD_IN_SAMPLE_END.date()}): {n_events}")
    if n_events < 200:
        print("STOPPING RULE 5 FIRED -- fewer than 200 usable events in the "
              "in-sample window. Do not proceed to SUE computation.")
    else:
        print(f"Event floor passed ({n_events} events). Proceeding.")

    return income_df, prices_df, universe_df


if __name__ == "__main__":
    run_pead_ingest()
