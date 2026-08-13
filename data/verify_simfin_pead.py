"""
Stage 1 feasibility gate for the pre-registered PEAD mechanism
(earnings_drift_pead). This is stopping rule 5 from the locked test_plan:
if SimFin cannot support 200+ companies with 8+ consecutive quarters of PIT
EPS in the pre-registered small-cap universe, stop before testing.

Not part of the main pipeline; run directly with:

    PYTHONPATH=. python data/verify_simfin_pead.py

API key is read only from SIMFIN_API_KEY (never written to any file --
sf.set_api_key() sets it in-process memory only).
"""
import os
import sys

import numpy as np
import pandas as pd

MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 2_000_000_000
MIN_SMALL_CAP_COMPANIES = 300
MIN_EPS_HISTORY_COMPANIES = 200
MIN_CONSECUTIVE_QUARTERS = 8
IN_SAMPLE_START = pd.Timestamp("2022-07-01")  # amended 2026-07-10 (2nd amendment): actual usable-SUE data floor
IN_SAMPLE_END = pd.Timestamp("2023-12-31")


def _max_consecutive_run(mask: np.ndarray) -> int:
    """Longest run of consecutive True values in a boolean array."""
    if not mask.any():
        return 0
    # Run-length encode via cumulative-sum-on-breaks trick.
    breaks = np.diff(mask.astype(int)) != 0
    group_ids = np.concatenate(([0], np.cumsum(breaks)))
    best = 0
    for group_id in np.unique(group_ids):
        in_group = group_ids == group_id
        if mask[in_group][0]:  # this group is a run of True
            best = max(best, int(in_group.sum()))
    return best


def _first_usable_sue_date(group: pd.DataFrame) -> pd.Timestamp | None:
    """Publish Date of the 8th quarter in the FIRST run of MIN_CONSECUTIVE_QUARTERS
    consecutive non-null EPS quarters (chronological), i.e. the earliest date this
    company's SUE becomes computable at all -- not the raw Report Date floor."""
    valid = group["eps_basic"].notna().to_numpy()
    run_len = 0
    for i, v in enumerate(valid):
        run_len = run_len + 1 if v else 0
        if run_len >= MIN_CONSECUTIVE_QUARTERS:
            return group.iloc[i]["Publish Date"]
    return None


def main() -> None:
    api_key = os.environ.get("SIMFIN_API_KEY")
    if not api_key:
        raise RuntimeError("SIMFIN_API_KEY not set")

    import simfin as sf

    sf.set_api_key(api_key)  # in-memory only for this process; never persisted
    sf.set_data_dir("data/simfin_cache/")

    print("Loading quarterly US income statements...")
    try:
        income = sf.load_income(variant="quarterly", market="us")
    except Exception as e:
        print(f"ERROR loading income statements: {e}")
        print("STOPPING RULE 5 FIRED")
        sys.exit(1)

    results = {}

    # ---- Check 1: Publish Date (PIT) exists and is genuinely distinct ----
    print("\n--- Check 1: PIT publish_date ---")
    cols = list(income.columns) + list(income.index.names)
    has_report_date = "Report Date" in income.index.names
    has_publish_date = "Publish Date" in income.columns
    print(f"Columns (incl. index): {cols}")
    print(f"'Report Date' present: {has_report_date}")
    print(f"'Publish Date' present: {has_publish_date}")

    if not (has_report_date and has_publish_date):
        results["check1"] = False
        print("Check 1: FAIL -- required PIT columns missing")
    else:
        report_dates = pd.to_datetime(income.index.get_level_values("Report Date"))
        publish_dates = pd.to_datetime(income["Publish Date"])
        gap_days = (publish_dates - report_dates).dt.days
        sample = gap_days.dropna()
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(sample), size=min(100, len(sample)), replace=False)
        sample_gaps = sample.iloc[sample_idx]
        median_gap = float(sample_gaps.median())
        min_gap, max_gap = float(sample_gaps.min()), float(sample_gaps.max())
        print(f"Sampled {len(sample_gaps)} companies -- median gap: {median_gap:.1f} days, "
              f"min: {min_gap:.1f}, max: {max_gap:.1f}")
        results["check1"] = 30 <= median_gap <= 90
        print(f"Check 1: {'PASS' if results['check1'] else 'FAIL'}")

    # ---- Check 2: small-cap coverage (market cap proxy from most recent shareprices row) ----
    print("\n--- Check 2: small-cap coverage ---")
    print("Loading daily US share prices (for market cap proxy)...")
    try:
        prices = sf.load_shareprices(variant="daily", market="us")
    except Exception as e:
        print(f"ERROR loading share prices: {e}")
        results["check2"] = False
        n_smallcap = 0
    else:
        prices = prices.copy()
        prices["market_cap"] = prices["Close"] * prices["Shares Outstanding"]
        latest = prices.reset_index().sort_values("Date").groupby("Ticker").tail(1)
        smallcap_mask = latest["market_cap"].between(MIN_MARKET_CAP, MAX_MARKET_CAP)
        smallcap_tickers = set(latest.loc[smallcap_mask, "Ticker"])
        n_smallcap = len(smallcap_tickers)
        print(f"n_companies in $300M-$2B range (most-recent-price proxy): {n_smallcap}")
        results["check2"] = n_smallcap >= MIN_SMALL_CAP_COMPANIES
        print(f"Check 2: {'PASS' if results['check2'] else 'FAIL'} (threshold: {MIN_SMALL_CAP_COMPANIES})")

    # ---- Check 3: EPS history depth (8+ consecutive quarters) among small-caps ----
    print("\n--- Check 3: EPS history depth ---")
    if results.get("check2") is None or n_smallcap == 0:
        print("Skipping (no small-cap universe from Check 2)")
        results["check3"] = False
        n_with_history = 0
    else:
        inc = income.reset_index()
        inc = inc[inc["Ticker"].isin(smallcap_tickers)].copy()
        # EPS Basic = Net Income (Common) / Shares (Basic); not a direct SimFin
        # column -- documented derivation (see load_income_pit's docstring in
        # data/ingest_pead.py for the same convention used at ingest time).
        inc["eps_basic"] = inc["Net Income (Common)"] / inc["Shares (Basic)"]
        inc["Report Date"] = pd.to_datetime(inc["Report Date"])
        inc["Publish Date"] = pd.to_datetime(inc["Publish Date"])
        inc = inc.sort_values(["Ticker", "Report Date"])

        n_with_history = 0
        sample_rows = None
        first_usable_dates = {}
        for ticker, group in inc.groupby("Ticker"):
            group = group.reset_index(drop=True)
            valid = group["eps_basic"].notna().to_numpy()
            if _max_consecutive_run(valid) >= MIN_CONSECUTIVE_QUARTERS:
                n_with_history += 1
                if sample_rows is None:
                    sample_rows = group[group["eps_basic"].notna()][
                        ["Ticker", "Report Date", "Publish Date", "eps_basic"]
                    ].tail(5)
            fu_date = _first_usable_sue_date(group)
            if fu_date is not None:
                first_usable_dates[ticker] = fu_date

        print(f"n_companies with {MIN_CONSECUTIVE_QUARTERS}+ consecutive quarters "
              f"of non-null EPS (small-cap range): {n_with_history}")
        results["check3"] = n_with_history >= MIN_EPS_HISTORY_COMPANIES
        print(f"Check 3: {'PASS' if results['check3'] else 'FAIL'} (threshold: {MIN_EPS_HISTORY_COMPANIES})")
        if sample_rows is not None:
            print("\nSample rows (ticker, report_date, publish_date, eps_basic):")
            print(sample_rows.to_string(index=False))

    # ---- Check 4: usable-SUE date floor (not the raw Report Date floor --
    # what matters is when SUE first becomes computable at all, which lags
    # the raw Report Date floor by the 8-quarter warmup). ----
    print("\n--- Check 4: usable-SUE date floor ---")
    if not first_usable_dates:
        print("Skipping (no companies with usable SUE from Check 3)")
        results["check4"] = False
    else:
        fu_series = pd.Series(first_usable_dates)
        earliest_usable_sue = fu_series.min()
        latest_report_date = inc["Report Date"].max()
        print(f"Earliest usable-SUE publish_date across universe: {earliest_usable_sue.date()}")
        print(f"Latest Report Date in dataset: {latest_report_date.date()}")
        results["check4"] = (
            earliest_usable_sue <= IN_SAMPLE_START and latest_report_date >= IN_SAMPLE_END
        )
        print(f"Check 4: {'PASS' if results['check4'] else 'FAIL'} "
              f"(need earliest usable-SUE date <= {IN_SAMPLE_START.date()} "
              f"and latest Report Date >= {IN_SAMPLE_END.date()})")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("FEASIBILITY VERDICT")
    print("=" * 70)
    print(f"Check 1 (PIT publish_date):    {'PASS' if results.get('check1') else 'FAIL'}")
    print(f"Check 2 (small-cap coverage):  {'PASS' if results.get('check2') else 'FAIL'}")
    print(f"Check 3 (EPS history depth):   {'PASS' if results.get('check3') else 'FAIL'}")
    print(f"Check 4 (date range):          {'PASS' if results.get('check4') else 'FAIL'}")
    overall = all(results.get(k) for k in ("check1", "check2", "check3", "check4"))
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    if not overall:
        print("\nSTOPPING RULE 5 FIRED -- data pipeline build aborted. Do not proceed to Stage 2.")
        sys.exit(1)
    else:
        print("\nFEASIBILITY GATE PASSED -- proceeding to Stage 2.")


if __name__ == "__main__":
    main()
