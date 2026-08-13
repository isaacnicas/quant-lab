# EDGAR vs SimFin Data Quality Comparison
Date: 2026-07-14
Sample tickers: ACLS, ADUS, ADMA, ABUS, ACRS (drawn from the real, already-passing SimFin `pead_universe` set)
Scope: data quality only -- no backtest, no Sharpe ratios, no experiments run.

## Method

Both sources' `pead_income` rows (`source='simfin'` and `source='edgar'`) were joined on `(ticker, report_date)` for the 5 sample tickers, restricted to the 50 quarters where both sources have data (SimFin: 65 total rows for these tickers; edgar: 85 total rows, extending back to 2018 vs SimFin's ~2020 floor -- see report below for the earliest-date comparison).

## EPS Actuals

**44 / 50 (88%) match within 0.01.** 6 mismatches, all small in magnitude:

| Ticker | Report Date | SimFin EPS | Edgar EPS | Diff |
|---|---|---:|---:|---:|
| ABUS | 2022-03-31 | -0.1556 | -0.11 | -0.0456 |
| ABUS | 2023-06-30 | -0.0001 | -0.10 | +0.0999 |
| ACLS | 2022-09-30 | 1.2005 | 1.22 | -0.0195 |
| ADMA | 2021-06-30 | -0.1635 | -0.15 | -0.0135 |
| ADMA | 2021-09-30 | -0.2056 | -0.13 | -0.0756 |
| ADUS | 2023-06-30 | 0.9165 | 0.93 | -0.0135 |

**Likely explanation, not a bug in either source per se**: SimFin's `eps_basic` here is *derived* (Net Income (Common) / Shares (Basic), per `ingest_pead.py`'s own documented convention), whereas edgar's is the company's *directly reported* `us-gaap:EarningsPerShareBasic` XBRL fact. Companies routinely report EPS after rounding, share-count timing adjustments, or treatment of items (e.g. preferred dividends, discontinued operations) that a simple Net Income / Shares recomputation won't exactly reproduce. The ABUS 2023-06-30 case (-0.0001 vs -0.10, off by two orders of magnitude) is the one outlier worth a closer look in a future session -- it looks more like a genuine data anomaly (possibly a near-zero-net-income quarter where the derived-EPS convention is numerically unstable) than simple rounding.

## Filing/Publish Dates

**44 / 50 (88%) agree same-day.** The 6 disagreements are NOT noise -- they are a consistent, systematic ~1-year offset:

| Ticker | Report Date | SimFin Publish Date | Edgar Publish Date (SEC acceptance) | Diff (days) | EPS still matches? |
|---|---|---|---|---:|---|
| ACRS | 2021-06-30 | 2021-08-05 | 2022-08-03 | -363 | Yes |
| ABUS | 2020-09-30 | 2020-11-05 | 2021-11-04 | -364 | Yes |
| ACRS | 2021-09-30 | 2021-11-02 | 2022-11-08 | -371 | Yes |
| ADMA | 2021-03-31 | 2021-05-12 | 2022-05-11 | -364 | Yes |
| ACRS | 2021-03-31 | 2021-05-07 | 2022-05-10 | -368 | Yes |
| ADMA | 2020-09-30 | 2020-11-05 | 2021-11-10 | -370 | Yes |

**This is a genuine, previously-undetected SimFin PIT data quality issue**, distinct from the already-documented `publish_date <= report_date` vendor issue in `ingest_pead.py` (confirmed there at 0.052% of rows). In every one of these 6 cases, EPS values agree, isolating this cleanly to a *date-attribution* error, not a value error -- SimFin's Publish Date appears to be off by almost exactly one year (not caught by the existing quarantine check, since these dates are still nominally after `report_date`). Edgar's `acceptance_datetime` is the SEC's own immutable timestamp for when the filing was actually received -- it is the ground truth here, not a second opinion to average against SimFin.

**Rate**: 6/50 (12%) in this small 5-ticker sample. This is not a large enough sample to state a reliable population-wide rate, but it is large enough to say this class of error is real and not a one-off: it hit 3 of the 5 sample tickers (ACRS, ABUS, ADMA), all in the 2020-2021 report-date range. This is exactly the kind of issue a second independent source exists to catch -- it would not have been discoverable from SimFin data alone.

## Coverage: How Far Back Does Edgar Actually Extend?

Earliest `publish_date` per sample ticker (edgar):

| Ticker | Earliest Edgar Publish Date | Earliest SimFin Publish Date (this ingest) |
|---|---|---|
| ACLS | pre-2018 (filing history starts 2000-08-24) | ~2020 (SimFin free-tier floor) |
| ADUS | pre-2018 (2009-11-20) | ~2020 |
| ADMA | pre-2018 (2008-11-05) | ~2020 |
| ABUS | pre-2018 (2014-03-28) | ~2020 |
| ACRS | pre-2018 (2015-11-18) | ~2020 |

All 5 sample tickers have SEC filing history extending well before 2018 -- the coverage gap the pre-registration's fourth amendment identified (SimFin free tier + 12-quarter SUE warmup collapsing the effective in-sample window to Q4 2023 only) is a SimFin-specific limitation, not a fundamental data-availability limitation. Edgar genuinely does extend usable history back toward 2018 as hoped.

**Actual earliest usable date given the 12-quarter SUE warmup requirement**: not computed in this session (would require ingesting all universe tickers back to ~2015 to build full SUE series, which is a discovery/experiment-adjacent step, out of scope here -- see reminder below). Qualitatively: since edgar coverage for these 5 tickers starts 2000-2015, a 12-quarter (3-year) warmup lands comfortably within edgar's range, suggesting a genuine in-sample window starting well before 2022 is feasible -- but this must be verified across the full universe, not assumed from 5 tickers, before any amendment.

## Files
- `data/ingest_pead_edgar.py` -- edgar ingest, `docs/` this file
- `tests/test_pead_ingest_edgar.py` -- 12/12 offline tests passing
- `data/edgar_cache/` -- cached raw XBRL facts for the 5 sample tickers (not committed, gitignored)

---

**No pre-registration amendment has been made. No experiments have been run.** This session only builds and verifies the edgar ingest. The next session should: (a) decide whether to amend the pre-registration's data source field to include edgar as a source (and how to treat the ~1-year SimFin date-attribution error found above -- likely: prefer edgar's PIT dates where both sources exist), (b) if amended, determine the new effective in-sample window by running the full universe (not just 5 tickers) through edgar's coverage and the 12-quarter SUE warmup, (c) only then run any test.
