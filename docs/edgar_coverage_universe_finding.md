# Edgar Coverage vs. Universe Definition: Why the Proposed 5th Amendment Was Not Made
Date: 2026-07-14
Status: no pre-registration amendment made this session. `earnings_drift_pead` remains locked at hash `1060018b72725b02d7964e8ae81f969b65fecdb570443d1fbe3fffc4cfd4c674`, 3 amendment_log entries, Q4 2023-only in-sample window, unchanged.

## What Was Asked

Amend the locked `earnings_drift_pead` pre-registration (5th amendment) to designate edgar as PIT-authoritative and extend the in-sample window, computing the new start date from edgar's actual coverage across the universe (not guessed), using "the 10th percentile earliest-coverage date."

## Why This Didn't Happen

Computing that percentile honestly surfaced a problem with the premise, not just an implementation detail.

Using the 388 tickers already cross-checked against edgar in this session's prior investigation (a coverage sample, not a cherry-picked one), the distribution of each ticker's earliest available quarterly EPS `report_date` is:

| Percentile | Earliest-coverage date | Interpretation |
|---|---|---|
| min | 2008-06-30 | |
| p10 | 2009-09-27 | Barely later than the raw minimum |
| p25 | 2010-06-30 | |
| **p50 (median)** | **2011-03-31** | Matches this project's own prior convention (2nd amendment used a median-based estimate: "median 2022-08-08") |
| p75 | 2017-10-23 | |
| p80 | 2019-06-30 | |
| **p90** | **2020-09-30** | The date by which 90% of the universe has *any* coverage at all |
| p95 | 2021-06-30 | |

**The distribution is bimodal, not a smooth curve with a few early outliers.** There's a cluster of tickers with data back to 2010-2011, then a sharp jump to 2017-2020. This is not an edgar coverage limitation -- it reflects that a substantial share of this specific $300M-$2B small/mid-cap universe consists of companies that were not public before roughly 2017-2020 (recent IPOs, SPAC mergers, spin-offs). No data source, including edgar, can supply pre-listing filing history for a company that didn't exist as a public filer yet.

**The literal "10th percentile" instruction does not achieve its own stated purpose.** The stated rationale was to "protect against a few early outlier tickers skewing the estimate." But p10 (2009-09-27) is barely different from the raw minimum (2008-06-30) -- both sit in the same early cluster. Protecting against early-outlier skew requires moving toward the *high* end of the distribution (e.g. p90), not the low end, since the outliers being protected against are the earliest-covered tickers, and p10 doesn't exclude them at all.

**But the statistically honest choice undermines the amendment's premise.** p90 (2020-09-30) -- the threshold by which the bulk of the universe genuinely has coverage -- lands almost exactly at SimFin's existing free-tier floor (~2020). At that threshold, edgar does not meaningfully extend the usable in-sample window beyond what SimFin already provides for this universe. The "pre-2018+" extension motivating this amendment holds for individual established companies (as shown for the 5 original sample tickers, all covering 2000-2015), but not for the *universe as currently defined*, which includes many companies too young to have pre-2018 history regardless of source.

Using the median (p50 = 2011-03-31, consistent with the project's own prior amendment convention) would extend the window substantially, but only by counting on data availability for the subset of tickers old enough to have it -- effectively changing what fraction of the declared universe is actually being tested in the earlier years, a different question than "what does edgar cover" and one that deserves deliberate discussion, not a percentile default.

## Conclusion

**This is a universe-definition question, not a data-source question**, and amending the pre-registration to extend the in-sample window requires first deciding how to handle a universe where coverage genuinely varies by company age -- not simply picking a percentile of edgar's reach. That decision is out of scope for this session.

## What Still Stands

None of the following is invalidated by this finding:
- `data/ingest_pead_edgar.py` -- the edgar ingest infrastructure, additive schema migration, and tests (committed `ee674e9`).
- `docs/edgar_vs_simfin_data_quality.md` -- the SimFin date-attribution defect finding (6/50 sampled quarters, ~365-day error) and edgar's PIT-authority case, both independent of the universe-coverage question above.
- The full-population investigation confirming that defect does not appear to have contaminated the already-reported Q4 2023 pilot result (0/312 verifiable events affected).

## Separate Finding: amendment_log Has Only 3 Entries, Not 4

While reading the current pre-registration (Step 0), the stored `amendment_log` array has **3 entries**, but the third entry's own text says "Fourth amendment," and its reasoning explicitly references content from a "third amendment" (the 8-vs-12-prior-quarters correction) that has no dedicated log entry of its own. Git history confirms a real commit titled "Third amendment to earnings_drift_pead pre-registration: data-floor window" (`fcbdc33`) exists, but since `data/store.duckdb` is gitignored, no diff or content from that amendment is recoverable from git -- only the commit message survives. This means a real amendment's dedicated log entry is missing from the array, and its exact original content cannot be reconstructed. This does not affect the current hash's validity (the hash is computed from current content, which is internally consistent with itself) -- it's a completeness gap in the human-readable audit trail, not a tamper-detection failure. Flagging for awareness; not fixed here, since backfilling a plausible-sounding entry for content I cannot verify would itself be a form of tampering with the historical record.

## Pre-registration State

`earnings_drift_pead` is unchanged: `prereg_locked=True`, `prereg_hash=1060018b72725b02d7964e8ae81f969b65fecdb570443d1fbe3fffc4cfd4c674`, 3 amendment_log entries, in-sample window still Q4 2023 only. No 5th amendment was made. No experiment was run. The universe-definition question above is deferred to a later, separate session.

## Files
`data/edgar_cache/` -- the 394-ticker cache this analysis was computed from (not committed, gitignored).
