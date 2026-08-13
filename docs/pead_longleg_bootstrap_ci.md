# PEAD Long-Leg Sharpe: Bootstrap CI Attempt — STOPPED at Validation
Date: 2026-07-15
Branch: `fix/audit-lag-drawdown-trades`
Scope: read-only evidence-gathering. No pre-registration amendment, no new experiment, no change to the locked `earnings_drift_pead` hypothesis or any knowledge-graph gate.

## Outcome: validation check failed — bootstrap was not run

Per the task's own explicit branching instruction, the reconstructed daily return series' Sharpe was checked against the reported 6.49 **before** running any bootstrap. It does not match — not closely, and not even in sign. Per instruction, this stops the exercise at that point rather than proceeding to quantify uncertainty around a series that doesn't reproduce the number in question, and rather than adjusting the reconstruction until it does.

## Step 0 — Does a daily return series already exist?

No. Checked directly, not assumed:
- `data/store.duckdb`'s `experiments` table schema holds only aggregate columns (`sharpe`, `max_drawdown`, `win_rate`, `cagr`, `trades`, `monte_carlo_p_value`, etc.) — structurally incapable of holding a time series.
- The one PEAD row (`experiment_id = PEAD_PILOT_Q4_2023_sue_srw_longshort_q4_2023_1783703388.375924`) has a `params` JSON blob with per-run configuration and summary stats only: `{"n_events": 481, "sue_threshold": 0, "hold_days": 20, "stop_loss": 0.1, "cost_bps": 5.0, "dsr": ..., "long_sharpe": 6.488725272627753, "short_sharpe": -6.643468683190577, ...}` — confirms the reported figure (6.49) and the `hold_days`/`stop_loss` conventions used, but again no series.
- No pilot script survives anywhere: not in git history, not untracked on disk, not referenced in any of the `pipeline_run*.log` files. This matches the earlier finding that the pilot was never version-controlled.
- The "488-event population" and "0/312 verifiable events" contamination check (`docs/edgar_coverage_universe_finding.md` line 43) also has **no surviving reconstruction script or intermediate artifact** — only its written conclusion remains. The task's instruction to "reuse that exact reconstruction" could not be followed literally, because it does not exist as a reusable artifact. Instead, Step 1 rebuilt the population directly from the current canonical `features/engineer_pead.py` functions and the raw `data/pead_store.duckdb` tables, using the experiment's own logged `params` as the ground-truth configuration (`hold_days=20`, `stop_loss=0.1`, `sue_threshold=0`) — this is disclosed as a substitution, not presented as the original artifact.

## Step 1 — Reconstruction

**Population**: `pead_universe` (`passes_all_filters=TRUE`, `source='simfin'`, the only source that existed at pilot time) joined to `pead_sue` (non-null `sue_srw`), `publish_date` in `[2023-10-01, 2023-12-31]`: **488 events total** — matching the task's own stated population size. Long leg (`sue_srw > 0`): **231 events**. Short leg: 257.

**Note on a population discrepancy, disclosed rather than silently resolved**: the experiment's logged `params.n_events` is 481 (both legs), not 488. The 18-row `pead_exclusions` table does not explain the gap (none of its Q4-2023-dated rows are in the passing-filters set to begin with). This 7-event (1.4%) difference was not fully traced — plausibly the original pilot ran against a slightly different price-data extract with fewer trailing days available for a few late-quarter events (`compute_forward_returns` silently drops events without enough forward price rows). This is disclosed rather than investigated further, since the validation check below fails by a far larger margin than 1.4% could explain either way.

**Entry/hold/exit conventions — confirmed from `features/engineer_pead.py`, not assumed:**
- Entry (`engineer_pead.py:113-122, 183-188`): the first trading day **strictly after** `publish_date`, entering at that day's **open** (`entry_price = opens[entry_idx]`) — **not T+1 close**, contrary to the task prompt's guess. This was checked against the actual source, not assumed.
- Hold period (`engineer_pead.py:107`, confirmed against the logged `params.hold_days = 20`): 20 trading days, unless a direction-aware stop-loss (10%, `stop_loss_pct=0.10`) triggers first (`engineer_pead.py:190-206`).
- `gross_return = direction * (exit_price - entry_price) / entry_price` (`engineer_pead.py:207`); long leg has `direction=+1` throughout, so no sign flip is needed.

Ran `compute_forward_returns()` (imported directly and unmodified) on the 231 long-leg events against `pead_prices`: **228 events produced a forward return** (3 dropped for insufficient trailing price data — consistent with the note above). Per-event `gross_return`: mean +6.56%, std 14.85% (n=228).

**Daily path reconstruction per event** (to preserve the actual overlapping-position structure rather than resampling events independently):
- Entry day: `(close − open) / open` on the entry date.
- Every subsequent day through the exit day: ordinary close-to-close return, `(close_k − close_{k-1}) / close_{k-1}`.
- **Validated, not assumed**: chaining these multiplicatively telescopes exactly to `(exit_price − entry_price) / entry_price`. Checked directly against every event's own `gross_return` from `compute_forward_returns()` — **0 mismatches across all 228 events** (tolerance 1e-8).

**Portfolio blend**: for each calendar trading day, the portfolio return is the equal-weighted mean of that day's return across every event with an open position that day. Result: **60 trading days**, 2023-10-05 to 2023-12-29, with `n_open_positions` ranging from 1 (the first and a few thin early/late days) to 161 (mid-quarter, mean 64.9). **Zero gap days** — every trading day in the series' own date range had at least one long-leg position open, so no zero-return/exclusion decision was actually needed in practice (disclosed as the choice that would have applied, per instruction, even though it never triggered).

## Validation Check: FAILED

| | Value |
|---|---|
| Reconstructed daily series | mean = −0.2677%, std = 3.03% (ddof=1), n = 60 |
| Reconstructed annualized Sharpe (mean/std × √252, ddof=1) | **−1.401** |
| Reconstructed annualized Sharpe (ddof=0, matching `backtest/evaluate.py::calculate_sharpe`'s exact convention) | **−1.413** |
| Originally reported `long_sharpe` | **+6.4887** |
| Delta | **≈ −7.9, sign-discordant** |

This is not "reasonably close." Two alternate hypotheses for how 6.49 might have been computed were checked and ruled out:
- Per-event Sharpe, annualized by trade frequency (`√(252/20)` on the 228 event-level `gross_return`s): **1.568** — still far off.
- Per-event Sharpe, annualized as if each event were one daily observation (`√252` directly on event-level returns — methodologically inconsistent with "daily," but checked anyway since it's cheap): **7.01** — closer in magnitude, but not an exact match, and not the "continuous daily portfolio series" method the task asked for regardless.

**Likely mechanism, offered as a diagnostic, not a fix**: the reconstructed series' most extreme days are concentrated exactly where `n_open_positions` is smallest. The single first day (2023-10-05, 1 position open) returned −5.15%; 2023-10-12 (1 position) returned −13.44%; 2023-10-18 (2 positions) returned −11.56%. These thin, high-variance early days get **equal weight** as "one day's portfolio return" alongside mid-quarter days with 100+ simultaneously open, well-diversified positions. That equal-per-calendar-day weighting is a legitimate, literal reading of "the same shape as a normal strategy's daily P&L" (per the task's own instruction) — but it evidently is not how the reported 6.49 was actually produced, since a data set with a clearly positive mean per-event return (+6.56%) turns Sharpe-negative under it. Concluding what construction *does* reproduce 6.49 would require the original pilot script, which does not survive (Step 0) — this is reported as an open, unresolved discrepancy, not guessed at further.

## Steps 2 and 3 — Not Executed

Per the task's explicit instruction, the stationary block bootstrap (Step 2) and its sanity checks (Step 3) were **not run**. Running a bootstrap on a series whose own point-estimate Sharpe doesn't match the number the exercise was supposed to explain would produce a confidence interval around the wrong quantity, dressed up with the appearance of rigor. No bootstrapped_sharpes, no 95% interval, no standard error are reported here, because none would mean what they'd appear to mean.

## Position

**This does not support or undercut the "treat with extreme suspicion" concern raised in the five-counsel PEAD review — it sidesteps the question entirely, and that itself is informative.** The concern was about whether a Sharpe of 6.49 from a single-quarter pilot is statistically believable. This exercise cannot answer that, because the daily series constructed by the most direct, source-verified reading of the pilot's own stated conventions doesn't reproduce 6.49 in the first place — it produces a negative number. That is arguably a *more* serious finding than a wide-but-positive confidence interval would have been: it means the reported 6.49 rests on a return-aggregation methodology that isn't recoverable from the current codebase and data alone, not just on a small sample. The original "extreme suspicion" framing was about sample size; this finding adds a second, independent reason for suspicion — the number can't currently be reproduced by any straightforward reconstruction of the strategy's own daily P&L, faithful to its own documented entry/hold/exit rules. Recommended next step (not undertaken here, out of scope for a read-only exercise): before any further reliance on 6.49, either recover the actual original aggregation method (unlikely, given the script is gone) or treat 6.49 as unverified pending a from-scratch, newly pre-registered re-derivation — which is a decision for a separate session, not something to resolve by adjusting this reconstruction until it matches.
