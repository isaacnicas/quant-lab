# Research methodology: the 3-Stage Research Architecture

This document governs *how* research moves through this pipeline, as opposed
to `README.md`, which documents *what the pipeline is* (the
`data → features → regimes → signals → backtest → validation → journal` code
path) and the current QQQ-signal status. This is process discipline, not a
one-off decision for a single signal — it is the standing default for every
future research thread in `quant_lab`.

## Why this document exists

The pipeline's current `validation/robustness.py` step runs Monte Carlo,
FDR correction, and regime-consistency checks together, at scale, across the
full exploratory sweep (`main.py`: 10 tickers × 200 signals/ticker = 2,000
experiments/run, all gated simultaneously — see `README.md`'s "Current
status"). DSR was wired in as a permanent fourth gate alongside these
(commit `e55c0d8`). That is a defensible way to report a *finished* search
honestly — the QQQ survivor's FDR-significance verdict depends on being
judged against the full 200-signal width it was drawn from, not a width of
one, and the pipeline gets that part right.

It is a different thing to run FDR/DSR/PBO **while still exploring** — as a
filter that decides which candidates get refined further, rather than as the
final judgment on a search that is already complete. Applied too early, a
strict multiple-testing correction does two things at once, and neither is
good: it kills promising ideas before they have had a chance to be refined
into their best form (the correction is being paid against a search width
that hasn't stabilized yet), and it invites the search width itself to be
gamed retroactively (rerun a narrower sweep after seeing what worked, and the
correction looks better without the idea having gotten any more real). The
fix is not a better statistic — DSR and PBO are the right tools — it is
applying them at the right *stage*, once, at the end, against the honest
cumulative search width.

## The three stages

### 1. Exploration

**Goal:** generate and cheaply screen a wide set of candidate hypotheses.
**Gate:** loose — raw Sharpe, basic economic sense, does the signal even
produce trades. No multiple-testing correction applied here. The search
width is not final at this stage, so paying a correction against it is
premature; the point of Exploration is breadth, not certainty.
**Output:** a short list of candidates worth stressing, not a verdict on any
of them.

### 2. Stressing

**Goal:** find out how a candidate breaks, cheaply, before spending the
Confirmation-stage budget on it. Adversarial tests specific to the
candidate's own failure modes — not generic statistics:
- out-of-sample / holdout splits and walk-forward validation
  (`validation/robustness.py`'s `walk_forward_split()`);
- regime-consistency checks across the candidate's stated operating
  environment;
- clairvoyance / look-ahead-shift falsification — give the candidate
  unrealistic perfect foresight and check whether even that helps; if it
  doesn't, the concept is structurally broken, not just poorly implemented
  or laggy (this test's real precedent: the 2026-08-12 regime-engine
  clairvoyance test in the `TrendFollowing` project — both live regime
  engines showed negligible improvement even under perfect 2-day-early
  foresight, confirming a structural rather than latency failure; see
  `quant-portfolio/docs/factor_timing_equity_carry_schedule_review.md`);
- mechanism-specific stress tests where the candidate has a known failure
  mode (e.g. a yield-carry candidate's yield-trap / quality filter — see
  `quant-portfolio/docs/equity_carry_prereg.md`'s Gate 1).
**Gate:** candidate-specific pre-registered pass/fail criteria, written
*before* the stress test is run. A candidate that fails here is killed or
honestly relabeled (e.g. "this is a value tilt, not an independent carry
premium" — again, `equity_carry_prereg.md`'s Gate 2) — it does not proceed
to Confirmation regardless of how promising Exploration made it look.
**Output:** a much shorter list of candidates that have survived an honest
attempt to kill them on their own specific terms.

### 3. Confirmation

**Goal:** the final, statistically rigorous verdict — the only stage where
DSR and PBO belong.
**Gate:** DSR / PBO computed against the **honest, cumulative** search width
— every candidate actually tried across Exploration for this line of
research, not just the survivor being tested. `n_shuffles=2000`,
SHA-256-derived seeds (bug 14's fix), pre-registered significance
thresholds. FDR correction, if still relevant at this stage, uses the same
honest width.
**Output:** a genuine go/no-go — live/paper deployment decision, or a
recorded, dated kill in `research-falsification-log`.

## The rule this replaces

Default habit without this document: run DSR/PBO/FDR together, once, at
whatever point in the pipeline they're easiest to call — usually right after
the backtest step, before any dedicated stress-testing has happened at all.
That is what the current `main.py` sweep does, and it is not wrong for
reporting a finished 2,000-signal search honestly. It is wrong as a *habit*
for narrower, single-mechanism research threads (a single new sleeve, a
single new gate) where Exploration and Confirmation are not naturally the
same step — collapsing them either kills real ideas too early or lets a
retroactively-narrowed search width flatter a result that hasn't actually
been stressed. From this document forward, new research threads default to
the three explicit stages above; collapsing Stressing into Confirmation (or
skipping it) requires a stated reason, not silence.
