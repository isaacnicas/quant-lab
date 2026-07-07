# Quant Lab — Alpha Discovery Pipeline

A rule-based signal research pipeline: fetch price data, engineer features,
label market regimes, generate combinatorial rule-based signals, backtest
each one, validate with Monte Carlo + FDR + regime-consistency checks, and
journal every experiment to DuckDB.

```
data → features → regimes → signals → backtest → validation (MC/FDR/regime) → journal (DuckDB)
```

- `data/ingest.py` — yfinance fetch, DuckDB `prices` cache
- `features/engineer.py` — rolling returns, volatility, RSI(14), MACD, z-scores
- `regimes/detect.py` — bull/bear x high/low-vol regime labels
- `signals/generate.py` — up to 200 rule-based signal strings per ticker
- `signals/apply.py` — shared signal-masking + 1-bar-lag primitives, used by
  every consumer below (this is the module bug #6 introduced to stop the
  masking logic from drifting across three separate implementations)
- `backtest/evaluate.py` — `run_backtest()`: lagged entry, stop-loss,
  transaction costs, Sharpe/CAGR/drawdown/win-rate
- `validation/robustness.py` — `monte_carlo_test()`, `fdr_correction()`,
  `regime_consistency_check()`, `walk_forward_split()`
- `journal/` — `log_experiment()` / `query_experiments()` against
  `data/store.duckdb`'s `experiments` table
- `main.py` — runs all of the above across `Config.tickers` (10 tickers,
  2018-01-01 to 2023-12-31, 200 signals/ticker = 2,000 experiments/run)

## Current status

One signal is the standout of the 2,000-signal sweep: **QQQ,
`ret_1d_z < 1.0 AND rsi < 30`**. It has real, consistent economics across
6 in-sample years and a genuine 2024-2025 holdout. It does **not** survive
Benjamini-Hochberg FDR correction against the 200-signals-per-ticker search
width it was drawn from.

**Honest characterization: promising signal with strong holdout consistency;
not FDR-certified at current search width.**

### Final reconciled numbers (post-audit, 2000 MC shuffles, SHA-256 seeded)

| Metric | In-sample (2018-2023) | Holdout (2024-2025) |
|---|---|---|
| Gross Sharpe | 1.366 | 1.011 |
| Net Sharpe (5bps cost) | 1.284 | 0.964 |
| Discrete trades | 43 | 10 |
| Trade-level win rate | 83.7% | 80.0% |
| Full-allocation Net CAGR | 10.5% | 9.9% |
| Full-allocation Max Drawdown | -10.33% | -4.11% |
| MC p-value | 0.0015 (stable) | N/A — holdout isn't run through the MC/FDR gate |
| FDR-significant | **False** | N/A |

Total discrete trades across both periods: **53** (not the ~101 previously
assumed, which counted days-in-position rather than discrete trades — see
bug #13).

Across the full 2,000-experiment sweep: **0 signals are FDR-significant**,
and **0 signals clear all three gates** (Sharpe > 1, FDR-significant,
regime-consistent) simultaneously. The QQQ signal above is the closest —
lowest p-value of any of its ticker's 200 candidates — but 0.0015 doesn't
clear the ≈0.00025 bar Benjamini-Hochberg requires for the best-ranked
hypothesis out of 200 at alpha=0.05.

See `paper_trading_spec.md` for how this signal is being tracked live.

## Bug log

Numbered in the order each was found and fixed during development of this
pipeline.

**Bug 1 — yfinance dtype mismatch broke the price schema.**
`test_fetch_and_cache` failed: `close` round-tripped as `float32` instead of
`float64`. Root cause was two-layered: yfinance returned `float32` OHLC
columns, *and* the DuckDB `prices` table declared `close FLOAT` (32-bit) —
so even casting the DataFrame to `float64` before insert wouldn't have
survived, since DuckDB truncates on insert to the column's declared width
regardless of source dtype. Fixed by casting to `float64` in
`_reshape_to_long()` **and** changing the table schema to `DOUBLE`. Verified:
7/7 tests pass.

**Bug 2 — `numpy.bool_` crashed DuckDB inserts.**
The pipeline crashed on the very first experiment logged, before anything
was written: `_duckdb.NotImplementedException: Unable to transform python
value of type 'numpy.bool' to DuckDB LogicalType`. `fdr_significant` and
`regime_consistent` were `numpy.bool_`, which DuckDB's Python binding can't
bind directly. Fixed with an explicit `bool(...)` cast in
`log_experiment()`. Verified: full pipeline ran to completion, 2,000
experiments logged (was 0).

**Bug 3 — `monte_carlo_test()` ignored the `signal` argument entirely.**
It computed Sharpe on the raw, unconditional return series and shuffled
that same series for the null distribution — shuffling doesn't change a
series' mean/std, so the "null" was statistically identical to "actual"
every time, driving p-values to ~1.0 regardless of which signal was passed
in. Verified: one ticker's 200 different signals produced as few as 1
distinct p-value across all of them. Fixed by masking returns with the
signal before computing Sharpe and shuffling.

**Bug 4 — `regime_consistency_check()` had the identical bug.**
Same root cause — it never masked returns by the signal, just checked raw
buy-and-hold returns in bull/bear windows. Verified: 1 distinct
`regime_consistent` value per ticker across all 200 of that ticker's
signals. Fixed the same way as bug 3.

**Bug 5 — self-contradictory generated signals.**
`signals/generate.py` could produce combinations like
`ret_1d_z > 1.0 AND ret_1d_z < 1.0` (impossible, always 0 trades) — 70 of
2,000 experiments in one run. Fixed by skipping any combination that
references the same feature more than once.

**Bug 6 — signal-lag inconsistency across the three consumers.**
`run_backtest` correctly lagged trade entry by one bar (via
`calculate_returns`'s internal shift); the bug-3/4 fixes above had
`monte_carlo_test`/`regime_consistency_check` mask returns by signal but
**without** the lag — reintroducing same-bar circularity for contemporaneous
features like `ret_1d_z` (a signal partly restating the very return it's
measured against). Fixed by extracting one shared module,
`signals/apply.py` (`compute_signal_mask`, `apply_signal_lagged`,
`lag_mask`), used by all three consumers instead of three independent
copies. Verified: FDR-significant count across the 2,000-experiment sweep
dropped from **1,096 to 6**; a circular signal's Sharpe (`ret_1d_z > 1.0`)
dropped from ~22 to ~-1.

**Bug 7 — transaction-cost/position_size unit mismatch (caught before
shipping).** The first transaction-cost implementation subtracted a flat
`cost_bps` from position-scaled returns without scaling the cost by
`position_size` — since Sharpe is scale-invariant to `position_size` but a
flat bps charge isn't, this made net Sharpe implode to -3.3 for a signal
whose gross Sharpe was +1.37, a pure units artifact. Fixed by scaling the
per-transition cost by `position_size` to match the units of the return
series it's charged against.

**Bug 8 — `calculate_win_rate()` used the wrong denominator.**
It divided winning days by total calendar days in the series — since ~95%
of days aren't in a trade at all (return = 0, not counted as a win), this
structurally produced tiny win rates (2-7%) regardless of how good the
signal actually was. Fixed to compute wins ÷ discrete trades via
`group_trades()`. Verified: the QQQ survivor's real trade-level win rate is
75-100% per year, not 2-7%.

**Bug 9 — CAGR was only ever reported position-scaled.**
At `position_size=0.01`, CAGR values were ~4e-6 — technically correct but
uninterpretable. Added a second, full-allocation CAGR (position_size
divided back out before compounding) alongside the existing position-scaled
figure, both labeled explicitly.

**Bug 10 — CAGR's annualization exponent used raw day count, not years.**
`calculate_cagr` used `final ** (1/len(returns)) - 1`, treating the row
count as if it were a year count — for a ~1,500-row series this crushed any
real compounding to near-zero regardless of position sizing. Fixed to
compute actual elapsed years from the date span (`dates`), falling back to
a 252-trading-day approximation when dates aren't available. Verified: the
QQQ survivor's full-allocation CAGR is genuinely ~3-23%/year across
in-sample years, not ~0.03-0.08% as previously reported.

**Bug 11 — `walk_forward_split()` was never updated in the bug-6 fix.**
It still called the unlagged `compute_signal_mask()` directly, reintroducing
the exact circularity bug 6 eliminated everywhere else. Verified before fix:
`ret_1d_z > 0.5` mean walk-forward Sharpe was 31.04 (10 windows); `rsi < 30`
was -19.06. Fixed with the same one-line change as bug 6
(`apply_signal_lagged`, lag computed within each walk-forward window only).
Verified after fix: `ret_1d_z > 0.5` collapsed to 0.13; `rsi < 30` flipped to
**+21.98** — a sign flip, not "roughly unchanged" as originally hypothesized
(unlagged RSI<30 measures the down-day that caused the oversold reading;
lagged, it measures the bounce that follows).

**Bug 12 — Max Drawdown was scaled by `position_size`, unlike Sharpe.**
Verified: `position_size=0.01` gave Max DD -0.00108; `position_size=1.0` gave
-0.10330 for the identical signal with identical Sharpe (1.283625 both
times) — a ~100x difference purely from an arbitrary sizing convention,
making risk look far smaller than it is. Fixed: primary `"Max Drawdown"` is
now computed on full-allocation (unscaled) returns, matching the existing
full-allocation CAGR convention; position-scaled values are kept under
explicit `"(Position-Scaled)"` keys. Verified after fix: both `position_size`
values now report identically (-0.103301).

**Bug 13 — "Trades" counted raw days-in-position, not discrete trades.**
`run_backtest` reported `int(np.sum(raw_mask))` — pre-lag days the signal
condition was true, not the number of distinct trades. Verified: for the
QQQ survivor, `raw_mask.sum()=78`, `trade_mask.sum()=78`, but
`len(group_trades(trade_mask))=43` — the real count is roughly half.
Fixed: `"Trades"` now reports `len(group_trades(trade_mask))`; `"Days In
Position"` added as a separate field for the old meaning. Corrected sample
size: **43 in-sample + 10 holdout = 53 discrete trades**, not ~101.

**Bug 14 — Monte Carlo p-values weren't reproducible run-to-run.**
`monte_carlo_test` used `np.random.permutation()` with no fixed seed and
only 500 shuffles; at the resolution that implies (1/500 = 0.002), the QQQ
survivor's p-value bounced between 0.000 and 0.006 across identical calls —
meaning its FDR pass/fail status was decided by sampling noise, not the
signal's actual properties. Fixed in two parts: (1) raised
`n_shuffles` to 2,000 (std across repeated calls dropped from ~0.0026 to
~0.00052); (2) added an optional `seed` parameter using an isolated
`np.random.default_rng(seed)` (never touches global random state), with
`main.py` deriving a deterministic per-`(ticker, signal)` seed. The
originally-planned `hash((ticker, signal))` was caught and rejected during
implementation — Python randomizes string hashing per-process
(`PYTHONHASHSEED`) by default, so the same tuple hashes to a different value
on every separate run, verified empirically with two independent
`python -c` invocations. Replaced with a SHA-256-based seed, verified stable
across processes. **Result of the fix:** the QQQ survivor's p-value is now
stable at **0.0015**, and FDR-significant is now **False** — the previously
reported "6 significant" and "1 signal clears all three gates" were
themselves artifacts of 500-shuffle sampling noise. With accurately-resolved
Monte Carlo tests, **0 of the 2,000 experiments are FDR-significant.**
