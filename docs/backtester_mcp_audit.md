# backtester-mcp Audit: DSR and PBO Comparison
Date: 2026-07-13
Library version: backtester-mcp 0.1.0
Branch: fix/audit-lag-drawdown-trades

## DSR Comparison

**Custom (`validation/robustness.py:deflated_sharpe_ratio`)** implements the full Bailey & Lopez de Prado (2014) formula in general form:
- `expected_max_sharpe(n_trials, mean_sharpe, std_sharpe)` accepts an arbitrary trial-population mean/std, not just the null case — `E[max] = mean + std * ((1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne)))`.
- Skew and kurtosis are computed directly from the actual per-period return series (`scipy.stats.skew`, `kurtosis(fisher=False)`), not assumed.
- All three Sharpe inputs (`observed_sharpe`, `mean_sharpe_trials`, `std_sharpe_trials`) are de-annualized (÷√252) before the formula is applied, since the formula's √(T-1) term and skew/kurtosis correction are derived on per-period Sharpe; `sr0` is re-annualized only for reporting.
- Returns `"dsr"` as **Φ(z) — a probability in [0,1]**, matching the literature's actual definition (Prob[SR* > SR0]). The production gate in `main.py` checks `dsr > 0.95`.

**Library (`backtester_mcp.robustness.deflated_sharpe`)**:
- `expected_max_sharpe` **hardcodes mean=0, std=1** — there is no parameter to supply an arbitrary trial-population mean/std. This is the null-hypothesis special case only, not the general form.
- Skew/kurtosis are **optional scalar arguments defaulting to 0.0/3.0** (normal distribution) — the function never computes them from data itself; the caller must pass real values manually or silently get the normal-distribution assumption.
- No de-annualization step, and the docstring does not state whether `observed_sharpe` should be annualized or per-period — this is genuinely ambiguous from the source.
- Returns `"dsr"` as the **raw Z-score/test-statistic** (not a probability), plus a separate `"p_value"` field equal to `1 - Φ(test_stat)`. So `Φ(test_stat) = 1 - p_value` is the quantity actually comparable to the custom implementation's `"dsr"` — the two libraries use the *same variable name* for two different quantities.

**Test Case A (synthetic, 252 obs, n_trials=3600, mean=0, std=1):** Custom DSR=0.0000499, library (all three variants: naive/matched-skew-kurt/de-annualized, converted to probability) = 0.0000 each, delta ≈ 5e-5 — within tolerance. **This agreement is not very informative**: the realized synthetic Sharpe (-0.3116, a below-average draw for this seed) sits so far below SR0 (3.60) that every variant of the library call collapses to the same "obviously zero" answer regardless of the formula differences above. It does not exercise the actual points of divergence (arbitrary mean/std, real skew/kurtosis, annualization convention) in a way that would show up in the final rounded number.

**Test Case C (real QQQ candidate signal, `ret_1d_z < 1.0 AND rsi < 30`, mechanism_id `rsi_oversold_bounce`):** Using the real empirical population (`mean_sharpe_full=0.0320, std_sharpe_full=0.4954`, n_trials=3600), custom DSR = **0.0146** — confirmed to 9 decimal places against `validate_dsr_pbo.py`'s own live authoritative output, run independently. *(Note: an earlier reference value of DSR=0.042 for this same mechanism reflects the original 2026-07-07 sweep; the experiments table was fully regenerated 2026-07-09–07-10 against a later `end_date` — `Config.end_date` rolls forward by design — producing a new population and a new honest DSR. 0.0146 is the current, correct ground truth, not a stale or buggy result.)* The library **cannot reproduce this at all**: because it hardcodes mean=0/std=1, it has no way to use the real population's mean=0.032/std=0.4954, and its best-effort output (matched skew/kurtosis, annualized Sharpe as given) rounds to a probability of 0.0000 — not because of a formula-precision difference, but because the library is structurally incapable of representing this population.

**Verdict: DIVERGE.** The disagreement is not numerical noise — the library's `expected_max_sharpe` is a strict special case (mean=0, std=1 only) of what the custom implementation supports, and this special case does not match how `main.py` actually computes DSR in production (empirical mean/std of `all_sharpes`, not 0/1). The library is structurally unable to reproduce the production gate's real behavior.

## PBO Comparison

**Custom (`probability_of_backtest_overfitting`)**: exhaustive Combinatorially Symmetric Cross-Validation — all C(16,8)=12,870 combinations for n_splits=16. Blocks built via `np.array_split` (near-equal sizes, e.g. 252 periods → twelve 16-blocks + four 15-blocks — **all periods used**). Relative OOS rank scaled by `N+1` (`rank_position/(N+1)`) specifically to avoid `logit(0)`/`logit(1)`. `pbo = mean(logit ≤ 0)`.

**Library (`backtester_mcp.robustness.pbo`)**: same CSCV concept, n_splits=16 supported, but two structural differences: (1) blocks are built via `block_size = n_periods // n_splits` contiguous slicing — for T=252, n_splits=16, this gives `block_size=15`, using only 240 of 252 periods and **silently dropping the last 12**. (2) Combinations are **capped at a random subsample of 500** (`rng.choice(len(combos), 500, replace=False)`, library's own hardcoded seed=42) rather than the full 12,870 — a stochastic approximation, not exhaustive enumeration. Rank scaling uses raw `N` (not `N+1`); the final PBO decision is a direct `rank/N > 0.5` threshold on unclipped `w`, computed separately from the logit array (which is only used for the returned `logits` diagnostic, with `np.clip(w, 0.01, 0.99)` applied there only).

**Test Case B (synthetic 252×50 pure-noise matrix, n_splits=16):** Custom PBO=0.5924 (12,870 combinations), library PBO=0.5920 (500 combinations), delta=0.0004 — within tolerance for this instance. This single-seed agreement is genuine, but it does not validate the structural differences: the library's PBO estimate carries irreducible run-to-run sampling variance from subsampling only 500 of 12,870 combinations (mitigated only by its own hardcoded seed=42, not a designed reproducibility guarantee), and its contiguous block construction silently discards data whenever `T` is not evenly divisible by `n_splits` — a property that will matter more, not less, on datasets with less favorable divisibility than this test case's.

**Verdict: DIVERGE.** Close numerical agreement on one synthetic instance does not offset that the library's PBO is a subsampled approximation (not exhaustive) with silent data truncation on uneven splits — a materially different reliability profile for a live gating decision than the custom implementation's exhaustive, no-data-loss method.

## Recommendation

**KEEP — DIVERGED.**

**Root cause of divergence:** backtester-mcp 0.1.0 is a small (3 GitHub stars, single PyPI release), general-purpose backtesting toolkit built with simplified, self-contained defaults — not a drop-in replacement engineered to this project's specific requirements. For DSR, it hardcodes the null-hypothesis special case (mean=0, std=1) where the custom implementation supports the general form `main.py` actually needs (empirical trial-population mean/std). For PBO, it trades exhaustive combinatorial coverage and full data usage for speed (500-combination subsample, contiguous-block truncation) — a reasonable default for a lightweight general tool, but not compatible with this project's exhaustive, reproducible, no-data-loss requirements for a live gating decision.

**Action: none required.** Both custom implementations are retained as-is. This is a single, unified root cause (the library was not built for this project's generality/exhaustiveness requirements) explaining both divergences — not two independent issues — so no partial replacement applies.

## Files
- `compare_pbo_dsr.py` and `compare_summary.json` — stay in `C:\QuantTrading\backtester-mcp-audit\`, not committed to quant_lab.
