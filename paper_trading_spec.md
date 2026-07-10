# Paper Trading Spec — QQQ `ret_1d_z < 1.0 AND rsi < 30`

> ## ⚠️ KILLED 2026-07-09 — DO NOT DEPLOY
>
> This candidate was killed by Deflated Sharpe Ratio (DSR) selection-bias
> correction: **DSR = 0.042 at n_trials = 3,600** (the honest number, since
> this signal was selected as the best of the full 18-ticker × 200-signal
> sweep, not just competing against its own 200 QQQ variants) — well below
> the noise ceiling. Observed net Sharpe 1.284 does not even clear
> **SR0 = 1.817**, the expected maximum Sharpe that 3,600 pure-noise trials
> would produce by chance. DSR fails even at the more forgiving n_trials=200
> (0.170). This is the exact failure mode DSR exists to catch: the signal
> looked FDR-significant in isolation (p=0.0015) precisely because FDR
> controls expected false discoveries across a batch but does not correct
> the winner's-curse bias of having picked the single best of thousands of
> trials.
>
> **No capital or paper trades should be deployed against this signal.**
> The mechanism record (`rsi_oversold_bounce` in the knowledge graph,
> `journal/knowledge_graph.py`) is marked `status="killed"` with this
> evidence in its notes. See `validate_dsr_pbo.py` for the full computation.
>
> The rest of this document is **retained as a template** for the next
> candidate — specifically, one that goes through pre-registration
> (`journal/knowledge_graph.py`'s `pre_register()`/`test_plan` schema)
> *before* being tested, so its n_trials is honest by construction rather
> than reconstructed after the fact.

Status: live experiment, not a certified strategy. See `README.md` for the
full backtest/audit history behind this signal.

## Signal definition

**Rule:** `ret_1d_z < 1.0 AND rsi < 30`, evaluated on QQQ daily closes.

**Feature definitions** (both computed in `features/engineer.py`):
- `ret_1d_z` — 20-day rolling z-score of the 1-day return.
- `rsi` — 14-period RSI (Wilder's formulation as implemented in
  `features/engineer.py`).

**Evaluation and entry timing:** the signal is evaluated on day T's close
(both `ret_1d_z` and `rsi` use only data through day T). The trade is
entered at day T+1's open — a 1-bar lag, matching `signals/apply.py`'s
`apply_signal_lagged()`. Do not enter on day T's close; do not evaluate the
signal intraday.

**Exit:** the position is held for as long as the signal condition remains
true on each subsequent day's close (re-evaluated daily, same 1-bar-lag
convention); exit at the open of the first day the condition is no longer
true, or on a stop-loss trigger (below), whichever comes first.

## Position sizing

Two sizing options — pick one before starting, do not switch mid-experiment:

**Conservative (recommended starting point): 1% of portfolio per trade.**
This is the exact sizing used in the backtest (`position_size=0.01`), so
live results are directly comparable to the reported Sharpe and drawdown
without any rescaling.

**Moderate: 5% of portfolio per trade.**
Still well within the tested drawdown envelope (full-allocation Max
Drawdown was -10.33% in-sample / -4.11% holdout; at 5% sizing, realized
drawdown scales to roughly -0.5% / -0.2% of total portfolio).

**Important — CAGR does not scale the way you'd expect.** The
full-allocation CAGR (~10%/year) assumes 100% of the portfolio is deployed
on every trade, which never actually happens (trades are rare and brief —
~13/year historically). At 1% sizing, the realized contribution to
portfolio CAGR is approximately 0.1%/year; at 5% sizing, approximately
0.5%/year. This is the realistic expectation at prudent sizing — this
signal is not a portfolio return driver, it is a low-frequency, high-win-rate
edge sized as a small satellite position.

## Stop loss

5% stop on the position, matching the backtest's `stop_loss=0.05` parameter.
Evaluated at next-day close, not intraday — do not exit on an intraday
touch of the 5% level; check only at the close of each day following entry,
and exit if that close is ≥5% below the entry price.

## Trade logging

Every paper trade must be recorded before moving to the next. Log to
`paper_trades.csv` with exactly these columns:

```
ticker, entry_date, entry_price, exit_date, exit_price, gross_return, net_return, win, notes
```

- `gross_return`: `(exit_price / entry_price) - 1`
- `net_return`: `gross_return` minus round-trip transaction costs (use
  5bps = 0.0005 total, matching the backtest's `cost_bps=5.0`, unless your
  actual broker costs are known and materially different — if so, use the
  real cost and note it in `notes`)
- `win`: `1` if `net_return > 0`, else `0`

After every trade, update two running statistics (compute from all rows in
`paper_trades.csv` so far):
- **Running win rate**: wins ÷ total trades so far.
- **Running Sharpe**: only start reporting this once there are **5 or more**
  trades logged (below 5 trades, the number is too noisy to be meaningful —
  record it as `n/a` in your own notes, not in the CSV).

## Kill criteria (pre-committed — do not change once trading starts)

- **Pause** if live win rate drops below **50%** over any rolling 20 trades.
- **Pause** if live Sharpe drops below **0** over any rolling 20 trades.
- **Pause** if a single trade's loss exceeds **3x the average historical
  win** (average historical win = mean of positive `net_return` values
  across the 53 in-sample + holdout backtest trades).
- **Review (not an automatic pause)** if 6 months pass with fewer than 5
  trades logged — this signal fires ~13 times/year historically, so this
  low a count suggests either bad luck or a possible regime change, not
  necessarily a broken signal.

**Resume criteria:** after any pause, do not resume automatically. Resume
only after explicitly reviewing whether market structure has changed
(volatility regime, QQQ's mean-reversion characteristics, etc.). Write a
dated note in `paper_trades.csv`'s `notes` column (on a zero-row placeholder
entry if needed) documenting the review and the decision before any new
trade is logged.

## Confidence context

**What the evidence supports:** a Monte Carlo p-value of ≈0.0015 (stable,
SHA-256-seeded, 2,000 shuffles) — the effect is unlikely to be pure noise
on its own. Performance is consistent across all 6 in-sample years
(2018-2023), including a positive result in the 2022 bear market. Holdout
performance (2024-2025) is in line with in-sample: Sharpe 0.96 on 10
discrete trades, 80% win rate.

**What the evidence does not support:** FDR certification against the
200-candidate search width it was drawn from (p≈0.0015 does not clear the
≈0.00025 threshold Benjamini-Hochberg requires for the best-ranked
hypothesis out of 200 at alpha=0.05). High statistical confidence at
n=53 discrete trades total — this is a small sample by any standard.

**Expected confidence timeline:** meaningful additional live evidence after
roughly 40-50 more discrete trades, which at the historical firing rate of
~13 signals/year is approximately **3-4 years** of paper trading.

**This is a live experiment, not a certified strategy.** Size and monitor
it accordingly.

## Next review date

Formal review after the **first 20 live trades or 12 months**, whichever
comes first. At that review: reconcile live win rate and Sharpe against
the backtest numbers above, and decide whether to continue as-is, resize,
or kill the experiment.
