"""
Deflated Sharpe Ratio (DSR) + Probability of Backtest Overfitting (PBO) for
the QQQ candidate signal, applied to the EXISTING 3,600-experiment result in
data/store.duckdb. Must be run before any future change narrows the search
width (fewer tickers/signals), since that would change the trial counts DSR
depends on and make this result stale.

Not part of the main pipeline loop; run directly with:

    PYTHONPATH=. python validate_dsr_pbo.py

Reuses the exact same masking/lagging/cost machinery as run_backtest (via
signals.apply and backtest.evaluate) rather than reimplementing it, per the
project's single-path principle.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

from config import Config
from data.ingest import load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.apply import compute_signal_mask, lag_mask
from backtest.evaluate import (
    _coerce_price_series,
    apply_stop_loss,
    apply_transaction_costs,
    calculate_returns,
    calculate_sharpe,
)
from validation.robustness import deflated_sharpe_ratio, probability_of_backtest_overfitting

TICKER = "QQQ"
CANDIDATE_SIGNAL = "ret_1d_z < 1.0 AND rsi < 30"
STOP_LOSS = 0.05
POSITION_SIZE = 0.01
COST_BPS = 5.0
PBO_N_SPLITS = 16


def build_net_returns(featured: pd.DataFrame, signal: str) -> pd.Series:
    """Same masking/lagging/cost pipeline run_backtest uses -- reused, not
    reimplemented (signals.apply + backtest.evaluate are the single source
    of truth for this)."""
    price_series = _coerce_price_series(featured)
    raw_mask = compute_signal_mask(featured, signal)
    raw_mask = apply_stop_loss(raw_mask, price_series, STOP_LOSS)
    trade_mask = lag_mask(raw_mask, price_series.index).to_numpy()
    gross_returns = calculate_returns(trade_mask, price_series, POSITION_SIZE)
    return apply_transaction_costs(gross_returns, trade_mask, COST_BPS, POSITION_SIZE)


def main() -> None:
    cfg = Config()

    if not os.path.exists(cfg.duckdb_path):
        print(f"ERROR: {cfg.duckdb_path} does not exist. Run `PYTHONPATH=. python main.py` "
              f"first to generate the experiments table, then rerun this script.")
        sys.exit(1)

    conn = duckdb.connect(cfg.duckdb_path)
    total_experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    if total_experiments == 0:
        conn.close()
        print("ERROR: experiments table is empty. Run `PYTHONPATH=. python main.py` first, "
              "then rerun this script.")
        sys.exit(1)

    all_sharpes = conn.execute("SELECT sharpe FROM experiments WHERE sharpe IS NOT NULL").fetchdf()["sharpe"].to_numpy()
    qqq_df = conn.execute(
        "SELECT signal, sharpe FROM experiments WHERE ticker = ? AND sharpe IS NOT NULL ORDER BY signal",
        [TICKER],
    ).fetchdf()
    candidate_row = conn.execute(
        "SELECT sharpe FROM experiments WHERE ticker = ? AND signal = ?",
        [TICKER, CANDIDATE_SIGNAL],
    ).fetchone()
    conn.close()

    if candidate_row is None:
        print(f"ERROR: candidate signal not found for {TICKER}: {CANDIDATE_SIGNAL!r}")
        sys.exit(1)
    candidate_sharpe_db = float(candidate_row[0])

    n_trials_full = int(total_experiments)
    n_trials_ticker = int(len(qqq_df))
    mean_sharpe_full, std_sharpe_full = float(np.mean(all_sharpes)), float(np.std(all_sharpes, ddof=0))
    mean_sharpe_ticker, std_sharpe_ticker = float(qqq_df["sharpe"].mean()), float(qqq_df["sharpe"].std(ddof=0))

    print(f"Population sizes: full={n_trials_full}, {TICKER}-only={n_trials_ticker}")
    print(f"Full population Sharpe: mean={mean_sharpe_full:.4f}, std={std_sharpe_full:.4f}")
    print(f"{TICKER}-only Sharpe:    mean={mean_sharpe_ticker:.4f}, std={std_sharpe_ticker:.4f}")
    print()

    # Rebuild real return series -- reusing run_backtest's own machinery -- for
    # the candidate (DSR skew/kurtosis/T) and for all 200 QQQ signals (PBO matrix).
    prices = load_prices([TICKER], cfg.start_date, cfg.end_date, cfg.duckdb_path)
    featured = build_features(prices)
    featured["regime"] = label_regime(featured)

    candidate_returns = build_net_returns(featured, CANDIDATE_SIGNAL)
    candidate_sharpe_recomputed = calculate_sharpe(candidate_returns)
    if not np.isclose(candidate_sharpe_recomputed, candidate_sharpe_db, atol=1e-6):
        print(f"WARNING: recomputed Sharpe ({candidate_sharpe_recomputed:.6f}) does not match "
              f"the value stored in the DB ({candidate_sharpe_db:.6f}) -- using the DB value.")
    observed_sharpe = candidate_sharpe_db

    dsr_ticker = deflated_sharpe_ratio(
        observed_sharpe, candidate_returns, n_trials_ticker, mean_sharpe_ticker, std_sharpe_ticker,
    )
    dsr_full = deflated_sharpe_ratio(
        observed_sharpe, candidate_returns, n_trials_full, mean_sharpe_full, std_sharpe_full,
    )

    print(f"DSR at n_trials={n_trials_ticker} (QQQ's own signal pool only):")
    print(f"  {dsr_ticker}")
    print(f"DSR at n_trials={n_trials_full} (full cross-ticker population):")
    print(f"  {dsr_full}")
    print()
    print("Which is honest: the candidate was selected by scanning ALL 18 tickers x 200")
    print("signals -- it wasn't just competing against QQQ's other 199 signals, it was")
    print("selected as the single best across the ENTIRE 3,600-trial sweep. The n_trials=200")
    print(f"figure (DSR={dsr_ticker['dsr']:.4f}) understates the true selection pressure;")
    print(f"n_trials={n_trials_full} (DSR={dsr_full['dsr']:.4f}) is the honest number.")
    print()

    # PBO across QQQ's 200-signal candidate set.
    all_qqq_signals = qqq_df["signal"].tolist()
    returns_columns = []
    for signal in all_qqq_signals:
        r = build_net_returns(featured, signal)
        returns_columns.append(r.to_numpy())
    returns_matrix = np.column_stack(returns_columns)

    pbo_result = probability_of_backtest_overfitting(returns_matrix, n_splits=PBO_N_SPLITS)

    print(f"PBO across {TICKER}'s {returns_matrix.shape[1]} candidate signals "
          f"({returns_matrix.shape[0]} periods, {pbo_result['n_combinations']} combinations):")
    print(f"  PBO = {pbo_result['pbo']:.4f}")
    print()

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Observed net Sharpe (annualized):        {observed_sharpe:.4f}")
    print(f"SR0 at n_trials={n_trials_ticker} (QQQ-only, annualized):   {dsr_ticker['sr0']:.4f}")
    print(f"SR0 at n_trials={n_trials_full} (full population, annualized): {dsr_full['sr0']:.4f}")
    print(f"DSR at n_trials={n_trials_ticker}:  {dsr_ticker['dsr']:.4f}  (clears 0.95? {dsr_ticker['dsr'] > 0.95})")
    print(f"DSR at n_trials={n_trials_full}: {dsr_full['dsr']:.4f}  (clears 0.95? {dsr_full['dsr'] > 0.95})")
    print(f"PBO: {pbo_result['pbo']:.4f}  (clears < 0.5? {pbo_result['pbo'] < 0.5})")
    print()

    honest_dsr = dsr_full["dsr"]
    survives_dsr = honest_dsr > 0.95
    survives_pbo = pbo_result["pbo"] < 0.5

    if survives_dsr and survives_pbo:
        verdict = (f"The QQQ candidate SURVIVES selection-bias correction: DSR={honest_dsr:.4f} "
                    f"clears the 0.95 threshold even at the honest n_trials={n_trials_full}, and "
                    f"PBO={pbo_result['pbo']:.4f} is below 0.5.")
    elif not survives_dsr and survives_pbo:
        verdict = (f"The QQQ candidate DOES NOT SURVIVE selection-bias correction: at the honest "
                    f"n_trials={n_trials_full}, DSR={honest_dsr:.4f} does NOT clear the 0.95 "
                    f"threshold -- once you account for the fact that this signal was picked as "
                    f"the best of {n_trials_full} trials, its Sharpe is not distinguishable from "
                    f"what the best of {n_trials_full} pure-noise trials would produce. PBO="
                    f"{pbo_result['pbo']:.4f} is still below 0.5, so the signal isn't flagged as "
                    f"actively worse than random selection, but DSR is the stricter and decisive "
                    f"test here, and it fails.")
    elif survives_dsr and not survives_pbo:
        verdict = (f"Mixed result: DSR={honest_dsr:.4f} clears 0.95 at n_trials={n_trials_full}, "
                    f"but PBO={pbo_result['pbo']:.4f} is >= 0.5, meaning the process that selected "
                    f"the IS-best signal from this candidate pool performs no better than chance "
                    f"OOS. Treat this as a fail: PBO measures something DSR does not (the actual "
                    f"selection process on this specific candidate set), and it's an unambiguous "
                    f"warning sign.")
    else:
        verdict = (f"The QQQ candidate FAILS both checks: DSR={honest_dsr:.4f} at n_trials="
                    f"{n_trials_full} does not clear 0.95, and PBO={pbo_result['pbo']:.4f} is >= "
                    f"0.5. This is a valid and valuable result, not a failure of the analysis -- "
                    f"it means the apparent edge is most likely an artifact of testing 3,600 "
                    f"hypotheses and picking the best one, not a real, exploitable effect.")

    print(verdict)


if __name__ == "__main__":
    main()
