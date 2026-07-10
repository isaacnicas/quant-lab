import numpy as np

from config import Config, IN_SAMPLE_END
from data.ingest import assert_discovery_in_sample, fetch_and_cache, load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.generate import generate_signals
from backtest.evaluate import run_backtest, strategy_returns
from validation.robustness import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    fdr_correction,
    monte_carlo_test,
    regime_consistency_check,
)
from journal.log_experiment import log_experiment


def run_pipeline(config: Config) -> None:
    # fetch_and_cache is NOT bounded by IN_SAMPLE_END: caching data through
    # config.end_date (today) is harmless and useful for later evaluation of
    # already-pre-registered hypotheses. Only the discovery READ below is
    # bounded -- see data/ingest.py's assert_discovery_in_sample().
    fetch_and_cache(config.tickers, config.start_date, config.end_date, config.duckdb_path)

    # The sweep's declared trial budget, knowable in advance -- used for DSR
    # regardless of how many tickers actually complete. Using a count that
    # shrinks after the fact (e.g. only tickers that had enough history) is
    # exactly the "launder the denominator" failure mode DSR exists to catch.
    n_trials_total = len(config.tickers) * config.max_signal_combinations

    all_records = []
    backtest_calls = 0

    # ---- stage 1 (per ticker): MC + backtest + regime for every signal, ----
    # ---- then FDR correction within that ticker's own family of tests.  ----
    for ticker in config.tickers:
        # Discovery must never see data past IN_SAMPLE_END -- this is the
        # project's only clean 2024-2025 holdout, and once discovery touches
        # it, it is permanently spent. Bounding by IN_SAMPLE_END here (not
        # config.end_date) is the primary fix; assert_discovery_in_sample()
        # below is the structural backstop that catches it even if this call
        # site is ever changed back by mistake.
        prices = load_prices([ticker], config.start_date, IN_SAMPLE_END, config.duckdb_path)
        assert_discovery_in_sample(prices, IN_SAMPLE_END)
        if prices.empty or len(prices) < 260:
            print(f"Skipping {ticker}: insufficient history")
            continue

        featured = build_features(prices)
        featured["regime"] = label_regime(featured)

        feature_dict = {col: featured[col].to_numpy() for col in featured.columns if col.endswith("_z") or col == "rsi"}
        signals = generate_signals(feature_dict, config.max_signal_combinations)

        ticker_mc, ticker_metrics, ticker_returns, ticker_consistent, p_values = [], [], [], [], []
        for signal in signals:
            # ticker= derives a deterministic per-(ticker, signal) seed inside
            # monte_carlo_test (validation/robustness.py) -- same pair always
            # gets the same p-value on the same data, different signals get
            # different seeds, and re-ingesting new price data naturally
            # changes the result.
            mc = monte_carlo_test(signal, featured, config.monte_carlo_shuffles, ticker=ticker)
            metrics = run_backtest(signal, featured)
            backtest_calls += 1
            # Same net-of-cost return series run_backtest computed internally
            # (via the shared _compute_trade_returns path) -- needed here for
            # DSR's skew/kurtosis/T, not recomputed a third way.
            returns = strategy_returns(signal, featured)
            consistent = regime_consistency_check(signal, featured, ["bull", "bear"])

            ticker_mc.append(mc)
            ticker_metrics.append(metrics)
            ticker_returns.append(returns)
            ticker_consistent.append(consistent)
            p_values.append(mc["p_value"])

        # FDR is per-ticker: it controls the false discovery rate within a
        # family of tests on the SAME asset. DSR (stage 2/3 below, computed
        # globally after ALL tickers) corrects the winner's curse of
        # selecting the max across the ENTIRE search -- a different
        # correction at a different scope. This distinction is the whole
        # reason FDR passed things DSR kills: the QQQ signal looked
        # FDR-significant (p=0.0015) because FDR only ever compared it to
        # QQQ's other 199 signals, never to the full 3,600-trial sweep it
        # was actually selected from.
        significance_flags = fdr_correction(p_values, config.fdr_alpha)

        for signal, mc, metrics, returns, consistent, significant in zip(
            signals, ticker_mc, ticker_metrics, ticker_returns, ticker_consistent, significance_flags
        ):
            all_records.append({
                "ticker": ticker,
                "signal": signal,
                "metrics": metrics,
                "returns": returns,
                "mc_p_value": mc["p_value"],
                "fdr_significant": significant,
                "regime_consistent": consistent,
            })
        print(f"{ticker}: evaluated {len(signals)} signals")

    assert backtest_calls == len(all_records), (
        f"backtest_calls ({backtest_calls}) != len(all_records) ({len(all_records)}) -- "
        f"every signal must be backtested exactly once per run."
    )

    # ---- stage 2 (after ALL tickers): pool the Sharpe distribution across ----
    # ---- the entire sweep and compute SR0 once, against the declared     ----
    # ---- n_trials_total, not the per-ticker count.                      ----
    all_sharpes = np.array([
        r["metrics"]["Sharpe"] for r in all_records if np.isfinite(r["metrics"]["Sharpe"])
    ])
    mean_sharpe_trials = float(np.mean(all_sharpes)) if len(all_sharpes) else float("nan")
    std_sharpe_trials = float(np.std(all_sharpes, ddof=0)) if len(all_sharpes) else float("nan")
    sr0 = expected_max_sharpe(n_trials_total, mean_sharpe_trials, std_sharpe_trials)

    # ---- stage 3: DSR per record against that single shared SR0, then log ----
    gate_counts = {"mc": 0, "fdr": 0, "regime": 0, "dsr": 0, "all_four": 0}
    for record in all_records:
        dsr_result = deflated_sharpe_ratio(
            record["metrics"]["Sharpe"], record["returns"], n_trials_total,
            mean_sharpe_trials, std_sharpe_trials,
        )
        dsr_value = dsr_result["dsr"]
        dsr_significant = bool(np.isfinite(dsr_value) and dsr_value > 0.95)

        log_experiment(
            config.duckdb_path, record["signal"], record["ticker"], record["metrics"],
            monte_carlo_p_value=record["mc_p_value"],
            fdr_significant=record["fdr_significant"],
            regime_consistent=record["regime_consistent"],
            sharpe_deflated=dsr_value,
            dsr_significant=dsr_significant,
            params={"sr0": dsr_result["sr0"], "n_trials": n_trials_total},
        )

        mc_pass = record["mc_p_value"] < 0.05
        fdr_pass = bool(record["fdr_significant"])
        regime_pass = bool(record["regime_consistent"])
        gate_counts["mc"] += mc_pass
        gate_counts["fdr"] += fdr_pass
        gate_counts["regime"] += regime_pass
        gate_counts["dsr"] += dsr_significant
        gate_counts["all_four"] += mc_pass and fdr_pass and regime_pass and dsr_significant

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total experiments logged:      {len(all_records)}")
    print(f"n_trials (declared budget):    {n_trials_total}")
    print(f"Trial Sharpe distribution:     mean={mean_sharpe_trials:.4f}, std={std_sharpe_trials:.4f}")
    print(f"SR0 (expected max under null): {sr0:.4f}")
    print(f"Passing MC (p<0.05):           {gate_counts['mc']}")
    print(f"Passing FDR:                   {gate_counts['fdr']}")
    print(f"Passing regime consistency:    {gate_counts['regime']}")
    print(f"Passing DSR (>0.95):           {gate_counts['dsr']}")
    print(f"Passing ALL FOUR gates:        {gate_counts['all_four']}")


if __name__ == "__main__":
    run_pipeline(Config())
