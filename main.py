from config import Config, IN_SAMPLE_END
from data.ingest import assert_discovery_in_sample, fetch_and_cache, load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.generate import generate_signals
from backtest.evaluate import run_backtest
from validation.robustness import monte_carlo_test, fdr_correction, regime_consistency_check
from journal.log_experiment import log_experiment


def run_pipeline(config: Config) -> None:
    # fetch_and_cache is NOT bounded by IN_SAMPLE_END: caching data through
    # config.end_date (today) is harmless and useful for later evaluation of
    # already-pre-registered hypotheses. Only the discovery READ below is
    # bounded -- see data/ingest.py's assert_discovery_in_sample().
    fetch_and_cache(config.tickers, config.start_date, config.end_date, config.duckdb_path)

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

        mc_results, p_values = [], []
        for signal in signals:
            # ticker= derives a deterministic per-(ticker, signal) seed inside
            # monte_carlo_test (validation/robustness.py) -- same pair always
            # gets the same p-value on the same data, different signals get
            # different seeds, and re-ingesting new price data naturally
            # changes the result.
            mc = monte_carlo_test(signal, featured, config.monte_carlo_shuffles, ticker=ticker)
            mc_results.append(mc)
            p_values.append(mc["p_value"])

        significance_flags = fdr_correction(p_values, config.fdr_alpha)

        for signal, mc, significant in zip(signals, mc_results, significance_flags):
            metrics = run_backtest(signal, featured)
            consistent = regime_consistency_check(signal, featured, ["bull", "bear"])
            log_experiment(
                config.duckdb_path, signal, ticker, metrics,
                monte_carlo_p_value=mc["p_value"],
                fdr_significant=significant,
                regime_consistent=consistent,
            )
        print(f"{ticker}: logged {len(signals)} experiments")


if __name__ == "__main__":
    run_pipeline(Config())
