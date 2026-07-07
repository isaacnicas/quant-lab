import hashlib

from config import Config
from data.ingest import fetch_and_cache, load_prices
from features.engineer import build_features
from regimes.detect import label_regime
from signals.generate import generate_signals
from backtest.evaluate import run_backtest
from validation.robustness import monte_carlo_test, fdr_correction, regime_consistency_check
from journal.log_experiment import log_experiment


def _deterministic_seed(ticker: str, signal: str) -> int:
    """Python's built-in hash() randomizes str hashing per-process (PYTHONHASHSEED),
    so hash((ticker, signal)) gives a DIFFERENT value on every run -- defeating
    the point of a reproducible seed. SHA-256 is stable across processes."""
    digest = hashlib.sha256(f"{ticker}|{signal}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def run_pipeline(config: Config) -> None:
    fetch_and_cache(config.tickers, config.start_date, config.end_date, config.duckdb_path)

    for ticker in config.tickers:
        prices = load_prices([ticker], config.start_date, config.end_date, config.duckdb_path)
        if prices.empty or len(prices) < 260:
            print(f"Skipping {ticker}: insufficient history")
            continue

        featured = build_features(prices)
        featured["regime"] = label_regime(featured)

        feature_dict = {col: featured[col].to_numpy() for col in featured.columns if col.endswith("_z") or col == "rsi"}
        signals = generate_signals(feature_dict, config.max_signal_combinations)

        mc_results, p_values = [], []
        for signal in signals:
            # Deterministic per-(ticker, signal) seed: same pair always gets the
            # same p-value on the same data, different signals get different
            # seeds, and re-ingesting new price data naturally changes the result.
            seed = _deterministic_seed(ticker, signal)
            mc = monte_carlo_test(signal, featured, config.monte_carlo_shuffles, seed=seed)
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
