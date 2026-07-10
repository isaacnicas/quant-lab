from dataclasses import dataclass, field
from datetime import datetime
from typing import List

# IN_SAMPLE_END bounds DISCOVERY. end_date (below, on Config) bounds
# EVALUATION. They are not interchangeable -- do not use one where the other
# is called for.
#
# IN_SAMPLE_END is frozen and never rolls: it is the fixed boundary of the
# original in-sample research window (2018-01-01 through this date) that
# produced the audited 3,600-experiment result. Discovery (signal generation,
# Monte Carlo, FDR, backtest, log_experiment -- see main.py's discovery guard)
# must NEVER see data past this date; the 2024-2025 window is this project's
# only clean holdout, and once discovery touches it, it is permanently spent.
# Config.end_date rolls forward with the calendar so out-of-sample EVALUATION
# of already-pre-registered hypotheses can use fresh data over time.
IN_SAMPLE_END = datetime(2023, 12, 31)


@dataclass
class Config:
    # Original 10 (2 broad-market ETFs + 8 mega-cap stocks) plus 8 factor/
    # sector/cross-asset ETFs added in Phase 2 to test mechanisms across more
    # market regimes: IWM (small cap), DIA (Dow), GLD (gold), TLT (20yr
    # Treasury), XLE/XLF/XLK/XLV (energy/financial/tech/healthcare sectors).
    # Sector ETFs are liquid enough for the transaction cost model to stay
    # realistic and represent meaningfully different regimes/mechanisms than
    # SPY/QQQ; GLD and TLT add cross-asset coverage. No new individual stocks
    # were added -- liquidity and data quality are harder to control there,
    # and mechanism testing on ETFs is cleaner.
    tickers: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "NVDA", "META", "NFLX",
        "IWM", "DIA", "GLD", "TLT", "XLE", "XLF", "XLK", "XLV",
    ])
    start_date: datetime = datetime(2018, 1, 1)
    # Rolls forward to "today" on every fresh process run (field(default_factory=...)
    # is required here, not a bare `= datetime.today()` default, since a plain
    # dataclass field default is evaluated once at class-definition time and
    # would otherwise freeze at whatever moment this module was first imported).
    # See IN_SAMPLE_END above for the fixed 2018-2023 boundary the existing
    # audit/holdout results depend on.
    end_date: datetime = field(default_factory=datetime.today)
    duckdb_path: str = "data/store.duckdb"
    regime_lookback: int = 90
    max_signal_combinations: int = 200
    fdr_alpha: float = 0.05
    monte_carlo_shuffles: int = 2000
