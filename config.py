from dataclasses import dataclass, field
from datetime import datetime
from typing import List

# Fixed boundary of the original in-sample research window (2018-01-01
# through this date) that produced the audited 3,600-experiment result and
# that the 2024-2025 holdout scripts (validate_qqq_signal_holdout.py, etc.)
# are built around. Config.end_date below now rolls forward with the
# calendar for day-to-day pipeline runs -- it is intentionally NOT tied to
# this constant, so a future main.py run will pull in more recent data
# (including what was previously the 2024-2025 holdout window) rather than
# silently redefining what "holdout" has always meant in this project.
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
