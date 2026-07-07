from dataclasses import dataclass, field
from datetime import datetime
from typing import List


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
    end_date: datetime = datetime(2023, 12, 31)
    duckdb_path: str = "data/store.duckdb"
    regime_lookback: int = 90
    max_signal_combinations: int = 200
    fdr_alpha: float = 0.05
    monte_carlo_shuffles: int = 2000
