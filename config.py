from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "NVDA", "META", "NFLX"
    ])
    start_date: datetime = datetime(2018, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)
    duckdb_path: str = "data/store.duckdb"
    regime_lookback: int = 90
    max_signal_combinations: int = 200
    fdr_alpha: float = 0.05
    monte_carlo_shuffles: int = 2000
