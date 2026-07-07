import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import duckdb


def _ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id VARCHAR PRIMARY KEY,
            signal VARCHAR,
            ticker VARCHAR,
            sharpe DOUBLE,
            max_drawdown DOUBLE,
            win_rate DOUBLE,
            cagr DOUBLE,
            trades INTEGER,
            monte_carlo_p_value DOUBLE,
            fdr_significant BOOLEAN,
            regime_consistent BOOLEAN,
            params JSON,
            created_at TIMESTAMP
        )
    """)


def log_experiment(
    duckdb_path: str,
    signal: str,
    ticker: str,
    backtest_metrics: Dict[str, float],
    monte_carlo_p_value: Optional[float] = None,
    fdr_significant: Optional[bool] = None,
    regime_consistent: Optional[bool] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Log one experiment's full result set. Returns the experiment_id."""
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)

    experiment_id = f"{ticker}_{signal}_{datetime.now(timezone.utc).timestamp()}"
    conn.execute(
        """
        INSERT OR REPLACE INTO experiments VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            experiment_id,
            signal,
            ticker,
            backtest_metrics.get("Sharpe"),
            backtest_metrics.get("Max Drawdown"),
            backtest_metrics.get("Win Rate"),
            backtest_metrics.get("CAGR"),
            backtest_metrics.get("Trades"),
            monte_carlo_p_value,
            None if fdr_significant is None else bool(fdr_significant),
            None if regime_consistent is None else bool(regime_consistent),
            json.dumps(params or {}),
            datetime.now(timezone.utc),
        ],
    )
    conn.close()
    return experiment_id
