import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd

_PROVENANCE_SOURCE_FILES = [
    "features/engineer.py",
    "signals/apply.py",
    "backtest/evaluate.py",
    "validation/robustness.py",
]

_provenance_cache: Optional[Dict[str, str]] = None


def _git_commit() -> str:
    """Short SHA of HEAD, with a '-dirty' suffix if the tree has uncommitted
    changes. Never raises -- git being unavailable or this not being a repo
    yields 'unknown' rather than failing the pipeline."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
        return f"{sha}-dirty" if status.strip() else sha
    except Exception:
        return "unknown"


def _code_hash() -> str:
    """sha256 of the concatenated source of the files that actually determine
    an experiment's outcome (feature engineering, signal masking/lagging,
    backtest, and validation). Missing files are skipped rather than crashing
    the pipeline -- code_hash then reflects whatever subset was readable."""
    digest = hashlib.sha256()
    for relative_path in _PROVENANCE_SOURCE_FILES:
        try:
            with open(relative_path, "rb") as f:
                digest.update(f.read())
        except OSError:
            continue
    return digest.hexdigest()


def _get_provenance() -> Dict[str, str]:
    """Computed once per process (not once per experiment) and cached -- these
    values are identical for every row a single pipeline run logs."""
    global _provenance_cache
    if _provenance_cache is None:
        _provenance_cache = {
            "git_commit": _git_commit(),
            "code_hash": _code_hash(),
            "python_version": sys.version.split()[0],
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
    return _provenance_cache


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
            created_at TIMESTAMP,
            git_commit VARCHAR,
            code_hash VARCHAR,
            python_version VARCHAR,
            pandas_version VARCHAR,
            numpy_version VARCHAR
        )
    """)
    # ALTER ... ADD COLUMN IF NOT EXISTS covers the case where `experiments`
    # already exists from a prior run with the old (pre-provenance) schema --
    # CREATE TABLE IF NOT EXISTS above is a no-op there, so the existing rows
    # (and all their data) are preserved; this backfills the new columns
    # (NULL on old rows) without destroying anything.
    for column, col_type in [
        ("git_commit", "VARCHAR"), ("code_hash", "VARCHAR"),
        ("python_version", "VARCHAR"), ("pandas_version", "VARCHAR"),
        ("numpy_version", "VARCHAR"),
    ]:
        conn.execute(f"ALTER TABLE experiments ADD COLUMN IF NOT EXISTS {column} {col_type}")


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

    provenance = _get_provenance()
    experiment_id = f"{ticker}_{signal}_{datetime.now(timezone.utc).timestamp()}"
    conn.execute(
        """
        INSERT OR REPLACE INTO experiments VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            provenance["git_commit"],
            provenance["code_hash"],
            provenance["python_version"],
            provenance["pandas_version"],
            provenance["numpy_version"],
        ],
    )
    conn.close()
    return experiment_id
