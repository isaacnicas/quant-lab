import duckdb
import pandas as pd


def query_experiments(
    duckdb_path: str,
    min_sharpe: float = 1.0,
    fdr_significant_only: bool = True,
    min_regime_consistent: bool = True,
) -> pd.DataFrame:
    """Return experiments passing the promotion bar: Sharpe, FDR significance, regime consistency."""
    conn = duckdb.connect(duckdb_path)
    conditions = [f"sharpe > {min_sharpe}"]
    if fdr_significant_only:
        conditions.append("fdr_significant = TRUE")
    if min_regime_consistent:
        conditions.append("regime_consistent = TRUE")
    where_clause = " AND ".join(conditions)
    df = conn.execute(f"""
        SELECT * FROM experiments
        WHERE {where_clause}
        ORDER BY sharpe DESC
    """).fetchdf()
    conn.close()
    return df


__all__ = ["query_experiments"]
