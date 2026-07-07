import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

_ALLOWED_FIELDS = {
    "name", "mechanism", "asset_class", "variables", "direction", "horizon",
    "source_type", "source_ref", "evidence_strength", "status",
    "experiment_ids", "notes",
}
_JSON_FIELDS = {"variables", "experiment_ids"}


def _ensure_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mechanisms (
            mechanism_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            mechanism VARCHAR,
            asset_class VARCHAR,
            variables JSON,
            direction VARCHAR,
            horizon VARCHAR,
            source_type VARCHAR,
            source_ref VARCHAR,
            evidence_strength VARCHAR,
            status VARCHAR,
            experiment_ids JSON,
            notes VARCHAR,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    """)


def add_mechanism(duckdb_path: str, **fields: Any) -> str:
    """Insert (or replace) one mechanism row. Requires mechanism_id in fields;
    all other schema columns are optional and default to None/empty."""
    mechanism_id = fields.get("mechanism_id")
    if not mechanism_id:
        raise ValueError("add_mechanism requires a mechanism_id field")
    unknown = set(fields) - _ALLOWED_FIELDS - {"mechanism_id"}
    if unknown:
        raise ValueError(f"Unknown mechanism field(s): {sorted(unknown)}")

    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    now = datetime.now(timezone.utc)
    conn.execute(
        """
        INSERT OR REPLACE INTO mechanisms VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            mechanism_id,
            fields.get("name"),
            fields.get("mechanism"),
            fields.get("asset_class"),
            json.dumps(fields.get("variables") or []),
            fields.get("direction"),
            fields.get("horizon"),
            fields.get("source_type"),
            fields.get("source_ref"),
            fields.get("evidence_strength"),
            fields.get("status", "untested"),
            json.dumps(fields.get("experiment_ids") or []),
            fields.get("notes"),
            now,
            now,
        ],
    )
    conn.close()
    return mechanism_id


def update_mechanism(duckdb_path: str, mechanism_id: str, **fields: Any) -> None:
    """Partial update of an existing mechanism row. created_at is untouched;
    updated_at is bumped to now."""
    if not fields:
        return
    unknown = set(fields) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unknown mechanism field(s): {sorted(unknown)}")

    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    set_clauses, params = [], []
    for key, value in fields.items():
        if key in _JSON_FIELDS:
            value = json.dumps(value or [])
        set_clauses.append(f"{key} = ?")
        params.append(value)
    set_clauses.append("updated_at = ?")
    params.append(datetime.now(timezone.utc))
    params.append(mechanism_id)

    conn.execute(
        f"UPDATE mechanisms SET {', '.join(set_clauses)} WHERE mechanism_id = ?",
        params,
    )
    conn.close()


def link_experiments(duckdb_path: str, mechanism_id: str, experiment_ids: List[str]) -> None:
    """Merge experiment_ids into a mechanism's existing list (union, deduped),
    rather than overwriting it."""
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    row = conn.execute(
        "SELECT experiment_ids FROM mechanisms WHERE mechanism_id = ?", [mechanism_id]
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No mechanism with id {mechanism_id!r}; add it first via add_mechanism()")

    existing = json.loads(row[0]) if row[0] else []
    merged = sorted(set(existing) | set(experiment_ids))
    conn.execute(
        "UPDATE mechanisms SET experiment_ids = ?, updated_at = ? WHERE mechanism_id = ?",
        [json.dumps(merged), datetime.now(timezone.utc), mechanism_id],
    )
    conn.close()


def query_mechanisms(
    duckdb_path: str,
    status: Optional[str] = None,
    asset_class: Optional[str] = None,
    evidence_strength: Optional[str] = None,
) -> pd.DataFrame:
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    conditions, params = [], []
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    if asset_class is not None:
        conditions.append("asset_class = ?")
        params.append(asset_class)
    if evidence_strength is not None:
        conditions.append("evidence_strength = ?")
        params.append(evidence_strength)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    df = conn.execute(
        f"SELECT * FROM mechanisms {where_clause} ORDER BY created_at", params
    ).fetchdf()
    conn.close()
    return df


def get_untested(duckdb_path: str) -> pd.DataFrame:
    return query_mechanisms(duckdb_path, status="untested")


__all__ = [
    "add_mechanism",
    "update_mechanism",
    "link_experiments",
    "query_mechanisms",
    "get_untested",
]
