import hashlib
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

# Pre-registration fields are intentionally NOT in _ALLOWED_FIELDS: they must
# only ever be set by pre_register() (which computes and locks prereg_hash),
# never by the generic update_mechanism() partial-update path. That's what
# makes prereg_locked meaningful -- if update_mechanism() could silently
# overwrite test_plan, the "immutable once frozen" guarantee would be fake.

# test_plan JSON schema (all fields required, no nulls -- a pre-registration
# that doesn't state what would falsify it is not a pre-registration):
#   {
#     "signal_spec":                "<exact signal string, e.g. 'rsi < 30'>",
#     "ticker":                     "<ticker, or list>",
#     "predicted_sharpe_direction": "positive" | "negative",
#     "predicted_mechanism":        "<one-sentence economic rationale>",
#     "n_trials_declared":          <int>,  # hypotheses in THIS batch
#     "gates": {"dsr_threshold": 0.95, "pbo_threshold": 0.5, "fdr_alpha": 0.05},
#     "stopping_rule":              "<what result would kill this hypothesis>",
#     "entry_rule":                 "<e.g. T+1 open>",
#     "exit_rule":                  "<e.g. 5% stop, next-day close>",
#   }
_REQUIRED_TEST_PLAN_FIELDS = [
    "signal_spec", "ticker", "predicted_sharpe_direction", "predicted_mechanism",
    "n_trials_declared", "gates", "stopping_rule", "entry_rule", "exit_rule",
]
_REQUIRED_GATE_FIELDS = ["dsr_threshold", "pbo_threshold", "fdr_alpha"]


def _validate_test_plan(test_plan: Dict[str, Any]) -> None:
    missing = [f for f in _REQUIRED_TEST_PLAN_FIELDS if test_plan.get(f) in (None, "", [], {})]
    if missing:
        raise ValueError(f"test_plan is missing required field(s): {missing}")
    gates = test_plan.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("test_plan['gates'] must be a dict")
    missing_gates = [f for f in _REQUIRED_GATE_FIELDS if gates.get(f) is None]
    if missing_gates:
        raise ValueError(f"test_plan['gates'] is missing required field(s): {missing_gates}")
    if test_plan["predicted_sharpe_direction"] not in ("positive", "negative"):
        raise ValueError("test_plan['predicted_sharpe_direction'] must be 'positive' or 'negative'")


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
            updated_at TIMESTAMP,
            pre_registration_date TIMESTAMP,
            test_plan JSON,
            prereg_hash VARCHAR,
            prereg_locked BOOLEAN
        )
    """)
    # Non-destructive migration for a mechanisms table created before
    # pre-registration existed -- CREATE TABLE IF NOT EXISTS above is a no-op
    # there, so the existing seeded rows (and all their data) survive; this
    # just backfills the new columns (NULL on old rows).
    for column, col_type in [
        ("pre_registration_date", "TIMESTAMP"), ("test_plan", "JSON"),
        ("prereg_hash", "VARCHAR"), ("prereg_locked", "BOOLEAN"),
    ]:
        conn.execute(f"ALTER TABLE mechanisms ADD COLUMN IF NOT EXISTS {column} {col_type}")


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

    # INSERT OR REPLACE replaces the whole row -- preserve any existing
    # pre-registration so re-adding/reseeding a mechanism_id can never
    # silently destroy a locked test_plan. Only pre_register() may set
    # these four columns.
    existing = conn.execute(
        "SELECT pre_registration_date, test_plan, prereg_hash, prereg_locked "
        "FROM mechanisms WHERE mechanism_id = ?",
        [mechanism_id],
    ).fetchone()
    prereg_date, test_plan, prereg_hash, prereg_locked = existing if existing else (None, None, None, False)

    now = datetime.now(timezone.utc)
    conn.execute(
        """
        INSERT OR REPLACE INTO mechanisms VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            prereg_date,
            test_plan,
            prereg_hash,
            prereg_locked,
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


def pre_register(duckdb_path: str, mechanism_id: str, test_plan: Dict[str, Any]) -> str:
    """Freeze a test plan for a mechanism. A hypothesis selected from a
    threshold sweep and a hypothesis specified in advance from an economic
    mechanism are different objects, even if the signal string is identical
    -- only the second has an honest n_trials. This is what makes that
    distinction checkable: once frozen, prereg_hash is a tamper-evident
    fingerprint of the exact plan (see verify_prereg), and prereg_locked
    blocks re-registration.

    Raises ValueError if the mechanism doesn't exist, is already locked, or
    the test_plan is missing any required field (see the schema documented
    above _ensure_table)."""
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    row = conn.execute(
        "SELECT prereg_locked FROM mechanisms WHERE mechanism_id = ?", [mechanism_id]
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No mechanism with id {mechanism_id!r}; add it first via add_mechanism()")
    if row[0]:
        conn.close()
        raise ValueError(
            f"Mechanism {mechanism_id!r} is already pre-registered (prereg_locked=True) -- "
            f"a pre-registration is immutable once frozen. Register a new mechanism_id for "
            f"a revised hypothesis instead of altering this one."
        )

    _validate_test_plan(test_plan)

    plan_json = json.dumps(test_plan, sort_keys=True)
    prereg_hash = hashlib.sha256(plan_json.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    conn.execute(
        """
        UPDATE mechanisms SET
            test_plan = ?, prereg_hash = ?, pre_registration_date = ?,
            prereg_locked = TRUE, status = 'pre_registered', updated_at = ?
        WHERE mechanism_id = ?
        """,
        [plan_json, prereg_hash, now, now, mechanism_id],
    )
    conn.close()
    return prereg_hash


def verify_prereg(duckdb_path: str, mechanism_id: str, test_plan: Dict[str, Any]) -> bool:
    """Recompute the hash from the supplied plan and compare it to the stored
    prereg_hash. False means the plan differs from what was frozen at
    registration -- the tamper check."""
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    row = conn.execute(
        "SELECT prereg_hash FROM mechanisms WHERE mechanism_id = ?", [mechanism_id]
    ).fetchone()
    conn.close()
    if row is None or row[0] is None:
        return False
    recomputed = hashlib.sha256(json.dumps(test_plan, sort_keys=True).encode()).hexdigest()
    return recomputed == row[0]


def assert_pre_registered(duckdb_path: str, mechanism_id: str) -> None:
    """Raise unless the mechanism has prereg_locked=True and a non-null
    prereg_hash. This is the gate the (not-yet-built) hypothesis compiler
    must call before testing anything -- left unused here on purpose; its
    existence is what makes the compiler unable to be built without it."""
    conn = duckdb.connect(duckdb_path)
    _ensure_table(conn)
    row = conn.execute(
        "SELECT prereg_locked, prereg_hash FROM mechanisms WHERE mechanism_id = ?", [mechanism_id]
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"No mechanism with id {mechanism_id!r}.")
    locked, prereg_hash = row
    if not locked or prereg_hash is None:
        raise ValueError(
            f"Mechanism {mechanism_id!r} is not pre-registered (prereg_locked={locked}, "
            f"prereg_hash={prereg_hash!r}). Call pre_register() with a complete test_plan "
            f"before testing this hypothesis -- an unregistered hypothesis has no honest "
            f"n_trials and cannot be evaluated."
        )


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
    "pre_register",
    "verify_prereg",
    "assert_pre_registered",
]
