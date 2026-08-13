"""
One-time seed for the research knowledge graph: 12 known anomalies as
status="untested", then promotes the QQQ survivor (rsi_oversold_bounce) to
status="live_paper" and links it to its experiment_id from the pipeline's
current `experiments` table. Not part of the main pipeline loop; run
directly with:

    PYTHONPATH=. python seed_knowledge_graph.py

Notes on fields not explicitly specified in each mechanism's description:
`horizon` and `direction` were inferred from each mechanism's standard
academic/practitioner characterization where not stated outright, and
`asset_class` reflects the classic setting each effect is documented in
(individual stocks for firm-level anomalies like PEAD/low-vol/momentum,
liquid ETFs for market-wide/regime effects). Adjust via update_mechanism()
if any of these judgment calls should be different.
"""
import duckdb

from knowledge_graph import add_mechanism, link_experiments, query_mechanisms, update_mechanism

DUCKDB_PATH = "data/store.duckdb"

MECHANISMS = [
    dict(
        mechanism_id="rsi_oversold_bounce",
        name="RSI oversold bounce",
        mechanism="Short-term mean reversion in liquid ETFs when RSI(14) < 30",
        asset_class="equity_etf",
        variables=["rsi"],
        direction="long",
        horizon="swing",
        source_type="academic",
        source_ref="observation + Jegadeesh (1990) short-term reversal",
        evidence_strength="moderate",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="momentum_12_1",
        name="12-1 month momentum",
        mechanism="12-month minus 1-month return predicts continuation",
        asset_class="equity_stock",
        variables=["ret_20d", "ret_5d"],
        direction="long",
        horizon="position",
        source_type="academic",
        source_ref="Jegadeesh & Titman (1993)",
        evidence_strength="strong",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="vol_regime_filter",
        name="Volatility regime filter",
        mechanism="High-volatility regimes predict poor forward returns for trend signals",
        asset_class="equity_etf",
        variables=["vol_60d_z", "vol_20d_z"],
        direction="either",
        horizon="position",
        source_type="academic",
        source_ref="Moreira & Muir (2017)",
        evidence_strength="strong",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="macd_crossover",
        name="MACD crossover",
        mechanism="MACD line crossing above signal line as momentum entry",
        asset_class="equity_etf",
        variables=["macd_line_z", "macd_signal_z"],
        direction="long",
        horizon="swing",
        source_type="blog",
        source_ref="practitioner/blog convention",
        evidence_strength="weak",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="mean_reversion_zscore",
        name="5-day return z-score reversal",
        mechanism="Extreme z-score in 5-day return predicts reversal",
        asset_class="equity_stock",
        variables=["ret_5d_z"],
        direction="either",
        horizon="swing",
        source_type="academic",
        source_ref="Lo & MacKinlay (1990)",
        evidence_strength="moderate",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="low_vol_anomaly",
        name="Low volatility anomaly",
        mechanism="Low-volatility stocks outperform on a risk-adjusted basis",
        asset_class="equity_stock",
        variables=["vol_20d_z", "vol_60d_z"],
        direction="long",
        horizon="position",
        source_type="academic",
        source_ref="Baker, Bradley & Wurgler (2011)",
        evidence_strength="strong",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="trend_following_ma",
        name="Trend-following moving average filter",
        mechanism="Price above long-term MA as trend filter",
        asset_class="equity_etf",
        variables=["ret_20d_z"],
        direction="long",
        horizon="position",
        source_type="academic",
        source_ref="Faber (2007)",
        evidence_strength="moderate",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="earnings_drift_pead",
        name="Post-earnings announcement drift",
        mechanism="Prices continue drifting in the direction of an earnings surprise",
        asset_class="equity_stock",
        variables=[],
        direction="either",
        horizon="swing",
        source_type="academic",
        source_ref="N/A -- deferred, needs earnings data",
        evidence_strength="untested",
        status="untested",
        notes="needs earnings feature -- deferred; requires earnings data not currently in pipeline",
    ),
    dict(
        mechanism_id="volatility_clustering",
        name="Volatility clustering",
        mechanism="High volatility follows high volatility (GARCH effect)",
        asset_class="equity_etf",
        variables=["vol_20d_z", "vol_60d_z"],
        direction="either",
        horizon="swing",
        source_type="academic",
        source_ref="Engle (1982)",
        evidence_strength="strong",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="reversal_1month",
        name="1-month reversal",
        mechanism="1-month losers outperform 1-month winners",
        asset_class="equity_stock",
        variables=["ret_20d_z"],
        direction="long",
        horizon="swing",
        source_type="academic",
        source_ref="Jegadeesh (1990)",
        evidence_strength="moderate",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="rsi_overbought_short",
        name="RSI overbought short",
        mechanism="Mean reversion from overbought conditions (RSI > 70)",
        asset_class="equity_etf",
        variables=["rsi"],
        direction="short",
        horizon="swing",
        source_type="observation",
        source_ref="internal observation",
        evidence_strength="weak",
        status="untested",
        notes=None,
    ),
    dict(
        mechanism_id="overnight_gap_fade",
        name="Overnight gap fade",
        mechanism="Large overnight gaps tend to partially fill",
        asset_class="equity_etf",
        variables=["ret_1d_z"],
        direction="either",
        horizon="intraday",
        source_type="blog",
        source_ref="practitioner convention",
        evidence_strength="weak",
        status="untested",
        notes=None,
    ),
]


def main() -> None:
    for m in MECHANISMS:
        add_mechanism(DUCKDB_PATH, **m)

    print("=== All 12 mechanisms seeded ===")
    df = query_mechanisms(DUCKDB_PATH)
    print(df.to_string())
    print(f"\nrow count: {len(df)}")

    conn = duckdb.connect(DUCKDB_PATH)
    row = conn.execute(
        "SELECT experiment_id FROM experiments WHERE ticker='QQQ' "
        "AND signal='ret_1d_z < 1.0 AND rsi < 30'"
    ).fetchone()
    conn.close()
    if row is None:
        raise RuntimeError("QQQ survivor experiment not found -- run main.py first")
    survivor_experiment_id = row[0]

    update_mechanism(DUCKDB_PATH, "rsi_oversold_bounce", status="live_paper")
    link_experiments(DUCKDB_PATH, "rsi_oversold_bounce", [survivor_experiment_id])

    print("\n=== rsi_oversold_bounce after promotion to live_paper ===")
    updated = query_mechanisms(DUCKDB_PATH, status="live_paper")
    print(updated[["mechanism_id", "status", "experiment_ids", "updated_at"]].to_string())


if __name__ == "__main__":
    main()
