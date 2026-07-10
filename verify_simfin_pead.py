"""
Minimal SimFin capability check for the pre-registered PEAD mechanism
(earnings_drift_pead in the knowledge graph) -- confirms the data source can
support the pre-registered universe BEFORE any data pipeline code is built.
This is NOT signal testing; it only checks that point-in-time (PIT) fields
exist and that enough history is available for one representative small-cap
name (LOVE -- Lovesac Company).

Not part of the main pipeline; run directly with:

    PYTHONPATH=. python verify_simfin_pead.py

The API key is read from the SIMFIN_API_KEY environment variable and is
never written to any file -- sf.set_api_key() only sets it in-process
memory (confirmed by reading the installed package's source: there is no
save_api_key() function, only load_api_key(path=...), which this script
does not call).

NOTE ON IMPORT PATH: the pre-registration task text suggested
`from simfin.api import SimFinApi`, but the actual installed `simfin`
PyPI package (v1.0.2) has no such module -- it exposes a flat
`import simfin as sf` API with `sf.set_api_key()` / `sf.load_income()`,
confirmed via introspection (`dir(simfin)`, `help(simfin.load)`) before
writing this script.
"""
import os
import sys

TICKER = "LOVE"  # Lovesac Company -- small-cap with public earnings history


def main() -> None:
    api_key = os.environ.get("SIMFIN_API_KEY")
    if not api_key:
        print("SIMFIN_API_KEY is not set in the environment.")
        print("This script never reads or writes the key to a file -- it is")
        print("only accepted via the environment variable. To run this check:")
        print()
        print("  PowerShell:  $env:SIMFIN_API_KEY = 'your_key_here'")
        print("  bash:        export SIMFIN_API_KEY=your_key_here")
        print()
        print("Skipping SimFin verification gracefully.")
        return

    import simfin as sf

    sf.set_api_key(api_key)  # in-memory only for this process; never persisted
    sf.set_data_dir(os.path.join(os.environ.get("TEMP", "/tmp"), "simfin_cache"))

    print(f"Fetching quarterly income statements (market=us) to check ticker {TICKER!r}...")
    income = sf.load_income(variant="quarterly", market="us")

    if TICKER not in income.index.get_level_values("Ticker"):
        print(f"ERROR: {TICKER!r} not found in the quarterly US income dataset.")
        sys.exit(1)

    df = income.loc[TICKER].sort_index()
    df = df[(df.index.get_level_values("Report Date") >= "2020-01-01") &
            (df.index.get_level_values("Report Date") <= "2023-12-31")]

    print()
    print(f"Rows for {TICKER} in 2020-2023: {len(df)}")

    has_report_date = "Report Date" in df.index.names or "Report Date" in df.columns
    has_publish_date = "Publish Date" in df.columns
    print(f"'Report Date' present:  {has_report_date}")
    print(f"'Publish Date' present: {has_publish_date}")
    if has_publish_date:
        report_dates = df.index.get_level_values("Report Date")
        distinct = (df["Publish Date"].to_numpy() != report_dates.to_numpy()).any()
        print(f"'Publish Date' distinct from 'Report Date' (i.e. genuinely PIT, "
              f"not just the period-end date relabeled): {distinct}")

    has_eps_column = any("EPS" in c.upper() for c in df.columns)
    print(f"Direct 'EPS' column present: {has_eps_column}")
    if not has_eps_column:
        derivable = "Net Income" in df.columns and any("Shares" in c for c in df.columns)
        print(f"EPS derivable from Net Income / Shares columns instead: {derivable}")

    print(f"At least 8 quarters of data: {len(df) >= 8}")

    print()
    print("First 5 rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
