# GUIDE_data

## Part 1: Conceptual explanation

`data/` stores tracked offline artifacts required for clone-and-run reproducibility.

The runtime contract now requires raw intraday data under `data/raw/` only.

There should be no runtime parquet dependencies under any other `data/` subfolder.

## Part 2: Code reference

- `raw/raw_manifest.json`: raw dataset metadata contract (coverage, schema, file/row counts).
- `raw/intraday_bars/symbol=<SYMBOL>/part-000.parquet`: symbol-partitioned intraday bars.

Required raw columns:

- `symbol`
- `timestamp`
- `price`
- `volume`

The raw payload is intentionally compact and stays under repository-friendly size limits.

## Part 3: Short journal

- 2026-04-19: Migrated tracked runtime data from cache snapshots to raw intraday bars.
