# GUIDE_data_access

## Part 1: Conceptual explanation

`data_access/` owns raw data contracts and loading behavior for offline runs.

The runtime path is intentionally narrow:

- load one tracked manifest,
- discover raw parquet partitions,
- normalize and session-filter bars,
- expose one clean intraday bars DataFrame to downstream modules.

ClickHouse extraction logic is isolated behind an explicit optional command and is
never called in the offline runtime path.

## Part 2: Code reference

- `raw_manifest.py`: manifest dataclass, required-key validation, and parser.
- `raw_bars.py`: symbol path discovery, parquet loading, deduplication, and
  regular-session filtering.
- `clickhouse_extract.py`: explicit optional extraction message for offline mode.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added raw-data access layer as part of realized-variance refactor.
