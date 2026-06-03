# Raw Data Contract

The offline runtime path reads raw intraday parquet files from `data/raw/`.

## Required files

- `data/raw/raw_manifest.json`
- `data/raw/intraday_bars/symbol=<SYMBOL>/part-000.parquet` for each symbol

## Required parquet columns

- `symbol` (string)
- `timestamp` (timezone-aware UTC timestamp)
- `price` (float)
- `volume` (integer)

## Partition layout

`data/raw/intraday_bars/symbol=<SYMBOL>/part-000.parquet`

## Session and timezone convention

- Stored timestamps are UTC.
- Regular-session filtering uses `America/New_York` local time.
- Session window is `09:30` through `15:59` inclusive.

## Manifest fields

The manifest must include:

- dataset identity (`dataset_name`)
- symbol universe (`symbols`)
- date coverage (`start_date`, `end_date`)
- bar metadata (`bar_frequency`, `timezone`, `trading_session_timezone`, `trading_session`)
- payload size metadata (`row_count`, `file_count`, `total_bytes`)
- schema metadata (`required_columns`, `partition_layout`)

## Payload sizing rule

Tracked raw payload should stay below `100 MB` when practical.

Current test suite enforces this guardrail.
