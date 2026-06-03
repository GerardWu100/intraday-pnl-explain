"""Raw dataset manifest loading and contract validation utilities.

This module defines a small typed contract for the tracked raw dataset.
The contract is intentionally strict so offline reproducibility is explicit:

- one manifest file under ``data/raw/raw_manifest.json``
- one parquet family under ``data/raw/intraday_bars``
- one narrow column schema used by the runtime path
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REQUIRED_MANIFEST_KEYS = {
    "dataset_name",
    "symbols",
    "start_date",
    "end_date",
    "bar_frequency",
    "timezone",
    "trading_session_timezone",
    "trading_session",
    "row_count",
    "file_count",
    "total_bytes",
    "required_columns",
    "partition_layout",
}


@dataclass(frozen=True)
class RawManifest:
    """In-memory representation of raw dataset metadata.

    Parameters
    ----------
    dataset_name
        Human-readable dataset identifier.
    symbols
        List of symbol identifiers included in the raw payload.
    start_date
        First trading date covered by the raw payload, formatted YYYY-MM-DD.
    end_date
        Last trading date covered by the raw payload, formatted YYYY-MM-DD.
    bar_frequency
        Intraday bar frequency string, for example ``1min``.
    timezone
        Timestamp timezone for stored raw timestamps.
    trading_session_timezone
        Local exchange timezone used by session filtering.
    trading_session
        Session window string, for example ``09:30-15:59``.
    row_count
        Total number of rows across all raw parquet files.
    file_count
        Total number of parquet files in the raw payload.
    total_bytes
        Total file size in bytes across all raw parquet files.
    required_columns
        Ordered list of required raw parquet columns.
    partition_layout
        Text description of the partition path convention.
    """

    dataset_name: str
    symbols: list[str]
    start_date: str
    end_date: str
    bar_frequency: str
    timezone: str
    trading_session_timezone: str
    trading_session: str
    row_count: int
    file_count: int
    total_bytes: int
    required_columns: list[str]
    partition_layout: str


def load_raw_manifest(raw_root: Path) -> RawManifest:
    """Load and validate the raw manifest from ``data/raw``.

    Parameters
    ----------
    raw_root
        Absolute path to the root raw-data folder.

    Returns
    -------
    RawManifest
        Parsed and validated raw manifest object.
    """
    manifest_path = raw_root / "raw_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Fail fast when the tracked manifest drifts from the runtime contract.
    missing_keys = REQUIRED_MANIFEST_KEYS.difference(payload)
    if missing_keys:
        missing_keys_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"Raw manifest is missing keys: {missing_keys_text}")

    return RawManifest(
        dataset_name=str(payload["dataset_name"]),
        symbols=list(payload["symbols"]),
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]),
        bar_frequency=str(payload["bar_frequency"]),
        timezone=str(payload["timezone"]),
        trading_session_timezone=str(payload["trading_session_timezone"]),
        trading_session=str(payload["trading_session"]),
        row_count=int(payload["row_count"]),
        file_count=int(payload["file_count"]),
        total_bytes=int(payload["total_bytes"]),
        required_columns=list(payload["required_columns"]),
        partition_layout=str(payload["partition_layout"]),
    )
