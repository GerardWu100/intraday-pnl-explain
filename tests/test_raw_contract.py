"""Tests for tracked raw intraday parquet contract guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_raw_manifest_exists_and_matches_payload_files() -> None:
    """Manifest values should reconcile to the actual tracked raw files."""
    manifest_path = Path("data/raw/raw_manifest.json")
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_paths = sorted(Path("data/raw/intraday_bars").rglob("*.parquet"))
    assert len(raw_paths) > 0

    computed_total_bytes = sum(path.stat().st_size for path in raw_paths)
    assert payload["file_count"] == len(raw_paths)
    assert payload["total_bytes"] == computed_total_bytes

    computed_total_rows = 0
    required_columns = payload["required_columns"]
    for raw_path in raw_paths:
        frame = pd.read_parquet(raw_path)
        computed_total_rows += len(frame.index)
        assert list(frame.columns) == required_columns

    assert payload["row_count"] == computed_total_rows


def test_parquet_files_under_data_live_only_in_raw_folder() -> None:
    """All parquet files under data should be routed to data/raw only."""
    data_root = Path("data")
    parquet_paths = sorted(data_root.rglob("*.parquet"))
    assert len(parquet_paths) > 0

    for parquet_path in parquet_paths:
        assert str(parquet_path).startswith("data/raw/")


def test_tracked_raw_payload_stays_below_size_limit() -> None:
    """Tracked raw payload must remain comfortably below 100 MB."""
    total_bytes = sum(
        path.stat().st_size for path in Path("data/raw").rglob("*") if path.is_file()
    )
    assert total_bytes < 100 * 1024 * 1024
