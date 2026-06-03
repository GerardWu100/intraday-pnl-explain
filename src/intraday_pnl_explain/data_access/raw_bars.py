"""Offline raw intraday parquet loading and normalization.

The runtime contract here is simple and explicit:

- read from ``data/raw`` only,
- load a narrow schema,
- normalize timestamps,
- sort and deduplicate rows,
- filter to regular trading session bars.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd

from intraday_pnl_explain.data_access.raw_manifest import RawManifest

# Regular NYSE cash session window in exchange-local time.
SESSION_START_TIME = time(9, 30)
SESSION_END_TIME = time(15, 59)


def discover_raw_parquet_paths(raw_root: Path, symbols: list[str]) -> list[Path]:
    """Resolve expected symbol-level raw parquet paths from the manifest symbols.

    Parameters
    ----------
    raw_root
        Root path to the tracked raw data folder.
    symbols
        Symbols expected in the raw payload.

    Returns
    -------
    list[pathlib.Path]
        Existing parquet paths in stable sorted order.
    """
    parquet_paths: list[Path] = []

    for symbol in sorted(symbols):
        symbol_directory = raw_root / "intraday_bars" / f"symbol={symbol}"
        symbol_paths = sorted(symbol_directory.glob("part-*.parquet"))
        parquet_paths.extend(symbol_paths)

    return parquet_paths


def load_raw_intraday_bars(raw_root: Path, manifest: RawManifest) -> pd.DataFrame:
    """Load and clean tracked raw bars for offline realized-variance workflows.

    Parameters
    ----------
    raw_root
        Root path to tracked raw data.
    manifest
        Raw manifest object with schema and symbol expectations.

    Returns
    -------
    pandas.DataFrame
        Normalized intraday bars with columns:
        ``symbol``, ``timestamp``, ``price``, ``volume``.
    """
    parquet_paths = discover_raw_parquet_paths(
        raw_root=raw_root, symbols=manifest.symbols
    )
    if not parquet_paths:
        raise ValueError(
            "No tracked raw parquet files were found under data/raw/intraday_bars"
        )

    frames: list[pd.DataFrame] = []
    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path)
        frames.append(frame)

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.loc[:, manifest.required_columns].copy()

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    bars = bars.drop_duplicates(subset=["symbol", "timestamp"], keep="last")

    # Keep only bars inside the regular session window in exchange-local time.
    local_timestamps = bars["timestamp"].dt.tz_convert(
        manifest.trading_session_timezone
    )
    local_times = local_timestamps.dt.time
    in_session_mask = (local_times >= SESSION_START_TIME) & (
        local_times <= SESSION_END_TIME
    )
    bars = bars.loc[in_session_mask].reset_index(drop=True)

    return bars
