"""Optional ClickHouse extraction entrypoints.

This module is intentionally isolated from the offline runtime path.
Offline execution does not depend on any database connectivity.
"""

from __future__ import annotations


def raise_optional_clickhouse_message() -> None:
    """Raise a clear message for optional one-time ClickHouse extraction.

    Raises
    ------
    RuntimeError
        Always raised to indicate this command is outside offline runtime scope.
    """
    raise RuntimeError(
        "ClickHouse extraction is optional and one-time only. "
        "Offline runs read tracked data/raw parquet files directly."
    )
