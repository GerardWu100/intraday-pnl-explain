"""Tests for realized variance construction from intraday bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from intraday_pnl_explain.realized_variance.construct import (
    construct_daily_realized_variance,
)


def test_realized_variance_matches_hand_computed_log_return_sum() -> None:
    """Daily realized variance should equal sum of squared intraday log returns."""
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "timestamp": pd.to_datetime(
                [
                    "2026-04-01 13:30:00+00:00",
                    "2026-04-01 13:31:00+00:00",
                    "2026-04-01 13:32:00+00:00",
                    "2026-04-01 13:33:00+00:00",
                ],
                utc=True,
            ),
            "price": [100.0, 101.0, 99.0, 100.0],
            "volume": [1000, 1000, 1000, 1000],
        }
    )

    realized = construct_daily_realized_variance(bars=bars)

    r1 = np.log(101.0 / 100.0)
    r2 = np.log(99.0 / 101.0)
    r3 = np.log(100.0 / 99.0)
    expected_rv = float(r1**2 + r2**2 + r3**2)

    assert len(realized.index) == 1
    assert realized.loc[0, "realized_variance"] == expected_rv
    assert realized.loc[0, "realized_variance"] >= 0.0
    assert realized.loc[0, "bar_count"] == 3
