"""Tests for lagged and rolling feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_pnl_explain.features.build_features import build_feature_table


def test_feature_builder_uses_only_information_available_at_feature_date() -> None:
    """Feature row for day d should predict log(RV_{d+1}) without leakage."""
    rv_daily = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 8,
            "date": pd.to_datetime(
                [
                    "2026-03-24",
                    "2026-03-25",
                    "2026-03-26",
                    "2026-03-27",
                    "2026-03-30",
                    "2026-03-31",
                    "2026-04-01",
                    "2026-04-02",
                ]
            ),
            "realized_variance": [
                0.010,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
                0.016,
                0.017,
            ],
            "bar_count": [389] * 8,
            "is_complete_session": [True] * 8,
        }
    )

    feature_table = build_feature_table(rv_daily=rv_daily)
    assert len(feature_table.index) > 0

    first_row = feature_table.iloc[0]
    feature_date = pd.Timestamp(first_row["feature_date"])
    source_row = rv_daily[rv_daily["date"] == feature_date].iloc[0]
    next_day_source_row = rv_daily[rv_daily["date"] > feature_date].iloc[0]

    assert first_row["lag_1_log_rv"] == np.log(source_row["realized_variance"])
    assert first_row["target_log_rv_next_day"] == np.log(
        next_day_source_row["realized_variance"]
    )
