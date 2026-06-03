"""Feature engineering utilities for next-day log realized variance forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_pnl_explain.realized_variance.construct import (
    EXPECTED_BAR_RETURNS_PER_DAY,
)

EPSILON_VARIANCE = 1e-12

FEATURE_COLUMNS = [
    "lag_1_log_rv",
    "lag_5_mean_log_rv",
    "lag_5_std_log_rv",
    "prev_day_range_proxy",
    "bar_completeness",
]


def build_feature_table(rv_daily: pd.DataFrame) -> pd.DataFrame:
    """Build leak-free features for predicting next-day log realized variance.

    Parameters
    ----------
    rv_daily
        Daily realized variance table with columns:
        ``symbol``, ``date``, ``realized_variance``, ``bar_count``.

    Returns
    -------
    pandas.DataFrame
        Feature table with one row per symbol and feature date ``d`` where the
        target is ``log(RV_{d+1})``.
    """
    frame = rv_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Log transform stabilizes scale; clip avoids log(0) on quiet days.
    clipped_variance = frame["realized_variance"].clip(lower=EPSILON_VARIANCE)
    frame["log_rv"] = np.log(clipped_variance)

    # Features at date d use only information through d (same-day log RV and its lags).
    frame["lag_1_log_rv"] = frame["log_rv"]
    frame["lag_5_mean_log_rv"] = frame.groupby("symbol")["log_rv"].transform(
        lambda values: values.rolling(5).mean()
    )
    frame["lag_5_std_log_rv"] = frame.groupby("symbol")["log_rv"].transform(
        lambda values: values.rolling(5).std()
    )
    frame["prev_day_range_proxy"] = frame.groupby("symbol")["log_rv"].transform(
        lambda values: values.diff().abs()
    )
    frame["bar_completeness"] = frame["bar_count"] / float(EXPECTED_BAR_RETURNS_PER_DAY)

    # Target is next-day log RV; rows without a realized next day are dropped below.
    frame["feature_date"] = frame["date"]
    frame["target_date"] = frame.groupby("symbol")["date"].shift(-1)
    frame["target_log_rv_next_day"] = frame.groupby("symbol")["log_rv"].shift(-1)

    required_columns = FEATURE_COLUMNS + ["target_log_rv_next_day", "target_date"]
    feature_table = frame.dropna(subset=required_columns).copy()

    return feature_table.loc[
        :,
        [
            "symbol",
            "feature_date",
            "target_date",
            *FEATURE_COLUMNS,
            "target_log_rv_next_day",
        ],
    ].reset_index(drop=True)
