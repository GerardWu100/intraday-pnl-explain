"""Daily realized variance construction from intraday price bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Regular session has 389 one-minute return intervals between 09:30 and 15:59 ET.
EXPECTED_BAR_RETURNS_PER_DAY = 389


def _sum_squared_log_returns(log_returns: pd.Series) -> float:
    """Sum r_t^2 over non-missing intraday log returns for one symbol-day."""
    valid_returns = log_returns.dropna()
    return float(np.square(valid_returns).sum())


def _non_missing_log_return_count(log_returns: pd.Series) -> int:
    """Count intraday returns available after the first bar of the session."""
    return int(log_returns.notna().sum())


def construct_daily_realized_variance(bars: pd.DataFrame) -> pd.DataFrame:
    """Construct daily realized variance from intraday bars.

    Realized variance is computed as the sum of squared intraday log returns:

    ``RV_d = sum_t r_t^2`` where ``r_t = log(P_t / P_{t-1})`` for one symbol-day.

    Parameters
    ----------
    bars
        Intraday bars containing at least ``symbol``, ``timestamp``, and ``price``.

    Returns
    -------
    pandas.DataFrame
        Daily table with columns:
        ``symbol``, ``date``, ``realized_variance``, ``realized_volatility``,
        ``bar_count``, ``is_complete_session``.
    """
    prepared_bars = bars.copy()
    prepared_bars["timestamp"] = pd.to_datetime(prepared_bars["timestamp"], utc=True)
    prepared_bars = prepared_bars.sort_values(["symbol", "timestamp"]).reset_index(
        drop=True
    )

    # Assign each bar to its New York trading date before aggregating within the day.
    local_date = prepared_bars["timestamp"].dt.tz_convert("America/New_York").dt.date
    prepared_bars["date"] = pd.to_datetime(local_date)

    group_columns = ["symbol", "date"]
    prepared_bars["log_return"] = prepared_bars.groupby(group_columns)[
        "price"
    ].transform(lambda values: np.log(values / values.shift(1)))

    realized_daily = (
        prepared_bars.groupby(group_columns, as_index=False)
        .agg(
            realized_variance=("log_return", _sum_squared_log_returns),
            bar_count=("log_return", _non_missing_log_return_count),
        )
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    realized_daily["realized_volatility"] = np.sqrt(realized_daily["realized_variance"])
    realized_daily["is_complete_session"] = (
        realized_daily["bar_count"] >= EXPECTED_BAR_RETURNS_PER_DAY
    )

    return realized_daily
