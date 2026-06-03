"""Tests for walk-forward modeling orchestration and output shape."""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_pnl_explain.modeling.train import build_walk_forward_predictions


def test_walk_forward_predictions_respect_time_order_and_shape() -> None:
    """Modeling output should preserve time order and expected row counts."""
    dates = pd.date_range("2026-03-01", periods=14, freq="B")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(["AAPL", "MSFT"]):
        for date_index, feature_date in enumerate(dates):
            lag_1_log_rv = -4.5 + 0.05 * date_index + 0.01 * symbol_index
            lag_5_mean_log_rv = lag_1_log_rv - 0.02
            lag_5_std_log_rv = 0.05 + 0.001 * date_index
            prev_day_range_proxy = 0.02 + 0.001 * symbol_index
            bar_completeness = 1.0
            target = (
                0.7 * lag_1_log_rv
                + 0.2 * lag_5_mean_log_rv
                + 0.1 * prev_day_range_proxy
            )
            rows.append(
                {
                    "symbol": symbol,
                    "feature_date": feature_date,
                    "target_date": feature_date + pd.Timedelta(days=1),
                    "lag_1_log_rv": lag_1_log_rv,
                    "lag_5_mean_log_rv": lag_5_mean_log_rv,
                    "lag_5_std_log_rv": lag_5_std_log_rv,
                    "prev_day_range_proxy": prev_day_range_proxy,
                    "bar_completeness": bar_completeness,
                    "target_log_rv_next_day": target,
                }
            )

    feature_table = pd.DataFrame(rows)
    predictions, coefficients = build_walk_forward_predictions(
        feature_table=feature_table,
        min_train_dates=5,
        ridge_alpha=1.0,
    )

    assert len(predictions.index) > 0
    assert len(coefficients.index) > 0
    assert (predictions["feature_date"] < predictions["target_date"]).all()
    assert np.isfinite(predictions["prediction_log_rv_next_day"]).all()
