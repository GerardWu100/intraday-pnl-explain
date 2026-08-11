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
            current_log_rv = -4.5 + 0.05 * date_index + 0.01 * symbol_index
            trailing_5_mean_log_rv = current_log_rv - 0.02
            trailing_5_std_log_rv = 0.05 + 0.001 * date_index
            absolute_1day_log_rv_change = 0.02 + 0.001 * symbol_index
            bar_completeness = 1.0
            target = (
                0.7 * current_log_rv
                + 0.2 * trailing_5_mean_log_rv
                + 0.1 * absolute_1day_log_rv_change
            )
            rows.append(
                {
                    "symbol": symbol,
                    "feature_date": feature_date,
                    "target_date": feature_date + pd.Timedelta(days=1),
                    "current_log_rv": current_log_rv,
                    "trailing_5_mean_log_rv": trailing_5_mean_log_rv,
                    "trailing_5_std_log_rv": trailing_5_std_log_rv,
                    "absolute_1day_log_rv_change": absolute_1day_log_rv_change,
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


def test_walk_forward_excludes_labels_unavailable_at_forecast_cutoff() -> None:
    """A missing symbol date must not let a future target enter training."""
    dates = pd.date_range("2026-03-02", periods=7, freq="B")
    rows: list[dict[str, object]] = []
    for date_index, feature_date in enumerate(dates):
        target_date = feature_date + pd.offsets.BDay(1)
        if date_index == 2:
            # Simulate a gap: this label is not known at the next forecast origin.
            target_date = feature_date + pd.offsets.BDay(3)
        rows.append(
            {
                "symbol": "AAPL",
                "feature_date": feature_date,
                "target_date": target_date,
                "current_log_rv": -5.0 + date_index,
                "trailing_5_mean_log_rv": -5.1 + date_index,
                "trailing_5_std_log_rv": 0.2,
                "absolute_1day_log_rv_change": 0.1,
                "bar_completeness": 1.0,
                "target_log_rv_next_day": -4.9 + date_index,
            }
        )

    predictions, coefficients = build_walk_forward_predictions(
        feature_table=pd.DataFrame(rows),
        min_train_dates=3,
        ridge_alpha=1.0,
    )

    # The first candidate test date has only two eligible training dates after
    # purging the unavailable label, so scoring starts one date later.
    assert predictions["feature_date"].min() == dates[4]
    assert coefficients["train_end_date"].min() == dates[3]
