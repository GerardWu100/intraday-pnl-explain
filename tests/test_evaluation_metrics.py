"""Tests for feasible out-of-sample forecast comparisons."""

from __future__ import annotations

import pandas as pd
import pytest

from intraday_pnl_explain.evaluation.metrics import compute_model_metrics


def test_forecast_skill_uses_persistence_squared_error() -> None:
    """Skill should compare model SSE with persistence on identical rows."""
    rows: list[dict[str, object]] = []
    for model_name, predictions in {
        "persistence": [1.0, 5.0],
        "ridge": [2.0, 4.0],
    }.items():
        for symbol, actual, prediction in zip(
            ["AAPL", "MSFT"], [3.0, 3.0], predictions, strict=True
        ):
            rows.append(
                {
                    "model_name": model_name,
                    "symbol": symbol,
                    "feature_date": pd.Timestamp("2026-04-01"),
                    "target_date": pd.Timestamp("2026-04-02"),
                    "actual_log_rv_next_day": actual,
                    "prediction_log_rv_next_day": prediction,
                }
            )

    metrics = compute_model_metrics(pd.DataFrame(rows))

    assert metrics["persistence"]["skill_vs_persistence"] == pytest.approx(0.0)
    assert metrics["ridge"]["skill_vs_persistence"] == pytest.approx(0.75)
