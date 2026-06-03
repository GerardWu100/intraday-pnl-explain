"""Evaluation metrics for next-day log realized variance forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute out-of-sample coefficient of determination."""
    residual_sum_squares = np.square(actual - predicted).sum()
    total_sum_squares = np.square(actual - actual.mean()).sum()
    if float(total_sum_squares) == 0.0:
        return 0.0
    return float(1.0 - (residual_sum_squares / total_sum_squares))


def compute_model_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute RMSE, MAE, and out-of-sample R-squared by model name.

    Parameters
    ----------
    predictions
        Prediction table with ``model_name``, ``actual_log_rv_next_day``, and
        ``prediction_log_rv_next_day`` columns.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping from model name to metric values.
    """
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, group in predictions.groupby("model_name"):
        actual_values = group["actual_log_rv_next_day"].to_numpy(dtype=float)
        predicted_values = group["prediction_log_rv_next_day"].to_numpy(dtype=float)
        residual_values = actual_values - predicted_values

        rmse_value = float(np.sqrt(np.square(residual_values).mean()))
        mae_value = float(np.abs(residual_values).mean())
        r2_value = _r2_score(actual=actual_values, predicted=predicted_values)

        metrics_by_model[str(model_name)] = {
            "rmse": rmse_value,
            "mae": mae_value,
            "r2_oos": r2_value,
        }

    return metrics_by_model
