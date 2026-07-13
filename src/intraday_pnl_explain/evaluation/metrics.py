"""Evaluation metrics for next-day log realized variance forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


PERSISTENCE_MODEL_NAME = "persistence"


def compute_model_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute errors and forecast skill relative to persistence.

    Parameters
    ----------
    predictions
        Prediction table with ``model_name``, ``actual_log_rv_next_day``, and
        ``prediction_log_rv_next_day`` columns.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested mapping from model name to root mean squared error (RMSE), mean
        absolute error (MAE), and squared-error skill versus persistence. Skill
        is ``1 - SSE_model / SSE_persistence`` on common forecast rows.
    """
    persistence_rows = predictions[
        predictions["model_name"] == PERSISTENCE_MODEL_NAME
    ].copy()
    if persistence_rows.empty:
        raise ValueError("Persistence predictions are required as the skill benchmark")

    observation_columns = ["symbol", "feature_date", "target_date"]
    persistence_rows = persistence_rows.loc[
        :, [*observation_columns, "prediction_log_rv_next_day"]
    ].rename(
        columns={"prediction_log_rv_next_day": "persistence_prediction"}
    )
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, group in predictions.groupby("model_name"):
        scored_group = group.merge(
            persistence_rows,
            on=observation_columns,
            how="inner",
            validate="one_to_one",
        )
        if len(scored_group.index) != len(group.index):
            raise ValueError(
                f"Model {model_name!r} does not share every row with persistence"
            )

        actual_values = scored_group["actual_log_rv_next_day"].to_numpy(dtype=float)
        predicted_values = scored_group["prediction_log_rv_next_day"].to_numpy(dtype=float)
        persistence_values = scored_group["persistence_prediction"].to_numpy(dtype=float)
        residual_values = actual_values - predicted_values
        persistence_residual_values = actual_values - persistence_values

        rmse_value = float(np.sqrt(np.square(residual_values).mean()))
        mae_value = float(np.abs(residual_values).mean())
        model_sse = float(np.square(residual_values).sum())
        persistence_sse = float(np.square(persistence_residual_values).sum())
        skill_value = (
            0.0
            if persistence_sse == 0.0 and model_sse == 0.0
            else float(1.0 - model_sse / persistence_sse)
        )

        metrics_by_model[str(model_name)] = {
            "rmse": rmse_value,
            "mae": mae_value,
            "skill_vs_persistence": skill_value,
        }

    return metrics_by_model
