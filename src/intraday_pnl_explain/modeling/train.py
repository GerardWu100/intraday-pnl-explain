"""Walk-forward training pipeline for baseline and ridge models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from intraday_pnl_explain.features.build_features import FEATURE_COLUMNS

# Each baseline reads one pre-built feature column on the test date.
BASELINE_FEATURE_COLUMNS: dict[str, str] = {
    "persistence": "current_log_rv",
    "rolling_mean": "trailing_5_mean_log_rv",
}


def _standardize(
    train_values: np.ndarray, test_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize train and test matrices using train moments only."""
    mean_values = train_values.mean(axis=0)
    std_values = train_values.std(axis=0)
    # Avoid division by zero when a feature is constant in the training window.
    std_values = np.where(std_values == 0.0, 1.0, std_values)

    train_scaled = (train_values - mean_values) / std_values
    test_scaled = (test_values - mean_values) / std_values
    return train_scaled, test_scaled


def _append_prediction_rows(
    prediction_rows: list[dict[str, object]],
    model_name: str,
    test_frame: pd.DataFrame,
    predicted_log_rv: np.ndarray,
) -> None:
    """Append one out-of-sample row per symbol for a single walk-forward test date."""
    for test_row, prediction_value in zip(
        test_frame.itertuples(index=False), predicted_log_rv
    ):
        prediction_rows.append(
            {
                "model_name": model_name,
                "symbol": str(test_row.symbol),
                "feature_date": pd.Timestamp(test_row.feature_date),
                "target_date": pd.Timestamp(test_row.target_date),
                "actual_log_rv_next_day": float(test_row.target_log_rv_next_day),
                "prediction_log_rv_next_day": float(prediction_value),
            }
        )


def _append_ridge_coefficient_rows(
    coefficient_rows: list[dict[str, object]],
    train_end_date: pd.Timestamp,
    ridge_model: Ridge,
) -> None:
    """Record ridge intercept and feature weights for one training window."""
    coefficient_rows.append(
        {
            "train_end_date": train_end_date,
            "feature_name": "intercept",
            "coefficient": float(ridge_model.intercept_),
        }
    )
    for feature_name, coefficient_value in zip(FEATURE_COLUMNS, ridge_model.coef_):
        coefficient_rows.append(
            {
                "train_end_date": train_end_date,
                "feature_name": str(feature_name),
                "coefficient": float(coefficient_value),
            }
        )


def build_walk_forward_predictions(
    feature_table: pd.DataFrame,
    min_train_dates: int = 10,
    ridge_alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build walk-forward predictions for baselines and ridge regression.

    Parameters
    ----------
    feature_table
        Leak-free feature table produced by ``features.build_features``.
    min_train_dates
        Minimum distinct feature dates required before out-of-sample prediction.
    ridge_alpha
        L2 regularization strength for ridge regression.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Two tables:

        - prediction rows across all models and walk-forward dates,
        - ridge coefficient rows for interpretation.
    """
    frame = feature_table.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame = frame.sort_values(["feature_date", "symbol"]).reset_index(drop=True)

    unique_dates = sorted(frame["feature_date"].unique())
    prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    # Expand one forecast-origin date at a time. A training label is eligible only
    # when its target has been observed by the test date's post-close cutoff.
    for date_index in range(min_train_dates, len(unique_dates)):
        test_date = unique_dates[date_index]
        train_frame = frame[
            (frame["feature_date"] < test_date) & (frame["target_date"] <= test_date)
        ].copy()
        test_frame = frame[frame["feature_date"] == test_date].copy()
        eligible_train_dates = sorted(train_frame["feature_date"].unique())
        if len(eligible_train_dates) < min_train_dates or test_frame.empty:
            continue

        for model_name, feature_column in BASELINE_FEATURE_COLUMNS.items():
            baseline_predictions = test_frame[feature_column].to_numpy(dtype=float)
            _append_prediction_rows(
                prediction_rows=prediction_rows,
                model_name=model_name,
                test_frame=test_frame,
                predicted_log_rv=baseline_predictions,
            )

        train_x = train_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        train_y = train_frame["target_log_rv_next_day"].to_numpy(dtype=float)
        test_x = test_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)

        train_x_scaled, test_x_scaled = _standardize(
            train_values=train_x, test_values=test_x
        )

        ridge_model = Ridge(alpha=ridge_alpha)
        ridge_model.fit(train_x_scaled, train_y)
        ridge_predictions = ridge_model.predict(test_x_scaled)

        _append_prediction_rows(
            prediction_rows=prediction_rows,
            model_name="ridge",
            test_frame=test_frame,
            predicted_log_rv=ridge_predictions,
        )
        _append_ridge_coefficient_rows(
            coefficient_rows=coefficient_rows,
            train_end_date=pd.Timestamp(eligible_train_dates[-1]),
            ridge_model=ridge_model,
        )

    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients
