"""Diagnostic plotting utilities for offline model evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write one matplotlib figure to disk and release its resources."""
    figure.savefig(output_path)
    plt.close(figure)


def write_diagnostic_figures(
    rv_daily: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate static diagnostic figures for offline demo outputs.

    Parameters
    ----------
    rv_daily
        Daily realized-variance table.
    predictions
        Walk-forward prediction table for all models.
    coefficients
        Ridge coefficient table.
    figures_directory
        Destination directory for PNG figures.
    """
    figures_directory.mkdir(parents=True, exist_ok=True)

    rv_history = (
        rv_daily.groupby("date", as_index=False)["realized_variance"]
        .mean()
        .sort_values("date")
    )
    history_figure, history_axis = plt.subplots(
        figsize=(11, 4), dpi=140, constrained_layout=True
    )
    history_axis.plot(
        rv_history["date"], rv_history["realized_variance"], color="#1f77b4"
    )
    history_axis.set_title("Average Daily Realized Variance")
    history_axis.set_xlabel("Date")
    history_axis.set_ylabel("Variance (decimal squared)")
    _save_figure(history_figure, figures_directory / "realized_variance_history.png")

    ridge_predictions = predictions[predictions["model_name"] == "ridge"].copy()
    ridge_predictions = ridge_predictions.sort_values("target_date")
    prediction_figure, prediction_axis = plt.subplots(
        figsize=(11, 4),
        dpi=140,
        constrained_layout=True,
    )
    prediction_axis.plot(
        ridge_predictions["target_date"],
        ridge_predictions["actual_log_rv_next_day"],
        label="Actual",
        color="#2ca02c",
    )
    prediction_axis.plot(
        ridge_predictions["target_date"],
        ridge_predictions["prediction_log_rv_next_day"],
        label="Ridge prediction",
        color="#ff7f0e",
    )
    prediction_axis.set_title("Ridge Prediction Versus Actual")
    prediction_axis.set_xlabel("Target Date")
    prediction_axis.set_ylabel("log(realized variance)")
    prediction_axis.legend()
    _save_figure(prediction_figure, figures_directory / "prediction_vs_actual.png")

    residual_values = (
        ridge_predictions["actual_log_rv_next_day"]
        - ridge_predictions["prediction_log_rv_next_day"]
    )
    residual_figure, residual_axis = plt.subplots(
        figsize=(8, 4),
        dpi=140,
        constrained_layout=True,
    )
    residual_axis.hist(residual_values, bins=20, color="#17becf", alpha=0.85)
    residual_axis.set_title("Ridge Residual Distribution")
    residual_axis.set_xlabel("Residual")
    residual_axis.set_ylabel("Count")
    _save_figure(residual_figure, figures_directory / "residual_distribution.png")

    coefficient_summary = (
        coefficients[coefficients["feature_name"] != "intercept"]
        .groupby("feature_name", as_index=False)["coefficient"]
        .mean()
        .sort_values("coefficient", ascending=False)
    )
    coefficient_figure, coefficient_axis = plt.subplots(
        figsize=(9, 4),
        dpi=140,
        constrained_layout=True,
    )
    coefficient_axis.bar(
        coefficient_summary["feature_name"],
        coefficient_summary["coefficient"],
        color="#9467bd",
    )
    coefficient_axis.set_title("Average Ridge Coefficients")
    coefficient_axis.set_xlabel("Feature")
    coefficient_axis.set_ylabel("Coefficient")
    coefficient_axis.tick_params(axis="x", labelrotation=30)
    _save_figure(coefficient_figure, figures_directory / "coefficient_bar_chart.png")
