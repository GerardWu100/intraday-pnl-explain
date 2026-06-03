"""End-to-end offline realized-variance research demo pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from intraday_pnl_explain.app.config import load_app_config
from intraday_pnl_explain.data_access.raw_bars import load_raw_intraday_bars
from intraday_pnl_explain.data_access.raw_manifest import load_raw_manifest
from intraday_pnl_explain.evaluation.diagnostics import write_diagnostic_figures
from intraday_pnl_explain.evaluation.metrics import compute_model_metrics
from intraday_pnl_explain.features.build_features import build_feature_table
from intraday_pnl_explain.modeling.train import build_walk_forward_predictions
from intraday_pnl_explain.realized_variance.construct import (
    construct_daily_realized_variance,
)


def run_offline_demo(output_directory: Path) -> None:
    """Run the full offline research workflow and write artifacts.

    Parameters
    ----------
    output_directory
        Destination folder for pipeline artifacts:
        metrics, predictions, coefficients, and figures.
    """
    config = load_app_config()

    manifest = load_raw_manifest(raw_root=config.raw_root)
    raw_bars = load_raw_intraday_bars(raw_root=config.raw_root, manifest=manifest)
    rv_daily = construct_daily_realized_variance(bars=raw_bars)
    feature_table = build_feature_table(rv_daily=rv_daily)

    # Walk-forward uses a short warm-up so the demo runs on the tracked sample size.
    predictions, coefficients = build_walk_forward_predictions(
        feature_table=feature_table,
        min_train_dates=4,
        ridge_alpha=1.0,
    )

    metrics_payload = compute_model_metrics(predictions=predictions)

    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory = output_directory / "figures"
    write_diagnostic_figures(
        rv_daily=rv_daily,
        predictions=predictions,
        coefficients=coefficients,
        figures_directory=figures_directory,
    )

    (output_directory / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    predictions.to_parquet(output_directory / "predictions.parquet", index=False)
    coefficients.to_csv(output_directory / "coefficients.csv", index=False)


if __name__ == "__main__":
    default_output_directory = load_app_config().default_output_directory
    run_offline_demo(output_directory=default_output_directory)
