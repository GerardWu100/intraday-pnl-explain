"""Integration tests for end-to-end offline research pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from intraday_pnl_explain.pipeline.run_offline_demo import run_offline_demo


def test_offline_demo_writes_expected_artifacts(tmp_path: Path) -> None:
    """Pipeline should produce metrics, predictions, coefficients, and figures."""
    output_directory = tmp_path / "demo_run"
    run_offline_demo(output_directory=output_directory)

    metrics_path = output_directory / "metrics.json"
    predictions_path = output_directory / "predictions.parquet"
    coefficients_path = output_directory / "coefficients.csv"
    figures_directory = output_directory / "figures"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert coefficients_path.exists()
    assert figures_directory.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics_payload) == {"persistence", "rolling_mean", "ridge"}
    assert set(metrics_payload["ridge"]) == {
        "rmse",
        "mae",
        "skill_vs_persistence",
    }
    assert metrics_payload["persistence"]["skill_vs_persistence"] == 0.0

    predictions_frame = pd.read_parquet(predictions_path)
    assert set(predictions_frame["model_name"].unique()) == {
        "persistence",
        "rolling_mean",
        "ridge",
    }

    assert len(list(output_directory.rglob("*.html"))) == 0
