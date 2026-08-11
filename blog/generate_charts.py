"""Generate frozen data and technical figures for the bilingual blog post.

The script reads the project's tracked intraday parquet files and the frozen
walk-forward outputs under ``blog/data/demo_run``. It writes one derived CSV
and three publication-ready figures under ``blog/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intraday_pnl_explain.app.config import load_app_config
from intraday_pnl_explain.data_access.raw_bars import load_raw_intraday_bars
from intraday_pnl_explain.data_access.raw_manifest import load_raw_manifest
from intraday_pnl_explain.realized_variance.construct import (
    construct_daily_realized_variance,
)

BLOG_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = BLOG_ROOT / "data"
DEMO_DIRECTORY = DATA_DIRECTORY / "demo_run"
IMAGES_DIRECTORY = BLOG_ROOT / "images"

FIGURE_DPI = 180
SYMBOL_COLORS = {
    "AAPL": "#4C78A8",
    "MSFT": "#72B7B2",
    "NVDA": "#F58518",
    "XOM": "#E45756",
    "CVX": "#B279A2",
    "JPM": "#54A24B",
}
MODEL_COLORS = {
    "persistence": "#4C78A8",
    "rolling_mean": "#F58518",
    "ridge": "#54A24B",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "rolling_mean": "5-day mean",
    "ridge": "Ridge",
}


def _apply_plot_style() -> None:
    """Apply a restrained style shared by all post figures."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "font.size": 10,
            "grid.alpha": 0.22,
        }
    )


def _load_and_freeze_realized_variance() -> pd.DataFrame:
    """Construct daily realized variance and freeze the post-level CSV."""
    config = load_app_config()
    manifest = load_raw_manifest(config.raw_root)
    bars = load_raw_intraday_bars(config.raw_root, manifest)
    realized = construct_daily_realized_variance(bars)

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frozen_path = DATA_DIRECTORY / "realized_variance_daily.csv"
    realized.to_csv(frozen_path, index=False)
    return realized


def _plot_realized_variance(realized: pd.DataFrame) -> None:
    """Plot daily realized variance in one panel per symbol."""
    symbols = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "CVX"]
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
    )

    for axis, symbol in zip(axes.flat, symbols, strict=True):
        symbol_data = realized[realized["symbol"] == symbol].sort_values("date")
        scaled_variance = symbol_data["realized_variance"] * 1_000_000
        axis.plot(
            symbol_data["date"],
            scaled_variance,
            color=SYMBOL_COLORS[symbol],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
        )
        axis.fill_between(
            symbol_data["date"],
            scaled_variance,
            color=SYMBOL_COLORS[symbol],
            alpha=0.10,
        )
        axis.set_title(symbol, loc="left")
        axis.set_ylabel("RV × 10⁶")
        axis.grid(axis="y")
        axis.tick_params(axis="x", rotation=30)

    figure.suptitle(
        "Daily realized variance from one-minute regular-session returns",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(
        IMAGES_DIRECTORY / "01_realized_variance_by_symbol.png", dpi=FIGURE_DPI
    )
    plt.close(figure)


def _plot_model_errors() -> None:
    """Compare root mean squared error and mean absolute error by model."""
    metrics = json.loads((DEMO_DIRECTORY / "metrics.json").read_text(encoding="utf-8"))
    models = ["persistence", "rolling_mean", "ridge"]
    rmse_values = [metrics[model]["rmse"] for model in models]
    mae_values = [metrics[model]["mae"] for model in models]

    x_positions = np.arange(len(models), dtype=float)
    bar_width = 0.34
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    axis.bar(
        x_positions - bar_width / 2,
        rmse_values,
        width=bar_width,
        color="#315A7D",
        label="RMSE",
    )
    axis.bar(
        x_positions + bar_width / 2,
        mae_values,
        width=bar_width,
        color="#D8893D",
        label="MAE",
    )

    for position, value in zip(x_positions - bar_width / 2, rmse_values, strict=True):
        axis.text(position, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    for position, value in zip(x_positions + bar_width / 2, mae_values, strict=True):
        axis.text(position, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)

    axis.set_title("Forecast errors on the single held-out date")
    axis.set_ylabel("Error in log realized-variance units")
    axis.set_xticks(x_positions, [MODEL_LABELS[model] for model in models])
    axis.set_ylim(0.0, max(rmse_values) * 1.18)
    axis.grid(axis="y")
    axis.legend(frameon=False)
    figure.savefig(IMAGES_DIRECTORY / "02_model_error_comparison.png", dpi=FIGURE_DPI)
    plt.close(figure)


def _plot_cross_section_forecasts() -> None:
    """Plot actual and model-predicted log variance across six symbols."""
    predictions = pd.read_parquet(DEMO_DIRECTORY / "predictions.parquet")
    symbol_order = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "CVX"]
    predictions["symbol"] = pd.Categorical(
        predictions["symbol"], categories=symbol_order, ordered=True
    )
    predictions = predictions.sort_values(["model_name", "symbol"])

    actual = (
        predictions.drop_duplicates("symbol")
        .set_index("symbol")
        .reindex(symbol_order)["actual_log_rv_next_day"]
    )
    x_positions = np.arange(len(symbol_order), dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    axis.plot(
        x_positions,
        actual.to_numpy(),
        color="#111827",
        marker="o",
        linewidth=2.7,
        markersize=7,
        label="Actual",
        zorder=5,
    )

    for model in ["persistence", "rolling_mean", "ridge"]:
        model_values = (
            predictions[predictions["model_name"] == model]
            .set_index("symbol")
            .reindex(symbol_order)["prediction_log_rv_next_day"]
        )
        axis.plot(
            x_positions,
            model_values.to_numpy(),
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=1.7,
            markersize=5,
            alpha=0.92,
            label=MODEL_LABELS[model],
        )

    axis.set_title("Next-day forecasts across the six-symbol test cross-section")
    axis.set_xlabel("Symbol")
    axis.set_ylabel("log(realized variance)")
    axis.set_xticks(x_positions, symbol_order)
    axis.grid(axis="y")
    axis.legend(frameon=False, ncol=4, loc="lower center")
    figure.savefig(IMAGES_DIRECTORY / "03_forecast_cross_section.png", dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    """Generate the frozen daily series and all article figures."""
    _apply_plot_style()
    IMAGES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    realized = _load_and_freeze_realized_variance()
    _plot_realized_variance(realized)
    _plot_model_errors()
    _plot_cross_section_forecasts()


if __name__ == "__main__":
    main()
