# intraday-pnl-explain

`intraday-pnl-explain` is now an offline-first quantitative research demo focused on one question:

Can simple lagged intraday realized-variance features forecast next-day realized variance better than naive baselines?

The project runs entirely from tracked local parquet files under `data/raw/`.

## Offline Architecture

```text
data/raw/intraday_bars/*.parquet
  -> intraday_pnl_explain.data_access (manifest + bars)
  -> intraday_pnl_explain.realized_variance.construct
  -> intraday_pnl_explain.features.build_features
  -> intraday_pnl_explain.modeling.train (walk-forward: persistence, rolling mean, ridge)
  -> intraday_pnl_explain.evaluation (metrics + diagnostics)
  -> outputs/demo_run/{metrics.json,predictions.parquet,coefficients.csv,figures/*.png}
```

ClickHouse support is optional and one-time only for refreshing raw cache payloads. It is not required for default runtime commands or notebook execution.

## Install

`uv sync --extra dev`

## Run Offline Demo

`uv run python -m intraday_pnl_explain.app.cli run-offline-demo`

Optional output location:

`uv run python -m intraday_pnl_explain.app.cli run-offline-demo --output-dir outputs/demo_run`

Optional one-time refresh entrypoint (expected offline-only failure in this repo clone):

`uv run python -m intraday_pnl_explain.app.cli build-raw-manifest`

## Run Notebook

`uv run python -m nbconvert --to notebook --execute --inplace notebooks/intraday_variance_walkthrough.ipynb`

## Run Tests

`uv run python -m pytest -v`

## What This Demonstrates In Interviews

- A complete offline research workflow from raw bars to evaluated forecasts.
- Correct realized-variance target construction from intraday log returns.
- Leak-aware feature engineering and walk-forward evaluation.
- Baseline-versus-linear-model comparison with interpretable outputs.
- Portable reproducibility: clone + tracked `data/raw/` is enough to run.

## Scope Limits

- One target only: next-day daily realized variance (modeled in log space).
- One dataset family only: intraday bars.
- Two naive baselines plus one ridge benchmark.
- No HTML reporting layer, dashboards, or web application surface.
- No hyperparameter search, deep learning, or live database dependency.
