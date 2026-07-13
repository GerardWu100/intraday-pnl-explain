# Project Overview

## Purpose

`intraday-pnl-explain` is an offline-first realized-variance research project.

The central research question is:

Can simple lagged intraday realized-variance features forecast next-day realized variance better than naive baselines?

The project runs from tracked local parquet files and does not require live database access on the default runtime path.

## Project Layout

```text
intraday-pnl-explain/
|- src/
|  |- intraday_pnl_explain/    # Importable package root
|     |- app/                 # CLI orchestration for offline commands
|     |- data_access/         # Raw manifest and raw bar loading
|     |- realized_variance/   # Daily target construction
|     |- features/            # Leak-aware feature engineering
|     |- modeling/            # Baselines and ridge walk-forward training
|     |- evaluation/          # Metrics and diagnostic plots
|     |- pipeline/            # End-to-end offline pipeline
|- data/raw/                   # Tracked intraday bars and raw manifest
|- notebooks/                  # Teaching notebook walkthrough
|- tests/                      # Unit and integration tests
|- docs/                       # User docs, reference docs, plan/spec artifacts
|- README.md                   # Quick start and project story
|- pyproject.toml              # Dependencies and packaging
```

## Architecture and Data Flow

Default runtime flow:

1. Load and validate `data/raw/raw_manifest.json`.
2. Load and normalize bars from `data/raw/intraday_bars/`.
3. Build daily realized variance from intraday log returns.
4. Build lagged and rolling predictors with next-day target alignment.
5. At each post-close forecast origin, purge labels not yet observable.
6. Fit persistence, rolling-mean, and ridge models with expanding walk-forward splits.
7. Export errors and squared-error skill relative to persistence.

Conceptual flow:

```text
CLI command
  -> raw manifest + raw bars
  -> realized variance
  -> feature matrix
  -> walk-forward models
  -> metrics + parquet/csv/png outputs
```

## Quant Model (High Level)

For symbol $i$ and intraday bar index $t$:

- $P_{i,t}$ is bar price.
- $r_{i,t} = \log(P_{i,t} / P_{i,t-1})$ is intraday log return.

For date $d$, daily realized variance is:

$$
RV_{i,d} = \sum_{t \in d} r_{i,t}^2
$$

The forecast target is next-day log realized variance:

$$
\log(RV_{i,d+1})
$$

All predictors are constructed from information available after day $d$ closes.
Training rows must also satisfy `target_date <= d`, so a missing symbol date cannot
make an unobserved future label enter the fit.

The daily measure has units of decimal return squared. The project keeps the
one-session target unannualized. Multiplying variance by 252, or volatility by
$\sqrt{252}$, would only rescale the target under a same-distribution assumption.

## Offline Data Contract

Tracked payload under `data/raw/` includes:

- `raw_manifest.json` with dataset schema and coverage metadata,
- partitioned parquet bars under `intraday_bars/symbol=.../part-000.parquet`,
- narrow required columns: `symbol`, `timestamp`, `price`, `volume`.

This contract is sufficient for offline pipeline and notebook execution.

## Outputs and Surfaces

- CLI commands: `run-offline-demo`, `build-raw-manifest`.
- Artifacts: `metrics.json`, `predictions.parquet`, `coefficients.csv`, `figures/*.png`.
- Teaching surface: `notebooks/intraday_variance_walkthrough.ipynb`.

## Scope and Limits

- One target only: next-day realized variance.
- Two naive baselines plus one ridge benchmark.
- The demo has one held-out date and six forecasts, so model rankings have no
  useful statistical power.
- Raw-data provenance is absent from the tracked manifest. Treat the payload as
  a workflow fixture, not verified historical market data.
- No HTML report surface, dashboard, or web app.
- Optional ClickHouse step is separated from default runtime.
