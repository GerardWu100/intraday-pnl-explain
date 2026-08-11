# intraday-pnl-explain

An offline quant research demo that asks one question: do simple lagged intraday realized-variance features forecast next-day realized variance better than naive baselines? Everything runs from parquet bars already tracked in this repo, so cloning is enough to reproduce a run.

## What it does

The pipeline builds daily realized variance per symbol from intraday log returns, engineers lagged/rolling features, and walk-forward evaluates three forecasters of next-day log realized variance:

- persistence (tomorrow = today)
- rolling mean
- ridge regression on standardized features

For symbol $i$ and bar $t$, the intraday log return is $r_{i,t} = \log(P_{i,t} / P_{i,t-1})$, and daily realized variance is:

$$
RV_{i,d} = \sum_{t \in d} r_{i,t}^2
$$

The model target is $\log(RV_{i,d+1})$. Each forecast origin only sees information available after day $d$ closes; a label-availability purge keeps unobserved future targets out of training. Full assumptions and formulas are in `docs/reference/assumptions.md`.

The tracked demo payload covers six symbols (AAPL, CVX, JPM, MSFT, NVDA, XOM) over a short date range with one held-out forecast date. That is enough to exercise the workflow end to end, not enough to draw statistical conclusions about real market behavior — see `docs/reference/assumptions.md` and `docs/reference/raw_data_contract.md` for why (the raw manifest also does not state data vendor or provenance).

## Requirements

- Python >= 3.13
- No external service is required for the default runtime path.
- `.env` declares `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY` for a possible future one-time cache-refresh path, but no code currently reads them. The `build-raw-manifest` CLI command is a stub that always raises a clear "offline only" error.

## Setup

```
uv sync --extra dev
```

The `dev` extra brings in `pytest`, `jupyter`, and `ipykernel`, needed for tests and the notebook.

## Usage

- `uv run python -m intraday_pnl_explain.app.cli run-offline-demo` — run the full pipeline and write artifacts to `outputs/demo_run` (override with `--output-dir`).
- `uv run python -m intraday_pnl_explain.app.cli build-raw-manifest` — stub for a one-time ClickHouse refresh; always fails offline with an explanatory message.
- `uv run python -m nbconvert --to notebook --execute --inplace notebooks/intraday_variance_walkthrough.ipynb` — run the teaching notebook, which calls the real `src/` modules.
- `uv run python -m pytest -v` — run the test suite.

## Configuration

Settings live in `src/intraday_pnl_explain/app/config.toml`:

- `[paths] raw_root` — where tracked raw parquet bars live (`data/raw`).
- `[paths] default_output_directory` — default artifact directory (`outputs/demo_run`).
- `[modeling] min_train_dates` — feature dates required before the first out-of-sample date is scored. The tracked payload has five feature dates, so values above 4 produce no predictions.
- `[modeling] ridge_alpha` — L2 regularization strength for the ridge benchmark.

## Layout

```
src/intraday_pnl_explain/   importable package: app, data_access, realized_variance, features, modeling, evaluation, pipeline
data/raw/                   tracked intraday parquet bars and raw_manifest.json
notebooks/                  teaching notebook that imports src/ modules
tests/                      unit and integration tests
docs/                       user runbook and quant assumptions/data contract
outputs/                    generated run artifacts (not tracked in git)
```

## Output

`run-offline-demo` writes to `outputs/demo_run/` (or `--output-dir`):

- `metrics.json` — error metrics and skill versus persistence
- `predictions.parquet` — per-date, per-symbol forecasts
- `coefficients.csv` — fitted ridge coefficients
- `figures/*.png` — realized-variance history, prediction-vs-actual, residual distribution, coefficient bar chart

## License

All rights reserved. See [LICENSE](LICENSE).
