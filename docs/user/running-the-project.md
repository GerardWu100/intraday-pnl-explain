# Running The Project

## Install dependencies

`uv sync --extra dev`

## Run the offline pipeline demo

Default output path (`outputs/demo_run`):

`uv run python -m intraday_pnl_explain.app.cli run-offline-demo`

Custom output path:

`uv run python -m intraday_pnl_explain.app.cli run-offline-demo --output-dir outputs/demo_run`

## Execute the notebook offline

`uv run python -m nbconvert --to notebook --execute --inplace notebooks/intraday_variance_walkthrough.ipynb`

## Run tests

`uv run python -m pytest -v`

## Optional one-time ClickHouse refresh path

The runtime path does not require ClickHouse. This command exists only to make
the one-time refresh path explicit:

`uv run python -m intraday_pnl_explain.app.cli build-raw-manifest`

In this offline-first repository clone, the command returns a clear failure
message because no external ClickHouse source is configured.
