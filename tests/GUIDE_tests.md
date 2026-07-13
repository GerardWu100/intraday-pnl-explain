# GUIDE_tests

## Part 1: Conceptual explanation

`tests/` protects the offline realized-variance pipeline against regressions.

Coverage is intentionally focused on:

- raw-data contract integrity,
- formula correctness,
- leakage prevention,
- walk-forward modeling behavior,
- target-label availability at each forecast origin,
- forecast skill relative to persistence on matched rows,
- end-to-end artifact generation,
- notebook structure and execution.

## Part 2: Code reference

- `test_raw_contract.py`: manifest consistency, raw parquet location constraints, and payload size guardrail.
- `test_realized_variance.py`: hand-checked `sum(log_return^2)` validation.
- `test_feature_building.py`: feature-target alignment and lookahead leakage prevention.
- `test_modeling_pipeline.py`: walk-forward ordering and prediction-shape checks.
- `test_evaluation_metrics.py`: hand-checked persistence-relative skill score.
- `test_offline_demo.py`: end-to-end offline artifact generation and non-HTML output check.
- `test_cli.py`: package version plus CLI command behavior.
- `test_notebook_walkthrough.py`: markdown/code alternation and offline execution.

Primary verification command:

- `uv run python -m pytest -v`

## Part 3: Short journal

- 2026-04-19: Replaced legacy PnL/stress tests with realized-variance research tests.
- 2026-07-13: Added regressions for label purging and feasible benchmark scoring.
