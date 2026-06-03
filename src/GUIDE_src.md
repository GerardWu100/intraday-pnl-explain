# GUIDE_src

## Part 1: Conceptual explanation

`src/` contains all executable runtime logic for offline realized-variance research.

Module boundaries are organized around pipeline stages instead of product surfaces:

- data access,
- target construction,
- feature engineering,
- modeling,
- evaluation,
- orchestration.

This design keeps each file focused on one stage and makes notebook usage straightforward.

## Part 2: Code reference

- `intraday_pnl_explain/`: importable package root with version metadata.
- `intraday_pnl_explain/app/`: CLI command parsing and config loading.
- `intraday_pnl_explain/data_access/`: raw manifest validation and raw bar loading.
- `intraday_pnl_explain/realized_variance/`: daily realized-variance construction.
- `intraday_pnl_explain/features/`: lagged and rolling feature building.
- `intraday_pnl_explain/modeling/`: baseline and ridge walk-forward training logic.
- `intraday_pnl_explain/evaluation/`: metrics and diagnostic figure generation.
- `intraday_pnl_explain/pipeline/`: end-to-end orchestration entrypoint.

Suggested read order:

1. `intraday_pnl_explain/pipeline/run_offline_demo.py`
2. `intraday_pnl_explain/data_access/raw_bars.py`
3. `intraday_pnl_explain/realized_variance/construct.py`
4. `intraday_pnl_explain/features/build_features.py`
5. `intraday_pnl_explain/modeling/train.py`
6. `intraday_pnl_explain/evaluation/metrics.py`

## Part 3: Short journal

- 2026-04-19: Replaced PnL explain architecture with realized-variance research pipeline modules.
