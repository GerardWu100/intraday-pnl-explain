# GUIDE_ROOT

## Part 1: Conceptual explanation

The repository root is organized as a compact offline quantitative research project.

Core root-level roles:

- `src/` contains runtime implementation code only.
- `data/raw/` contains tracked intraday parquet bars and one raw manifest.
- `tests/` verifies contract, formula, and end-to-end behavior.
- `notebooks/` provides a teaching walkthrough that imports `src/` modules.
- `docs/` documents run workflows and modeling assumptions.

The main runtime path is:

`data/raw -> src pipeline -> outputs/demo_run`

## Part 2: Code reference

- `README.md`: project story, architecture, and run commands.
- `pyproject.toml`: dependencies and packaging metadata.
- `src/`: offline data access, realized variance, features, modeling, evaluation, and pipeline orchestration.
- `data/raw/`: tracked raw parquet files and `raw_manifest.json`.
- `tests/`: unit and integration tests aligned to the realized-variance domain.
- `notebooks/intraday_variance_walkthrough.ipynb`: executable teaching notebook.
- `docs/user/running-the-project.md`: user runbook for offline pipeline and notebook.
- `docs/reference/assumptions.md`: quant assumptions and scope boundaries.
- `GUIDE_OVERVIEW.md`: architecture and data-flow summary.

Suggested read order:

1. `README.md`
2. `GUIDE_OVERVIEW.md`
3. `src/intraday_pnl_explain/pipeline/run_offline_demo.py`
4. `src/intraday_pnl_explain/realized_variance/construct.py`
5. `src/intraday_pnl_explain/modeling/train.py`

## Part 3: Short journal

- 2026-04-19: Rewrote root guide for offline realized-variance research refactor.
- 2026-07-13: Replaced the infeasible held-out-mean score with forecast skill
  versus persistence and documented the demo data's missing provenance.
