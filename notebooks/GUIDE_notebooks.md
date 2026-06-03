# GUIDE_notebooks

## Part 1: Conceptual explanation

`notebooks/` contains the teaching surface for the project.

Notebook design rules:

- substantial markdown before each code cell,
- strict markdown/code alternation,
- imports from `src/` modules instead of duplicating logic,
- offline execution from tracked `data/raw/` only.

## Part 2: Code reference

- `intraday_variance_walkthrough.ipynb`:
  - defines the research question and offline contract,
  - shows raw manifest and parquet coverage,
  - constructs realized variance from intraday bars,
  - builds features and trains walk-forward models,
  - writes and inspects offline artifacts.

Validation command:

- `uv run python -m nbconvert --to notebook --execute --inplace notebooks/intraday_variance_walkthrough.ipynb`

## Part 3: Short journal

- 2026-04-19: Replaced PnL explain walkthrough with realized-variance research notebook.
