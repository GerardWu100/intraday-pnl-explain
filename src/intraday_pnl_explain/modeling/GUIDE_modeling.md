# GUIDE_modeling

## Part 1: Conceptual explanation

`modeling/` trains simple, defensible forecasting models with walk-forward
evaluation logic.

Current model set is intentionally small:

- persistence baseline,
- rolling-mean baseline,
- ridge regression on standardized features.

## Part 2: Code reference

- `train.py`: walk-forward split loop, baseline feature-column predictors,
  ridge fitting, prediction export rows, and coefficient collection.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added walk-forward baseline and ridge modeling layer.
- 2026-05-19: Inlined baseline predictors into `train.py` and removed `baselines.py`.
