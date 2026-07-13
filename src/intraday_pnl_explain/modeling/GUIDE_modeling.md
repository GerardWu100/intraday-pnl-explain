# GUIDE_modeling

## Part 1: Conceptual explanation

`modeling/` trains three forecasting rules with expanding walk-forward logic.
At forecast origin $d$, a row enters training only if its feature date is before
$d$ and its target date is no later than $d$. The second condition purges labels
that would still be unavailable when a symbol has a data gap.

Current model set is intentionally small:

- persistence baseline,
- rolling-mean baseline,
- ridge regression on standardized features.

Ridge scaling uses training-window means and standard deviations only. Stored
slope coefficients are therefore changes in predicted log variance for a one
training-standard-deviation change in the corresponding feature. With just one
held-out date, coefficient signs are diagnostics rather than stable estimates.

## Part 2: Code reference

- `train.py`: walk-forward split loop, baseline feature-column predictors,
  ridge fitting, prediction export rows, and coefficient collection.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added walk-forward baseline and ridge modeling layer.
- 2026-05-19: Inlined baseline predictors into `train.py` and removed `baselines.py`.
- 2026-07-13: Added target-date purging at each post-close forecast origin.
