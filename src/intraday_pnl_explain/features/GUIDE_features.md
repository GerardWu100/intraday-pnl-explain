# GUIDE_features

## Part 1: Conceptual explanation

`features/` builds leak-free predictor columns from the daily realized-variance
target series.

Feature rows are indexed by feature date $d$ and target date $d+1$, so every
feature uses information available at the end of day $d$ only.

## Part 2: Code reference

- `build_features.py`: log-transform target construction, lagged features,
  rolling summary features, and next-day target alignment.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added compact feature builder for next-day RV forecasting.
