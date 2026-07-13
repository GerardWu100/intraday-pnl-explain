# GUIDE_features

## Part 1: Conceptual explanation

`features/` builds leak-free predictor columns from the daily realized-variance
target series.

Feature rows are indexed by feature date $d$ and the next observed target date.
The forecast is formed after day $d$ closes, so current-day log realized
variance, its trailing five-session mean and standard deviation, its absolute
one-session change, and session completeness are available. The explicit names
avoid calling a current-day value a lag or calling a variance change a range.

## Part 2: Code reference

- `build_features.py`: log-transform target construction, lagged features,
  rolling summary features, and next-day target alignment.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added compact feature builder for next-day RV forecasting.
- 2026-07-13: Renamed predictors to match their timing and financial meaning.
