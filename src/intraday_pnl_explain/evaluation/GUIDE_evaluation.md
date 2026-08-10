# GUIDE_evaluation

## Part 1: Conceptual explanation

`evaluation/` computes out-of-sample error metrics and writes static diagnostic
figures.

Metric scope is constrained to interpretable defaults:

- root mean squared error (RMSE),
- mean absolute error (MAE),
- squared-error forecast skill relative to persistence.

For model $M$, skill is $1-SSE_M/SSE_P$, where $SSE_M$ is the model's sum of
squared errors and $SSE_P$ is persistence's sum on the same forecast rows. Zero
matches persistence; positive values improve on it; negative values are worse.
This feasible benchmark replaces an earlier statistic whose denominator used
the held-out outcomes' own mean.

## Part 2: Code reference

- `metrics.py`: per-model metric aggregation.
- `diagnostics.py`: realized-variance history, prediction-vs-actual,
  residual histogram, and coefficient bar chart exports.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added lightweight evaluation module for offline artifacts.
- 2026-07-13: Replaced held-out-mean $R^2$ with forecast skill versus persistence.
- 2026-08-10: Prediction-versus-actual figure now plots markers instead of a
  connecting line. Several symbols share one target date, so a line drew a path
  between different symbols rather than a time series.
