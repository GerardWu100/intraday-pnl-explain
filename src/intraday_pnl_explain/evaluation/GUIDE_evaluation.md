# GUIDE_evaluation

## Part 1: Conceptual explanation

`evaluation/` computes out-of-sample error metrics and writes static diagnostic
figures.

Metric scope is constrained to interpretable defaults:

- root mean squared error (RMSE),
- mean absolute error (MAE),
- out-of-sample coefficient of determination ($R^2$).

## Part 2: Code reference

- `metrics.py`: per-model metric aggregation.
- `diagnostics.py`: realized-variance history, prediction-vs-actual,
  residual histogram, and coefficient bar chart exports.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added lightweight evaluation module for offline artifacts.
