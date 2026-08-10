# GUIDE_pipeline

## Part 1: Conceptual explanation

`pipeline/` is the end-to-end orchestration layer for the offline research demo.

Data flow:

1. raw manifest load,
2. raw bars load and normalization,
3. realized variance construction,
4. feature matrix build,
5. label-availability-purged walk-forward training,
6. error and persistence-skill export,
7. diagnostic export.

## Part 2: Code reference

- `run_offline_demo.py`: one callable function that runs the full workflow and
  writes offline artifacts.
- `__init__.py`: package marker only.

Walk-forward settings come from `app/config.toml`:

- `[modeling] min_train_dates`: feature dates required before the first scored
  date. The tracked payload has five feature dates, so values above 4 leave no
  forecast origin and the pipeline raises a clear error instead of running.
- `[modeling] ridge_alpha`: L2 regularization strength for the ridge benchmark.

## Part 3: Short journal

- 2026-04-19: Added unified offline demo pipeline command.
- 2026-07-13: Pipeline metrics now use persistence as the feasible forecast benchmark.
- 2026-08-10: Moved `min_train_dates` and `ridge_alpha` out of the call site into
  `app/config.toml`, and added an explicit error when the walk-forward loop
  produces no out-of-sample predictions.
