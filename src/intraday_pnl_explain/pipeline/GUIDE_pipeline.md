# GUIDE_pipeline

## Part 1: Conceptual explanation

`pipeline/` is the end-to-end orchestration layer for the offline research demo.

Data flow:

1. raw manifest load,
2. raw bars load and normalization,
3. realized variance construction,
4. feature matrix build,
5. walk-forward model training,
6. metrics and diagnostics export.

## Part 2: Code reference

- `run_offline_demo.py`: one callable function that runs the full workflow and
  writes offline artifacts.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added unified offline demo pipeline command.
