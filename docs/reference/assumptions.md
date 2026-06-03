# Modeling Assumptions

- Project question: forecast next-day realized variance from lagged intraday-derived features.
- Intraday return is log return: `r_t = log(P_t / P_{t-1})`.
- Daily realized variance is `RV_d = sum_t r_t^2` for each symbol-day.
- Modeling target is `log(RV_{d+1})`, not variance in levels.
- Feature date is day `d`; target date is day `d+1`; no lookahead information is allowed.
- Evaluation uses walk-forward time splits only; no random train/test shuffles.
- Baselines are persistence and rolling-mean forecasts in log-RV space.
- Linear benchmark is ridge regression on standardized feature columns.
- Annualization is not required for this target because comparisons are in one-day horizon log-RV units.
- Offline runs read tracked `data/raw` parquet files only.
- Optional ClickHouse extraction is one-time cache population and is excluded from default runtime path.
- Output scope is non-HTML artifacts only: JSON, parquet, CSV, and PNG.
