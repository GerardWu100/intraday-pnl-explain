# Modeling Assumptions

- Project question: forecast next-day realized variance from lagged intraday-derived features.
- Intraday return is log return: `r_t = log(P_t / P_{t-1})`.
- Daily realized variance is `RV_d = sum_t r_t^2` for each symbol-day.
- Modeling target is `log(RV_{d+1})`, not variance in levels.
- The forecast origin is after day `d` closes. Feature date is day `d`; target
  date is the next observed session for that symbol.
- Evaluation uses walk-forward time splits only; no random train/test shuffles.
- A training row is eligible only when its target date is no later than the
  current forecast origin. This label-availability purge protects gaps.
- Baselines are persistence and rolling-mean forecasts in log-RV space.
- Linear benchmark is ridge regression on standardized feature columns.
- Scaling moments are estimated from each training window only. Reported slope
  coefficients therefore represent a one-training-standard-deviation feature move.
- Model skill is `1 - SSE_model / SSE_persistence`, where `SSE` is sum of squared
  error on identical held-out rows. Persistence has skill zero by construction.
- Annualization is not applied. Daily RV is in decimal-return-squared units;
  annual variance would be `252 * RV` under a comparable-day assumption.
- The six symbols from one held-out date are correlated. Six forecast rows do
  not provide six independent time-series observations or credible inference.
- The raw manifest does not state data vendor, extraction method, or whether the
  prices are synthetic. Results are workflow evidence only.
- Offline runs read tracked `data/raw` parquet files only.
- Optional ClickHouse extraction is one-time cache population and is excluded from default runtime path.
- Output scope is non-HTML artifacts only: JSON, parquet, CSV, and PNG.
