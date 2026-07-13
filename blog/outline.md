# Outline proposal

## Project scan summary

- Project archetype candidate: mixed (`data-pipeline` plus `risk-model` forecasting).
- Supporting evidence from files: the project normalizes tracked one-minute parquet bars, constructs daily realized variance, aligns day-$d$ features to a day-$d+1$ target, and compares three forecasting methods with expanding walk-forward splits. The tracked manifest contains 23,400 rows for AAPL, MSFT, NVDA, XOM, CVX, and JPM over 2026-03-24 to 2026-04-06.

## Blueprint selection

- Selected blueprint: mixed.
- Why this blueprint fits this project: the data contract and timestamp handling are as important as the ridge model. A forecasting article that skips the pipeline would hide the main sources of leakage and measurement error.
- Planned section order: the forecasting question; bars-to-realized-variance construction; leak-aware feature and target alignment; walk-forward comparison; observed results; what the sample cannot support.

## Planned equations

1. Intraday log return and daily realized variance:
   - Purpose: define the measured quantity from one-minute prices.
   - Symbols: $P_{i,d,t}$ is the price for symbol $i$, date $d$, and minute $t$; $r_{i,d,t}$ is its intraday log return; $RV_{i,d}$ is daily realized variance.
   - Delimiter: display.
2. Next-day target alignment:
   - Purpose: make the no-lookahead convention explicit.
   - Symbols: $x_{i,d}$ is the feature vector known after day $d$; $y_{i,d+1}=\log(RV_{i,d+1})$ is the target.
   - Delimiter: display.
3. Ridge objective:
   - Purpose: define the regularized linear benchmark.
   - Symbols: $n$ is the number of training observations, $\beta_0$ is the intercept, $\boldsymbol{\beta}$ is the coefficient vector, and $\lambda$ is the L2 penalty strength.
   - Delimiter: display.
4. Evaluation metrics:
   - Purpose: interpret root mean squared error, mean absolute error, and squared-error forecast skill relative to persistence.
   - Symbols: $y_j$ and $\hat y_j$ are actual and forecast log realized variance; $SSE_M$ and $SSE_P$ are model and persistence squared errors on matched rows.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/intraday_pnl_explain/realized_variance/construct.py`
   - Function/block: grouped log-return construction and squared-return aggregation.
   - Why include this excerpt: it is the measurement step on which every downstream result depends.
2. File: `src/intraday_pnl_explain/features/build_features.py`
   - Function/block: day-$d$ predictors and grouped shift to day-$d+1$ target.
   - Why include this excerpt: it shows the anti-leakage alignment directly.
3. File: `src/intraday_pnl_explain/modeling/train.py`
   - Function/block: expanding walk-forward split and train-only standardization.
   - Why include this excerpt: it distinguishes a legitimate time-series evaluation from a random split.

## Planned technical graphs

1. Graph type: six-symbol realized-variance small multiples.
   - Source (reuse or generate): generate from the tracked parquet files through project modules.
   - Expected takeaway: realized variance changes materially by symbol and day even in this narrow ten-session sample.
2. Graph type: forecast error comparison by model.
   - Source (reuse or generate): generate from the pipeline's frozen metrics and predictions.
   - Expected takeaway: compare persistence, rolling mean, and ridge without implying that this sample ranks them reliably out of sample.
3. Graph type: actual versus ridge-predicted next-day log realized variance.
   - Source (reuse or generate): generate from frozen walk-forward predictions.
   - Expected takeaway: show where the linear benchmark tracks the cross-section and where it misses.

## Risks, gaps, and assumptions

- Data gaps: the tracked payload covers only ten trading sessions and six symbols; after rolling-feature warm-up and walk-forward warm-up, the evaluation set is very small.
- Assumptions: the raw `price` field is a minute observation, but the manifest does not establish its exact bar semantics or provenance; microstructure noise and overnight variance are outside the target; the regular session is 09:30 through 15:59 America/New_York; the target is evaluated in log realized-variance units.
- Validation checks to run before final draft: execute the offline pipeline into `blog/data/`; verify session counts, feature dates, prediction counts, and metric calculations; regenerate all graphs; run project tests; run the blog validator on both languages; verify every referenced path.
- Deployment note: the canonical workspace is `intraday-pnl-explain/blog/`. Per the user's instruction, no publish bundle will be copied to `~/projects/website`, no Hugo build will be run there, and no website commit or push will occur in this task.
