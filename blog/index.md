---
title: "Forecasting Tomorrow's Realized Variance Without Borrowing Tomorrow's Data"
description: "A post-close, walk-forward realized-variance experiment with explicit units, label purging, a feasible persistence benchmark, and honest limits on a ten-session demo."
date: 2026-07-13
image: images/cover-realized-variance-forecast.png
categories: ["Quantitative Research", "Risk Management"]
---

# Forecasting Tomorrow's Realized Variance Without Borrowing Tomorrow's Data

A volatility model can look impressive because of one bad timestamp. A rolling window can reach into the target session, a scaler can learn from the test set, or a score can compare the model with a benchmark that was unknowable when the forecast was made. None of those failures requires complicated code.

This project asks a deliberately small question: after today's regular session closes, can today's and recent intraday realized variance forecast the next session better than persistence? The tracked fixture has 23,400 one-minute rows for AAPL, MSFT, NVDA, JPM, XOM, and CVX over ten sessions from March 24 through April 6, 2026.

That sample is useful for testing a research pipeline. It cannot establish a trading result. The manifest also omits the data vendor, extraction method, and whether the prices are synthetic. I therefore treat the bars as a reproducible fixture, not verified historical market evidence.

## The forecast clock

The forecast origin is after the cash close on date $d$. At that point, every regular-session price on date $d$ is known. The prediction concerns the next observed session, denoted $d+1$ for compactness.

| Stage | Latest information allowed | Shape in this fixture |
|---|---|---:|
| Raw bars | 09:30 to 15:59 New York time on date $d$ | 390 prices per symbol-session |
| Daily measure | Returns formed only within symbol and date $d$ | 389 returns per symbol-session |
| Features | Current and trailing realized variance through date $d$ | 30 symbol-date rows after warm-up |
| Training labels | Target dates no later than forecast origin $d$ | 24 rows across four dates |
| Held-out forecasts | Features dated April 3, target dated April 6 | 6 rows on one date |

The six held-out rows are a cross-section, not six independent test periods. A market-wide volatility shock can move all six stocks together. The effective time-series test size is one.

## From minute prices to realized variance

For symbol $i$, trading date $d$, and intraday observation $t$, let $P_{i,d,t}$ be the price in dollars. The one-minute log return $r_{i,d,t}$ is

$$
r_{i,d,t}
= \log\left(\frac{P_{i,d,t}}{P_{i,d,t-1}}\right).
$$

A log return is dimensionless. Squaring it gives decimal return squared. Daily realized variance $RV_{i,d}$ sums the squared returns within the regular session:

$$
RV_{i,d}
= \sum_{t=1}^{T_d} r_{i,d,t}^{2},
$$

where $T_d$ is the number of usable within-session returns. The loader first converts Coordinated Universal Time (UTC) timestamps to `America/New_York`. The return shift then occurs inside each symbol-date group, which excludes the overnight price move.

```python
group_columns = ["symbol", "date"]
prepared_bars["log_return"] = prepared_bars.groupby(group_columns)["price"].transform(
    lambda values: np.log(values / values.shift(1))
)

realized_daily = prepared_bars.groupby(group_columns, as_index=False).agg(
    realized_variance=("log_return", _sum_squared_log_returns),
    bar_count=("log_return", _non_missing_log_return_count),
)
```

Every one of the 60 symbol-session groups has 390 prices and 389 returns. That count establishes internal completeness, but not market-data provenance or price semantics. The raw contract calls the field `price`. It does not say whether that value is a bar close, midpoint, or trade.

![Daily realized variance for six symbols](images/01_realized_variance_by_symbol.png)

The panels show a wide scale range. NVDA reaches about $5.89 \times 10^{-6}$, while the sample minimum is about $9.45 \times 10^{-10}$. The model uses log variance so the largest observation does not dominate a squared-error regression in levels.

The target stays in one-session units. If $A=252$ is the assumed number of trading sessions per year, annualized variance and volatility would be

$$
RV^{\mathrm{ann}}_{i,d}=A\,RV_{i,d},
\qquad
\sigma^{\mathrm{ann}}_{i,d}=\sqrt{A\,RV_{i,d}}.
$$

This annualization assumes comparable daily variance across the year. In log space it only adds a constant:

$$
\log(RV^{\mathrm{ann}}_{i,d})=\log(A)+\log(RV_{i,d}).
$$

Adding the same constant to actuals and forecasts leaves every forecast error unchanged, so annualization would not change the model ranking.

## Features and target alignment

Define $y_{i,d+1}$ as next-session log realized variance and $\mathbf{x}_{i,d}$ as the feature vector available after date $d$ closes:

$$
y_{i,d+1}=\log(RV_{i,d+1}),
\qquad
\widehat y_{i,d+1}=f(\mathbf{x}_{i,d}).
$$

The five features are current log variance, its trailing five-session mean, its trailing five-session standard deviation, its absolute one-session change, and the fraction of expected returns present. Their names now state their timing directly. The earlier names described current variance as `lag_1` and an absolute variance change as a `range_proxy`, which invited the wrong financial interpretation.

```python
frame["current_log_rv"] = frame["log_rv"]
frame["trailing_5_mean_log_rv"] = frame.groupby("symbol")["log_rv"].transform(
    lambda values: values.rolling(5).mean()
)
frame["target_date"] = frame.groupby("symbol")["date"].shift(-1)
frame["target_log_rv_next_day"] = frame.groupby("symbol")["log_rv"].shift(-1)
```

The rolling window consumes the first four sessions. That leaves five feature dates. For the held-out forecast origin on April 3, training feature dates end on April 2 and their targets end on April 3.

A date-order split alone is not enough when a symbol has missing sessions. A row dated earlier can point to a target later than the current forecast origin. The walk-forward fit now applies both conditions:

```python
train_frame = frame[
    (frame["feature_date"] < test_date) & (frame["target_date"] <= test_date)
].copy()
```

The second condition is a label-availability purge. At the April 3 post-close cutoff, the April 3 realized variance is observable and may serve as a training label. April 6 realized variance is not observable and remains the test target.

## Three forecasting rules

Persistence predicts that next-session log variance equals current log variance. The trailing-mean rule predicts the five-session average. Both are feasible at the forecast origin.

Ridge regression combines the five features. For $n$ training rows, let $y_j$ be the target, $\mathbf{z}_j$ the standardized feature vector, $\beta_0$ the intercept, $\boldsymbol{\beta}$ the five slopes, and $\lambda \ge 0$ the penalty. The fitted parameters solve

$$
(\widehat{\beta}_0,\widehat{\boldsymbol{\beta}})
= \arg\min_{\beta_0,\boldsymbol{\beta}}
\left[
\sum_{j=1}^{n}
\left(y_j-\beta_0-\mathbf{z}_j^{\mathsf T}\boldsymbol{\beta}\right)^2
+\lambda\sum_{k=1}^{5}\beta_k^2
\right].
$$

For feature $k$, the code calculates the training mean $\mu_k$ and training standard deviation $s_k$, then sets

$$
z_{j,k}=\frac{x_{j,k}-\mu_k}{s_k}.
$$

The test row uses the same $\mu_k$ and $s_k$. It never contributes to them. The project sets $\lambda=1$ without tuning because four training dates cannot support a credible hyperparameter search.

The 24 training rows are pooled across six symbols. They are not 24 independent volatility histories. The five features are also strongly related by construction, so the single fitted coefficient vector is unstable. Its standardized slopes range from -0.713 for current log variance to 0.942 for trailing variability, but those signs should not be read as economic estimates.

## A feasible score against persistence

For $m$ held-out forecasts, let $e_{M,j}=y_j-\widehat y_{M,j}$ be model $M$'s log-variance error. Root mean squared error (RMSE) and mean absolute error (MAE) are

$$
\mathrm{RMSE}_M
=\sqrt{\frac{1}{m}\sum_{j=1}^{m}e_{M,j}^2},
\qquad
\mathrm{MAE}_M
=\frac{1}{m}\sum_{j=1}^{m}|e_{M,j}|.
$$

Let $SSE_M=\sum_j e_{M,j}^2$ be model squared error and $SSE_P$ be persistence squared error on the identical rows. Persistence-relative forecast skill is

$$
\mathrm{Skill}_{M\mid P}
=1-\frac{SSE_M}{SSE_P}.
$$

Persistence has skill zero by construction. Positive skill improves on persistence. Negative skill is worse. This replaces the earlier reported statistic whose denominator used the held-out outcomes' own cross-sectional mean. That mean is known only after the outcomes arrive, so it was not a feasible forecasting benchmark.

| Model | RMSE | MAE | Skill vs. persistence |
|---|---:|---:|---:|
| Persistence | 1.300 | 1.209 | 0.000 |
| Five-session mean | 1.604 | 1.427 | -0.522 |
| Ridge | 1.190 | 1.089 | 0.162 |

![Forecast error comparison](images/02_model_error_comparison.png)

Ridge reduces squared error by 16.19 percent relative to persistence on this one cross-section. The five-session mean increases it by 52.20 percent. Ridge's RMSE is 8.45 percent lower than persistence's because RMSE takes the square root of average squared error.

The absolute errors are economically large. Since the target is logarithmic, an absolute log error $a$ corresponds to a multiplicative variance ratio of $\exp(a)$. Exponentiating ridge's MAE gives $\exp(1.089)\approx2.97$. This is a scale summary, not a confidence interval, but it makes clear that “best” does not mean “accurate.”

![Actual and forecast next-day log realized variance](images/03_forecast_cross_section.png)

Actual log variance on April 6 ranges from -16.86 for JPM to -14.41 for CVX. Ridge overpredicts four symbols, not all six: it underpredicts CVX and XOM. The line follows part of the cross-sectional ordering, but one date cannot tell whether that ordering came from a reusable relation or chance.

## Why this is not a P&L model

Realized variance is a risk input, not profit and loss (P&L). A variance forecast can inform option hedging, volatility targeting, margin, or scenario sizing, but each use needs another layer that maps variance into positions and cash flows.

For intuition only, consider a delta-hedged option over a short horizon. Let $S$ be spot price, $\Gamma$ be option gamma in currency per price squared, $RV$ be realized variance over the horizon, and $v_{\mathrm{imp}}$ be the horizon-matched implied variance priced at inception. Holding spot and gamma fixed gives the rough approximation

$$
\mathrm{P\&L}_{\Delta\text{-hedged}}
\approx \frac{1}{2}\Gamma S^2\left(RV-v_{\mathrm{imp}}\right).
$$

The project does not test that equation. Its $RV$ excludes overnight returns, while an option lives through the full clock. Gamma changes with spot and time, transaction costs matter, and a forecast of $\log(RV)$ is not automatically a forecast of mean variance. Because the exponential function is convex, $\exp(\widehat y)$ estimates a conditional median under common log-error assumptions, not the conditional mean, unless a bias correction is added.

## What the experiment proves

The reproducible run verifies session assignment, within-session return construction, feature timing, label availability, train-only scaling, three forecast rules, and artifact generation. Regression tests include a hand-calculated realized-variance example, a missing-date purge case, and a hand-calculated persistence-skill case.

It does not prove predictive power. A serious study needs years of documented market data, many walk-forward dates across volatility regimes, symbol- and date-level error reporting, uncertainty for paired loss differences, and a decision-specific treatment of overnight variance. Sampling frequency also deserves testing because one-minute prices can contain market microstructure noise.

The most defensible result is therefore procedural: on six forecasts from one date, ridge has positive skill against persistence, but the sample has essentially no power to distinguish a durable forecasting edge from noise.

## References

- Andersen, Bollerslev, Diebold, and Labys, [“The Distribution of Realized Exchange Rate Volatility”](https://doi.org/10.1198/016214501750332965), *Journal of the American Statistical Association*, 2001.
- Corsi, [“A Simple Approximate Long-Memory Model of Realized Volatility”](https://doi.org/10.1093/jjfinec/nbp001), *Journal of Financial Econometrics*, 2009.
- Hansen and Lunde, [“Realized Variance and Market Microstructure Noise”](https://doi.org/10.1198/073500106000000071), *Journal of Business & Economic Statistics*, 2006.
- Hoerl and Kennard, [“Ridge Regression: Biased Estimation for Nonorthogonal Problems”](https://doi.org/10.1080/00401706.1970.10488634), *Technometrics*, 1970.
- Campbell and Thompson, [“Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?”](https://doi.org/10.1093/rfs/hhm055), *Review of Financial Studies*, 2008. The paper's feasible benchmark logic motivates the persistence-relative skill comparison here.
