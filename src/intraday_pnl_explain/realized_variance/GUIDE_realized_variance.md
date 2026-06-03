# GUIDE_realized_variance

## Part 1: Conceptual explanation

`realized_variance/` converts intraday price bars into one daily target per symbol.

For intraday return $r_t = log(P_t / P_{t-1})$, daily realized variance is:

$$
RV_d = \sum_{t \in d} r_t^2
$$

The module also reports realized volatility, bar-count coverage, and a simple
session-completeness flag.

## Part 2: Code reference

- `construct.py`: computes intraday log returns and daily realized variance table.
- `__init__.py`: package marker only.

## Part 3: Short journal

- 2026-04-19: Added daily realized-variance constructor for offline pipeline.
