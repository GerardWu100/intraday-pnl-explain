# GUIDE_docs

## Part 1: Conceptual explanation

`docs/` stores user documentation, reference assumptions, and planning artifacts.

For this project shape:

- `docs/user/` explains how to run offline pipeline and notebook commands.
- `docs/reference/` defines quantitative assumptions and raw-data contracts.
When runtime behavior changes, update user and reference docs in the same session.

## Part 2: Code reference

- `user/running-the-project.md`: canonical runbook for offline demo, tests, and notebook execution.
- `reference/assumptions.md`: realized-variance modeling assumptions and limits.
- `reference/raw_data_contract.md`: tracked raw parquet schema and partition layout.

## Part 3: Short journal

- 2026-04-19: Rewrote docs guide for offline realized-variance project scope.
