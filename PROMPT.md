Build a backend-focused quant demo project called `intraday-pnl-explain`.

Project description:
This project should explain why an equity portfolio made or lost money during the day and support simple stress testing under hypothetical market moves. The goal is a serious but lightweight GitHub project I can discuss in interviews.

Keep it practical, laptop-friendly, and runnable on a normal machine without GPU or special infrastructure. Do not spend effort on frontend, dashboards, or HTML apps. Build it as a proper reusable codebase with real files, modules, config, tests, and docs.

Notebook requirement:
Create a new notebook folder and add a teaching notebook for this project. The notebook should explain the core ideas in markdown, then walk through the project step by step with code cells that call the real project scripts/modules rather than duplicating logic inline. The notebook is for understanding the project, not for fancy visualization.

Cache requirement:
If the repo uses external market data or ClickHouse/Parquet caching, implement and verify an offline-first Parquet cache policy. A valid cache must mean both the Parquet file and its metadata sidecar exist and pass checks. When cache is valid, load from Parquet with no DB call. When cache is missing or invalid and ClickHouse is available, query the DB and refresh the cache. When cache is missing or invalid and ClickHouse is unavailable, fail clearly with an actionable message naming the required files and exact directory. Required input cache Parquet files must be tracked in Git rather than ignored. Update config, tests, ignore rules, and docs accordingly.
