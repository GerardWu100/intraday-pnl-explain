"""Configuration helpers for offline research CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime configuration for offline CLI commands.

    Parameters
    ----------
    project_root
        Absolute path to the repository root, derived from this file's location.
    raw_root
        Absolute path to the tracked raw-data folder.
    default_output_directory
        Absolute path used for demo artifacts when the CLI gets no --output-dir.
    min_train_dates
        Distinct feature dates required before the walk-forward loop scores its
        first out-of-sample date.
    ridge_alpha
        L2 regularization strength used by the ridge benchmark.
    """

    project_root: Path
    raw_root: Path
    default_output_directory: Path
    min_train_dates: int
    ridge_alpha: float


def load_app_config() -> AppConfig:
    """Load TOML config and resolve script-relative project paths."""
    config_path = Path(__file__).resolve().with_name("config.toml")
    project_root = config_path.parents[3]

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = payload["paths"]
    modeling = payload["modeling"]
    raw_root = project_root / paths["raw_root"]
    default_output_directory = project_root / paths["default_output_directory"]

    return AppConfig(
        project_root=project_root,
        raw_root=raw_root,
        default_output_directory=default_output_directory,
        min_train_dates=int(modeling["min_train_dates"]),
        ridge_alpha=float(modeling["ridge_alpha"]),
    )
