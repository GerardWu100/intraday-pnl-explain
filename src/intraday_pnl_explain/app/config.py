"""Configuration helpers for offline research CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime configuration for offline CLI commands."""

    project_root: Path
    raw_root: Path
    default_output_directory: Path


def load_app_config() -> AppConfig:
    """Load TOML config and resolve script-relative project paths."""
    config_path = Path(__file__).resolve().with_name("config.toml")
    project_root = config_path.parents[3]

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = payload["paths"]
    raw_root = project_root / paths["raw_root"]
    default_output_directory = project_root / paths["default_output_directory"]

    return AppConfig(
        project_root=project_root,
        raw_root=raw_root,
        default_output_directory=default_output_directory,
    )
