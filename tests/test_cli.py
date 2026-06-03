"""Smoke and integration tests for package and CLI research workflows."""

from pathlib import Path
from subprocess import run

from intraday_pnl_explain import __version__


def test_package_exposes_version() -> None:
    """Verify the package can be imported and exposes a semantic version."""
    assert __version__ == "0.1.0"


def test_run_offline_demo_command_runs_successfully(tmp_path: Path) -> None:
    """CLI should execute the offline realized-variance demo end to end."""
    output_directory = tmp_path / "demo_run"
    completed = run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "intraday_pnl_explain.app.cli",
            "run-offline-demo",
            "--output-dir",
            str(output_directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Offline demo completed" in completed.stdout
    assert (output_directory / "metrics.json").exists()
    assert (output_directory / "predictions.parquet").exists()


def test_build_raw_manifest_command_without_clickhouse_fails_cleanly() -> None:
    """Raw-manifest refresh should fail cleanly in offline-only mode."""
    completed = run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "intraday_pnl_explain.app.cli",
            "build-raw-manifest",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ClickHouse extraction is optional" in completed.stderr


def test_teaching_notebook_exists() -> None:
    """Teaching walkthrough notebook should be tracked in the repository."""
    notebook_path = Path("notebooks/intraday_variance_walkthrough.ipynb")
    assert notebook_path.exists()
