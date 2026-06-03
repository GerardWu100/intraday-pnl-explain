"""CLI entry points for offline realized-variance research workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from intraday_pnl_explain.app.config import load_app_config
from intraday_pnl_explain.data_access.clickhouse_extract import (
    raise_optional_clickhouse_message,
)
from intraday_pnl_explain.pipeline.run_offline_demo import run_offline_demo


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for offline demo and optional extract command."""
    parser = argparse.ArgumentParser(prog="intraday-pnl-explain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline_demo_parser = subparsers.add_parser("run-offline-demo")
    offline_demo_parser.add_argument(
        "--output-dir",
        required=False,
        help="Directory where metrics, predictions, coefficients, and figures are written",
    )

    subparsers.add_parser("build-raw-manifest")

    return parser


def run_selected_command(args: argparse.Namespace) -> int:
    """Route parsed CLI arguments to the selected command handler."""
    config = load_app_config()

    if args.command == "run-offline-demo":
        output_directory = (
            Path(args.output_dir).resolve()
            if args.output_dir is not None
            else config.default_output_directory
        )
        run_offline_demo(output_directory=output_directory)
        print(f"Offline demo completed. Artifacts written to: {output_directory}")
        return 0

    # Optional extract path is offline-only in this repo; surface a clear failure.
    try:
        raise_optional_clickhouse_message()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1


def main() -> None:
    """Parse command-line arguments and exit with command status code."""
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_selected_command(args))


if __name__ == "__main__":
    main()
