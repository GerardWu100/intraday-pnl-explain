"""Tests for notebook teaching structure and offline execution."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import run


def test_walkthrough_notebook_alternates_markdown_and_code_blocks() -> None:
    """Notebook should teach in markdown/code pairs rather than code dumps."""
    notebook_path = Path("notebooks/intraday_variance_walkthrough.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_types = [cell["cell_type"] for cell in payload["cells"]]
    assert cell_types.count("markdown") >= 6
    assert cell_types.count("code") >= 6
    for left, right in zip(cell_types, cell_types[1:]):
        assert not (left == "code" and right == "code")


def test_walkthrough_notebook_executes_offline() -> None:
    """Notebook should execute on a fresh clone using only tracked cache."""
    completed = run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "notebooks/intraday_variance_walkthrough.ipynb",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
