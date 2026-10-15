"""``python -m openral_cli.main`` registers the full command set (#403).

The ``if __name__ == "__main__"`` guard used to sit mid-file, so running the
module directly invoked ``app()`` before most commands were registered.
"""

from __future__ import annotations

import os
import subprocess
import sys

import typer.main
from openral_cli.main import app


def test_module_entry_registers_every_command() -> None:
    expected = set(typer.main.get_command(app).commands)  # type: ignore[attr-defined] # reason: click.Group
    proc = subprocess.run(
        [sys.executable, "-m", "openral_cli.main", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "200"},
        check=True,
        timeout=120,
    )
    missing = {name for name in expected if f" {name} " not in proc.stdout}
    assert not missing, f"commands missing from `python -m openral_cli.main --help`: {missing}"
