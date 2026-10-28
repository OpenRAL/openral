"""The InternVLA-N1 launcher rebuilds a cached sidecar venv made for another Python.

Real ``uv venv`` interpreters; the 3.11 case skips when uv cannot provide one.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "internvla_n1_sidecar.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("internvla_n1_sidecar", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uv_venv(path: Path, python: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    done = subprocess.run(
        [uv, "venv", "--quiet", "--python", python, str(path)],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"uv cannot provide Python {python}: {done.stderr.decode()[-200:]}")


def test_reads_the_interpreter_of_a_current_venv(tmp_path: Path) -> None:
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    _uv_venv(tmp_path / "venv", current)
    assert _load()._venv_python(tmp_path / "venv") == current


def test_an_old_cached_venv_differs_from_the_pin(tmp_path: Path) -> None:
    module = _load()
    _uv_venv(tmp_path / "venv", "3.11")
    assert module._venv_python(tmp_path / "venv") == "3.11"
    assert module._venv_python(tmp_path / "venv") != module._PYTHON


def test_no_venv_reads_as_none(tmp_path: Path) -> None:
    assert _load()._venv_python(tmp_path / "missing") is None
