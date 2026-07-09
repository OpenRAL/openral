"""Unit tests for the shared sidecar interpreter environment.

:func:`openral_sim._sidecar_common.make_isolated_env` builds the env handed to
the gr00t / rldx model server before ``exec``. It must default the CUDA
caching-allocator to ``expandable_segments:True`` so the 3B NF4 checkpoint load
co-exists with the main process's sim render context on an 8 GB GPU — without
it the load is OOM-killed mid-shard (``rldx_sidecar_died_during_boot
returncode=-9`` in ``openral benchmark run``, whose suite path doesn't export
the var the ``benchmark scene`` smoke wrapper did). Pure-function; no GPU/zmq.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_sim._sidecar_common import make_isolated_env


def test_defaults_expandable_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = make_isolated_env(Path("/tmp/sidecar/.venv"))
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_caller_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit caller value is preserved (setdefault semantics)."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    env = make_isolated_env(Path("/tmp/sidecar/.venv"))
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_drops_workspace_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 3.12 workspace PYTHONPATH/PYTHONHOME must not leak into the 3.10 venv."""
    monkeypatch.setenv("PYTHONPATH", "/home/x/workspace/openral/python/sim/src")
    monkeypatch.setenv("PYTHONHOME", "/usr")
    venv = Path("/tmp/sidecar/.venv")
    env = make_isolated_env(venv)
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].startswith(str(venv / "bin"))
