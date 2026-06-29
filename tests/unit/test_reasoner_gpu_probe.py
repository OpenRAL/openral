"""Unit tests for the reasoner's torch-free GPU-total VRAM probe (ADR-0077).

`_detect_gpu_total_vram_gb` shells out to ``nvidia-smi`` so the reasoner_node can
size the VLA+reward pair check without importing torch. It must parse MiB→GiB and
degrade to ``0.0`` (→ the caller skips the check) on any failure.

Run with:
    uv run pytest tests/unit/test_reasoner_gpu_probe.py -v
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

# reasoner_node imports rclpy at module scope.
pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs.msg")

from openral_reasoner_ros.reasoner_node import _detect_gpu_total_vram_gb


def _fake_run(stdout: str) -> Any:
    def run(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    return run


def test_parses_mib_to_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi reports MiB; 8188 MiB → ~7.99 GiB (the usable 8 GB card)."""
    monkeypatch.setattr(subprocess, "run", _fake_run("8188\n"))
    assert _detect_gpu_total_vram_gb() == pytest.approx(8188 / 1024.0, abs=1e-6)


def test_first_gpu_when_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple GPUs → GPU 0's total is used."""
    monkeypatch.setattr(subprocess, "run", _fake_run("24576\n49140\n"))
    assert _detect_gpu_total_vram_gb() == pytest.approx(24576 / 1024.0, abs=1e-6)


def test_missing_nvidia_smi_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No nvidia-smi → 0.0 (caller skips the check, never blocks dispatch)."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _detect_gpu_total_vram_gb() == 0.0


def test_unparseable_output_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage / empty output → 0.0, not a crash."""
    monkeypatch.setattr(subprocess, "run", _fake_run("no-gpu-here\n"))
    assert _detect_gpu_total_vram_gb() == 0.0
    monkeypatch.setattr(subprocess, "run", _fake_run(""))
    assert _detect_gpu_total_vram_gb() == 0.0
