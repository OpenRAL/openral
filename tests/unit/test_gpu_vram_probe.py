"""Unit test for the shared ``openral_core.detect_gpu_vram_gb`` probe.

Lifted out of ``openral_cli.deploy_sim`` and
``openral_reasoner_ros.reasoner_node`` (CLAUDE.md §1.11: no mocks — this runs
the real ``nvidia-smi`` subprocess against whatever GPU is actually present).
"""

from __future__ import annotations

import shutil

import pytest
from openral_core import detect_gpu_vram_gb


def test_detect_gpu_vram_gb_reads_real_gpu_total() -> None:
    if not shutil.which("nvidia-smi"):
        pytest.skip("no nvidia-smi on this host")
    total = detect_gpu_vram_gb("memory.total")
    assert isinstance(total, float)
    assert total > 0.0


def test_detect_gpu_vram_gb_reads_real_gpu_free() -> None:
    if not shutil.which("nvidia-smi"):
        pytest.skip("no nvidia-smi on this host")
    free = detect_gpu_vram_gb("memory.free")
    assert isinstance(free, float)
    assert free > 0.0


def test_detect_gpu_vram_gb_bad_field_returns_zero_not_raise() -> None:
    if not shutil.which("nvidia-smi"):
        pytest.skip("no nvidia-smi on this host")
    # An unknown --query-gpu column makes nvidia-smi exit non-zero; the probe
    # must degrade to 0.0 rather than raise (callers skip their check).
    assert detect_gpu_vram_gb("not.a.real.field") == 0.0
