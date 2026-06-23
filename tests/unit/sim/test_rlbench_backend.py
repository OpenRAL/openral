"""Unit coverage for RLBench backend launch helpers."""

from __future__ import annotations

import pytest
from openral_sim.backends import rlbench


def _fake_xvfb(name: str) -> str:
    assert name == "xvfb-run"
    return "/usr/bin/xvfb-run"


def test_sidecar_display_prefix_uses_xvfb_when_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(rlbench.shutil, "which", _fake_xvfb)

    assert rlbench._sidecar_display_prefix() == [
        "/usr/bin/xvfb-run",
        "-a",
        "--server-args=-screen 0 1280x1024x24",
    ]


def test_sidecar_display_prefix_preserves_existing_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(rlbench.shutil, "which", _fake_xvfb)

    assert rlbench._sidecar_display_prefix() == []
