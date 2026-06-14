"""HIL test: running on a real NVIDIA Jetson.

Skipped automatically on non-Jetson hosts (detected by the absence of
``/etc/nv_tegra_release`` and ``/proc/device-tree/model``).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = Path("/proc/device-tree/model")
    if model.exists():
        return "tegra" in model.read_text(errors="ignore").lower()
    return False


@pytest.mark.skipif(not _is_jetson(), reason="Not running on a Jetson host")
def test_jetson_probe_returns_typed_info() -> None:  # pragma: no cover
    from openral_detect.probes import probe_gpus

    result = probe_gpus(warnings=[])
    assert result.jetson is not None
    assert "Jetson" in result.jetson.board or "Tegra" in result.jetson.board.upper()
    # TOPS lookup: known boards have non-zero TOPS; unknown boards
    # yield 0.0 — the probe must not raise either way.
    assert result.jetson.tops >= 0.0
    # The supported_dtypes list must include FP16 and INT8 on every
    # current Jetson SoC (Xavier / Orin).
    from openral_core.schemas import QuantizationDtype

    if "Orin" in result.jetson.board or "Xavier" in result.jetson.board:
        assert QuantizationDtype.FP16 in result.jetson.supported_dtypes
        assert QuantizationDtype.INT8 in result.jetson.supported_dtypes
