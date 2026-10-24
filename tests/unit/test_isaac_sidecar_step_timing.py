"""The sidecar's opt-in per-step timing log (``OPENRAL_SIDECAR_TIMING=1``).

``tools/isaac_sidecar.py`` imports without Isaac (module scope needs only numpy), like
``test_isaac_sidecar_version_probe``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "isaac_sidecar.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("isaac_sidecar_timing_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reply_sizes_are_listed_by_array_path() -> None:
    reply = {
        "observation": {
            "images": {"top": np.zeros((10, 10, 3), np.uint8)},
            "depth_frames": {"cam": {"depth": np.zeros((100, 100), np.float32), "k": [1.0]}},
        }
    }
    sizes = _load_tool()._nbytes_tree(reply)
    assert sizes == {
        "/observation/images/top": pytest.approx(300 / 1e6),
        "/observation/depth_frames/cam/depth": pytest.approx(40000 / 1e6),
    }


def test_the_mean_is_printed_every_hundred_steps_of_a_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timing = _load_tool()._StepTiming()
    for _ in range(99):
        timing.add("fresh", {"physics": 10.0, "pack": 2.0})
    assert capsys.readouterr().out == ""
    timing.add("fresh", {"physics": 10.0, "pack": 2.0})
    out = capsys.readouterr().out
    assert "frames=fresh n=100" in out
    assert "physics=10.0" in out
    assert "pack=2.0" in out
