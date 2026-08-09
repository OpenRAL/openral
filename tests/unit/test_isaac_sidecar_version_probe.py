"""Unit tests for the Isaac sidecar's pre-Kit dependency-floor probe.

Issue #89: the sidecar venv's ``nvidia-nvjitlink-cu12`` (12.6.85, pulled in by
torch 2.7) is too old for the CUDA-12.8 ``libcusparse`` that Isaac's own
``omni.isaac.ml_archive`` extension prebundles. The mismatch does not raise —
extension startup fails, Kit stays alive, and the client burns its whole
``boot_timeout_s`` before reporting the wrong cause. ``_check_required_versions``
runs before ``SimulationApp`` so the failure is a one-second, actionable error.

The floors it is handed come from ``_ISAAC_CUDA_FLOORS`` on the openral side —
asserted here so the two never drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "isaac_sidecar.py"
_PYTEST_VERSION = pytest.__version__


def _load_tool() -> Any:
    """Import ``tools/isaac_sidecar.py`` as a standalone module.

    Safe outside an Isaac venv: every ``isaacsim`` / ``omni.*`` import lives
    inside ``main`` (the SimulationApp-before-omni order), so module scope needs
    only numpy.
    """
    spec = importlib.util.spec_from_file_location("isaac_sidecar_under_test", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_satisfied_floor_passes() -> None:
    # pytest is installed in the test env, so its own version is its own floor.
    _load_tool()._check_required_versions([f"pytest>={_PYTEST_VERSION}"])


def test_too_old_dist_fails_with_both_versions() -> None:
    with pytest.raises(SystemExit) as excinfo:
        # 999 is unreachable, so the installed pytest is always "too old".
        _load_tool()._check_required_versions(["pytest>=999.0"])
    msg = str(excinfo.value)
    assert f"pytest: {_PYTEST_VERSION} installed, need >=999.0" in msg
    # Carries the repair, not just the diagnosis.
    assert "pip install --upgrade --no-deps" in msg
    assert "OPENRAL_ISAAC_AUTO_PROVISION=1" in msg


def test_missing_dist_is_reported_as_missing() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_tool()._check_required_versions(["nvidia-nvjitlink-cu12-not-real>=12.8"])
    assert "not installed (need >=12.8)" in str(excinfo.value)


def test_all_failures_are_reported_together() -> None:
    """One boot, one error — not a fix-rerun-fix loop over each floor in turn."""
    with pytest.raises(SystemExit) as excinfo:
        _load_tool()._check_required_versions(["pytest>=999.0", "no-such-dist-here>=1.0"])
    msg = str(excinfo.value)
    assert "pytest" in msg
    assert "no-such-dist-here" in msg


def test_release_segments_compare_numerically_not_lexically() -> None:
    """A floor one minor step away must be honored in both directions.

    Segment-wise integer comparison, not string comparison — otherwise a
    two-digit minor (``12.10`` vs ``12.8``) sorts backwards. Asserted through
    the probe, with pytest's real installed version as the movable floor.
    """
    tool = _load_tool()
    major, minor = (int(part) for part in _PYTEST_VERSION.split(".")[:2])
    # A floor one minor above the installed version must fail even when the
    # extra digit makes it lexically smaller (e.g. installed 8.9 vs floor 8.10).
    with pytest.raises(SystemExit):
        tool._check_required_versions([f"pytest>={major}.{minor + 1}"])
    # …and one minor below must pass.
    if minor > 0:
        tool._check_required_versions([f"pytest>={major}.{minor - 1}"])


def test_malformed_requirement_is_rejected() -> None:
    with pytest.raises(SystemExit, match="DIST>=VERSION"):
        _load_tool()._check_required_versions(["nvidia-nvjitlink-cu12==12.8"])


def test_no_floors_is_a_noop() -> None:
    _load_tool()._check_required_versions([])


def test_floors_match_the_openral_side_pins() -> None:
    """The probe's floors and the pip pins are one source of truth."""
    pytest.importorskip("openral_sim", reason="needs the openral workspace")
    from openral_sim.backends.isaac_sim import _ISAAC_CUDA_DEPS, _ISAAC_CUDA_FLOORS

    assert _ISAAC_CUDA_FLOORS["nvidia-nvjitlink-cu12"] == "12.8"
    for dist, floor in _ISAAC_CUDA_FLOORS.items():
        assert f"{dist}>={floor},<13" in _ISAAC_CUDA_DEPS
