"""The world-voxel resolution knob, and the cap that has to move with it.

``deploy_e2e.launch.py`` pinned the kernel's ``world_voxel_max_cells`` to the
literal ``614125`` with a comment saying it was ``85^3``, the worst case for
``_octomap_coverage_radius()`` at 25 mm cells. That is a *derived* quantity kept
by hand, and the failure mode when it drifts is the wrong direction: a kernel
reserving too few cells rejects every grid it is sent, which reads as an empty
world and is a fail-**open** on the world check.

These pin the derivation against the constant it replaced, and pin the override
that PLAN.md §5's resolution battery runs behind.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch"] = module
    spec.loader.exec_module(module)
    return module


def test_the_cap_reproduces_the_constant_it_replaced(launch_module: object) -> None:
    """25 mm must still give exactly 614 125, or this refactor changed the shipped graph."""
    assert launch_module._world_voxel_max_cells(0.025) == 614125
    assert launch_module._octomap_coverage_radius() == 1.05


def test_the_cap_grows_with_a_finer_grid(launch_module: object) -> None:
    """A 15 mm ball needs 141^3 cells; a kernel still reserving 85^3 would reject every grid.

    That rejection is a fail-*open* on the world check — the kernel would carry
    on with no world obstacles at all — which is why the cap is derived here
    rather than left for someone to remember.
    """
    assert launch_module._world_voxel_max_cells(0.015) == 141**3 == 2803221
    assert launch_module._world_voxel_max_cells(0.0125) == 169**3


def test_the_resolution_override_is_honoured_within_its_range(
    launch_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRAL_OCTOMAP_RESOLUTION_M", "0.015")
    assert launch_module._octomap_resolution("sim") == 0.015
    assert launch_module._octomap_resolution("real") == 0.015


def test_an_unusable_resolution_falls_back_to_the_shipped_default(
    launch_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anything the lattice cannot place must not become a silently empty grid.

    ``build_lattice`` refuses a non-positive resolution and
    ``kMaxCellsPerAxis`` refuses one too fine, and both refusals publish an
    *empty* grid — every obstacle dropped. Falling back to the shipped default
    is the safe reading of a typo.
    """
    for bad in ("", "   ", "not-a-number", "0", "-0.015", "1.5"):
        monkeypatch.setenv("OPENRAL_OCTOMAP_RESOLUTION_M", bad)
        assert launch_module._octomap_resolution("sim") == 0.025, bad
        assert launch_module._octomap_resolution("real") == 0.05, bad


def test_without_the_override_the_shipped_defaults_are_unchanged(
    launch_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENRAL_OCTOMAP_RESOLUTION_M", raising=False)
    assert launch_module._octomap_resolution("sim") == 0.025
    assert launch_module._octomap_resolution("real") == 0.05
