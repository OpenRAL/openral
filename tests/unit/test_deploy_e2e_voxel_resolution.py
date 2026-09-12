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
        assert launch_module._octomap_resolution("sim") == 0.015, bad
        assert launch_module._octomap_resolution("real") == 0.02, bad


def test_the_shipped_defaults_are_15_mm_in_sim_and_20_mm_on_hardware(
    launch_module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#253's ruling, and the half of it that deliberately did NOT move.

    Sim ships 15 mm: replayed on the real kernel over identical payload poses,
    the DOP-only field-typical payload goes from 154 false stops of 182
    genuinely-clear poses (box @ 25 mm) to 15 (refined @ 15 mm), every real
    contact still caught. The terms ADD, so this only pays alongside #266 --
    15 mm alone was 10%.

    Real hardware ships 20 mm -- finer than the 50 mm it replaces, coarser than
    sim's 15 mm, and on sim evidence alone. No real map has been measured at
    any resolution: sim rasterises a kitchen's own solid geometry, a real map
    is depth-camera derived and dilated by the octree->grid bridge, and its
    error budget is larger, with depth noise and extrinsics sitting beside the
    quantisation term this shrinks.

    The two values stay DIFFERENT on purpose, which is the point of this test:
    a later edit that collapses them to one number would be asserting the two
    maps have the same error budget, and nothing has shown that.
    """
    monkeypatch.delenv("OPENRAL_OCTOMAP_RESOLUTION_M", raising=False)
    assert launch_module._octomap_resolution("sim") == 0.015
    assert launch_module._octomap_resolution("real") == 0.02
    assert launch_module._octomap_resolution("sim") != launch_module._octomap_resolution("real")
    # Both inside `kMaxCells` (4 000 000), with the real grid the roomier one.
    assert launch_module._world_voxel_max_cells(0.02) == 1191016
    # 15 mm is the finest value that ships without touching `kMaxCells`:
    # 141/axis is 2 803 221 against the 4 000 000 guard, where 12.5 mm needs
    # 4 826 809 and is refused outright.
    assert launch_module._world_voxel_max_cells(0.015) == 2803221
    assert launch_module._world_voxel_max_cells(0.015) < 4_000_000


def _load_tool(name: str) -> object:
    """``tools/`` is not an installed package — load by path, as the other tool tests do."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_transport_probe_sizes_the_grid_the_kernel_reserves(launch_module: object) -> None:
    """The probe's cell derivation is a hand-copy of the launch file's; pin them together.

    ``voxel_transport_probe`` re-derives the grid from its own ``RADIUS_M``
    literal rather than importing ``_octomap_coverage_radius``, because the
    launch file is not importable as a package. If the shipped radius ever
    moves, the probe would keep timing a message of the *old* size while the
    kernel reserved the new one — and the wire latency it reports is the term
    the whole 25 -> 15 mm trade is settled on. A wrong-sized message would not
    fail; it would quietly measure the wrong lever.
    """
    pytest.importorskip("openral_msgs", reason="the probe imports openral_msgs at module level")
    probe = _load_tool("voxel_transport_probe")

    assert launch_module._octomap_coverage_radius() == probe.RADIUS_M
    for resolution in (0.025, 0.015, 0.0125):
        assert probe.per_axis(resolution) ** 3 == launch_module._world_voxel_max_cells(resolution)


def test_the_quantisation_gain_matches_the_matrix_budget_it_is_derived_from() -> None:
    """8.66 mm is the difference of two half body-diagonals, not a typed-in constant.

    ``stop_ee_speed.QUANTISATION_GAIN_M`` is what every staleness figure in
    PLAN.md §5 is weighed against, and ``validation_matrix.quantization_budget_m``
    is the canonical form of the same derivation. They are written out
    separately, so they can drift apart silently.
    """
    stop_ee_speed = _load_tool("stop_ee_speed")
    validation_matrix = _load_tool("validation_matrix")

    expected = validation_matrix.quantization_budget_m(
        0.025
    ) - validation_matrix.quantization_budget_m(0.015)
    assert pytest.approx(expected) == stop_ee_speed.QUANTISATION_GAIN_M
    assert pytest.approx(8.66, abs=0.01) == stop_ee_speed.QUANTISATION_GAIN_M * 1e3
