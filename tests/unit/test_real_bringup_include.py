# SPDX-License-Identifier: Apache-2.0
"""`deploy run` must start its robot's vendor ros2_control graph itself.

Before `_build_real_bringup_include`, `deploy_e2e.launch.py` assumed the
`controller_manager` graph was already up, so a real bring-up meant launching it
by hand from a second terminal. That put a second `/joint_states` publisher on
the bus, which is exactly what `openral_cli._dds_scope` refuses to launch over
(#227) — so the documented path needed that guard disarmed by an env var to work
at all. With the bringup inside the graph the refusal became unwaivable.

The wiring is a convention, not a manifest field: a HAL package that ships
`launch/real_bringup.launch.py` declares by that fact alone which controller
graph its real HAL publishes to. `hal_package` is already threaded into the
launch file, so nothing new has to be declared anywhere.

These tests cover the resolution rule. That the OpenArm bringup's own arguments
agree with `OpenArmRealHAL`'s constants is
`tests/unit/test_openarm_real_bringup_launch.py`; that the controllers and
joints agree on real hardware is `tests/hil/test_openarm_bringup_agreement.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LAUNCH_PATH = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"


def _load_deploy_e2e() -> object:
    """Import deploy_e2e.launch.py by path.

    Guard on ``launch.actions`` rather than bare ``launch``: several packages ship
    an unpackaged ROS ``launch/`` directory that resolves as an implicit namespace
    package with no real ROS on the path, silently satisfying a bare
    ``importorskip("launch")`` (CLAUDE.md §1.11).
    """
    pytest.importorskip("launch.actions")
    pytest.importorskip("ament_index_python.packages")
    # deploy_e2e.launch.py imports the bringup packages at module top; they exist
    # only with the OpenRAL overlay sourced, as under ament_cmake_pytest.
    pytest.importorskip("openral_foxglove_bringup")
    pytest.importorskip("launch_ros.actions")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch", _LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_openarm_hal_package_ships_the_conventional_bringup() -> None:
    """The real OpenArm graph is reachable under the conventional name.

    This is the half of the contract that lives in the repo rather than in an
    installed overlay, so it holds without ROS sourced: rename the file and the
    include silently stops resolving, which would put us back to a `deploy run`
    that needs a second terminal.
    """
    conventional = (
        _REPO_ROOT / "packages" / "openral_hal_openarm" / "launch" / "real_bringup.launch.py"
    )
    assert conventional.is_file(), (
        f"{conventional} is the name deploy_e2e.launch.py looks for; "
        "renaming it disables the automatic real bringup"
    )


def test_bringup_name_is_the_one_the_launch_looks_for() -> None:
    """`REAL_BRINGUP_LAUNCH` and the shipped file name are one fact, not two.

    The test above hardcodes the name so it holds without ROS sourced; this one
    resolves it through the constant the launch file actually reads, so the two
    cannot drift apart silently.
    """
    deploy_e2e = _load_deploy_e2e()
    shipped = (
        _REPO_ROOT / "packages" / "openral_hal_openarm" / "launch" / deploy_e2e.REAL_BRINGUP_LAUNCH  # type: ignore[attr-defined]  # reason: module loaded by path
    )
    assert shipped.is_file()


def test_missing_package_yields_no_include() -> None:
    """An unknown HAL package is not an error — it simply ships no bringup.

    Most HAL packages are pure Python with no share directory at all
    (`PackageNotFoundError`), and a real robot whose controller graph is started
    by a vendor daemon legitimately has nothing to include. Raising here would
    break `deploy run` for every one of them.
    """
    deploy_e2e = _load_deploy_e2e()
    assert deploy_e2e._build_real_bringup_include("no_such_hal_package_exists") is None  # type: ignore[attr-defined]  # reason: as above


def test_package_without_a_bringup_yields_no_include() -> None:
    """A package that resolves but ships no `real_bringup.launch.py` returns None."""
    deploy_e2e = _load_deploy_e2e()
    # `openral_msgs` is an installed package with a share directory and no
    # bringup launch — the "resolves, but nothing to include" branch.
    pytest.importorskip("openral_msgs")
    assert deploy_e2e._build_real_bringup_include("openral_msgs") is None  # type: ignore[attr-defined]  # reason: as above
