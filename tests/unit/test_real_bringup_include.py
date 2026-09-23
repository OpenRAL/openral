# SPDX-License-Identifier: Apache-2.0
"""`deploy run` must start its robot's vendor ros2_control graph itself.

Before `_build_real_bringup_include`, `deploy_e2e.launch.py` assumed the
`controller_manager` graph was already up, so a real bring-up meant launching it
by hand from a second terminal. That put a second `/joint_states` publisher on
the bus, which is exactly what `openral_cli._dds_scope` refuses to launch over
(#227) — so the documented path needed that guard disarmed by an env var to work
at all. With the bringup inside the graph the refusal became unwaivable.

The manifest's `hal.real_bringup` (`"<pkg>:<file>.launch.py"`) is the only way
a robot declares that bringup — every robot runs the one generic
`openral_hal_node` package, so there is no per-robot package to carry it by
convention.

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


def test_undeclared_bringup_yields_no_include() -> None:
    """A manifest with no `hal.real_bringup` is not an error — it needs none.

    A real robot whose controller graph is started by a vendor daemon or a
    robot-side controller legitimately has nothing to include; raising here
    would break `deploy run` for every one of them.
    """
    deploy_e2e = _load_deploy_e2e()
    assert deploy_e2e._build_real_bringup_include(None) is None  # type: ignore[attr-defined]  # reason: module loaded by path


def test_openarm_manifest_declares_its_bringup() -> None:
    """`robots/openarm/robot.yaml` names the bringup explicitly via `hal.real_bringup`.

    The declared file must exist in-repo, so the manifest field and the shipped
    launch file cannot drift apart. Holds without ROS sourced.
    """
    from openral_core import RobotDescription

    desc = RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / "openarm" / "robot.yaml"))
    assert desc.hal.real_bringup is not None
    pkg, _, launch_file = desc.hal.real_bringup.partition(":")
    assert (_REPO_ROOT / "packages" / pkg / "launch" / launch_file).is_file()


def test_declared_bringup_of_missing_package_raises() -> None:
    """An explicit `hal.real_bringup` that is not installed fails loud (§1.4)."""
    deploy_e2e = _load_deploy_e2e()
    with pytest.raises(RuntimeError, match="not installed"):
        deploy_e2e._build_real_bringup_include(  # type: ignore[attr-defined]  # reason: module loaded by path
            "no_such_hal_package_exists:real_bringup.launch.py"
        )


def test_declared_bringup_file_missing_raises() -> None:
    """An installed package that lacks the declared launch file fails loud too."""
    deploy_e2e = _load_deploy_e2e()
    pytest.importorskip("openral_msgs")
    with pytest.raises(RuntimeError, match="does not exist"):
        deploy_e2e._build_real_bringup_include(  # type: ignore[attr-defined]  # reason: module loaded by path
            "openral_msgs:real_bringup.launch.py"
        )
