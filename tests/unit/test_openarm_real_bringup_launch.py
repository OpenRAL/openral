# SPDX-License-Identifier: Apache-2.0
"""`openarm_real_bringup.launch.py` must agree with `OpenArmRealHAL`'s constants.

The launch file's CAN-interface and controller-name defaults are a hand-copied mirror of
`openral_hal.openarm_real`'s private constants (see that module's docstring: the launch file
is what starts the `ros2_control` graph those constants assume is already running). Nothing
else compares the two, so a rename on either side would silently desync — the launch file
would bring up controllers the HAL never talks to, or vice versa, and the arm would simply not
move. Same failure shape `tests/hil/test_openarm_bringup_agreement.py` guards one hop further
out (against `openarm_bringup`'s own controller config); this file guards the hop between the
HAL and the launch file that stands the controllers up.

Pure source-level check — no ROS install, no hardware, no `openarm_bringup` on the ament
prefix path required.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_LAUNCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "openral_hal_openarm"
    / "launch"
    / "openarm_real_bringup.launch.py"
)

try:
    import launch  # noqa: F401
    import launch_ros  # noqa: F401
except ImportError:
    _LAUNCH_AVAILABLE = False
else:
    _LAUNCH_AVAILABLE = True

requires_launch = pytest.mark.skipif(
    not _LAUNCH_AVAILABLE,
    reason="`launch`/`launch_ros` not importable — source a ROS 2 environment",
)


def _launch_constants() -> dict[str, str]:
    """Read the launch file's module-level string constants without importing it.

    Importing would execute `from launch import LaunchDescription`, which needs a sourced
    ROS 2 install — and on a host without one, the package's own `launch/` directory is
    itself picked up as a namespace package named `launch`, so the import fails confusingly
    rather than skipping. Parsing keeps this drift guard running everywhere, including CI,
    which is the point of having it: a rename on either side must fail the build, not skip.
    """
    tree = ast.parse(_LAUNCH_PATH.read_text())
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value.value, str)
    }


def _load_launch_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("openarm_real_bringup_launch", _LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_can_interface_defaults_match_the_hal() -> None:
    """A drifted default here silently points the bringup at the wrong CAN bus."""
    from openral_hal import openarm_real

    constants = _launch_constants()
    assert constants["_LEFT_CAN_INTERFACE"] == openarm_real._LEFT_CAN_INTERFACE
    assert constants["_RIGHT_CAN_INTERFACE"] == openarm_real._RIGHT_CAN_INTERFACE


def test_robot_controller_choice_yields_the_hal_s_controller_names() -> None:
    """Upstream's `robot_controller` arg picks a controller-name *pair*, not a literal name.

    `openarm.bimanual.launch.py` maps `robot_controller="joint_trajectory_controller"` to
    spawning `left_joint_trajectory_controller` / `right_joint_trajectory_controller` — the
    other legal value (`forward_position_controller`) takes `Float64MultiArray`, not the
    `JointTrajectory` messages `OpenArmRealHAL` publishes.
    """
    from openral_hal import openarm_real

    controller = _launch_constants()["_ROBOT_CONTROLLER"]
    assert controller == "joint_trajectory_controller"
    assert f"left_{controller}" == openarm_real._LEFT_ARM_CONTROLLER
    assert f"right_{controller}" == openarm_real._RIGHT_ARM_CONTROLLER


@requires_launch
def test_generates_a_launch_description_declaring_both_can_args() -> None:
    module = _load_launch_module()
    description = module.generate_launch_description()
    entities = description.entities
    arg_names = {e.name for e in entities if e.__class__.__name__ == "DeclareLaunchArgument"}
    assert arg_names == {"left_can_interface", "right_can_interface"}
