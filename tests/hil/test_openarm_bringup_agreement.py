# SPDX-License-Identifier: Apache-2.0
"""`OpenArmRealHAL` and `openarm_bringup` must agree, checked from the configs.

The HAL names four controllers and the sixteen joints it hands them. Those
names live in **`openarm_bringup`'s** controller YAML, in a different repo on
a different release cadence. Nothing has ever compared the two: the HAL's
table (`openarm_real._command_groups`) is a hand-copied transcription, and a
rename upstream would leave it silently wrong.

Silently is the operative word. A `JointTrajectoryController` handed joints it
does not own **rejects the whole message**. There is no exception on the
publisher side, nothing on `/joint_states` looks unusual, and the arm simply
does not move — which is indistinguishable from a policy that chose to hold
still. That is the same failure shape ADR-0102 was written to end, one layer
further out.

The gripper is where this is most likely to bite, and the reason this file
exists: the bimanual bringup calls the gripper joint
``openarm_left_finger_joint1``, **not** ``openarm_left_gripper``. Every
plausible guess is wrong, so the only safe source is the config itself.

Needs no hardware at all — not even the CAN links — only a host where
``openarm_bringup`` is on the ament prefix path. It answers statically what
would otherwise need a powered cell: bringing the controllers up to look at
them calls ``OpenArmHW::on_activate`` → ``openarm_->enable_all()``, which
energises all sixteen motors. Reading the config energises nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CONTROLLER_CONFIG = Path("config") / "controllers" / "openarm_bimanual_controllers.yaml"


def _bringup_controller_config() -> dict | None:
    """The committed bimanual controller YAML, or None when bringup is absent."""
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        share = Path(get_package_share_directory("openarm_bringup"))
    except Exception:  # reason: PackageNotFoundError, plus whatever ament raises off-rig
        return None
    path = share / _CONTROLLER_CONFIG
    if not path.is_file():
        return None
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else None


_CONFIG = _bringup_controller_config()

requires_bringup = pytest.mark.skipif(
    _CONFIG is None,
    reason=(
        "openarm_bringup not on the ament prefix path — source the workspace "
        "carrying openarm_ros2 to compare the HAL against its controller config"
    ),
)


def _hal_groups() -> list[tuple[str, list[str]]]:
    """``(command topic, joint names)`` per controller, straight from the HAL."""
    from openral_hal.openarm_real import OpenArmRealHAL

    hal = OpenArmRealHAL(require_can_links=False)
    return [(topic, list(names)) for topic, _, names in hal._command_groups]


def _controller_joints(controller: str) -> list[str]:
    assert _CONFIG is not None
    return list(_CONFIG[controller]["ros__parameters"]["joints"])


@requires_bringup
def test_every_controller_the_hal_commands_is_declared_by_bringup() -> None:
    """A topic with no controller behind it publishes into the void."""
    assert _CONFIG is not None
    declared = set(_CONFIG["controller_manager"]["ros__parameters"]) - {"update_rate"}
    for topic, _ in _hal_groups():
        controller = topic.strip("/").split("/")[0]
        assert controller in declared, (
            f"the HAL publishes to {topic!r}, but openarm_bringup declares no "
            f"controller named {controller!r}. Declared: {sorted(declared)}"
        )


@requires_bringup
def test_each_controller_is_a_joint_trajectory_controller() -> None:
    """The HAL's topic suffix and message type both assume it.

    A ``ForwardCommandController`` takes ``Float64MultiArray`` on ``/commands``
    — publishing a ``JointTrajectory`` at it reaches nothing.
    """
    assert _CONFIG is not None
    declared = _CONFIG["controller_manager"]["ros__parameters"]
    for topic, _ in _hal_groups():
        controller = topic.strip("/").split("/")[0]
        assert declared[controller]["type"] == (
            "joint_trajectory_controller/JointTrajectoryController"
        ), f"{controller} is a {declared[controller]['type']}, so {topic} is the wrong interface"


@requires_bringup
def test_the_joint_names_the_hal_sends_are_the_ones_the_controller_owns() -> None:
    """The check that a wrong guess about the gripper would fail.

    `openarm_bringup` calls it ``openarm_left_finger_joint1``; ``_gripper``,
    ``_gripper_joint`` and ``_finger_joint`` are all wrong, and all of them
    fail by the controller quietly rejecting the message.
    """
    for topic, hal_names in _hal_groups():
        controller = topic.strip("/").split("/")[0]
        assert hal_names == _controller_joints(controller), (
            f"{controller}: the HAL sends {hal_names}, the controller owns "
            f"{_controller_joints(controller)}. A JointTrajectory naming joints "
            "the controller does not own is rejected whole — silently."
        )


@requires_bringup
def test_the_four_controllers_cover_all_sixteen_joints_exactly_once() -> None:
    """No joint commanded twice, none left out, and the HAL's order preserved."""
    from openral_hal.openarm_real import OpenArmRealHAL

    hal = OpenArmRealHAL(require_can_links=False)
    fanned: list[str] = []
    for topic, _ in _hal_groups():
        fanned.extend(_controller_joints(topic.strip("/").split("/")[0]))
    assert fanned == hal.ros2_control_joint_names(), (
        "concatenating the four controllers' joints must reproduce the HAL's "
        "16-DoF vector order exactly — that ordering is what the slice spans mean"
    )
    assert len(set(fanned)) == 16
