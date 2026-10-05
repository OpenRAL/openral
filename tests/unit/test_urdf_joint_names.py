# SPDX-License-Identifier: Apache-2.0
"""A sim HAL's joint states reach ``robot_state_publisher`` under the URDF's joint names.

``robot_state_publisher`` moves only the links whose URDF joint names it is sent. A sim
HAL publishes the manifest's logical names (``left_joint1``); the OpenArm's vendored
URDF names the same joint ``openarm_left_joint1`` (``JointSpec.sim_joint_name``), so
under ``deploy sim`` no moving OpenArm link was on ``/tf`` and every tf2 consumer of a
link (the vision attachment leg's TCP and attach link, the octomap bridge's payload
clearing) found nothing. ``openral_hal.resolver.urdf_joint_names`` is the map the
launch hands the HAL; these tests pin it on real manifests and URDFs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal.lifecycle import urdf_named_joint_state
from openral_hal.resolver import urdf_joint_names

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


def test_openarm_states_are_renamed_to_the_vendored_urdf() -> None:
    desc = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    names = urdf_joint_names(desc, (_ROBOTS / "openarm" / "openarm.urdf").read_text())
    assert names == [j.sim_joint_name for j in desc.joints]
    assert names[0] == "openarm_left_joint1"
    assert names[15] == "openarm_right_finger_joint1"


def test_a_urdf_that_already_agrees_needs_no_renamed_stream() -> None:
    desc = RobotDescription.from_yaml(str(_ROBOTS / "so101_follower" / "robot.yaml"))
    urdf = next((_ROBOTS / "so101_follower").glob("*.urdf"))
    assert urdf_joint_names(desc, urdf.read_text()) == []


def test_a_joint_the_urdf_lacks_is_left_out_of_the_renamed_state() -> None:
    desc = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    urdf = (_ROBOTS / "openarm" / "openarm.urdf").read_text()
    urdf = urdf.replace('name="openarm_right_finger_joint1"', 'name="renamed_upstream"')
    names = urdf_joint_names(desc, urdf)
    assert names[15] == ""
    state = SimpleNamespace(
        header=None,
        name=[j.name for j in desc.joints],
        position=[float(i) for i in range(16)],
        velocity=[],
        effort=[0.0] * 16,
    )
    out = urdf_named_joint_state(state, names)
    assert out.name == names[:15]
    assert list(out.position) == [float(i) for i in range(15)]
    assert list(out.velocity) == []
    assert list(out.effort) == [0.0] * 15
    assert state.name[0] == "left_joint1"  # the manifest-named message is untouched


def test_unparseable_urdf_is_a_config_error() -> None:
    desc = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    with pytest.raises(ROSConfigError, match="does not parse"):
        urdf_joint_names(desc, "<robot")
