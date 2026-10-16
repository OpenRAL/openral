# SPDX-License-Identifier: Apache-2.0
"""robot_state_publisher gets the URDF's units: the TF jaw is the manifest's mimic-map jaw.

The HAL publishes a fraction-unit gripper (``[0, 1]``) on ``/joint_states``, the unit the
safety kernel, the grasp trigger and the runner read. ``robot_state_publisher`` read that
fraction as the URDF jaw joint's radians: the SO-101 jaw sat at ``f`` rad instead of the
``1.74533 f`` Isaac drives (21.5 deg off at the LeIsaac PickOrange deploy trial g19's
ATTACH). ``urdf_joint_map`` sends the URDF joint the manifest's own map; these tests pose
the URDF joint with it (the way robot_state_publisher does: origin, then the axis motion)
and compare with the manifest's mimic chain (the kernel's and the vision bridge's model).
Real manifests and URDFs (CLAUDE.md §1.11).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.schemas import MimicJoint
from openral_hal.lifecycle import URDF_JOINT_UNITS_UNSET, urdf_joint_units, urdf_named_joint_state
from openral_hal.resolver import UrdfJointMap, urdf_joint_map, urdf_joint_names
from openral_hal.vision_attachment_bridge import _edge_chain, _joint_motion

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


def _robot(name: str, urdf: str) -> tuple[RobotDescription, str]:
    return (
        RobotDescription.from_yaml(str(_ROBOTS / name / "robot.yaml")),
        (_ROBOTS / name / urdf).read_text(),
    )


def _rpy_quat(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (r, p, y) for f in (math.cos, math.sin))
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _urdf_joint_pose(urdf: str, joint: str, value: float) -> NDArray[np.float64]:
    """parent <- child of a revolute URDF joint at ``value`` rad, as robot_state_publisher."""
    node = next(j for j in ET.fromstring(urdf).iter("joint") if j.get("name") == joint)
    origin = node.find("origin")
    xyz = [
        float(v) for v in ((origin.get("xyz") if origin is not None else None) or "0 0 0").split()
    ]
    rpy = [
        float(v) for v in ((origin.get("rpy") if origin is not None else None) or "0 0 0").split()
    ]
    axis_node = node.find("axis")
    axis = np.asarray(
        [
            float(v)
            for v in ((axis_node.get("xyz") if axis_node is not None else None) or "1 0 0").split()
        ]
    )
    axis = axis / np.linalg.norm(axis)
    half = value / 2.0
    q = (*(axis * math.sin(half)), math.cos(half))
    return np.asarray(
        homogeneous_from_quat_xyz((xyz[0], xyz[1], xyz[2]), _rpy_quat(*rpy))
        @ homogeneous_from_quat_xyz((0.0, 0.0, 0.0), q),
        dtype=np.float64,
    )


def _manifest_link_pose(
    robot: RobotDescription, root: str, link: str, q: float
) -> NDArray[np.float64]:
    """root <- link down the manifest's own edge chain at gripper position ``q``."""
    chain = _edge_chain(robot, root, link)
    assert chain is not None
    pose = np.eye(4)
    for edge, origin in chain:
        value = edge.value_at(q) if isinstance(edge, MimicJoint) else q
        pose = pose @ origin @ _joint_motion(edge, value)
    return np.asarray(pose, dtype=np.float64)


def _sent(robot: RobotDescription, urdf: str, joint: str, q: float) -> tuple[str, float]:
    """What robot_state_publisher is sent for manifest joint ``joint`` at ``q``."""
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    names = [j.name for j in robot.joints]
    msg = SimpleNamespace(
        header=None,
        name=names,
        position=[q if n == joint else 0.0 for n in names],
        velocity=[],
        effort=[],
    )
    out = urdf_named_joint_state(msg, m.names, m.scales, m.offsets)
    i = [j.name for j in robot.joints].index(joint)
    return m.names[i], float(out.position[list(out.name).index(m.names[i])])


@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.378, 1.0])
def test_the_so101_tf_jaw_is_the_mimic_map_jaw(fraction: float) -> None:
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    urdf_joint, value = _sent(robot, urdf, "gripper", fraction)
    assert urdf_joint == "gripper"
    assert value == pytest.approx(1.74533 * fraction)
    np.testing.assert_allclose(
        _urdf_joint_pose(urdf, urdf_joint, value),
        _manifest_link_pose(robot, "gripper", "jaw", fraction),
        atol=1e-6,
    )


def test_the_old_stream_was_21_degrees_off_at_g19() -> None:
    """The bug this fixes: the fraction read as radians, at g19's ATTACH fraction.

    The jaw is off by 0.74533 f rad — 16.1 deg at f = 0.378. (The deploy-trial bag gave
    21.5 deg against a reconstruction from its own tf samples; this is the model value.)
    """
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    f = 0.37767
    wrong = _urdf_joint_pose(urdf, "gripper", f)[:3, :3]
    right = _manifest_link_pose(robot, "gripper", "jaw", f)[:3, :3]
    angle = math.degrees(math.acos(np.clip((np.trace(wrong.T @ right) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle == pytest.approx(math.degrees(0.74533 * f), abs=0.05)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_the_droid_robotiq_tf_fingers_are_the_mimic_map_fingers(fraction: float) -> None:
    robot, urdf = _robot("droid_franka_robotiq", "droid_franka_robotiq.urdf")
    urdf_joint, value = _sent(robot, urdf, "finger_joint", fraction)
    assert urdf_joint == "finger_joint"
    assert value == pytest.approx(math.pi / 4 * (1.0 - fraction), abs=1e-6)
    leader = next(j for j in robot.joints if j.name == "finger_joint")
    np.testing.assert_allclose(
        _urdf_joint_pose(urdf, "finger_joint", value),
        _manifest_link_pose(robot, leader.parent_link, "left_outer_knuckle", fraction),
        atol=1e-6,
    )
    # robot_state_publisher derives every URDF <mimic> follower from finger_joint by the
    # URDF's own multiplier and offset: they must agree with the manifest's rows.
    followers = {
        j.get("name"): j.find("mimic")
        for j in ET.fromstring(urdf).iter("joint")
        if j.find("mimic") is not None and j.find("mimic").get("joint") == "finger_joint"  # type: ignore[union-attr]
    }
    assert followers, "the Robotiq URDF declares its mimic followers"
    rows = {m.name: m for m in robot.mimic_joints}
    for name, mimic in followers.items():
        assert mimic is not None
        urdf_value = float(mimic.get("multiplier", "1")) * value + float(mimic.get("offset", "0"))
        assert urdf_value == pytest.approx(rows[str(name)].value_at(fraction), abs=1e-6), name


def test_robots_whose_names_and_units_agree_need_no_map() -> None:
    for name, urdf in (("ur5e", "ur5e.urdf"), ("h1", "h1.urdf")):
        robot, text = _robot(name, urdf)
        assert urdf_joint_map(robot, text) is None, name


def test_the_openarm_map_is_its_renaming_at_unit_scale() -> None:
    robot, urdf = _robot("openarm", "openarm.urdf")
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    assert list(m.names) == urdf_joint_names(robot, urdf)
    assert set(m.scales) == {1.0} and set(m.offsets) == {0.0}


def test_the_manifest_units_on_joint_states_are_untouched() -> None:
    """The renamed, re-unit copy is a copy: /joint_states keeps the fraction."""
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    names = [j.name for j in robot.joints]
    msg = SimpleNamespace(
        header=None,
        name=list(names),
        position=[0.0] * 5 + [0.4],
        velocity=[0.0] * 5 + [0.2],
        effort=[],
    )
    out = urdf_named_joint_state(msg, m.names, m.scales, m.offsets)
    assert msg.position[-1] == 0.4 and msg.velocity[-1] == 0.2
    assert out.position[-1] == pytest.approx(1.74533 * 0.4)
    assert out.velocity[-1] == pytest.approx(1.74533 * 0.2)


def test_the_so101_map_passes_the_parameter_parsing_unchanged() -> None:
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    n = len(robot.joints)
    assert urdf_joint_units(n, m.scales, m.offsets) == (list(m.scales), list(m.offsets))


def test_an_unset_array_is_identity_and_explicit_zeros_are_kept() -> None:
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    m = urdf_joint_map(robot, urdf)
    assert m is not None
    n = len(robot.joints)
    unset = list(URDF_JOINT_UNITS_UNSET)
    assert urdf_joint_units(n, unset, unset) == ([1.0] * n, [0.0] * n)
    assert urdf_joint_units(n, m.scales, unset) == (list(m.scales), [0.0] * n)
    assert urdf_joint_units(n, unset, m.offsets) == ([1.0] * n, list(m.offsets))
    # A real single-joint [0.0] is a value, not the old "unset" sentinel.
    assert urdf_joint_units(1, [0.0], [0.0]) == ([0.0], [0.0])


def test_a_scale_array_of_the_wrong_length_is_a_config_error() -> None:
    robot, _ = _robot("so101_follower", "so101_follower.urdf")
    with pytest.raises(ROSConfigError, match="urdf_joint_scales"):
        urdf_joint_units(len(robot.joints), [1.0], list(URDF_JOINT_UNITS_UNSET))


@pytest.mark.parametrize("hal_mode", ["sim", "real"])
def test_the_deploy_launch_maps_the_so101_jaw_for_robot_state_publisher(hal_mode: str) -> None:
    """The real SO-101 (serial HAL, no ros2_control broadcaster) publishes the manifest's
    units on /joint_states as the twin does, so /tf needs the same map in both modes."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    import importlib.util
    import sys

    from openral_hal.resolver import ros_control_hal

    launch = _ROBOTS.parent / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_rsp_units", launch)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_rsp_units"] = module
    spec.loader.exec_module(module)
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    assert not ros_control_hal(robot, mode="real")
    so101_map = module._rsp_urdf_joint_map(
        robot, str(_ROBOTS / "so101_follower/robot.yaml"), hal_mode
    )
    assert so101_map is not None and so101_map == urdf_joint_map(robot, urdf)
    # A ros2_control arm (the OpenArm): the vendor broadcaster's URDF names, on real only.
    openarm = RobotDescription.from_yaml(str(_ROBOTS / "openarm/robot.yaml"))
    openarm_map = module._rsp_urdf_joint_map(openarm, str(_ROBOTS / "openarm/robot.yaml"), hal_mode)
    assert ros_control_hal(openarm, mode="real")
    assert (openarm_map is None) == (hal_mode == "real")


_FINITE = st.floats(allow_nan=False, allow_infinity=False)


@given(st.lists(st.tuples(st.text(), _FINITE, _FINITE), max_size=8))
def test_the_urdf_joint_map_round_trips_and_its_json_schema_validates(
    rows: list[tuple[str, float, float]],
) -> None:
    m = UrdfJointMap(
        names=tuple(r[0] for r in rows),
        scales=tuple(r[1] for r in rows),
        offsets=tuple(r[2] for r in rows),
    )
    assert UrdfJointMap.model_validate_json(m.model_dump_json()) == m
    assert set(UrdfJointMap.model_json_schema()["required"]) == {"names", "scales", "offsets"}


def test_the_so101_urdf_joint_map_is_frozen_and_parallel() -> None:
    robot, urdf = _robot("so101_follower", "so101_follower.urdf")
    m = urdf_joint_map(robot, urdf)
    assert m is not None and len(m.names) == len(robot.joints)
    with pytest.raises(ValueError, match="frozen"):
        m.names = ()  # type: ignore[misc]  # reason: asserting the model refuses mutation
    with pytest.raises(ValueError, match="parallel"):
        UrdfJointMap(names=m.names, scales=m.scales[:-1], offsets=m.offsets)
