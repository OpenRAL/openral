"""Lock-step guards for the MakerMods Metal single-arm and station manifests.

``robots/makermods_metal_station/robot.yaml`` is two copies of
``robots/makermods_metal/robot.yaml`` with ``left_`` / ``right_`` prefixes,
mounted at ``y = ±STATION_ARM_SPACING_M / 2`` on ``station_base``. Nothing
generates the station manifest, so these tests catch drift between the two
YAML files, between the station mounts and the asset module's spacing
constant. Pure YAML: no network. The checks against the URDF / MJCF derived
from the pinned upstream clone (a ``git clone`` on a cold cache, too slow for
the unit tier) live in ``tests/sim/test_makermods_metal_hal_mujoco.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from openral_core import RobotDescription
from openral_hal._makermods_metal_assets import STATION_ARM_SPACING_M

_ROOT = Path(__file__).resolve().parents[2]
_PREFIXES: tuple[str, ...] = ("left_", "right_")
# Fields that carry an arm-local name and therefore differ by the prefix.
_NAMED_JOINT_FIELDS: tuple[str, ...] = ("name", "sim_joint_name", "parent_link", "child_link")


@pytest.fixture(scope="module")
def arm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROOT / "robots" / "makermods_metal" / "robot.yaml"))


@pytest.fixture(scope="module")
def station() -> RobotDescription:
    return RobotDescription.from_yaml(
        str(_ROOT / "robots" / "makermods_metal_station" / "robot.yaml")
    )


def _prefixed(dump: dict[str, Any], prefix: str, fields: tuple[str, ...]) -> dict[str, Any]:
    out = dict(dump)
    for field in fields:
        if out.get(field) is not None:
            out[field] = prefix + str(out[field])
    return out


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_station_joints_equal_single_arm_joints(
    arm: RobotDescription, station: RobotDescription, prefix: str
) -> None:
    """Every station joint is the single-arm joint with the arm prefix applied."""
    station_joints = {j.name: j for j in station.joints if j.name.startswith(prefix)}
    assert [j.name for j in station.joints if j.name.startswith(prefix)] == [
        prefix + j.name for j in arm.joints
    ]
    for joint in arm.joints:
        expected = _prefixed(joint.model_dump(), prefix, _NAMED_JOINT_FIELDS)
        assert station_joints[prefix + joint.name].model_dump() == expected, joint.name


def test_station_joint_order_is_left_then_right(
    arm: RobotDescription, station: RobotDescription
) -> None:
    """Station joint order = left arm, then right arm (the MJCF actuator order)."""
    assert [j.name for j in station.joints] == [p + j.name for p in _PREFIXES for j in arm.joints]


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_station_gripper_attachment_equals_single_arm(
    arm: RobotDescription, station: RobotDescription, prefix: str
) -> None:
    """Each arm's ``gripper_base`` fixed attachment matches the single arm."""
    by_name = {a.name: a for a in station.fixed_attachments}
    for att in arm.fixed_attachments:
        expected = _prefixed(att.model_dump(), prefix, ("name", "parent_link", "child_link"))
        assert by_name[prefix + att.name].model_dump() == expected


def test_station_mounts_match_spacing_constant(station: RobotDescription) -> None:
    """``left_mount`` / ``right_mount`` sit at ``y = ±STATION_ARM_SPACING_M / 2``."""
    by_name = {a.name: a for a in station.fixed_attachments}
    half = STATION_ARM_SPACING_M / 2.0
    for prefix, y in (("left_", half), ("right_", -half)):
        mount = by_name[f"{prefix}mount"]
        assert mount.parent_link == station.base_frame == "station_base"
        assert mount.child_link == f"{prefix}base_link"
        assert mount.origin_xyz == pytest.approx((0.0, y, 0.0))
        assert mount.origin_rpy == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_station_end_effector_equals_single_arm(
    arm: RobotDescription, station: RobotDescription, prefix: str
) -> None:
    """Each station gripper end effector is the single-arm one (bar name / hand)."""
    (single,) = arm.end_effectors
    by_name = {e.name: e for e in station.end_effectors}
    got = by_name[prefix + single.name].model_dump()
    want = single.model_dump()
    assert got.pop("hand") == prefix.rstrip("_")
    want.pop("hand")
    assert got == {**want, "name": prefix + single.name}


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_station_sim_gripper_equals_single_arm(
    arm: RobotDescription, station: RobotDescription, prefix: str
) -> None:
    """Station gripper sim wiring = single-arm wiring shifted by the arm's offsets."""
    assert arm.sim is not None and station.sim is not None
    (single,) = arm.sim.grippers
    by_joint = {g.joint: g for g in station.sim.grippers}
    got = by_joint[prefix + single.joint]
    arm_index = _PREFIXES.index(prefix)
    qpos_per_arm = 8  # joint1..6, mimic finger joint7, driven finger joint8
    act_per_arm = len(arm.joints)
    assert got.ctrl_range == single.ctrl_range
    assert got.qpos_scale == single.qpos_scale
    assert got.read_mode == single.read_mode
    assert got.write_mode == single.write_mode
    assert got.qpos_addrs == tuple(a + arm_index * qpos_per_arm for a in single.qpos_addrs)
    assert got.actuator_index == arm_index * act_per_arm + act_per_arm - 1
    assert station.sim.seed_ctrl_from_qpos == arm.sim.seed_ctrl_from_qpos
    assert arm.sim.joint_qpos_addr is not None and station.sim.joint_qpos_addr is not None
    for name, addr in arm.sim.joint_qpos_addr.items():
        assert station.sim.joint_qpos_addr[prefix + name] == addr + arm_index * qpos_per_arm


@pytest.mark.parametrize("prefix", _PREFIXES)
def test_station_wrist_camera_equals_single_arm(
    arm: RobotDescription, station: RobotDescription, prefix: str
) -> None:
    """Each station wrist camera is the single-arm ``wrist`` camera on that arm."""
    (single,) = [s for s in arm.sensors if s.name == "wrist"]
    side = prefix.rstrip("_")
    (got,) = [s for s in station.sensors if s.name == f"wrist_{side}"]
    assert got.parent_frame == prefix + str(single.parent_frame)
    assert got.frame_id == f"{prefix}{single.frame_id}"
    assert got.static_transform_calibrated is single.static_transform_calibrated is False
    assert got.intrinsics == single.intrinsics
    assert single.sim_placement is not None and got.sim_placement is not None
    want = single.sim_placement.model_dump()
    want["parent_body"] = prefix + str(want["parent_body"])
    assert got.sim_placement.model_dump() == want
