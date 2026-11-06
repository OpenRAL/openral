"""Lock-step guards for the MakerMods Metal single-arm and station manifests.

``robots/makermods_metal_station/robot.yaml`` is two copies of
``robots/makermods_metal/robot.yaml`` with ``left_`` / ``right_`` prefixes,
mounted at ``y = ±STATION_ARM_SPACING_M / 2`` on ``station_base``. Nothing
generates the station manifest, so these tests catch drift between the two
YAML files, between the station mounts and the asset module's spacing
constant, between the station ``sim:`` maps and the derived MJCF, and between
the single-arm manifest and the URDF derived from the pinned upstream clone.

The URDF / MJCF checks need the pinned ``makermods-robotics/metal-python-ros``
clone; on a host that cannot fetch it they skip on ``AssetFetchError``
(CLAUDE.md §1.11), never fake the model.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from openral_core import RobotDescription
from openral_core.assets import AssetFetchError, AssetKind, resolve_asset
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


def _resolve(ref: str, kind: AssetKind) -> Path:
    try:
        path = resolve_asset(ref, kind)
    except AssetFetchError as exc:
        pytest.skip(f"pinned MakerMods Metal clone not fetchable on this host: {exc}")
    assert path is not None
    return path


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


def test_station_sim_maps_match_station_mjcf(station: RobotDescription) -> None:
    """Station qpos / actuator maps point at the MJCF joints / actuators they name."""
    mujoco = pytest.importorskip("mujoco")
    assert station.assets.mjcf is not None and station.sim is not None
    model = mujoco.MjModel.from_xml_path(str(_resolve(station.assets.mjcf, "mjcf")))
    assert station.sim.joint_qpos_addr is not None
    assert station.sim.actuator_index is not None
    assert model.nu == len(station.joints)
    for joint in station.joints:
        sim_name = joint.sim_joint_name or joint.name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
        assert jid >= 0, sim_name
        assert int(model.jnt_qposadr[jid]) == station.sim.joint_qpos_addr[joint.name], joint.name
        act = station.sim.actuator_index[joint.name]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act) == sim_name
        assert int(model.actuator_trnid[act, 0]) == jid
    for gripper in station.sim.grippers:
        gripper_sim = next(j.sim_joint_name for j in station.joints if j.name == gripper.joint)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gripper_sim)
        assert gripper.qpos_addrs == (int(model.jnt_qposadr[jid]),)
        assert gripper.actuator_index == station.sim.actuator_index[gripper.joint]
        assert tuple(float(v) for v in model.jnt_range[jid]) == pytest.approx(gripper.ctrl_range)


def test_single_arm_sim_maps_match_mjcf(arm: RobotDescription) -> None:
    """Single-arm qpos map and default actuator order match the derived MJCF."""
    mujoco = pytest.importorskip("mujoco")
    assert arm.assets.mjcf is not None and arm.sim is not None
    model = mujoco.MjModel.from_xml_path(str(_resolve(arm.assets.mjcf, "mjcf")))
    assert model.nu == len(arm.joints)
    assert arm.sim.joint_qpos_addr is not None
    for index, joint in enumerate(arm.joints):
        sim_name = joint.sim_joint_name or joint.name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
        assert int(model.jnt_qposadr[jid]) == arm.sim.joint_qpos_addr[joint.name]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) == sim_name


def _urdf_joints(path: Path) -> dict[str, ET.Element]:
    return {str(j.get("name")): j for j in ET.parse(path).getroot().iter("joint")}


def _floats(text: str | None) -> tuple[float, ...]:
    return tuple(float(v) for v in str(text).split())


def test_single_arm_matches_derived_urdf(arm: RobotDescription) -> None:
    """Revolute joint origins / axes / limits equal the URDF from the pinned clone."""
    assert arm.assets.urdf is not None and arm.assets.urdf.ref is not None
    urdf = _urdf_joints(_resolve(arm.assets.urdf.ref, "urdf"))
    for joint in arm.joints:
        if joint.role == "gripper":
            continue
        src = urdf[joint.sim_joint_name or joint.name]
        origin, axis, limit = src.find("origin"), src.find("axis"), src.find("limit")
        assert origin is not None and axis is not None and limit is not None
        assert src.get("type") == joint.joint_type.value, joint.name
        parent, child = src.find("parent"), src.find("child")
        assert parent is not None and child is not None
        assert (parent.get("link"), child.get("link")) == (joint.parent_link, joint.child_link)
        assert joint.origin_xyz == pytest.approx(_floats(origin.get("xyz")), abs=1e-6)
        assert joint.origin_rpy == pytest.approx(_floats(origin.get("rpy")), abs=1e-9)
        assert joint.axis_xyz == pytest.approx(_floats(axis.get("xyz")))
        assert joint.position_limits == pytest.approx(
            (float(str(limit.get("lower"))), float(str(limit.get("upper"))))
        )
        assert joint.effort_limit == pytest.approx(float(str(limit.get("effort"))))
        assert joint.velocity_limit == pytest.approx(float(str(limit.get("velocity"))))


def test_single_arm_gripper_matches_derived_urdf(arm: RobotDescription) -> None:
    """The normalised gripper maps onto the URDF driven-finger travel and effort."""
    assert arm.assets.urdf is not None and arm.assets.urdf.ref is not None
    assert arm.sim is not None
    urdf = _urdf_joints(_resolve(arm.assets.urdf.ref, "urdf"))
    (gripper,) = [j for j in arm.joints if j.role == "gripper"]
    assert gripper.sim_joint_name is not None
    src = urdf[gripper.sim_joint_name]
    limit, parent, child = src.find("limit"), src.find("parent"), src.find("child")
    assert limit is not None and parent is not None and child is not None
    assert (parent.get("link"), child.get("link")) == (gripper.parent_link, gripper.child_link)
    travel = (float(str(limit.get("lower"))), float(str(limit.get("upper"))))
    (sim_gripper,) = arm.sim.grippers
    assert sim_gripper.ctrl_range == pytest.approx(travel)
    assert sim_gripper.qpos_scale == pytest.approx(travel[1] - travel[0])
    assert gripper.effort_limit == pytest.approx(float(str(limit.get("effort"))))
    by_name = {a.name: a for a in arm.fixed_attachments}
    fixed = urdf["gripper_base"]
    origin = fixed.find("origin")
    assert origin is not None
    assert by_name["gripper_base"].origin_xyz == pytest.approx(_floats(origin.get("xyz")))
    assert by_name["gripper_base"].origin_rpy == pytest.approx(_floats(origin.get("rpy")))


def test_station_urdf_mounts_match_manifest(station: RobotDescription) -> None:
    """The derived station URDF mounts the arms where ``fixed_attachments`` says."""
    assert station.assets.urdf is not None and station.assets.urdf.ref is not None
    urdf = _urdf_joints(_resolve(station.assets.urdf.ref, "urdf"))
    for att in station.fixed_attachments:
        src = urdf[att.name]
        origin = src.find("origin")
        assert origin is not None, att.name
        assert att.origin_xyz == pytest.approx(_floats(origin.get("xyz")), abs=1e-6)
        assert att.origin_rpy == pytest.approx(_floats(origin.get("rpy")))
    for joint in station.joints:
        assert (joint.sim_joint_name or joint.name) in urdf, joint.name


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
