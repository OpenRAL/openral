# SPDX-License-Identifier: Apache-2.0
"""OpenAMRobot 2.0's head ZED: OpenArm's intrinsics, at the shelf cell's head pose.

The intrinsics match OpenArm's, so an OpenArm rSkill gets its raster and field of view.
The mount is the fixed OpenArm shelf cell's head-camera pose relative to the arm
bases (the robot owner confirmed the real head ZED looks down at the tote and shelf like
the cell's), on ``openarm_left_base_link``, 2 mm forward so the lens clears the Getac
placeholder. One pose, three readers, which must agree: TF (robot_state_publisher's URDF
chain + the deploy launch's static ``parent_frame -> frame_id`` from
``static_transform_xyz_rpy``), Isaac (``default_camera_mounts``: the same transform,
``axes: world``) and the MuJoCo twin (the camera rig's ``sim_placement``: ``-z`` toward
``target``, ``+y`` toward ``up``).
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import BoxShape, RobotDescription, SensorSpec
from openral_core.geometry import homogeneous_from_quat_xyz, look_at_quat_wxyz
from openral_core.kinematics import link_transform, transform_from_xyz_rpy

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_AMR = _ROBOTS / "openamrobot_v2"
# MuJoCo camera axes (x right, y up, looks down -z) -> body axes (x forward, y left, z up).
_MJ_CAMERA_TO_BODY = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
# What the head ZED's depth may see of the robot from its lens: the base's front deck,
# the Getac face just behind and below the lens (the bottom rows look 5 deg past
# straight down) and the head light under it. The self-filter masks these.
_HEAD_ZED_SEES = {"base_link", "getac_tablet_link", "head_light_link"}


@functools.cache
def _desc(robot: str = "openamrobot_v2") -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))


def _sensor(robot: str, name: str) -> SensorSpec:
    return next(s for s in _desc(robot).sensors if s.name == name)


def _rot_wxyz(q: tuple[float, ...]) -> NDArray[np.float64]:
    w, x, y, z = q
    return homogeneous_from_quat_xyz((0.0, 0.0, 0.0), (x, y, z, w))[:3, :3]


def _pose(link: str) -> NDArray[np.float64]:
    """``base_footprint <- link`` at the zero configuration (the manifest's chain)."""
    desc = _desc()
    return link_transform(desc, "base_footprint", link, {j.name: 0.0 for j in desc.joints})


@functools.cache
def _urdf_link_pose(link: str) -> NDArray[np.float64]:
    """``base_footprint <- link`` through the URDF: the chain robot_state_publisher sends."""
    root = ET.parse(_AMR / "openamrobot_v2.urdf").getroot()
    parent_of = {
        j.find("child").get("link"): j  # type: ignore[union-attr] # reason: URDF joints have one child
        for j in root.iter("joint")
    }
    pose = np.eye(4)
    while link != "base_footprint":
        joint = parent_of[link]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]  # type: ignore[union-attr] # reason: every joint here has an origin
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]  # type: ignore[union-attr] # reason: as above
        pose = transform_from_xyz_rpy(xyz, rpy) @ pose
        link = joint.find("parent").get("link")  # type: ignore[union-attr,assignment] # reason: URDF joints have one parent
    return pose


def _angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


@functools.cache
def _tf_pose() -> NDArray[np.float64]:
    """``base_footprint <- zed_camera_link`` through TF (body frame: x forward, z up)."""
    zed = _sensor("openamrobot_v2", "head_zed")
    assert zed.parent_frame is not None and zed.static_transform_xyz_rpy is not None
    x, y, z, roll, pitch, yaw = zed.static_transform_xyz_rpy
    mount: NDArray[np.float64] = _urdf_link_pose(zed.parent_frame) @ transform_from_xyz_rpy(
        [x, y, z], [roll, pitch, yaw]
    )
    return mount


@pytest.mark.parametrize("name", ["head_zed", "top"])
def test_intrinsics_match_openarm(name: str) -> None:
    amr, arm = _sensor("openamrobot_v2", name), _sensor("openarm", name)
    assert amr.modality == arm.modality
    assert amr.intrinsics == arm.intrinsics
    assert amr.vla_feature_key == arm.vla_feature_key


@pytest.mark.parametrize(
    "link",
    ["openarm_left_base_link", "head_camera_link", "head_camera_bracket_link", "getac_tablet_link"],
)
def test_the_manifest_chain_is_the_urdf_chain(link: str) -> None:
    """The kernel poses the camera body and its neighbours from the manifest, TF from the URDF."""
    assert np.allclose(_pose(link), _urdf_link_pose(link), atol=1e-6)


def test_mount_is_the_openarm_cells_head_pose_relative_to_the_arm_bases() -> None:
    """The arm bases sit at +/-0.031 m on the carriage, oriented as OpenArm's.

    So the cell's ``openarm_base`` (their midpoint) pose is, on the left base, the same
    pose 0.031 m to the right, and its rotation is OpenArm's measured one. The lens is
    2.013 mm further forward than the cell's, clear of the Getac placeholder.
    """
    left, right = _pose("openarm_left_base_link"), _pose("openarm_right_base_link")
    assert np.allclose(left[:3, :3], np.eye(3)) and np.allclose(right[:3, :3], np.eye(3))
    assert np.allclose(left[:3, 3] - right[:3, 3], [0.0, 0.062, 0.0])
    zed, arm = _sensor("openamrobot_v2", "head_zed"), _sensor("openarm", "head_zed")
    assert zed.parent_frame == "openarm_left_base_link"
    assert not zed.static_transform_calibrated  # a design pose, not a measurement
    assert zed.static_transform_xyz_rpy is not None and arm.static_transform_xyz_rpy is not None
    assert zed.static_transform_xyz_rpy[3:] == arm.static_transform_xyz_rpy[3:]
    midpoint = (left[:3, 3] + right[:3, 3]) / 2.0
    in_midpoint = left[:3, 3] + np.array(zed.static_transform_xyz_rpy[:3]) - midpoint
    # The cell's camera_mounts xyz on openarm_base, to the manifest's 6 decimals, + 2 mm x.
    assert np.allclose(in_midpoint, [0.077587 + 0.002013, 0.027239, 0.099276], atol=1e-6)


def test_tf_isaac_and_twin_place_the_camera_at_one_pose() -> None:
    """The camera's pose at the zero configuration, by every path, within 1 mm / 0.1 deg."""
    from openral_sim.backends.isaac_sim import default_camera_mounts

    desc = _desc()
    urdf_links = {
        str(link.get("name")) for link in ET.parse(_AMR / "openamrobot_v2.urdf").iter("link")
    }
    tf = _tf_pose()
    mounts = default_camera_mounts(
        desc.model_copy(update={"assets": desc.assets.model_copy(update={"mjcf": None})}),
        _AMR,
        urdf_links | {desc.base_frame},
    )
    by_name = {s.name: s for s in desc.sensors}
    for name in ("head_zed", "top"):
        # Isaac: the link's pose, then the mount; `axes: world` is x forward, z up.
        mount = mounts[name]
        assert mount.axes == "world"
        assert mount.link is not None and mount.quat_wxyz is not None
        isaac = _urdf_link_pose(mount.link) @ transform_from_xyz_rpy(
            list(mount.xyz), [0.0, 0.0, 0.0]
        )
        isaac[:3, :3] = isaac[:3, :3] @ _rot_wxyz(mount.quat_wxyz)
        # The MuJoCo twin's camera rig: -z looks at `target`, +y toward `up`.
        placement = by_name[name].sim_placement
        assert placement is not None and placement.parent_body is not None
        twin = _urdf_link_pose(placement.parent_body) @ transform_from_xyz_rpy(
            list(placement.pos), [0.0, 0.0, 0.0]
        )
        quat = look_at_quat_wxyz(placement.pos, placement.target, up=placement.up, view_axis="-z")
        twin[:3, :3] = twin[:3, :3] @ _rot_wxyz(quat) @ _MJ_CAMERA_TO_BODY
        for other in (isaac, twin):
            assert np.linalg.norm(other[:3, 3] - tf[:3, 3]) < 1e-3
            assert _angle_deg(other[:3, :3], tf[:3, :3]) < 0.1
    # The ZED body (head_camera_link, upstream's placeholder box) rides the lens, 2.5 mm
    # behind it, with the same axes; its optical frame IS the lens.
    body = _urdf_link_pose("head_camera_link")
    assert _angle_deg(body[:3, :3], tf[:3, :3]) < 0.01
    assert np.allclose(
        np.linalg.inv(tf)[:3, :] @ np.append(body[:3, 3], 1.0), [-0.0025, 0, 0], atol=1e-6
    )
    optical = _urdf_link_pose("head_camera_optical_frame")
    assert np.linalg.norm(optical[:3, 3] - tf[:3, 3]) < 1e-6


def test_the_camera_looks_down_in_front_of_the_arms() -> None:
    """65.6 deg down (the cell's pitch); the centre ray crosses the arm bases' height
    0.12 m in front of them, between the two arms: where the carried tote sits."""
    tf = _tf_pose()
    view = tf[:3, 0]
    assert np.degrees(np.arcsin(-view[2])) == pytest.approx(65.6, abs=0.1)
    left, right = _pose("openarm_left_base_link")[:3, 3], _pose("openarm_right_base_link")[:3, 3]
    s = (left[2] - tf[2, 3]) / view[2]
    hit = tf[:3, 3] + s * view
    assert hit[0] - left[0] == pytest.approx(0.125, abs=0.01)
    assert right[1] < hit[1] < left[1]


def _head_zed_rays() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The lens (base_footprint) and unit pixel rays of ``head_zed``, every 16th pixel."""
    k = _sensor("openamrobot_v2", "head_zed").intrinsics
    assert k is not None
    u, v = np.meshgrid(np.arange(0, k.width, 16), np.arange(0, k.height, 16))
    # Optical (x right, y down, z forward) -> body (x forward, y left, z up).
    body = np.stack([np.ones(u.size), -(u.ravel() - k.cx) / k.fx, -(v.ravel() - k.cy) / k.fy])
    tf = _tf_pose()
    rays = tf[:3, :3] @ (body / np.linalg.norm(body, axis=0))
    return tf[:3, 3], rays.T


def test_the_self_filter_casts_the_head_zed_from_outside_every_robot_box() -> None:
    """The octomap self-filter casts each depth pixel from the lens into the manifest's
    collision boxes (slab entry). A lens inside a box enters it at t = 0 on every pixel,
    so the whole image reads as robot (holes UNCERTAIN, robot-occluded). The lens clears
    every box by at least 1 mm, and no ray enters one at its origin."""
    origin, rays = _head_zed_rays()
    for geom in _desc().collision_geometry or []:
        frame = _pose(geom.link_name) @ transform_from_xyz_rpy(
            list(geom.origin_xyz_rpy[:3]), list(geom.origin_xyz_rpy[3:])
        )
        assert isinstance(geom.shape, BoxShape), geom.link_name  # the body is all boxes
        half = np.array(geom.shape.half_extents_m)
        o = np.linalg.inv(frame)[:3, :] @ np.append(origin, 1.0)
        assert np.max(np.abs(o) - half) >= 1e-3 - 1e-9, geom.link_name
        d = rays @ frame[:3, :3]  # each ray in the box frame
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        near = np.nanmax(np.minimum(t1, t2), axis=1)
        far = np.nanmin(np.maximum(t1, t2), axis=1)
        entered = (near <= far) & (far > 0.0)
        assert np.all(near[entered] > 0.0), geom.link_name


def test_the_twin_cameras_sit_at_the_tf_mount_and_see_past_the_robot() -> None:
    """The twin's compiled cameras are at the TF pose. Cast from the lens, as a depth
    ray-caster does: ``top`` (the policy camera) sees only the base's front deck, the
    wider ``head_zed`` also the Getac face and head light below it, never the column,
    the arms or its own body."""
    mujoco = pytest.importorskip("mujoco")
    from openral_core import ROSConfigError
    from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL

    hal = OpenAMRobotV2MujocoHAL(description=_desc())
    try:
        hal.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamrobot_v2 twin unavailable: {exc}")
    try:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        mujoco.mj_forward(model, data)
        world_from_footprint = np.eye(4)
        footprint = model.body("base_footprint").id
        world_from_footprint[:3, 3] = data.xpos[footprint]
        world_from_footprint[:3, :3] = data.xmat[footprint].reshape(3, 3)
        tf = world_from_footprint @ _tf_pose()
        robot_root = model.body_rootid[footprint]
        geom = np.zeros(1, dtype=np.int32)
        for name, allowed in (("top", {"base_link"}), ("head_zed", _HEAD_ZED_SEES)):
            cam = model.camera(name).id
            assert np.linalg.norm(data.cam_xpos[cam] - tf[:3, 3]) < 1e-3
            rot = data.cam_xmat[cam].reshape(3, 3)
            assert _angle_deg(rot @ _MJ_CAMERA_TO_BODY, tf[:3, :3]) < 0.1
            k = _sensor("openamrobot_v2", name).intrinsics
            assert k is not None
            origin = data.cam_xpos[cam].copy()
            step = max(8, k.width // 84)
            seen: set[str] = set()
            for v in range(0, k.height, step):
                for u in range(0, k.width, step):
                    pixel = np.array([(u - k.cx) / k.fx, -(v - k.cy) / k.fy, -1.0])
                    ray = rot @ pixel / np.linalg.norm(pixel)
                    hit = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geom)
                    body = model.geom_bodyid[geom[0]]
                    if hit >= 0 and model.body_rootid[body] == robot_root:
                        seen.add(str(model.body(body).name))
            assert seen <= allowed, name
    finally:
        hal.disconnect()
