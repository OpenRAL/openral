# SPDX-License-Identifier: Apache-2.0
"""OpenAMRobot 2.0's head ZED: OpenArm's intrinsics, at the restock cell's head pose.

The intrinsics match OpenArm's, so an OpenArm rSkill gets its raster and field of view.
The mount is the fixed OpenArm restock cell's head-camera pose relative to the arm
bases (the robot owner confirmed the real head ZED looks down at the tote and shelf like
the cell's), on ``openarm_left_base_link``. One pose, three readers, which must agree:
TF (robot_state_publisher's URDF chain + the deploy launch's static
``parent_frame -> frame_id`` from ``static_transform_xyz_rpy``), Isaac
(``default_camera_mounts``: the same transform, ``axes: world``) and the MuJoCo twin
(the camera rig's ``sim_placement``: ``-z`` toward ``target``, ``+y`` toward ``up``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import RobotDescription, SensorSpec
from openral_core.geometry import look_at_quat_wxyz
from openral_core.kinematics import transform_from_xyz_rpy

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_AMR = _ROBOTS / "openamrobot_v2"
# MuJoCo camera axes (x right, y up, looks down -z) -> body axes (x forward, y left, z up).
_MJ_CAMERA_TO_BODY = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T


def _sensor(robot: str, name: str) -> SensorSpec:
    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    return next(s for s in desc.sensors if s.name == name)


def _quat_wxyz_matrix(q: tuple[float, ...]) -> NDArray[np.float64]:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _urdf_link_pose(link: str) -> NDArray[np.float64]:
    """``base_footprint <- link`` at the zero configuration: the URDF chain RSP publishes."""
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


def test_mount_is_the_restock_cells_head_pose_relative_to_the_arm_bases() -> None:
    """The arm bases sit at +/-0.031 m on the carriage, oriented as OpenArm's.

    So the cell's ``openarm_base`` (their midpoint) pose is, on the left base, the same
    pose 0.031 m to the right, and its rotation is OpenArm's measured one.
    """
    left, right = (
        _urdf_link_pose("openarm_left_base_link"),
        _urdf_link_pose("openarm_right_base_link"),
    )
    assert np.allclose(left[:3, :3], np.eye(3)) and np.allclose(right[:3, :3], np.eye(3))
    assert np.allclose(left[:3, 3] - right[:3, 3], [0.0, 0.062, 0.0])
    zed, arm = _sensor("openamrobot_v2", "head_zed"), _sensor("openarm", "head_zed")
    assert zed.parent_frame == "openarm_left_base_link"
    assert not zed.static_transform_calibrated  # a design pose, not a measurement
    assert zed.static_transform_xyz_rpy is not None and arm.static_transform_xyz_rpy is not None
    assert zed.static_transform_xyz_rpy[3:] == arm.static_transform_xyz_rpy[3:]
    midpoint = (left[:3, 3] + right[:3, 3]) / 2.0
    in_midpoint = left[:3, 3] + np.array(zed.static_transform_xyz_rpy[:3]) - midpoint
    # The cell's camera_mounts xyz on openarm_base, to the manifest's 6 decimals.
    assert np.allclose(in_midpoint, [0.077587, 0.027239, 0.099276], atol=1e-6)


def test_tf_isaac_and_twin_place_the_camera_at_one_pose() -> None:
    """The camera's pose at the zero configuration, by every path, within 1 mm / 0.1 deg."""
    from openral_sim.backends.isaac_sim import default_camera_mounts

    desc = RobotDescription.from_yaml(str(_AMR / "robot.yaml"))
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
        isaac[:3, :3] = isaac[:3, :3] @ _quat_wxyz_matrix(mount.quat_wxyz)
        # The MuJoCo twin's camera rig: -z looks at `target`, +y toward `up`.
        placement = by_name[name].sim_placement
        assert placement is not None and placement.parent_body is not None
        twin = _urdf_link_pose(placement.parent_body) @ transform_from_xyz_rpy(
            list(placement.pos), [0.0, 0.0, 0.0]
        )
        quat = look_at_quat_wxyz(placement.pos, placement.target, up=placement.up, view_axis="-z")
        twin[:3, :3] = twin[:3, :3] @ _quat_wxyz_matrix(quat) @ _MJ_CAMERA_TO_BODY
        for other in (isaac, twin):
            assert np.linalg.norm(other[:3, 3] - tf[:3, 3]) < 1e-3
            assert _angle_deg(other[:3, :3], tf[:3, :3]) < 0.1


def test_the_camera_looks_down_and_forward_at_the_tote() -> None:
    """The tote rides the carriage in front of and below the arms: the view goes there."""
    tf = _tf_pose()
    view = tf[:3, 0]
    assert view[0] > 0.0 and view[2] < 0.0
    assert np.degrees(np.arcsin(-view[2])) == pytest.approx(65.6, abs=0.1)  # the cell's pitch
    # The centre ray reaches the arm bases' height ahead of them, then keeps descending.
    arm_base = _urdf_link_pose("openarm_left_base_link")[:3, 3]
    s = (arm_base[2] - tf[2, 3]) / view[2]
    assert s > 0.0 and tf[0, 3] + s * view[0] > arm_base[0]


def test_the_twin_cameras_sit_at_the_tf_mount_and_see_past_the_robot() -> None:
    """The twin's compiled cameras are at the TF pose; ``top`` sees only the base's deck.

    Rays start at the render's near plane, as the renderer draws them: the lens sits on
    the face of upstream's Getac placeholder box. The policy camera may show the base's
    front deck at its bottom edge, never the tablet, the column or the head.
    """
    mujoco = pytest.importorskip("mujoco")
    from openral_core import ROSConfigError
    from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL

    desc = RobotDescription.from_yaml(str(_AMR / "robot.yaml"))
    hal = OpenAMRobotV2MujocoHAL(description=desc)
    try:
        hal.connect()
    except ROSConfigError as exc:  # pinned mesh clone unreachable (offline host)
        pytest.skip(f"openamrobot_v2 twin unavailable: {exc}")
    try:
        handles = hal.mujoco_handles()
        assert handles is not None
        model, data = handles
        mujoco.mj_forward(model, data)
        tf = _tf_pose()
        world_from_footprint = np.eye(4)
        footprint = model.body("base_footprint").id
        world_from_footprint[:3, 3] = data.xpos[footprint]
        world_from_footprint[:3, :3] = data.xmat[footprint].reshape(3, 3)
        tf = world_from_footprint @ tf
        for name in ("head_zed", "top"):
            cam = model.camera(name).id
            assert np.linalg.norm(data.cam_xpos[cam] - tf[:3, 3]) < 1e-3
            rot = data.cam_xmat[cam].reshape(3, 3) @ _MJ_CAMERA_TO_BODY
            assert _angle_deg(rot, tf[:3, :3]) < 0.1

        top = _sensor("openamrobot_v2", "top")
        assert top.intrinsics is not None
        k = top.intrinsics
        cam = model.camera("top").id
        origin, rot = data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3)
        znear = float(model.vis.map.znear * model.stat.extent)
        robot_root = model.body_rootid[footprint]
        geom = np.zeros(1, dtype=np.int32)
        seen: set[str] = set()
        for v in range(0, k.height, 8):
            for u in range(0, k.width, 8):
                pixel = np.array([(u - k.cx) / k.fx, -(v - k.cy) / k.fy, -1.0])
                ray = rot @ pixel / np.linalg.norm(pixel)
                start = origin + ray * znear * float(np.linalg.norm(pixel))
                hit = mujoco.mj_ray(model, data, start, ray, None, 1, -1, geom)
                body = model.geom_bodyid[geom[0]]
                if hit >= 0 and model.body_rootid[body] == robot_root:
                    seen.add(str(model.body(body).name))
        assert seen <= {"base_link"}
    finally:
        hal.disconnect()
