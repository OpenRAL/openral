# SPDX-License-Identifier: Apache-2.0
"""OpenAMRobot 2.0's head ZED: OpenArm's intrinsics, on upstream's head-camera mount.

The intrinsics match OpenArm's, so an OpenArm rSkill gets its raster and field of view.
The mount does not: OpenArm's measured mount, re-expressed from the arm bases (which
sit on the lift carriage as on the OpenArm body), lands inside upstream's Getac tablet,
so the camera rides upstream's ``head_camera_link`` instead. The TF mount and the sim
camera must agree, and the camera must see the world, not the robot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openral_core import RobotDescription, SensorSpec

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"


def _sensor(robot: str, name: str) -> SensorSpec:
    desc = RobotDescription.from_yaml(str(_ROBOTS / robot / "robot.yaml"))
    return next(s for s in desc.sensors if s.name == name)


@pytest.mark.parametrize("name", ["head_zed", "top"])
def test_intrinsics_match_openarm(name: str) -> None:
    amr, arm = _sensor("openamrobot_v2", name), _sensor("openarm", name)
    assert amr.modality == arm.modality
    assert amr.intrinsics == arm.intrinsics
    assert amr.vla_feature_key == arm.vla_feature_key


@pytest.mark.parametrize("name", ["head_zed", "top"])
def test_sim_camera_sits_at_the_tf_mount_on_upstreams_head_link(name: str) -> None:
    zed, spec = _sensor("openamrobot_v2", "head_zed"), _sensor("openamrobot_v2", name)
    assert zed.parent_frame == "head_camera_link"
    assert zed.static_transform_xyz_rpy is not None
    x, y, z, roll, pitch, yaw = zed.static_transform_xyz_rpy
    assert (roll, pitch, yaw) == (0.0, 0.0, 0.0)  # the ZED body frame IS head_camera_link's
    placement = spec.sim_placement
    assert placement is not None and placement.parent_body == "head_camera_link"
    assert np.allclose(placement.pos, [x, y, z])
    view = np.asarray(placement.target) - np.asarray(placement.pos)
    assert np.allclose(view / np.linalg.norm(view), [1.0, 0.0, 0.0])  # body +x
    assert np.allclose(placement.up, [0.0, 0.0, 1.0])


def test_the_head_camera_sees_the_world_not_the_robot() -> None:
    """Ray-cast the ``top`` raster in the twin: no pixel may land on the robot.

    OpenArm's mount on this robot put 100 % of the pixels on the Getac tablet.
    """
    mujoco = pytest.importorskip("mujoco")
    from openral_core import ROSConfigError
    from openral_hal.openamrobot_v2 import OpenAMRobotV2MujocoHAL

    desc = RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))
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
        top = _sensor("openamrobot_v2", "top")
        assert top.intrinsics is not None
        k = top.intrinsics
        cam = model.camera("top").id
        origin, rot = data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3)
        robot_root = model.body_rootid[model.body("base_footprint").id]
        geom = np.zeros(1, dtype=np.int32)
        on_robot = 0
        for v in range(0, k.height, 8):
            for u in range(0, k.width, 8):
                ray = rot @ np.array([(u - k.cx) / k.fx, -(v - k.cy) / k.fy, -1.0])
                hit = mujoco.mj_ray(
                    model, data, origin, ray / np.linalg.norm(ray), None, 1, -1, geom
                )
                if hit >= 0 and model.body_rootid[model.geom_bodyid[geom[0]]] == robot_root:
                    on_robot += 1
        assert on_robot == 0
    finally:
        hal.disconnect()
