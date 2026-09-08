# SPDX-License-Identifier: Apache-2.0
"""The world-voxel grid must be genuinely ``base_link``-referenced in x, y AND z.

``openral_msgs/OccupancyVoxels`` carries no TF of its own — ``header.frame_id``
names the frame ``origin`` is measured in, and the C++ safety kernel rasterizes
FK'd link capsules against that grid untransformed. So the grid producer and the
kernel's collision FK must agree, to the millimetre, on what ``base_link`` means.

RoboCasa's ``PandaMobile`` has a pedestal duality: MJCF chassis root
``mobilebase0_base`` (world z 0) vs. elevated arm mount ``mobilebase0_support``
(+0.70 m) — TF's ``base_link`` is the latter (``MobileBaseBridge`` publishes
``odom -> base_link`` from ``base_pose_6dof()``). Pre-ADR-0095 the camera
extrinsic was measured against the chassis root instead, so the depth cloud (and
the OctoMap/grid lowered from it) was 0.70 m off in z for every TF consumer
(Nav2, SLAM, the dashboard).

These tests pin the ADR-0095 Option A contract on a real ``mujoco.MjModel``
reproducing the pedestal topology:

    mobilebase0_base (world z 0.000)      chassis root, wheels
      └── mobilebase0_support (+0.700)    arm mount == ``base_link``
            └── robot0_link1  (+0.333)    Franka URDF joint-1 origin

``base_link`` stays at the pedestal top (a past regression moved
``world_to_base.position.z`` to 0.0 instead); the camera extrinsic moves onto it.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# RoboCasa's OmronMobileBase pedestal: how far the arm mount (and therefore
# TF's ``base_link``) sits above the ground-level chassis root.
ROBOCASA_PEDESTAL_M = 0.700
# Franka URDF ``panda_joint1`` origin above ``panda_link0``.
FRANKA_JOINT1_Z_M = 0.333
# World height of the counter slab's top face in the fixture below.
COUNTER_TOP_WORLD_Z_M = 0.920

# Real MuJoCo model: RoboCasa pedestal topology + arm-mounted camera on a counter
# slab, mirroring ``robot0_agentview_left`` on ``PickPlaceCounterToCabinet``. Body/joint
# names follow robosuite's auto-prefix convention, matching manifest-driven resolution.
_PEDESTAL_MJCF = """
<mujoco model="robocasa_pedestal_frame_contract">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="mobilebase0_base" pos="0 0 0">
      <joint name="mobilebase0_joint_mobile_forward" type="slide" axis="1 0 0"/>
      <joint name="mobilebase0_joint_mobile_side" type="slide" axis="0 1 0"/>
      <joint name="mobilebase0_joint_mobile_yaw" type="hinge" axis="0 0 1"/>
      <geom name="mobilebase0_wheeled_base_col" type="box" size="0.35 0.25 0.1" pos="0 0 0.1"/>
      <body name="mobilebase0_support" pos="0 0 0.7">
        <geom name="mobilebase0_support_col" type="box" size="0.2 0.2 0.01"/>
        <camera name="robot0_agentview_left" pos="0.25 0 0.5" zaxis="-0.70710678 0 0.70710678"/>
        <body name="robot0_link1" pos="0 0 0.333">
          <geom name="robot0_link1_col" type="capsule" size="0.06" fromto="0 0 0 0 0 0.2"/>
        </body>
      </body>
    </body>
    <body name="counter_1_left_group_main" pos="0.5 0 0.46">
      <geom name="counter_1_left_group_top_left_1" type="box" size="0.3 0.5 0.015" pos="0 0 0.445"/>
    </body>
  </worldbody>
</mujoco>
"""

# Pinhole intrinsics for the fixture's depth raster (small: this is a unit test).
_W, _H, _F = 48, 36, 30.0


def _pedestal_model_and_data() -> tuple[object, object]:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_PEDESTAL_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body_pose(model: object, data: object, name: str) -> tuple[NDArray[np.float64], ...]:
    """``(world position, world rotation matrix)`` of a named MJCF body."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} missing from the fixture"
    return (
        np.asarray(data.xpos[bid], dtype=np.float64),  # type: ignore[attr-defined]
        np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3),  # type: ignore[attr-defined]
    )


def _quat_xyzw_to_matrix(quat: tuple[float, float, float, float]) -> NDArray[np.float64]:
    x, y, z, w = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _depth_cloud_in_optical_frame(model: object, data: object) -> NDArray[np.float64]:
    """The raster the sim bridge publishes: self-filtered, camera-optical frame."""
    from openral_hal.depth_cloud import robot_self_body_ids
    from openral_sim.backends.depth_camera import synthesize_depth_pointcloud

    description = _panda_mobile_description()
    self_bodies = robot_self_body_ids(
        model,
        [j.sim_joint_name for j in description.joints],  # type: ignore[attr-defined]
    )
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="robot0_agentview_left",
        width=_W,
        height=_H,
        fx=_F,
        fy=_F,
        cx=_W / 2.0,
        cy=_H / 2.0,
        max_range_m=6.0,
        exclude_body_ids=self_bodies,
    )
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _cloud_in_base_frame(model: object, data: object) -> NDArray[np.float64]:
    """The published depth cloud, placed by the published ``base_link -> optical`` TF."""
    from openral_hal.depth_cloud import camera_optical_tf_to_base, resolve_base_frame_body_name

    base_body = resolve_base_frame_body_name(model, description=_panda_mobile_description())
    assert base_body is not None
    xyz, quat = camera_optical_tf_to_base(
        model=model,
        data=data,
        camera_name="robot0_agentview_left",
        base_body_name=base_body,
    )
    rot = _quat_xyzw_to_matrix(quat)
    return _depth_cloud_in_optical_frame(model, data) @ rot.T + np.asarray(xyz)


def _panda_mobile_description() -> object:
    from openral_core import RobotDescription

    return RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")


# ── the frame contract ────────────────────────────────────────────────────────


def test_base_frame_body_is_the_elevated_arm_mount_not_the_chassis_root() -> None:
    """``base_link``'s MJCF body is the pedestal-top mount, 0.70 m above the chassis root.

    Chassis root = depth self-filter's ``mj_multiRay`` exclude anchor; arm mount =
    what ``MobileBaseBridge`` publishes as ``odom -> base_link``.
    """
    from openral_hal.depth_cloud import resolve_base_body_name, resolve_base_frame_body_name

    model, data = _pedestal_model_and_data()
    description = _panda_mobile_description()

    assert resolve_base_body_name(model, description=description) == "mobilebase0_base"
    assert resolve_base_frame_body_name(model, description=description) == "mobilebase0_support"

    chassis, _ = _body_pose(model, data, "mobilebase0_base")
    mount, _ = _body_pose(model, data, "mobilebase0_support")
    assert float(mount[2] - chassis[2]) == pytest.approx(ROBOCASA_PEDESTAL_M, abs=1e-9)


_FIXED_BASE_MJCF = """
<mujoco model="libero_style_fixed_arm">
  <worldbody>
    <body name="robot0_base" pos="-0.66 0 0.912">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="robot0_link1" pos="0 0 0.333"><geom type="sphere" size="0.05"/></body>
    </body>
  </worldbody>
</mujoco>
"""


def test_fixed_base_arm_resolves_the_same_body_as_before() -> None:
    """No ``_support`` body → ADR-0095 changes nothing for fixed-base robosuite arms
    (LIBERO franka/ur5e): chassis root and ``base_frame`` body are the same MJCF
    body; only pedestal robots see any change.
    """
    mujoco = pytest.importorskip("mujoco")
    from openral_core import RobotDescription
    from openral_hal.depth_cloud import resolve_base_body_name, resolve_base_frame_body_name

    model = mujoco.MjModel.from_xml_string(_FIXED_BASE_MJCF)
    description = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
    assert resolve_base_frame_body_name(model, description=description) == resolve_base_body_name(
        model, description=description
    )
    assert resolve_base_frame_body_name(model) == "robot0_base"


def test_depth_cloud_lands_where_tf_says_it_does_in_all_three_axes() -> None:
    """Composing the published camera extrinsic with the published ``base_link`` pose
    (``octomap_server``'s ``odom <- base_link <- <cam>_optical_frame`` chain) must put
    every point back at MuJoCo's own world truth; a frame mismatch shows up here as a
    rigid 0.700 m z offset.
    """
    model, data = _pedestal_model_and_data()

    cloud_base = _cloud_in_base_frame(model, data)
    assert len(cloud_base) > 100, "fixture camera saw nothing; the raster is not exercised"

    # ``base_link``'s pose on /tf: the arm mount, per ``base_pose_6dof()``.
    base_pos, base_rot = _body_pose(model, data, "mobilebase0_support")
    cloud_world_via_tf = cloud_base @ base_rot.T + base_pos

    # MuJoCo's own truth for the same rays: the camera's live world pose.
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)  # type: ignore[attr-defined]
    cam_rot = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)  # type: ignore[attr-defined]
    optical_in_cam = np.diag([1.0, -1.0, -1.0])  # REP-103 optical frame vs MuJoCo camera
    cloud_world_truth = (
        _depth_cloud_in_optical_frame(model, data) @ (cam_rot @ optical_in_cam).T + cam_pos
    )

    residual = cloud_world_via_tf - cloud_world_truth
    assert np.abs(residual).max() < 1e-6, (
        "cloud placed by TF disagrees with MuJoCo world truth by "
        f"{np.abs(residual).max():.4f} m (mean {residual.mean(axis=0)})"
    )


def test_grid_cells_over_the_counter_sit_at_the_base_link_height_of_the_counter() -> None:
    """Occupied grid cells are base-referenced in z, not world-referenced.

    ``openral_octomap_bridge`` rasterizes voxel centres in ``header.frame_id``
    (``base_link``): world z 0.920 counter top -> grid z 0.920 - 0.700 = 0.220.
    Pre-ADR-0095 it landed at 0.920 (PR #103's FK-root offset masked the mismatch).
    """
    model, data = _pedestal_model_and_data()
    cloud_base = _cloud_in_base_frame(model, data)

    resolution = 0.025
    origin = np.asarray([-0.8, -0.8, -0.3])  # the bridge's deploy-sim grid
    cells = np.floor((cloud_base - origin) / resolution).astype(np.int64)
    centres = origin + (cells + 0.5) * resolution

    # Points over the counter footprint, in base_link (counter spans world
    # x in [0.2, 0.8], y in [-0.5, 0.5]; the mount sits at world x=y=0).
    over_counter = (centres[:, 0] > 0.25) & (centres[:, 0] < 0.75) & (np.abs(centres[:, 1]) < 0.45)
    assert over_counter.sum() > 20, "no rasterized cells over the counter footprint"

    expected_z = COUNTER_TOP_WORLD_Z_M - ROBOCASA_PEDESTAL_M
    top_z = float(centres[over_counter, 2].max())
    assert top_z == pytest.approx(expected_z, abs=resolution), (
        f"counter top rasterized at base-frame z={top_z:.4f}, expected {expected_z:.4f} "
        f"(world {COUNTER_TOP_WORLD_Z_M} minus the {ROBOCASA_PEDESTAL_M} m pedestal)"
    )


def test_collision_fk_root_places_link1_where_the_mjcf_does() -> None:
    """``panda_joint1``'s manifest origin is measured from the same ``base_link`` the
    kernel FKs capsules from: plain Franka URDF 0.333 m, not 0.333 + the 0.700 m
    pedestal (already carried by ``odom -> base_link`` — double-counted if also
    added to the FK root).
    """
    model, data = _pedestal_model_and_data()
    description = _panda_mobile_description()
    joint1 = next(j for j in description.joints if j.name == "panda_joint1")  # type: ignore[attr-defined]

    mount_pos, mount_rot = _body_pose(model, data, "mobilebase0_support")
    link1_pos, _ = _body_pose(model, data, "robot0_link1")
    link1_in_base = mount_rot.T @ (link1_pos - mount_pos)

    assert joint1.origin_xyz == (0.0, 0.0, FRANKA_JOINT1_Z_M)
    assert link1_in_base == pytest.approx(joint1.origin_xyz, abs=1e-9)
    # …and the world height the two conventions must agree on.
    assert float(link1_pos[2]) == pytest.approx(ROBOCASA_PEDESTAL_M + FRANKA_JOINT1_Z_M, abs=1e-9)


def test_frame_alignment_does_not_change_the_kernels_protective_envelope() -> None:
    """ADR-0095's two equal-and-opposite -0.700 m shifts (grid content; FK root
    1.033 -> 0.333) leave their difference — the only thing a capsule-vs-voxel
    distance depends on — invariant.
    """
    # Pre-ADR-0095: grid content at world z, FK root at 1.033 above base_link.
    before = (COUNTER_TOP_WORLD_Z_M) - (1.033)
    # Post-ADR-0095: grid content base-referenced, FK root at 0.333.
    after = (COUNTER_TOP_WORLD_Z_M - ROBOCASA_PEDESTAL_M) - (FRANKA_JOINT1_Z_M)
    assert before == pytest.approx(after, abs=1e-12)
