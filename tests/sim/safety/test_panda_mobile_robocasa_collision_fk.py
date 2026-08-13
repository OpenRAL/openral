"""RoboCasa's ``base_link``, the depth extrinsic and the safety FK share one frame.

ADR-0095. ``base_link`` on ``/tf`` is RoboCasa's arm mount — the top of the
OmronMobileBase's 0.70 m pedestal — because ``MobileBaseBridge`` publishes
``odom -> base_link`` from ``base_pose_6dof()`` (robosuite's
``robot0_base_pos``). These tests pin, against the live kitchen, that:

* the MJCF body the sim bridge measures every ``base_link -> …`` extrinsic
  against really is the body ``base_pose_6dof()`` reports, and
* the manifest's collision-FK root is measured from that same body,

so the ``/openral/world_voxels`` grid the kernel rasterizes capsules against
and the capsules themselves live in one frame — the frame TF names.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.sim.conftest import mujoco_renderer_probe_error

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOT = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"


def _robocasa_unavailable() -> str:
    if importlib.util.find_spec("robocasa") is None:
        return "robocasa not installed"
    from openral_sim._deps import _has_robocasa_kitchen

    return "" if _has_robocasa_kitchen() else "RoboCasa kitchen fork is not active"


_ROBOCASA_ERROR = _robocasa_unavailable()
_RENDERER_ERROR = mujoco_renderer_probe_error() if not _ROBOCASA_ERROR else ""
# scipy arrives transitively with robosuite; it is not a declared OpenRAL dependency,
# so the lean CI env has to skip rather than fail collection.
_SCIPY_MISSING = importlib.util.find_spec("scipy") is None

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_ROBOCASA_ERROR), reason=_ROBOCASA_ERROR or "RoboCasa unavailable"),
    pytest.mark.skipif(
        bool(_RENDERER_ERROR), reason=_RENDERER_ERROR or "no MuJoCo offscreen renderer"
    ),
    pytest.mark.skipif(_SCIPY_MISSING, reason="scipy not installed"),
]


@pytest.fixture(scope="module")
def panda_mobile_hal() -> Iterator[Any]:
    from openral_core import RobotDescription
    from openral_hal import build_hal

    hal = build_hal(
        RobotDescription.from_yaml(_ROBOT),
        mode="sim",
        sim_env_yaml=str(_SCENE),
    )
    hal.connect()
    try:
        yield hal
    finally:
        hal.disconnect()


def test_base_frame_body_is_the_body_tf_publishes_as_base_link(panda_mobile_hal: Any) -> None:
    """The extrinsic parent body is the one ``odom -> base_link`` actually carries.

    ``resolve_base_frame_body_name`` is a *static* name resolution; this is the
    empirical link that makes it correct. If robosuite ever moved
    ``robot0_base_pos`` to another body, the sim camera extrinsic would silently
    drift from TF again and the world-voxel grid would stop being
    ``base_link``-referenced — which is exactly the ADR-0095 defect.
    """
    import mujoco
    from openral_hal.depth_cloud import resolve_base_frame_body_name

    model, data = panda_mobile_hal.mujoco_handles()
    base_body = resolve_base_frame_body_name(model, description=panda_mobile_hal.description)
    assert base_body == "mobilebase0_support"

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    published = panda_mobile_hal.base_pose_6dof()
    assert published is not None, "RoboCasa HAL must surface the 6-DoF base pose"
    assert np.asarray(data.xpos[base_id], dtype=np.float64) == pytest.approx(
        np.asarray(published[0], dtype=np.float64), abs=1e-6
    )


def test_collision_root_matches_robocasa_arm_mount(panda_mobile_hal: Any) -> None:
    """The manifest joint-1 origin matches RoboCasa link1 relative to base_link.

    Measured against the arm-mount body (see above), not the ground-level
    chassis root: PR #103 measured it against the chassis root and therefore
    pinned 1.033. ADR-0095 fixes the producer at source, so the FK root is the
    plain Franka 0.333 again and both halves land in the same commit — a
    half-applied change leaves the kernel a full pedestal off (HZ-0095-1).
    """
    import mujoco
    from openral_hal.depth_cloud import resolve_base_frame_body_name

    model, data = panda_mobile_hal.mujoco_handles()
    base_body = resolve_base_frame_body_name(model, description=panda_mobile_hal.description)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    link1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_link1")
    base_rotation = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    link1_in_base = base_rotation.T @ (
        np.asarray(data.xpos[link1_id], dtype=np.float64)
        - np.asarray(data.xpos[base_id], dtype=np.float64)
    )
    joint1 = next(
        joint for joint in panda_mobile_hal.description.joints if joint.name == "panda_joint1"
    )

    assert link1_in_base == pytest.approx(joint1.origin_xyz, abs=1e-6)


def test_depth_extrinsic_places_the_cloud_where_tf_says_it_does(panda_mobile_hal: Any) -> None:
    """End-to-end: sim depth -> published extrinsic -> ``base_link`` -> world truth.

    The real chain the kernel's world-voxel check depends on, on the real
    kitchen. ``octomap_server`` lifts the published cloud through
    ``odom <- base_link <- <cam>_optical_frame``; composing the published
    extrinsic with the published ``base_link`` pose must therefore reproduce the
    ray-cast's own world points. Before ADR-0095 this residual was a rigid
    +0.700 m in z — the pedestal — so every obstacle in the OctoMap, and every
    Nav2 / SLAM / dashboard consumer of it, sat a pedestal too high.
    """
    import mujoco
    from openral_hal.depth_cloud import (
        camera_optical_tf_to_base,
        depth_synth_kwargs,
        is_depth_sensor,
        resolve_base_frame_body_name,
        robot_self_body_ids,
    )
    from openral_sim.backends.depth_camera import synthesize_depth_pointcloud

    description = panda_mobile_hal.description
    spec = next(s for s in description.sensors if is_depth_sensor(s))
    model, data = panda_mobile_hal.mujoco_handles()
    kwargs = depth_synth_kwargs(spec, max_range_default=4.0)
    points = np.asarray(
        synthesize_depth_pointcloud(
            model=model,
            data=data,
            stride=4,
            exclude_body_ids=robot_self_body_ids(
                model, [j.sim_joint_name for j in description.joints]
            ),
            **kwargs,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    assert len(points) > 100, "the depth raster returned nothing to check"

    base_body = resolve_base_frame_body_name(model, description=description)
    xyz, quat = camera_optical_tf_to_base(
        model=model, data=data, camera_name=kwargs["camera_name"], base_body_name=base_body
    )
    qx, qy, qz, qw = quat
    rot_base_optical = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    cloud_base = points @ rot_base_optical.T + np.asarray(xyz)

    # ``base_link``'s pose on /tf, straight from the HAL's own publisher input.
    base_pos, base_quat = panda_mobile_hal.base_pose_6dof()
    bx, by, bz, bw = base_quat
    rot_world_base = np.asarray(
        [
            [1 - 2 * (by * by + bz * bz), 2 * (bx * by - bz * bw), 2 * (bx * bz + by * bw)],
            [2 * (bx * by + bz * bw), 1 - 2 * (bx * bx + bz * bz), 2 * (by * bz - bx * bw)],
            [2 * (bx * bz - by * bw), 2 * (by * bz + bx * bw), 1 - 2 * (bx * bx + by * by)],
        ],
        dtype=np.float64,
    )
    cloud_world_via_tf = cloud_base @ rot_world_base.T + np.asarray(base_pos, dtype=np.float64)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, kwargs["camera_name"])
    cam_rot = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    cloud_world_truth = points @ (cam_rot @ np.diag([1.0, -1.0, -1.0])).T + np.asarray(
        data.cam_xpos[cam_id], dtype=np.float64
    )

    residual = cloud_world_via_tf - cloud_world_truth
    assert np.abs(residual).max() < 1e-6, (
        f"TF-placed cloud is off by {np.abs(residual).max():.4f} m "
        f"(mean {residual.mean(axis=0)}) — base_link and the depth extrinsic disagree"
    )


def test_arm_boxes_enclose_the_real_robocasa_collision_meshes(
    panda_mobile_hal: Any,
) -> None:
    """Every world-collision OBB contains its matching RoboCasa arm mesh."""
    import mujoco
    from openral_core import BoxShape
    from scipy.spatial.transform import Rotation

    model, _data = panda_mobile_hal.mujoco_handles()
    for link_index in range(1, 8):
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"robot0_link{link_index}_collision",
        )
        geometry = next(
            item
            for item in panda_mobile_hal.description.collision_geometry
            if item.link_name == f"panda_link{link_index}"
        )
        assert isinstance(geometry.shape, BoxShape)

        aabb_center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
        aabb_half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64)
        geom_rotation = Rotation.from_quat(
            [
                model.geom_quat[geom_id, 1],
                model.geom_quat[geom_id, 2],
                model.geom_quat[geom_id, 3],
                model.geom_quat[geom_id, 0],
            ]
        ).as_matrix()
        corners = np.asarray(
            [
                aabb_center + aabb_half * np.asarray([x, y, z], dtype=np.float64)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        corners_in_link = corners @ geom_rotation.T + np.asarray(model.geom_pos[geom_id])

        box_origin = np.asarray(geometry.origin_xyz_rpy[:3])
        box_rotation = Rotation.from_euler("xyz", geometry.origin_xyz_rpy[3:]).as_matrix()
        corners_in_box = (corners_in_link - box_origin) @ box_rotation

        assert np.all(
            np.abs(corners_in_box)
            <= np.asarray(geometry.shape.half_extents_m, dtype=np.float64) + 1e-6
        ), geometry.link_name
