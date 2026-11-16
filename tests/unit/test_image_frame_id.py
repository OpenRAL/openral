"""``image_frame_id``: the tf2 frame a camera's pixels are in, and the Isaac `top` lift through it.

On the OpenAMRobot 2.0 the detector watches ``top`` (``frame_id: world``, a sim stand-in;
``shares_mount_with: head_zed``). Its pixels are in head_zed's optical frame, which TF places
on the mobile base; ``world`` is not on its tree, so the lift used to drop every batch.

Real fixtures (CLAUDE.md §1.11): the in-tree manifests, the Isaac backend's own render
description (``_render_desc`` at the aisle scene's 224 px) and a real ``VoxelFrustumLifter``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from openral_core import ObjectDetection2D, RobotDescription, scale_intrinsics_to
from openral_core.depth_extrinsic import image_frame_id
from openral_world_state import VoxelFrustumLifter

_AMR = "robots/openamrobot_v2/robot.yaml"


def test_a_shared_mount_camera_is_in_the_mount_optical_frame() -> None:
    amr = RobotDescription.from_yaml(_AMR).sensors
    frames = {s.name: image_frame_id(s, amr) for s in amr if s.modality == "rgb"}
    assert frames == {
        "top": "zed_camera_link_optical_frame",
        "wrist_left": "left_wrist_camera_optical_frame",
        "wrist_right": "right_wrist_camera_optical_frame",
    }
    so101 = RobotDescription.from_yaml("robots/so101_follower/robot.yaml").sensors
    assert {s.name: image_frame_id(s, so101) for s in so101 if s.modality == "rgb"} == {
        s.name: s.frame_id for s in so101 if s.modality == "rgb"
    }


def test_isaac_reports_the_k_it_renders_top_with() -> None:
    from openral_sim.backends.isaac_sim import (
        _render_desc,
        _rendered_rgb_intrinsics,
        _sensor_dict,
    )

    desc = RobotDescription.from_yaml(_AMR)
    render, _ = _render_desc(desc, {"top": 224}, depth_cameras=True)
    ks = _rendered_rgb_intrinsics({"sensors": [_sensor_dict(s) for s in render.sensors]})
    manifest = next(s for s in desc.sensors if s.name == "top").intrinsics
    assert manifest is not None
    assert ks["top"] == scale_intrinsics_to(manifest, 400, 224)
    assert set(ks) == {"top", "wrist_left", "wrist_right"}


def _rot_rpy(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Fixed-axis RPY (tf2 / URDF): Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def test_a_top_box_lifts_through_the_head_zed_optical_frame() -> None:
    """A 224 px Isaac `top` box lifts onto the object through the frame TF resolves."""
    desc = RobotDescription.from_yaml(_AMR)
    top = next(s for s in desc.sensors if s.name == "top")
    zed = next(s for s in desc.sensors if s.name == "head_zed")
    assert image_frame_id(top, desc.sensors) == f"{zed.frame_id}_optical_frame"
    mount = zed.static_transform_xyz_rpy
    assert mount is not None and top.intrinsics is not None
    # parent <- zed_camera_link (manifest mount) <- its optical child (BODY_TO_OPTICAL).
    t_parent_body = np.eye(4)
    t_parent_body[:3, :3] = _rot_rpy(*mount[3:])
    t_parent_body[:3, 3] = mount[:3]
    t_body_opt = np.eye(4)
    t_body_opt[:3, :3] = _rot_rpy(-math.pi / 2, 0.0, -math.pi / 2)
    t_cam_from_parent = np.asarray(np.linalg.inv(t_parent_body @ t_body_opt), dtype=np.float64)

    k = scale_intrinsics_to(top.intrinsics, 400, 224)
    obj = (t_parent_body @ t_body_opt @ np.array([0.05, 0.03, 0.8, 1.0]))[:3]
    p = t_cam_from_parent @ np.append(obj, 1.0)
    u, v = k.fx * p[0] / p[2] + k.cx, k.fy * p[1] / p[2] + k.cy
    assert 0 < u < 400 and 0 < v < 224, (u, v)
    offs = (np.arange(5) - 2) * 0.02
    cube = np.array([obj + np.array([x, y, z]) for x in offs for y in offs for z in offs])

    (lifted,) = VoxelFrustumLifter().lift(
        detections=[
            ObjectDetection2D(
                label="box",
                confidence=0.9,
                bbox_xyxy=(int(u - 15), int(v - 15), int(u + 15), int(v + 15)),
            )
        ],
        occupied_centers_base=cube,
        intrinsics=k,
        frame_size=(400, 224),
        t_cam_from_base=t_cam_from_parent,
        t_map_from_base=np.eye(4),
        map_frame=str(zed.parent_frame),
    )
    assert np.linalg.norm(np.array(lifted.pose.xyz) - obj) < 0.02, lifted.pose.xyz
