"""``image_frame_id``: the tf2 frame a camera's pixels are in, and the Isaac `top` lift through it.

On the OpenAMRobot 2.0 the detector watches ``top`` (``frame_id: world``, a sim stand-in;
``shares_mount_with: head_zed``). Its pixels are in head_zed's optical frame, which TF places
on the mobile base; ``world`` is not on its tree, so the lift used to drop every batch.

Real fixtures (CLAUDE.md §1.11): the in-tree manifests, the Isaac backend's own render
description (``_render_desc`` at the aisle scene's 224 px) and a real ``VoxelFrustumLifter``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
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


def test_a_body_convention_mount_is_in_its_optical_child() -> None:
    """MakerMods ``top`` is mounted at ``top_camera`` (x forward): its pixels are in the
    REP-103 optical child the sim bridge hangs under it, never the body frame itself."""
    metal = RobotDescription.from_yaml("robots/makermods_metal/robot.yaml").sensors
    top = next(s for s in metal if s.name == "top")
    assert top.frame_id == "top_camera" and top.parent_frame is not None
    assert image_frame_id(top, metal) == "top_camera_optical_frame"


def test_a_rendered_lens_reaches_the_camera_info() -> None:
    """Isaac renders some cameras with a lens; the CameraInfo carries it, and a fisheye
    (``equidistant``) keeps its model even with zero coefficients — it is never pinhole."""
    pytest.importorskip("sensor_msgs")
    from openral_hal.depth_cloud import camera_info_from_intrinsics

    def info(model: str, d: list[float]) -> tuple[str, list[float]]:
        msg = camera_info_from_intrinsics(
            width=4,
            height=4,
            fx=1.0,
            fy=1.0,
            cx=2.0,
            cy=2.0,
            frame_id="f",
            distortion_model=model,
            d=d,
        )
        return msg.distortion_model, list(msg.d)

    rational = [0.1, -0.2, 0.0, 0.0, 0.05, 0.01, 0.0, 0.0]
    assert info("rational_polynomial", rational) == ("rational_polynomial", rational)
    assert info("equidistant", []) == ("equidistant", [0.0] * 4)
    assert info("none", []) == ("plumb_bob", [0.0] * 5)


def test_isaac_reports_the_k_it_renders_top_with() -> None:
    from openral_sim.backends.isaac_sim import (
        _render_desc,
        _rendered_rgb_intrinsics,
        _sensor_dict,
    )

    desc = RobotDescription.from_yaml(_AMR)
    render, _ = _render_desc(desc, {"top": 224}, depth_cameras=True)
    sensors = [_sensor_dict(s) for s in render.sensors]
    ks = _rendered_rgb_intrinsics({"sensors": sensors})
    manifest = next(s for s in desc.sensors if s.name == "top").intrinsics
    assert manifest is not None
    assert ks["top"] == scale_intrinsics_to(manifest, 400, 224)
    assert set(ks) == {"top", "wrist_left", "wrist_right"}
    # A mount that sets hfov_deg renders an ideal pinhole from it: no K is claimed.
    mounts = {"wrist_left": {"hfov_deg": 70.0}, "wrist_right": {"hfov_deg": None}}
    ks = _rendered_rgb_intrinsics({"sensors": sensors, "camera_mounts": mounts})
    assert set(ks) == {"top", "wrist_right"}


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
    """A 224 px Isaac `top` box lifts onto the object through the frame TF resolves.

    The camera pose is what a real tf2 buffer answers for the frame ``image_frame_id``
    names, fed the manifest mount and the bridge's ``BODY_TO_OPTICAL_QUAT_XYZW`` child —
    independent of the RPY reference used to place the object."""
    tf2_ros = pytest.importorskip("tf2_ros")
    rclpy = pytest.importorskip("rclpy")
    from geometry_msgs.msg import TransformStamped
    from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
    from openral_hal.depth_cloud import BODY_TO_OPTICAL_QUAT_XYZW

    desc = RobotDescription.from_yaml(_AMR)
    top = next(s for s in desc.sensors if s.name == "top")
    zed = next(s for s in desc.sensors if s.name == "head_zed")
    mount = zed.static_transform_xyz_rpy
    assert mount is not None and top.intrinsics is not None and zed.parent_frame is not None
    # Reference: parent <- zed_camera_link (manifest mount) <- its optical child (REP-103).
    t_parent_body = np.eye(4)
    t_parent_body[:3, :3] = _rot_rpy(*mount[3:])
    t_parent_body[:3, 3] = mount[:3]
    t_body_opt = np.eye(4)
    t_body_opt[:3, :3] = _rot_rpy(-math.pi / 2, 0.0, -math.pi / 2)

    def static(
        parent: str, child: str, xyz: tuple[float, ...], q: tuple[float, ...]
    ) -> Any:  # reason: geometry_msgs TransformStamped, imported lazily
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = q
        return t

    w, x, y, z = rotation_to_quat_wxyz(t_parent_body[:3, :3])
    buffer = tf2_ros.Buffer()
    buffer.set_transform_static(
        static(str(zed.parent_frame), str(zed.frame_id), tuple(mount[:3]), (x, y, z, w)), "test"
    )
    optical = f"{zed.frame_id}_optical_frame"
    buffer.set_transform_static(
        static(str(zed.frame_id), optical, (0.0, 0.0, 0.0), BODY_TO_OPTICAL_QUAT_XYZW), "test"
    )
    frame = image_frame_id(top, desc.sensors)
    tf = buffer.lookup_transform(frame, str(zed.parent_frame), rclpy.time.Time()).transform
    tr, q = tf.translation, tf.rotation
    t_cam_from_parent = homogeneous_from_quat_xyz((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))

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
