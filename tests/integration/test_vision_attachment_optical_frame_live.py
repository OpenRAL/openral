# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: grasp masks back-project in the depth header's optical frame, through the driver's K.

Verified on the OpenArm Thor cell 2026-10-02: the ZED publishes depth
(``depth/depth_registered``) and rectified RGB both in
``zed_left_camera_frame_optical``, with ``depth/camera_info`` K = [1498.18, 0,
936.11; 0, 1498.18, 541.81]. The bridge used to look tf2 up for the manifest's
``head_zed.frame_id`` — ``zed_camera_link``, the ZED *body* frame — and project
through the manifest's nominal fx = 960, so every payload it measured was
rotated by the body->optical rotation and scaled by 56 %.

Real rclpy, real ``VisionAttachmentBridge`` on the bimanual OpenArm manifest,
real tf2 with the manifest's ``openarm_base -> zed_camera_link`` mount and the
driver's REP-103 ``zed_camera_link -> zed_left_camera_frame_optical`` hop, real
``openral_msgs`` on the wire. The segmenter is a ``SegmentInView`` peer that
masks a disc around the TCP pixel it is sent — a real DDS service at a process
boundary standing in for SAM 2 (CLAUDE.md §1.11); it also asserts the request
arrives in the optical frame (empty ``frame_id``).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_vision_attachment_optical_frame_live.py
"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first."
)

pytestmark = pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)

_ROBOT_YAML = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"
_CAMERA = "head_zed"
_OPTICAL = "zed_left_camera_frame_optical"
# The Thor ZED Mini's /zed/zed_node/depth/camera_info, measured 2026-10-02.
_THOR_K = (1498.18, 1498.18, 936.11, 541.81)
_FULL = (1080, 1920)
_SERVICE = "/openral/perception/segment_in_view_optical_itest"
# Where the test puts the TCP, in the optical frame — inside the producer's
# 1 m depth gate, slightly off-axis so a wrong principal point would show.
_TCP_IN_OPTICAL = np.array([0.03, -0.02, 0.6])
_SLAB_HALF_M = 0.04
_MASK_RADIUS_PX = 50


def _wait_until(predicate: Any, *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis RPY (tf2 / URDF convention): Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
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
    return rz @ ry @ rx


def _homogeneous(rot: np.ndarray, xyz: Any) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def _quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w-major; enough for the rotations used here)."""
    w = math.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    if w > 1e-6:
        return (
            (rot[2, 1] - rot[1, 2]) / (4 * w),
            (rot[0, 2] - rot[2, 0]) / (4 * w),
            (rot[1, 0] - rot[0, 1]) / (4 * w),
            w,
        )
    x = math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])) / 2.0
    return (x, (rot[0, 1] + rot[1, 0]) / (4 * x), (rot[0, 2] + rot[2, 0]) / (4 * x), 0.0)


def _tf_msg(parent: str, child: str, t: np.ndarray) -> Any:
    from geometry_msgs.msg import TransformStamped

    msg = TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
        float(v) for v in t[:3, 3]
    )
    q = _quat_xyzw(t[:3, :3])
    msg.transform.rotation.x, msg.transform.rotation.y = q[0], q[1]
    msg.transform.rotation.z, msg.transform.rotation.w = q[2], q[3]
    return msg


def _pixel(point: np.ndarray, k: tuple[float, float, float, float]) -> tuple[float, float]:
    fx, fy, cx, cy = k
    return fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy


def _depth_image(shape: tuple[int, int], k: tuple[float, float, float, float]) -> Any:
    """A 32FC1 frame: a flat slab at the TCP's depth, nothing (0 = invalid) elsewhere."""
    from sensor_msgs.msg import Image

    height, width = shape
    grid = np.zeros(shape, dtype=np.float32)
    u, v = _pixel(_TCP_IN_OPTICAL, k)
    half_px = k[0] * _SLAB_HALF_M / _TCP_IN_OPTICAL[2]
    rows = slice(max(0, int(v - half_px)), int(v + half_px) + 1)
    cols = slice(max(0, int(u - half_px)), int(u + half_px) + 1)
    grid[rows, cols] = _TCP_IN_OPTICAL[2]
    msg = Image()
    msg.header.frame_id = _OPTICAL
    msg.height, msg.width = height, width
    msg.encoding = "32FC1"
    msg.step = 4 * width
    msg.data = grid.tobytes()
    return msg


def _camera_info(shape: tuple[int, int], k: tuple[float, float, float, float]) -> Any:
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.header.frame_id = _OPTICAL
    msg.height, msg.width = shape
    fx, fy, cx, cy = k
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return msg


def test_grasp_masks_back_project_in_the_depth_headers_optical_frame() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    from openral_core import (
        Action,
        AttachmentEvidenceKind,
        ControlMode,
        JointState,
        RobotDescription,
    )
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.srv import SegmentInView
    from rcl_interfaces.msg import Log
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    left = next(j for j in description.joints if j.name == "left_gripper")
    camera = next(spec for spec in description.sensors if spec.name == _CAMERA)
    assert camera.frame_id == "zed_camera_link", "the manifest names the ZED body frame"
    tcp_in_link = np.asarray(left.origin_xyz, dtype=np.float64)
    gate_radius = 0.12  # VisionGateConfig.containment_radius_m default

    # ── The TF tree the Thor cell publishes. ────────────────────────────────
    mount = camera.static_transform_xyz_rpy
    assert mount is not None and camera.parent_frame == "openarm_base"
    t_base_body = _homogeneous(_rot_rpy(*mount[3:]), mount[:3])
    # The ZED driver's REP-103 hop: optical +z forward = body +x, +x right = body -y.
    t_body_opt = _homogeneous(_rot_rpy(-math.pi / 2, 0.0, -math.pi / 2), (0.0, 0.0, 0.0))
    # Put the attach link wherever makes the TCP land at _TCP_IN_OPTICAL, with a
    # generic orientation so no axis lines up by accident.
    r_opt_link = _rot_rpy(0.3, -0.2, 0.5)
    t_opt_link = _homogeneous(r_opt_link, _TCP_IN_OPTICAL - r_opt_link @ tcp_in_link)
    t_base_link = t_base_body @ t_body_opt @ t_opt_link
    attach_link = left.parent_link

    rclpy.init()
    node = Node("test_vision_optical_hal")
    peer = Node("test_vision_optical_peer")
    segmenter = Node("test_vision_optical_segmenter")
    requests: list[Any] = []
    # Seconds the mask's capture stamp sits from the (unstamped, t = 0) depth frames.
    mask_skew_s = {"value": 0.0}

    def segment(request: Any, response: Any) -> Any:
        """Mask a disc around the TCP pixel, at the RGB (full) resolution."""
        requests.append(request)
        point = np.array([request.tcp_point.x, request.tcp_point.y, request.tcp_point.z])
        u, v = _pixel(point, _THOR_K)
        rows, cols = np.mgrid[0 : _FULL[0], 0 : _FULL[1]]
        disc = (rows + 0.5 - v) ** 2 + (cols + 0.5 - u) ** 2 <= _MASK_RADIUS_PX**2
        mask = Image()
        mask.height, mask.width = _FULL
        mask.encoding = "mono8"
        mask.step = _FULL[1]
        mask.data = (disc.astype(np.uint8) * 255).tobytes()
        skew_ns = round(mask_skew_s["value"] * 1e9)
        mask.header.stamp.sec, mask.header.stamp.nanosec = divmod(skew_ns, 1_000_000_000)
        response.ok = True
        response.camera = request.camera
        response.masks = [mask]
        response.mask_scores_advisory = [0.9]
        return response

    segmenter.create_service(SegmentInView, _SERVICE, segment)
    logs: list[str] = []
    peer.create_subscription(Log, "/rosout", lambda msg: logs.append(str(msg.msg)), 100)
    # The HAL node spins single-threaded (`rclpy.spin`); a MultiThreadedExecutor
    # skips the busy node's camera_info while a 1080p depth callback holds its one
    # callback group, and starves it outright when depth is always ready.
    executor = SingleThreadedExecutor()
    for each in (node, peer, segmenter):
        executor.add_node(each)

    broadcaster = StaticTransformBroadcaster(peer)
    broadcaster.sendTransform(
        [
            _tf_msg("openarm_base", "zed_camera_link", t_base_body),
            _tf_msg("zed_camera_link", _OPTICAL, t_body_opt),
            _tf_msg("openarm_base", attach_link, t_base_link),
        ]
    )
    sensor_qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    bridges: list[Any] = []

    def grasp(tag: str, *, depth_shape: tuple[int, int], k: Any, publish_info: bool) -> Any:
        """One real bridge, one depth stream, one ATTACH; returns the leg's attachment."""
        depth_topic = f"/test_vision_optical/{tag}/depth"
        info_topic = f"/test_vision_optical/{tag}/camera_info"
        bridge = VisionAttachmentBridge(
            node,
            description,
            config=VisionAttachmentConfig(
                camera=_CAMERA,
                depth_topic=depth_topic,
                camera_info_topic=info_topic,
                service_name=_SERVICE,
                deadline_s=5.0,
            ),
        )
        bridges.append(bridge)
        bridge.setup()
        depth_pub = peer.create_publisher(Image, depth_topic, sensor_qos)
        info_pub = peer.create_publisher(CameraInfo, info_topic, sensor_qos)
        depth = _depth_image(depth_shape, k)
        info = _camera_info(depth_shape, k)

        def inputs_cached() -> bool:
            depth_pub.publish(depth)
            if publish_info:
                info_pub.publish(info)
            return bridge._depth is not None and (
                bridge._camera_info is not None or not publish_info
            )

        assert _wait_until(inputs_cached), f"{tag}: the bridge never cached its inputs"
        assert _wait_until(bridge._client.service_is_ready), "SegmentInView never discovered"
        # Each bridge owns a fresh tf2 listener; let it hear /tf_static first.
        assert _wait_until(lambda: bridge._lookup(attach_link, _OPTICAL) is not None)
        # Close both jaws; the left one stalls 0.2 rad short (an object), the right
        # rests closed on nothing. Effort is the real driver's zeros.
        bridge.observe_command(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.0] * len(description.joints)],
            )
        )
        for tick in range(12):
            bridge.observe_joint_state(
                JointState(
                    name=["left_gripper", "right_gripper"],
                    position=[0.2, -0.0116],
                    effort=[0.0, 0.0],
                    stamp_ns=1_000 + tick,
                )
            )
            if not bridge.attachment_action_ack_ready():
                break
        assert not bridge.attachment_action_ack_ready(), f"{tag}: the trigger never fired"
        assert _wait_until(bridge.attachment_action_ack_ready), f"{tag}: never settled"
        attachment = bridge._legs[0].attachment
        assert attachment is not None
        return bridge, attachment

    try:
        # ── 1. Full resolution, the driver's K: the payload is measured at the TCP.
        full, attached = grasp("full", depth_shape=_FULL, k=_THOR_K, publish_info=True)
        assert attached.evidence_kind is not AttachmentEvidenceKind.GRIPPER_CLOSURE, (
            f"fell back; log: {[line for line in logs if 'vision attachment' in line]}"
        )
        request = requests[-1]
        assert request.frame_id == "", "the prompt must be sent in the optical frame"
        expected_tcp = t_opt_link @ np.append(tcp_in_link, 1.0)
        sent = np.array([request.tcp_point.x, request.tcp_point.y, request.tcp_point.z])
        np.testing.assert_allclose(sent, expected_tcp[:3], atol=1e-6)
        centre_full = np.asarray(attached.pose_in_link.xyz)
        miss = float(np.linalg.norm(centre_full - tcp_in_link))
        assert miss <= gate_radius, f"centre {miss * 1000:.1f} mm from the TCP"
        assert any(
            f"depth header's frame '{_OPTICAL}'" in line and "'zed_camera_link'" in line
            for line in logs
        ), "the frame substitution was not logged"

        # The old frame choice: the same camera-frame centroid, carried into the
        # link through link <- zed_camera_link instead of link <- optical, lands
        # far outside the containment gate.
        t_link_opt = full._lookup(full.tf_frame(attach_link), _OPTICAL)
        t_link_body = full._lookup(full.tf_frame(attach_link), str(camera.frame_id))
        assert t_link_opt is not None and t_link_body is not None
        centroid_cam = np.linalg.inv(t_link_opt) @ np.append(centre_full, 1.0)
        old_centre = (t_link_body @ centroid_cam)[:3]
        assert np.linalg.norm(old_centre - tcp_in_link) > gate_radius, (
            "link <- zed_camera_link would have put the payload at the TCP too; "
            "the test geometry does not discriminate the frames"
        )

        # ── 2. No CameraInfo: the conservative box, with the reason named. ────
        _, fallback = grasp("noinfo", depth_shape=_FULL, k=_THOR_K, publish_info=False)
        assert fallback.evidence_kind is AttachmentEvidenceKind.GRIPPER_CLOSURE
        assert _wait_until(
            lambda: any(
                "fell back to GRIPPER_CLOSURE" in line
                and "ROSPerceptionStale: no CameraInfo yet on '/test_vision_optical/noinfo/" in line
                for line in logs
            ),
            timeout_s=3.0,
        ), "the missing-CameraInfo fallback did not log its typed reason"

        # ── 3. Half-resolution depth + K/2, full-resolution mask: resampled. ──
        half_k = tuple(value / 2.0 for value in _THOR_K)
        _, halved = grasp(
            "half", depth_shape=(_FULL[0] // 2, _FULL[1] // 2), k=half_k, publish_info=True
        )
        assert halved.evidence_kind is not AttachmentEvidenceKind.GRIPPER_CLOSURE, (
            f"fell back; log: {[line for line in logs if 'vision attachment' in line]}"
        )
        drift = float(np.linalg.norm(np.asarray(halved.pose_in_link.xyz) - centre_full))
        assert drift <= 0.020, f"half-res centroid {drift * 1000:.1f} mm from full-res"

        # ── 4. A mask captured 0.5 s from the depth frame: never back-projected. ──
        mask_skew_s["value"] = 0.5
        _, skewed = grasp("skewed", depth_shape=_FULL, k=_THOR_K, publish_info=True)
        assert skewed.evidence_kind is AttachmentEvidenceKind.GRIPPER_CLOSURE
        assert _wait_until(
            lambda: any(
                "fell back to GRIPPER_CLOSURE: ROSPerceptionStale: SegmentInView mask captured "
                "0.500 s from the depth frame" in line
                for line in logs
            ),
            timeout_s=3.0,
        ), "the mask/depth skew fallback did not log its typed reason"
    finally:
        for bridge in bridges:
            with suppress(Exception):
                bridge.teardown()
        executor.shutdown()
        spin.join(timeout=5.0)
        for each in (segmenter, peer, node):
            each.destroy_node()
        rclpy.shutdown()
