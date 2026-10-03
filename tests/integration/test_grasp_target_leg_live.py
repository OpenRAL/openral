# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the pre-grasp target producer leg measures the declared target's region.

Real rclpy, real ``VisionAttachmentBridge`` on the bimanual OpenArm manifest with
``grasp_target_enabled``, real tf2 with the Thor cell's measured ZED mount
(``robots/openarm/units/thor.yaml``) and the driver's REP-103 optical hop, real
``openral_msgs`` on the wire: a ``GraspDeclaration`` with a search box on
``/openral/grasp_declaration``, an ``OccupancyVoxels`` grid on
``/openral/world_voxels`` holding a box on a support plane, a depth ``Image`` +
``CameraInfo`` (Thor K, 1920x1080) ray-cast from that same box, and a
``SegmentInView`` peer that masks a disc around the pixel it is prompted at — a
real DDS service at a process boundary standing in for SAM 2.1 (CLAUDE.md §1.11),
the same pattern as ``test_vision_attachment_optical_frame_live.py``.

Asserts, on ``/openral/attachment_state``: a valid region that contains the box
and whose lower face sits above the *measured* table top within one voxel, though
the search box's bottom reaches below it; the grid
going silent retracts it once the freeze TTL has run from the region's own
stamp, with the reason logged; masks captured 0.5 s off the depth frame are
refused as ``mask_depth_skew``; two equal boxes in the search box are refused as
``ambiguous`` and no region is ever published; dispatch retracting the
declaration removes it from the next heartbeat.

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    OPENRAL_TEST_ROS_LIVE=1 pytest tests/integration/test_grasp_target_leg_live.py
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
import yaml

from tests.integration.test_vision_attachment_optical_frame_live import (
    _camera_info,
    _homogeneous,
    _rot_rpy,
    _tf_msg,
    _wait_until,
)

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first.",
)

_REPO = Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
_THOR_YAML = _REPO / "robots" / "openarm" / "units" / "thor.yaml"
_BASE = "openarm_base"
_OPTICAL = "zed_left_camera_frame_optical"
# The Thor ZED Mini's depth camera_info, measured 2026-10-02.
_THOR_K = (1498.18, 1498.18, 936.11, 541.81)
_FULL = (1080, 1920)
_SERVICE = "/openral/perception/segment_in_view_grasp_target_itest"
_RES = 0.02
_SUPPORT_Z = -0.40  # table top, base frame
_BOX_HALF = np.array([0.04, 0.04, 0.04])  # the target: an 8 cm cube, faces on cell faces
_MASK_RADIUS_PX = 400
_FREEZE_S = 2.0


def _thor_mount() -> np.ndarray:
    unit = yaml.safe_load(_THOR_YAML.read_text())
    xyzrpy = next(s for s in unit["sensors"] if s["name"] == "head_zed")["static_transform_xyz_rpy"]
    return _homogeneous(_rot_rpy(*xyzrpy[3:]), xyzrpy[:3])


def _box_centres(t_base_opt: np.ndarray) -> list[np.ndarray]:
    """Where the optical axis meets the table, snapped so the box faces are cell faces."""
    origin, axis = t_base_opt[:3, 3], t_base_opt[:3, 2]
    hit = origin + axis * (_SUPPORT_Z - origin[2]) / axis[2]
    x = round(hit[0] / _RES) * _RES
    return [np.array([x, 0.0, _SUPPORT_Z + _BOX_HALF[2]])]


def _voxels(centres: list[np.ndarray], stamp: Any) -> Any:
    """A base-aligned 20 mm lattice: the table layer under the search box + solid boxes."""
    from openral_msgs.msg import OccupancyVoxels

    origin = np.array([0.0, -0.30, _SUPPORT_Z - _RES])  # layer k=0 is the table surface
    size = (30, 30, 10)
    occ = np.zeros(size[::-1], dtype=np.uint8)  # [k, j, i] so ravel() is x-fastest
    occ[0, :, :] = 1
    for centre in centres:
        lo = np.round((centre - _BOX_HALF - origin) / _RES).astype(int)
        hi = np.round((centre + _BOX_HALF - origin) / _RES).astype(int)
        occ[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] = 1
    msg = OccupancyVoxels()
    msg.header.frame_id = _BASE
    msg.header.stamp = stamp
    msg.origin.x, msg.origin.y, msg.origin.z = (float(v) for v in origin)
    msg.orientation.w = 1.0
    msg.resolution = _RES
    msg.size_x, msg.size_y, msg.size_z = size
    msg.occupancy = occ.ravel().tolist()
    return msg


def _depth(t_base_opt: np.ndarray, centres: list[np.ndarray]) -> np.ndarray:
    """Ray-cast the boxes (only) into a 32FC1 z-depth raster; 0 = no return."""
    fx, fy, cx, cy = _THOR_K
    rows, cols = np.mgrid[0 : _FULL[0], 0 : _FULL[1]]
    rays_cam = np.stack(
        ((cols + 0.5 - cx) / fx, (rows + 0.5 - cy) / fy, np.ones(_FULL)), axis=-1
    ).reshape(-1, 3)
    rays = rays_cam @ t_base_opt[:3, :3].T  # parameter t along these is the optical z
    origin = t_base_opt[:3, 3]
    depth = np.full(rays.shape[0], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for centre in centres:
            lo, hi = centre - _BOX_HALF, centre + _BOX_HALF
            t1, t2 = (lo - origin) / rays, (hi - origin) / rays
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (near <= far) & (near > 0)
            depth[hit] = np.minimum(depth[hit], near[hit])
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(_FULL).astype(np.float32)


def _depth_msg(grid: np.ndarray, stamp: Any) -> Any:
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.frame_id = _OPTICAL
    msg.header.stamp = stamp
    msg.height, msg.width = _FULL
    msg.encoding = "32FC1"
    msg.step = 4 * _FULL[1]
    msg.data = grid.tobytes()
    return msg


def _declaration(stamp_ns: int, centre: np.ndarray, *, active: bool = True) -> Any:
    from openral_core import GraspDeclaration, PlaceRegion, Pose6D
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg

    # A lifted detection bbox: its bottom face 3 cm *below* the table top. The
    # producer must measure the table, not stand the region on this face.
    search_box = PlaceRegion(
        frame_id=_BASE,
        pose=Pose6D(
            xyz=(float(centre[0]), float(centre[1]), _SUPPORT_Z + 0.12),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=_BASE,
        ),
        half_extents=(0.15, 0.20, 0.15),
    )
    declaration = GraspDeclaration(
        target_id="cell:restock_box",
        contact_links=("openarm_left_finger_pair", "openarm_right_finger_pair"),
        rskill_id="openral/itest-grasp-target",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=60.0,
        stamp_ns=stamp_ns,
        active=active,
        search_box=search_box,
    )
    msg = GraspDeclarationMsg()
    declaration.fill_idl(msg)
    return msg


def test_grasp_target_leg_measures_freezes_refuses_and_retracts() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    from openral_core import JointState, PlaceRegion, RobotDescription
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import AttachmentState, OccupancyVoxels
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.srv import SegmentInView
    from rcl_interfaces.msg import Log
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    t_base_body = _thor_mount()
    t_body_opt = _homogeneous(_rot_rpy(-math.pi / 2, 0.0, -math.pi / 2), (0.0, 0.0, 0.0))
    t_base_opt = t_base_body @ t_body_opt
    t_opt_base = np.linalg.inv(t_base_opt)
    (one,) = _box_centres(t_base_opt)
    two = [one + np.array([0.0, 0.08, 0.0]), one - np.array([0.0, 0.08, 0.0])]
    scene = {"one": (_depth(t_base_opt, [one]), [one]), "two": (_depth(t_base_opt, two), two)}
    assert (scene["one"][0] > 0).sum() > 10_000, "the box must be in view"

    rclpy.init()
    node = Node("test_grasp_target_hal")
    peer = Node("test_grasp_target_peer")
    segmenter = Node("test_grasp_target_segmenter")
    prompts: list[np.ndarray] = []

    def segment(request: Any, response: Any) -> Any:
        """Mask a disc around the prompt's pixel; the prompt arrives in the optical frame."""
        assert request.frame_id == "" and list(request.negative_points) == []
        point = np.array([request.tcp_point.x, request.tcp_point.y, request.tcp_point.z])
        prompts.append(point)
        fx, fy, cx, cy = _THOR_K
        u, v = fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy
        rows, cols = np.mgrid[0 : _FULL[0], 0 : _FULL[1]]
        disc = (rows + 0.5 - v) ** 2 + (cols + 0.5 - u) ** 2 <= _MASK_RADIUS_PX**2
        mask = Image()
        mask.height, mask.width = _FULL
        mask.encoding = "mono8"
        mask.step = _FULL[1]
        mask.data = (disc.astype(np.uint8) * 255).tobytes()
        # The real segmenter stamps a mask with its RGB frame's capture stamp; here
        # the RGB frame is the depth frame asked about, unless a skew is injected.
        skew_ns = round(state["mask_skew_s"] * 1e9)
        stamp_ns = request.stamp.sec * 1_000_000_000 + request.stamp.nanosec + skew_ns
        mask.header.stamp.sec, mask.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        response.ok = True
        response.camera = request.camera
        response.masks = [mask]
        response.mask_scores_advisory = [0.9]
        return response

    segmenter.create_service(SegmentInView, _SERVICE, segment)
    logs: list[str] = []
    peer.create_subscription(Log, "/rosout", lambda msg: logs.append(str(msg.msg)), 200)
    envelopes: list[Any] = []
    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    peer.create_subscription(AttachmentState, "/openral/attachment_state", envelopes.append, 50)
    declaration_pub = peer.create_publisher(
        GraspDeclarationMsg, "/openral/grasp_declaration", latched
    )
    voxel_pub = peer.create_publisher(
        OccupancyVoxels,
        "/openral/world_voxels",
        QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=1),
    )
    depth_topic, info_topic = "/test_grasp_target/depth", "/test_grasp_target/camera_info"
    sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=1)
    depth_pub = peer.create_publisher(Image, depth_topic, sensor_qos)
    info_pub = peer.create_publisher(CameraInfo, info_topic, sensor_qos)
    StaticTransformBroadcaster(peer).sendTransform(
        [
            _tf_msg(_BASE, "zed_camera_link", t_base_body),
            _tf_msg("zed_camera_link", _OPTICAL, t_body_opt),
        ]
    )

    bridge = VisionAttachmentBridge(
        node,
        description,
        config=VisionAttachmentConfig(
            camera="head_zed",
            depth_topic=depth_topic,
            camera_info_topic=info_topic,
            service_name=_SERVICE,
            deadline_s=2.0,
            grasp_target_enabled=True,
            grasp_target_freeze_s=_FREEZE_S,
        ),
    )
    bridge.setup()
    state: dict[str, Any] = {"scene": "one", "grid": True, "mask_skew_s": 0.0}
    info = _camera_info(_FULL, _THOR_K)

    def feed() -> None:
        """Effort evidence for the heartbeat (no grasp), plus the sensor streams."""
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[0.0, 0.0],
                effort=[0.01, 0.01],
                stamp_ns=time.monotonic_ns(),
            )
        )

    def sensors() -> None:
        stamp = node.get_clock().now().to_msg()
        raster, centres = scene[state["scene"]]
        depth_pub.publish(_depth_msg(raster, stamp))
        info_pub.publish(info)
        if state["grid"]:
            voxel_pub.publish(_voxels(centres, stamp))

    node.create_timer(0.05, feed)
    peer.create_timer(0.2, sensors)
    executor = MultiThreadedExecutor()
    for each in (node, peer, segmenter):
        executor.add_node(each)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    def now_ns() -> int:
        return int(node.get_clock().now().nanoseconds)

    def latest() -> Any:
        return envelopes[-1] if envelopes else None

    def region_live() -> bool:
        msg = latest()
        return (
            msg is not None and msg.grasp_declaration_valid and msg.grasp_declaration.region_valid
        )

    try:
        # ── 1. One box: a measured region that contains it, above the table. ──
        declaration_pub.publish(_declaration(now_ns(), one))
        assert _wait_until(region_live, timeout_s=20.0), (
            f"no region; log: {[line for line in logs if 'grasp target' in line]}"
        )
        envelope = latest()
        declared = envelope.grasp_declaration
        assert declared.target_id == "cell:restock_box"
        assert declared.rskill_id == "openral/itest-grasp-target"
        assert declared.search_box_valid
        region = PlaceRegion.from_idl(declared.region)
        assert region.frame_id == _BASE
        assert region.evidence_ref.startswith("segment_in_view:openral/itest-grasp-target@")
        assert 0 < region.stamp_ns <= now_ns()
        bottom = region.pose.xyz[2] - region.half_extents[2]
        assert _SUPPORT_Z < bottom <= _SUPPORT_Z + _RES + 1e-6, f"lower face at {bottom:.3f}"
        corners = np.array(
            [one + _BOX_HALF * np.array([sx, sy, 1.0]) for sx in (-1, 1) for sy in (-1, 1)]
            + [one + np.array([0.0, 0.0, -_BOX_HALF[2] + _RES + 0.005])]
        )
        rot = homogeneous_from_quat_xyz((0.0, 0.0, 0.0), region.pose.quat_xyzw)[:3, :3]
        local = np.abs((corners - np.asarray(region.pose.xyz)) @ rot)
        assert np.all(local <= np.asarray(region.half_extents) + 1e-6), (
            f"box not contained: {local} vs {region.half_extents}"
        )
        # The prompt was the cluster's top-centre, sent in the optical frame.
        top_in_opt = t_opt_base @ np.append(one + np.array([0.0, 0.0, _BOX_HALF[2]]), 1.0)
        assert np.linalg.norm(prompts[-1] - top_in_opt[:3]) < _RES

        # ── 2. The grid goes silent: frozen, then retracted after the TTL. ──
        state["grid"] = False
        assert _wait_until(lambda: not region_live(), timeout_s=10.0), "region outlived its TTL"
        last_held = next(
            msg for msg in reversed(envelopes) if msg.grasp_declaration.region_valid
        ).grasp_declaration.region.stamp_ns
        first_gone = latest()
        gone_ns = first_gone.header.stamp.sec * 1_000_000_000 + first_gone.header.stamp.nanosec
        assert gone_ns - last_held >= _FREEZE_S * 1e9 - 0.25e9, "retracted before the TTL"
        assert first_gone.grasp_declaration_valid, "the declaration itself is still live"
        assert _wait_until(
            lambda: any("retracted — freeze_ttl" in line for line in logs), timeout_s=3.0
        ), f"no TTL reason logged: {[line for line in logs if 'grasp target' in line]}"
        assert any("grid_stale" in line for line in logs), "the lost-view reason was not logged"

        # ── 3. Two equal boxes: ambiguous, no region ever. ───────────────────
        # The detection box centres on one of them (the leg anchors the target on
        # the column nearest the box centre); the other, as big, stands beside it.
        state.update(scene="two", grid=True)
        mark = len(envelopes)
        declaration_pub.publish(_declaration(now_ns(), two[0]))
        assert _wait_until(
            lambda: any("refused — ambiguous" in line for line in logs), timeout_s=10.0
        ), f"no ambiguous refusal: {[line for line in logs if 'grasp target' in line]}"
        time.sleep(1.0)
        assert not any(msg.grasp_declaration.region_valid for msg in envelopes[mark:])

        # ── 4. Dispatch retracts: gone from the next heartbeat. ──────────────
        state["scene"] = "one"
        stamp = now_ns()
        declaration_pub.publish(_declaration(stamp, one))
        assert _wait_until(region_live, timeout_s=20.0)
        declaration_pub.publish(_declaration(stamp, one, active=False))
        assert _wait_until(lambda: not latest().grasp_declaration_valid, timeout_s=1.0), (
            "dispatch retraction did not reach the heartbeat"
        )

        # ── 5. Masks captured 0.5 s off the depth frame: refused, no region. ─
        state["mask_skew_s"] = 0.5
        mark = len(envelopes)
        declaration_pub.publish(_declaration(now_ns(), one))
        assert _wait_until(
            lambda: any("refused — mask_depth_skew" in line for line in logs), timeout_s=10.0
        ), f"no skew refusal: {[line for line in logs if 'grasp target' in line]}"
        time.sleep(1.0)
        assert not any(msg.grasp_declaration.region_valid for msg in envelopes[mark:])
    finally:
        with suppress(Exception):
            bridge.teardown()
        executor.shutdown()
        spin.join(timeout=5.0)
        for each in (segmenter, peer, node):
            each.destroy_node()
        rclpy.shutdown()
