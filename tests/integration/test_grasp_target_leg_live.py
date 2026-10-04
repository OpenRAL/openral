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


def _voxels(centres: list[np.ndarray], stamp: Any, source: Any = None) -> Any:
    """A base-aligned 20 mm lattice: the table layer under the search box + solid boxes.

    ``source`` is the grid's ``source_stamp`` (the world data's capture time; default
    ``stamp``) — a stalled octomap keeps ``stamp`` fresh and ``source`` frozen.
    """
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
    msg.source_stamp = stamp if source is None else source
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

    from openral_core import Action, ControlMode, JointState, PlaceRegion, RobotDescription
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
    state: dict[str, Any] = {
        "scene": "one",
        "grid": True,
        "mask_skew_s": 0.0,
        "close": False,
        "source": None,
    }
    close_both = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_names=["left_gripper", "right_gripper"],
        joint_targets=[[0.0, 0.0]],
    )
    info = _camera_info(_FULL, _THOR_K)

    def feed() -> None:
        """Jaw-position evidence for the heartbeat; with ``close``, a left-hand stall."""
        if state["close"]:
            bridge.observe_command(close_both)
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[0.2 if state["close"] else 0.5, -0.0116],
                effort=[0.0, 0.0],
                stamp_ns=time.monotonic_ns(),
            )
        )

    def sensors() -> None:
        stamp = node.get_clock().now().to_msg()
        raster, centres = scene[state["scene"]]
        depth_pub.publish(_depth_msg(raster, stamp))
        info_pub.publish(info)
        if state["grid"]:
            voxel_pub.publish(_voxels(centres, stamp, state["source"]))

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
        assert _wait_until(lambda: any("grid_stale" in line for line in logs), timeout_s=3.0), (
            "the lost-view reason was not logged"
        )

        # ── 2b. The octomap stalls: the grid keeps arriving with a fresh header but
        #        its source_stamp frozen. The region is stamped no later than that
        #        world data, so it ages out under the freeze, never refreshed.
        state["grid"] = True
        declaration_pub.publish(_declaration(now_ns(), one))
        assert _wait_until(region_live, timeout_s=20.0)
        state["source"] = node.get_clock().now().to_msg()
        stalled_ns = state["source"].sec * 1_000_000_000 + state["source"].nanosec
        assert _wait_until(lambda: not region_live(), timeout_s=10.0), (
            f"a stalled octomap kept the region alive: "
            f"{[line for line in logs if 'grasp target' in line]}"
        )
        gone = latest()
        gone_ns = gone.header.stamp.sec * 1_000_000_000 + gone.header.stamp.nanosec
        assert gone_ns - stalled_ns <= (_FREEZE_S + 1.0) * 1e9, "held past the stalled data"
        assert _wait_until(
            lambda: any("grid_source_stale" in line for line in logs), timeout_s=3.0
        ), f"no stalled-source reason: {[line for line in logs if 'grasp target' in line]}"
        state["source"] = None

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

        # ── 6. Handover: the left hand at the region stalls closed — the latched
        #       region IS the payload, named as the declaration, no segmentation.
        state["mask_skew_s"] = 0.0
        declaration_pub.publish(_declaration(now_ns(), one))
        assert _wait_until(region_live, timeout_s=20.0)
        published = PlaceRegion.from_idl(latest().grasp_declaration.region)
        # The payload is the held (tight) fit; the kernel gets it cell-closed — grown
        # sideways and up, never down (``GraspTargetLeg.fill``).
        assert bridge._grasp_target is not None
        region = bridge._grasp_target.tracker.region
        assert region is not None
        # (a re-fit may land between the two reads: same static scene, millimetres apart)
        assert published.pose.xyz[:2] == pytest.approx(region.pose.xyz[:2], abs=1e-3)
        assert published.pose.xyz[2] - published.half_extents[2] == pytest.approx(
            region.pose.xyz[2] - region.half_extents[2], abs=1e-3
        )
        assert all(p > h for p, h in zip(published.half_extents, region.half_extents, strict=True))
        left = next(j for j in description.joints if j.name == "left_gripper")
        # link7 placed so the left TCP (the gripper joint origin) sits at the region centre.
        t_base_link7 = _homogeneous(
            np.eye(3), np.asarray(region.pose.xyz) - np.asarray(left.origin_xyz)
        )
        StaticTransformBroadcaster(peer).sendTransform(
            [_tf_msg(_BASE, left.parent_link, t_base_link7)]
        )
        assert _wait_until(lambda: bridge.jaw_point(left.child_link, _BASE) is not None)
        segments_before = len(prompts)
        state["close"] = True
        assert _wait_until(lambda: bool(latest().objects), timeout_s=5.0), (
            f"no attachment; log: {[line for line in logs if 'vision attachment' in line]}"
        )
        (held,) = latest().objects
        assert held.evidence_kind == "grasp_target_region"
        assert held.object_id == "cell:restock_box", "the declaration names it (no object_id)"
        assert held.attach_link == left.parent_link
        np.testing.assert_allclose(
            [
                held.pose_in_link.position.x,
                held.pose_in_link.position.y,
                held.pose_in_link.position.z,
            ],
            left.origin_xyz,
            atol=1e-6,
        )
        assert _wait_until(
            lambda: any("from the grasp-target region" in line for line in logs), timeout_s=3.0
        )
        time.sleep(0.5)
        assert len(prompts) <= segments_before + 2, "the attach re-segmented the target"
    finally:
        # Stop the executor first: a timer callback waiting on the bridge lock must not
        # run on entities the teardown destroyed.
        executor.shutdown()
        spin.join(timeout=5.0)
        with suppress(Exception):
            bridge.teardown()
        for each in (segmenter, peer, node):
            each.destroy_node()
        rclpy.shutdown()


def test_an_approaching_hand_arms_the_target_with_no_named_target() -> None:
    """Approach-armed target: the runner's goal-scope declaration names no target and no
    search box; the left hand's TCP 5 cm above the box arms ``approach:<left finger>:<n>``
    (left contact link only) and the region measured there contains the box and not the
    neighbour 20 cm away; the right hand, far from everything, arms nothing; lifting the
    hand retracts the region; moving it over the neighbour measures the neighbour. Two picks
    in one goal: the jaws stall on the neighbour (ATTACH, its region the payload) and close
    onto nothing (DETACH) — the region leaves the envelope at once; the release window times
    out with the hand still there and the hand does not re-arm on the payload it released;
    lifted and brought over the first box it arms ``:<n + 1>`` under the goal's stamp and
    measures it. The goal ending retracts it all. Same real stack as the test above, plus the
    hands' tf and the left jaw's trigger."""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    from openral_core import (
        Action,
        ControlMode,
        GraspDeclaration,
        JointState,
        PlaceRegion,
        RobotDescription,
    )
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
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    origin = {
        j.child_link: np.array(j.origin_xyz) for j in description.joints if j.role == "gripper"
    }
    t_base_body = _thor_mount()
    t_body_opt = _homogeneous(_rot_rpy(-math.pi / 2, 0.0, -math.pi / 2), (0.0, 0.0, 0.0))
    t_base_opt = t_base_body @ t_body_opt
    (one,) = _box_centres(t_base_opt)
    neighbour = one + np.array([0.0, 0.20, 0.0])
    boxes = [one, neighbour]
    raster = _depth(t_base_opt, boxes)

    rclpy.init()
    node = Node("test_grasp_approach_hal")
    peer = Node("test_grasp_approach_peer")
    segmenter = Node("test_grasp_approach_segmenter")
    service = _SERVICE + "_approach"

    def segment(request: Any, response: Any) -> Any:
        point = np.array([request.tcp_point.x, request.tcp_point.y, request.tcp_point.z])
        fx, fy, cx, cy = _THOR_K
        u, v = fx * point[0] / point[2] + cx, fy * point[1] / point[2] + cy
        rows, cols = np.mgrid[0 : _FULL[0], 0 : _FULL[1]]
        disc = (rows + 0.5 - v) ** 2 + (cols + 0.5 - u) ** 2 <= _MASK_RADIUS_PX**2
        mask = Image()
        mask.height, mask.width = _FULL
        mask.encoding = "mono8"
        mask.step = _FULL[1]
        mask.data = (disc.astype(np.uint8) * 255).tobytes()
        mask.header.stamp = request.stamp
        response.ok = True
        response.camera = request.camera
        response.masks = [mask]
        response.mask_scores_advisory = [0.9]
        return response

    segmenter.create_service(SegmentInView, service, segment)
    logs: list[str] = []
    peer.create_subscription(Log, "/rosout", lambda msg: logs.append(str(msg.msg)), 200)
    envelopes: list[Any] = []
    peer.create_subscription(AttachmentState, "/openral/attachment_state", envelopes.append, 50)
    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    declaration_pub = peer.create_publisher(
        GraspDeclarationMsg, "/openral/grasp_declaration", latched
    )
    voxel_pub = peer.create_publisher(
        OccupancyVoxels,
        "/openral/world_voxels",
        QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=1),
    )
    depth_topic, info_topic = "/test_grasp_approach/depth", "/test_grasp_approach/camera_info"
    sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=1)
    depth_pub = peer.create_publisher(Image, depth_topic, sensor_qos)
    info_pub = peer.create_publisher(CameraInfo, info_topic, sensor_qos)
    StaticTransformBroadcaster(peer).sendTransform(
        [
            _tf_msg(_BASE, "zed_camera_link", t_base_body),
            _tf_msg("zed_camera_link", _OPTICAL, t_body_opt),
        ]
    )
    hands_tf = TransformBroadcaster(peer)
    # Where each hand's TCP is (base frame); the ee frame = TCP - the jaw joint's origin.
    tcp = {
        "left": one + np.array([0.0, 0.0, _BOX_HALF[2] + 0.05]),
        "right": np.array([0.30, -0.25, 0.20]),  # high above the empty table edge
    }

    bridge = VisionAttachmentBridge(
        node,
        description,
        config=VisionAttachmentConfig(
            camera="head_zed",
            depth_topic=depth_topic,
            camera_info_topic=info_topic,
            service_name=service,
            deadline_s=2.0,
            grasp_target_enabled=True,
            grasp_target_freeze_s=_FREEZE_S,
            grasp_target_approach_m=0.10,
            release_timeout_s=1.0,
            tf_frames={
                "openarm_left_link7": "openarm_left_ee_base_link",
                "openarm_right_link7": "openarm_right_ee_base_link",
            },
        ),
    )
    bridge.setup()
    info = _camera_info(_FULL, _THOR_K)
    # The left jaw: open and uncommanded until ``close`` (then commanded shut, read at
    # ``left_q``: 0.2 stalls on an object, 0.0 closes onto nothing).
    jaw: dict[str, Any] = {"close": False, "left_q": 0.5}
    close_left = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_names=["left_gripper"],
        joint_targets=[[0.0]],
    )

    def feed() -> None:
        if jaw["close"]:
            bridge.observe_command(close_left)
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[jaw["left_q"], 0.0],
                effort=[0.01, 0.01],
                stamp_ns=time.monotonic_ns(),
            )
        )

    def sensors() -> None:
        stamp = node.get_clock().now().to_msg()
        depth_pub.publish(_depth_msg(raster, stamp))
        info_pub.publish(info)
        voxel_pub.publish(_voxels(boxes, stamp))
        frames = []
        for side, point in tcp.items():
            t = np.eye(4)
            t[:3, 3] = point - origin[f"openarm_{side}_finger_pair"]
            msg = _tf_msg(_BASE, f"openarm_{side}_ee_base_link", t)
            msg.header.stamp = stamp
            frames.append(msg)
        hands_tf.sendTransform(frames)

    node.create_timer(0.05, feed)
    peer.create_timer(0.1, sensors)
    executor = MultiThreadedExecutor()
    for each in (node, peer, segmenter):
        executor.add_node(each)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    def latest() -> Any:
        return envelopes[-1] if envelopes else None

    def region_around(centre: np.ndarray) -> bool:
        msg = latest()
        if msg is None or not (msg.grasp_declaration_valid and msg.grasp_declaration.region_valid):
            return False
        region = PlaceRegion.from_idl(msg.grasp_declaration.region)
        return bool(np.all(np.abs(np.asarray(region.pose.xyz)[:2] - centre[:2]) < _RES))

    stamp_ns = int(node.get_clock().now().nanoseconds)
    goal = GraspDeclaration(
        target_id="approach",
        contact_links=("openarm_left_finger_pair", "openarm_right_finger_pair"),
        rskill_id="openral/itest-approach",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        timeout_s=60.0,
        stamp_ns=stamp_ns,
    )

    def publish(declaration: GraspDeclaration) -> None:
        msg = GraspDeclarationMsg()
        declaration.fill_idl(msg)
        declaration_pub.publish(msg)

    try:
        # ── 1. The left hand over the box: armed for the left hand, the box measured. ──
        publish(goal)
        assert _wait_until(lambda: region_around(one), timeout_s=20.0), (
            f"no region; log: {[line for line in logs if 'grasp target' in line]}"
        )
        declared = latest().grasp_declaration
        first_id = declared.target_id
        assert first_id.startswith("approach:openarm_left_finger_pair:")
        assert list(declared.contact_links) == ["openarm_left_finger_pair"]
        assert declared.rskill_id == "openral/itest-approach"
        assert declared.stamp_ns == stamp_ns
        region = PlaceRegion.from_idl(declared.region)
        assert region.pose.xyz[1] + region.half_extents[1] < neighbour[1] - _BOX_HALF[1], (
            "the region reaches the neighbour"
        )
        bottom = region.pose.xyz[2] - region.half_extents[2]
        assert _SUPPORT_Z < bottom <= _SUPPORT_Z + _RES + 1e-6, f"lower face at {bottom:.3f}"
        assert not any("openarm_right_finger_pair" in line and "armed" in line for line in logs)

        # ── 2. The hand lifts clear: retracted at once, back to the goal-scope one. ──
        tcp["left"] = tcp["left"] + np.array([0.0, 0.0, 0.30])
        assert _wait_until(
            lambda: (
                latest().grasp_declaration.target_id == "approach"
                and not latest().grasp_declaration.region_valid
            ),
            timeout_s=5.0,
        ), f"not retracted: {[line for line in logs if 'grasp target' in line]}"
        assert _wait_until(
            lambda: any("retracted — approach_ended" in line for line in logs), timeout_s=3.0
        )

        # ── 2b. The right hand low over the empty table: the table is not a target.
        #        Its surface layer is not counted, so nothing arms, refuses, re-arms.
        tcp["right"] = np.array([one[0], -0.25, _SUPPORT_Z + 0.05])
        mark = len(logs)
        time.sleep(1.5)
        assert latest().grasp_declaration.target_id == "approach"
        assert not any("grasp target" in line for line in logs[mark:]), (
            f"the empty table armed a hand: {logs[mark:]}"
        )
        tcp["right"] = np.array([0.30, -0.25, 0.20])

        # ── 3. The hand goes to the neighbour instead: the policy picked it. ──
        tcp["left"] = neighbour + np.array([0.0, 0.0, _BOX_HALF[2] + 0.05])
        assert _wait_until(lambda: region_around(neighbour), timeout_s=20.0), (
            f"no neighbour region: {[line for line in logs if 'grasp target' in line]}"
        )
        # Nothing was handed over: the same pick, so the same identity.
        assert latest().grasp_declaration.target_id == first_id

        # ── 3b. Pick 1: the TCP into the neighbour, the jaws stall on it (ATTACH). ──
        tcp["left"] = neighbour.copy()
        time.sleep(0.5)
        jaw["close"], jaw["left_q"] = True, 0.2
        assert _wait_until(lambda: bool(latest().objects), timeout_s=5.0), (
            f"no attachment; log: {[line for line in logs if 'attachment' in line]}"
        )
        assert latest().objects[0].evidence_kind == "grasp_target_region"
        assert _wait_until(lambda: any("region kept" in line for line in logs), timeout_s=3.0), (
            "not handed over"
        )

        # ── 3c. DETACH: the region leaves the envelope at once, the window opens. A
        #        process stall longer than the window (seen under load: ~1.7 s with no
        #        callback) closes it before any heartbeat shows it; then the next one
        #        already shows the completed pick, never the region. ──
        jaw["left_q"] = 0.0
        assert _wait_until(
            lambda: (
                latest().grasp_declaration.target_id in (first_id, "approach")
                and not latest().grasp_declaration.region_valid
            ),
            timeout_s=5.0,
        ), f"region not dropped at DETACH: {[line for line in logs if 'grasp' in line]}"
        assert _wait_until(
            lambda: any("released its payload — region dropped" in line for line in logs),
            timeout_s=3.0,
        )
        assert _wait_until(
            lambda: any("release window opened" in line for line in logs), timeout_s=3.0
        )

        # ── 3d. The window times out, the hand still by the payload: pick complete, and
        #        the hand does not re-arm on what it just released. ──
        assert _wait_until(
            lambda: latest().grasp_declaration.target_id == "approach", timeout_s=5.0
        ), f"pick never completed: {[line for line in logs if 'grasp' in line]}"
        assert _wait_until(
            lambda: any("holds nothing; may re-arm" in line for line in logs), timeout_s=3.0
        )
        time.sleep(1.5)
        assert latest().grasp_declaration.target_id == "approach", (
            "the hand re-armed on its just-released payload"
        )

        # ── 3e. Lifted clear, then over the first box: pick 2, a fresh identity. The
        #        first box is within a voxel of the released payload's (inflated) record,
        #        so only a tick that samples the hand lifted ends the guard (HZ-0115-11):
        #        hold the lift until one did — a stall can swallow a fixed sleep. ──
        tcp["left"] = neighbour + np.array([0.0, 0.0, 0.40])
        assert _wait_until(
            lambda: any("cleared the payload it released" in line for line in logs),
            timeout_s=10.0,
        ), f"the lift never ended the guard: {[line for line in logs if 'grasp' in line]}"
        tcp["left"] = one + np.array([0.0, 0.0, _BOX_HALF[2] + 0.05])
        assert _wait_until(lambda: region_around(one), timeout_s=20.0), (
            f"no pick-2 region: {[line for line in logs if 'grasp target' in line]}"
        )
        second = latest().grasp_declaration
        n_first = int(first_id.rsplit(":", 1)[1])
        assert second.target_id == f"approach:openarm_left_finger_pair:{n_first + 1}"
        assert second.stamp_ns == stamp_ns, "the goal's stamp: its timeout backstop"

        # ── 4. The goal ends: gone from the next heartbeat. ──
        publish(goal.model_copy(update={"active": False}))
        assert _wait_until(lambda: not latest().grasp_declaration_valid, timeout_s=1.0)
    finally:
        # Stop the executor first: a timer callback waiting on the bridge lock must not
        # run on entities the teardown destroyed.
        executor.shutdown()
        spin.join(timeout=5.0)
        with suppress(Exception):
            bridge.teardown()
        for each in (segmenter, peer, node):
            each.destroy_node()
        rclpy.shutdown()
