# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: a released payload stays a checked, frozen attached record until the jaws are clear.

Design note ``docs/reference/real-pick-place-design.md`` §1.6 / §2.3 "Release". Gripper-effort
DETACH fires when the jaws *open*, while the fingers still surround the released object. The
vision leg used to publish the empty set at once; the octomap bridge then stopped clearing the
object's cells, re-marked them within ~0.2 s, and the fingers — still inside the 20 mm world
margin of those cells — were stopped on the first retreat chunk (``KIND_COLLISION`` on
``*_finger_pair``).

Two tests, both real components (CLAUDE.md §1.11):

``test_a_released_payload_stays_frozen_until_the_jaws_are_clear`` — real rclpy, real
``VisionAttachmentBridge`` on the bimanual OpenArm manifest with the Thor cell's ``tf_frames``
(``openarm_<side>_link7`` -> ``openarm_<side>_ee_base_link``), real tf2, a real depth frame and
an offered-but-unanswered ``SegmentInView`` (the attach deadline settles every grasp on the
fallback box). ATTACH -> DETACH: the next snapshots still carry the payload, now attached to
``openarm_base`` at its DETACH pose, and that pose and the revision stay put while the hand's TF
moves 10 cm; once the hand is clear the set goes empty at a fresh revision. A new ATTACH during a
window replaces the frozen record with the new grasp. Every snapshot goes through a real
``WorldStateAggregator``.

``test_the_kernel_accepts_the_retreat_only_while_the_frozen_record_is_published`` — the real
``safety_kernel_node`` with the real-cell parameters (manifest collision model, 20 mm world
margin on 20 mm cells, attached check on at margin 0 as the vision leg turns it on). There is no
octomap bridge here, so the grid is published the way payload clearing would leave it — an
explicit assumption: WITHOUT the payload's cells while the frozen record is on
``/openral/world_state_fast`` (``payload_clearing.hpp`` clears any attached object present on
that message) and WITH them once it is gone. Rows, all at the release configuration (the first
waypoint of a retreat):

==============================================================  =============================
row                                                             verdict
==============================================================  =============================
frozen record published, payload cells cleared                  ACCEPTED
frozen record published, a shelf cell under the payload         REFUSED, ``attach:<id>``
record dropped, payload cells re-marked                         REFUSED, ``*_finger_pair``
==============================================================  =============================

The first row is what the window buys; the second shows a frozen payload is still checked
against the world (nothing is exempted); the third is the defect the window closes.

Gated on ``OPENRAL_TEST_ROS_LIVE=1`` (the kernel test also on a colcon-built
``openral_safety_kernel``). ``scripts/ros_live_tests.sh`` is the runner.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from typing import Any

import numpy as np
import pytest
from openral_core import CameraTopicKind, camera_topic

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy node + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 "
    "and source install/setup.bash first.",
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_ROBOT_YAML = _REPO / "robots" / "openarm" / "robot.yaml"
_BASE = "openarm_base"
_HAND = "openarm_left_link7"
_FINGER = "openarm_left_finger_pair"
#: The Thor cell's ``tf_frames`` (scenes/deploy/openarm_real_world_voxels.yaml).
_TF_FRAMES = {
    "openarm_left_link7": "openarm_left_ee_base_link",
    "openarm_right_link7": "openarm_right_ee_base_link",
}
_CAMERA = "head_zed"
_SERVICE = "/openral/perception/segment_in_view_release_itest"
_ATTACH_DEADLINE_S = 0.5
_PERIOD_S = 0.2
_FEED_PERIOD_S = 1.0 / 30.0


def _wait_until(predicate: Any, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _hand_at_zero(description: Any) -> np.ndarray:
    """``openarm_base <- openarm_left_link7`` at q = 0 (the manifest chain's origins composed).

    The same FK the kernel runs at q = 0 (``origin · joint motion``, motion = identity).
    """
    from openral_hal.vision_attachment_bridge import _xyz_rpy_matrix

    edges = {e.child_link: e for e in [*description.joints, *description.fixed_attachments]}
    t, link = np.eye(4), _HAND
    while link in edges:
        edge = edges[link]
        t = _xyz_rpy_matrix(edge.origin_xyz, edge.origin_rpy) @ t
        link = edge.parent_link
    assert link == _BASE
    return t


def test_a_released_payload_stays_frozen_until_the_jaws_are_clear() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")

    from geometry_msgs.msg import TransformStamped
    from openral_core import AttachedCollisionObject, JointState, RobotDescription
    from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
    from openral_hal.vision_attachment_bridge import (
        VisionAttachmentBridge,
        VisionAttachmentConfig,
    )
    from openral_msgs.msg import AttachmentState
    from openral_msgs.srv import SegmentInView
    from openral_world_state import WorldStateAggregator
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    left = next(j for j in description.joints if j.name == "left_gripper")
    camera = next(spec for spec in description.sensors if spec.name == _CAMERA)
    depth_topic = camera_topic(_CAMERA, CameraTopicKind.DEPTH_IMAGE)
    config = VisionAttachmentConfig(
        camera=_CAMERA,
        depth_topic=depth_topic,
        service_name=_SERVICE,
        deadline_s=_ATTACH_DEADLINE_S,
        tf_frames=_TF_FRAMES,
        # Long enough that only separation or a new ATTACH can end a window here; the
        # timeout is pinned by the unit tier.
        release_timeout_s=60.0,
    )
    t_hand = _hand_at_zero(description)

    rclpy.init()
    node = Node("test_release_window_hal")
    peer = Node("test_release_window_peer")
    unanswered = Node("test_release_window_segmenter")
    unanswered.create_service(SegmentInView, _SERVICE, lambda _req, resp: resp)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)

    aggregator = WorldStateAggregator(description)
    aggregator_errors: list[str] = []
    # (receipt monotonic s, revision, decoded objects)
    received: list[tuple[float, int, list[AttachedCollisionObject]]] = []
    feed: dict[str, Any] = {"on": False, "left": 0.01, "bridge": None}

    def on_state(msg: Any) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        objects = [AttachedCollisionObject.from_idl(item) for item in msg.objects]
        received.append((time.monotonic(), int(msg.revision), objects))
        try:
            aggregator.update_attached_objects(
                objects, revision=int(msg.revision), stamp_ns=stamp_ns
            )
        except ValueError as exc:
            aggregator_errors.append(str(exc))

    peer.create_subscription(
        AttachmentState,
        "/openral/attachment_state",
        on_state,
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        ),
    )

    def on_feed() -> None:
        bridge = feed["bridge"]
        if not feed["on"] or bridge is None:
            return
        bridge.observe_joint_state(
            JointState(
                name=["left_gripper", "right_gripper"],
                position=[0.0, 0.0],
                effort=[feed["left"], 0.01],
                stamp_ns=time.time_ns(),
            )
        )

    node.create_timer(_FEED_PERIOD_S, on_feed)

    hand_frame = _TF_FRAMES[_HAND]
    # The camera is rigid on the hand (static); the hand moves, so it is on /tf at 20 Hz the
    # way robot_state_publisher streams it, lifted by ``hand["dz"]``.
    cam = TransformStamped()
    cam.header.frame_id = hand_frame
    cam.child_frame_id = str(camera.frame_id)
    cam.transform.translation.z = 0.3
    cam.transform.rotation.w = 1.0
    StaticTransformBroadcaster(peer).sendTransform([cam])
    hand_tf = TransformBroadcaster(peer)
    hand: dict[str, float] = {"dz": 0.0}

    def publish_hand() -> None:
        tf = TransformStamped()
        tf.header.stamp = peer.get_clock().now().to_msg()
        tf.header.frame_id = _BASE
        tf.child_frame_id = hand_frame
        x, y, z = t_hand[:3, 3]
        tf.transform.translation.x, tf.transform.translation.y = float(x), float(y)
        tf.transform.translation.z = float(z) + hand["dz"]
        w, qx, qy, qz = rotation_to_quat_wxyz(t_hand[:3, :3])
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, w
        hand_tf.sendTransform(tf)

    peer.create_timer(0.05, publish_hand)

    def place_hand(dz: float) -> None:
        """Lift the hand ``dz`` above its release pose; settle one TF period."""
        hand["dz"] = dz
        time.sleep(0.1)

    def since(index: int) -> list[tuple[float, int, list[AttachedCollisionObject]]]:
        return received[index:]

    def held_on(link: str, start: int) -> tuple[float, int, list[AttachedCollisionObject]] | None:
        return next((r for r in since(start) if r[2] and r[2][0].attach_link == link), None)

    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    bridge = None
    try:
        place_hand(0.0)
        bridge = VisionAttachmentBridge(node, description, config=config)
        feed["bridge"] = bridge
        bridge.setup()
        feed["on"] = True
        assert _wait_until(lambda: bool(received)), "no heartbeat once effort was live"

        depth_pub = peer.create_publisher(
            Image,
            depth_topic,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            ),
        )
        depth = Image()
        depth.header.frame_id = str(camera.frame_id)
        depth.height, depth.width = 8, 8
        depth.encoding = "32FC1"
        depth.step = 4 * depth.width
        depth.data = np.full((8, 8), 0.25, dtype=np.float32).tobytes()

        def depth_cached() -> bool:
            depth.header.stamp = peer.get_clock().now().to_msg()
            depth_pub.publish(depth)
            return bridge._depth is not None

        assert _wait_until(depth_cached), "the bridge never cached a depth frame"
        assert _wait_until(bridge._client.service_is_ready), "SegmentInView never discovered"

        def grasp() -> tuple[float, int, list[AttachedCollisionObject]]:
            start = len(received)
            feed["left"] = 0.9 * float(left.effort_limit)
            assert _wait_until(lambda: held_on(_HAND, start) is not None), "no fallback box"
            row = held_on(_HAND, start)
            assert row is not None
            return row

        def release() -> tuple[float, int, list[AttachedCollisionObject]]:
            start = len(received)
            feed["left"] = 0.01
            assert _wait_until(lambda: held_on(_BASE, start) is not None), (
                "DETACH dropped the payload instead of freezing it"
            )
            row = held_on(_BASE, start)
            assert row is not None
            return row

        # ── ATTACH: the held box rides the hand. ─────────────────────────────────────
        _, held_revision, (held,) = grasp()
        assert held.attach_link == _HAND and held.touch_links == [_FINGER]

        # ── DETACH: frozen in the base frame at the DETACH pose, ack untouched. ──────
        opened_at, frozen_revision, (frozen,) = release()
        assert frozen_revision == held_revision + 1
        assert frozen.object_id == held.object_id
        assert frozen.attach_link == _BASE == frozen.pose_in_link.frame_id
        assert frozen.touch_links == [_HAND, _FINGER]
        assert frozen.primitives == held.primitives
        expected = t_hand @ homogeneous_from_quat_xyz(
            held.pose_in_link.xyz, held.pose_in_link.quat_xyzw
        )
        np.testing.assert_allclose(frozen.pose_in_link.xyz, expected[:3, 3], atol=1e-6)
        assert bridge.attachment_action_ack_ready(), "a release window must not hold the ack"

        # ── The hand retreats 10 cm: still inside the jaws' reach of the box. ────────
        place_hand(0.10)
        moved_at = time.monotonic()
        time.sleep(6 * _PERIOD_S)
        window = [r for r in received if r[0] >= opened_at]
        after_move = [r for r in window if r[0] >= moved_at + _PERIOD_S]
        assert len(after_move) >= 3, f"only {len(after_move)} heartbeats after the move"
        for _, revision, objects in window:
            assert revision == frozen_revision, "a heartbeat re-revisioned the frozen record"
            assert objects == [frozen], "the frozen record moved, changed, or vanished"
        assert bridge.attachment_action_ack_ready()

        # ── The hand is clear: the record goes, at a fresh revision. ─────────────────
        start = len(received)
        place_hand(0.50)
        assert _wait_until(lambda: any(not r[2] for r in since(start)), timeout_s=2.0), (
            "the window never closed once the jaws were clear"
        )
        closed = next(r for r in since(start) if not r[2])
        assert closed[1] == frozen_revision + 1

        # ── A new ATTACH during a window ends it: the new grasp replaces it. ─────────
        place_hand(0.0)
        time.sleep(2 * _PERIOD_S)
        _, regrasp_revision, _ = grasp()
        _, refrozen_revision, _ = release()
        start = len(received)
        _, attach_revision, (rehold,) = grasp()
        assert rehold.attach_link == _HAND
        assert attach_revision > refrozen_revision > regrasp_revision
        time.sleep(3 * _PERIOD_S)
        tail = [r for r in since(start) if r[1] >= attach_revision]
        assert tail and all(
            len(objects) == 1 and objects[0].attach_link == _HAND for _, _, objects in tail
        ), "the frozen record outlived the new ATTACH"

        assert aggregator_errors == [], aggregator_errors
    finally:
        feed["on"] = False
        if bridge is not None:
            with suppress(Exception):
                bridge.teardown()
        executor.shutdown()
        spin.join(timeout=5.0)
        unanswered.destroy_node()
        peer.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


# ── The kernel row ──────────────────────────────────────────────────────────────────────────

#: The real cell's grid (``_octomap_resolution("real")``), laid out as the grasp band test's.
_RES = 0.02
_SX, _SY, _SZ = 6, 8, 7
_ORIGIN = (-0.06, 0.10, -0.68)


def _index(i: int, j: int, k: int) -> int:
    return i + _SX * (j + _SY * k)


#: The payload's cells: 2 x 4 x 2 cells, centres x ±0.01, y 0.15..0.21, z -0.63 / -0.61. At
#: q = 0 the left finger pair's hull contains them (the grasp band test pins this).
_PAYLOAD_CELLS = frozenset(_index(i, j, k) for i in (2, 3) for j in (2, 3, 4, 5) for k in (2, 3))
#: A shelf strip under the payload (layer centre z = -0.65): 33.5 mm from the finger hull at
#: q = 0 (grasp band test), so only the payload, whose bottom face sits 9 mm into it, reaches it.
_SHELF_CELLS = frozenset(_index(i, j, 1) for i in (2, 3) for j in (2, 3, 4, 5))
#: The released payload in ``openarm_base``: a box over exactly those cells' centres.
_PAYLOAD_CENTRE = (0.0, 0.18, -0.625)
_PAYLOAD_HALF = (0.019, 0.039, 0.024)
_Q = [0.0] * 16
_OBJECT_ID = "grasped_payload:left_gripper"


def _kernel_params(description: Any) -> dict[str, object]:
    """The safety kernel's parameters as ``deploy_e2e.launch.py`` builds them on the real cell
    with the vision leg on (attached check on, margin 0, 1000 ms deadline)."""
    from openral_core.depth_extrinsic import REAL_WORLD_VOXEL_MARGIN_M
    from openral_safety.envelope_loader import (
        collision_params_from_description,
        compute_intersection,
        ee_link_index_from_collision_params,
        kernel_params_from_envelope,
    )

    collision = collision_params_from_description(description)
    params: dict[str, object] = {
        **kernel_params_from_envelope(compute_intersection(description, skill=None, deploy=None)),
        **collision,
    }
    params["collision_joint_names"] = [j.name for j in description.joints]
    params["collision_seed_dt_s"] = 0.0
    params["collision_state_deadline_ms"] = 1000.0
    params["collision_ee_link_index"] = ee_link_index_from_collision_params(collision)
    params["world_voxel_enabled"] = True
    params["world_voxel_margin_m"] = REAL_WORLD_VOXEL_MARGIN_M
    params["world_voxel_max_cells"] = 4096
    params["world_voxel_deadline_ms"] = 2000.0
    params["attached_collision_enabled"] = True
    params["attached_collision_margin_m"] = 0.0
    params["attached_collision_deadline_ms"] = 1000.0
    return params


def test_the_kernel_accepts_the_retreat_only_while_the_frozen_record_is_published(
    reset_kernel_estop: Any,
) -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS 2 not sourced")
    import rclpy
    from geometry_msgs.msg import Point, Quaternion
    from openral_core import (
        AttachedCollisionObject,
        AttachedCollisionPrimitive,
        AttachmentEvidenceKind,
        BoxShape,
        Pose6D,
        RobotDescription,
    )
    from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
    from openral_hal.vision_attachment_bridge import freeze_released_attachment
    from openral_msgs.msg import ActionChunk, FailureTrigger, OccupancyVoxels, WorldStateStamped
    from openral_msgs.msg import AttachedCollisionObject as AttachedMsg
    from openral_msgs.msg import AttachedCollisionPrimitive as PrimitiveMsg
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty
    from std_srvs.srv import Trigger

    from tests.sim.safety._kernel_subprocess import (
        activate_kernel_node,
        isolated_domain_id,
        start_kernel,
        terminate_kernel,
    )

    description = RobotDescription.from_yaml(str(_ROBOT_YAML))
    # What the left leg held: the payload box, in the hand frame at the release pose.
    t_hand = _hand_at_zero(description)
    in_hand = np.linalg.inv(t_hand) @ homogeneous_from_quat_xyz(_PAYLOAD_CENTRE, (0, 0, 0, 1))
    w, x, y, z = rotation_to_quat_wxyz(in_hand[:3, :3])
    held = AttachedCollisionObject(
        object_id=_OBJECT_ID,
        attach_link=_HAND,
        touch_links=[_FINGER],
        primitives=[
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=_PAYLOAD_HALF),
                pose_in_object=Pose6D(
                    xyz=(0.0, 0.0, 0.0), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=_OBJECT_ID
                ),
            )
        ],
        pose_in_link=Pose6D(
            xyz=tuple(float(v) for v in in_hand[:3, 3]), quat_xyzw=(x, y, z, w), frame_id=_HAND
        ),
        confidence=0.9,
        evidence_kind=AttachmentEvidenceKind.VISION_SEGMENTATION,
        evidence_ref="segmenter_mask:release-row",
        stamp_ns=time.time_ns(),
    )
    # The record the bridge publishes during the window — the code under test.
    frozen = freeze_released_attachment(held, base_link=_BASE, t_base_from_link=t_hand)
    np.testing.assert_allclose(frozen.pose_in_link.xyz, _PAYLOAD_CENTRE, atol=1e-9)

    node_name = f"safety_kernel_release_{uuid.uuid4().hex[:8]}"
    joint_names = [j.name for j in description.joints]
    state: dict[str, Any] = {"occupied": frozenset(), "objects": [frozen], "revision": 1}
    with tempfile.TemporaryDirectory() as td:
        log_path = pathlib.Path(td) / "kernel.log"
        proc = start_kernel(
            _kernel_params(description),
            node_name,
            isolated_domain_id(),
            log_path=log_path,
            params_file=pathlib.Path(td) / "kernel_params.yaml",
        )
        try:
            time.sleep(1.5)
            rclpy.init()
            try:
                helper = rclpy.create_node("release_window_helper")
                assert activate_kernel_node(node_name, helper), "kernel activation failed"
                executor = SingleThreadedExecutor()
                executor.add_node(helper)
                safe: dict[str, Any] = {}
                failures: list[Any] = []
                estops: list[Any] = []
                chunk_pub = helper.create_publisher(ActionChunk, "/openral/candidate_action", 10)
                reset_client = helper.create_client(Trigger, "/openral/estop_reset")
                safe_sub = helper.create_subscription(
                    ActionChunk,
                    "/openral/safe_action",
                    lambda m: safe.__setitem__(m.trace_id, m),
                    10,
                )
                helper.create_subscription(
                    FailureTrigger, "/openral/failure/safety", failures.append, 50
                )
                helper.create_subscription(Empty, "/openral/estop", estops.append, 10)
                reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                voxel_pub = helper.create_publisher(
                    OccupancyVoxels, "/openral/world_voxels", reliable
                )
                state_pub = helper.create_publisher(
                    WorldStateStamped, "/openral/world_state_fast", reliable
                )
                joint_pub = helper.create_publisher(JointState, "/joint_states", 10)

                def publish_grid() -> None:
                    grid = OccupancyVoxels()
                    grid.header.frame_id = _BASE
                    grid.header.stamp = helper.get_clock().now().to_msg()
                    grid.source_stamp = grid.header.stamp
                    grid.origin = Point(x=_ORIGIN[0], y=_ORIGIN[1], z=_ORIGIN[2])
                    grid.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    grid.resolution = _RES
                    grid.size_x, grid.size_y, grid.size_z = _SX, _SY, _SZ
                    occupancy = [0] * (_SX * _SY * _SZ)
                    for idx in state["occupied"]:
                        occupancy[idx] = 1
                    grid.occupancy = occupancy
                    voxel_pub.publish(grid)

                def publish_joint_state() -> None:
                    js = JointState()
                    js.header.stamp = helper.get_clock().now().to_msg()
                    js.name = joint_names
                    js.position = list(_Q)
                    joint_pub.publish(js)

                def publish_world_state() -> None:
                    now_ns = helper.get_clock().now().nanoseconds
                    msg = WorldStateStamped()
                    msg.header.frame_id = _BASE
                    msg.header.stamp = helper.get_clock().now().to_msg()
                    msg.stamp_ns = now_ns
                    msg.attachment_revision = state["revision"]
                    msg.attachment_stamp_ns = now_ns
                    for obj in state["objects"]:
                        item = AttachedMsg()
                        obj.fill_idl(item, primitive_factory=PrimitiveMsg)
                        msg.attached_objects.append(item)
                    state_pub.publish(msg)

                for period, publish in (
                    (0.2, publish_grid),
                    (0.1, publish_joint_state),
                    (0.1, publish_world_state),
                ):
                    publish()
                    helper.create_timer(period, publish)

                def spin(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)

                deadline = time.time() + 5.0
                while time.time() < deadline and not (
                    chunk_pub.get_subscription_count() >= 1 and safe_sub.get_publisher_count() >= 1
                ):
                    executor.spin_once(timeout_sec=0.05)
                spin(0.6)

                def send(trace: str, *, expect_accept: bool) -> None:
                    seen = len(failures)
                    chunk = ActionChunk()
                    chunk.control_mode = 0  # JOINT_POSITION
                    chunk.horizon = 1
                    chunk.n_dof = len(_Q)
                    chunk.flat = list(_Q)
                    chunk.rskill_id = "openral/release-window"
                    chunk.trace_id = trace
                    chunk_pub.publish(chunk)
                    end = time.time() + 10.0
                    while time.time() < end:
                        executor.spin_once(timeout_sec=0.02)
                        if expect_accept and trace in safe:
                            break
                        if not expect_accept and len(failures) > seen and estops:
                            break
                    spin(0.4)

                def refused(trace: str) -> dict[str, Any]:
                    assert trace not in safe, f"{trace} reached /openral/safe_action"
                    assert estops, f"{trace} must fire /openral/estop"
                    trigger = failures[-1]
                    assert trigger.kind == FailureTrigger.KIND_COLLISION
                    evidence: dict[str, Any] = json.loads(trigger.evidence_json)
                    assert evidence["collision_kind"] == "world"
                    return evidence

                # ── Window open: frozen record published, its cells cleared → ACCEPTED ─────
                state.update(occupied=frozenset(), objects=[frozen], revision=1)
                spin(0.6)
                send("frozen-cleared", expect_accept=True)
                assert "frozen-cleared" in safe, "the frozen record must let the retreat start"
                assert not estops

                # ── Window open, a shelf cell under the payload → the payload is checked ───
                state.update(occupied=_SHELF_CELLS)
                spin(0.6)
                send("frozen-shelf", expect_accept=False)
                evidence = refused("frozen-shelf")
                assert evidence["link_a"] == f"attached:{_OBJECT_ID}", evidence
                assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _SHELF_CELLS}
                reset_kernel_estop(reset_client, executor, spin, estops)

                # ── Window closed while the jaws are still there: cells re-marked → STOP ───
                state.update(occupied=_PAYLOAD_CELLS, objects=[], revision=2)
                spin(0.6)
                send("dropped-remarked", expect_accept=False)
                evidence = refused("dropped-remarked")
                assert evidence["link_a"] == _FINGER, evidence
                assert evidence["link_b_or_object"] in {f"voxel_{i}" for i in _PAYLOAD_CELLS}
            finally:
                rclpy.shutdown()
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n\n--- safety_kernel_node log ---\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            ) from exc
        finally:
            terminate_kernel(proc)
