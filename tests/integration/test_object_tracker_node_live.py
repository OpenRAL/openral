# SPDX-License-Identifier: Apache-2.0
"""Live-ROS: the #418 object-tracker node grounds, relays, binds and survives a restart.

Real rclpy, real ``openral_msgs`` on the wire, the real ``object_tracker_node`` (mode
``bind``) and the real World State lifecycle node with a real ``WorldStateAggregator``.
Inputs are the SO-101 PickOrange ``top_depth`` camera from the committed scene, ray-cast
(``tests/unit/_tracker_scenes``) with a ∅4.8 cm orange standing where the SO-101's pinch
volume is at a measured hold configuration: depth (32FC1) + ``CameraInfo`` + an all-free
robot mask at the depth stamp, the voxel grid a single-view octomap of that depth would
hold, a static tf ``base <- optical`` and the grasp declaration naming ``orange`` with a
seed box. The one double is the ``SegmentInView`` peer at the service boundary: it
answers with the ray-cast's own label mask under the prompt (``test_grasp_target_leg_live``
does the same).

Asserts: the set reaches CONFIRMED on ``/openral/tracked_objects`` with a revision seeded
from wall time; ``world_state_fast`` carries it with ``tracked_objects_valid``; an ATTACH
at the hold publishes ``/openral/attachment_state`` with a ``TRACKED_OBJECT`` binding of
that id and the set shows it BOUND; with the tracker deactivated the relayed set goes
invalid within the aggregator's staleness window; a fresh tracker node's first revision is
above every revision before the restart (the kernel refuses a regressing one).

Gated on ``OPENRAL_TEST_ROS_LIVE=1``. Locally::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    PYTHONPATH=$PWD/packages/world_state:$PWD:$PYTHONPATH OPENRAL_TEST_ROS_LIVE=1 \\
        pytest tests/integration/test_object_tracker_node_live.py
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
pytestmark = pytest.mark.skipif(
    not _LIVE_ROS,
    reason="live rclpy nodes + colcon openral_msgs overlay — set OPENRAL_TEST_ROS_LIVE=1 and "
    "source install/setup.bash first.",
)

_ROBOT = "robots/so101_follower/robot.yaml"
_OPTICAL = "top_depth_optical"
_R = 0.024


def _wait(predicate: Any, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_tracker_node_grounds_relays_binds_and_never_regresses() -> None:
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("openral_msgs")
    from geometry_msgs.msg import TransformStamped
    from openral_core import (
        CameraTopicKind,
        GraspDeclaration,
        ObjectQuery,
        ObjectRole,
        PlaceRegion,
        Pose6D,
        RobotDescription,
        TrackStatus,
        camera_topic,
    )
    from openral_core.geometry import rotation_to_quat_wxyz
    from openral_core.perception_geometry import project_point
    from openral_msgs.msg import (
        AttachmentState,
        GraspEvent,
        OccupancyVoxels,
        TrackedObjectSet,
        WorldStateStamped,
    )
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import ObjectQuery as ObjectQueryMsg
    from openral_msgs.srv import SegmentInView
    from openral_world_state import WorldStateAggregator
    from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
    from openral_world_state_ros.object_tracker_node import make_object_tracker_node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from tf2_ros import StaticTransformBroadcaster

    from tests.unit._object_tracker_fixtures import arm_at_hold
    from tests.unit._tracker_scenes import grid_from_depth, render, so101_top, sphere

    _, q, pinch = arm_at_hold("so101_follower")
    # The orange stands in the pinch volume, on a support plane under it.
    rig = replace(
        so101_top(),
        centre_xy=(float(pinch[0]), float(pinch[1])),
        support_z=float(pinch[2]) - _R,
    )
    orange = sphere((float(pinch[0]), float(pinch[1]), float(pinch[2])), _R)
    depth, labels = render([orange], rig)
    grid = grid_from_depth(depth, rig)
    k = rig.k

    latched = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    reliable = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1
    )
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=3)
    helper = rclpy.create_node("tracker_live_helper")
    tracked: list[Any] = []
    attach: list[Any] = []
    fast: list[Any] = []
    helper.create_subscription(
        TrackedObjectSet, "/openral/tracked_objects", tracked.append, reliable
    )
    helper.create_subscription(AttachmentState, "/openral/attachment_state", attach.append, latched)
    helper.create_subscription(
        WorldStateStamped, "/openral/world_state_fast", fast.append, reliable
    )

    def segment(req: Any, res: Any) -> Any:  # the ray-cast's own label under the prompt
        p = req.tcp_point
        uv = project_point((p.x, p.y, p.z), np.eye(4), k)
        n = 0 if uv is None else int(labels[int(uv[1]), int(uv[0])])
        res.ok, res.camera = n > 0, req.camera
        if n:
            img = Image(height=k.height, width=k.width, encoding="mono8", step=k.width)
            img.data = ((labels == n).astype(np.uint8) * 255).tobytes()
            res.masks = [img]
        return res

    helper.create_service(SegmentInView, "/openral/perception/segment_in_view", segment)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "base", _OPTICAL
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
        float(v) for v in rig.t[:3, 3]
    )
    w, x, y, z = rotation_to_quat_wxyz(rig.t[:3, :3])
    tf.transform.rotation.x, tf.transform.rotation.y = x, y
    tf.transform.rotation.z, tf.transform.rotation.w = z, w
    StaticTransformBroadcaster(helper).sendTransform(tf)
    sensor = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )
    pub_depth = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.DEPTH_IMAGE), sensor
    )
    pub_info = helper.create_publisher(
        CameraInfo, camera_topic("top", CameraTopicKind.DEPTH_CAMERA_INFO), sensor
    )
    pub_mask = helper.create_publisher(
        Image, camera_topic("top", CameraTopicKind.ROBOT_MASK), sensor
    )
    pub_grid = helper.create_publisher(OccupancyVoxels, "/openral/world_voxels", reliable)
    pub_decl = helper.create_publisher(GraspDeclarationMsg, "/openral/grasp_declaration", latched)
    pub_js = helper.create_publisher(JointState, "/joint_states", sensor)
    pub_event = helper.create_publisher(
        GraspEvent,
        "/openral/grasp_events",
        QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        ),
    )

    def tracker(mode: str) -> Any:
        node = make_object_tracker_node()
        node.set_parameters(
            [
                Parameter("robot_yaml", value=_ROBOT),
                Parameter("mode", value=mode),
                Parameter("cameras", value=["top"]),
                Parameter("grounding_wait_s", value=5.0),
                Parameter("cell_m", value=float(grid.resolution)),
            ]
        )
        executor.add_node(node)
        node.trigger_configure()
        node.trigger_activate()
        return node

    def frame() -> None:
        now = helper.get_clock().now().to_msg()
        info = CameraInfo(width=k.width, height=k.height)
        info.header.frame_id = _OPTICAL
        info.k = [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0]
        pub_info.publish(info)
        img = Image(height=k.height, width=k.width, encoding="32FC1", step=4 * k.width)
        img.header.stamp, img.header.frame_id = now, _OPTICAL
        img.data = depth.astype("<f4").tobytes()
        mask = Image(height=k.height, width=k.width, encoding="mono8", step=k.width)
        mask.header.stamp, mask.header.frame_id = now, _OPTICAL
        mask.data = bytes(k.width * k.height)
        g = OccupancyVoxels()
        g.header.frame_id, g.header.stamp, g.source_stamp = grid.frame_id, now, now
        g.origin.x, g.origin.y, g.origin.z = grid.origin
        g.orientation.w = 1.0
        g.resolution = grid.resolution
        g.size_x, g.size_y, g.size_z = grid.size
        g.occupancy = grid.occupancy.tolist()
        pub_grid.publish(g)
        pub_mask.publish(mask)
        pub_depth.publish(img)

    aggregator = WorldStateAggregator(RobotDescription.from_yaml(_ROBOT))
    world_state = _WorldStateLifecycleNode(aggregator=aggregator)
    executor.add_node(helper)
    executor.add_node(world_state)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    world_state.trigger_configure()
    world_state.trigger_activate()
    first = None
    try:
        activated_ns = time.time_ns()
        first = tracker("bind")
        seed = PlaceRegion(
            frame_id="base",
            pose=Pose6D(
                xyz=tuple(float(v) for v in pinch), quat_xyzw=(0, 0, 0, 1), frame_id="base"
            ),
            half_extents=(0.06, 0.06, 0.05),
        )
        decl = GraspDeclaration(
            target_id="orange",
            contact_links=RobotDescription.from_yaml(_ROBOT).grasp_contact_links(),
            timeout_s=120.0,
            stamp_ns=helper.get_clock().now().nanoseconds,
            target_queries=(
                ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, seed=seed),
            ),
        )
        decl_msg = GraspDeclarationMsg()
        decl.fill_idl(decl_msg, query_factory=ObjectQueryMsg)
        pub_decl.publish(decl_msg)

        def confirmed() -> bool:
            frame()
            time.sleep(0.2)
            return any(
                o.status == TrackStatus.CONFIRMED.value for s in tracked[-1:] for o in s.objects
            )

        assert _wait(confirmed), [[o.status for o in s.objects] for s in tracked[-3:]]
        [obj] = tracked[-1].objects
        assert obj.object_id == "trk:1" and obj.declaration_key == f"orange@{decl.stamp_ns}"
        assert all(s.revision >= activated_ns for s in tracked)
        assert _wait(lambda: bool(fast) and fast[-1].tracked_objects_valid, 5.0)
        assert [o.object_id for o in fast[-1].tracked_objects.objects] == ["trk:1"]

        # ATTACH at the measured hold: the tracker owns attachment_state in `bind`.
        js = JointState(name=list(q), position=[float(v) for v in q.values()])
        js.header.stamp = helper.get_clock().now().to_msg()
        pub_js.publish(js)
        assert _wait(lambda: bool(attach), 5.0)  # the empty, fresh set first
        event = GraspEvent(kind=GraspEvent.KIND_ATTACH, gripper_joint="gripper")
        event.jaw_measured = event.hold_position = event.jaw_commanded = q["gripper"]
        event.stamp = helper.get_clock().now().to_msg()
        pub_event.publish(event)
        assert _wait(lambda: bool(attach[-1].objects), 5.0)
        [held] = attach[-1].objects
        assert held.object_id == "trk:1" and held.evidence_kind == "tracked_object"
        assert attach[-1].grasp_declaration_valid and not attach[-1].grasp_declaration.region_valid
        assert _wait(lambda: tracked[-1].objects[0].status == TrackStatus.BOUND.value, 5.0)
        last_revision = max(s.revision for s in tracked)

        # The tracker stops: world_state_fast drops the set within the staleness window.
        first.trigger_deactivate()
        assert _wait(lambda: not fast[-1].tracked_objects_valid, 4.0)
        first.trigger_cleanup()
        executor.remove_node(first)
        first.destroy_node()
        first = None

        # A restarted tracker never regresses the revision the kernel last accepted.
        n_before = len(tracked)
        first = tracker("bind")
        assert _wait(lambda: len(tracked) > n_before, 5.0)
        assert tracked[n_before].revision > last_revision
    finally:
        if first is not None:
            first.destroy_node()
        executor.shutdown()
        world_state.destroy_node()
        helper.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=5.0)
