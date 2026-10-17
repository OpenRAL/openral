# SPDX-License-Identifier: Apache-2.0
"""Live-ROS rig for the #418 object tracker: real nodes, the SO-101 PickOrange camera.

Real rclpy, real ``openral_msgs`` on the wire, the real World State lifecycle node with a
real ``WorldStateAggregator``, and the real ``object_tracker_node`` (started by
``TrackerRig.tracker``). Inputs are the SO-101 PickOrange ``top_depth`` camera from the
committed scene, ray-cast (``tests/unit/_tracker_scenes``) with a ∅4.8 cm orange standing
where the SO-101's pinch volume is at a measured hold configuration: depth (32FC1) +
``CameraInfo`` + an all-free robot mask at the depth stamp, the voxel grid a single-view
octomap of that depth would hold, a static tf ``base <- optical`` and the grasp
declaration naming ``orange`` with a seed box. The one double is the ``SegmentInView``
peer at the service boundary: it answers with the ray-cast's own label mask under the
prompt (``test_grasp_target_leg_live`` does the same).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

ROBOT = "robots/so101_follower/robot.yaml"
OPTICAL = "top_depth_optical"
ORANGE_R = 0.024


def wait(predicate: Callable[[], bool], timeout_s: float = 20.0) -> bool:
    """Poll ``predicate`` every 50 ms until it holds or ``timeout_s`` passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TrackerRig:
    """The helper node's peers, publishers and recorders around one World State node."""

    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped
        from openral_core import CameraTopicKind, RobotDescription, camera_topic
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
        from openral_msgs.srv import SegmentInView
        from openral_world_state import WorldStateAggregator
        from openral_world_state_ros.lifecycle_node import _WorldStateLifecycleNode
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from tf2_ros import StaticTransformBroadcaster

        from tests.unit._object_tracker_fixtures import arm_at_hold
        from tests.unit._tracker_scenes import grid_from_depth, render, so101_top, sphere

        _, self.q, self.pinch = arm_at_hold("so101_follower")
        # The orange stands in the pinch volume, on a support plane under it.
        rig = replace(
            so101_top(),
            centre_xy=(float(self.pinch[0]), float(self.pinch[1])),
            support_z=float(self.pinch[2]) - ORANGE_R,
        )
        x, y, z = (float(v) for v in self.pinch)
        orange = sphere((x, y, z), ORANGE_R)
        self.depth, labels = render([orange], rig)
        self.grid = grid_from_depth(self.depth, rig)
        k = self.k = rig.k
        self.description = RobotDescription.from_yaml(ROBOT)

        self.latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        rclpy.init()
        self.executor = MultiThreadedExecutor(num_threads=3)
        helper = self.helper = rclpy.create_node("tracker_live_helper")
        self.tracked: list[Any] = []
        self.attach: list[Any] = []
        self.fast: list[Any] = []
        helper.create_subscription(
            TrackedObjectSet, "/openral/tracked_objects", self.tracked.append, self.reliable
        )
        helper.create_subscription(
            AttachmentState, "/openral/attachment_state", self.attach.append, self.latched
        )
        helper.create_subscription(
            WorldStateStamped, "/openral/world_state_fast", self.fast.append, self.reliable
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
        tf.header.frame_id, tf.child_frame_id = "base", OPTICAL
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(v) for v in rig.t[:3, 3]
        )
        w, x, y, z = rotation_to_quat_wxyz(rig.t[:3, :3])
        tf.transform.rotation.x, tf.transform.rotation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = z, w
        StaticTransformBroadcaster(helper).sendTransform(tf)
        self._msgs = (CameraInfo, Image, OccupancyVoxels, JointState, GraspEvent)
        self.pub_depth = helper.create_publisher(
            Image, camera_topic("top", CameraTopicKind.DEPTH_IMAGE), sensor
        )
        self.pub_info = helper.create_publisher(
            CameraInfo, camera_topic("top", CameraTopicKind.DEPTH_CAMERA_INFO), sensor
        )
        self.pub_mask = helper.create_publisher(
            Image, camera_topic("top", CameraTopicKind.ROBOT_MASK), sensor
        )
        self.pub_grid = helper.create_publisher(
            OccupancyVoxels, "/openral/world_voxels", self.reliable
        )
        self.pub_decl = helper.create_publisher(
            GraspDeclarationMsg, "/openral/grasp_declaration", self.latched
        )
        self.pub_js = helper.create_publisher(JointState, "/joint_states", sensor)
        self.pub_event = helper.create_publisher(
            GraspEvent,
            "/openral/grasp_events",
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            ),
        )
        self.world_state = _WorldStateLifecycleNode(
            aggregator=WorldStateAggregator(self.description)
        )
        self.executor.add_node(helper)
        self.executor.add_node(self.world_state)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()
        self.world_state.trigger_configure()
        self.world_state.trigger_activate()

    def tracker(self, mode: str) -> Any:
        """Create, configure and activate a real ``object_tracker_node`` in ``mode``."""
        from openral_world_state_ros.object_tracker_node import make_object_tracker_node
        from rclpy.parameter import Parameter

        node = make_object_tracker_node()
        node.set_parameters(
            [
                Parameter("robot_yaml", value=ROBOT),
                Parameter("mode", value=mode),
                Parameter("cameras", value=["top"]),
                Parameter("grounding_wait_s", value=5.0),
                Parameter("cell_m", value=float(self.grid.resolution)),
            ]
        )
        self.executor.add_node(node)
        node.trigger_configure()
        node.trigger_activate()
        return node

    def frame(self, grid_frame: str | None = None) -> None:
        """Publish one depth frame, its info and robot mask, and the grid (in ``grid_frame``)."""
        camera_info, image, voxels, _, _ = self._msgs
        k, grid = self.k, self.grid
        now = self.helper.get_clock().now().to_msg()
        info = camera_info(width=k.width, height=k.height)
        info.header.frame_id = OPTICAL
        info.k = [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0]
        self.pub_info.publish(info)
        img = image(height=k.height, width=k.width, encoding="32FC1", step=4 * k.width)
        img.header.stamp, img.header.frame_id = now, OPTICAL
        img.data = self.depth.astype("<f4").tobytes()
        mask = image(height=k.height, width=k.width, encoding="mono8", step=k.width)
        mask.header.stamp, mask.header.frame_id = now, OPTICAL
        mask.data = bytes(k.width * k.height)
        g = voxels()
        g.header.frame_id, g.header.stamp, g.source_stamp = grid_frame or grid.frame_id, now, now
        g.origin.x, g.origin.y, g.origin.z = grid.origin
        g.orientation.w = 1.0
        g.resolution = grid.resolution
        g.size_x, g.size_y, g.size_z = grid.size
        g.occupancy = grid.occupancy.tolist()
        self.pub_grid.publish(g)
        self.pub_mask.publish(mask)
        self.pub_depth.publish(img)

    def declare(self) -> int:
        """Publish the grasp declaration naming ``orange`` around the pinch; its stamp."""
        from openral_core import GraspDeclaration, ObjectQuery, ObjectRole, PlaceRegion, Pose6D
        from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
        from openral_msgs.msg import ObjectQuery as ObjectQueryMsg

        seed = PlaceRegion(
            frame_id="base",
            pose=Pose6D(
                xyz=tuple(float(v) for v in self.pinch), quat_xyzw=(0, 0, 0, 1), frame_id="base"
            ),
            half_extents=(0.06, 0.06, 0.05),
        )
        decl = GraspDeclaration(
            target_id="orange",
            contact_links=self.description.grasp_contact_links(),
            timeout_s=120.0,
            stamp_ns=self.helper.get_clock().now().nanoseconds,
            target_queries=(
                ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET, seed=seed),
            ),
        )
        msg = GraspDeclarationMsg()
        decl.fill_idl(msg, query_factory=ObjectQueryMsg)
        self.pub_decl.publish(msg)
        return decl.stamp_ns

    def ground(self) -> bool:
        """Feed frames until the set on ``/openral/tracked_objects`` holds a CONFIRMED track."""
        from openral_core import TrackStatus

        def confirmed() -> bool:
            self.frame()
            time.sleep(0.2)
            return any(
                o.status == TrackStatus.CONFIRMED.value
                for s in self.tracked[-1:]
                for o in s.objects
            )

        return wait(confirmed)

    def publish_hold(self) -> None:
        """Publish the measured hold configuration on ``/joint_states``."""
        joint_state = self._msgs[3]
        js = joint_state(name=list(self.q), position=[float(v) for v in self.q.values()])
        js.header.stamp = self.helper.get_clock().now().to_msg()
        self.pub_js.publish(js)

    def attach_event(self) -> None:
        """Publish an ATTACH grasp event at the hold."""
        grasp_event = self._msgs[4]
        event = grasp_event(kind=grasp_event.KIND_ATTACH, gripper_joint="gripper")
        event.jaw_measured = event.hold_position = event.jaw_commanded = self.q["gripper"]
        event.stamp = self.helper.get_clock().now().to_msg()
        self.pub_event.publish(event)

    def close(self) -> None:
        import rclpy

        self.executor.shutdown()
        self.world_state.destroy_node()
        self.helper.destroy_node()
        rclpy.shutdown()
        self._spin.join(timeout=5.0)


@contextmanager
def tracker_rig() -> Iterator[TrackerRig]:
    """A ``TrackerRig`` torn down (rclpy shut down) on exit."""
    rig = TrackerRig()
    try:
        yield rig
    finally:
        rig.close()
