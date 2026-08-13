"""HAL-side client for the vision attachment-evidence path.

The ROS wiring that turns :mod:`~openral_hal._grasp_trigger` events into
``openral_msgs/srv/SegmentInView`` calls and feeds the replies to
:class:`~openral_hal._vision_attachment_evidence.VisionAttachmentEvidenceProducer`,
publishing the resulting ``AttachmentState`` the safety kernel already consumes.

The real-hardware sibling of the attachment leg in
:mod:`~openral_hal.sim_sensor_bridge`, which reads MuJoCo ground truth. Enable
it explicitly (``vision_attachment_enabled``); the two must not both drive
``/openral/attachment_state``.

**Torch-free, by construction.** Everything model-shaped lives behind the
service, in ``openral_perception_ros.segmenter_node``. This module imports
numpy, rclpy and ``openral_core`` — never ``torch``, never ``transformers``, and
never the runner's segmenter backend.

The deferred-ack barrier
------------------------

The HAL already holds the ``action_applied`` acknowledgement of a grouped tick
until attached-payload perception settles (``attachment_action_ack_ready`` /
``_on_attachment_perception_ready``). In the simulator that wait is for a
transparent depth frame — about 100 ms. Segmentation **rides inside that same
wait**: a warmed SAM 2.1 call was measured at ~53 ms on the reference GPU, so on
that host the barrier does not grow at all. This bridge exposes the same
``attachment_action_ack_ready`` shape, so the node holds the tick for it exactly
as it does for the simulator's depth frames.

Three properties of that wait are non-negotiable:

* **It is bounded.** Every request carries a deadline
  (:attr:`VisionAttachmentConfig.deadline_s`). When it expires the attachment is
  resolved from what is in hand — nothing — rather than waiting longer. A robot
  that stalls because a perception node is wedged is a worse failure than a
  conservatively-shaped payload.
* **It never skips.** A timeout, a service failure, a missing depth frame and a
  missing transform all end in the producer's conservative jaw-span box, stamped
  :attr:`~openral_core.AttachmentEvidenceKind.GRIPPER_FORCE` at low confidence.
  Something is in the jaws either way; the collision checker must see *some*
  geometry.
* **It is visible.** Every fallback logs its typed reason and every attachment
  logs the gate report (CLAUDE.md §1.4). Nothing here degrades silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

from openral_hal._grasp_trigger import GraspEvent, GraspTriggerConfig, GripperEffortTrigger
from openral_hal._vision_attachment_evidence import (
    VisionAttachmentEvidenceProducer,
    VisionGateConfig,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import IntrinsicsPinhole, JointState, RobotDescription

__all__ = [
    "DEFAULT_SEGMENT_SERVICE",
    "SegmentOutcome",
    "VisionAttachmentBridge",
    "VisionAttachmentConfig",
    "decode_mono8_mask",
    "resolve_segment_outcome",
]

#: Service the perception-side segmenter node offers by default.
DEFAULT_SEGMENT_SERVICE = "/openral/perception/segment_in_view"

#: What one grasp needs from TF and the manifest before it can prompt:
#: ``(t_link_from_cam, tcp_in_link, negative_points_in_link)``.
_PromptContext = tuple[
    "NDArray[np.float64]",
    tuple[float, float, float],
    list[tuple[float, float, float]],
]


@dataclass(frozen=True)
class VisionAttachmentConfig:
    """Wiring for :class:`VisionAttachmentBridge`.

    Attributes:
        camera: Logical camera id — a ``SensorSpec`` name in the robot manifest,
            which is where its intrinsics and optical frame come from. Empty
            selects the manifest's first camera-like depth sensor.
        depth_topic: ``sensor_msgs/Image`` topic carrying that camera's metric
            depth (``32FC1`` metres or ``16UC1`` millimetres).
        service_name: ``SegmentInView`` service to call.
        deadline_s: Upper bound on one segmentation, from request to reply.
            *Calibration point*, ``0.25`` s — a warmed call was measured at
            ~53 ms on the reference GPU plus a service round trip, so this is
            roughly a 4x margin over the measured path while staying the same
            order as the ~100 ms barrier it rides inside. A **CPU-only** host is
            far slower than that and must raise this deliberately; leaving it at
            the default there means every grasp falls back, which is safe,
            visible in the log, and useless.
        tcp_frame: tf2 frame of the tool center point. Empty resolves to the
            gripper joint's ``child_link`` — the moving jaw — which is the
            closest thing every manifest already declares. A robot with a
            calibrated mid-jaw frame should name it here rather than accept the
            approximation.
        jaw_tip_frames: Optional tf2 frames used as **negative** prompt points,
            pushing the mask off the gripper fingers. Empty = none; the single
            positive point at the TCP is what makes the prompt meaningful.
        object_id: Identity stamped on the attachment. Vision cannot name what
            it segmented, so this is a stable slot ("the thing in the jaws"),
            not a recognition result.
    """

    camera: str = ""
    depth_topic: str = ""
    service_name: str = DEFAULT_SEGMENT_SERVICE
    deadline_s: float = 0.25
    tcp_frame: str = ""
    jaw_tip_frames: tuple[str, ...] = ()
    object_id: str = "grasped_payload"


@dataclass(frozen=True)
class SegmentOutcome:
    """What one ``SegmentInView`` round trip yielded, and why.

    Attributes:
        use_masks: Whether the reply carried candidates worth gating. ``False``
            sends the producer down its conservative fallback path.
        reason: Typed, human-readable explanation, empty only on the clean path.
            Logged and carried into the attachment trace.
    """

    use_masks: bool
    reason: str


def resolve_segment_outcome(
    *,
    timed_out: bool,
    ok: bool,
    failure_reason: str,
    mask_count: int,
) -> SegmentOutcome:
    """Decide how one service round trip resolves — pure, so it is testable.

    Every branch that is not "the segmenter returned candidates" ends in the
    conservative fallback, and every one of them names itself. There is no
    branch that skips the attachment.

    Args:
        timed_out: The deadline expired before a reply arrived.
        ok: The reply's ``ok`` flag.
        failure_reason: The reply's typed ``failure_reason``.
        mask_count: How many candidate masks the reply carried.

    Returns:
        The :class:`SegmentOutcome`.

    Example:
        >>> late = resolve_segment_outcome(
        ...     timed_out=True, ok=False, failure_reason="", mask_count=0
        ... )
        >>> late.use_masks, late.reason.split(":")[0]
        (False, 'ROSDeadlineMissed')
        >>> resolve_segment_outcome(timed_out=False, ok=True, failure_reason="", mask_count=3)
        SegmentOutcome(use_masks=True, reason='')
    """
    if timed_out:
        return SegmentOutcome(
            use_masks=False,
            reason="ROSDeadlineMissed: SegmentInView did not answer within the attach deadline",
        )
    if not ok:
        return SegmentOutcome(
            use_masks=False,
            reason=failure_reason or "ROSPerceptionStale: SegmentInView returned ok=False",
        )
    if mask_count == 0:
        return SegmentOutcome(
            use_masks=False,
            reason="ROSPerceptionStale: SegmentInView returned ok=True with no candidate masks",
        )
    return SegmentOutcome(use_masks=True, reason="")


def decode_mono8_mask(data: bytes, *, height: int, width: int) -> NDArray[np.bool_]:
    """Decode ``mono8`` mask bytes into an ``(H, W)`` boolean array.

    The reader half of ``segmenter_node.mono8_bytes_from_mask``. Any non-zero
    pixel is set — the producer writes 255, but a mask that survived a lossy hop
    must not silently become empty because it arrived as 254.

    Args:
        data: ``height * width`` tightly-packed bytes.
        height: Rows.
        width: Columns.

    Returns:
        The boolean mask.

    Raises:
        ROSConfigError: If the payload length does not match ``height * width``.

    Example:
        >>> decode_mono8_mask(bytes([255, 0]), height=1, width=2).tolist()
        [[True, False]]
    """
    flat = np.frombuffer(data, dtype=np.uint8)
    if flat.size != height * width:
        raise ROSConfigError(
            f"decode_mono8_mask: {flat.size} bytes for a {height}x{width} mono8 mask."
        )
    return np.asarray(flat.reshape(height, width) != 0, dtype=bool)


class VisionAttachmentBridge:
    """Drive SegmentInView from grasp events and hold the ack barrier for it.

    Args:
        node: The HAL lifecycle node. Owns the executor these callbacks run on.
        description: The robot manifest — gripper joints, camera intrinsics,
            frames.
        on_perception_ready: Called when a held barrier is released, so the node
            can publish the deferred ``action_applied`` tick. Same callback the
            simulator bridge is given.
        config: Wiring and the segmentation deadline.
        gate_config: Geometric gate thresholds handed to the producer.
        trigger_config: Effort thresholds and debounce for the grasp trigger.

    Raises:
        ROSConfigError: If the manifest cannot support the producer or the
            trigger (no gripper joints, no effort limit, no camera intrinsics).
    """

    def __init__(
        self,
        node: Any,
        description: RobotDescription,
        *,
        on_perception_ready: Any = None,
        config: VisionAttachmentConfig | None = None,
        gate_config: VisionGateConfig | None = None,
        trigger_config: GraspTriggerConfig | None = None,
    ) -> None:
        """Resolve manifest-derived wiring; create no ROS entities yet."""
        self._node = node
        self._description = description
        self._config = config or VisionAttachmentConfig()
        self._on_perception_ready = on_perception_ready
        self._producer = VisionAttachmentEvidenceProducer(description, config=gate_config)
        self._trigger = GripperEffortTrigger(description, config=trigger_config)
        self._camera, self._intrinsics = self._resolve_camera()
        self._tcp_frame = self._config.tcp_frame or self._resolve_tcp_frame()

        self._depth: tuple[NDArray[np.float64], int] | None = None
        self._depth_sub: Any = None
        self._client: Any = None
        self._attachment_pub: Any = None
        self._tf_buffer: Any = None
        self._tf_listener: Any = None
        self._deadline_timer: Any = None
        self._inflight: Any = None
        self._pending = False
        self._revision = 0

    # ── wiring ───────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Create the depth subscription, TF listener, service client, publisher."""
        import tf2_ros
        from openral_msgs.msg import AttachmentState
        from openral_msgs.srv import SegmentInView
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import Image

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        depth_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._depth_sub = self._node.create_subscription(
            Image, self._depth_topic(), self._on_depth, depth_qos
        )
        # Same class the simulator bridge publishes on: the kernel's authoritative
        # attachment snapshot is latched description-class data.
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._attachment_pub = self._node.create_publisher(
            AttachmentState, "/openral/attachment_state", state_qos
        )
        self._client = self._node.create_client(SegmentInView, self._config.service_name)
        self._node.get_logger().info(
            f"vision attachment bridge: camera={self._camera!r} depth={self._depth_topic()!r} "
            f"service={self._config.service_name!r} tcp_frame={self._tcp_frame!r} "
            f"deadline={self._config.deadline_s:.3f}s "
            f"effort_thresholds_n={self._trigger.thresholds_n}"
        )

    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent, and never leaves the barrier shut."""
        self._cancel_deadline()
        if self._depth_sub is not None:
            self._node.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        if self._attachment_pub is not None:
            self._node.destroy_publisher(self._attachment_pub)
            self._attachment_pub = None
        self._client = None
        self._tf_listener = None
        self._tf_buffer = None
        self._inflight = None
        # A bridge torn down mid-flight must not leave the node deferring an
        # acknowledgement forever.
        self._pending = False

    # ── barrier ──────────────────────────────────────────────────────────────

    def attachment_action_ack_ready(self) -> bool:
        """Whether attached-payload perception has settled for this tick.

        The same shape :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`
        exposes, so the node's deferred-ack path treats both identically.
        """
        return not self._pending

    # ── trigger ──────────────────────────────────────────────────────────────

    def observe_joint_state(self, state: JointState) -> None:
        """Fold one HAL read into the grasp trigger and act on any transition.

        Args:
            state: The tick's joint state, straight from the HAL.
        """
        event = self._trigger.update(state)
        if event is None:
            return
        if event is GraspEvent.DETACH:
            self._node.get_logger().info("grasp trigger: DETACH — publishing an empty attachment")
            self._publish_attachment([])
            return
        self._node.get_logger().info(
            f"grasp trigger: {event.value.upper()} — holding the action ack for segmentation"
        )
        self._begin_segmentation(stamp_ns=int(state.stamp_ns))

    @property
    def missing_effort_ticks(self) -> int:
        """Ticks whose gripper carried no effort value — a driver-health signal."""
        return self._trigger.missing_effort_ticks

    # ── segmentation round trip ──────────────────────────────────────────────

    def _begin_segmentation(self, *, stamp_ns: int) -> None:
        """Close the barrier and dispatch one bounded SegmentInView request."""
        self._pending = True
        context = self._gather_context()
        if isinstance(context, str):  # a typed precondition failure
            self._finish(stamp_ns=stamp_ns, masks=[], scores=[], reason=context)
            return
        t_link_from_cam, tcp_in_link, negatives = context
        if self._client is None or not self._client.service_is_ready():
            self._finish(
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason=(
                    f"ROSDispatchUnavailable: no server on {self._config.service_name!r}; "
                    "is the segmenter node active?"
                ),
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return

        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import Point
        from openral_msgs.srv import SegmentInView

        request = SegmentInView.Request()
        request.stamp = Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        request.camera = self._camera
        request.frame_id = self._producer.attach_link
        request.tcp_point = Point(x=tcp_in_link[0], y=tcp_in_link[1], z=tcp_in_link[2])
        request.negative_points = [Point(x=x, y=y, z=z) for x, y, z in negatives]

        future = self._client.call_async(request)
        self._inflight = future
        future.add_done_callback(
            lambda fut: self._on_reply(
                fut,
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
        )
        self._deadline_timer = self._node.create_timer(
            self._config.deadline_s,
            lambda: self._on_deadline(
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            ),
        )

    def _on_reply(
        self,
        future: Any,
        *,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve the attachment from a SegmentInView reply (or its exception)."""
        if future is not self._inflight:
            return  # already resolved by the deadline; this reply is late
        self._cancel_deadline()
        self._inflight = None
        try:
            response = future.result()
        except Exception as exc:  # a service exception must degrade, not propagate
            self._finish(
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason=f"ROSDispatchUnavailable: SegmentInView raised {type(exc).__name__}: {exc}",
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return
        outcome = resolve_segment_outcome(
            timed_out=False,
            ok=bool(response.ok),
            failure_reason=str(response.failure_reason),
            mask_count=len(response.masks),
        )
        masks = (
            [
                decode_mono8_mask(bytes(image.data), height=image.height, width=image.width)
                for image in response.masks
            ]
            if outcome.use_masks
            else []
        )
        self._finish(
            stamp_ns=stamp_ns,
            masks=masks,
            scores=[float(s) for s in response.mask_scores_advisory] if outcome.use_masks else [],
            reason=outcome.reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _on_deadline(
        self,
        *,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve conservatively when the segmenter overran its budget."""
        self._cancel_deadline()
        if self._inflight is None:
            return
        self._inflight.cancel()
        self._inflight = None
        outcome = resolve_segment_outcome(timed_out=True, ok=False, failure_reason="", mask_count=0)
        self._finish(
            stamp_ns=stamp_ns,
            masks=[],
            scores=[],
            reason=outcome.reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _finish(
        self,
        *,
        stamp_ns: int,
        masks: Sequence[NDArray[np.bool_]],
        scores: Sequence[float],
        reason: str,
        t_link_from_cam: NDArray[np.float64] | None = None,
        tcp_in_link: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Gate, publish, log, and release the barrier — on every path.

        ``masks`` empty is not an error branch: the producer answers it with its
        conservative jaw-span box, so the collision checker always receives
        geometry.
        """
        depth = self._depth[0] if self._depth is not None else np.zeros((0, 0), dtype=np.float64)
        transform = t_link_from_cam if t_link_from_cam is not None else np.eye(4, dtype=np.float64)

        def _fit(candidates: Sequence[NDArray[np.bool_]], advisory: Sequence[float]) -> Any:
            return self._producer.on_grasp(
                masks=candidates,
                depth_m=depth,
                intrinsics=self._intrinsics,
                t_link_from_cam=transform,
                tcp_in_link=tcp_in_link,
                object_id=self._config.object_id,
                stamp_ns=stamp_ns,
                mask_scores_advisory=advisory,
            )

        try:
            attachment, report = _fit(masks, scores)
        except ROSConfigError as exc:
            # Shape disagreements (mask vs depth vs intrinsics) are configuration
            # bugs, not grasp outcomes. They must neither stall the tick nor
            # leave the payload invisible, so the conservative box still ships.
            reason = f"ROSConfigError: {exc}"
            self._node.get_logger().error(f"vision attachment producer rejected inputs: {exc}")
            attachment, report = _fit([], [])
        if reason:
            self._node.get_logger().warning(
                f"vision attachment fell back to GRIPPER_FORCE: {reason}"
            )
        self._node.get_logger().info(
            f"vision attachment: accepted={report.accepted} rejections={list(report.rejections)} "
            f"candidate={report.candidate_index}/{report.candidate_count} "
            f"points={report.point_count} extents_m={report.extents_m} "
            f"depth_valid={report.depth_valid_fraction:.2f} "
            f"mask_score_advisory={report.mask_score_advisory:.3f} (advisory only)"
        )
        self._publish_attachment([attachment])
        self._release_barrier()

    def _release_barrier(self) -> None:
        """Reopen the barrier and let the node publish its deferred tick."""
        self._pending = False
        if callable(self._on_perception_ready):
            self._on_perception_ready()

    def _cancel_deadline(self) -> None:
        """Cancel and drop the one-shot deadline timer, if any."""
        if self._deadline_timer is None:
            return
        self._deadline_timer.cancel()
        self._node.destroy_timer(self._deadline_timer)
        self._deadline_timer = None

    # ── inputs ───────────────────────────────────────────────────────────────

    def _gather_context(self) -> _PromptContext | str:
        """Collect depth, transforms and prompt geometry, or name what is missing.

        Returns the ``(t_link_from_cam, tcp_in_link, negative_points)`` tuple, or
        a typed reason string when a precondition is unmet. The caller turns
        either into an attachment; neither turns into a stall.
        """
        if self._depth is None:
            return f"ROSPerceptionStale: no depth frame yet on {self._depth_topic()!r}"
        camera_frame = self._camera_frame()
        t_link_from_cam = self._lookup(self._producer.attach_link, camera_frame)
        if t_link_from_cam is None:
            return f"ROSPerceptionStale: no tf2 {self._producer.attach_link} <- {camera_frame}"
        t_link_from_tcp = self._lookup(self._producer.attach_link, self._tcp_frame)
        if t_link_from_tcp is None:
            return f"ROSPerceptionStale: no tf2 {self._producer.attach_link} <- {self._tcp_frame}"
        tcp = (
            float(t_link_from_tcp[0, 3]),
            float(t_link_from_tcp[1, 3]),
            float(t_link_from_tcp[2, 3]),
        )
        negatives: list[tuple[float, float, float]] = []
        for frame in self._config.jaw_tip_frames:
            tip = self._lookup(self._producer.attach_link, frame)
            if tip is not None:
                negatives.append((float(tip[0, 3]), float(tip[1, 3]), float(tip[2, 3])))
        return t_link_from_cam, tcp, negatives

    def _on_depth(self, msg: Any) -> None:
        """Cache the newest metric-depth raster for the attach camera."""
        from openral_hal.depth_cloud import depth_grid_from_image

        try:
            grid = depth_grid_from_image(msg)
        except ROSConfigError as exc:
            self._node.get_logger().warning(f"vision attachment depth dropped: {exc}")
            return
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._depth = (grid, stamp_ns)

    def _lookup(self, target: str, source: str) -> NDArray[np.float64] | None:
        """Latest ``target <- source`` transform as a 4x4, or ``None``."""
        if target == source:
            return np.eye(4, dtype=np.float64)
        import rclpy
        from openral_core.geometry import homogeneous_from_quat_xyz

        try:
            tf = self._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception as exc:  # tf2 raises several distinct lookup errors
            self._node.get_logger().debug(f"tf {target} <- {source} unavailable: {exc}")
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _publish_attachment(self, objects: Sequence[Any]) -> None:
        """Publish one authoritative attachment snapshot at a fresh revision."""
        if self._attachment_pub is None:
            return
        from openral_msgs.msg import (
            AttachedCollisionObject,
            AttachedCollisionPrimitive,
            AttachmentState,
        )

        self._revision += 1
        msg = AttachmentState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.revision = self._revision
        for obj in objects:
            item = AttachedCollisionObject()
            obj.fill_idl(item, primitive_factory=AttachedCollisionPrimitive)
            msg.objects.append(item)
        self._attachment_pub.publish(msg)

    # ── manifest resolution ──────────────────────────────────────────────────

    def _sensors(self) -> list[Any]:
        """Every SensorSpec on the manifest, standalone and bundled."""
        specs = list(self._description.sensors)
        for bundle in self._description.sensor_bundles:
            specs.extend(bundle.sensors)
        return specs

    def _resolve_camera(self) -> tuple[str, IntrinsicsPinhole]:
        """Resolve the attach camera's id and intrinsics from the manifest."""
        specs = [spec for spec in self._sensors() if spec.intrinsics is not None]
        if self._config.camera:
            specs = [spec for spec in specs if spec.name == self._config.camera]
            if not specs:
                raise ROSConfigError(
                    f"vision attachment: camera {self._config.camera!r} has no SensorSpec with "
                    f"intrinsics on {self._description.name!r}."
                )
        if not specs:
            raise ROSConfigError(
                f"vision attachment needs a camera SensorSpec with intrinsics on "
                f"{self._description.name!r}; none declares any."
            )
        return str(specs[0].name), specs[0].intrinsics

    def _camera_frame(self) -> str:
        """tf2 optical frame of the attach camera."""
        for spec in self._sensors():
            if spec.name == self._camera:
                return str(spec.frame_id)
        raise ROSConfigError(f"vision attachment: camera {self._camera!r} vanished from manifest.")

    def _resolve_tcp_frame(self) -> str:
        """Default TCP frame: the gripper joint's moving-jaw child link.

        An approximation, and named as one: no schema field carries a calibrated
        tool center point today, and the moving jaw is the closest frame every
        manifest already declares. ``VisionAttachmentConfig.tcp_frame`` overrides
        it on a robot that knows better.
        """
        joints = [joint for joint in self._description.joints if joint.role == "gripper"]
        if not joints:
            raise ROSConfigError(
                "vision attachment needs a role='gripper' joint to derive the TCP frame."
            )
        return str(joints[0].child_link)

    def _depth_topic(self) -> str:
        """Configured depth topic, or the conventional per-camera default."""
        return self._config.depth_topic or f"/openral/cameras/{self._camera}/depth"
