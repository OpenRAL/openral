"""HAL-side client for the vision attachment-evidence path.

ROS wiring that turns ``_grasp_trigger`` events into
``openral_msgs/srv/SegmentInView`` calls, feeds the replies to
``VisionAttachmentEvidenceProducer``, and publishes the resulting
``AttachmentState`` the safety kernel already consumes.

The real-hardware sibling of the attachment leg in
``sim_sensor_bridge``, which reads MuJoCo ground truth. Enable it
explicitly (``vision_attachment_enabled``); the two must not both drive
``/openral/attachment_state``.

**Torch-free, by construction.** Everything model-shaped lives behind
the service, in ``openral_perception_ros.segmenter_node``. This module
imports numpy, rclpy and ``openral_core`` — never ``torch``, never
``transformers``, and never the runner's segmenter backend.

The deferred-ack barrier: the HAL already holds the ``action_applied``
acknowledgement of a grouped tick until attached-payload perception
settles (``attachment_action_ack_ready`` /
``_on_attachment_perception_ready``). In the simulator that wait is for
a transparent depth frame — about 100 ms. Segmentation rides inside
that same wait: a warmed SAM 2.1 call was measured at ~53 ms on the
reference GPU, so on that host the barrier does not grow at all. This
bridge exposes the same ``attachment_action_ack_ready`` shape, so the
node holds the tick for it exactly as it does for the simulator's depth
frames.

Three properties of that wait are non-negotiable:

* **It is bounded.** Every request carries a deadline
  (``VisionAttachmentConfig.deadline_s``). When it expires the
  attachment is resolved from what is in hand — nothing — rather than
  waiting longer. A robot that stalls because a perception node is
  wedged is a worse failure than a conservatively-shaped payload.
* **It never skips.** A timeout, a service failure, a missing depth
  frame and a missing transform all end in the producer's conservative
  jaw-span box, stamped ``AttachmentEvidenceKind.GRIPPER_FORCE`` at low
  confidence. Something is in the jaws either way; the collision
  checker must see *some* geometry.
* **It is visible.** Every fallback logs its typed reason and every
  attachment logs the gate report (CLAUDE.md §1.4). Nothing here
  degrades silently.

The kernel's attached-payload check fails closed on a snapshot it has never
heard (``attachment_stamp_ns == 0``) or one older than its deadline, so the
set is republished on a 0.2 s heartbeat — the same period as the simulator
bridge's. That heartbeat is a claim about the jaws, so it is only made while
the claim has evidence behind it: every gripper's effort channel must have
reported within ``VisionAttachmentConfig.evidence_timeout_s``, for at least
``GraspTriggerConfig.consecutive_ticks`` samples in a row, with no grasp
being resolved. A dead effort channel therefore ages into a kernel drop, never
into a stale "nothing attached".
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import CameraTopicKind, camera_topic
from openral_core.exceptions import ROSConfigError

from openral_hal._grasp_trigger import (
    GraspEvent,
    GraspTriggerConfig,
    GripperEffortTrigger,
    gripper_joints,
)
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


@dataclass
class _GripperLeg:
    """One gripper's trigger, producer, and in-flight segmentation state.

    A bimanual robot carries one leg per ``role: gripper`` joint; the two hands
    grasp independently, and each leg owns its own slot in the published
    attachment set.

    Attributes:
        joint_name: The gripper joint this leg watches.
        trigger: Effort trigger on that joint.
        producer: Evidence producer attached to that joint's parent link.
        object_id: Attachment identity, unique across legs.
        tcp_frame: tf2 frame to look the TCP up from, or empty to use
            ``tcp_in_link`` directly.
        tcp_in_link: TCP in the attach link — the gripper joint's
            ``origin_xyz``, i.e. the jaw child link's origin in the parent link.
            Needs no TF, so it holds even where the attach link is missing from
            the published tree (the vendored OpenArm URDF has no ``link7``).
        attachment: What this leg currently holds, or ``None``.
        inflight: The outstanding ``SegmentInView`` future, if any.
        deadline_timer: The one-shot deadline timer for that future, if any.
        pending: Whether this leg is holding the ack barrier.
    """

    joint_name: str
    trigger: GripperEffortTrigger
    producer: VisionAttachmentEvidenceProducer
    object_id: str
    tcp_frame: str
    tcp_in_link: tuple[float, float, float]
    attachment: Any = None
    inflight: Any = None
    deadline_timer: Any = None
    pending: bool = False


#: Heartbeat period, seconds — the simulator bridge's attachment heartbeat period.
_HEARTBEAT_PERIOD_S = 0.2


class _EffortEvidence:
    """Is every gripper's effort channel alive right now? — the heartbeat's gate.

    Pure bookkeeping on a caller-supplied monotonic clock. A sample counts only
    when *every* leg read an effort value from it; one missing value, or a gap
    longer than the timeout, restarts the count, because a heartbeat claims
    something about every hand at once and a channel that just came back has
    not yet shown it is steady.

    Args:
        required_samples: Consecutive complete samples before evidence is live.
        timeout_s: How old the newest complete sample may be.

    Example:
        >>> evidence = _EffortEvidence(required_samples=2, timeout_s=0.5)
        >>> evidence.observe(complete=True, now_s=0.0)
        >>> evidence.live(now_s=0.0)
        False
        >>> evidence.observe(complete=True, now_s=0.1)
        >>> evidence.live(now_s=0.5), evidence.live(now_s=0.7)
        (True, False)
    """

    def __init__(self, *, required_samples: int, timeout_s: float) -> None:
        """Start with no evidence."""
        self._required = required_samples
        self._timeout_s = timeout_s
        self._streak = 0
        self._last_s: float | None = None

    def observe(self, *, complete: bool, now_s: float) -> None:
        """Fold one joint-state sample in."""
        stale = self._last_s is not None and now_s - self._last_s > self._timeout_s
        if not complete or stale:
            self._streak = 0
        if not complete:
            return
        self._streak += 1
        self._last_s = now_s

    def live(self, *, now_s: float) -> bool:
        """Whether enough fresh, complete samples back a claim about the jaws."""
        return (
            self._streak >= self._required
            and self._last_s is not None
            and now_s - self._last_s <= self._timeout_s
        )


#: What one grasp needs from TF and the manifest before it can prompt:
#: ``(t_link_from_cam, tcp_in_link, negative_points_in_link)``.
_PromptContext = tuple[
    "NDArray[np.float64]",
    tuple[float, float, float],
    list[tuple[float, float, float]],
]


@dataclass(frozen=True)
class VisionAttachmentConfig:
    """Wiring for ``VisionAttachmentBridge``.

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
        tcp_frame: tf2 frame of the tool center point. Empty resolves, on a
            single-gripper robot, to the gripper joint's ``child_link`` — the
            moving jaw — looked up through tf2, and on a multi-gripper robot to
            each gripper joint's ``origin_xyz`` (the same jaw link origin,
            read from the manifest without TF). A robot with a calibrated
            mid-jaw frame should name it here rather than accept the
            approximation. Single-gripper only: set on a robot with several
            gripper joints it is a ``ROSConfigError``.
        jaw_tip_frames: Optional tf2 frames used as **negative** prompt points,
            pushing the mask off the gripper fingers. Empty = none; the single
            positive point at the TCP is what makes the prompt meaningful.
            Single-gripper only, like ``tcp_frame``.
        object_id: Identity stamped on the attachment, suffixed
            ``:<gripper joint>`` so each hand's slot is unique. Vision cannot
            name what it segmented, so this is a stable slot ("the thing in the
            jaws"), not a recognition result.
        tf_frames: Manifest link name -> tf2 frame name, for robots whose
            published TF tree spells a link differently from the manifest. Used
            only for tf2 lookups (attach link <- camera / TCP / jaw tips) and as
            the ``SegmentInView`` request frame; the published attachment keeps
            the manifest link, because the kernel's collision model is keyed on
            manifest links. Every entry must be a **proven identity** — the
            same rigid body with the same origin — never a nearby frame. The
            OpenArm case: the manifest's ``openarm_<side>_link7`` is the body
            joint7 drives, which the MJCF and the vendor URDF the real cell
            publishes both call ``openarm_<side>_ee_base_link`` (origin at
            joint7). Empty = look every link up by its manifest name.
        evidence_timeout_s: How old the newest joint-state sample carrying an
            effort value for every gripper may be before the attachment
            heartbeat stops. *Calibration point*, ``0.5`` s — several HAL read
            ticks at any rate the cell runs, and well inside the kernel's
            attached-collision deadline, so a dead effort channel surfaces as a
            kernel drop rather than as a stale "nothing attached".
    """

    camera: str = ""
    depth_topic: str = ""
    service_name: str = DEFAULT_SEGMENT_SERVICE
    deadline_s: float = 0.25
    tcp_frame: str = ""
    jaw_tip_frames: tuple[str, ...] = ()
    object_id: str = "grasped_payload"
    tf_frames: Mapping[str, str] = field(default_factory=dict)
    evidence_timeout_s: float = 0.5


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
        The ``SegmentOutcome``.

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

    One leg per ``role: gripper`` joint: each has its own effort trigger,
    evidence producer, and in-flight request, and the barrier stays shut while
    any leg is pending. Every publish carries the union of every leg's current
    attachment, so one hand's grasp never erases the other's payload.

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
            trigger (no gripper joints, no effort limit, no camera intrinsics),
            or ``tcp_frame`` / ``jaw_tip_frames`` is set on a robot with more
            than one gripper joint.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> bridge = VisionAttachmentBridge(
        ...     None, d, config=VisionAttachmentConfig(camera="head_zed")
        ... )
        >>> bridge.gripper_joint_names
        ('left_gripper', 'right_gripper')
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
        self._legs = self._build_legs(gate_config, trigger_config)
        self._camera, self._intrinsics = self._resolve_camera()
        links = {j.parent_link for j in description.joints} | {
            j.child_link for j in description.joints
        }
        unknown = sorted(set(self._config.tf_frames) - links)
        if unknown:
            raise ROSConfigError(
                f"vision attachment: tf_frames names {unknown}, which are not links of "
                f"{description.name!r}."
            )

        self._depth: tuple[NDArray[np.float64], int] | None = None
        self._depth_sub: Any = None
        self._client: Any = None
        self._attachment_pub: Any = None
        self._tf_buffer: Any = None
        self._tf_listener: Any = None
        self._heartbeat_timer: Any = None
        self._heartbeat_open: bool | None = None
        self._evidence = _EffortEvidence(
            required_samples=(trigger_config or GraspTriggerConfig()).consecutive_ticks,
            timeout_s=self._config.evidence_timeout_s,
        )
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
        # The bridge is rebuilt on every HAL activate, and the aggregator rejects
        # a revision that moves backwards, so a counter restarting at 0 would
        # wedge the second activation. Seed from the node clock instead: host
        # wall time on a real cell, monotonic within a launch under sim time;
        # +1 per event after that. Nothing persists the last revision across
        # processes — the upgrade path, if wall time ever proves unfit, is to
        # read the latched topic's last revision before the first publish.
        self._revision = int(self._node.get_clock().now().nanoseconds)
        self._heartbeat_timer = self._node.create_timer(_HEARTBEAT_PERIOD_S, self._heartbeat)
        for leg in self._legs:
            self._node.get_logger().info(
                f"vision attachment bridge: gripper={leg.joint_name!r} camera={self._camera!r} "
                f"depth={self._depth_topic()!r} service={self._config.service_name!r} "
                f"attach_link={leg.producer.attach_link!r} "
                f"tcp={leg.tcp_frame or leg.tcp_in_link!r} "
                f"deadline={self._config.deadline_s:.3f}s "
                f"effort_thresholds_n={leg.trigger.thresholds_n}"
            )

    def teardown(self) -> None:
        """Destroy every ROS entity; idempotent, and never leaves the barrier shut."""
        for leg in self._legs:
            self._cancel_deadline(leg)
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._node.destroy_timer(self._heartbeat_timer)
            self._heartbeat_timer = None
        if self._depth_sub is not None:
            self._node.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        if self._attachment_pub is not None:
            self._node.destroy_publisher(self._attachment_pub)
            self._attachment_pub = None
        self._client = None
        self._tf_listener = None
        self._tf_buffer = None
        # A bridge torn down mid-flight must not leave the node deferring an
        # acknowledgement forever. Clearing the flag is not enough now that the
        # node re-checks every barrier holder before releasing: a swallowed
        # notify is only safe because each holder re-issues one when it settles,
        # so the holder that goes away has to issue its last one here.
        for leg in self._legs:
            leg.inflight = None
            if leg.pending:
                self._release_barrier(leg)

    # ── barrier ──────────────────────────────────────────────────────────────

    def attachment_action_ack_ready(self) -> bool:
        """Whether attached-payload perception has settled for this tick.

        The same shape ``SimSensorBridge``
        exposes, so the node's deferred-ack path treats both identically.
        """
        return not any(leg.pending for leg in self._legs)

    def tf_frame(self, link: str) -> str:
        """The tf2 frame a manifest link is looked up as (``tf_frames``, else itself).

        Args:
            link: A manifest link name.

        Returns:
            The tf2 frame name.
        """
        return self._config.tf_frames.get(link, link)

    @property
    def gripper_joint_names(self) -> tuple[str, ...]:
        """The gripper joints this bridge watches, one leg each, in manifest order."""
        return tuple(leg.joint_name for leg in self._legs)

    # ── trigger ──────────────────────────────────────────────────────────────

    def observe_joint_state(self, state: JointState) -> None:
        """Fold one HAL read into every leg's grasp trigger and act on transitions.

        Also the heartbeat's liveness evidence: a sample counts only when every
        leg read an effort value from it.

        Args:
            state: The tick's joint state, straight from the HAL.
        """
        complete = True
        for leg in self._legs:
            missing_before = leg.trigger.missing_effort_ticks
            event = leg.trigger.update(state)
            complete = complete and leg.trigger.missing_effort_ticks == missing_before
            if event is None:
                continue
            if event is GraspEvent.DETACH:
                self._node.get_logger().info(
                    f"grasp trigger {leg.joint_name}: DETACH — dropping its attachment"
                )
                leg.attachment = None
                self._publish_attachment()
                continue
            self._node.get_logger().info(
                f"grasp trigger {leg.joint_name}: {event.value.upper()} — holding the action "
                "ack for segmentation"
            )
            self._begin_segmentation(leg, stamp_ns=int(state.stamp_ns))
        self._evidence.observe(complete=complete, now_s=time.monotonic())

    @property
    def missing_effort_ticks(self) -> int:
        """Ticks whose gripper carried no effort value, summed over every leg.

        A driver-health signal.
        """
        return sum(leg.trigger.missing_effort_ticks for leg in self._legs)

    # ── segmentation round trip ──────────────────────────────────────────────

    def _begin_segmentation(self, leg: _GripperLeg, *, stamp_ns: int) -> None:
        """Close the barrier and dispatch one bounded SegmentInView request."""
        leg.pending = True
        context = self._gather_context(leg)
        if isinstance(context, str):  # a typed precondition failure
            self._finish(leg, stamp_ns=stamp_ns, masks=[], scores=[], reason=context)
            return
        t_link_from_cam, tcp_in_link, negatives = context
        if self._client is None or not self._client.service_is_ready():
            self._finish(
                leg,
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
        request.frame_id = self.tf_frame(leg.producer.attach_link)
        request.tcp_point = Point(x=tcp_in_link[0], y=tcp_in_link[1], z=tcp_in_link[2])
        request.negative_points = [Point(x=x, y=y, z=z) for x, y, z in negatives]

        future = self._client.call_async(request)
        leg.inflight = future
        future.add_done_callback(
            lambda fut: self._on_reply(
                leg,
                fut,
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
        )
        leg.deadline_timer = self._node.create_timer(
            self._config.deadline_s,
            lambda: self._on_deadline(
                leg,
                stamp_ns=stamp_ns,
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            ),
        )

    def _on_reply(
        self,
        leg: _GripperLeg,
        future: Any,
        *,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve the attachment from a SegmentInView reply (or its exception)."""
        if future is not leg.inflight:
            return  # already resolved by the deadline; this reply is late
        self._cancel_deadline(leg)
        leg.inflight = None
        try:
            response = future.result()
        except Exception as exc:  # a service exception must degrade, not propagate
            self._finish(
                leg,
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason=f"ROSDispatchUnavailable: SegmentInView raised {type(exc).__name__}: {exc}",
                t_link_from_cam=t_link_from_cam,
                tcp_in_link=tcp_in_link,
            )
            return
        if response is None:
            # A cancelled future resolves to None without raising. Belt and
            # braces behind the in-flight guard above: an AttributeError escaping
            # here would be swallowed by rclpy's callback wrapper and leave the
            # barrier held with no explanation anywhere.
            self._finish(
                leg,
                stamp_ns=stamp_ns,
                masks=[],
                scores=[],
                reason="ROSDispatchUnavailable: SegmentInView future resolved with no response",
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
            leg,
            stamp_ns=stamp_ns,
            masks=masks,
            scores=[float(s) for s in response.mask_scores_advisory] if outcome.use_masks else [],
            reason=outcome.reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _on_deadline(
        self,
        leg: _GripperLeg,
        *,
        stamp_ns: int,
        t_link_from_cam: NDArray[np.float64],
        tcp_in_link: tuple[float, float, float],
    ) -> None:
        """Resolve conservatively when the segmenter overran its budget."""
        self._cancel_deadline(leg)
        inflight = leg.inflight
        if inflight is None:
            return
        # Drop the handle BEFORE cancelling. ``rclpy.Future.cancel`` completes
        # the future, which synchronously runs the done callback we registered —
        # and that callback's "is this still the in-flight one?" guard is the
        # only thing keeping it from resolving a reply that never arrived.
        leg.inflight = None
        inflight.cancel()
        outcome = resolve_segment_outcome(timed_out=True, ok=False, failure_reason="", mask_count=0)
        self._finish(
            leg,
            stamp_ns=stamp_ns,
            masks=[],
            scores=[],
            reason=outcome.reason,
            t_link_from_cam=t_link_from_cam,
            tcp_in_link=tcp_in_link,
        )

    def _finish(
        self,
        leg: _GripperLeg,
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
            return leg.producer.on_grasp(
                masks=candidates,
                depth_m=depth,
                intrinsics=self._intrinsics,
                t_link_from_cam=transform,
                tcp_in_link=tcp_in_link,
                object_id=leg.object_id,
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
                f"vision attachment {leg.joint_name} fell back to GRIPPER_FORCE: {reason}"
            )
        self._node.get_logger().info(
            f"vision attachment {leg.joint_name}: accepted={report.accepted} "
            f"rejections={list(report.rejections)} "
            f"candidate={report.candidate_index}/{report.candidate_count} "
            f"points={report.point_count} extents_m={report.extents_m} "
            f"depth_valid={report.depth_valid_fraction:.2f} "
            f"mask_score_advisory={report.mask_score_advisory:.3f} (advisory only)"
        )
        leg.attachment = attachment
        self._publish_attachment()
        self._release_barrier(leg)

    def _release_barrier(self, leg: _GripperLeg) -> None:
        """Reopen this leg's hold and let the node re-check the barrier."""
        leg.pending = False
        if callable(self._on_perception_ready):
            self._on_perception_ready()

    def _cancel_deadline(self, leg: _GripperLeg) -> None:
        """Cancel and drop a leg's one-shot deadline timer, if any."""
        if leg.deadline_timer is None:
            return
        leg.deadline_timer.cancel()
        self._node.destroy_timer(leg.deadline_timer)
        leg.deadline_timer = None

    # ── inputs ───────────────────────────────────────────────────────────────

    def _gather_context(self, leg: _GripperLeg) -> _PromptContext | str:
        """Collect depth, transforms and prompt geometry, or name what is missing.

        Returns the ``(t_link_from_cam, tcp_in_link, negative_points)`` tuple, or
        a typed reason string when a precondition is unmet. The caller turns
        either into an attachment; neither turns into a stall.
        """
        if self._depth is None:
            return f"ROSPerceptionStale: no depth frame yet on {self._depth_topic()!r}"
        link = self.tf_frame(leg.producer.attach_link)
        camera_frame = self._camera_frame()
        t_link_from_cam = self._lookup(link, camera_frame)
        if t_link_from_cam is None:
            return f"ROSPerceptionStale: no tf2 {link} <- {camera_frame}"
        tcp = leg.tcp_in_link
        if leg.tcp_frame:
            t_link_from_tcp = self._lookup(link, leg.tcp_frame)
            if t_link_from_tcp is None:
                return f"ROSPerceptionStale: no tf2 {link} <- {leg.tcp_frame}"
            tcp = (
                float(t_link_from_tcp[0, 3]),
                float(t_link_from_tcp[1, 3]),
                float(t_link_from_tcp[2, 3]),
            )
        negatives: list[tuple[float, float, float]] = []
        for frame in self._config.jaw_tip_frames:
            tip = self._lookup(link, frame)
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

    def _publish_attachment(self) -> None:
        """Publish every leg's current attachment as one snapshot at a fresh revision."""
        if self._attachment_pub is None:
            return
        self._revision += 1
        self._publish_snapshot()

    def _heartbeat(self) -> None:
        """Republish the current set at the current revision, while evidence backs it.

        Fail-closed: no live effort evidence, a grasp still being resolved, or a
        trigger that believes the jaws are loaded with no attachment yet each
        stop the heartbeat, so the kernel's snapshot ages into a drop instead of
        a stale claim. Each open/close transition is logged once.
        """
        if self._attachment_pub is None:
            return
        if not self._evidence.live(now_s=time.monotonic()):
            reason = "no live effort evidence from every gripper"
        elif any(leg.pending for leg in self._legs):
            reason = "a grasp is being resolved"
        elif any(leg.trigger.attached != (leg.attachment is not None) for leg in self._legs):
            reason = "a trigger and its attachment disagree"
        else:
            reason = ""
        is_open = not reason
        if is_open != self._heartbeat_open:
            self._heartbeat_open = is_open
            if is_open:
                self._node.get_logger().info(
                    "vision attachment heartbeat: publishing — effort evidence is live"
                )
            else:
                self._node.get_logger().warning(
                    f"vision attachment heartbeat: withheld — {reason}; the kernel's "
                    "attachment snapshot will age out"
                )
        if is_open:
            self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        """Publish the union of the legs' attachments at the current revision, stamped now."""
        objects = [leg.attachment for leg in self._legs if leg.attachment is not None]
        from openral_msgs.msg import (
            AttachedCollisionObject,
            AttachedCollisionPrimitive,
            AttachmentState,
        )

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

    def _build_legs(
        self,
        gate_config: VisionGateConfig | None,
        trigger_config: GraspTriggerConfig | None,
    ) -> list[_GripperLeg]:
        """One leg per gripper joint, with its TCP resolved.

        The TCP is the gripper's moving-jaw child link — an approximation, and
        named as one: no schema field carries a calibrated tool center point
        today, and the jaw link is the closest frame every manifest declares.
        With one gripper it is looked up through tf2 (``tcp_frame``, defaulting
        to the child link) exactly as before. With several, it is the joint's
        ``origin_xyz`` — the child link origin in the attach link, read from
        the manifest — because the attach link may be absent from the published
        TF tree, and per-hand frame overrides are not supported.
        """
        joints = gripper_joints(self._description)
        if len(joints) > 1 and (self._config.tcp_frame or self._config.jaw_tip_frames):
            raise ROSConfigError(
                f"vision attachment: tcp_frame / jaw_tip_frames are single-gripper settings, "
                f"but {self._description.name!r} has gripper joints "
                f"{[joint.name for joint in joints]}."
            )
        return [
            _GripperLeg(
                joint_name=joint.name,
                trigger=GripperEffortTrigger(
                    self._description, joint_name=joint.name, config=trigger_config
                ),
                producer=VisionAttachmentEvidenceProducer(
                    self._description, gripper_joint=joint.name, config=gate_config
                ),
                object_id=f"{self._config.object_id}:{joint.name}",
                tcp_frame=(self._config.tcp_frame or joint.child_link) if len(joints) == 1 else "",
                tcp_in_link=joint.origin_xyz,
            )
            for joint in joints
        ]

    def _depth_topic(self) -> str:
        """Configured depth topic, or the conventional per-camera default."""
        return self._config.depth_topic or camera_topic(self._camera, CameraTopicKind.DEPTH_IMAGE)
