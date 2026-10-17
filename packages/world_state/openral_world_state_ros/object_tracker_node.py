r"""openral_world_state object-tracker lifecycle node (issue #418, WP D1).

Wraps ``openral_world_state.object_tracker.TrackerSession`` (all decisions live there) and
owns the wire:

* **Inputs.** Per configured camera: metric depth + its ``CameraInfo`` + the robot
  self-filter's ``robot_mask`` (``camera_topic(<cam>, DEPTH_IMAGE | DEPTH_CAMERA_INFO |
  ROBOT_MASK)``); ``/openral/world_voxels`` (grid + ``source_stamp``); ``/joint_states``;
  tf2 (``T_base<-cam`` at the depth stamp); the live ``/openral/grasp_declaration`` /
  ``/openral/place_declaration``; ``/openral/grasp_events`` (the HAL trigger's ATTACH /
  REGRASP / DETACH); in bind / on, ``/openral/safe_action`` (the approved gripper command: the
  tracker derives its own stalled-jaw ATTACH from it, ``JawStallDetector``).
* **Frames.** A depth frame waits (wall) at most ``mask_tf_wait_s`` for the mask with its
  exact stamp and for tf at that stamp; missing either after that, it is fed for
  **ageing only** (design step 0) — never fitted.
* **Requests.** ``TrackerOutput.requests`` (at most one in flight; the core rate-caps) go
  to ``LocateInView`` (``locate_service``; empty = no detector, answered empty) or
  ``SegmentInView`` (``segment_service``: box prompt plus the prompt pixel back-projected
  through the frame's own depth, in the depth optical frame); the answer is fed back
  with the frame it was issued on.
* **Outputs.** ``/openral/tracked_objects`` (``TrackedObjectSet``; RELIABLE, VOLATILE,
  KEEP_LAST 1) on every change and as a ``heartbeat_hz`` heartbeat. In ``bind`` / ``on``
  the node also owns ``/openral/attachment_state`` (RELIABLE, TRANSIENT_LOCAL, KEEP_LAST
  1): every hand's binding plus the live declarations relayed **without** a target
  region (dispatch never measures one; the HAL producer is gated off in these modes) but
  with the grasp declaration's ``workspace_box`` as ``workspace_region`` while the
  tracker's own grid confirms it (``TrackerSession.workspace_region``, the producer's
  rule), heartbeated only while joint states are fresh. In
  ``shadow`` nothing but ``tracked_objects`` and the logged predicted bindings leaves
  the node.
* **Revisions** are seeded from wall time (ns) at every activate plus the store's own
  counter, so they never regress across a node restart or re-activation (the kernel
  refuses a regressing ``tracked_objects`` revision, and the aggregator a regressing
  attachment revision).

``bind`` / ``on`` are sim-only until Safety-WG review: the node refuses them with
``hal_mode:=real`` (the CLI and the launch refuse first).

Usage::

    ros2 run openral_world_state object_tracker_node --ros-args \\
        -p robot_yaml:=robots/so101_follower/robot.yaml -p mode:=shadow \\
        -p cameras:="['top']" -p grounding_wait_s:=8.0 -p cell_m:=0.015
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, ObjectQueryEvidence, ObjectsMetadata
from openral_core.exceptions import ROSConfigError
from openral_world_state.object_tracker.grounding import QueryEscalation, StandingQuery

log = logging.getLogger(__name__)
slog = structlog.get_logger(__name__)

__all__ = [
    "backproject_pixel",
    "detector_boxes",
    "escalation_evidence",
    "image_array",
    "intrinsics_from_info",
    "main",
    "pick_masks",
]

#: ``SegmentInView`` server the HAL's vision bridge also calls.
DEFAULT_SEGMENT_SERVICE = "/openral/perception/segment_in_view"
_NS = 1_000_000_000
#: Robot masks kept per camera while their depth frame waits.
_MASKS_KEPT = 8


def image_array(msg: Any) -> NDArray[Any]:  # reason: duck-typed sensor_msgs/Image
    """Decode a ``sensor_msgs/Image`` into ``(H, W)``: depth in metres, or ``mono8``.

    ``32FC1`` (metres) and ``16UC1`` (millimetres) become float64 metres (``0`` stays the
    REP-118 "no return"); ``mono8`` stays ``uint8``. Row padding (``step``) is honoured.

    Raises:
        ROSConfigError: On another encoding or a payload shorter than ``step * height``.

    Example:
        >>> from types import SimpleNamespace
        >>> m = SimpleNamespace(
        ...     encoding="16UC1", height=1, width=2, step=4, data=bytes([232, 3, 0, 0])
        ... )
        >>> image_array(m).tolist()
        [[1.0, 0.0]]
    """
    kinds = {"32FC1": ("<f4", 1.0), "16UC1": ("<u2", 1e-3), "mono8": ("u1", None)}
    kind = kinds.get(str(msg.encoding))
    if kind is None:
        raise ROSConfigError(f"image_array: unsupported encoding {msg.encoding!r}.")
    dtype, scale = kind
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if raw.size < step * h:
        raise ROSConfigError(f"image_array: {raw.size} bytes for {h} rows of {step}.")
    item = np.dtype(dtype).itemsize
    rows = raw[: step * h].reshape(h, step)[:, : w * item]
    arr = np.ascontiguousarray(rows).view(dtype).reshape(h, w)
    return arr.copy() if scale is None else arr.astype(np.float64) * scale


def intrinsics_from_info(msg: Any) -> IntrinsicsPinhole:  # reason: duck-typed CameraInfo
    """The driver's live pinhole ``K`` (never the manifest's nominal one).

    Example:
        >>> from types import SimpleNamespace
        >>> k = [500.0, 0, 320, 0, 500.0, 240, 0, 0, 1]
        >>> intrinsics_from_info(SimpleNamespace(width=640, height=480, k=k)).cx
        320.0
    """
    k = [float(v) for v in msg.k]
    return IntrinsicsPinhole(
        width=int(msg.width), height=int(msg.height), fx=k[0], fy=k[4], cx=k[2], cy=k[5]
    )


def _bbox_iou(mask: NDArray[np.bool_], box: tuple[float, float, float, float]) -> float:
    """IoU of ``mask``'s pixel bounding rectangle with ``box`` (``x0, y0, x1, y1``)."""
    rows, cols = np.nonzero(mask)
    mx0, my0, mx1, my1 = cols.min(), rows.min(), cols.max() + 1, rows.max() + 1
    ix = max(0.0, min(mx1, box[2]) - max(mx0, box[0]))
    iy = max(0.0, min(my1, box[3]) - max(my0, box[1]))
    inter = ix * iy
    union = (mx1 - mx0) * (my1 - my0) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def pick_masks(
    masks: Sequence[NDArray[np.bool_]],
    box_xyxy: tuple[float, float, float, float] | None,
    within_xyxy: tuple[float, float, float, float] | None = None,
    *,
    inside_fraction: float = 0.9,
) -> list[NDArray[np.bool_]]:
    """The segmenter's qualifying nested candidates, best first, for a **containing** fit.

    With a box prompt (a detector's box, or a track's projected box) the candidates at
    least ``inside_fraction`` inside it, the one whose own rectangle matches the box best
    first: the detector already said how far the object reaches, and SAM's nested
    candidates are a subpart, the object and the object spilled onto what it stands on.
    Without one, the candidates inside ``within_xyxy`` (a map seed's rectangle) or all of
    them, largest first (a subpart under-contains). The tracker falls back to the next
    candidate only when one cannot be fitted or splits on the map. *Calibration point:*
    ``inside_fraction``.

    Example:
        >>> a = np.zeros((8, 8), bool)
        >>> a[3:5, 3:5] = True
        >>> b = np.ones((8, 8), bool)
        >>> [int(m.sum()) for m in pick_masks([a, b], None)]
        [64, 4]
        >>> [int(m.sum()) for m in pick_masks([a, b], None, (3, 3, 5, 5))]
        [4]
        >>> [int(m.sum()) for m in pick_masks([b, a], (2, 2, 6, 6))]
        [4]
    """
    ok = [m for m in masks if m.any()]
    bound = box_xyxy if box_xyxy is not None else within_xyxy
    if bound is not None:
        x0, y0, x1, y1 = (round(v) for v in bound)
        ok = [
            m
            for m in ok
            if m[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)].sum()
            >= inside_fraction * m.sum()
        ]
    if box_xyxy is not None:
        return sorted(ok, key=lambda m: -_bbox_iou(m, box_xyxy))
    return sorted(ok, key=lambda m: -int(m.sum()))


def backproject_pixel(
    depth_m: NDArray[np.float64], k: IntrinsicsPinhole, uv: tuple[float, float], *, win: int = 2
) -> tuple[float, float, float] | None:
    """The optical-frame point under pixel ``uv``: median valid depth in a ``2*win+1`` window.

    Example:
        >>> k = IntrinsicsPinhole(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
        >>> backproject_pixel(np.full((4, 4), 1.0), k, (3.0, 2.0))
        (0.5, 0.0, 1.0)
    """
    u, v = round(uv[0]), round(uv[1])
    h, w = depth_m.shape
    patch = depth_m[max(v - win, 0) : min(v + win + 1, h), max(u - win, 0) : min(u + win + 1, w)]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if not valid.size:
        return None
    z = float(np.median(valid))
    return ((uv[0] - k.cx) * z / k.fx, (uv[1] - k.cy) * z / k.fy, z)


def detector_scores(metadata_json: str) -> tuple[float, ...]:
    """``LocateInView``'s detection scores, parallel to ``detector_boxes``.

    Example:
        >>> md = ObjectsMetadata(
        ...     sensor_id="top",
        ...     model_id="m",
        ...     frame_width=64,
        ...     frame_height=48,
        ...     detections=[{"label": "box", "confidence": 0.2, "bbox_xyxy": (1, 2, 3, 4)}],
        ... )
        >>> detector_scores(md.model_dump_json())
        (0.2,)
    """
    if not metadata_json:
        return ()
    md = ObjectsMetadata.model_validate(json.loads(metadata_json))
    return tuple(float(d.confidence) for d in md.detections)


def detector_boxes(
    metadata_json: str, k: IntrinsicsPinhole
) -> tuple[tuple[float, float, float, float], ...]:
    """``LocateInView``'s ``ObjectsMetadata`` boxes, rescaled to the depth raster ``k``.

    Example:
        >>> md = ObjectsMetadata(
        ...     sensor_id="top",
        ...     model_id="m",
        ...     frame_width=640,
        ...     frame_height=480,
        ...     detections=[{"label": "orange", "confidence": 0.9, "bbox_xyxy": (64, 48, 128, 96)}],
        ... )
        >>> k = IntrinsicsPinhole(width=320, height=240, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        >>> detector_boxes(md.model_dump_json(), k)
        ((32.0, 24.0, 64.0, 48.0),)
    """
    if not metadata_json:
        return ()
    md = ObjectsMetadata.model_validate(json.loads(metadata_json))
    sx, sy = k.width / md.frame_width, k.height / md.frame_height
    return tuple(
        (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in (d.bbox_xyxy for d in md.detections)
    )


def _resize_nearest(mask: NDArray[np.bool_], shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """``mask`` on a ``shape`` raster by nearest neighbour (a registered RGB/depth pair)."""
    if mask.shape == shape:
        return mask
    rows = (np.arange(shape[0]) * mask.shape[0] // shape[0]).astype(int)
    cols = (np.arange(shape[1]) * mask.shape[1] // shape[1]).astype(int)
    return np.asarray(mask[np.ix_(rows, cols)], dtype=np.bool_)


@dataclass
class _Camera:
    """One tracked camera's latest inputs."""

    name: str
    info: IntrinsicsPinhole | None = None
    pending: tuple[Any, float] | None = None  # (depth msg, monotonic receive time)
    masks: dict[int, NDArray[np.uint8]] = field(default_factory=dict)


def escalation_evidence(
    escalation: QueryEscalation, query: StandingQuery, attempts: int
) -> ObjectQueryEvidence:
    """The typed failure evidence for one tracker escalation (issue #418 Addendum G).

    ``waited_s`` is the source time since the query's current search started; ``attempts``
    the detection requests the node issued for it.

    Example:
        >>> from openral_core import ObjectQuery, ObjectQueryStatus
        >>> q = StandingQuery(
        ...     "grasp:obj:orange@1",
        ...     "0",
        ...     ObjectQuery(label="orange", count="one", role="target"),
        ...     "obj:orange@1",
        ...     ObjectQueryStatus.EXHAUSTED,
        ...     since_ns=0,
        ... )
        >>> e = QueryEscalation("exhausted", q.goal_id, "0", "orange", None, 60 * _NS)
        >>> escalation_evidence(e, q, 9).waited_s
        60.0
    """
    return ObjectQueryEvidence(
        escalation=escalation.kind,
        goal_id=escalation.goal_id,
        query_id=escalation.query_id,
        label=escalation.label,
        role=query.query.role,
        waited_s=max(0.0, (escalation.stamp_ns - query.since_ns) / _NS),
        attempts=attempts,
        last_seen=escalation.last_seen,
    )


def _stamp_ns(stamp: Any) -> int:  # reason: builtin_interfaces/Time
    return int(stamp.sec) * _NS + int(stamp.nanosec)


def main() -> None:
    """Entry point for ``ros2 run openral_world_state object_tracker_node``."""
    import rclpy
    from openral_observability import configure_observability
    from rclpy.executors import ExternalShutdownException

    configure_observability(service_name="openral.object_tracker")
    rclpy.init()
    node = make_object_tracker_node()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def make_object_tracker_node() -> Any:  # reason: rclpy node, untyped
    """Build the lifecycle node (class defined here so the module imports without rclpy)."""
    from geometry_msgs.msg import Point
    from openral_core import (
        GraspDeclaration,
        ObjectQuery,
        PlaceDeclaration,
        RobotDescription,
        camera_topic,
    )
    from openral_core.geometry import homogeneous_from_quat_xyz
    from openral_core.perception_geometry import lattice_from_msg
    from openral_core.schemas import CameraTopicKind
    from openral_msgs.msg import ActionChunk as ActionChunkMsg
    from openral_msgs.msg import AttachedCollisionObject as AttachedMsg
    from openral_msgs.msg import AttachedCollisionPrimitive as PrimitiveMsg
    from openral_msgs.msg import AttachmentState as AttachmentStateMsg
    from openral_msgs.msg import GraspDeclaration as GraspDeclarationMsg
    from openral_msgs.msg import GraspEvent as GraspEventMsg
    from openral_msgs.msg import ObjectQuery as ObjectQueryMsg
    from openral_msgs.msg import OccupancyVoxels
    from openral_msgs.msg import PlaceDeclaration as PlaceDeclarationMsg
    from openral_msgs.msg import TrackedObject as TrackedObjectMsg
    from openral_msgs.msg import TrackedObjectSet as TrackedObjectSetMsg
    from openral_msgs.srv import LocateInView, SegmentInView
    from openral_observability import FailureBusPublisher, FailureSource, world_state_span
    from openral_observability import semconv as _semconv
    from openral_observability.failure_bus import (
        KIND_OBJECT_QUERY,
        SEVERITY_FAIL,
        SEVERITY_WARN,
    )
    from openral_world_state.object_tracker import (
        CameraFrame,
        TrackerRequest,
        TrackerResult,
    )
    from openral_world_state.object_tracker.session import (
        GraspEventKind,
        TrackerSession,
        TrackingMode,
    )
    from openral_world_state.object_tracker.tracker import TrackerParams
    from pydantic import TypeAdapter
    from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, Image, JointState

    sensor_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=5,
    )
    reliable_1 = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1
    )
    latched_1 = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1,
    )
    events_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )
    kinds = {
        GraspEventMsg.KIND_ATTACH: GraspEventKind.ATTACH,
        GraspEventMsg.KIND_REGRASP: GraspEventKind.REGRASP,
        GraspEventMsg.KIND_DETACH: GraspEventKind.DETACH,
    }

    class ObjectTrackerNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
        """The object-tracker lifecycle node (module docstring)."""

        def __init__(self) -> None:
            super().__init__("openral_object_tracker")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("mode", "shadow")
            self.declare_parameter("hal_mode", "sim")
            self.declare_parameter("cameras", [""])
            self.declare_parameter("cell_m", 0.015)
            self.declare_parameter("grounding_wait_s", 0.0)
            self.declare_parameter("grounding_required", False)
            self.declare_parameter("release_dwell_s", 0.3)
            # MUST equal the kernel's tracked_release_timeout_s and the voxel bridge's
            # released_clear_timeout_s (the launch passes one value to all three).
            self.declare_parameter("released_clear_timeout_s", 3.0)
            # The deploy's prompt-time ObjectQuery list (JSON), grounded at start.
            self.declare_parameter("object_queries_json", "")
            self.declare_parameter("segment_service", DEFAULT_SEGMENT_SERVICE)
            self.declare_parameter("locate_service", "")
            self.declare_parameter("mask_tf_wait_s", 0.3)
            self.declare_parameter("heartbeat_hz", 2.0)
            self.declare_parameter("process_hz", 15.0)
            self.declare_parameter("joint_states_topic", "/joint_states")
            self.declare_parameter("attachment_evidence_timeout_s", 1.0)
            # Addendum G active-search budget (source seconds): an unresolved query
            # escalates after search_start_s, is EXHAUSTED after search_max_s.
            self.declare_parameter("search_start_s", TrackerParams().search_start_s)
            self.declare_parameter("search_max_s", TrackerParams().search_max_s)
            self._failure_bus: Any = None
            self._attempts: dict[tuple[str, str], int] = {}
            self.session: TrackerSession | None = None
            self.mode = TrackingMode.SHADOW
            self._cams: dict[str, _Camera] = {}
            self._subs: list[Any] = []
            self._timers: list[Any] = []
            self._pub_tracked: Any = None
            self._pub_attach: Any = None
            self._segment: Any = None
            self._locate: Any = None
            self._tf_buffer: Any = None
            self._tf_listener: Any = None
            self._aliases: dict[str, str] = {}
            self._active = False
            self._rev_base = 0
            self._last_tracked_rev = -1
            self._attach_key: tuple[int, int, int] | None = None
            self._attach_rev = 0
            self._joint_rx_ns = 0
            self._joint_order: list[str] = []

        def _p(self, name: str) -> Any:  # reason: rclpy parameter value
            return self.get_parameter(name).value

        # -- lifecycle -------------------------------------------------------------------

        def on_configure(self, state: object) -> TransitionCallbackReturn:
            import tf2_ros

            try:
                mode = TrackingMode(str(self._p("mode")))
                wait_s = float(self._p("grounding_wait_s"))
                if wait_s <= 0.0:
                    raise ROSConfigError("object tracker: grounding_wait_s must be set (> 0).")
                if mode is not TrackingMode.SHADOW and str(self._p("hal_mode")) != "sim":
                    raise ROSConfigError(
                        f"object tracker: mode={mode.value!r} is sim-only until Safety-WG "
                        "review; only 'shadow' runs on a real HAL."
                    )
                description = RobotDescription.from_yaml(str(self._p("robot_yaml")))
                self.session = TrackerSession(
                    description,
                    cell_m=float(self._p("cell_m")),
                    grounding_wait_s=wait_s,
                    grounding_required=bool(self._p("grounding_required")),
                    release_dwell_s=float(self._p("release_dwell_s")),
                    released_clear_timeout_s=float(self._p("released_clear_timeout_s")),
                    require_hands=mode is not TrackingMode.SHADOW,
                    # The tracker reads a stalled jaw itself in bind/on (sim only, enforced
                    # above): the HAL trigger is unchanged and a real ATTACH is never doubled.
                    derive_attach=mode is not TrackingMode.SHADOW,
                    params=TrackerParams(
                        search_start_s=float(self._p("search_start_s")),
                        search_max_s=float(self._p("search_max_s")),
                    ),
                )
                raw = str(self._p("object_queries_json"))
                if raw:
                    self.session.open_standing_queries(
                        TypeAdapter(list[ObjectQuery]).validate_json(raw)
                    )
            except (ROSConfigError, ValueError) as exc:
                self.get_logger().error(f"object tracker configure refused: {exc}")
                return TransitionCallbackReturn.FAILURE
            self.mode = mode
            self._joint_order = [j.name for j in description.joints]
            for j in description.joints:
                self._aliases[j.name] = j.name
                if j.sim_joint_name:
                    self._aliases[j.sim_joint_name] = j.name
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

            for name in [c for c in self._p("cameras") if c]:
                cam = self._cams[name] = _Camera(name)
                self._subs += [
                    self.create_subscription(
                        Image,
                        camera_topic(name, CameraTopicKind.DEPTH_IMAGE),
                        lambda m, c=cam: self._on_depth(c, m),
                        sensor_qos,
                    ),
                    self.create_subscription(
                        CameraInfo,
                        camera_topic(name, CameraTopicKind.DEPTH_CAMERA_INFO),
                        lambda m, c=cam: setattr(c, "info", intrinsics_from_info(m)),
                        sensor_qos,
                    ),
                    self.create_subscription(
                        Image,
                        camera_topic(name, CameraTopicKind.ROBOT_MASK),
                        lambda m, c=cam: self._on_mask(c, m),
                        sensor_qos,
                    ),
                ]
            self._subs += [
                self.create_subscription(
                    OccupancyVoxels, "/openral/world_voxels", self._on_voxels, reliable_1
                ),
                self.create_subscription(
                    JointState, str(self._p("joint_states_topic")), self._on_joints, sensor_qos
                ),
                self.create_subscription(
                    GraspEventMsg, "/openral/grasp_events", self._on_grasp_event, events_qos
                ),
                self.create_subscription(
                    ActionChunkMsg, "/openral/safe_action", self._on_safe_action, events_qos
                ),
                self.create_subscription(
                    GraspDeclarationMsg,
                    "/openral/grasp_declaration",
                    self._on_grasp_declaration,
                    latched_1,
                ),
                self.create_subscription(
                    PlaceDeclarationMsg,
                    "/openral/place_declaration",
                    self._on_place_declaration,
                    latched_1,
                ),
            ]
            self._pub_tracked = self.create_publisher(
                TrackedObjectSetMsg, "/openral/tracked_objects", reliable_1
            )
            if self.mode is not TrackingMode.SHADOW:
                self._pub_attach = self.create_publisher(
                    AttachmentStateMsg, "/openral/attachment_state", latched_1
                )
            # Every fit / grounding refusal by (key, reason), so a tracker that grounds
            # nothing says why on the graph too (#418 restock), not only in its log.
            from diagnostic_msgs.msg import DiagnosticArray

            self._pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
            self._diag_ns = 0
            # Addendum G escalations reach the reasoner on the failure bus.
            self._failure_bus = FailureBusPublisher(self, FailureSource.WORLD_STATE)
            self._failure_bus.create_publisher()
            self._segment = self.create_client(SegmentInView, str(self._p("segment_service")))
            if self._p("locate_service"):
                self._locate = self.create_client(LocateInView, str(self._p("locate_service")))
            self.get_logger().info(
                f"object tracker configured: mode={self.mode.value} cameras={sorted(self._cams)} "
                f"frame={self.session.frame_id} hands={sorted(self.session.binders)} "
                f"locate={self._p('locate_service')!r}"
            )
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: object) -> TransitionCallbackReturn:
            # Wall-time seed: a restarted or re-activated tracker never regresses a revision.
            self._rev_base = time.time_ns()
            self._last_tracked_rev = -1
            self._attach_key = None
            self._active = True
            self._timers = [
                self.create_timer(1.0 / float(self._p("process_hz")), self._process),
                self.create_timer(1.0 / float(self._p("heartbeat_hz")), self._heartbeat),
            ]
            if self._failure_bus is not None:
                self._failure_bus.start()
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: object) -> TransitionCallbackReturn:
            self._active = False
            if self._failure_bus is not None:
                self._failure_bus.stop()
            for t in self._timers:
                self.destroy_timer(t)
            self._timers = []
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: object) -> TransitionCallbackReturn:
            for s in self._subs:
                self.destroy_subscription(s)
            self._subs = []
            for pub in (self._pub_tracked, self._pub_attach):
                if pub is not None:
                    self.destroy_publisher(pub)
            for cli in (self._segment, self._locate):
                if cli is not None:
                    self.destroy_client(cli)
            self._pub_tracked = self._pub_attach = self._segment = self._locate = None
            if self._failure_bus is not None:
                self._failure_bus.destroy()
                self._failure_bus = None
            self._attempts.clear()
            self._cams.clear()
            self._tf_listener = self._tf_buffer = None
            self.session = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: object) -> TransitionCallbackReturn:
            self.on_deactivate(state)
            return self.on_cleanup(state)

        # -- inputs ----------------------------------------------------------------------

        def _on_depth(self, cam: _Camera, msg: Any) -> None:
            cam.pending = (msg, time.monotonic())  # newest wins: the tracker is not a queue

        def _on_mask(self, cam: _Camera, msg: Any) -> None:
            cam.masks[_stamp_ns(msg.header.stamp)] = image_array(msg)
            while len(cam.masks) > _MASKS_KEPT:
                del cam.masks[min(cam.masks)]

        def _on_voxels(self, msg: Any) -> None:
            if self.session is None:
                return
            stamp = _stamp_ns(msg.source_stamp) or _stamp_ns(msg.header.stamp)
            self.session.on_grid(lattice_from_msg(msg), stamp)

        def _on_joints(self, msg: Any) -> None:
            if self.session is None:
                return
            positions = {
                self._aliases[n]: float(q)
                for n, q in zip(msg.name, msg.position, strict=False)
                if n in self._aliases
            }
            stamp = _stamp_ns(msg.header.stamp) or int(self.get_clock().now().nanoseconds)
            self._joint_rx_ns = int(self.get_clock().now().nanoseconds)
            self.session.on_joint_state(positions, stamp)
            self._publish_attachment()

        def _on_safe_action(self, msg: Any) -> None:
            """The approved command's gripper targets, for the tracker's own stalled-jaw read.

            The chunk's last row, like the HAL trigger reads it: a full-dof row at each gripper's
            manifest index, a compact row positionally by ``joint_names``. A slot of a group is
            not the applied command and is ignored.
            """
            if self.session is None or int(msg.tick_group_size) > 1:
                return
            n, flat = int(msg.n_dof), list(msg.flat)
            if n <= 0 or len(flat) < n:
                return
            row, names = flat[-n:], list(msg.joint_names)
            for joint in self.session.binders:
                if names:
                    index = names.index(joint) if joint in names and len(names) == n else None
                elif n == len(self._joint_order) and joint in self._joint_order:
                    index = self._joint_order.index(joint)
                else:
                    index = None
                if index is not None:
                    self.session.on_jaw_command(joint, float(row[index]), self.session.now_ns)

        def _on_grasp_event(self, msg: Any) -> None:
            kind = kinds.get(int(msg.kind))
            if self.session is None or kind is None:
                return
            self.session.on_grasp_event(
                kind,
                str(msg.gripper_joint),
                jaw=float(msg.jaw_measured),
                hold=float(msg.hold_position),
                stamp_ns=_stamp_ns(msg.stamp),
            )
            self._publish_tracked()
            self._publish_attachment()

        def _on_grasp_declaration(self, msg: Any) -> None:
            if self.session is not None:
                self.session.on_grasp_declaration(GraspDeclaration.from_idl(msg))
                self._publish_tracked()
                self._publish_attachment()

        def _on_place_declaration(self, msg: Any) -> None:
            if self.session is not None:
                self.session.on_place_declaration(PlaceDeclaration.from_idl(msg))
                self._publish_attachment()

        def _lookup(self, frame: str, stamp_ns: int) -> NDArray[np.float64] | None:
            assert self.session is not None
            at = Time(nanoseconds=stamp_ns)
            if not self._tf_buffer.can_transform(self.session.frame_id, frame, at):
                return None
            tf = self._tf_buffer.lookup_transform(self.session.frame_id, frame, at).transform
            t, q = tf.translation, tf.rotation
            return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

        # -- the frame loop --------------------------------------------------------------

        def _process(self) -> None:
            if self.session is None or not self._active:
                return
            for cam in self._cams.values():
                if cam.pending is None or cam.info is None:
                    continue
                msg, rx = cam.pending
                stamp = _stamp_ns(msg.header.stamp)
                mask = cam.masks.get(stamp)
                t_cam = self._lookup(str(msg.header.frame_id), stamp)
                waited = time.monotonic() - rx
                if (mask is None or t_cam is None) and waited < float(self._p("mask_tf_wait_s")):
                    continue
                cam.pending = None
                depth = image_array(msg)
                if mask is not None and mask.shape != depth.shape:
                    mask = None
                if mask is None or t_cam is None:
                    self.get_logger().debug(
                        f"object tracker {cam.name}@{stamp}: ageing only "
                        f"(mask={'ok' if mask is not None else 'missing'}, "
                        f"tf={'ok' if t_cam is not None else 'missing'})"
                    )
                frame = CameraFrame(cam.name, stamp, depth, cam.info, t_cam, mask)
                self._feed(frame, str(msg.header.frame_id), msg.header.stamp)

        def _feed(
            self,
            frame: CameraFrame,
            optical_frame: str,
            stamp_msg: Any,
            results: Sequence[TrackerResult] = (),
        ) -> None:
            assert self.session is not None
            out = self.session.on_frame(frame, wall_s=time.monotonic(), results=results)
            for e in out.escalations:
                self._escalate(e)
            self._publish_tracked()
            self._publish_attachment()
            if out.requests:
                self._issue(out.requests[0], frame, optical_frame, stamp_msg)

        def _answer(
            self,
            frame: CameraFrame,
            optical_frame: str,
            stamp_msg: Any,
            result: TrackerResult,
        ) -> None:
            if self._active and self.session is not None:
                self._feed(frame, optical_frame, stamp_msg, (result,))

        def _escalate(self, e: QueryEscalation) -> None:
            """Record one Addendum G escalation and send it to the reasoner.

            Typed failure ``ROSObjectNotInMemory`` (``ObjectQueryEvidence``): a structlog
            event (WARN unresolved, ERROR exhausted), a ``world_state.object_query`` span
            parented on the goal's skill-execution trace (its declaration's ``trace_id``)
            with the exception recorded, and a ``FailureTrigger`` on
            ``/openral/failure/world_state`` carrying that trace id (the recorded bag
            keeps it, CLAUDE.md §1.8). The tracker acts on none of it.
            """
            assert self.session is not None
            query = self.session.tracker.queries.get(e.goal_id, e.query_id)
            if query is None:  # the goal closed in the same tick: nothing to escalate
                return
            evidence = escalation_evidence(e, query, self._attempts.get(query.key, 0))
            exc = evidence.to_exception()
            s = self.session
            decl = {"grasp": s.grasp_declaration, "place": s.place_declaration}.get(
                e.goal_id.split(":", 1)[0]
            )
            trace_id = decl.trace_id if decl is not None else ""
            rskill_id = decl.rskill_id if decl is not None else ""
            attrs = {
                "escalation": evidence.escalation,
                "goal_id": evidence.goal_id,
                "query_id": evidence.query_id,
                "label": evidence.label,
                "role": evidence.role.value,
                "waited_s": evidence.waited_s,
                "attempts": evidence.attempts,
                "error_type": type(exc).__name__,
            }
            with world_state_span(
                _semconv.SPAN_WORLD_STATE_OBJECT_QUERY, traceparent=trace_id, **attrs
            ) as span:
                span.record_exception(exc)
                exhausted = evidence.escalation == "exhausted"
                emit = slog.error if exhausted else slog.warning
                emit("object_tracker.query_escalation", error=str(exc), **attrs)
                if self._failure_bus is not None:
                    self._failure_bus.publish(
                        kind=KIND_OBJECT_QUERY,
                        severity=SEVERITY_FAIL if exhausted else SEVERITY_WARN,
                        evidence=evidence,
                        rskill_id=rskill_id,
                        trace_id=trace_id or None,
                    )

        def _issue(
            self,
            req: TrackerRequest,
            frame: CameraFrame,
            optical_frame: str,
            stamp_msg: Any,
        ) -> None:
            if req.query_key is not None:
                self._attempts[req.query_key] = self._attempts.get(req.query_key, 0) + 1
            empty = TrackerResult(req.request_id)
            if req.service == "locate_in_view":
                if self._locate is None or not self._locate.service_is_ready():
                    self._answer(frame, optical_frame, stamp_msg, empty)
                    return
                lreq = LocateInView.Request(
                    query=req.prompt or req.label, camera=req.camera, detector=""
                )
                lreq.roi_norm_xyxy = list(req.roi_norm_xyxy or ())
                lreq.min_score = float(req.min_score)
                future = self._locate.call_async(lreq)

                def done_locate(f: Any) -> None:
                    res = f.result()
                    boxes = (
                        detector_boxes(str(res.metadata_json), frame.intrinsics)
                        if res is not None and res.found
                        else ()
                    )
                    scores = (
                        detector_scores(str(res.metadata_json))
                        if res is not None and res.found
                        else ()
                    )
                    result = TrackerResult(
                        req.request_id,
                        None,
                        boxes,
                        box_scores=scores if len(scores) == len(boxes) else (),
                    )
                    self._answer(frame, optical_frame, stamp_msg, result)

                future.add_done_callback(done_locate)
                return
            uv = req.point_uv
            if uv is None and req.box_xyxy is not None:
                x0, y0, x1, y1 = req.box_xyxy
                uv = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            point = None if uv is None else backproject_pixel(frame.depth_m, frame.intrinsics, uv)
            if point is None or not self._segment.service_is_ready():
                self._answer(frame, optical_frame, stamp_msg, empty)
                return
            sreq = SegmentInView.Request()
            sreq.stamp = stamp_msg
            sreq.camera = req.camera
            sreq.frame_id = optical_frame
            sreq.tcp_point.x, sreq.tcp_point.y, sreq.tcp_point.z = point
            for nuv in req.negative_uv:  # robot pixels beside a half-hidden object
                neg = backproject_pixel(frame.depth_m, frame.intrinsics, nuv)
                if neg is not None:
                    pt = Point()
                    pt.x, pt.y, pt.z = neg
                    sreq.negative_points.append(pt)
            sreq.box_valid = req.box_xyxy is not None
            if req.box_xyxy is not None:
                sreq.box_xyxy = [float(v) for v in req.box_xyxy]
            future = self._segment.call_async(sreq)

            def done_segment(f: Any) -> None:
                res = f.result()
                masks = (
                    [_resize_nearest(image_array(m) > 0, frame.depth_m.shape) for m in res.masks]
                    if res is not None and res.ok
                    else []
                )
                ok = pick_masks(masks, req.box_xyxy, req.within_xyxy)
                result = TrackerResult(
                    req.request_id, ok[0] if ok else None, alternatives=tuple(ok[1:])
                )
                self._answer(frame, optical_frame, stamp_msg, result)

            future.add_done_callback(done_segment)

        # -- outputs ---------------------------------------------------------------------

        def _publish_tracked(self, *, force: bool = False) -> None:
            if self.session is None or self._pub_tracked is None or not self._active:
                return
            snap = self.session.snapshot()
            rev = self._rev_base + snap.revision
            if not force and rev == self._last_tracked_rev:
                return
            self._last_tracked_rev = rev
            msg = TrackedObjectSetMsg()
            snap.model_copy(update={"revision": rev}).fill_idl(
                msg, object_factory=TrackedObjectMsg, primitive_factory=PrimitiveMsg
            )
            self._pub_tracked.publish(msg)

        def _publish_attachment(self, *, heartbeat: bool = False) -> None:
            s = self.session
            if s is None or self._pub_attach is None or not self._active:
                return
            key = (s.attachment_changes, id(s.grasp_declaration), id(s.place_declaration))
            if key != self._attach_key:
                self._attach_key = key
                self._attach_rev += 1
            elif not heartbeat:
                return
            now = self.get_clock().now()
            msg = AttachmentStateMsg()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = s.frame_id
            msg.revision = self._rev_base + self._attach_rev
            for obj in s.attachments():
                item = AttachedMsg()
                obj.fill_idl(item, primitive_factory=PrimitiveMsg)
                msg.objects.append(item)
            grasp, place = s.grasp_declaration, s.place_declaration
            msg.grasp_declaration_valid = grasp is not None
            if grasp is not None:  # dispatch's own declaration: no region, nothing exempted
                # The producer's workspace confirmation, redone on the tracker's grid.
                ws = s.workspace_region()
                grasp.model_copy(update={"region": None, "workspace_region": ws}).fill_idl(
                    msg.grasp_declaration, query_factory=ObjectQueryMsg
                )
            msg.place_declaration_valid = place is not None
            if place is not None:
                place.model_copy(update={"region": None}).fill_idl(msg.place_declaration)
            self._pub_attach.publish(msg)

        def _publish_refusals(self) -> None:
            """``/diagnostics``: the tracker's refusal counts, at most every 5 s."""
            from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

            now = int(self.get_clock().now().nanoseconds)
            if self.session is None or now - self._diag_ns < 5 * _NS:
                return
            self._diag_ns = now
            counts = self.session.tracker.refusal_counts
            status = DiagnosticStatus(
                level=DiagnosticStatus.OK,
                name="openral_object_tracker: refusals",
                hardware_id=self.session.frame_id,
                message=f"{sum(counts.values())} refusals",
                values=[
                    KeyValue(key=f"{k}|{why}", value=str(n))
                    for (k, why), n in sorted(counts.items())
                ],
            )
            msg = DiagnosticArray(status=[status])
            msg.header.stamp = self.get_clock().now().to_msg()
            self._pub_diag.publish(msg)

        def _heartbeat(self) -> None:
            self._publish_tracked(force=True)
            self._publish_refusals()
            now = int(self.get_clock().now().nanoseconds)
            timeout = float(self._p("attachment_evidence_timeout_s")) * _NS
            if self._joint_rx_ns and now - self._joint_rx_ns <= timeout:
                self._publish_attachment(heartbeat=True)  # else the kernel's copy ages out

    return ObjectTrackerNode()
