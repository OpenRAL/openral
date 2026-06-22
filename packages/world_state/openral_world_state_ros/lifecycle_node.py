r"""openral_world_state ROS 2 lifecycle node.

Wraps :class:`openral_world_state.WorldStateAggregator` as a managed
lifecycle node. Subscribes to ``/joint_states`` and (per ADR-0018 F2)
publishes a typed :class:`openral_msgs.msg.WorldStateStamped` snapshot
on two topics:

- ``/openral/world_state_fast`` (30 Hz, ``RELIABLE+VOLATILE+KL=1``) —
  dashboards, observability, fast consumers.
- ``/openral/world_state_slow`` (5 Hz, ``RELIABLE+VOLATILE+KL=1``) —
  the reasoner.

Both topics carry the same payload built from a single in-memory
snapshot per fast tick (the slow topic re-publishes the same message
every Nth fast tick where ``N = round(fast_hz / slow_hz)``). The
legacy JSON publication on ``/world_state`` is removed by this PR —
typed is the only path (ADR-0018 §2; capability review F2).

Lifecycle transitions
---------------------
- ``configure``  → initialise aggregator + subscriptions + publishers
- ``activate``   → start fast publish timer (drives both topics)
- ``deactivate`` → stop timer
- ``cleanup``    → destroy subscriptions and aggregator

Usage (after colcon build + source install/setup.bash)::

    ros2 run openral_world_state world_state_node \\
        --ros-args -p robot_name:=so100 \\
          -p publish_rate_hz_fast:=30.0 \\
          -p publish_rate_hz_slow:=5.0

Parameters
----------
robot_name (str)
    Short robot identifier used to build the RobotDescription stub.
    Default: ``"robot"``.
publish_rate_hz_fast (float)
    Fast-topic snapshot rate (``/openral/world_state_fast``).
    Default: ``30.0``.
publish_rate_hz_slow (float)
    Slow-topic snapshot rate (``/openral/world_state_slow``).
    Default: ``5.0``.
staleness_limit_s (float)
    Age threshold after which a sensor is marked stale. Default: ``0.5``
    (0.1 s equals the 10 Hz camera period and made diagnostics flap).
"""

from __future__ import annotations

import logging

from opentelemetry import trace

log = logging.getLogger(__name__)

try:
    import rclpy  # type: ignore[import-untyped]
    from openral_observability import log_lifecycle_errors
    from rclpy.executors import (  # type: ignore[import-untyped]
        ExternalShutdownException,
    )
    from rclpy.lifecycle import (  # type: ignore[import-untyped]
        LifecycleNode,
        TransitionCallbackReturn,
    )
    from rclpy.qos import (  # type: ignore[import-untyped]
        QoSDurabilityPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


def main() -> None:
    """Entry point for ``ros2 run openral_world_state world_state_node``."""
    if not _ROS2_AVAILABLE:
        log.error("rclpy not found — cannot start without ROS 2.")
        raise SystemExit(1)

    from openral_observability import configure_observability

    # Idempotent + no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset. Lets
    # ``sensors.read_latest`` + ``world_state.snapshot`` spans flow when
    # the node is launched standalone (the composed ``runtime_node``
    # already calls this, so spans flow there regardless).
    configure_observability(service_name="openral.world_state")

    rclpy.init()
    node = _WorldStateLifecycleNode()  # type: ignore[name-defined]
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy. On
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread, spin instead raises ExternalShutdownException.
            # Either way the context is already shut down by the time we
            # reach the `finally` below, so the bare `rclpy.shutdown()`
            # we used to call there raised
            # `RCLError: rcl_shutdown already called` — the
            # `try_shutdown()` switch below is the corresponding fix.
            pass
        finally:
            node.destroy_node()
    finally:
        # Idempotent — no-op when the SIGINT handler (or whoever fired
        # ExternalShutdownException) already shut down the context.
        rclpy.try_shutdown()


if _ROS2_AVAILABLE:
    # Module-level constants — keep on the same source so tests and
    # downstream consumers (rqt adapters, F4 reasoner) can import them
    # without depending on the generated IDL.
    TOPIC_FAST = "/openral/world_state_fast"
    TOPIC_SLOW = "/openral/world_state_slow"

    # Diagnostic status enum mirroring ``WorldStateStamped.DIAG_*``. The
    # generated message ships the constants too, but exposing them here
    # lets the snapshot builder run without a round-trip through the
    # IDL when constructing the parallel arrays.
    _DIAG_STATUS = {
        "ok": 0,
        "warn": 1,
        "stale": 2,
        "error": 3,
    }

    class _WorldStateLifecycleNode(LifecycleNode):  # type: ignore[misc]
        """Managed lifecycle node for the World State aggregator.

        Parameters (ROS 2 node params):
            robot_name (str): Short robot name for the description stub.
            publish_rate_hz_fast (float): Fast-topic publish rate.
                Default: 30.0.
            publish_rate_hz_slow (float): Slow-topic publish rate.
                Default: 5.0.
            staleness_limit_s (float): Staleness threshold. Default: 0.5.
        """

        def __init__(self, aggregator: object | None = None) -> None:
            """Declare parameters; does not open any connections.

            Args:
                aggregator: Optional pre-constructed
                    ``openral_world_state.WorldStateAggregator`` instance.
                    When supplied (ADR-0018 §3 / F1
                    ``compose_so100_runtime`` composition), the node
                    reuses it and skips internal construction on
                    ``on_configure``. ``None`` preserves the standalone
                    behaviour — the node owns its own aggregator built
                    against a stub ``RobotDescription``.
            """
            super().__init__("openral_world_state")
            self.declare_parameter("robot_name", "robot")
            self.declare_parameter("publish_rate_hz_fast", 30.0)
            self.declare_parameter("publish_rate_hz_slow", 5.0)
            # 0.5 s, not 0.1 s: with 10 Hz cameras a 0.1 s window equals the
            # frame period, so the per-sensor diagnostics flapped OK↔STALE on
            # every snapshot. See WorldStateAggregator.DEFAULT_STALENESS_S.
            self.declare_parameter("staleness_limit_s", 0.5)
            # Camera image topics to subscribe to. Each entry yields a
            # `sensor_msgs/Image` subscription that lands the bytes on
            # `WorldStateAggregator.update_image_frame(<name>, ...)`
            # so the aggregator's snapshot carries pixels into the
            # rSkill. Names match the keys the rSkill's
            # `image_preprocessing.aliases` expects (e.g. `top`,
            # `left_wrist`, `right_wrist`).
            self.declare_parameter("camera_names", [""])
            self.declare_parameter("camera_topic_prefix", "/openral/cameras")
            self.declare_parameter("object_lift_enabled", True)
            self.declare_parameter("object_detections_topic", "/openral/perception/objects")
            self.declare_parameter("object_voxels_topic", "/openral/world_voxels")
            # ADR-0035 amendment (#11) — depth point cloud used as the lift's
            # depth source when no octomap voxel grid is available (octomap is
            # often disabled in dense scenes to avoid kernel false positives, yet
            # the lift still needs depth to place a 2D box in 3D). Empty disables
            # the fallback.
            self.declare_parameter(
                "object_depth_points_topic", "/openral/cameras/front_depth/points"
            )
            self.declare_parameter("object_lift_depth_max_points", 4000)
            self.declare_parameter("object_lift_map_frame", "map")
            self.declare_parameter("object_lift_k_nearest", 25)
            self.declare_parameter("object_lift_min_voxels", 3)
            self.declare_parameter("object_lift_iou_threshold", 0.3)
            self.declare_parameter("object_lift_memory_cadence_hz", 2.0)
            self.declare_parameter("object_lift_max_misses", 1)
            self.declare_parameter("object_lift_voxel_staleness_s", 1.0)

            self._aggregator = aggregator
            self._owns_aggregator = aggregator is None
            self._timer = None
            self._pub_fast = None
            self._pub_slow = None
            self._joint_sub = None
            self._camera_subs: dict[str, object] = {}
            self._slow_divider = 1
            self._tick_count = 0
            # ADR-0018 F8 — heartbeat.
            self._heartbeat: object | None = None
            # ADR-0035 — object lift state.
            self._lift_enabled = False
            self._tf_buffer: object | None = None
            self._tf_listener: object | None = None
            self._det_sub: object | None = None
            self._voxel_sub: object | None = None
            self._memory_timer: object | None = None
            self._lifter: object | None = None
            self._memory: object | None = None
            self._latest_voxels: object | None = None
            self._voxel_stamp_ns: int = 0
            # #11 — depth point-cloud fallback for the lift (octomap-free path).
            self._depth_points_sub: object | None = None
            self._latest_depth_points: object | None = None
            self._depth_points_stamp_ns: int = 0
            self._depth_max_points: int = 0
            self._candidate_buffer: list[object] = []
            self._seen_sensor_ids: set[str] = set()
            self._map_frame = "map"
            self._voxel_staleness_ns = 0

            self.get_logger().info("WorldState node initialised.")

        @log_lifecycle_errors
        def on_configure(self, state: object) -> TransitionCallbackReturn:
            """Initialise the aggregator (if owned) and topic plumbing."""
            from openral_msgs.msg import (  # type: ignore[import-untyped]
                WorldStateStamped,
            )
            from openral_observability import DiagnosticsHeartbeat, Level
            from sensor_msgs.msg import (
                JointState as RosJointState,  # type: ignore[import-untyped]
            )

            robot_name: str = self.get_parameter("robot_name").get_parameter_value().string_value
            staleness: float = (
                self.get_parameter("staleness_limit_s").get_parameter_value().double_value
            )

            # ADR-0018 §3 — `WorldStateAggregator` is the only subscriber of
            # `/joint_states`. When an aggregator was supplied at
            # construction (compose_so100_runtime path), reuse it; otherwise
            # build the standalone-mode stub-backed aggregator preserved for
            # back-compat.
            if self._aggregator is None:
                from openral_core import (
                    ControlMode,
                    EmbodimentKind,
                    JointSpec,
                    JointType,
                    RobotCapabilities,
                    RobotDescription,
                    SafetyEnvelope,
                )
                from openral_world_state import WorldStateAggregator

                desc = RobotDescription(
                    name=robot_name,
                    embodiment_kind=EmbodimentKind.MANIPULATOR,
                    joints=[
                        JointSpec(
                            name="j0",
                            joint_type=JointType.REVOLUTE,
                            parent_link="base_link",
                            child_link="link_0",
                        )
                    ],
                    capabilities=RobotCapabilities(
                        supported_control_modes=[ControlMode.JOINT_POSITION],
                    ),
                    safety=SafetyEnvelope(),
                )
                self._aggregator = WorldStateAggregator(desc, staleness_limit_s=staleness)

            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=5,
            )
            self._joint_sub = self.create_subscription(
                RosJointState,
                "/joint_states",
                self._on_joint_state,
                sensor_qos,
            )

            # Per-camera image subscriptions. The HAL publishes
            # `sensor_msgs/Image` with RELIABLE QoS on
            # `<prefix>/<name>/image`; we mirror RELIABLE so the
            # subscription matches and we never miss a frame on
            # bring-up.
            from sensor_msgs.msg import Image as RosImage

            camera_names_raw = list(
                self.get_parameter("camera_names").get_parameter_value().string_array_value,
            )
            camera_names = [n for n in camera_names_raw if n]
            topic_prefix: str = (
                self.get_parameter("camera_topic_prefix").get_parameter_value().string_value
            )
            image_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            for name in camera_names:
                topic = f"{topic_prefix}/{name}/image"
                self._camera_subs[name] = self.create_subscription(
                    RosImage,
                    topic,
                    lambda msg, _name=name, _topic=topic: self._on_image(_name, _topic, msg),
                    image_qos,
                )
            if camera_names:
                self.get_logger().info(
                    f"WorldState subscribing to {len(camera_names)} camera(s): "
                    f"{', '.join(camera_names)}",
                )

            # ADR-0018 §1 — RELIABLE+VOLATILE+KL=1 on both world_state topics.
            ws_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._pub_fast = self.create_publisher(WorldStateStamped, TOPIC_FAST, ws_qos)
            self._pub_slow = self.create_publisher(WorldStateStamped, TOPIC_SLOW, ws_qos)

            # ADR-0018 F8 — heartbeat.
            def _status() -> tuple[int, str, dict[str, str]]:
                # Aggregator presence is the meaningful per-node fact here;
                # joint-state staleness lands when F2's typed publication
                # surfaces ``staleness_ms[]`` alongside the snapshot.
                if self._aggregator is None:
                    return Level.ERROR, "aggregator missing", {"robot": robot_name}
                return (
                    Level.OK,
                    "aggregator ready",
                    {
                        "robot": robot_name,
                        "staleness_s": f"{staleness:.3f}",
                    },
                )

            self._heartbeat = DiagnosticsHeartbeat(
                self,
                hardware_id=f"openral_world_state:{robot_name}",
                component_name="openral_world_state",
                status_fn=_status,
            )
            self._heartbeat.create_publisher()  # type: ignore[union-attr]

            self.get_logger().info(f"WorldState configured for robot '{robot_name}'.")
            self._lift_enabled = (
                self.get_parameter("object_lift_enabled").get_parameter_value().bool_value
            )
            if self._lift_enabled:
                import tf2_ros
                from openral_msgs.msg import OccupancyVoxels, PromptStamped
                from openral_world_state import ObjectMemory, VoxelFrustumLifter

                self._map_frame = (
                    self.get_parameter("object_lift_map_frame").get_parameter_value().string_value
                )
                self._voxel_staleness_ns = int(
                    self.get_parameter("object_lift_voxel_staleness_s")
                    .get_parameter_value()
                    .double_value
                    * 1e9
                )
                self._lifter = VoxelFrustumLifter(
                    k_nearest=self.get_parameter("object_lift_k_nearest")
                    .get_parameter_value()
                    .integer_value,
                    min_voxels=self.get_parameter("object_lift_min_voxels")
                    .get_parameter_value()
                    .integer_value,
                )
                self._memory = ObjectMemory(
                    iou_threshold=self.get_parameter("object_lift_iou_threshold")
                    .get_parameter_value()
                    .double_value,
                    max_misses=self.get_parameter("object_lift_max_misses")
                    .get_parameter_value()
                    .integer_value,
                )
                self._tf_buffer = tf2_ros.Buffer()
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

                det_topic = (
                    self.get_parameter("object_detections_topic").get_parameter_value().string_value
                )
                vox_topic = (
                    self.get_parameter("object_voxels_topic").get_parameter_value().string_value
                )
                det_qos = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=5,
                )
                vox_qos = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE,
                    depth=1,
                )
                self._det_sub = self.create_subscription(
                    PromptStamped,
                    det_topic,
                    self._on_objects,
                    det_qos,
                )
                self._voxel_sub = self.create_subscription(
                    OccupancyVoxels,
                    vox_topic,
                    self._on_voxels,
                    vox_qos,
                )
                depth_topic = self._setup_depth_points_fallback(det_qos)
                self.get_logger().info(
                    f"WorldState object lift enabled (map='{self._map_frame}', "
                    f"detections='{det_topic}', voxels='{vox_topic}', "
                    f"depth_fallback='{depth_topic or '(off)'}')."
                )
            return TransitionCallbackReturn.SUCCESS

        @log_lifecycle_errors
        def on_activate(self, state: object) -> TransitionCallbackReturn:
            """Start the fast snapshot publish timer (drives both topics)."""
            fast_hz: float = (
                self.get_parameter("publish_rate_hz_fast").get_parameter_value().double_value
            )
            slow_hz: float = (
                self.get_parameter("publish_rate_hz_slow").get_parameter_value().double_value
            )
            # Slow-topic divider — emit on the slow publisher every Nth
            # fast tick. Clamped to ≥1 so slow_hz ≥ fast_hz still publishes
            # something on the slow topic (degenerate but explicit).
            self._slow_divider = max(1, round(fast_hz / max(slow_hz, 1e-6)))
            self._tick_count = 0
            period_s = 1.0 / max(fast_hz, 1.0)
            self._timer = self.create_timer(period_s, self._publish_snapshot)
            if self._lift_enabled:
                cadence: float = (
                    self.get_parameter("object_lift_memory_cadence_hz")
                    .get_parameter_value()
                    .double_value
                )
                self._memory_timer = self.create_timer(
                    1.0 / max(cadence, 0.1),
                    self._on_memory_tick,
                )
            if self._heartbeat is not None:
                self._heartbeat.start()  # type: ignore[union-attr]
            self.get_logger().info(
                f"WorldState publishing fast={fast_hz:.1f} Hz, "
                f"slow={slow_hz:.1f} Hz (divider={self._slow_divider}).",
            )
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, state: object) -> TransitionCallbackReturn:
            """Stop the publish timer."""
            if self._heartbeat is not None:
                self._heartbeat.stop()  # type: ignore[union-attr]
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._memory_timer is not None:
                self._memory_timer.cancel()
                self._memory_timer = None
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: object) -> TransitionCallbackReturn:
            """Destroy subscriptions, publishers, and (if owned) aggregator."""
            if self._heartbeat is not None:
                self._heartbeat.destroy()  # type: ignore[union-attr]
                self._heartbeat = None
            if self._joint_sub is not None:
                self.destroy_subscription(self._joint_sub)
                self._joint_sub = None
            for sub in self._camera_subs.values():
                self.destroy_subscription(sub)  # type: ignore[arg-type]
            self._camera_subs.clear()
            if self._det_sub is not None:
                self.destroy_subscription(self._det_sub)  # type: ignore[arg-type]
                self._det_sub = None
            if self._voxel_sub is not None:
                self.destroy_subscription(self._voxel_sub)  # type: ignore[arg-type]
                self._voxel_sub = None
            self._tf_listener = None
            self._tf_buffer = None
            self._latest_voxels = None
            self._candidate_buffer = []
            self._seen_sensor_ids = set()
            if self._pub_fast is not None:
                self.destroy_publisher(self._pub_fast)
                self._pub_fast = None
            if self._pub_slow is not None:
                self.destroy_publisher(self._pub_slow)
                self._pub_slow = None
            # Only the standalone-mode node owns the aggregator. When the
            # compose factory injected one, cleanup must not drop it — the
            # composed rskill_runner_node still holds the snapshot reference.
            if self._owns_aggregator:
                self._aggregator = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: object) -> TransitionCallbackReturn:
            """Force cleanup on shutdown."""
            return self.on_cleanup(state)

        def _on_image(self, sensor_name: str, topic: str, msg: object) -> None:
            """Convert ROS Image → SensorFrame and hand to aggregator.

            Emits a ``sensors.read_latest`` OTel span per frame so the
            dashboard's Perception card populates (modality, encoding,
            geometry, age, JPEG thumbnail). The span name + attribute
            shape mirror :meth:`DeployRunner._tick_impl`'s sensor read
            so a single dashboard consumer handles both topologies.
            """
            import time

            from openral_core.schemas import FrameEncoding, SensorFrame
            from openral_observability import producer as ral_producer
            from openral_observability import semconv

            if self._aggregator is None:
                return

            _ENCODING_MAP = {  # noqa: N806  # reason: ALL-CAPS preserved because this maps to module-level FrameEncoding enum constants
                "rgb8": FrameEncoding.RGB8,
                "bgr8": FrameEncoding.BGR8,
                "mono8": FrameEncoding.MONO8,
            }
            encoding_str: str = str(getattr(msg, "encoding", "rgb8") or "rgb8").lower()
            encoding = _ENCODING_MAP.get(encoding_str)
            if encoding is None:
                # Unknown encoding — skip with a warn rather than crashing
                # the executor on a misconfigured publisher.
                self.get_logger().warn(
                    f"_on_image({sensor_name!r}): unsupported encoding {encoding_str!r}; "
                    f"expected one of {sorted(_ENCODING_MAP)}.",
                )
                return

            width = int(getattr(msg, "width", 0) or 0)
            height = int(getattr(msg, "height", 0) or 0)
            if width <= 0 or height <= 0:
                self.get_logger().warn(
                    f"_on_image({sensor_name!r}): width={width} height={height} invalid; skipping.",
                )
                return

            data = bytes(getattr(msg, "data", b"") or b"")
            now_ns = time.time_ns()
            # Source-stamp the frame: pull header.stamp when present so
            # the perception age reflects the publisher's clock, not the
            # subscriber's. Falls back to wall time when the publisher
            # leaves the stamp empty.
            header = getattr(msg, "header", None)
            stamp_ros = getattr(header, "stamp", None) if header is not None else None
            stamp_sec = int(getattr(stamp_ros, "sec", 0) or 0)
            stamp_nsec = int(getattr(stamp_ros, "nanosec", 0) or 0)
            stamp_wall_ns = stamp_sec * 1_000_000_000 + stamp_nsec if stamp_sec else now_ns
            frame = SensorFrame(
                sensor_id=sensor_name,
                stamp_monotonic_ns=stamp_wall_ns,
                stamp_wall_ns=stamp_wall_ns,
                encoding=encoding,
                width=width,
                height=height,
                channels=3 if encoding in (FrameEncoding.RGB8, FrameEncoding.BGR8) else 1,
                data=data,
            )
            age_ms = max(0.0, (now_ns - stamp_wall_ns) / 1e6)
            tracer = trace.get_tracer("openral_world_state_ros")
            with tracer.start_as_current_span(
                semconv.SPAN_SENSORS_READ_LATEST,
                attributes={
                    semconv.SENSORS_SOURCE: sensor_name,
                },
            ) as sensor_span:
                modality = ral_producer.modality_for_encoding(encoding)
                thumb = ral_producer.encode_frame_thumbnail(frame)
                ral_producer.record_sensor_frame_attrs(
                    sensor_span,
                    modality=modality,
                    encoding=encoding.value,
                    width=width,
                    height=height,
                    channels=frame.channels,
                    age_ms=age_ms,
                    thumbnail_bytes=thumb,
                )
                self._aggregator.update_image_frame(sensor_name, frame)

        def _on_joint_state(self, msg: object) -> None:
            """Convert ROS JointState → Pydantic JointState and update aggregator."""
            import time

            from openral_core.schemas import JointState
            from sensor_msgs.msg import (
                JointState as RosJointState,  # type: ignore[import-untyped]
            )

            if self._aggregator is None:
                return
            ros_msg: RosJointState = msg  # type: ignore[assignment]
            js = JointState(
                name=list(ros_msg.name),
                position=list(ros_msg.position),
                velocity=list(ros_msg.velocity) if ros_msg.velocity else [],
                effort=list(ros_msg.effort) if ros_msg.effort else [],
                stamp_ns=time.time_ns(),
            )
            self._aggregator.update_joint_state(js)

        def _on_voxels(self, msg: object) -> None:  # OccupancyVoxels
            """Store the latest occupancy voxel grid (best-effort; cheap)."""
            self._latest_voxels = msg
            self._voxel_stamp_ns = self.get_clock().now().nanoseconds

        def _setup_depth_points_fallback(self, qos: object) -> str:
            """Subscribe the depth point cloud used as the #11 octomap-free lift source.

            Returns the resolved topic (empty when the fallback is disabled).
            """
            from sensor_msgs.msg import PointCloud2

            self._depth_max_points = (
                self.get_parameter("object_lift_depth_max_points")
                .get_parameter_value()
                .integer_value
            )
            depth_topic = (
                self.get_parameter("object_depth_points_topic").get_parameter_value().string_value
            )
            if depth_topic:
                self._depth_points_sub = self.create_subscription(
                    PointCloud2, depth_topic, self._on_depth_points, qos
                )
            return depth_topic

        def _on_depth_points(self, msg: object) -> None:  # PointCloud2
            """Store the latest depth point cloud (#11 octomap-free lift fallback)."""
            self._latest_depth_points = msg
            self._depth_points_stamp_ns = self.get_clock().now().nanoseconds

        def _depth_centers_base(self, base_frame: str, now_ns: int) -> object | None:
            """Depth-cloud points as ``(N, 3)`` in the base frame, or ``None``.

            #11 — fallback depth source for the object lift when no octomap voxel
            grid is published (e.g. ``--no-enable-octomap``). Decodes the latest
            ``sensor_msgs/PointCloud2``, drops non-finite returns, subsamples to a
            bounded count, and transforms it from the cloud's optical frame into
            the robot base frame (where the lifter expects ``occupied_centers``).
            Returns ``None`` on a missing/stale cloud or unavailable TF — the lift
            then simply skips, exactly as it does without voxels.
            """
            from openral_world_state import depth_cloud_to_centers_base

            cloud = self._latest_depth_points
            if cloud is None:
                return None
            if (
                self._voxel_staleness_ns
                and (now_ns - self._depth_points_stamp_ns) > self._voxel_staleness_ns
            ):
                return None
            try:
                from sensor_msgs_py import point_cloud2

                raw = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"))
            except Exception as exc:  # decode is best-effort; never kill the callback
                self.get_logger().debug(f"depth cloud decode failed: {exc}")
                return None
            cloud_frame = str(getattr(cloud.header, "frame_id", "") or "")  # type: ignore[attr-defined]
            t_base_from_cloud = self._lookup_4x4(base_frame, cloud_frame)
            if t_base_from_cloud is None:
                return None
            centers = depth_cloud_to_centers_base(
                raw, t_base_from_cloud, max_points=self._depth_max_points
            )
            return centers if centers.shape[0] else None

        def _lookup_4x4(self, target: str, source: str) -> object | None:
            """Latest target<-source transform as a 4x4 numpy array, or None."""
            import rclpy
            from openral_world_state import homogeneous_from_quat_xyz

            try:
                tf = self._tf_buffer.lookup_transform(  # type: ignore[union-attr]
                    target,
                    source,
                    rclpy.time.Time(),
                )
            except Exception as exc:  # tf2 raises several lookup errors
                self.get_logger().debug(f"tf {target}<-{source} unavailable: {exc}")
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

        def _sensor_spec(self, sensor_id: str) -> object | None:
            """Find a SensorSpec by name across sensors + sensor_bundles."""
            desc = self._aggregator.description  # type: ignore[union-attr]
            for s in desc.sensors:
                if s.name == sensor_id:
                    return s
            for bundle in desc.sensor_bundles:
                for s in bundle.sensors:
                    if s.name == sensor_id:
                        return s
            return None

        def _on_objects(self, msg: object) -> None:  # PromptStamped
            """Decode detections, lift to 3D, buffer candidates (best-effort)."""
            from openral_core.schemas import ObjectsMetadata
            from openral_world_state import decode_occupied_centers

            if not self._lift_enabled or self._aggregator is None:
                return
            now_ns = self.get_clock().now().nanoseconds
            grid = self._latest_voxels
            voxels_fresh = grid is not None and not (
                self._voxel_staleness_ns
                and (now_ns - self._voxel_stamp_ns) > self._voxel_staleness_ns
            )
            try:
                md = ObjectsMetadata.model_validate_json(msg.metadata_json)  # type: ignore[attr-defined]
            except Exception as exc:
                self.get_logger().debug(f"bad detection metadata_json: {exc}")
                return
            spec = self._sensor_spec(md.sensor_id)
            if spec is None or spec.intrinsics is None:
                self.get_logger().debug(f"no intrinsics for sensor '{md.sensor_id}'")
                return
            # Remember this detection camera so eviction can project its FOV
            # every memory tick from the camera pose alone (the real detector
            # publishes nothing when it sees nothing — ADR-0035).
            self._seen_sensor_ids.add(md.sensor_id)
            base_frame = self._aggregator.description.base_frame
            t_cam_from_base = self._lookup_4x4(spec.frame_id, base_frame)
            t_map_from_base = self._lookup_4x4(self._map_frame, base_frame)
            if t_cam_from_base is None or t_map_from_base is None:
                return  # best-effort: missing TF => skip
            # Prefer the octomap voxel grid (filtered, persistent); fall back to
            # the raw depth point cloud when no fresh grid exists (#11) so the
            # lift — and thus spatial-memory ingest + recall_object — works even
            # with octomap disabled. Both yield occupied centres in the base frame.
            if voxels_fresh:
                centers = decode_occupied_centers(
                    origin=(grid.origin.x, grid.origin.y, grid.origin.z),  # type: ignore[attr-defined]
                    resolution=grid.resolution,  # type: ignore[attr-defined]
                    size_xyz=(grid.size_x, grid.size_y, grid.size_z),  # type: ignore[attr-defined]
                    occupancy=bytes(grid.occupancy),  # type: ignore[attr-defined]
                )
            else:
                centers = self._depth_centers_base(base_frame, now_ns)
                if centers is None:
                    return  # no voxels and no usable depth cloud => skip
            cands = self._lifter.lift(  # type: ignore[union-attr]
                detections=md.detections,
                occupied_centers_base=centers,
                intrinsics=spec.intrinsics,
                frame_size=(md.frame_width, md.frame_height),
                t_cam_from_base=t_cam_from_base,
                t_map_from_base=t_map_from_base,
                map_frame=self._map_frame,
            )
            self._candidate_buffer.append((md.sensor_id, cands))

        def _on_memory_tick(self) -> None:
            """Associate buffered candidates, evict in-view misses, feed aggregator."""
            import numpy as np
            from openral_world_state import build_in_fov_predicate

            if not self._lift_enabled or self._aggregator is None or self._memory is None:
                return
            buffered = self._candidate_buffer
            self._candidate_buffer = []
            cands = [c for _sid, cs in buffered for c in cs]
            stamp_ns = self.get_clock().now().nanoseconds

            preds = []
            base_frame = self._aggregator.description.base_frame
            for sid in self._seen_sensor_ids:
                spec = self._sensor_spec(sid)
                if spec is None or spec.intrinsics is None:
                    continue
                t_cam_from_base = self._lookup_4x4(spec.frame_id, base_frame)
                t_map_from_base = self._lookup_4x4(self._map_frame, base_frame)
                if t_cam_from_base is None or t_map_from_base is None:
                    continue
                t_cam_from_map = t_cam_from_base @ np.linalg.inv(t_map_from_base)
                preds.append(
                    build_in_fov_predicate(
                        intrinsics=spec.intrinsics,
                        t_cam_from_map=t_cam_from_map,
                    )
                )

            def in_fov(obj: object) -> bool:
                return any(p(obj) for p in preds)

            objects = self._memory.tick(cands, stamp_ns=stamp_ns, in_fov=in_fov)
            self._aggregator.update_detected_objects(objects)

        def _publish_snapshot(self) -> None:
            """Timer callback: snapshot → typed WorldStateStamped → publish.

            Builds one ``WorldStateStamped`` per fast tick and publishes
            it on the fast topic; every ``_slow_divider`` ticks the same
            message also goes out on the slow topic (single snapshot, two
            publishers, two rates — ADR-0018 F2).
            """
            if self._aggregator is None or self._pub_fast is None or self._pub_slow is None:
                return
            ws = self._aggregator.snapshot()
            msg = build_world_state_stamped_msg(self, ws)
            self._pub_fast.publish(msg)
            self._tick_count += 1
            if self._tick_count % self._slow_divider == 0:
                self._pub_slow.publish(msg)


_IDL_DIAG_TO_STR: dict[int, str] = {
    0: "ok",  # DIAG_OK
    1: "warn",  # DIAG_WARN
    2: "stale",  # DIAG_STALE
    3: "error",  # DIAG_ERROR
}


def world_state_from_idl(msg: object) -> object:
    """Translate an ``openral_msgs.msg.WorldStateStamped`` → ``openral_core.WorldState``.

    Symmetric inverse of :func:`build_world_state_stamped_msg`. Used by
    the reasoner_node to feed real joint / ee / diagnostic state into
    its ``ContextRenderer`` instead of the previous
    ``world_state=None`` placeholder.

    The IDL carries image topic refs (parallel arrays) but no inline
    pixel bytes, so the returned ``WorldState.image_frames`` is always
    ``None`` — the colocated skill_runner picks up frames from the
    shared aggregator instance, and the cross-process reasoner does
    not need pixels for context rendering.

    Args:
        msg: An ``openral_msgs.msg.WorldStateStamped`` instance.

    Returns:
        A populated :class:`openral_core.WorldState`.
    """
    from openral_core.schemas import DetectedObject, JointState, Pose6D, WorldState

    js_msg = msg.joint_state  # type: ignore[attr-defined]
    joint_state = JointState(
        name=list(js_msg.name),
        position=list(js_msg.position),
        velocity=list(js_msg.velocity) if js_msg.velocity else [],
        effort=list(js_msg.effort) if js_msg.effort else [],
        stamp_ns=int(msg.stamp_ns),  # type: ignore[attr-defined]
    )

    base_pose: object | None = None
    if bool(msg.base_pose_valid):  # type: ignore[attr-defined]
        base_pose = _ros_pose_to_pose6d(
            msg.base_pose,  # type: ignore[attr-defined]
            frame_id=str(msg.header.frame_id),  # type: ignore[attr-defined]
        )

    base_twist: tuple[float, float, float, float, float, float] | None = None
    if bool(msg.base_twist_valid):  # type: ignore[attr-defined]
        tw = msg.base_twist  # type: ignore[attr-defined]
        base_twist = (
            float(tw.linear.x),
            float(tw.linear.y),
            float(tw.linear.z),
            float(tw.angular.x),
            float(tw.angular.y),
            float(tw.angular.z),
        )

    ee_names = list(msg.ee_names)  # type: ignore[attr-defined]
    ee_poses_msg = list(msg.ee_poses)  # type: ignore[attr-defined]
    header_frame = str(msg.header.frame_id)  # type: ignore[attr-defined]
    ee_poses = {
        name: _ros_pose_to_pose6d(p, frame_id=header_frame)
        for name, p in zip(ee_names, ee_poses_msg, strict=True)
    }

    images = dict(
        zip(
            list(msg.image_sensor_ids),  # type: ignore[attr-defined]
            list(msg.image_topics),  # type: ignore[attr-defined]
            strict=True,
        )
    )

    diag_keys = list(msg.diagnostic_keys)  # type: ignore[attr-defined]
    diag_vals = list(msg.diagnostic_statuses)  # type: ignore[attr-defined]
    diagnostics: dict[str, str] = {
        k: _IDL_DIAG_TO_STR.get(int(v), "error") for k, v in zip(diag_keys, diag_vals, strict=True)
    }

    battery_pct: float | None = (
        float(msg.battery_pct)  # type: ignore[attr-defined]
        if bool(msg.battery_valid)  # type: ignore[attr-defined]
        else None
    )

    detected_objects = [
        DetectedObject(
            label=label,
            confidence=float(conf),
            pose=Pose6D(
                xyz=(float(pos.x), float(pos.y), float(pos.z)),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=msg.detected_object_frame or "map",  # type: ignore[attr-defined]
            ),
            track_id=(int(tid) if int(tid) >= 0 else None),
        )
        for label, conf, pos, tid in zip(
            msg.detected_object_labels,  # type: ignore[attr-defined]
            msg.detected_object_confidences,  # type: ignore[attr-defined]
            msg.detected_object_positions,  # type: ignore[attr-defined]
            msg.detected_object_track_ids,  # type: ignore[attr-defined]
            strict=False,
        )
    ]

    return WorldState(
        stamp_ns=int(msg.stamp_ns),  # type: ignore[attr-defined]
        joint_state=joint_state,
        base_pose=base_pose,
        base_twist=base_twist,
        ee_poses=ee_poses,
        images=images,
        image_frames=None,
        battery_pct=battery_pct,
        diagnostics=diagnostics,
        detected_objects=detected_objects,
    )


def _ros_pose_to_pose6d(ros_pose: object, *, frame_id: str) -> object:
    """Convert ``geometry_msgs/Pose`` (position + orientation) → :class:`Pose6D`."""
    from openral_core.schemas import Pose6D

    return Pose6D(
        xyz=(
            float(ros_pose.position.x),  # type: ignore[attr-defined]
            float(ros_pose.position.y),  # type: ignore[attr-defined]
            float(ros_pose.position.z),  # type: ignore[attr-defined]
        ),
        quat_xyzw=(
            float(ros_pose.orientation.x),  # type: ignore[attr-defined]
            float(ros_pose.orientation.y),  # type: ignore[attr-defined]
            float(ros_pose.orientation.z),  # type: ignore[attr-defined]
            float(ros_pose.orientation.w),  # type: ignore[attr-defined]
        ),
        frame_id=frame_id,
    )


def build_world_state_stamped_msg(node: object, world_state: object) -> object:
    """Translate a Pydantic :class:`WorldState` into ``WorldStateStamped``.

    Pure-Python translation: parallel arrays are deterministic-ordered
    (sorted by key) so two consumers comparing snapshots at the same
    timestamp see the same byte layout.

    Args:
        node: The publishing :class:`rclpy.lifecycle.LifecycleNode`. Only
            used to stamp ``header.stamp`` via ``node.get_clock().now()``.
        world_state: An :class:`openral_core.WorldState` snapshot.

    Returns:
        A populated ``openral_msgs.msg.WorldStateStamped``.
    """
    from geometry_msgs.msg import (  # type: ignore[import-untyped]
        Pose,
        Quaternion,
        Twist,
        Vector3,
    )
    from openral_msgs.msg import WorldStateStamped  # type: ignore[import-untyped]
    from openral_observability import propagation
    from sensor_msgs.msg import JointState as RosJointState  # type: ignore[import-untyped]

    msg = WorldStateStamped()
    if node is not None:
        msg.header.stamp = node.get_clock().now().to_msg()  # type: ignore[attr-defined]
    msg.header.frame_id = ""
    msg.trace_id = propagation.current_traceparent() or ""
    msg.stamp_ns = int(world_state.stamp_ns)  # type: ignore[attr-defined]

    msg.joint_state = _build_joint_state_msg(
        world_state.joint_state,  # type: ignore[attr-defined]
        msg.header.stamp,
        RosJointState,
    )
    _fill_base_pose_twist(
        msg,
        world_state,
        pose_cls=Pose,
        quat_cls=Quaternion,
        twist_cls=Twist,
        vec3_cls=Vector3,
    )
    _fill_ee_and_images(msg, world_state, pose_cls=Pose, quat_cls=Quaternion)
    _fill_diagnostics(msg, world_state)
    _fill_battery(msg, world_state)
    _fill_detected_objects(msg, world_state)

    return msg


def _build_joint_state_msg(js: object, stamp: object, ros_joint_state_cls: type) -> object:
    """Build a ``sensor_msgs/JointState`` from an :class:`openral_core.JointState`."""
    ros_js = ros_joint_state_cls()
    ros_js.header.stamp = stamp
    ros_js.name = list(js.name)  # type: ignore[attr-defined]
    ros_js.position = list(js.position)  # type: ignore[attr-defined]
    ros_js.velocity = list(js.velocity)  # type: ignore[attr-defined]
    ros_js.effort = list(js.effort)  # type: ignore[attr-defined]
    return ros_js


def _fill_base_pose_twist(
    msg: object,
    world_state: object,
    *,
    pose_cls: type,
    quat_cls: type,
    twist_cls: type,
    vec3_cls: type,
) -> None:
    """Populate ``base_pose``/``base_twist`` and append the base frame to ``frame_ids``."""
    base_pose = world_state.base_pose  # type: ignore[attr-defined]
    if base_pose is not None:
        msg.base_pose_valid = True  # type: ignore[attr-defined]
        msg.base_pose = _pose6d_to_ros_pose(  # type: ignore[attr-defined]
            base_pose,
            pose_cls=pose_cls,
            quat_cls=quat_cls,
        )
        msg.frame_ids.append(base_pose.frame_id)  # type: ignore[attr-defined]
    else:
        msg.base_pose_valid = False  # type: ignore[attr-defined]

    base_twist = world_state.base_twist  # type: ignore[attr-defined]
    if base_twist is not None:
        vx, vy, vz, wx, wy, wz = base_twist
        msg.base_twist_valid = True  # type: ignore[attr-defined]
        msg.base_twist = twist_cls(  # type: ignore[attr-defined]
            linear=vec3_cls(x=float(vx), y=float(vy), z=float(vz)),
            angular=vec3_cls(x=float(wx), y=float(wy), z=float(wz)),
        )
    else:
        msg.base_twist_valid = False  # type: ignore[attr-defined]


def _fill_ee_and_images(
    msg: object,
    world_state: object,
    *,
    pose_cls: type,
    quat_cls: type,
) -> None:
    """Populate the EE-pose and sensor-image parallel arrays, deterministic-ordered."""
    ee_items = sorted(world_state.ee_poses.items())  # type: ignore[attr-defined]
    msg.ee_names = [name for name, _ in ee_items]  # type: ignore[attr-defined]
    msg.ee_poses = [  # type: ignore[attr-defined]
        _pose6d_to_ros_pose(pose, pose_cls=pose_cls, quat_cls=quat_cls) for _, pose in ee_items
    ]
    for _, pose in ee_items:
        if pose.frame_id and pose.frame_id not in msg.frame_ids:  # type: ignore[attr-defined]
            msg.frame_ids.append(pose.frame_id)  # type: ignore[attr-defined]

    image_items = sorted(world_state.images.items())  # type: ignore[attr-defined]
    msg.image_sensor_ids = [sid for sid, _ in image_items]  # type: ignore[attr-defined]
    msg.image_topics = [topic for _, topic in image_items]  # type: ignore[attr-defined]


def _fill_diagnostics(msg: object, world_state: object) -> None:
    """Populate the diagnostic + staleness parallel arrays.

    Staleness data lives only on the aggregator's internal ``_emit_*``
    path; the public WorldState surfaces it via ``diagnostics`` (ok /
    stale / error). The typed message therefore carries the categorical
    status (mandatory) and an empty ``staleness_*`` array by default —
    the F2 follow-up that surfaces per-component ages in the snapshot
    will land them here without an IDL bump.
    """
    diag_items = sorted(world_state.diagnostics.items())  # type: ignore[attr-defined]
    msg.diagnostic_keys = [k for k, _ in diag_items]  # type: ignore[attr-defined]
    msg.diagnostic_statuses = [  # type: ignore[attr-defined]
        _DIAG_STATUS.get(v, _DIAG_STATUS["error"]) for _, v in diag_items
    ]
    msg.staleness_keys = []  # type: ignore[attr-defined]
    msg.staleness_ms = []  # type: ignore[attr-defined]


def _fill_battery(msg: object, world_state: object) -> None:
    """Populate the battery_valid / battery_pct fields."""
    battery = world_state.battery_pct  # type: ignore[attr-defined]
    if battery is not None:
        msg.battery_valid = True  # type: ignore[attr-defined]
        msg.battery_pct = float(battery)  # type: ignore[attr-defined]
    else:
        msg.battery_valid = False  # type: ignore[attr-defined]
        msg.battery_pct = 0.0  # type: ignore[attr-defined]


def _fill_detected_objects(msg: object, world_state: object) -> None:
    """Populate the detected_object_* parallel arrays from ``WorldState.detected_objects``."""
    from geometry_msgs.msg import Point  # type: ignore[import-untyped]

    objects = world_state.detected_objects  # type: ignore[attr-defined]
    msg.detected_object_labels = [o.label for o in objects]  # type: ignore[attr-defined]
    msg.detected_object_confidences = [float(o.confidence) for o in objects]  # type: ignore[attr-defined]
    msg.detected_object_positions = [  # type: ignore[attr-defined]
        Point(x=float(o.pose.xyz[0]), y=float(o.pose.xyz[1]), z=float(o.pose.xyz[2]))
        for o in objects
    ]
    msg.detected_object_track_ids = [  # type: ignore[attr-defined]
        int(o.track_id) if o.track_id is not None else -1 for o in objects
    ]
    msg.detected_object_frame = (  # type: ignore[attr-defined]
        objects[0].pose.frame_id if objects else ""
    )


def _pose6d_to_ros_pose(pose: object, *, pose_cls: type, quat_cls: type) -> object:
    """Convert an :class:`openral_core.Pose6D` to ``geometry_msgs/Pose``.

    Pose6D already carries a quaternion (xyzw), so this is a straight
    field copy — no rpy↔quat math, no tf_transformations dependency.

    Args:
        pose: The :class:`openral_core.Pose6D` to convert.
        pose_cls: The injected ``geometry_msgs/Pose`` class (passed in
            so this helper stays importable without rclpy).
        quat_cls: The injected ``geometry_msgs/Quaternion`` class.
    """
    x, y, z = pose.xyz  # type: ignore[attr-defined]
    qx, qy, qz, qw = pose.quat_xyzw  # type: ignore[attr-defined]
    out = pose_cls()
    out.position.x = float(x)
    out.position.y = float(y)
    out.position.z = float(z)
    out.orientation = quat_cls(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    return out
