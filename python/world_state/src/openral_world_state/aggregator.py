"""WorldStateAggregator — tf2-aware, injectable snapshot producer.

Collects joint state, sensor image refs, EE poses, base pose, and battery
level from injected update callables and produces a ``WorldState`` Pydantic
snapshot on demand.

Diagnostics
-----------
Every tracked component (joint state, each sensor bundle, each EE) gets a
per-key entry in ``WorldState.diagnostics``:

- ``"ok"``    — updated within ``staleness_limit_s``
- ``"stale"`` — not updated within ``staleness_limit_s``
- ``"error"`` — reserved; set by callers via :meth:`set_error`

Staleness entries are **latched**: once a component goes stale it stays
``"stale"`` in the diagnostics dict until a fresh update arrives.  This
makes the diagnostic history visible to the Reasoner even across snapshot
boundaries.

Hot path
--------
:meth:`snapshot` is the only hot-path method.  It acquires a lock, samples
all injectable state, classifies staleness, and returns an immutable Pydantic
model.  All update methods are designed to be called from subscriber
callbacks in the ROS 2 node wrapper.

Example:
    >>> import time
    >>> from openral_core import (
    ...     RobotDescription,
    ...     EmbodimentKind,
    ...     JointSpec,
    ...     JointType,
    ...     RobotCapabilities,
    ...     SafetyEnvelope,
    ...     ControlMode,
    ... )
    >>> from openral_core.schemas import JointState
    >>> from openral_world_state.aggregator import WorldStateAggregator
    >>> desc = RobotDescription(
    ...     name="test",
    ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
    ...     joints=[
    ...         JointSpec(
    ...             name="j0", joint_type=JointType.REVOLUTE, parent_link="base", child_link="link0"
    ...         )
    ...     ],
    ...     capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
    ...     safety=SafetyEnvelope(),
    ... )
    >>> agg = WorldStateAggregator(desc)
    >>> agg.update_joint_state(JointState(name=["j0"], position=[0.0], stamp_ns=time.time_ns()))
    >>> ws = agg.snapshot()
    >>> ws.joint_state.name
    ['j0']
    >>> ws.diagnostics["joint_state"]
    'ok'
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Literal

import structlog
from openral_core.schemas import (
    DetectedObject,
    JointState,
    Pose6D,
    RobotDescription,
    SensorFrame,
    WorldState,
)
from openral_observability import metrics as ral_metrics
from openral_observability import producer as _producer
from openral_observability import semconv
from opentelemetry import trace

__all__ = ["WorldStateAggregator"]


def _tracer() -> trace.Tracer:
    # Resolved per call — never cache at module import. Caching binds to
    # whatever TracerProvider was global at import time and silently
    # swallows spans when downstream code (or tests) swap the provider.
    return trace.get_tracer("openral")


log = structlog.get_logger(__name__)

DiagStatus = Literal["ok", "warn", "stale", "error"]

# Snapshot rate advertised in diagnostics; not enforced here (ROS node does that).
DEFAULT_RATE_HZ: float = 30.0
# One staleness window covers every heterogeneous-rate component (30 Hz joint
# state, 10 Hz cameras/depth). 0.1 s == the camera period (10 Hz), so any
# scheduling jitter made the camera diagnostics flap OK↔STALE every snapshot.
# 0.5 s gives a comfortable margin over the slowest expected stream while still
# flagging a genuinely dead component. This is a freshness indicator, not a
# safety gate (the C++ kernel owns enforcement).
DEFAULT_STALENESS_S: float = 0.5


class WorldStateAggregator:
    """Aggregates sensor data and produces ``WorldState`` snapshots.

    The aggregator holds no ROS 2 imports.  All data arrives via typed update
    methods called from subscriber callbacks in the enclosing ROS 2 lifecycle
    node.  :meth:`snapshot` can be called from any thread; internal state is
    protected by a reentrant lock.

    Args:
        description: Normative ``RobotDescription`` for the target robot.
            Used to initialise diagnostic keys for all declared sensor bundles
            and end-effectors.
        staleness_limit_s: Maximum age (seconds) for a component reading
            before it is classified as ``"stale"``.  Default ``0.1 s``
            (3 frames at 30 Hz).
        clock_fn: Callable returning the current time in nanoseconds.
            Defaults to ``time.time_ns``.  Override in tests to control time.

    Example:
        >>> from openral_core.schemas import JointState
        >>> from openral_world_state.aggregator import WorldStateAggregator
        >>> import time
        >>> from openral_core import (
        ...     RobotDescription,
        ...     EmbodimentKind,
        ...     JointSpec,
        ...     JointType,
        ...     RobotCapabilities,
        ...     SafetyEnvelope,
        ...     ControlMode,
        ... )
        >>> desc = RobotDescription(
        ...     name="t",
        ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
        ...     joints=[
        ...         JointSpec(
        ...             name="j0",
        ...             joint_type=JointType.REVOLUTE,
        ...             parent_link="base",
        ...             child_link="l0",
        ...         )
        ...     ],
        ...     capabilities=RobotCapabilities(
        ...         supported_control_modes=[ControlMode.JOINT_POSITION]
        ...     ),
        ...     safety=SafetyEnvelope(),
        ... )
        >>> agg = WorldStateAggregator(desc)
        >>> agg.update_joint_state(JointState(name=["j0"], position=[0.5], stamp_ns=time.time_ns()))
        >>> agg.snapshot().joint_state.position
        [0.5]
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        staleness_limit_s: float = DEFAULT_STALENESS_S,
        clock_fn: Callable[[], int] | None = None,
    ) -> None:
        """Initialise the aggregator; does not open any connection."""
        self.description = description
        self._staleness_limit_ns: int = int(staleness_limit_s * 1e9)
        self._clock_fn: Callable[[], int] = clock_fn or time.time_ns
        self._lock = threading.RLock()

        # ── Tracked state (all protected by _lock) ────────────────────────
        self._joint_state: JointState | None = None
        self._joint_state_stamp_ns: int = 0

        # sensor_name → (topic_ref, stamp_ns)
        self._images: dict[str, tuple[str, int]] = {}
        # sensor_name → (SensorFrame, stamp_ns) — actual bytes for the
        # consumer that wants pixels in-process (skill_runner →
        # rSkill). Populated by `update_image_frame`; surfaced through
        # `WorldState.image_frames` so a Skill can read pixels without
        # opening its own ROS subscription.
        self._image_frames: dict[str, tuple[SensorFrame, int]] = {}
        # ee_name → (Pose6D, stamp_ns)
        self._ee_poses: dict[str, tuple[Pose6D, int]] = {}
        # base pose + stamp
        self._base_pose: Pose6D | None = None
        self._base_pose_stamp_ns: int = 0
        self._base_twist: tuple[float, float, float, float, float, float] | None = None
        # battery
        self._battery_pct: float | None = None
        # ADR-0035 — latest object-memory snapshot (already deduped/evicted by
        # ObjectMemory in the enclosing node). Stored verbatim; no staleness
        # logic here (the memory owns lifecycle and refreshes every tick).
        self._detected_objects: list[DetectedObject] = []
        # latched diagnostics for explicitly set errors
        self._forced_errors: dict[str, DiagStatus] = {}
        # Stale components from the previous snapshot — used by snapshot() to
        # emit a ``staleness_latched`` span event only on the tick where a
        # component first goes stale (not every tick it stays stale).
        self._prev_stale_components: set[str] = set()
        # Same for latched errors so we don't re-emit on every snapshot.
        self._prev_latched_errors: set[str] = set()

        # Initialise diagnostic keys from description
        self._sensor_names: set[str] = {
            s.name for bundle in description.sensor_bundles for s in bundle.sensors
        }
        # End-effector poses are lazily registered (like cameras, see
        # update_ee_pose / update_image_frame): a declared EE only appears in
        # the diagnostics once it has received at least one pose. Pre-populating
        # from description.end_effectors meant every robot reported its
        # gripper(s) permanently STALE whenever no pose source was wired (the
        # tf2/forward-kinematics EE feed is not active on the sim deploy path),
        # which is pure noise. Track what the description *declares* separately
        # for reference/logging without forcing it into the stale ledger.
        self._declared_ee_names: frozenset[str] = (
            frozenset(ee.name for ee in description.end_effectors)
            if description.end_effectors
            else frozenset()
        )
        self._ee_names: set[str] = set()

        log.info(
            "world_state.aggregator.init",
            robot=description.name,
            staleness_limit_s=staleness_limit_s,
            sensor_count=len(self._sensor_names),
            ee_count=len(self._declared_ee_names),
        )

    # ── Update methods (called from ROS 2 subscriber callbacks) ──────────────

    def update_joint_state(self, state: JointState) -> None:
        """Record a fresh joint state reading.

        Args:
            state: Latest ``JointState`` from the ``/joint_states`` topic.
        """
        with self._lock:
            self._joint_state = state
            self._joint_state_stamp_ns = self._clock_fn()
        log.debug("world_state.joint_state.updated", robot=self.description.name)

    def update_image(self, sensor_name: str, topic: str, stamp_ns: int) -> None:
        """Record a fresh image frame arrival for a named sensor.

        Args:
            sensor_name: Matches a ``SensorSpec.name`` in the description.
            topic: ROS 2 topic the image was published on.
            stamp_ns: Frame timestamp in nanoseconds.
        """
        with self._lock:
            self._images[sensor_name] = (topic, self._clock_fn())
        log.debug("world_state.image.updated", sensor=sensor_name)

    def update_image_frame(self, sensor_name: str, frame: SensorFrame) -> None:
        """Record an inline pixel payload for a named sensor.

        Unlike :meth:`update_image` (which just records "a frame arrived
        on topic X"), this method stores the actual :class:`SensorFrame`
        — usually with ``frame.data`` set — so the next
        :meth:`snapshot` carries pixels inline through
        :attr:`WorldState.image_frames`. This is the path a Skill uses
        when it needs RGB pixels without opening its own ROS
        subscription (CLAUDE.md §6.1 — Layer 1 (Sensors) writes through
        Layer 2 (World State) into Layer 3 (rSkill)).

        Accepts arbitrary ``sensor_name`` values not declared in the
        :class:`RobotDescription` ``sensor_bundles`` list — synthetic
        digital-twin cameras live in the active MJCF, not the robot
        manifest, and the aggregator must aggregate whatever streams
        in.

        Args:
            sensor_name: Camera id; used as the key in
                :attr:`WorldState.image_frames`.
            frame: Validated :class:`SensorFrame` (``data`` /
                ``topic`` / ``handle``).
        """
        stamp = self._clock_fn()
        with self._lock:
            self._image_frames[sensor_name] = (frame, stamp)
            # Mirror into `_images` so the diagnostics + the
            # `WorldState.images` topic map stay populated even for
            # sensors that are not in the robot.yaml. Use the frame's
            # `topic` when set, otherwise a placeholder.
            self._images[sensor_name] = (
                frame.topic if frame.topic is not None else f"<inline:{sensor_name}>",
                stamp,
            )
            self._sensor_names.add(sensor_name)
        log.debug(
            "world_state.image_frame.updated",
            sensor=sensor_name,
            encoding=str(frame.encoding),
            width=frame.width,
            height=frame.height,
        )

    def update_ee_pose(self, ee_name: str, pose: Pose6D) -> None:
        """Record a fresh end-effector pose from a tf2 lookup.

        Args:
            ee_name: Matches an ``EndEffectorSpec.name`` in the description.
            pose: Current 6D pose in the world frame.
        """
        with self._lock:
            self._ee_poses[ee_name] = (pose, self._clock_fn())
            # Lazy registration: only EEs that have actually been observed enter
            # the diagnostics ledger (see __init__ for why pre-population is noise).
            self._ee_names.add(ee_name)
        log.debug("world_state.ee_pose.updated", ee=ee_name)

    def update_base_pose(
        self,
        pose: Pose6D,
        twist: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """Record a fresh base link pose (and optional twist) from tf2.

        Args:
            pose: Base link 6D pose in the world/map frame.
            twist: Optional (vx, vy, vz, wx, wy, wz) twist.
        """
        with self._lock:
            self._base_pose = pose
            self._base_pose_stamp_ns = self._clock_fn()
            self._base_twist = twist
        log.debug("world_state.base_pose.updated")

    def update_battery(self, pct: float) -> None:
        """Record the latest battery percentage.

        Args:
            pct: Battery percentage in [0, 100].
        """
        with self._lock:
            self._battery_pct = pct

    def update_detected_objects(self, objects: list[DetectedObject]) -> None:
        """Replace the remembered detected-object set (ADR-0035).

        Called from the world-state node's memory tick with the current
        ``ObjectMemory`` output (already associated, frozen, and evicted). A
        copy is stored so later external mutation of ``objects`` cannot alter
        the snapshot.

        Args:
            objects: Current spatial-memory objects, anchored in the map frame.
        """
        with self._lock:
            self._detected_objects = list(objects)
        log.debug("world_state.detected_objects.updated", count=len(objects))

    def set_error(self, component: str, status: DiagStatus = "error") -> None:
        """Latch an explicit diagnostic status for a named component.

        Use to surface hardware faults or driver errors that go beyond
        mere staleness.  The forced status persists until :meth:`clear_error`
        is called.

        Args:
            component: Diagnostic key (e.g. ``"joint_state"``, sensor name).
            status: Diagnostic level to latch; typically ``"error"`` or
                ``"warn"``.
        """
        with self._lock:
            self._forced_errors[component] = status
        log.warning("world_state.error.latched", component=component, status=status)

    def clear_error(self, component: str) -> None:
        """Remove a forced diagnostic entry for a named component.

        Args:
            component: Diagnostic key to clear.
        """
        with self._lock:
            self._forced_errors.pop(component, None)

    # ── Snapshot (hot path) ───────────────────────────────────────────────────

    def snapshot(self) -> WorldState:
        """Produce a typed ``WorldState`` snapshot from current aggregated data.

        Staleness is evaluated at call time.  Components older than
        ``staleness_limit_s`` appear as ``"stale"`` in
        ``WorldState.diagnostics``.  Latched errors override staleness.

        Emits a ``world_state.snapshot`` OTel span recording
        ``openral.world_state.components_stale`` and
        ``openral.world_state.has_latched_error``. When a component first
        transitions to stale (or first acquires a latched error) the span
        carries a ``openral.event.staleness_latched`` (or
        ``..._error_latched``) event. Per-component staleness ages are
        recorded on the ``openral.world_state.staleness_ms`` histogram.

        Returns:
            An immutable ``WorldState`` snapshot.

        Raises:
            RuntimeError: If no joint state has ever been received (``None``
                state would make the snapshot unusable).
        """
        with (
            self._lock,
            _tracer().start_as_current_span(semconv.SPAN_WORLD_STATE_SNAPSHOT) as span,
        ):
            now_ns = self._clock_fn()
            diag: dict[str, str] = {}
            ages_ms: dict[str, float] = {}

            # Joint state
            if self._joint_state is None:
                # Create a zeroed state so Skills always get a valid object;
                # mark diagnostic as stale.
                joint_count = len(self.description.joints)
                joint_names = [j.name for j in self.description.joints]
                js = JointState(
                    name=joint_names,
                    position=[0.0] * joint_count,
                    velocity=[0.0] * joint_count,
                    effort=[0.0] * joint_count,
                    stamp_ns=now_ns,
                )
                diag["joint_state"] = "stale"
            else:
                js = self._joint_state
                age_ns = now_ns - self._joint_state_stamp_ns
                diag["joint_state"] = "ok" if age_ns <= self._staleness_limit_ns else "stale"
                ages_ms["joint_state"] = age_ns / 1e6

            # Images — topic refs from last received frames
            images: dict[str, str] = {}
            for sensor_name in self._sensor_names:
                if sensor_name in self._images:
                    topic, stamp = self._images[sensor_name]
                    age_ns = now_ns - stamp
                    diag[sensor_name] = "ok" if age_ns <= self._staleness_limit_ns else "stale"
                    images[sensor_name] = topic
                    ages_ms[sensor_name] = age_ns / 1e6
                else:
                    diag[sensor_name] = "stale"

            # EE poses
            ee_poses: dict[str, Pose6D] = {}
            for ee_name in self._ee_names:
                if ee_name in self._ee_poses:
                    pose, stamp = self._ee_poses[ee_name]
                    age_ns = now_ns - stamp
                    diag[ee_name] = "ok" if age_ns <= self._staleness_limit_ns else "stale"
                    ee_poses[ee_name] = pose
                    ages_ms[ee_name] = age_ns / 1e6
                else:
                    diag[ee_name] = "stale"

            # Forced errors override staleness classification
            diag.update(self._forced_errors)

            self._emit_snapshot_telemetry(span, diag, ages_ms)
            # Surface the embodiment view on the same span the dashboard
            # consumes — ee poses (named) + battery + joint stamp + the
            # per-component diagnostics list. Stale data still flows so
            # the dashboard can render a warn pill on the stale row.
            _producer.record_ee_poses(span, ee_poses)
            if self._battery_pct is not None:
                span.set_attribute("openral.world_state.battery_pct", float(self._battery_pct))
            span.set_attribute("openral.world_state.diagnostics_keys", sorted(diag.keys()))
            span.set_attribute(
                "openral.world_state.diagnostics_values",
                [diag[k] for k in sorted(diag.keys())],
            )
            span.set_attribute(semconv.HAL_JOINT_NAMES, list(js.name))
            span.set_attribute(
                semconv.HAL_JOINT_POSITIONS,
                [round(float(v), 3) for v in js.position],
            )

            # Inline pixel payloads: surface every camera that fed
            # `update_image_frame` since the last reset. The Pydantic
            # field is Optional; pass None when nothing has arrived so
            # the snapshot stays cheap when no cameras are wired.
            image_frames: dict[str, SensorFrame] | None = None
            if self._image_frames:
                image_frames = {n: f for n, (f, _) in self._image_frames.items()}

            return WorldState(
                stamp_ns=now_ns,
                joint_state=js,
                base_pose=self._base_pose,
                base_twist=self._base_twist,
                ee_poses=ee_poses,
                images=images,
                image_frames=image_frames,
                battery_pct=self._battery_pct,
                diagnostics=diag,
                detected_objects=list(self._detected_objects),
            )

    def _emit_snapshot_telemetry(
        self,
        span: trace.Span,
        diag: dict[str, str],
        ages_ms: dict[str, float],
    ) -> None:
        """Lift this tick's snapshot diagnostics onto the span + metric instruments.

        Called from inside :meth:`snapshot` under ``self._lock``. Compares
        the current stale / latched-error sets against the previous tick
        to fire ``openral.event.staleness_latched`` /
        ``openral.event.error_latched`` events only on transitions.
        """
        stale_now = {k for k, v in diag.items() if v == "stale"}
        errors_now = set(self._forced_errors)
        components_stale = len(stale_now)
        has_latched_error = bool(errors_now)

        span.set_attribute(semconv.WORLD_STATE_COMPONENTS_STALE, components_stale)
        span.set_attribute(semconv.WORLD_STATE_HAS_LATCHED_ERROR, has_latched_error)

        # Per-component staleness histogram. Ages are only known for
        # components we've seen at least one update from; never-seen
        # components contribute to ``components_stale`` but not to the
        # histogram (they have no age to record).
        staleness_hist = ral_metrics.get_world_state_staleness_ms()
        for component, age_ms in ages_ms.items():
            ral_metrics.record_histogram_ms(
                staleness_hist,
                age_ms,
                attributes={semconv.LABEL_COMPONENT: component},
            )

        # Up-down counter mirrors the current stale-set size.
        delta = components_stale - len(self._prev_stale_components)
        if delta != 0:
            ral_metrics.get_world_state_components_stale().add(delta)

        # Transition-only events: stale on this tick that wasn't stale last
        # tick, error latched this tick that wasn't latched last tick.
        for component in sorted(stale_now - self._prev_stale_components):
            attrs: dict[str, str | float] = {semconv.WORLD_STATE_COMPONENT: component}
            if component in ages_ms:
                attrs[semconv.SENSORS_AGE_MS] = ages_ms[component]
            span.add_event(semconv.EVENT_STALENESS_LATCHED, attributes=attrs)

        for component in sorted(errors_now - self._prev_latched_errors):
            span.add_event(
                semconv.EVENT_ERROR_LATCHED,
                attributes={
                    semconv.WORLD_STATE_COMPONENT: component,
                    "status": self._forced_errors[component],
                },
            )

        self._prev_stale_components = stale_now
        self._prev_latched_errors = errors_now
