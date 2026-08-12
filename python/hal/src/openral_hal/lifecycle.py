"""Generic ROS 2 managed lifecycle node wrapper for any HAL adapter.

Wraps any :class:`openral_hal.protocol.HAL` Protocol implementation as a
``rclpy.lifecycle.LifecycleNode`` so every per-robot package (``UR5e``,
``UR10e``, ``FrankaPanda``, ``SO100Follower``, ``OpenArm``, …) shares
the same publisher / subscriber / heartbeat / OTel-span wiring.

There are three ways to use this module, in decreasing order of preference:

1. **Manifest-driven** (preferred — issue #191): call
   :func:`make_lifecycle_main_from_manifest`, which spins up the generic
   :class:`ManifestHALLifecycleNode`. It reads ``robot_yaml`` + ``hal_mode``
   ROS parameters and builds its HAL through :func:`openral_hal.build_hal`,
   so a robot's construction kwargs (serial ``port``, ``robot_ip``, …) live
   in the manifest's ``hal.parameters.defaults`` block rather than
   a per-robot subclass. Adding a robot needs only a ``robot.yaml`` + a HAL
   class + a registry entry — no new node class.

2. **Zero-parameter HALs** (legacy): call :func:`make_lifecycle_main` with a
   callable that returns a fresh HAL instance. Suitable for adapters whose
   constructor has no ROS parameters worth exposing; superseded by (1) for
   robots whose manifest declares ``hal.sim`` / ``hal.real``.

3. **Bespoke parameterised HALs** (OpenArm cameras / viewer / MJCF scene;
   panda_mobile mobile base): subclass :class:`HALLifecycleNodeBase` and
   implement :meth:`HALLifecycleNodeBase._create_hal` plus the optional hooks
   (:meth:`_heartbeat_extra_fields`,
   :meth:`on_configure_post_hal`,
   :meth:`on_activate_post_subs`,
   :meth:`on_deactivate_pre_teardown`,
   :meth:`on_cleanup_pre_disconnect`). Tracked for collapse into (1) under
   issue #191 (Phases 2-3).

Either way, the base class owns:

* The standard publishers (``/joint_states`` + ``~/joint_states``).
* The standard subscribers (``/openral/safe_action``,
  ``/openral/estop``).
* The 1 Hz ``DiagnosticsHeartbeat``.
* The per-tick OTel ``hal.read_state`` + ``hal.send_action`` spans
  consumed by the live dashboard's Robot State / Commands / Identity
  cards.
* The estop latch (CLAUDE.md §1.5 defense in depth).

ROS 2 imports are deferred so this module imports cleanly without a
live ROS 2 installation (e.g. pure-Python CI / linting).

Lifecycle transitions
---------------------
- ``configure``  → construct the HAL (via :meth:`_create_hal`) and call
  ``connect()``; then run :meth:`on_configure_post_hal`.
- ``activate``   → start the joint-state publish timer + safe_action +
  estop subscriptions; then run :meth:`on_activate_post_subs`.
- ``deactivate`` → :meth:`on_deactivate_pre_teardown`; stop timers /
  destroy subs+pubs.
- ``cleanup``    → :meth:`on_cleanup_pre_disconnect`; call
  ``disconnect()`` on the HAL.
- ``shutdown``   → force-disconnect.

Example (UR5e — zero-parameter)::

    # In each per-robot package's lifecycle_node.py:
    from openral_hal.lifecycle import make_lifecycle_main
    from openral_hal import UR5eHAL

    main = make_lifecycle_main(
        node_name="openral_hal_ur5e",
        hal_factory=UR5eHAL,
    )

Example (SO-100 / franka — manifest-driven, the preferred path)::

    # In each per-robot package's lifecycle_node.py — no subclass needed:
    from openral_hal.lifecycle import make_lifecycle_main_from_manifest

    main = make_lifecycle_main_from_manifest(node_name="openral_hal_so100")
    # `openral deploy sim` injects `robot_yaml` + `hal_mode=sim`; real-HAL
    # construction kwargs (the SO-100's serial `port`) live in the manifest's
    # `hal.parameters` block, threaded by build_hal.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Any

from openral_hal.mobile_base_bridge import describes_mobile_base

if TYPE_CHECKING:
    from openral_core import RobotDescription

    from openral_hal.protocol import HAL

__all__ = [
    "HALLifecycleNodeBase",
    "ManifestHALLifecycleNode",
    "decode_action_chunk",
    "make_lifecycle_main",
    "make_lifecycle_main_from_manifest",
]


def decode_action_chunk(msg: object) -> object | None:
    """Reverse the action-chunk wire encoding back into a typed ``Action``.

    The publisher (``ros_publishing_hal._flatten_action_payload``) packs
    the typed :class:`openral_core.schemas.Action` into ``ActionChunk``'s
    ``flat`` + ``n_dof`` + ``horizon`` + ``control_mode`` fields. This
    is the inverse — used by the HAL lifecycle node's
    ``_on_safe_action`` callback after the C++ safety kernel
    republishes the clamped chunk on ``/openral/safe_action``.

    Returns ``None`` when the chunk is degenerate (empty flat, n_dof
    ≤ 0) or carries a ``control_mode`` not on the F1/F5 wire
    (``CARTESIAN_POSE``, ``FOOT_PLACEMENT``, ``DEX_HAND_JOINT``). The
    caller can drop or log; the kernel won't have produced one of
    those modes because the publisher rejects them at the encode side.

    Lives at module scope (not inside the rclpy guard) so the unit
    tests in ``python/hal/tests`` can exercise it without a ROS 2
    install. ``msg`` is duck-typed (``rosidl``-generated classes have
    no ``py.typed`` marker); we only read ``getattr`` fields.
    """
    from openral_core.schemas import UINT8_TO_CONTROL_MODE, Action, ControlMode

    flat = list(getattr(msg, "flat", []) or [])
    n_dof = int(getattr(msg, "n_dof", 0) or 0)
    if n_dof <= 0 or not flat:
        return None
    horizon = max(int(getattr(msg, "horizon", 1) or 1), 1)
    mode_uint = int(getattr(msg, "control_mode", 0) or 0)
    mode = UINT8_TO_CONTROL_MODE.get(mode_uint, ControlMode.JOINT_POSITION)

    rows: list[list[float]] = [flat[s * n_dof : (s + 1) * n_dof] for s in range(horizon)]
    kwargs: dict[str, Any] = {"control_mode": mode, "horizon": horizon}
    kwargs["ee_name"] = str(getattr(msg, "ee_name", "") or "") or None
    kwargs["frame_id"] = str(getattr(msg, "frame_id", "") or "") or None
    # Decode the wire value verbatim — no falsy-zero guard: confidence=0.0 is
    # a schema-legal "the policy explicitly disowns this action" and coercing
    # it to 1.0 would record/act on full confidence. A missing attribute
    # (pre-confidence IDL) still defaults to the schema's 1.0; an unset wire
    # field decodes as the IDL default 0.0, which errs in the safe (low-
    # confidence) direction.
    confidence_raw = getattr(msg, "confidence", None)
    kwargs["confidence"] = 1.0 if confidence_raw is None else float(confidence_raw)
    kwargs["tick_index"] = int(getattr(msg, "tick_index", 0) or 0)
    kwargs["tick_group_size"] = max(int(getattr(msg, "tick_group_size", 1) or 1), 1)
    if mode in (ControlMode.JOINT_POSITION, ControlMode.JOINT_TRAJECTORY):
        kwargs["joint_targets"] = rows
    elif mode is ControlMode.JOINT_VELOCITY:
        kwargs["joint_velocities"] = rows
    elif mode is ControlMode.JOINT_TORQUE:
        kwargs["joint_torques"] = rows
    elif mode is ControlMode.CARTESIAN_DELTA:
        kwargs["cartesian_delta"] = [tuple(r) for r in rows]
        scale = tuple(float(v) for v in (getattr(msg, "cartesian_delta_scale", []) or []))
        kwargs["cartesian_delta_scale"] = scale or None
    elif mode is ControlMode.CARTESIAN_TWIST:
        kwargs["cartesian_twist"] = [tuple(r) for r in rows]
    elif mode is ControlMode.BODY_TWIST:
        kwargs["body_twist"] = [tuple(r) for r in rows]
    elif mode in (ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION):
        # Gripper wire: flat is a horizon-long 1-D list (n_dof=1). The
        # typed ``Action.gripper`` field is the flat list itself, not
        # nested rows.
        kwargs["gripper"] = [float(v) for v in flat[:horizon]]
    elif mode is ControlMode.COMPOSITE_MODE:
        # Sim-only mux flag. Same wire layout as gripper
        # (n_dof=1, horizon 1-D values).
        kwargs["composite_mode"] = [float(v) for v in flat[:horizon]]
    else:
        return None
    return Action(**kwargs)


log = logging.getLogger(__name__)

try:
    import rclpy
    from openral_observability import log_lifecycle_errors
    from rclpy.executors import ExternalShutdownException
    from rclpy.lifecycle import (
        LifecycleNode,
        TransitionCallbackReturn,
    )

    from openral_hal.proprio_snapshot import ProprioFrame, ProprioSnapshot

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


HALFactory = Callable[..., "HAL"]


@contextmanager
def _hal_duration_metric(metric_name: str, hal_adapter: str) -> Iterator[None]:
    """Time a HAL call and record it on the matching duration histogram.

    Pairs with the `hal.read_state` / `hal.send_action` spans so trace and
    metric are emitted from one place. Both instruments were previously
    recorded only by :class:`openral_runner.DeployRunner`, which the ROS
    deploy graph does not instantiate — `rskill_runner_node` runs its own
    tick loop — so a live `openral deploy run` produced the spans and no
    latency histogram whatsoever.

    Never raises: an observability probe must not be able to disturb the
    control loop (CLAUDE.md §1.1).
    """
    started = perf_counter()
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            from openral_observability import metrics as _metrics
            from openral_observability import semconv

            getter = {
                semconv.METRIC_HAL_READ_STATE_DURATION: _metrics.get_hal_read_state_duration,
                semconv.METRIC_HAL_SEND_ACTION_DURATION: _metrics.get_hal_send_action_duration,
            }[metric_name]
            _metrics.record_histogram_ms(
                getter(),
                (perf_counter() - started) * 1000.0,
                {semconv.LABEL_HAL_ADAPTER: hal_adapter},
            )


def _hal_service_name(node_name: str) -> str:
    """Map a HAL ROS node name to a dotted ``service.name`` resource attribute.

    Mirrors the convention used by the other OpenRAL ROS nodes
    (``openral.reasoner``, ``openral.prompt_router``): ``openral_hal_franka``
    becomes ``openral.hal.franka`` so the dashboard's Identity card and span
    ``service.name`` reads cleanly per robot.
    """
    return "openral.hal." + node_name.removeprefix("openral_hal_")


def make_lifecycle_main(
    node_name: str,
    hal_factory: HALFactory,
) -> Callable[[], None]:
    """Build a ``main()`` entry point for a HAL lifecycle node.

    Args:
        node_name: ROS 2 node name (e.g. ``"openral_hal_ur5e"``).
        hal_factory: Zero-argument callable returning a fresh HAL
            instance. For HALs with ROS-parameterised constructors,
            subclass :class:`HALLifecycleNodeBase` directly and
            implement :meth:`_create_hal`.

    Returns:
        A zero-argument ``main()`` callable suitable as a console-script
        entry point.
    """

    def main() -> None:
        if not _ROS2_AVAILABLE:
            log.error("rclpy not found — cannot start lifecycle node without ROS 2.")
            raise SystemExit(1)

        # Install the OTLP exporters BEFORE rclpy.init() so the per-tick
        # `hal.read_state` / `hal.send_action` spans (and the SimSensorBridge
        # `sensors.read_latest` spans) reach the live dashboard. Without this
        # the HAL node creates spans against the global no-op TracerProvider and
        # the dashboard's Robot-state / Commands / Identity cards stay blank
        # even though `OTEL_EXPORTER_OTLP_ENDPOINT` is set in the node's env.
        # Idempotent + no-op when the endpoint env var is unset.
        from openral_observability import configure_observability

        configure_observability(service_name=_hal_service_name(node_name))

        rclpy.init()
        node = _FactoryHALLifecycleNode(node_name, hal_factory)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy; on
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread spin raises ExternalShutdownException instead. The
            # context is already down by the time we reach `finally`, so
            # the bare `rclpy.shutdown()` we used to call there raised
            # `RCLError: rcl_shutdown already called` — switched to the
            # idempotent `try_shutdown()` below.
            pass
        finally:
            node.destroy_node()
            # Idempotent — no-op when the SIGINT handler (or whoever
            # fired ExternalShutdownException) already shut the context.
            rclpy.try_shutdown()

    return main


def make_lifecycle_main_from_manifest(node_name: str) -> Callable[[], None]:
    """Build a ``main()`` for a manifest-driven HAL lifecycle node.

    Unlike :func:`make_lifecycle_main` (which pins a single hardcoded HAL
    class), the returned node reads two ROS parameters and constructs its HAL
    through the one resolver seam :func:`openral_hal.build_hal`:

    * ``robot_yaml`` (str, required) — path to ``robots/<id>/robot.yaml``.
    * ``hal_mode`` (str, default ``"sim"``) — ``"sim"`` (``deploy sim`` / the
      ``sim run`` harness) or ``"real"`` (``deploy run``, real hardware).

    So a single node serves both modes for every robot, and "add a robot"
    needs no per-package HAL class wiring — just a manifest declaring
    ``hal.sim`` / ``hal.real``. A robot whose manifest lacks the
    requested mode raises ``ROSCapabilityMismatch`` at configure time.

    Args:
        node_name: ROS 2 node name (e.g. ``"openral_hal_franka"``).

    Returns:
        A zero-argument ``main()`` console-script entry point.
    """

    def main() -> None:
        if not _ROS2_AVAILABLE:
            log.error("rclpy not found — cannot start lifecycle node without ROS 2.")
            raise SystemExit(1)

        # Install the OTLP exporters BEFORE rclpy.init() so the per-tick
        # `hal.read_state` / `hal.send_action` spans (and the SimSensorBridge
        # `sensors.read_latest` spans) reach the live dashboard. Without this
        # the HAL node creates spans against the global no-op TracerProvider and
        # the dashboard's Robot-state / Commands / Identity cards stay blank
        # even though `OTEL_EXPORTER_OTLP_ENDPOINT` is set in the node's env.
        # Idempotent + no-op when the endpoint env var is unset.
        from openral_observability import configure_observability

        configure_observability(service_name=_hal_service_name(node_name))

        rclpy.init()
        node = ManifestHALLifecycleNode(node_name)
        # Deliberately single-threaded. MuJoCo's EGL/GL context is
        # thread-affine, so a MultiThreadedExecutor (whose worker pool hops
        # threads between callbacks) crashes env.step with EGLError. Instead the
        # node offloads odom/joint_state to a dedicated publisher thread reading
        # the proprio snapshot, keeping all env.step / render on this one thread.
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy; on
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread spin raises ExternalShutdownException instead. The
            # context is already down by the time we reach `finally`, so
            # the bare `rclpy.shutdown()` we used to call there raised
            # `RCLError: rcl_shutdown already called` — switched to the
            # idempotent `try_shutdown()` below.
            pass
        finally:
            node.destroy_node()
            # Idempotent — no-op when the SIGINT handler (or whoever
            # fired ExternalShutdownException) already shut the context.
            rclpy.try_shutdown()

    return main


if _ROS2_AVAILABLE:

    class HALLifecycleNodeBase(LifecycleNode):  # type: ignore[misc]  # reason: rclpy is untyped at runtime
        """Generic managed lifecycle node base class wrapping a HAL adapter.

        Subclasses **must** override :meth:`_create_hal`. The other hook
        methods (``_heartbeat_extra_fields``, ``on_configure_post_hal``,
        ``on_activate_post_subs``, ``on_deactivate_pre_teardown``,
        ``on_cleanup_pre_disconnect``) have empty defaults — override
        only what's robot-specific (cameras, viewer, MJCF scene, …).

        See the module docstring for the full lifecycle contract.
        """

        def __init__(self, node_name: str) -> None:
            """Declare the standard ``publish_rate_hz`` parameter; opens no resources."""
            super().__init__(node_name)
            self._node_name = node_name
            self._hal: HAL | None = None
            self._timer: Any = None
            self._publisher: Any = None
            self._joint_state_pub: Any = None
            self._policy_state_pub: Any = None
            # Identity of the last ProprioFrame whose policy_state was
            # published. Frames are immutable and freshly constructed per
            # ``env.step`` capture, so publishing only on a NEW frame makes
            # /openral/policy_state a true "the simulator stepped" signal —
            # republishing the latch every timer tick would keep the
            # aggregator's staleness stamp fresh forever and the downstream
            # ROSPerceptionStale gate could never fire on a wedged sim.
            self._policy_state_last_frame: Any = None
            self._safe_action_sub: Any = None
            self._action_applied_pub: Any = None
            self._last_action_applied_tick: int = 0
            self._deferred_action_applied_tick: int = 0
            self._safe_group_tick: int | None = None
            self._safe_group_count: int = 0
            self._estop_sub: Any = None
            self._estop_reset_sub: Any = None
            # Decouple the cheap, latency-sensitive publishers (odom /
            # joint_state / TF) from the single executor thread, which is
            # head-of-line-blocked by env.step + render + scan raycast. They run
            # on a dedicated publisher thread reading ``_proprio`` (a plain-data
            # snapshot captured after each step), never touching MjData/GL off the
            # executor thread. A MultiThreadedExecutor was rejected: MuJoCo's
            # EGL/GL context is thread-affine, so callbacks hopping worker threads
            # crash env.step with EGLError. ``_proprio`` is set (and the thread
            # runs) only for sim-attached HALs (those exposing ``idle_step``); a
            # real HAL keeps it ``None`` and publishes via the legacy timers.
            self._proprio: ProprioSnapshot | None = None
            self._pub_thread: threading.Thread | None = None
            self._pub_stop: threading.Event | None = None
            # /clock publisher. Created at activate iff the
            # graph runs on sim time (the node's ``use_sim_time`` is True,
            # derived from ClockAuthority.origin=simulation) AND the HAL exposes a sim
            # clock; the publisher thread emits sim_time_ns so Nav2/slam/octomap
            # advance in lockstep with the sim. The HAL is the single /clock
            # authority (deploy-sim steps the sim, so only it knows sim time).
            self._clock_pub: Any = None
            # Uniform 1 Hz /diagnostics heartbeat. Built lazily
            # in on_configure so module import stays import-safe without
            # ``openral_observability`` on the path.
            self._heartbeat: Any = None
            # CLAUDE.md §1.5 — estop latch.
            self._estopped: bool = False
            # Monotonic tick counters stamped on hal.read_state /
            # hal.send_action spans so the dashboard correlates ticks
            # within a single goal lifecycle.
            self._read_tick_idx: int = 0
            self._send_tick_idx: int = 0
            self.declare_parameter("publish_rate_hz", 30.0)
            self.get_logger().info(f"{node_name} HAL node initialised.")

        # ── Subclass hooks ────────────────────────────────────────────────

        def _create_hal(self) -> HAL:
            """Construct and return a fresh HAL instance.

            **Must be overridden by every subclass.** Reads any
            ROS-parameter-driven constructor args via
            ``self.get_parameter(...).get_parameter_value().<kind>``.
            The base class calls ``connect()`` on the returned HAL.
            """
            raise NotImplementedError(
                f"{type(self).__name__}._create_hal must be overridden to construct a HAL instance."
            )

        def _heartbeat_extra_fields(self) -> dict[str, str]:
            """Return extra key/value fields to attach to the diagnostics heartbeat.

            Defaults to empty. Subclasses can return things like
            ``{"port": "/dev/ttyUSB0"}`` (SO-100) or
            ``{"mjcf": "/abs/path/openarm.xml"}`` (OpenArm) so the
            ``/diagnostics`` payload surfaces the robot-specific
            connection state alongside the standard ``robot`` / ``estopped``
            fields.
            """
            return {}

        def _heartbeat_status(self, robot_name: str) -> tuple[int, str, dict[str, str]]:
            """Return the generic status plus optional cached HAL health.

            Real hardware adapters may implement ``HALHealthProvider``. Its
            ``health()`` method must use cached state only; the 1 Hz diagnostics
            path must not perform device or network I/O.
            """
            from openral_observability import Level

            from openral_hal.protocol import HALHealthProvider

            extras = self._heartbeat_extra_fields()
            if self._estopped:
                return (
                    Level.ERROR,
                    "estop latched",
                    {"robot": robot_name, "estopped": "true", **extras},
                )
            if self._hal is None:
                return Level.ERROR, "hal disconnected", {"robot": robot_name, **extras}
            if isinstance(self._hal, HALHealthProvider):
                try:
                    report = self._hal.health()
                except Exception as exc:  # reason: diagnostics must surface, not hide, HAL faults
                    return (
                        Level.ERROR,
                        f"HAL health check failed: {exc}",
                        {"robot": robot_name, **extras},
                    )
                return (
                    Level.OK,
                    report.message,
                    {"robot": robot_name, "estopped": "false", **extras, **report.fields},
                )
            return (
                Level.OK,
                "hal ready",
                {"robot": robot_name, "estopped": "false", **extras},
            )

        def on_configure_post_hal(self) -> TransitionCallbackReturn:
            """Subclass extension point after the base wires HAL + heartbeat.

            Used for robot-specific setup that depends on
            ``self._hal`` being connected (e.g. opening an offscreen
            camera renderer on OpenArm). Default: return ``SUCCESS``.
            """
            return TransitionCallbackReturn.SUCCESS

        def on_activate_post_subs(self) -> TransitionCallbackReturn:
            """Subclass extension point after the base wires the standard subs.

            Used for robot-specific timers / publishers (e.g. the
            OpenArm camera render timer). Default: return ``SUCCESS``.
            """
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate_pre_teardown(self) -> None:
            """Subclass extension point before the base tears down subs.

            Used to stop robot-specific timers / destroy extra
            publishers. Default: no-op.
            """

        def on_cleanup_pre_disconnect(self) -> None:
            """Subclass extension point before the base disconnects the HAL.

            Used to tear down robot-specific resources (viewers,
            renderers). Default: no-op.
            """

        # ── Lifecycle ────────────────────────────────────────────────────

        @log_lifecycle_errors
        def on_configure(self, state: object) -> TransitionCallbackReturn:
            """Construct + connect the HAL, wire the heartbeat, then run post-hook."""
            from openral_core.exceptions import ROSConfigError, ROSRuntimeError
            from openral_observability import DiagnosticsHeartbeat

            try:
                self._hal = self._create_hal()
                self._hal.connect()
            except (ROSConfigError, ROSRuntimeError) as exc:
                self.get_logger().error(f"HAL connect failed: {exc}")
                return TransitionCallbackReturn.FAILURE

            robot_name = getattr(getattr(self._hal, "description", None), "name", self._node_name)

            def _status() -> tuple[int, str, dict[str, str]]:
                return self._heartbeat_status(str(robot_name))

            self._heartbeat = DiagnosticsHeartbeat(
                self,
                hardware_id=f"{self._node_name}:{robot_name}",
                component_name=self._node_name,
                status_fn=_status,
            )
            self._heartbeat.create_publisher()
            self.get_logger().info("HAL connected.")
            return self.on_configure_post_hal()

        @log_lifecycle_errors
        def on_activate(self, state: object) -> TransitionCallbackReturn:
            """Open the standard publishers + subscribers + timer."""
            from openral_msgs.msg import (
                ActionChunk,
            )
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import JointState as RosJointState
            from std_msgs.msg import Empty, Float32MultiArray, UInt64

            control_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            # HAL publishes /joint_states on the global topic
            # so the world_state aggregator's single subscriber reads it
            # without per-node remapping. The legacy `~/joint_states`
            # publication is kept for back-compat with existing CLI
            # consumers.
            self._joint_state_pub = self.create_publisher(
                RosJointState, "/joint_states", control_qos
            )
            self._publisher = self.create_publisher(RosJointState, "~/joint_states", control_qos)
            self._policy_state_pub = self.create_publisher(
                Float32MultiArray,
                "/openral/policy_state",
                control_qos,
            )
            self._action_applied_pub = self.create_publisher(
                UInt64,
                "/openral/action_applied",
                control_qos,
            )
            # Sim-attached HALs (those exposing ``idle_step``) read
            # MjData; publish odom/joint_state off a dedicated thread (below)
            # from a plain-data snapshot, so they aren't starved by env.step.
            # A real HAL keeps ``_proprio = None`` and uses the legacy timers.
            self._proprio = (
                ProprioSnapshot() if callable(getattr(self._hal, "idle_step", None)) else None
            )
            # Sim /clock publisher. When the graph is on sim
            # time (``use_sim_time`` true via ClockAuthority.origin=simulation) and this is a
            # sim-attached HAL, the publisher thread emits the captured
            # ``sim_time_ns`` on ``/clock``. RELIABLE so it satisfies any
            # downstream clock-subscription QoS. If the backend has no sim clock
            # (sidecar without sim_time, clock-less env), captured sim_time stays
            # ``None`` and the thread simply never publishes — the graph then
            # has no /clock and use_sim_time should not have been set (the CLI
            # resolves clock_origin against backend capability).
            self._clock_pub = None
            if self._proprio is not None and (
                self.get_parameter("use_sim_time").get_parameter_value().bool_value
            ):
                # Gate on a real sim clock: use_sim_time without a
                # /clock pins every node at t=0 — the exact frozen-clock failure
                # this whole effort fixed. If the backend exposes no sim time
                # (sidecar / clock-less env), refuse to claim the /clock role and
                # warn loudly rather than silently freeze the graph.
                probe = getattr(self._hal, "sim_time_ns", None)
                sim_clock_available = probe is not None and probe() is not None
                if sim_clock_available:
                    from rosgraph_msgs.msg import Clock as _ClockMsg

                    clock_qos = QoSProfile(
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        depth=10,
                    )
                    self._clock_pub = self.create_publisher(_ClockMsg, "/clock", clock_qos)
                    self.get_logger().info("publishing /clock from sim time.")
                else:
                    self.get_logger().error(
                        "use_sim_time=true but this backend exposes no sim clock "
                        "(sim_time_ns is None) — NO /clock will be published and the "
                        "graph would freeze at t=0. Use host_wall clock_origin for this backend."
                    )
            # Consume /openral/safe_action. Depth=50
            # mirrors the candidate_action upstream — depth=1 coalesces
            # the multi-slot chunks the safety kernel forwards per
            # policy tick (CARTESIAN_DELTA + GRIPPER_POSITION arrive
            # back-to-back) so only the last one survives, freezing
            # the arm in deploy_sim. See ros_publishing_hal.chunk_qos
            # + cpp/openral_safety_kernel chunk_qos() — all three sides
            # must stay aligned at 50.
            chunk_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=50,
            )
            self._safe_action_sub = self.create_subscription(
                ActionChunk,
                "/openral/safe_action",
                self._on_safe_action,
                chunk_qos,
            )
            # CLAUDE.md §1.5 — defense-in-depth estop.
            estop_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._estop_sub = self.create_subscription(
                Empty, "/openral/estop", self._on_estop, estop_qos
            )
            # Reset-cleared broadcast (symmetric to /openral/estop). The estop
            # TRIGGER is a topic every node latches on, but reset was a
            # kernel-only service, so the HAL stayed latched after
            # /openral/estop_reset — the robot never resumed until a restart.
            # The reset authority (the kernel via the dashboard, after its
            # cooldown-gated reset succeeds) publishes here so the HAL clears too.
            self._estop_reset_sub = self.create_subscription(
                Empty, "/openral/estop_cleared", self._on_estop_cleared, estop_qos
            )
            rate_hz: float = (
                self.get_parameter("publish_rate_hz").get_parameter_value().double_value
            )
            # For sim-attached HALs, joint_state (and odom, in
            # MobileBaseBridge) is published off a dedicated thread reading the
            # snapshot, NOT a timer on the single executor thread (which is busy
            # with env.step / render / raycast). Seed the snapshot first so the
            # thread never reads an empty one; the thread is started after the
            # bridges come up (end of on_activate). A real HAL keeps the timer.
            if self._proprio is not None:
                self._capture_proprio()
            else:
                self._timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._publish_joint_state)
            if self._heartbeat is not None:
                self._heartbeat.start()
            self.get_logger().info(f"HAL activated at {rate_hz:.1f} Hz.")
            result = self.on_activate_post_subs()
            # Start the proprio publisher thread after the bridges are
            # up (it may publish odom via ``self._mobile_base``). Sim HALs only.
            if result == TransitionCallbackReturn.SUCCESS and self._proprio is not None:
                self._start_publisher_thread(max(rate_hz, 1.0))
            return result

        def on_deactivate(self, state: object) -> TransitionCallbackReturn:
            """Stop timers + tear down subs/pubs. Calls the pre-teardown hook first."""
            # Stop the proprio publisher thread before tearing down the
            # publishers it writes to.
            self._stop_publisher_thread()
            if self._clock_pub is not None:
                self.destroy_publisher(self._clock_pub)
                self._clock_pub = None
            self.on_deactivate_pre_teardown()
            if self._heartbeat is not None:
                self._heartbeat.stop()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._safe_action_sub is not None:
                self.destroy_subscription(self._safe_action_sub)
                self._safe_action_sub = None
            if self._estop_sub is not None:
                self.destroy_subscription(self._estop_sub)
                self._estop_sub = None
            if self._estop_reset_sub is not None:
                self.destroy_subscription(self._estop_reset_sub)
                self._estop_reset_sub = None
            if self._action_applied_pub is not None:
                self.destroy_publisher(self._action_applied_pub)
                self._action_applied_pub = None
            if self._publisher is not None:
                self.destroy_publisher(self._publisher)
                self._publisher = None
            if self._joint_state_pub is not None:
                self.destroy_publisher(self._joint_state_pub)
                self._joint_state_pub = None
            if self._policy_state_pub is not None:
                self.destroy_publisher(self._policy_state_pub)
                self._policy_state_pub = None
            self._policy_state_last_frame = None
            return TransitionCallbackReturn.SUCCESS

        def on_cleanup(self, state: object) -> TransitionCallbackReturn:
            """Disconnect the HAL. Calls the pre-disconnect hook first."""
            self.on_cleanup_pre_disconnect()
            if self._heartbeat is not None:
                self._heartbeat.destroy()
                self._heartbeat = None
            if self._hal is not None:
                self._hal.disconnect()
                self._hal = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: object) -> TransitionCallbackReturn:
            """Force-disconnect on shutdown — mirrors :meth:`on_cleanup`."""
            return self.on_cleanup(state)

        # ── Internal callbacks (do not override) ─────────────────────────

        def _start_publisher_thread(self, rate_hz: float) -> None:
            """Start the dedicated proprio publisher thread (sim HALs).

            Publishes joint_state + odom/TF from the plain-data snapshot at
            ``rate_hz``, off the single executor thread (busy stepping/rendering
            the sim). It touches only the snapshot + rclpy publishers (both
            thread-safe) — never MjData/GL — so MuJoCo's thread-affine context is
            never used off the executor thread.
            """
            stop = threading.Event()
            self._pub_stop = stop
            period = 1.0 / max(rate_hz, 1.0)
            from rosgraph_msgs.msg import Clock as _ClockMsg

            def _publish_clock() -> None:
                # Emit the captured sim time on /clock so the
                # rest of the graph (use_sim_time) advances with the sim. Read
                # from the snapshot (never the simulator) — same thread-safety
                # contract as the other publishers. Published first so the node's
                # own clock updates before its odom/joint_state stamps.
                if self._clock_pub is None or self._proprio is None:
                    return
                frame = self._proprio.latest()
                if frame is None or frame.sim_time_ns is None:
                    return
                t = frame.sim_time_ns
                msg = _ClockMsg()
                msg.clock.sec = int(t // 1_000_000_000)
                msg.clock.nanosec = int(t % 1_000_000_000)
                self._clock_pub.publish(msg)

            def _loop() -> None:
                next_deadline = monotonic() + period
                while not stop.is_set():
                    try:
                        _publish_clock()
                        self._publish_joint_state()
                        mobile = getattr(self, "_mobile_base", None)
                        if mobile is not None:
                            mobile.publish_from_snapshot()
                    except Exception as exc:  # reason: a publish hiccup must not kill the thread
                        self.get_logger().warn(f"proprio publisher thread: {exc}")
                    remaining = next_deadline - monotonic()
                    if remaining > 0 and stop.wait(timeout=remaining):
                        return
                    next_deadline += period
                    if monotonic() > next_deadline + period:
                        next_deadline = monotonic() + period

            self._pub_thread = threading.Thread(
                target=_loop, name="openral_hal_proprio_pub", daemon=True
            )
            self._pub_thread.start()

        def _stop_publisher_thread(self) -> None:
            """Signal + join the publisher thread (idempotent)."""
            if self._pub_stop is not None:
                self._pub_stop.set()
            if self._pub_thread is not None:
                self._pub_thread.join(timeout=2.0)
            self._pub_thread = None
            self._pub_stop = None

        def _capture_proprio(self) -> None:
            """Snapshot the HAL's proprio into ``self._proprio``.

            MUST run in the default ("sim") callback group — right after an
            ``env.step`` (from ``_send_action_traced`` / the bridge's
            ``idle_step`` hook) or at activate. It reads the simulator-backed
            HAL accessors; the control group only ever reads the resulting
            plain-data frame, so it never touches MjData off this thread.
            No-op for real HALs (``_proprio is None``).
            """
            if self._proprio is None or self._hal is None:
                return
            state = self._hal.read_state()
            bp = getattr(self._hal, "base_pose", (0.0, 0.0, 0.0))
            getter = getattr(self._hal, "base_pose_6dof", None)
            pose_6dof = getter() if getter is not None else None
            twist = getattr(self._hal, "base_twist", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            # Capture sim time here (executor thread, safe
            # MjData read) so the publisher thread can emit /clock without racing
            # env.step. ``None`` for clock-less / sidecar HALs → no /clock.
            sim_time_getter = getattr(self._hal, "sim_time_ns", None)
            sim_time_ns = sim_time_getter() if sim_time_getter is not None else None
            policy_state_getter = getattr(self._hal, "read_policy_state", None)
            policy_state_raw = policy_state_getter() if policy_state_getter is not None else None
            policy_state = (
                tuple(float(value) for value in policy_state_raw)
                if policy_state_raw is not None
                else None
            )
            self._proprio.set(
                ProprioFrame(
                    state=state,
                    base_pose=(float(bp[0]), float(bp[1]), float(bp[2])),
                    base_pose_6dof=pose_6dof,
                    base_twist=tuple(float(v) for v in twist),
                    sim_time_ns=sim_time_ns,
                    policy_state=policy_state,
                )
            )

        def _publish_joint_state(self) -> None:
            """Timer callback: publish joint state.

            From the proprio snapshot for sim-attached HALs, else a live
            ``hal.read_state``. While the e-stop latch is set, snapshot-backed
            HALs keep publishing the last post-step frame so operators retain
            joint visibility through the latch; real HALs are skipped instead —
            their transport may already be torn down (``RESTART_REQUIRED``
            stops the whole vendor stack) and polling would only spam errors.
            """
            if self._hal is None or self._publisher is None:
                return
            if self._estopped and self._proprio is None:
                return

            from openral_observability import producer as ral_producer
            from openral_observability import semconv
            from opentelemetry import trace
            from sensor_msgs.msg import JointState as RosJointState

            tick_idx = self._read_tick_idx
            self._read_tick_idx += 1
            tracer = trace.get_tracer("openral_hal.lifecycle")
            hal_adapter_label = type(self._hal).__name__.lower()
            robot_model = getattr(self._hal.description, "name", self._node_name)
            # Record the duration histogram alongside the span. These two
            # instruments used to live only in `openral_runner.DeployRunner`,
            # which the ROS deploy graph never instantiates — so a real
            # `deploy run` emitted HAL spans and no HAL latency metric at all.
            # Emitting both from the one site is what keeps them consistent.
            with (
                _hal_duration_metric(semconv.METRIC_HAL_READ_STATE_DURATION, hal_adapter_label),
                tracer.start_as_current_span(
                    semconv.SPAN_HAL_READ_STATE,
                    attributes={
                        semconv.HAL_ADAPTER: hal_adapter_label,
                        semconv.HAL_ROBOT_MODEL: str(robot_model),
                        semconv.TICK_IDX: tick_idx,
                    },
                ) as hal_read_span,
            ):
                if self._proprio is not None:
                    # Read the post-step snapshot (plain data), never
                    # the simulator: this callback runs on the control thread
                    # concurrent with env.step.
                    frame = self._proprio.latest()
                    if frame is None:
                        return
                    state = frame.state
                else:
                    try:
                        state = self._hal.read_state()
                    except Exception as exc:  # reason: HAL surfaces typed errors; log + skip
                        hal_read_span.record_exception(exc)
                        self.get_logger().warn(f"read_state failed: {exc}")
                        return
                joint_specs = list(self._hal.description.joints)
                ral_producer.record_joint_state(
                    hal_read_span,
                    names=list(state.name),
                    positions=list(state.position),
                    velocities=list(state.velocity) if state.velocity else None,
                    efforts=list(state.effort) if state.effort else None,
                    position_limits=[j.position_limits for j in joint_specs] or None,
                    velocity_limits=[j.velocity_limit for j in joint_specs] or None,
                    effort_limits=[j.effort_limit for j in joint_specs] or None,
                    stamp_ns=state.stamp_ns,
                )

            msg = RosJointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(state.name)
            msg.position = list(state.position)
            msg.velocity = list(state.velocity) if state.velocity else []
            msg.effort = list(state.effort) if state.effort else []
            self._publisher.publish(msg)
            if self._joint_state_pub is not None:
                self._joint_state_pub.publish(msg)
            if self._policy_state_pub is not None:
                frame = self._proprio.latest() if self._proprio is not None else None
                # Publish only when the frame is NEW (one publish per env.step
                # capture) — see ``_policy_state_last_frame``. Latched
                # republishing would defeat the downstream staleness gate.
                if (
                    frame is not None
                    and frame.policy_state is not None
                    and frame is not self._policy_state_last_frame
                ):
                    from std_msgs.msg import Float32MultiArray

                    policy_msg = Float32MultiArray()
                    policy_msg.data = list(frame.policy_state)
                    self._policy_state_pub.publish(policy_msg)
                    self._policy_state_last_frame = frame

        def _on_safe_action(self, msg: object) -> None:
            """``/openral/safe_action`` callback.

            Decodes the action-chunk wire shape back into the typed
            :class:`Action` via :func:`decode_action_chunk`. Hardcoding
            ``ControlMode.JOINT_POSITION`` here (the prior behaviour)
            silently misrouted per-mode chunks: a 6-D CARTESIAN_DELTA
            arrived looking like a 6-joint JOINT_POSITION row and the
            HAL's per-mode packer rejected it with a row-width error
            that only surfaced as a single WARN — symptom: the arm
            never moves in ``deploy sim`` even though policy_step
            keeps publishing big deltas.
            """
            if self._hal is None or self._estopped:
                return
            action = decode_action_chunk(msg)
            if action is None:
                return
            if not self._send_action_traced(action, source="safe_action"):
                return
            self._publish_action_applied_if_complete(action)

        def _publish_action_applied_if_complete(self, action: Any) -> None:  # noqa: ANN401  # reason: typed Action is imported only on the ROS path
            """Acknowledge a grouped tick only after its HAL application completes."""
            if self._action_applied_pub is None:
                return
            group_size = int(action.tick_group_size)
            tick = int(action.tick_index)
            if group_size <= 1 or tick <= 0 or tick <= self._last_action_applied_tick:
                return
            committed_tick = getattr(self._hal, "last_committed_tick", None)
            if committed_tick is not None:
                complete = int(committed_tick) == tick
            else:
                if self._safe_group_tick is None:
                    self._safe_group_tick = tick
                elif self._safe_group_tick != tick:
                    self._safe_group_tick = tick
                    self._safe_group_count = 0
                self._safe_group_count += 1
                complete = self._safe_group_count == group_size
            if not complete:
                return
            bridge = getattr(self, "_bridge", None)
            ack_ready = getattr(bridge, "attachment_action_ack_ready", None)
            if callable(ack_ready) and not ack_ready():
                self._deferred_action_applied_tick = tick
                self.get_logger().info(
                    f"deferring action_applied tick={tick} for attachment perception"
                )
                return
            self._publish_action_applied_tick(tick)

        def _publish_action_applied_tick(self, tick: int) -> None:
            """Publish one completed tick and reset grouped-action bookkeeping."""
            if self._action_applied_pub is None or tick <= self._last_action_applied_tick:
                return
            from std_msgs.msg import UInt64

            msg = UInt64()
            msg.data = tick
            self._action_applied_pub.publish(msg)
            self._last_action_applied_tick = tick
            self._safe_group_tick = None
            self._safe_group_count = 0
            self._deferred_action_applied_tick = 0

        def _on_attachment_perception_ready(self) -> None:
            """Release the grouped tick held while attached depth clears the map."""
            if self._deferred_action_applied_tick > 0:
                self.get_logger().info(
                    "attachment perception ready; releasing action_applied "
                    f"tick={self._deferred_action_applied_tick}"
                )
                self._publish_action_applied_tick(self._deferred_action_applied_tick)

        def _send_action_traced(self, action: Any, *, source: str) -> bool:  # noqa: ANN401  # reason: action shape is HAL-adapter-specific (numpy ndarray / dict / typed namedtuple)
            """Forward ``action`` to ``self._hal.send_action`` inside a ``hal.send_action`` span.

            Centralises OTel wiring for the
            ``/openral/safe_action`` path; the ``source`` argument is
            stamped on the span so the dashboard's Commands card can
            disambiguate the originating subscription.
            """
            from openral_observability import producer as ral_producer
            from openral_observability import semconv
            from opentelemetry import trace

            if self._hal is None:
                return False
            tick_idx = self._send_tick_idx
            self._send_tick_idx += 1
            tracer = trace.get_tracer("openral_hal.lifecycle")
            hal_adapter_label = type(self._hal).__name__.lower()
            applied = True
            with (
                _hal_duration_metric(semconv.METRIC_HAL_SEND_ACTION_DURATION, hal_adapter_label),
                tracer.start_as_current_span(
                    semconv.SPAN_HAL_SEND_ACTION,
                    attributes={
                        semconv.HAL_ADAPTER: hal_adapter_label,
                        semconv.HAL_CONTROL_MODE: action.control_mode.value,
                        semconv.TICK_IDX: tick_idx,
                        "openral.hal.action.source": source,
                    },
                ) as hal_send_span,
            ):
                try:
                    self._hal.send_action(action)
                except Exception as exc:  # reason: HAL surfaces typed errors; log + skip
                    hal_send_span.record_exception(exc)
                    applied = False
                    self.get_logger().warn(f"send_action ({source}) failed: {exc}")
                # Refresh the proprio snapshot after the step (this
                # runs in the default/"sim" group, so the MjData read is safe).
                if applied:
                    self._capture_proprio()
                # Per-mode payload field carries the first row for the
                # dashboard's Commands card. Cover every actuation mode an
                # rSkill can emit — including ``joint_velocities`` /
                # ``joint_torques`` / ``cartesian_pose``, which the robocasa
                # composite/velocity skills use; omitting them left the card's
                # "next" row blank for those skills. Empty → None so the
                # producer records "no row" instead of crashing on ``[0]``.
                next_row: list[float] | None
                if action.joint_targets:
                    next_row = list(action.joint_targets[0])
                elif action.joint_velocities:
                    next_row = list(action.joint_velocities[0])
                elif action.joint_torques:
                    next_row = list(action.joint_torques[0])
                elif action.cartesian_pose:
                    pose0 = action.cartesian_pose[0]
                    next_row = [*pose0.xyz, *pose0.quat_xyzw]
                elif action.cartesian_delta:
                    next_row = list(action.cartesian_delta[0])
                elif action.cartesian_twist:
                    next_row = list(action.cartesian_twist[0])
                elif action.body_twist:
                    next_row = list(action.body_twist[0])
                elif action.gripper:
                    next_row = [float(action.gripper[0])]
                else:
                    next_row = None
                ral_producer.record_action(
                    hal_send_span,
                    next_row=next_row,
                    dim=len(next_row) if next_row else None,
                    horizon=action.horizon,
                    applied=applied,
                )
            return applied

        def _on_estop(self, _msg: object) -> None:
            """Latch and invoke downstream stop for HALs that explicitly opt in."""
            if self._estopped:
                return
            self._estopped = True
            self.get_logger().error(
                "openral_hal.estop_received; ignoring further commands until reset."
            )
            # Ordering is deliberate: latch, then STOP, then report. The latch
            # above is what makes `_on_safe_action` drop commands, and the
            # physical stop below must not queue behind telemetry. Reporting
            # runs in a `finally` so an e-stop is counted even when the vendor
            # stop path raises, and even for HALs that opt out of it entirely
            # (the early return in `_invoke_hal_estop`) — an uncounted e-stop
            # is exactly the blind spot this exists to close.
            try:
                self._invoke_hal_estop()
            finally:
                self._emit_estop_telemetry()

        def _invoke_hal_estop(self) -> None:
            """Call the vendor stop path for HALs that opt into it."""
            from openral_hal.protocol import LifecycleEStopHAL

            if self._hal is None or not isinstance(self._hal, LifecycleEStopHAL):
                return
            from openral_core.exceptions import ROSEStopRequested

            try:
                self._hal.estop()
            except ROSEStopRequested as exc:
                self.get_logger().error(f"hardware estop completed: {exc}")
            except Exception as exc:  # reason: latch must survive a vendor stop-path failure
                self.get_logger().fatal(f"hardware estop failed: {exc}")

        def _emit_estop_telemetry(self) -> None:
            """Report the e-stop on the OTLP path: span event + counter.

            The dashboard ingests OTLP, not ROS topics, so ``/openral/estop``
            is invisible to it. Its Command-band ``e-stops`` counter reads
            ``openral.event.estop_requested``, which nothing emitted — the
            widget therefore read **0 no matter how many e-stops fired**, and a
            safety indicator that cannot leave zero reads as "no e-stops have
            occurred". Same for the ``openral.hal.estop.count`` instrument.

            This node is the right and only chokepoint: it is the shared
            lifecycle base every robot HAL runs on, and it sits on the
            actuation side — if it fires, commands are already being dropped.
            Counting at the six ``/openral/estop`` publishers would need six
            call sites (one of them the C++ kernel) and counting at every
            subscriber would multiply a single e-stop into several.

            Never raises: an exception escaping here would propagate out of the
            e-stop callback, and telemetry must not be able to disturb the
            stop path (CLAUDE.md §1.1).
            """
            try:
                from openral_observability import metrics as ral_metrics
                from openral_observability import semconv
                from opentelemetry import trace

                adapter = type(self._hal).__name__ if self._hal is not None else "none"
                attrs = {semconv.HAL_ADAPTER: adapter}
                span = trace.get_current_span()
                if span.is_recording():
                    span.add_event(semconv.EVENT_ESTOP_REQUESTED, attributes=attrs)
                else:
                    # No active span in a bare ROS callback — open a transient
                    # one so the event still reaches the collector (mirrors
                    # `reasoner_node._emit_skill_failure_event`).
                    tracer = trace.get_tracer("openral.hal")
                    with tracer.start_as_current_span("hal.estop") as estop_span:
                        estop_span.add_event(semconv.EVENT_ESTOP_REQUESTED, attributes=attrs)
                ral_metrics.get_hal_estop_count().add(1, {semconv.LABEL_HAL_ADAPTER: adapter})
            except Exception as exc:  # reason: telemetry must never disturb the stop path
                self.get_logger().warning(f"estop telemetry failed: {exc!s}")

        def _on_estop_cleared(self, _msg: object) -> None:
            """Clear the estop latch when the reset authority broadcasts /openral/estop_cleared.

            Recovery is per-HAL policy. HALs without the hardware-estop opt-in
            just clear the local latch — without that the node dropped every
            command (``_on_safe_action`` returns early on ``_estopped``) and
            never resumed until a node restart. Opted-in HALs declare
            ``estop_recovery``: ``RESETTABLE`` HALs get ``reset_estop()`` and
            then resume, while ``RESTART_REQUIRED`` HALs (e.g. Galaxea A1,
            whose estop stops the entire ROS 1 sidecar) deliberately keep the
            latch — a full lifecycle restart with fresh alignment is the only
            way back. The kernel's cooldown gate has already passed by the
            time this fires (the dashboard publishes it only after estop_reset
            returns success).
            """
            if not self._estopped:
                return

            from openral_hal.protocol import (
                EStopRecovery,
                LifecycleEStopHAL,
                ResettableLifecycleEStopHAL,
            )

            if isinstance(self._hal, LifecycleEStopHAL):
                recovery = self._hal.estop_recovery
                if recovery == EStopRecovery.RESTART_REQUIRED:
                    self.get_logger().error(
                        "openral_hal.estop_clear_rejected; this real HAL requires a full "
                        "lifecycle restart and fresh alignment after hardware estop."
                    )
                    return
                if recovery != EStopRecovery.RESETTABLE:
                    self.get_logger().fatal(
                        f"openral_hal.estop_clear_rejected; unsupported recovery policy "
                        f"{recovery!r}."
                    )
                    return
                if not isinstance(self._hal, ResettableLifecycleEStopHAL):
                    self.get_logger().fatal(
                        "openral_hal.estop_clear_rejected; resettable HAL does not implement "
                        "reset_estop()."
                    )
                    return
                try:
                    self._hal.reset_estop()
                except Exception as exc:  # reason: never clear local latch after reset failure
                    self.get_logger().fatal(f"hardware estop reset failed: {exc}")
                    return
            self._estopped = False
            self.get_logger().info("openral_hal.estop_cleared; resuming command execution.")

    class _FactoryHALLifecycleNode(HALLifecycleNodeBase):
        """Thin subclass that takes a zero-arg HAL factory.

        Used by :func:`make_lifecycle_main` for HAL adapters whose
        constructor has no ROS parameters worth exposing (UR5e / UR10e
        / Franka). The factory is stored at construction time and
        invoked by :meth:`_create_hal`.
        """

        def __init__(self, node_name: str, hal_factory: HALFactory) -> None:
            super().__init__(node_name)
            self._hal_factory = hal_factory

        def _create_hal(self) -> HAL:
            return self._hal_factory()

    class ManifestHALLifecycleNode(HALLifecycleNodeBase):
        """Manifest-driven node: builds its HAL via ``build_hal(mode=...)``.

        Used by :func:`make_lifecycle_main_from_manifest`. Reads
        ``robot_yaml`` + ``hal_mode`` params and routes through the single
        resolver seam, so one node class serves sim and real for every robot.

        The HAL's construction kwargs (serial ``port``, ``robot_ip``, …) come
        from the manifest's ``hal.parameters.defaults`` block,
        threaded by :func:`openral_hal.build_hal` — so a parameterised robot
        needs no bespoke ``_create_hal`` subclass, only a manifest entry. This
        is the generic node that the per-robot lifecycle packages collapse
        into (issue #191).
        """

        def __init__(self, node_name: str) -> None:
            """Declare the manifest-driven params (``robot_yaml`` + ``hal_mode`` + sensor knobs)."""
            super().__init__(node_name)
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("hal_mode", "sim")
            self.declare_parameter("sim_env_yaml", "")
            # Real-HW transport overrides: `openral deploy run`
            # forwards the robot manifest's hal transport overrides (serial `port` /
            # `robot_ip` / `fci_ip`) + hal.params (calibration `id`) via the
            # HAL params file. They MUST be declared here or rclpy silently
            # drops them and build_hal falls back to the manifest's defaults —
            # observed as the SO-101 node connecting to /dev/ttyUSB0 while the
            # arm sat on /dev/ttyACM0, then reading with no calibration.
            # Empty string = unset (manifest default applies).
            self.declare_parameter("port", "")
            self.declare_parameter("robot_ip", "")
            self.declare_parameter("fci_ip", "")
            self.declare_parameter("id", "")
            # Calibration directory override. `deploy run` forwards
            # the deploy's `calibration_dir` HAL override so a deploy can
            # load a calibration committed next to its config instead of the
            # ambient HF cache (which may hold several stale `<id>.json` for one
            # arm). Empty string = unset (lerobot's default HF cache dir).
            self.declare_parameter("calibration_dir", "")
            # Scene-level MJCF composition (a `SceneComposition` as
            # JSON). `openral deploy sim` forwards the DeployScene's `composition`
            # here so the SCENE (not the robot manifest) owns its arena. Takes
            # precedence over `scene_defaults.composition`; "" = none.
            self.declare_parameter("scene_composition_json", "")
            self.declare_parameter("viewer_enabled", True)
            self.declare_parameter("walking_enabled", False)
            self.declare_parameter("camera_publish_rate_hz", 10.0)
            self.declare_parameter("viewer_sync_rate_hz", 30.0)
            # scan_* envelope params deploy_sim injects for lidar robots; declare
            # so rclpy accepts the override (Phase 2 consumes them). Defaults are
            # placeholders overridden from the manifest's lidar_2d sensor.
            self.declare_parameter("scan_publish_rate_hz", 10.0)
            self.declare_parameter("scan_n_beams", 360)
            self.declare_parameter("scan_max_range_m", 12.0)
            self.declare_parameter("scan_min_range_m", 0.05)
            # depth_* params for PointCloud2 streams (Phase 2).
            # Gated by bridge on live MuJoCo handles + manifest depth sensor.
            self.declare_parameter("depth_publish_rate_hz", 10.0)
            self.declare_parameter("depth_max_range_m", 5.0)
            self.declare_parameter("depth_pixel_stride", 4)
            # Mobile-base params (issue #191 Phase 3): consumed by MobileBaseBridge
            # only when the manifest declares `base_joints` (panda_mobile today).
            # Harmless for fixed-base arms — declared so deploy_sim's
            # `odom_publish_rate_hz` default + a `cmd_vel_topic` override are
            # accepted by rclpy.
            self.declare_parameter("odom_publish_rate_hz", 20.0)
            self.declare_parameter("cmd_vel_topic", "/cmd_vel")
            self._bridge: Any = None
            self._mobile_base: Any = None
            # Reflective ``ResetToPose`` service (issue #191 Phase 2):
            # opened in on_configure_post_hal only when the built
            # HAL exposes ``reset_to_pose`` (every MujocoArmHAL sim arm does;
            # PandaMobileHAL / SimAttachedHAL do not), so a robot needs no
            # bespoke service wiring.
            self._reset_to_pose_srv: Any = None

        def _create_hal(self) -> HAL:
            from openral_core import RobotDescription
            from openral_core.exceptions import ROSConfigError

            from openral_hal import build_hal

            robot_yaml = self.get_parameter("robot_yaml").get_parameter_value().string_value
            if not robot_yaml:
                raise ROSConfigError(
                    f"{self._node_name}: the 'robot_yaml' parameter is required "
                    "(path to robots/<id>/robot.yaml); `openral deploy sim`/`deploy run` inject it."
                )
            hal_mode = self.get_parameter("hal_mode").get_parameter_value().string_value or "sim"
            sim_env_yaml = (
                self.get_parameter("sim_env_yaml").get_parameter_value().string_value or None
            )
            description = RobotDescription.from_yaml(robot_yaml)
            self.get_logger().info(
                f"{hal_mode} mode: building HAL for robot={description.name} from {robot_yaml}"
                + (f" scene={sim_env_yaml}" if sim_env_yaml else "")
            )
            # Declarative MJCF scene composition (issue #191 Phase 3b).
            # When building a bare sim HAL (no scene-attach), call the named
            # composer and thread the resulting MJCF in as the HAL's `mjcf_path`.
            # The SCENE's `composition` (forwarded as `scene_composition_json`)
            # wins over the robot manifest's `scene_defaults.composition` — the
            # scene owns its arena, the robot manifest describes the robot.
            # The manifest fallback is retained for back-compat.
            from openral_core.schemas import SceneComposition

            transport: dict[str, object] = {}
            scene_comp_json = (
                self.get_parameter("scene_composition_json").get_parameter_value().string_value
            )
            composition: SceneComposition | None = None
            if scene_comp_json:
                composition = SceneComposition.model_validate_json(scene_comp_json)
            elif description.scene_defaults is not None:
                composition = description.scene_defaults.composition
            if hal_mode == "sim" and sim_env_yaml is None and composition is not None:
                transport["mjcf_path"] = self._compose_scene_mjcf(description, composition)
            # Real-HW transport overrides (`deploy run` → HAL params file →
            # the params declared in __init__). Only non-empty values are
            # threaded so build_hal's manifest-defaults fallback still applies
            # per-key. mode is validated by build_hal (sim|real).
            for _transport_key in ("port", "robot_ip", "fci_ip", "id", "calibration_dir"):
                _value = self.get_parameter(_transport_key).get_parameter_value().string_value
                if _value:
                    transport[_transport_key] = _value
            if self.get_parameter("walking_enabled").get_parameter_value().bool_value:
                transport["walking_enabled"] = True
            return build_hal(
                description,
                mode=hal_mode,  # type: ignore[arg-type]  # reason: hal_mode is a ROS param string validated as sim|real by build_hal
                transport=transport,
                sim_env_yaml=sim_env_yaml,
            )

        def _compose_scene_mjcf(self, description: RobotDescription, composition: object) -> str:
            """Run a manifest `scene_composition` composer, write the MJCF, return its path.

            The composer (a ``"module:fn"`` string) returns ``(xml, meshdir)``; the
            XML is written next to ``meshdir`` so relative mesh paths resolve, and
            the path is threaded into the HAL as ``mjcf_path``. Issue #191 Phase 3b
            replaces openarm's bespoke ``_create_hal`` scene splicing.
            """
            import importlib

            module_path, _, fn_name = composition.composer.partition(":")  # type: ignore[attr-defined]
            composer = getattr(importlib.import_module(module_path), fn_name)
            xml, meshdir = composer(**composition.params)  # type: ignore[attr-defined]
            scene_path = meshdir.parent / f"{description.name}_composed_scene.xml"
            scene_path.write_text(xml)
            self.get_logger().info(
                f"composed scene MJCF for {description.name} at {scene_path} "
                f"via {composition.composer}"  # type: ignore[attr-defined]
            )
            return str(scene_path)

        def _heartbeat_extra_fields(self) -> dict[str, str]:
            hal_mode = self.get_parameter("hal_mode").get_parameter_value().string_value or "sim"
            robot_yaml = self.get_parameter("robot_yaml").get_parameter_value().string_value
            return {"mode": hal_mode, "robot_yaml": robot_yaml}

        def on_configure_post_hal(self) -> TransitionCallbackReturn:
            """Open ``/openral/<robot>/reset_to_pose`` iff the HAL supports it.

            Generalises the openarm-only service to every manifest-driven HAL:
            reflect on the just-built HAL and wire the service only when it
            exposes ``reset_to_pose`` (the sim-arm starting-pose snap the
            ``skill_runner`` calls before the first inference tick). Robots whose
            HAL has no such method (panda_mobile, scene-attached twins) get no
            service — the call site falls back to its no-op exactly as today.
            """
            assert self._hal is not None
            if not callable(getattr(self._hal, "reset_to_pose", None)):
                return TransitionCallbackReturn.SUCCESS
            from pathlib import Path

            from openral_msgs.srv import (
                ResetToPose,
            )

            # Topic uses the robot_id (manifest directory name) to match what
            # `openral deploy sim` wires (`/openral/<robot_id>/reset_to_pose`),
            # which can differ from `description.name` (openarm dir "openarm" vs
            # name "openarm_v2"). Fall back to the HAL's name when robot_yaml is
            # absent (a directly-injected HAL in unit tests).
            robot_yaml = self.get_parameter("robot_yaml").get_parameter_value().string_value
            robot = (
                Path(robot_yaml).parent.name
                if robot_yaml
                else getattr(self._hal.description, "name", self._node_name)
            )
            topic = f"/openral/{robot}/reset_to_pose"
            self._reset_to_pose_srv = self.create_service(
                ResetToPose, topic, self._handle_reset_to_pose
            )
            self.get_logger().info(f"ResetToPose service ready at {topic}")
            return TransitionCallbackReturn.SUCCESS

        def _handle_reset_to_pose(self, request: object, response: object) -> object:
            """Forward ``request.pose`` to ``self._hal.reset_to_pose``.

            A failure surfaces as ``success=False`` + a typed ``failure_reason``
            rather than an exception across the IPC boundary (mirrors the
            original per-robot openarm handler the reflection replaces).
            """
            from openral_core.exceptions import ROSConfigError, ROSError

            pose = [float(v) for v in request.pose]  # type: ignore[attr-defined]  # reason: rosidl srv request is untyped
            self.get_logger().info(f"ResetToPose service: {len(pose)}-D pose received.")
            if self._hal is None:
                response.success = False  # type: ignore[attr-defined]  # reason: rosidl srv response is untyped
                response.failure_reason = "HAL not connected"  # type: ignore[attr-defined]
                return response
            try:
                self._hal.reset_to_pose(pose)  # type: ignore[attr-defined]  # reason: presence guaranteed by on_configure_post_hal reflection
            except ROSConfigError as exc:
                self.get_logger().error(f"ResetToPose: {exc!s}")
                response.success = False  # type: ignore[attr-defined]
                response.failure_reason = f"ROSConfigError: {exc!s}"  # type: ignore[attr-defined]
                return response
            except ROSError as exc:
                self.get_logger().error(f"ResetToPose runtime: {exc!s}")
                response.success = False  # type: ignore[attr-defined]
                response.failure_reason = f"{type(exc).__name__}: {exc!s}"  # type: ignore[attr-defined]
                return response
            # reset_to_pose mutates MjData directly but is neither an
            # idle_step nor a send_action, so the proprio snapshot the
            # /joint_states publisher serves would otherwise stay at the
            # PRE-reset pose. The policy's first inference fires ~20 ms after the
            # reset (inside the idle-hold window, before any post-reset
            # idle_step) and would read that stale state → out-of-distribution
            # first action → self-collision wedge. Refresh the snapshot now; this
            # handler runs on the MjData-owning callback group (same thread
            # _capture_proprio requires), so it is safe + synchronous before the
            # runner unblocks on the service response and ticks step 1.
            if self._proprio is not None:
                self._capture_proprio()
                # Push the fresh pose onto /joint_states immediately (don't wait
                # for the next publisher-thread tick) so the runner's post-reset
                # freshness gate passes ASAP. Safe: reads the just-set snapshot +
                # thread-safe publish; touches no MjData.
                self._publish_joint_state()
            response.success = True  # type: ignore[attr-defined]
            response.failure_reason = ""  # type: ignore[attr-defined]
            return response

        def on_activate_post_subs(self) -> TransitionCallbackReturn:
            """Attach the :class:`SimSensorBridge` (cameras / depth / scan / viewer)."""
            from openral_hal.sim_sensor_bridge import SimSensorBridge

            assert self._hal is not None
            self._bridge = SimSensorBridge(
                self,
                self._hal,
                self._hal.description,
                viewer_enabled=self.get_parameter("viewer_enabled")
                .get_parameter_value()
                .bool_value,
                camera_rate_hz=self.get_parameter("camera_publish_rate_hz")
                .get_parameter_value()
                .double_value,
                viewer_sync_rate_hz=self.get_parameter("viewer_sync_rate_hz")
                .get_parameter_value()
                .double_value,
                scan_rate_hz=self.get_parameter("scan_publish_rate_hz")
                .get_parameter_value()
                .double_value,
                scan_n_beams=self.get_parameter("scan_n_beams").get_parameter_value().integer_value,
                scan_max_range_m=self.get_parameter("scan_max_range_m")
                .get_parameter_value()
                .double_value,
                scan_min_range_m=self.get_parameter("scan_min_range_m")
                .get_parameter_value()
                .double_value,
                depth_rate_hz=self.get_parameter("depth_publish_rate_hz")
                .get_parameter_value()
                .double_value,
                depth_max_range_m=self.get_parameter("depth_max_range_m")
                .get_parameter_value()
                .double_value,
                depth_pixel_stride=self.get_parameter("depth_pixel_stride")
                .get_parameter_value()
                .integer_value,
                # Refresh the proprio snapshot after each idle step,
                # so odom/joint_state stay fresh while the scene idles.
                on_step=self._capture_proprio,
                on_attachment_perception_ready=self._on_attachment_perception_ready,
            )
            self._bridge.setup()

            # Mobile-base streams (issue #191 Phase 3): /odom + odom->base_link TF
            # + /cmd_vel->BODY_TWIST, attached generically when the manifest
            # declares a planar base (`base_joints`). Fixed-base arms skip it.
            if describes_mobile_base(self._hal.description):
                from openral_hal.mobile_base_bridge import MobileBaseBridge

                self._mobile_base = MobileBaseBridge(
                    self,
                    self._hal,
                    self._hal.description,
                    odom_rate_hz=self.get_parameter("odom_publish_rate_hz")
                    .get_parameter_value()
                    .double_value,
                    cmd_vel_topic=self.get_parameter("cmd_vel_topic")
                    .get_parameter_value()
                    .string_value,
                    # Odom published from the node's dedicated thread
                    # reading this snapshot, so it isn't starved by env.step.
                    proprio=self._proprio,
                )
                self._mobile_base.setup()
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate_pre_teardown(self) -> None:
            """Tear down the sensor + mobile-base bridges' publishers / timers / viewer."""
            if self._bridge is not None:
                self._bridge.teardown()
                self._bridge = None
            if self._mobile_base is not None:
                self._mobile_base.teardown()
                self._mobile_base = None

        def on_cleanup_pre_disconnect(self) -> None:
            """Idempotent teardown of the sensor + mobile-base bridges + ResetToPose."""
            # Idempotent safety net: a direct active->shutdown->cleanup path
            # (no deactivate) must still tear down the bridges' pubs/timers/viewer.
            if self._bridge is not None:
                self._bridge.teardown()
                self._bridge = None
            if self._mobile_base is not None:
                self._mobile_base.teardown()
                self._mobile_base = None
            if self._reset_to_pose_srv is not None:
                self.destroy_service(self._reset_to_pose_srv)
                self._reset_to_pose_srv = None

    # Back-compat alias: existing call sites + tests reference the old
    # internal `_HALLifecycleNode` name. Keep it pointing at the factory
    # subclass so legacy callers (and the UR5e lifecycle test) continue
    # to work without churn.
    _HALLifecycleNode = _FactoryHALLifecycleNode
    # Back-compat alias for the manifest node's prior private name (issue
    # #191 promoted it to public API). Existing imports keep working.
    _ManifestHALLifecycleNode = ManifestHALLifecycleNode
