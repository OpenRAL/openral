"""Generic ROS 2 managed lifecycle node wrapper for any HAL adapter.

Wraps any :class:`openral_hal.protocol.HAL` Protocol implementation as a
``rclpy.lifecycle.LifecycleNode`` so every per-robot package (``UR5e``,
``UR10e``, ``FrankaPanda``, ``SO100Follower``, ``OpenArm``, …) shares
the same publisher / subscriber / heartbeat / OTel-span wiring.

There are three ways to use this module, in decreasing order of preference:

1. **Manifest-driven** (preferred — ADR-0032 / issue #191): call
   :func:`make_lifecycle_main_from_manifest`, which spins up the generic
   :class:`ManifestHALLifecycleNode`. It reads ``robot_yaml`` + ``hal_mode``
   ROS parameters and builds its HAL through :func:`openral_hal.build_hal`,
   so a robot's construction kwargs (serial ``port``, ``robot_ip``, …) live
   in the manifest's ``hal.parameters.defaults`` block (ADR-0029) rather than
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
  ``/openral/estop``) per ADR-0018 F1/F5.
* The 1 Hz ``DiagnosticsHeartbeat`` (ADR-0018 F8).
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
    # `hal.parameters` block (ADR-0029), threaded by build_hal.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
    """Reverse the ADR-0028b wire encoding back into a typed ``Action``.

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
    if mode in (ControlMode.JOINT_POSITION, ControlMode.JOINT_TRAJECTORY):
        kwargs["joint_targets"] = rows
    elif mode is ControlMode.JOINT_VELOCITY:
        kwargs["joint_velocities"] = rows
    elif mode is ControlMode.JOINT_TORQUE:
        kwargs["joint_torques"] = rows
    elif mode is ControlMode.CARTESIAN_DELTA:
        kwargs["cartesian_delta"] = [tuple(r) for r in rows]
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
        # ADR-0028d — sim-only mux flag. Same wire layout as gripper
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
    """Build a ``main()`` for a manifest-driven HAL lifecycle node (ADR-0032).

    Unlike :func:`make_lifecycle_main` (which pins a single hardcoded HAL
    class), the returned node reads two ROS parameters and constructs its HAL
    through the one resolver seam :func:`openral_hal.build_hal`:

    * ``robot_yaml`` (str, required) — path to ``robots/<id>/robot.yaml``.
    * ``hal_mode`` (str, default ``"sim"``) — ``"sim"`` (``deploy sim`` / the
      ``sim run`` harness) or ``"real"`` (``deploy run``, real hardware).

    So a single node serves both modes for every robot, and "add a robot"
    needs no per-package HAL class wiring — just a manifest declaring
    ``hal.sim`` / ``hal.real`` (ADR-0031). A robot whose manifest lacks the
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
        # ADR-0049 — deliberately single-threaded. MuJoCo's EGL/GL context is
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
            self._safe_action_sub: Any = None
            self._estop_sub: Any = None
            # ADR-0049 — decouple the cheap, latency-sensitive publishers (odom /
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
            # ADR-0048 Phase 2 — /clock publisher. Created at activate iff the
            # graph runs on sim time (the node's ``use_sim_time`` is True, set by
            # the ``enable_sim_clock`` launch flag) AND the HAL exposes a sim
            # clock; the publisher thread emits sim_time_ns so Nav2/slam/octomap
            # advance in lockstep with the sim. The HAL is the single /clock
            # authority (deploy-sim steps the sim, so only it knows sim time).
            self._clock_pub: Any = None
            # ADR-0018 F8 — uniform 1 Hz /diagnostics heartbeat. Built lazily
            # in on_configure so module import stays import-safe without
            # ``openral_observability`` on the path.
            self._heartbeat: Any = None
            # ADR-0018 §F5 / CLAUDE.md §1.5 — estop latch.
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
            from openral_observability import DiagnosticsHeartbeat, Level

            try:
                self._hal = self._create_hal()
                self._hal.connect()
            except (ROSConfigError, ROSRuntimeError) as exc:
                self.get_logger().error(f"HAL connect failed: {exc}")
                return TransitionCallbackReturn.FAILURE

            robot_name = getattr(getattr(self._hal, "description", None), "name", self._node_name)

            def _status() -> tuple[int, str, dict[str, str]]:
                extras = self._heartbeat_extra_fields()
                if self._estopped:
                    return Level.ERROR, "estop latched", {"robot": str(robot_name), **extras}
                if self._hal is None:
                    return Level.ERROR, "hal disconnected", {"robot": str(robot_name), **extras}
                return (
                    Level.OK,
                    "hal ready",
                    {"robot": str(robot_name), "estopped": "false", **extras},
                )

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
            from std_msgs.msg import Empty

            control_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            # ADR-0018 F1: HAL publishes /joint_states on the global topic
            # so the world_state aggregator's single subscriber reads it
            # without per-node remapping. The legacy `~/joint_states`
            # publication is kept for back-compat with existing CLI
            # consumers.
            self._joint_state_pub = self.create_publisher(
                RosJointState, "/joint_states", control_qos
            )
            self._publisher = self.create_publisher(RosJointState, "~/joint_states", control_qos)
            # ADR-0049 — sim-attached HALs (those exposing ``idle_step``) read
            # MjData; publish odom/joint_state off a dedicated thread (below)
            # from a plain-data snapshot, so they aren't starved by env.step.
            # A real HAL keeps ``_proprio = None`` and uses the legacy timers.
            self._proprio = (
                ProprioSnapshot() if callable(getattr(self._hal, "idle_step", None)) else None
            )
            # ADR-0048 Phase 2 — sim /clock publisher. When the graph is on sim
            # time (``use_sim_time`` true via ``enable_sim_clock``) and this is a
            # sim-attached HAL, the publisher thread emits the captured
            # ``sim_time_ns`` on ``/clock``. RELIABLE so it satisfies any
            # downstream clock-subscription QoS. If the backend has no sim clock
            # (sidecar without sim_time, clock-less env), captured sim_time stays
            # ``None`` and the thread simply never publishes — the graph then
            # has no /clock and use_sim_time should not have been set (the CLI
            # gates ``enable_sim_clock`` on backend capability).
            self._clock_pub = None
            if self._proprio is not None and (
                self.get_parameter("use_sim_time").get_parameter_value().bool_value
            ):
                # Gate on a real sim clock (ADR-0048 §4): use_sim_time without a
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
                    self.get_logger().info("publishing /clock from sim time (ADR-0048 Phase 2).")
                else:
                    self.get_logger().error(
                        "use_sim_time=true but this backend exposes no sim clock "
                        "(sim_time_ns is None) — NO /clock will be published and the "
                        "graph would freeze at t=0. Re-run without --enable-sim-clock."
                    )
            # ADR-0018 F1/F5: consume /openral/safe_action. Depth=10
            # mirrors the candidate_action upstream — depth=1 coalesces
            # the multi-slot chunks the safety kernel forwards per
            # policy tick (CARTESIAN_DELTA + GRIPPER_POSITION arrive
            # back-to-back) so only the last one survives, freezing
            # the arm in deploy_sim. See ros_publishing_hal.chunk_qos
            # + cpp/openral_safety_kernel chunk_qos() — all three sides
            # must stay aligned at >=10.
            chunk_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._safe_action_sub = self.create_subscription(
                ActionChunk,
                "/openral/safe_action",
                self._on_safe_action,
                chunk_qos,
            )
            # ADR-0018 §F5 / CLAUDE.md §1.5 — defense-in-depth estop.
            estop_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
            )
            self._estop_sub = self.create_subscription(
                Empty, "/openral/estop", self._on_estop, estop_qos
            )

            rate_hz: float = (
                self.get_parameter("publish_rate_hz").get_parameter_value().double_value
            )
            # ADR-0049 — for sim-attached HALs, joint_state (and odom, in
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
            # ADR-0049 — start the proprio publisher thread after the bridges are
            # up (it may publish odom via ``self._mobile_base``). Sim HALs only.
            if result == TransitionCallbackReturn.SUCCESS and self._proprio is not None:
                self._start_publisher_thread(max(rate_hz, 1.0))
            return result

        def on_deactivate(self, state: object) -> TransitionCallbackReturn:
            """Stop timers + tear down subs/pubs. Calls the pre-teardown hook first."""
            # ADR-0049 — stop the proprio publisher thread before tearing down the
            # publishers it writes to.
            self._stop_publisher_thread()
            if self._clock_pub is not None:  # ADR-0048 Phase 2
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
            if self._publisher is not None:
                self.destroy_publisher(self._publisher)
                self._publisher = None
            if self._joint_state_pub is not None:
                self.destroy_publisher(self._joint_state_pub)
                self._joint_state_pub = None
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
            """Start the ADR-0049 dedicated proprio publisher thread (sim HALs).

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
                # ADR-0048 Phase 2 — emit the captured sim time on /clock so the
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
                while not stop.is_set():
                    try:
                        _publish_clock()
                        self._publish_joint_state()
                        mobile = getattr(self, "_mobile_base", None)
                        if mobile is not None:
                            mobile.publish_from_snapshot()
                    except Exception as exc:  # reason: a publish hiccup must not kill the thread
                        self.get_logger().warn(f"proprio publisher thread: {exc}")
                    stop.wait(period)

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
            """Snapshot the HAL's proprio into ``self._proprio`` (ADR-0049).

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
            # ADR-0048 Phase 2 — capture sim time here (executor thread, safe
            # MjData read) so the publisher thread can emit /clock without racing
            # env.step. ``None`` for clock-less / sidecar HALs → no /clock.
            sim_time_getter = getattr(self._hal, "sim_time_ns", None)
            sim_time_ns = sim_time_getter() if sim_time_getter is not None else None
            self._proprio.set(
                ProprioFrame(
                    state=state,
                    base_pose=(float(bp[0]), float(bp[1]), float(bp[2])),
                    base_pose_6dof=pose_6dof,
                    base_twist=tuple(float(v) for v in twist),
                    sim_time_ns=sim_time_ns,
                )
            )

        def _publish_joint_state(self) -> None:
            """Timer callback: publish joint state.

            From the ADR-0049 snapshot for sim-attached HALs, else a live
            ``hal.read_state``.
            """
            from openral_observability import producer as ral_producer
            from openral_observability import semconv
            from opentelemetry import trace
            from sensor_msgs.msg import JointState as RosJointState

            if self._hal is None or self._publisher is None:
                return
            tick_idx = self._read_tick_idx
            self._read_tick_idx += 1
            tracer = trace.get_tracer("openral_hal.lifecycle")
            hal_adapter_label = type(self._hal).__name__.lower()
            robot_model = getattr(self._hal.description, "name", self._node_name)
            with tracer.start_as_current_span(
                semconv.SPAN_HAL_READ_STATE,
                attributes={
                    semconv.HAL_ADAPTER: hal_adapter_label,
                    semconv.HAL_ROBOT_MODEL: str(robot_model),
                    semconv.TICK_IDX: tick_idx,
                },
            ) as hal_read_span:
                if self._proprio is not None:
                    # ADR-0049 — read the post-step snapshot (plain data), never
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

        def _on_safe_action(self, msg: object) -> None:
            """``/openral/safe_action`` callback (ADR-0018 F1/F5).

            Decodes the ADR-0028b wire shape back into the typed
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
            self._send_action_traced(action, source="safe_action")

        def _send_action_traced(self, action: Any, *, source: str) -> None:  # noqa: ANN401  # reason: action shape is HAL-adapter-specific (numpy ndarray / dict / typed namedtuple)
            """Forward ``action`` to ``self._hal.send_action`` inside a ``hal.send_action`` span.

            Centralises OTel wiring for the ADR-0018 F1/F5
            ``/openral/safe_action`` path; the ``source`` argument is
            stamped on the span so the dashboard's Commands card can
            disambiguate the originating subscription.
            """
            from openral_observability import producer as ral_producer
            from openral_observability import semconv
            from opentelemetry import trace

            if self._hal is None:
                return
            tick_idx = self._send_tick_idx
            self._send_tick_idx += 1
            tracer = trace.get_tracer("openral_hal.lifecycle")
            hal_adapter_label = type(self._hal).__name__.lower()
            applied = True
            with tracer.start_as_current_span(
                semconv.SPAN_HAL_SEND_ACTION,
                attributes={
                    semconv.HAL_ADAPTER: hal_adapter_label,
                    semconv.HAL_CONTROL_MODE: action.control_mode.value,
                    semconv.TICK_IDX: tick_idx,
                    "openral.hal.action.source": source,
                },
            ) as hal_send_span:
                try:
                    self._hal.send_action(action)
                except Exception as exc:  # reason: HAL surfaces typed errors; log + skip
                    hal_send_span.record_exception(exc)
                    applied = False
                    self.get_logger().warn(f"send_action ({source}) failed: {exc}")
                # ADR-0049 — refresh the proprio snapshot after the step (this
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

        def _on_estop(self, _msg: object) -> None:
            """ADR-0018 §F5 / CLAUDE.md §1.5 — latch the estop flag."""
            if self._estopped:
                return
            self._estopped = True
            self.get_logger().error(
                "openral_hal.estop_received; ignoring further commands until reset."
            )

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

        Used by :func:`make_lifecycle_main_from_manifest` (ADR-0032). Reads
        ``robot_yaml`` + ``hal_mode`` params and routes through the single
        resolver seam, so one node class serves sim and real for every robot.

        The HAL's construction kwargs (serial ``port``, ``robot_ip``, …) come
        from the manifest's ``hal.parameters.defaults`` block (ADR-0029),
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
            self.declare_parameter("viewer_enabled", True)
            self.declare_parameter("camera_publish_rate_hz", 10.0)
            self.declare_parameter("viewer_sync_rate_hz", 30.0)
            # scan_* envelope params deploy_sim injects for lidar robots; declare
            # so rclpy accepts the override (Phase 2 consumes them). Defaults are
            # placeholders overridden from the manifest's lidar_2d sensor.
            self.declare_parameter("scan_publish_rate_hz", 10.0)
            self.declare_parameter("scan_n_beams", 360)
            self.declare_parameter("scan_max_range_m", 12.0)
            self.declare_parameter("scan_min_range_m", 0.05)
            # depth_* params for ADR-0030 PointCloud2 streams (Phase 2).
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
            # Reflective ``ResetToPose`` service (issue #191 Phase 2 / ADR-0029
            # blocker #4): opened in on_configure_post_hal only when the built
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
            # issue #191 Phase 3b — declarative MJCF scene composition. When the
            # manifest declares `scene_defaults.composition` and we're building a
            # bare sim HAL (no scene-attach), call the named composer and thread
            # the resulting MJCF path in as the HAL's `mjcf_path` transport kwarg.
            transport: dict[str, object] = {}
            composition = (
                description.scene_defaults.composition
                if description.scene_defaults is not None
                else None
            )
            if hal_mode == "sim" and sim_env_yaml is None and composition is not None:
                transport["mjcf_path"] = self._compose_scene_mjcf(description, composition)
            # Phase 1 (ADR-0032) routes sim through the seam; real-HW transport
            # params (port / robot_ip / fci_ip) arrive with the `deploy run`
            # launch path. mode is validated by build_hal (sim|real).
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
            ADR-0024 openarm handler the reflection replaces).
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
            # ADR-0049 — reset_to_pose mutates MjData directly but is neither an
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
                # ADR-0049 — refresh the proprio snapshot after each idle step,
                # so odom/joint_state stay fresh while the scene idles.
                on_step=self._capture_proprio,
            )
            self._bridge.setup()

            # Mobile-base streams (issue #191 Phase 3): /odom + odom->base_link TF
            # + /cmd_vel->BODY_TWIST, attached generically when the manifest
            # declares a planar base (`base_joints`). Fixed-base arms skip it.
            if self._hal.description.base_joints:
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
                    # ADR-0049 — odom published from the node's dedicated thread
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
