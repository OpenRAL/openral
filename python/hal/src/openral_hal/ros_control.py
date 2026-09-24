"""RosControlHAL — ros2_control-backed Hardware Abstraction Layer adapter.

This adapter forwards ``Action`` chunks to a ``ros2_control`` joint trajectory
controller and reads ``JointState`` from the controller's state topic.  In
unit tests the underlying transport is replaced by a mock; the adapter itself
contains no ROS 2 imports so it can be tested without a live ROS 2 installation.

Example:
    >>> from openral_hal.ros_control import RosControlHAL
    >>> from openral_core import (
    ...     RobotDescription,
    ...     EmbodimentKind,
    ...     JointSpec,
    ...     JointType,
    ...     RobotCapabilities,
    ...     SafetyEnvelope,
    ...     ActionSpec,
    ...     ControlMode,
    ... )
    >>> desc = RobotDescription(
    ...     name="test_robot",
    ...     embodiment_kind=EmbodimentKind.MANIPULATOR,
    ...     joints=[
    ...         JointSpec(
    ...             name="j1", joint_type=JointType.REVOLUTE, parent_link="base", child_link="link1"
    ...         ),
    ...     ],
    ...     capabilities=RobotCapabilities(
    ...         supported_control_modes=[ControlMode.JOINT_POSITION],
    ...     ),
    ...     safety=SafetyEnvelope(joint_state_staleness_limit_s=0.5),  # required: no default
    ...     action_spec=ActionSpec(dim=1, control_freq_hz=30.0),  # required: sets the deadline
    ... )
    >>> hal = RosControlHAL(desc, controller_name="joint_trajectory_controller")
    >>> hal.connect()
    >>> hal.disconnect()
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import Action, JointState, RobotDescription

from openral_hal._base import HALBase, _raw_floats, resolve_staleness_limit_s
from openral_hal.protocol import EStopRecovery

__all__ = [
    "ControllerKind",
    "ControllerStopSeam",
    "ControllerStoppable",
    "ControllerSwitchReport",
    "DownstreamStopReport",
    "DownstreamStopReporting",
    "RosControlHAL",
    "TriggerReport",
]


class ControllerKind(StrEnum):
    """Which ros2_control controller sits behind one command topic.

    A command topic is not self-describing: the topic name says nothing about
    the message type the controller expects, and publishing the wrong type is
    *silent* — DDS simply never delivers it, so the arm does not move and
    nothing anywhere logs an error. That is the same class of failure as the
    no-op transport this layer already had to fix once, so the wire format is
    declared rather than assumed.

    A HAL names the kind per topic in `command_bindings`; `RosControlTransport`
    builds the matching publisher and message from it. Adding a robot whose
    controllers differ therefore means declaring a kind, not writing transport
    code.

    Only kinds this transport can actually build a message for belong here —
    an unlisted controller must fail loudly at wire-up rather than be
    approximated by a neighbouring one.
    """

    #: `joint_trajectory_controller/JointTrajectoryController`, commanded via
    #: `trajectory_msgs/JointTrajectory` on `~/joint_trajectory`. Every
    #: ros2_control robot in this repo uses this, grippers included (OpenArm's
    #: are 1-DoF instances of it).
    JOINT_TRAJECTORY = "joint_trajectory"

    #: The `forward_command_controller` family — `ForwardCommandController`,
    #: `position_controllers/JointGroupPositionController` and siblings —
    #: commanded via `std_msgs/Float64MultiArray` on `~/commands`. No robot in
    #: this repo uses one yet; it is here because it is the other type a
    #: ros2_control arm is commonly configured with, and because two kinds are
    #: what make the dispatch real rather than a rename of an assumption.
    FORWARD_COMMAND = "forward_command"


@dataclass(frozen=True)
class ControllerSwitchReport:
    """Acknowledged outcome of one ``controller_manager`` switch.

    ``ok`` is the switch service's own verdict **and** the post-switch
    confirmation: a transport reports ``ok=True`` only when every controller
    in ``controllers`` was listed in the requested state afterwards. ``states``
    carries what ``list_controllers`` (or the in-memory simulator) reported per
    controller so a refusal names the controller that did not move.
    """

    ok: bool
    controllers: tuple[str, ...]
    states: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    detail: str = ""


@dataclass(frozen=True)
class TriggerReport:
    """Acknowledged outcome of one ``std_srvs/Trigger``-shaped vendor call."""

    success: bool
    message: str = ""


@dataclass(frozen=True)
class DownstreamStopReport:
    """What a HAL knows about its last downstream stop.

    Produced by ``RosControlHAL.estop`` (and the Interbotix-backed
    ``AlohaHAL.estop``) and read by the lifecycle node, which logs FATAL and
    publishes ``downstream_stop=unacknowledged`` on ``/diagnostics`` when
    ``stopped`` is ``False``. ``stopped`` is only ``True`` when every step —
    controller deactivation **and** the vendor stop, where one exists — was
    acknowledged by the thing that executed it; a local latch alone never
    counts.
    """

    stopped: bool
    controllers: tuple[str, ...]
    controller_states: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    vendor_stop: str = ""
    detail: str = ""

    def fields(self) -> dict[str, str]:
        """Flatten for the ``/diagnostics`` heartbeat (string values only)."""
        return {
            "downstream_stop": "acknowledged" if self.stopped else "unacknowledged",
            "downstream_controllers": ",".join(self.controllers),
            "downstream_controller_states": ",".join(
                f"{name}={state}" for name, state in self.controller_states.items()
            ),
            "downstream_vendor_stop": self.vendor_stop or "none",
            "downstream_stop_detail": self.detail,
        }


@runtime_checkable
class ControllerStopSeam(Protocol):
    """What a transport must expose for a ros2_control HAL to stop its controllers.

    The production implementation is ``RosControlTransport`` (real
    ``controller_manager`` services on a live graph); the in-memory
    ``SimTransport`` implements the same seam over a simulated controller
    table so the unit lane exercises the identical HAL code path. A HAL never
    sees ``rclpy``: it asks the seam to deactivate / activate controllers by
    name and to fire the vendor-specific stop (``std_srvs/Trigger`` service or
    ``std_msgs/Empty`` topic) it declared, and gets back an acknowledgement it
    can put in its ``DownstreamStopReport``.
    """

    def deactivate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Deactivate every named controller and confirm each reads ``inactive``."""
        ...

    def activate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Activate every named controller and confirm each reads ``active``."""
        ...

    def call_trigger(self, service: str, *, timeout_s: float) -> TriggerReport:
        """Call one ``std_srvs/Trigger`` service the HAL declared in ``vendor_stop_services``."""
        ...

    def publish_empty(self, topic: str) -> None:
        """Publish one ``std_msgs/Empty`` on a topic the HAL declared in ``vendor_stop_topics``."""
        ...


@runtime_checkable
class ControllerStoppable(Protocol):
    """The stop-side counterpart of ``RosControlDrivable``.

    The lifecycle node attaches a ``ControllerStopSeam`` to any real-mode HAL
    that answers these, and **refuses to configure** a ``RosControlDrivable``
    HAL that does not — a robot that can be driven but cannot be stopped from
    ``/openral/estop`` must not come up. ``RosControlHAL`` answers all four,
    so every subclass is covered; the members exist so the fleet conformance
    test (``tests/unit/test_real_hal_estop_fleet_conformance.py``) can pin the
    contract structurally.
    """

    def controller_names(self) -> list[str]:
        """Every ``controller_manager`` controller this HAL commands."""
        ...

    def vendor_stop_services(self) -> list[str]:
        """``std_srvs/Trigger`` services the vendor stop calls (may be empty)."""
        ...

    def vendor_stop_topics(self) -> list[str]:
        """``std_msgs/Empty`` topics the vendor stop publishes on (may be empty)."""
        ...

    def attach_controller_stop(self, seam: ControllerStopSeam) -> None:
        """Bind the live stop seam after construction."""
        ...


@runtime_checkable
class DownstreamStopReporting(Protocol):
    """HALs that can say whether their last downstream stop was acknowledged."""

    @property
    def last_stop_report(self) -> DownstreamStopReport | None:
        """Report of the most recent ``estop()``; ``None`` before the first / after reset."""
        ...


log = structlog.get_logger(__name__)

#: How long ``estop`` waits for ``controller_manager`` and the vendor stop to
#: acknowledge. Bounded, like the Galaxea sidecar's 4 s: an e-stop callback
#: that hangs forever is as bad as one that never fires.
_DEFAULT_STOP_TIMEOUT_S = 5.0

# Type alias for the injectable transport callable used in tests.
# Signature: (topic: str, msg: dict[str, object]) -> None
_PublishFn = Callable[[str, dict[str, object]], None]


def _default_publish(topic: str, msg: dict[str, object]) -> None:  # pragma: no cover
    """No-op publish used when no real ROS 2 node is available.

    In production this is replaced by the actual ``rclpy`` publisher at
    ``connect()`` time.  The no-op is only reached in fully isolated unit tests
    that do not inject a custom transport.
    """
    log.debug("hal.publish", topic=topic, fields=list(msg.keys()))


class RosControlHAL(HALBase):
    """ros2_control-backed HAL adapter.

    The adapter does not import ``rclpy`` directly so it can be unit-tested
    without a live ROS 2 installation.  In integration / HIL tests, inject a
    real publisher via the ``publish_fn`` parameter.

    Args:
        description: Normative ``RobotDescription`` for the target robot.
        controller_name: Name of the ``ros2_control`` joint trajectory
            controller, e.g. ``"joint_trajectory_controller"``.
        joint_state_topic: ROS 2 topic that publishes ``sensor_msgs/JointState``.
            Defaults to ``"/joint_states"``.
        command_topic: ROS 2 topic for joint trajectory commands.
            Defaults to ``"/<controller_name>/joint_trajectory"``.
        publish_fn: Callable used to send messages to ROS 2 topics.  Defaults
            to a no-op logger; replace with a real publisher in integration
            tests.
        state_fn: Callable that returns the latest raw joint state as a dict.
            Defaults to ``None``, in which case the adapter returns a zeroed
            ``JointState``.  Replace with a real subscriber callback in
            integration tests.
        staleness_limit_s: Maximum age (seconds) of a ``read_state()`` reading
            before ``ROSPerceptionStale`` is raised. ``None`` (default) reads
            the manifest's ``safety.joint_state_staleness_limit_s``; neither
            raises ``ROSConfigError``.
        stop_timeout_s: How long ``estop`` / ``reset_estop`` wait for the
            controller switch and the vendor stop to be acknowledged.

    The control rate comes from the manifest's ``action_spec.control_freq_hz``
    — the same field the runner ticks at and the dataset recorder stamps as
    fps — so there is exactly one place a rig declares it. Every published
    trajectory point gets ``time_from_start = horizon / control_freq_hz``:
    the controller is asked to reach each target exactly when the next
    command is due. A manifest without the field is **refused at
    construction**, so ``build_hal(mode="real")`` — and with it the lifecycle
    node's configure — stops and names the missing value. There is no
    fallback deadline: the 100 ms constant that used to fill this gap asked a
    30 Hz stream to cover each step in a third of its period (issue #303).

    Raises:
        ROSConfigError: If ``description.joints`` is empty, or the manifest
            declares no positive ``action_spec.control_freq_hz``.

    **Lifecycle e-stop.** Every ``RosControlHAL`` implements
    ``LifecycleEStopHAL``: ``/openral/estop`` reaches ``estop()``, which
    deactivates every controller in ``controller_names()`` through the
    attached ``ControllerStopSeam`` (a ``controller_manager`` deactivation
    holds position on a position interface, zeroes velocity / effort ones,
    drops late trajectories because the controller's subscriber goes
    inactive, and writes nothing until re-activated), then runs the
    vendor-specific ``_vendor_stop`` hook, and records a
    ``DownstreamStopReport``. The default ``estop_recovery`` is
    ``RESTART_REQUIRED`` — the conservative policy; a subclass whose
    controllers can be safely re-armed in place (OpenArm) opts into
    ``RESETTABLE`` and gets ``reset_estop`` — re-activation through the same
    seam, acknowledged before the HAL reconnects.
    """

    #: Conservative default: after a hardware e-stop the operator restarts the
    #: lifecycle node and re-aligns. Vendor adapters override deliberately.
    estop_recovery: EStopRecovery = EStopRecovery.RESTART_REQUIRED

    def __init__(
        self,
        description: RobotDescription,
        controller_name: str,
        *,
        joint_state_topic: str = "/joint_states",
        command_topic: str | None = None,
        publish_fn: _PublishFn | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float | None = None,
        stop_timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
    ) -> None:
        """Initialise the adapter; does not open any connection yet."""
        if not description.joints:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' has no joints; "
                "cannot initialise RosControlHAL."
            )
        spec = description.action_spec
        control_rate_hz = None if spec is None else spec.control_freq_hz
        if control_rate_hz is None or not control_rate_hz > 0.0:
            raise ROSConfigError(
                f"RobotDescription '{description.name}' declares no positive "
                f"action_spec.control_freq_hz (got {control_rate_hz!r}); "
                f"{type(self).__name__} cannot be built without it. It sets every "
                "trajectory point's time_from_start (horizon / rate) and the runner's "
                "tick — add `action_spec: {dim, representation, control_freq_hz}` "
                "to the robot manifest."
            )
        self._control_rate_hz: float = float(control_rate_hz)
        self.description = description
        self._controller_name = controller_name
        self._joint_state_topic = joint_state_topic
        self._command_topic = command_topic or f"/{controller_name}/joint_trajectory"
        self._publish_fn: _PublishFn = publish_fn or _default_publish
        self._state_fn = state_fn
        self._staleness_limit_s = resolve_staleness_limit_s(description, staleness_limit_s)
        self._stop_timeout_s = stop_timeout_s
        self._stop_seam: ControllerStopSeam | None = None
        self._last_stop_report: DownstreamStopReport | None = None

        self._connected: bool = False
        self._last_state_time: float = 0.0
        # Set by `attach_transport` to the transport's real per-message arrival
        # clock. While it is None the HAL has no way to know how old its state
        # is (see `read_state`).
        self._stamp_fn: Callable[[], float] | None = None
        self._joint_names: list[str] = [j.name for j in description.joints]

    # ── Transport wiring ───────────────────────────────────────────────────────

    def attach_transport(
        self,
        publish_fn: _PublishFn,
        state_fn: Callable[[], dict[str, object]],
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind this HAL to a live transport after construction.

        ``build_hal`` constructs the HAL from the manifest alone, before any ROS
        node exists, so a real deployment cannot pass ``publish_fn``/``state_fn``
        to ``__init__``. The lifecycle node calls this once it has a node to
        create publishers on — which is what makes a real ros2_control robot
        work without any per-robot wiring code.

        Args:
            publish_fn: Sends one command dict to a controller topic.
            state_fn: Returns the newest joint state as a raw dict.
            stamp_fn: Returns the ``time.monotonic()`` timestamp of the newest
                joint state. Without it ``read_state`` can only measure age from
                ``connect()``, which is not a freshness check at all.

        Raises:
            ROSConfigError: If ``stamp_fn`` is given but is not callable.
        """
        # Checked here rather than at first use: `stamp_fn` feeds the staleness
        # watchdog, so a caller that passes a *value* (a property read once at
        # wire-up, say) instead of a callable would otherwise fail deep in the
        # hot path, on hardware, with the robot already connected.
        if stamp_fn is not None and not callable(stamp_fn):
            raise ROSConfigError(
                f"attach_transport(stamp_fn=...) needs a callable returning the newest "
                f"joint state's time.monotonic() timestamp, got {type(stamp_fn).__name__}. "
                "A property must be wrapped, e.g. `lambda: transport.last_stamp`."
            )
        self._publish_fn = publish_fn
        self._state_fn = state_fn
        self._stamp_fn = stamp_fn
        self._last_state_time = time.monotonic()

    def attach_controller_stop(self, seam: ControllerStopSeam) -> None:
        """Bind the live controller stop seam after construction.

        Like ``attach_transport``, this exists because ``build_hal`` runs
        before any ROS node exists. The lifecycle node attaches the
        ``RosControlTransport`` (which implements the seam) under
        ``hal_mode:=real``; unit tests attach a ``SimTransport``.

        Args:
            seam: Anything satisfying ``ControllerStopSeam``.

        Raises:
            ROSConfigError: If ``seam`` does not satisfy the protocol — the
                e-stop path must never discover a missing method on hardware.
        """
        if not isinstance(seam, ControllerStopSeam):
            raise ROSConfigError(
                f"attach_controller_stop() needs a ControllerStopSeam "
                f"(deactivate_controllers / activate_controllers / call_trigger / "
                f"publish_empty); got {type(seam).__name__}."
            )
        self._stop_seam = seam

    def controller_names(self) -> list[str]:
        """Return every ``controller_manager`` controller ``estop`` deactivates.

        One entry for a single-controller arm; robots whose controllers are
        split (the bimanual OpenArm) override this alongside
        ``command_bindings`` so the stop covers every topic they publish to.
        """
        return [self._controller_name]

    def vendor_stop_services(self) -> list[str]:
        """``std_srvs/Trigger`` services ``_vendor_stop`` calls; none by default.

        Declared up front so the transport creates the client at wire-up and
        the seam can refuse an undeclared service loudly instead of the stop
        path discovering a missing client on hardware.
        """
        return []

    def vendor_stop_topics(self) -> list[str]:
        """``std_msgs/Empty`` topics ``_vendor_stop`` publishes on; none by default."""
        return []

    @property
    def last_stop_report(self) -> DownstreamStopReport | None:
        """Report of the most recent ``estop()``; ``None`` before it / after ``reset_estop``."""
        return self._last_stop_report

    def command_bindings(self) -> dict[str, ControllerKind]:
        """Return each controller topic this HAL publishes to, with its kind.

        One entry for a single-controller arm; robots whose controllers are
        split (the bimanual OpenArm's arm+gripper per side) override this.
        Insertion order is the fan-out order `send_action` uses, so it is
        load-bearing.

        This is the override point for a new robot: declaring the topic *and*
        the message type its controller speaks is what lets the transport
        publish without guessing, so adding a robot needs no transport code.
        """
        return {self._command_topic: ControllerKind.JOINT_TRAJECTORY}

    def command_topics(self) -> list[str]:
        """Return every controller topic this HAL publishes to, in fan-out order.

        Derived from `command_bindings` so the topic list and the declared wire
        formats cannot drift apart. Override `command_bindings`, not this.
        """
        return list(self.command_bindings())

    def ros2_control_joint_names(self) -> list[str]:
        """Return the joint names in ros2_control's namespace, in action order.

        Defaults to the manifest's own names; robots whose URDF names differ
        from the manifest's (again, OpenArm) override it. ``/joint_states`` is
        keyed by these, so a transport matches on them.
        """
        return list(self._joint_names)

    @property
    def joint_state_topic(self) -> str:
        """The aggregated ``sensor_msgs/JointState`` topic this HAL reads."""
        return self._joint_state_topic

    @property
    def controller_name(self) -> str:
        """Name of the primary ``ros2_control`` controller this HAL commands."""
        return self._controller_name

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the connection to the robot hardware or simulator.

        Raises:
            ROSRuntimeError: If already connected.
        """
        if self._connected:
            raise ROSRuntimeError(f"RosControlHAL('{self.description.name}') is already connected.")
        log.info(
            "hal.connect",
            robot=self.description.name,
            controller=self._controller_name,
            joint_state_topic=self._joint_state_topic,
            command_topic=self._command_topic,
        )
        self._connected = True
        self._last_state_time = time.monotonic()

    # ── Hot path ───────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state snapshot.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.

        Returns:
            Latest ``JointState`` for all joints in ``description.joints``.
        """
        self._require_connected("read_state")
        # With a transport attached, age is measured from the arrival of the
        # newest joint state — the only thing that actually says whether the
        # robot is still talking to us. Without one, the best available
        # reference is `connect()`; that is a liveness floor, not a freshness
        # check, and it is why a transport must supply `stamp_fn` on real
        # hardware. A transport that has received nothing yet reports 0.0,
        # which must read as "infinitely stale", never as "fresh at epoch".
        if self._stamp_fn is not None:
            last = self._stamp_fn()
            age = float("inf") if last <= 0.0 else time.monotonic() - last
        else:
            age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        raw: dict[str, object] = {}
        if self._state_fn is not None:
            raw = self._state_fn()

        return JointState(
            name=self._joint_names,
            position=_raw_floats(raw, "position", n),
            velocity=_raw_floats(raw, "velocity", n),
            effort=_raw_floats(raw, "effort", n),
            stamp_ns=int(time.time_ns()),
        )

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the ros2_control joint trajectory controller.

        Args:
            action: The ``Action`` produced by a Skill.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not in the robot's
                ``supported_control_modes``, or if the joint target dimensions
                do not match the robot's joint count.
        """
        self._require_connected("send_action")
        self._validate_action(action)

        # `joint_names` travels with the command so the transport forwards what
        # the HAL chose rather than keeping a second copy of the mapping that
        # can drift from it (the rule ADR-0102 set for the OpenArm fan-out).
        msg: dict[str, object] = {
            "control_mode": action.control_mode,
            "horizon": action.horizon,
            "joint_names": self.ros2_control_joint_names(),
            "joint_targets": action.joint_targets,
            "stamp_ns": action.stamp_ns,
        }
        msg["time_from_start_s"] = self.time_from_start_s(action)
        self._publish_fn(self._command_topic, msg)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
        )

    def time_from_start_s(self, action: Action) -> float:
        """Trajectory deadline for ``action``: ``horizon / control_freq_hz``.

        One command arrives per control period, so a chunk of ``horizon``
        steps should be reached just as its replacement is due; the transport
        spreads the chunk's rows evenly up to this deadline.

        Example:
            >>> from openral_core.schemas import Action, ControlMode
            >>> from openral_hal.openarm_real import OpenArmRealHAL
            >>> hal = OpenArmRealHAL(require_can_links=False)  # manifest: 30 Hz
            >>> a = Action(
            ...     control_mode=ControlMode.JOINT_POSITION,
            ...     horizon=3,
            ...     joint_targets=[[0.0] * 16] * 3,
            ... )
            >>> round(hal.time_from_start_s(a), 3)
            0.1
        """
        return max(int(action.horizon), 1) / self._control_rate_hz

    # ── Safety ─────────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop: drop the connection, stop downstream, raise.

        Ordering is deliberate. The connection flag drops **first**, so no
        further ``send_action`` from this HAL can reach the wire even if the
        downstream stop below takes its full timeout. Then every controller in
        ``controller_names()`` is deactivated through the attached
        ``ControllerStopSeam`` (acknowledged by ``controller_manager`` and
        confirmed via its controller list), then the vendor-specific
        ``_vendor_stop`` runs. Both are attempted even if the first fails, so
        the report names everything that did and did not acknowledge.

        The outcome is recorded in ``last_stop_report`` rather than changing
        the exception type: the HAL Protocol promises ``ROSEStopRequested``
        from every ``estop()``, and the lifecycle node reads the report to log
        FATAL and publish ``downstream_stop=unacknowledged`` when the stop was
        not proven. With no seam attached (a unit test driving the HAL with
        bare callables) the report says so and ``stopped`` is ``False`` — a
        local latch is never reported as a downstream stop.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name, controllers=self.controller_names())
        self._connected = False
        report = self._stop_downstream()
        self._last_stop_report = report
        verdict = "acknowledged" if report.stopped else "NOT acknowledged"
        raise ROSEStopRequested(
            f"Emergency stop triggered on robot '{self.description.name}' "
            f"(downstream stop {verdict}: {report.detail})."
        )

    def _stop_downstream(self) -> DownstreamStopReport:
        """Deactivate the controllers, run the vendor stop, and report both."""
        names = tuple(self.controller_names())
        seam = self._stop_seam
        if seam is None:
            log.critical(
                "hal.estop.unproven",
                robot=self.description.name,
                reason="no ControllerStopSeam attached; only the local latch holds",
            )
            return DownstreamStopReport(
                stopped=False,
                controllers=names,
                detail="no controller stop seam attached; only the local latch holds",
            )

        # Both steps catch *every* exception, not only ``ROSError``: the seam
        # runs on rclpy, whose ``ShutdownException`` /
        # ``ExternalShutdownException`` / ``InvalidHandle`` are plain
        # ``Exception`` subclasses. One escaping here would skip the vendor
        # step, leave ``last_stop_report`` unassigned and break the Protocol's
        # "always ``ROSEStopRequested``" promise — the failure is recorded in
        # the report instead, never swallowed.
        problems: list[str] = []
        try:
            switch = seam.deactivate_controllers(names, timeout_s=self._stop_timeout_s)
        except Exception as exc:  # reason: a seam fault must land in the report, not escape estop()
            switch = ControllerSwitchReport(
                ok=False, controllers=names, detail=f"{type(exc).__name__}: {exc}"
            )
        if not switch.ok:
            problems.append(f"controller deactivation not acknowledged: {switch.detail}")

        vendor = ""
        try:
            vendor = self._vendor_stop(seam)
        except Exception as exc:  # reason: a seam fault must land in the report, not escape estop()
            problems.append(f"vendor stop failed: {type(exc).__name__}: {exc}")

        report = DownstreamStopReport(
            stopped=not problems,
            controllers=names,
            controller_states=switch.states,
            vendor_stop=vendor,
            detail="; ".join(problems) if problems else switch.detail or "controllers inactive",
        )
        if report.stopped:
            log.critical(
                "hal.estop.downstream_stopped", robot=self.description.name, **report.fields()
            )
        else:
            log.critical(
                "hal.estop.downstream_unacknowledged",
                robot=self.description.name,
                **report.fields(),
            )
        return report

    def _vendor_stop(self, seam: ControllerStopSeam) -> str:
        """Vendor-specific stop beyond controller deactivation; return its label.

        The base has none: for a plain ros2_control robot, deactivating the
        controller *is* the stop (and for ``franka_hardware`` it is what calls
        ``libfranka``'s ``stopRobot()``). Subclasses call one of the services
        / topics they declared in ``vendor_stop_services`` /
        ``vendor_stop_topics`` and return its name for the report, raising a
        ``ROSError`` subclass when it was not acknowledged.
        """
        return ""

    def _vendor_reset(self, seam: ControllerStopSeam) -> None:
        """Vendor-specific re-arm before controllers are re-activated; no-op by default."""
        return None

    def reset_estop(self) -> None:
        """Re-arm after ``estop()`` — only for ``RESETTABLE`` adapters.

        Runs the vendor reset, re-activates every controller through the seam
        and, only once ``controller_manager`` confirms them ``active``, marks
        the HAL connected again. The lifecycle node calls this from
        ``/openral/estop_cleared`` and keeps its latch when it raises, so no
        command can flow before the downstream re-arm succeeded.

        Raises:
            ROSRuntimeError: If this adapter's ``estop_recovery`` is not
                ``RESETTABLE`` (a ``RESTART_REQUIRED`` robot must never be
                re-armed in process), if no seam is attached, or if the
                re-activation was not acknowledged.
        """
        if self.estop_recovery is not EStopRecovery.RESETTABLE:
            raise ROSRuntimeError(
                f"{type(self).__name__} declares estop_recovery="
                f"{self.estop_recovery.value}; in-process reset is forbidden — restart the "
                "lifecycle node and re-align."
            )
        seam = self._stop_seam
        if seam is None:
            raise ROSRuntimeError(
                f"{type(self).__name__}.reset_estop(): no ControllerStopSeam attached, so the "
                "controllers cannot be re-activated."
            )
        names = tuple(self.controller_names())
        self._vendor_reset(seam)
        switch = seam.activate_controllers(names, timeout_s=self._stop_timeout_s)
        if not switch.ok:
            raise ROSRuntimeError(
                f"{type(self).__name__}.reset_estop(): controller re-activation not "
                f"acknowledged: {switch.detail}"
            )
        log.warning(
            "hal.estop.reset", robot=self.description.name, controllers=names, states=switch.states
        )
        self._connected = True
        self._last_state_time = time.monotonic()
        self._last_stop_report = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _validate_action(self, action: Action) -> None:
        supported = self.description.capabilities.supported_control_modes
        if supported and action.control_mode not in supported:
            raise ROSConfigError(
                f"Action control_mode '{action.control_mode}' is not in "
                f"supported_control_modes {supported} for robot "
                f"'{self.description.name}'."
            )
        self._validate_action_dims(action, len(self._joint_names))
