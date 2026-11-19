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
from openral_hal._slot_group import GRIPPER_MODES, SlotGroupStager, compose_slot_group_action
from openral_hal.protocol import EStopRecovery

__all__ = [
    "ControllerKind",
    "ControllerStopSeam",
    "ControllerStoppable",
    "ControllerSwapSeam",
    "ControllerSwitchReport",
    "DownstreamStopReport",
    "DownstreamStopReporting",
    "GroupedRosControlHAL",
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

    #: NVIDIA ``isaac_ros_deploy_ros2_control/ForwardJointCommandController``
    #: in trajectory mode, commanded via
    #: ``isaac_ros_deploy_interfaces/msg/JointCommandTrajectory`` (a named,
    #: chunked ``[horizon, joints]`` block with a uniform ``step_dt`` that the
    #: controller interpolates through). The Unitree hg family's real HAL
    #: (``openral_hal.unitree_real``) drives the upper body through it.
    JOINT_COMMAND_TRAJECTORY = "joint_command_trajectory"

    #: A ``geometry_msgs/msg/Twist`` velocity command read by a controller's
    #: topic input — NVIDIA ``InferenceController``'s ``command/body/velocity``
    #: source, which feeds the S0 whole-body policy. Commanded from the
    #: command dict's ``twist`` (6 floats: vx, vy, vz, wx, wy, wz), not from
    #: ``joint_targets``.
    TWIST = "twist"


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

    Produced by ``RosControlHAL.estop``, the Interbotix-backed
    ``AlohaHAL.estop``, the serial-bus ``SO100FollowerHAL.estop`` and the
    sidecar-backed ``GalaxeaA1HAL.estop``, and read by the lifecycle node, which logs FATAL and
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
class ControllerSwapSeam(Protocol):
    """A stop seam that can swap controllers in **one** ``switch_controller`` request.

    Plain deactivation is the stop for an arm: an inactive
    ``JointTrajectoryController`` holds its last position command. It is not a
    stop for a robot whose hardware interface *releases* every joint no
    controller claims — the Unitree hg ``SystemInterface`` sends motor mode 0
    (zero torque) for an unclaimed joint, so deactivating a standing humanoid's
    controllers drops it. Such a robot must hand its joints to a holding
    controller in the same request that releases them, so no control cycle
    runs with the joints unclaimed. The swap is requested STRICT: a BEST_EFFORT
    one drops an activation that cannot happen (a controller not ``inactive``)
    yet applies the deactivations — the joints released. STRICT refuses the
    whole request instead, so the seam also lists states and configures a
    controller (the pre-check ``UnitreeHgRealHAL`` runs). A failure *inside*
    the switch (an ``on_activate`` that errors) keeps the deactivations under
    either strictness, so the seam also activates one controller alone — the
    retry (#566).
    ``RosControlTransport`` and ``SimTransport`` both implement this; a HAL that
    needs it refuses a seam without it at ``attach_controller_stop`` time.
    """

    def swap_controllers(
        self,
        *,
        deactivate: Sequence[str],
        activate: Sequence[str],
        timeout_s: float,
    ) -> ControllerSwitchReport:
        """Deactivate ``deactivate`` and activate ``activate`` in one STRICT switch, then confirm.

        ``states`` carries the post-switch listing of every named controller even when
        the switch was refused.
        """
        ...

    def activate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Activate every named controller and confirm each reads ``active``."""
        ...

    def controller_states(self, *, timeout_s: float) -> dict[str, str]:
        """``{name: state}`` for every loaded controller; empty if the manager did not answer."""
        ...

    def configure_controller(self, name: str, *, timeout_s: float) -> bool:
        """Configure one loaded controller; ``True`` when the manager acknowledged it."""
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
    fallback deadline: a fixed 100 ms constant in this gap asks a
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
        stamp_ns = self._fresh_sample_stamp_ns()
        n = len(self._joint_names)
        raw: dict[str, object] = {}
        if self._state_fn is not None:
            raw = self._state_fn()

        return JointState(
            name=self._joint_names,
            position=_raw_floats(raw, "position", n),
            velocity=_raw_floats(raw, "velocity", n),
            effort=_raw_floats(raw, "effort", n),
            stamp_ns=stamp_ns,
        )

    def _fresh_sample_stamp_ns(self) -> int:
        """When the newest joint state arrived (ns), after the connection + staleness checks.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than ``staleness_limit_s``.
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
        sample_age_s = age if self._stamp_fn is not None else 0.0
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )
        # When the sample arrived, not when it was read: a consumer that pairs joint
        # states with other sensors by stamp (the robot self-filter's skew check) must
        # see how old the sample really is.
        return int(time.time_ns()) - int(max(sample_age_s, 0.0) * 1e9)

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

        Idempotent once acknowledged: a second ``estop()`` before ``reset_estop`` (a
        composed robot stopping every part after one part stopped itself, or the node's
        latch after the HAL's own stop) re-raises with the stored report instead of
        re-deactivating controllers that are already inactive, which a STRICT
        ``controller_manager`` switch refuses — that would overwrite an acknowledged
        report with an unacknowledged one. An unacknowledged stop is always retried.

        Raises:
            ROSEStopRequested: Always.
        """
        done = self._last_stop_report
        if not self._connected and done is not None and done.stopped:
            raise ROSEStopRequested(
                f"Emergency stop on robot '{self.description.name}': already stopped "
                f"(downstream stop acknowledged: {done.detail})."
            )
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
            switch = self._stop_switch(seam, names)
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

    def _stop_switch(
        self, seam: ControllerStopSeam, names: tuple[str, ...]
    ) -> ControllerSwitchReport:
        """The ``controller_manager`` switch ``estop`` performs; plain deactivation by default.

        A subclass whose hardware releases unclaimed joints (the Unitree hg
        humanoids) overrides this to hand ``names``' joints to a holding
        controller in the same request — see ``ControllerSwapSeam``. Whatever it
        returns is folded into the ``DownstreamStopReport``; an exception it
        raises is caught by ``_stop_downstream`` and reported, never escapes.
        """
        return seam.deactivate_controllers(names, timeout_s=self._stop_timeout_s)

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


class GroupedRosControlHAL(RosControlHAL):
    """``RosControlHAL`` whose joints are split across several trajectory controllers.

    A bimanual ros2_control arm (the OpenArm's arm + gripper per side, the dual
    Rizon's one controller per arm) commands one action vector through several
    ``JointTrajectoryController`` instances. This base owns that fan-out, so each
    robot declares only its ``groups`` — ``(controller_name, n_joints)`` in
    action-vector order — and its own preflight. Three properties hold for every
    subclass:

    * **All or nothing.** ``send_action`` builds every controller's message before
      publishing any, so a malformed action cannot leave one arm on a new chunk and
      the other on a stale one.
    * **Named, not positional.** Each message carries its own ``joint_names``, and
      ``read_state`` matches ``/joint_states`` by name, raising on a missing joint
      rather than zero-filling it (``sensor_msgs/JointState`` has no order, and each
      controller's joints arrive independently).
    * **Slot groups (ADR-0102).** A slot-dispatched tick is staged until every slot
      passed safety and is published as one reassembled vector.

    ros2_control joint names come from each joint's ``sim_joint_name`` (the URDF
    name), or its manifest ``name`` where the two agree.

    Args:
        description: The robot manifest.
        groups: ``(controller_name, n_joints)`` per controller, in action order;
            the counts must sum to the manifest's joint count.
        joint_state_topic / publish_fn / state_fn / staleness_limit_s: As
            ``RosControlHAL``.

    Raises:
        ROSConfigError: If ``groups`` is empty or does not cover every joint.

    Example:
        >>> from openral_hal.openarm_real import OpenArmRealHAL
        >>> OpenArmRealHAL(require_can_links=False).command_topics()[2]
        '/right_joint_trajectory_controller/joint_trajectory'
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        groups: Sequence[tuple[str, int]],
        joint_state_topic: str = "/joint_states",
        publish_fn: _PublishFn | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float | None = None,
    ) -> None:
        """Initialise the fan-out table; opens no connection."""
        names = [name for name, _ in groups]
        if (
            not groups
            or any(n <= 0 for _, n in groups)
            or len(set(names)) != len(names)
            or sum(n for _, n in groups) != len(description.joints)
        ):
            raise ROSConfigError(
                f"{type(self).__name__}: controller groups {list(groups)} must be distinct "
                f"controllers of at least one joint each, covering the "
                f"{len(description.joints)} joints of '{description.name}' exactly."
            )
        super().__init__(
            description,
            controller_name=groups[0][0],
            joint_state_topic=joint_state_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
        self._ros2_names: list[str] = [str(j.sim_joint_name or j.name) for j in description.joints]
        self._controller_names = names
        # (command topic, action slice, ros2_control joint names) per controller.
        table: list[tuple[str, slice, list[str]]] = []
        start = 0
        for name, n in groups:
            span = slice(start, start + n)
            table.append((f"/{name}/joint_trajectory", span, self._ros2_names[span]))
            start += n
        self._command_groups: tuple[tuple[str, slice, list[str]], ...] = tuple(table)
        self._slot_group = SlotGroupStager()

    def ros2_control_joint_names(self) -> list[str]:
        """Return the ros2_control joint names in action-vector order."""
        return list(self._ros2_names)

    def command_bindings(self) -> dict[str, ControllerKind]:
        """Return every controller's ``joint_trajectory`` topic, in fan-out order."""
        return {topic: ControllerKind.JOINT_TRAJECTORY for topic, _, _ in self._command_groups}

    def controller_names(self) -> list[str]:
        """Return every controller ``estop`` deactivates, in fan-out order."""
        return list(self._controller_names)

    def read_state(self) -> JointState:
        """Return the latest joint state, matched by name into manifest order.

        Raises:
            ROSRuntimeError: If not connected, or if the incoming state carries no
                joint names or is missing joints the manifest declares.
            ROSPerceptionStale: If the last reading exceeds the staleness deadline.
        """
        if self._state_fn is None:
            # No subscriber wired: the base returns a zeroed state in manifest order.
            return super().read_state()
        stamp_ns = self._fresh_sample_stamp_ns()
        raw = self._state_fn()  # one sample: the freshness check above is its own
        incoming = raw.get("name")
        if not isinstance(incoming, list):
            raise ROSRuntimeError(
                f"{self._joint_state_topic} sample carries no joint names: "
                f"{type(self).__name__} matches joints by name, never by position."
            )

        index_of = {str(n): i for i, n in enumerate(incoming)}
        absent = [n for n in self._ros2_names if n not in index_of]
        if absent:
            raise ROSRuntimeError(
                f"{self._joint_state_topic} is missing {len(absent)} joint(s) "
                f"declared by '{self.description.name}': {absent}. Check that every "
                "controller is active — each publishes its own joints."
            )

        def _reordered(key: str, *, absent: list[float]) -> list[float]:
            values = raw.get(key)
            if not isinstance(values, list) or len(values) != len(incoming):
                return absent
            return [float(values[index_of[n]]) for n in self._ros2_names]

        zeros = [0.0] * len(self._ros2_names)
        return JointState(
            name=list(self._joint_names),
            position=_reordered("position", absent=zeros),
            velocity=_reordered("velocity", absent=zeros),
            # An absent effort channel stays empty, never zero-filled: zeros would
            # fake an effort reading the driver never published.
            effort=_reordered("effort", absent=[]),
            stamp_ns=stamp_ns,
        )

    def send_action(self, action: Action) -> None:
        """Fan one action out to every controller, all or nothing.

        Every message is built and validated before the first publish. A publish that
        fails after an earlier controller already took its target cannot be undone, so
        it stops every controller instead (``estop``) and the tick is never committed.

        Raises:
            ROSEStopRequested: If a later controller's publish fails after an earlier
                one succeeded.
            ROSRuntimeError: If not connected.
            ROSConfigError: If the control mode is unsupported, the target width
                does not match the manifest, or a gripper action arrives outside a
                slot group.
        """
        self._require_connected("send_action")

        # ADR-0102: a slot-dispatched tick arrives as several typed actions, none a
        # whole-robot command on its own. Reassemble it and fall through to the
        # single-vector publish, so the all-or-nothing guarantee covers the group.
        group: list[Action] | None = None
        if int(action.tick_group_size) > 1:
            group = self._slot_group.stage(action)
            if group is None:
                return
            log.debug(
                "hal.send_action.slot_group",
                robot=self.description.name,
                slots=len(group),
                tick=group[0].tick_index,
                modes=[a.control_mode.value for a in group],
            )
            action = compose_slot_group_action(group, [j.name for j in self.description.joints])
        else:
            # Ungrouped ticked actions (starting-pose ramp, approach) share the
            # watermark, so a restarted runner's renumbering is seen here too.
            self._slot_group.admit(action)

        self._validate_action(action)

        if action.control_mode in GRIPPER_MODES:
            # A lone gripper action: the controllers are commanded from one joint
            # vector, so there is nowhere to put it. Refuse loudly — dropping it
            # would move the arms while the grippers never actuate.
            raise ROSConfigError(
                f"{type(self).__name__} cannot execute a standalone "
                f"{action.control_mode.value} action: its controllers are commanded as "
                "one joint vector, so a gripper command is only placeable inside a slot "
                "group (tick_group_size > 1, ADR-0102)."
            )

        if action.joint_targets is None:
            log.debug("hal.send_action.empty", robot=self.description.name)
            return

        deadline = self.time_from_start_s(action)
        outbound = [
            (
                topic,
                {
                    "control_mode": action.control_mode,
                    "horizon": action.horizon,
                    "joint_names": names,
                    "joint_targets": [list(step[span]) for step in action.joint_targets],
                    "stamp_ns": action.stamp_ns,
                    "time_from_start_s": deadline,
                },
            )
            for topic, span, names in self._command_groups
        ]
        for sent, (topic, msg) in enumerate(outbound):
            try:
                self._publish_fn(topic, msg)
            except Exception as exc:  # reason: a partial publish must stop every controller
                if sent == 0:
                    raise  # nothing reached a controller: an ordinary refused tick
                # An earlier controller already holds the new target and this one does
                # not: the arms are commanded inconsistently. Stop every controller
                # (raises ROSEStopRequested) rather than leave a half-applied tick.
                log.critical(
                    "hal.send_action.partial_publish",
                    robot=self.description.name,
                    published=[t for t, _ in outbound[:sent]],
                    failed=topic,
                    error=repr(exc),
                )
                self.estop()
        if group is not None:
            self._slot_group.commit(group, applied=action)
        else:
            self._slot_group.commit_tick(action)
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
            controllers=len(outbound),
        )

    def disconnect(self) -> None:
        """Close the connection, dropping any half-staged slot group (and the watermark)."""
        self._slot_group.reset()
        super().disconnect()

    @property
    def last_applied_action(self) -> Action | None:
        """The command the robot was last actually given (``None`` = none yet).

        For a slot group this is the composed full-dof ``JOINT_POSITION`` action; set
        only once the apply succeeded, in step with ``last_committed_tick``. The
        lifecycle node folds it into the grasp trigger.
        """
        return self._slot_group.last_applied_action

    @property
    def last_committed_tick(self) -> int:
        """Inference tick of the last slot group published (0 = none since disconnect)."""
        return self._slot_group.last_committed_tick

    @property
    def last_committed_session(self) -> int:
        """``runner_session_id`` of ``last_committed_tick`` (0 = none / session-less)."""
        return self._slot_group.last_committed_session

    def discard_staged_slots(self) -> int:
        """Drop a half-staged slot group, keeping the committed watermark.

        Called by the HAL lifecycle node on every ``/openral/estop`` latch and
        ``/openral/estop_cleared``, so a tick cut short by a stop never outlives it.

        Returns:
            How many staged slots were dropped.
        """
        return self._slot_group.discard()

    def estop(self) -> None:
        """Drop any half-staged slot group, then the base stop (deactivate every controller).

        The committed watermark is kept: a pre-stop tick replayed after the stop is
        still refused as stale.

        Raises:
            ROSEStopRequested: Always, from the base implementation.
        """
        self._slot_group.discard()
        super().estop()
