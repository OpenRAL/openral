"""Production ``rclpy`` transport for every ``RosControlHAL`` robot.

`RosControlHAL` is deliberately wire-format-free: it turns an `Action` into a
plain dict and hands it to an injected `publish_fn`. Something has to be that
callable on real hardware. Until this module existed nothing was — the default
was `_default_publish`, a `log.debug` no-op — so `openral deploy run` against a
real ros2_control arm sent commands into a logger and read back zeros, for every
robot in that family (OpenArm, UR5e, UR10e, Franka, Sawyer). Only the serial
arms (SO-100/SO-101, whose HAL owns its own bus) actually moved, which is why
the gap went unnoticed.

This transport is robot-agnostic on purpose. It is built entirely from what the
HAL already reports about itself:

* `command_bindings()` — one publisher per entry, typed by the entry's
  `ControllerKind`, so a single-controller UR and the four-controller bimanual
  OpenArm are the same code path.
* `ros2_control_joint_names()` — the names `/joint_states` is keyed by.
* `joint_state_topic` — the aggregated state topic.

Adding a robot therefore needs no transport code at all: give its HAL those
three (the base class already answers all of them) and the lifecycle node wires
this automatically. A robot whose controllers speak a format not yet in
`ControllerKind` is the one case that needs a change here — and it fails loudly
at wire-up rather than publishing a type DDS will silently drop.

Joint state is merged **by name, never by index**. Independently publishing
controllers give no ordering guarantee, and on a robot whose controllers are
split per limb each message carries only that limb's joints.

Arrival time is tracked per message and surfaced via `last_arrival`, which is
what lets `RosControlHAL.read_state` measure real staleness instead of time
since `connect()`.

**Stop seam.** The transport also implements `ControllerStopSeam`, which is
how `/openral/estop` reaches the controller: `deactivate_controllers` calls
`controller_manager`'s `switch_controller` (STRICT) for the HAL's
`controller_names()` and then confirms through `list_controllers` that every
one reads `inactive`; `activate_controllers` is the mirror for a `RESETTABLE`
re-arm. Vendor stops go through `call_trigger` (`std_srvs/Trigger` clients)
and `publish_empty` (`std_msgs/Empty` publishers), both created at wire-up
from what the HAL declared in `vendor_stop_services()` /
`vendor_stop_topics()`, so the stop path never creates an entity — and so a
vendor stop on an undeclared name is refused rather than silently absent.

Service calls run on a private helper node with its own executor: the e-stop
callback runs on the lifecycle node's single-threaded executor, and a future
issued from inside a callback can only complete if something *else* spins
the client. Bounded by the HAL's `stop_timeout_s`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from openral_hal.ros_control import ControllerKind, ControllerSwitchReport, TriggerReport

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from collections.abc import Sequence

    from rclpy.node import Node

__all__ = ["RosControlDrivable", "RosControlTransport"]

#: `controller_manager` namespace every ros2_control robot in this repo runs
#: its manager under. Overridable per transport for a namespaced manager.
_DEFAULT_CONTROLLER_MANAGER = "/controller_manager"

#: `controller_manager_msgs/srv/SwitchController` strictness: refuse the whole
#: switch if any named controller cannot move. A best-effort stop that leaves
#: one controller active is not a stop.
_STRICT = 2


@runtime_checkable
class RosControlDrivable(Protocol):
    """What a HAL must expose to be driven by `RosControlTransport`.

    Membership is **structural, not by ancestry**, so a HAL that reimplements
    the ros2_control fan-out on `HALBase` rather than inheriting still gets
    wired. Gating the lifecycle node on `isinstance(..., RosControlHAL)` would
    leave such a robot with no transport and no error — the failure this whole
    surface exists to prevent.

    A HAL opts in by answering these; the base class answers all four, so for
    most robots that is automatic.
    """

    def command_bindings(self) -> dict[str, ControllerKind]:
        """Controller topics this HAL publishes to, each with its wire format."""
        ...

    def ros2_control_joint_names(self) -> list[str]:
        """Joint names in ros2_control's namespace, in action-vector order."""
        ...

    @property
    def joint_state_topic(self) -> str:
        """Aggregated ``sensor_msgs/JointState`` topic to read."""
        ...

    def attach_transport(
        self,
        publish_fn: Callable[[str, dict[str, object]], None],
        state_fn: Callable[[], dict[str, object]],
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind the live transport's callables to this HAL."""
        ...


log = structlog.get_logger(__name__)

#: Command QoS per CLAUDE.md §2: control is RELIABLE, VOLATILE, shallow. A
#: trajectory that arrives late is worse than one that is dropped and resent.
_COMMAND_DEPTH = 1

#: State QoS. `joint_state_broadcaster` publishes RELIABLE, and a BEST_EFFORT
#: subscriber is compatible with a RELIABLE publisher, so this reads a live
#: broadcaster while still tolerating a lossy link.
_STATE_DEPTH = 10


class RosControlTransport:
    """Bridges one `RosControlHAL` to live ros2_control topics.

    Args:
        node: A live `rclpy` node the publishers/subscription are created on.
            Ownership stays with the caller (the lifecycle node).
        command_topics: Controller topics, from `hal.command_topics()`.
        joint_names: ros2_control joint names, from
            `hal.ros2_control_joint_names()`.
        joint_state_topic: Aggregated state topic, from `hal.joint_state_topic`.
        command_kinds: Wire format per topic, from `hal.command_bindings()`.
            A topic absent from the mapping defaults to
            `ControllerKind.JOINT_TRAJECTORY`, which is what every
            ros2_control robot in this repo runs.
        controller_names: Controllers the stop seam switches, from
            `hal.controller_names()`. Empty leaves the seam unwired (the HAL
            then reports its stop as unproven).
        trigger_services: `std_srvs/Trigger` services the HAL's vendor stop
            may call, from `hal.vendor_stop_services()`.
        empty_topics: `std_msgs/Empty` topics the HAL's vendor stop may
            publish on, from `hal.vendor_stop_topics()`.
        controller_manager: Namespace of the `controller_manager` node.

    Raises:
        ROSConfigError: If no command topic or no joint name was given — a HAL
            that reports neither cannot be driven, and failing here is far
            cheaper than a silent no-op on hardware.
    """

    def __init__(
        self,
        node: Node,
        *,
        command_topics: list[str],
        joint_names: list[str],
        joint_state_topic: str = "/joint_states",
        command_kinds: dict[str, ControllerKind] | None = None,
        controller_names: Sequence[str] = (),
        trigger_services: Sequence[str] = (),
        empty_topics: Sequence[str] = (),
        controller_manager: str = _DEFAULT_CONTROLLER_MANAGER,
    ) -> None:
        """Create one publisher per command topic and subscribe to the state topic."""
        # `openral_hal` must import without a ROS 2 install (unit tests, docs
        # builds, CI lanes with no rclpy), so every ROS symbol below is imported
        # at call time rather than at module scope — hence the PLC0415 waivers.
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        if not command_topics:
            raise ROSConfigError(
                "RosControlTransport needs at least one command topic; the HAL "
                "reported none, so nothing it publishes could reach a controller."
            )
        if not joint_names:
            raise ROSConfigError(
                "RosControlTransport needs the ros2_control joint names; the HAL "
                "reported none, so incoming /joint_states could not be matched."
            )

        kinds = dict(command_kinds or {})
        unknown = sorted(set(kinds) - set(command_topics))
        if unknown:
            raise ROSConfigError(
                f"RosControlTransport was given controller kinds for topic(s) {unknown} "
                f"that are not in command_topics {sorted(command_topics)}. A kind that "
                "names no topic silently governs nothing, which is how a controller ends "
                "up commanded with the wrong message type."
            )
        self._kinds = {t: kinds.get(t, ControllerKind.JOINT_TRAJECTORY) for t in command_topics}

        from rclpy.qos import (  # noqa: PLC0415  # reason: rclpy is a ROS-only dep; the HAL package must import without it
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import JointState as RosJointState  # noqa: PLC0415

        self._node = node
        self._joint_names = list(joint_names)
        self._joint_state_topic = joint_state_topic
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_arrival: float = 0.0

        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_COMMAND_DEPTH,
        )
        # One publisher per topic, typed by the kind the HAL declared. Getting
        # this wrong is invisible at runtime — DDS drops a mismatched type
        # without delivering it and without erroring — so the type is chosen
        # here, once, from the declaration rather than assumed at publish time.
        self._pubs = {
            topic: node.create_publisher(_message_type(self._kinds[topic]), topic, command_qos)
            for topic in command_topics
        }

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_STATE_DEPTH,
        )
        self._sub = node.create_subscription(
            RosJointState, joint_state_topic, self._on_joint_state, state_qos
        )

        # ── Stop seam ────────────────────────────────────────────────────────
        # Safety-class QoS (CLAUDE.md §2): RELIABLE, VOLATILE, KEEP_LAST=10.
        stop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        from std_msgs.msg import Empty  # noqa: PLC0415  # reason: ROS-only dep

        self._controller_names = tuple(controller_names)
        self._controller_manager = controller_manager.rstrip("/")
        self._empty_pubs = {
            topic: node.create_publisher(Empty, topic, stop_qos) for topic in empty_topics
        }
        self._empty_type: Any = Empty
        # Helper node + executor for service calls made from inside a callback
        # of `node` (see module docstring). Same context, so the same graph.
        import rclpy  # noqa: PLC0415  # reason: ROS-only dep
        from controller_manager_msgs.srv import (  # noqa: PLC0415  # reason: ROS-only dep
            ListControllers,
            SwitchController,
        )
        from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
        from rclpy.node import Node as _Node  # noqa: PLC0415
        from std_srvs.srv import Trigger  # noqa: PLC0415  # reason: ROS-only dep

        self._rclpy = rclpy
        self._switch_type: Any = SwitchController
        self._list_type: Any = ListControllers
        self._helper = _Node(f"{node.get_name()}_controller_stop", context=node.context)
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._helper)
        self._switch_client = self._helper.create_client(
            SwitchController, f"{self._controller_manager}/switch_controller"
        )
        self._list_client = self._helper.create_client(
            ListControllers, f"{self._controller_manager}/list_controllers"
        )
        self._trigger_clients: dict[str, Any] = {
            service: self._helper.create_client(Trigger, service) for service in trigger_services
        }
        log.info(
            "hal.transport.ready",
            command_topics=list(self._pubs),
            joint_state_topic=joint_state_topic,
            joints=len(self._joint_names),
            stop_controllers=list(self._controller_names),
            vendor_stop_services=list(self._trigger_clients),
            vendor_stop_topics=list(self._empty_pubs),
        )

    # ── Injected callables ────────────────────────────────────────────────────

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Send one HAL command dict to its controller in that controller's format.

        The message type follows the `ControllerKind` the HAL declared for this
        topic in `command_bindings` — `trajectory_msgs/JointTrajectory` for a
        `JointTrajectoryController`, `std_msgs/Float64MultiArray` for the
        forward-command family.

        The message's own `joint_names` are used verbatim, so this cannot
        misroute a value the HAL placed correctly; it falls back to the full
        joint list only for a HAL that omits them.

        Only the chunk's **last** step is published. A `JointTrajectory` point
        carries an absolute target plus a deadline, and the controller
        interpolates from where the robot actually is — replaying every
        intermediate step of an action chunk would fight that interpolation.

        Args:
            topic: One of `command_topics`.
            msg: The HAL's command dict.

        Raises:
            ROSConfigError: The topic is not one this transport owns, or the
                payload is a shape this transport cannot express — both are
                contract drift between HAL and transport, which must be loud.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        publisher = self._pubs.get(topic)
        if publisher is None:
            raise ROSConfigError(
                f"RosControlTransport has no publisher for {topic!r}; the HAL "
                f"publishes to topics it never declared in command_topics(). "
                f"Declared: {sorted(self._pubs)}"
            )

        # A payload this transport cannot express must be loud. Returning
        # quietly would drop a command on the actuation path with nothing above
        # debug in the logs — the exact silent no-op this transport exists to
        # end. A HAL whose controllers take something else (a scalar gripper
        # position, say) needs that message type implemented here, not skipped
        # at runtime.
        if "joint_targets" not in msg:
            raise ROSConfigError(
                f"RosControlTransport cannot express the command {sorted(msg)} sent to "
                f"{topic!r}: every ControllerKind it publishes is commanded from "
                f"'joint_targets', and this one ({self._kinds[topic].value}) is no "
                "exception. Add the kind that controller speaks to ControllerKind rather "
                "than letting the command be dropped."
            )
        targets = msg.get("joint_targets")
        if not isinstance(targets, list) or not targets:
            # Distinct from the above: the HAL had nothing to send this tick.
            log.debug("hal.transport.empty_command", topic=topic)
            return
        raw_names = msg.get("joint_names")
        names = (
            [str(n) for n in raw_names]
            if isinstance(raw_names, list) and raw_names
            else list(self._joint_names)
        )
        last_step = targets[-1]
        if not isinstance(last_step, list) or len(last_step) != len(names):
            raise ROSConfigError(
                f"RosControlTransport: {topic!r} names {len(names)} joint(s) but the "
                f"chunk's final step has "
                f"{len(last_step) if isinstance(last_step, list) else 'a non-list'}."
            )

        kind = self._kinds[topic]
        if kind is ControllerKind.FORWARD_COMMAND:
            from std_msgs.msg import Float64MultiArray  # noqa: PLC0415  # reason: ROS-only dep

            # A forward controller takes a bare vector in the order its own
            # `joints` parameter declares — there is no room in the message for
            # names, so the HAL's ordering is the whole contract. The length
            # check above is therefore the only guard available.
            command = Float64MultiArray()
            command.data = [float(v) for v in last_step]
            publisher.publish(command)
            return

        from trajectory_msgs.msg import (  # noqa: PLC0415  # reason: ROS-only dep
            JointTrajectory,
            JointTrajectoryPoint,
        )

        traj = JointTrajectory()
        traj.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in last_step]
        seconds = self._time_from_start_s(msg)
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = int((seconds - int(seconds)) * 1e9)
        traj.points.append(point)
        publisher.publish(traj)

    def state(self) -> dict[str, object]:
        """Return the newest joint state, projected onto the HAL's joint order.

        Includes `name` so a HAL that reorders by name (OpenArm) sees what
        actually arrived rather than an assumed layout.
        """
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self._joint_names:
            p, v, e = self._latest.get(name, (0.0, 0.0, 0.0))
            positions.append(p)
            velocities.append(v)
            efforts.append(e)
        return {
            "name": list(self._joint_names),
            "position": positions,
            "velocity": velocities,
            "effort": efforts,
        }

    def last_arrival(self) -> float:
        """`time.monotonic()` of the newest joint state; 0.0 if none has arrived."""
        return self._last_arrival

    # ── ControllerStopSeam ────────────────────────────────────────────────────

    def deactivate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Deactivate `names` through `switch_controller` (STRICT), then confirm `inactive`.

        A `JointTrajectoryController` that is deactivated holds its last
        position command (zeroes velocity / effort ones), stops accepting
        trajectories on its topic and writes nothing until re-activated —
        verified against `ros2_controllers` jazzy — so this is the stop. The
        report's `ok` is the manager's `ok` **and** the post-switch listing.
        """
        return self._switch(names, activate=False, timeout_s=timeout_s)

    def activate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Activate `names` through `switch_controller` (STRICT), then confirm `active`."""
        return self._switch(names, activate=True, timeout_s=timeout_s)

    def call_trigger(self, service: str, *, timeout_s: float) -> TriggerReport:
        """Call one declared `std_srvs/Trigger` service and return its response.

        Raises:
            ROSConfigError: If `service` was not declared at wire-up.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        client = self._trigger_clients.get(service)
        if client is None:
            raise ROSConfigError(
                f"RosControlTransport has no Trigger client for {service!r}; the HAL calls a "
                f"service it never declared in vendor_stop_services(). Declared: "
                f"{sorted(self._trigger_clients)}"
            )
        if not client.wait_for_service(timeout_sec=timeout_s):
            return TriggerReport(success=False, message=f"{service} not available")
        future = client.call_async(client.srv_type.Request())
        self._rclpy.spin_until_future_complete(
            self._helper, future, executor=self._executor, timeout_sec=timeout_s
        )
        if not future.done():
            return TriggerReport(success=False, message=f"{service} did not answer")
        exc = future.exception()
        if exc is not None:
            return TriggerReport(success=False, message=f"{service}: {exc}")
        response = future.result()
        return TriggerReport(success=bool(response.success), message=str(response.message))

    def publish_empty(self, topic: str) -> None:
        """Publish one `std_msgs/Empty` on a declared vendor-stop topic.

        Raises:
            ROSConfigError: If `topic` was not declared at wire-up.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        publisher = self._empty_pubs.get(topic)
        if publisher is None:
            raise ROSConfigError(
                f"RosControlTransport has no Empty publisher for {topic!r}; the HAL publishes "
                f"on a topic it never declared in vendor_stop_topics(). Declared: "
                f"{sorted(self._empty_pubs)}"
            )
        publisher.publish(self._empty_type())

    def controller_states(self, *, timeout_s: float) -> dict[str, str]:
        """Return `{name: state}` from `list_controllers`; empty if the manager did not answer."""
        if not self._list_client.wait_for_service(timeout_sec=timeout_s):
            return {}
        future = self._list_client.call_async(self._list_type.Request())
        self._rclpy.spin_until_future_complete(
            self._helper, future, executor=self._executor, timeout_sec=timeout_s
        )
        if not future.done() or future.exception() is not None:
            return {}
        return {str(c.name): str(c.state) for c in future.result().controller}

    def close(self) -> None:
        """Destroy the helper node (lifecycle cleanup)."""
        self._executor.remove_node(self._helper)
        self._helper.destroy_node()

    def _switch(
        self, names: Sequence[str], *, activate: bool, timeout_s: float
    ) -> ControllerSwitchReport:
        wanted = tuple(names)
        target = "active" if activate else "inactive"
        if not wanted:
            return ControllerSwitchReport(
                ok=False, controllers=wanted, detail="no controller names to switch"
            )
        deadline = time.monotonic() + timeout_s
        manager = f"{self._controller_manager}/switch_controller"
        if not self._switch_client.wait_for_service(timeout_sec=timeout_s):
            return ControllerSwitchReport(
                ok=False, controllers=wanted, detail=f"{manager} not available"
            )
        request = self._switch_type.Request()
        if activate:
            request.activate_controllers = list(wanted)
        else:
            request.deactivate_controllers = list(wanted)
        request.strictness = _STRICT
        request.activate_asap = True
        request.timeout.sec = int(timeout_s)
        request.timeout.nanosec = int((timeout_s - int(timeout_s)) * 1e9)
        future = self._switch_client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._helper, future, executor=self._executor, timeout_sec=timeout_s
        )
        if not future.done():
            return ControllerSwitchReport(
                ok=False, controllers=wanted, detail=f"{manager} did not answer"
            )
        exc = future.exception()
        if exc is not None:
            return ControllerSwitchReport(ok=False, controllers=wanted, detail=f"{manager}: {exc}")
        response = future.result()
        remaining = max(0.5, deadline - time.monotonic())
        states = self._controller_states_of(wanted, timeout_s=remaining)
        confirmed = all(states.get(n) == target for n in wanted)
        # `message` joined the response after Humble; read it defensively.
        message = str(getattr(response, "message", "") or "")
        detail = (
            f"switch_controller ok={bool(response.ok)} {message}".strip() + f"; listed {states}"
        )
        return ControllerSwitchReport(
            ok=bool(response.ok) and confirmed, controllers=wanted, states=states, detail=detail
        )

    def _controller_states_of(self, names: Sequence[str], *, timeout_s: float) -> dict[str, str]:
        listed = self.controller_states(timeout_s=timeout_s)
        return {n: listed.get(n, "not_listed") for n in names}

    # ── Introspection ─────────────────────────────────────────────────────────

    def seen_joints(self) -> set[str]:
        """Joint names that have actually appeared on the state topic."""
        return set(self._latest)

    def missing_joints(self) -> list[str]:
        """Expected joints the state topic has never reported."""
        return [n for n in self._joint_names if n not in self._latest]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _time_from_start_s(self, msg: dict[str, object]) -> float:
        """Deadline for the published point.

        Taken from the action's own `chunk_dt * horizon` when the HAL supplies
        it, because the controller's speed is `(target - measured) / deadline`
        — a deadline shorter than the chunk it represents asks the arm to cover
        the whole chunk in one step.
        """
        raw = msg.get("time_from_start_s")
        if isinstance(raw, int | float) and float(raw) > 0.0:
            return float(raw)
        return _DEFAULT_TIME_FROM_START_S

    def _on_joint_state(self, msg: Any) -> None:  # noqa: ANN401  # reason: duck-typed sensor_msgs/JointState keeps this module ROS-free at import
        """Merge one `sensor_msgs/JointState` into the cache, by name."""
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))
        for i, name in enumerate(names):
            self._latest[str(name)] = (
                positions[i] if i < len(positions) else 0.0,
                velocities[i] if i < len(velocities) else 0.0,
                efforts[i] if i < len(efforts) else 0.0,
            )
        self._last_arrival = time.monotonic()


#: Default trajectory deadline. Matches the 100 ms the production HALs assume
#: for a single-step command; overridden per message via `time_from_start_s`.
_DEFAULT_TIME_FROM_START_S = 0.1


def _message_type(kind: ControllerKind) -> type:
    """Return the ROS message class one `ControllerKind` is commanded with.

    Imported at call time so this module still imports without a ROS 2
    installation (unit tests, docs builds, CI lanes with no rclpy). Each class is
    bound to an explicitly annotated local on the way out: with no ROS 2 install
    to import from, mypy sees these as `Any`, and returning `Any` from a
    `-> type` function is exactly what `--strict` rejects.

    Raises:
        ROSConfigError: For a kind with no message mapping. Unreachable while
            the enum and this function agree; it exists so that adding a member
            without teaching this function fails at wire-up rather than
            publishing a plausible-but-wrong type onto the actuation path.
    """
    from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

    message_type: type
    if kind is ControllerKind.JOINT_TRAJECTORY:
        from trajectory_msgs.msg import JointTrajectory  # noqa: PLC0415  # reason: ROS-only dep

        message_type = JointTrajectory
        return message_type
    if kind is ControllerKind.FORWARD_COMMAND:
        from std_msgs.msg import Float64MultiArray  # noqa: PLC0415  # reason: ROS-only dep

        message_type = Float64MultiArray
        return message_type
    raise ROSConfigError(  # pragma: no cover - guarded by test_every_controller_kind_maps
        f"RosControlTransport has no message type for ControllerKind {kind!r}. "
        "Add one here in the same change that adds the enum member."
    )
