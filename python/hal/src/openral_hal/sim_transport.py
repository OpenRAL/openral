"""SimTransport -- in-memory simulated ros2_control transport.

Replaces ``MagicMock`` in unit tests for ``RosControlHAL`` with a typed,
stateful simulation that applies published joint trajectory commands to
internal state.  This creates a closed-loop test environment: calling
``send_action`` publishes via ``publish()``, which updates positions
that ``state()`` then exposes to ``read_state()``.

It also implements ``ControllerStopSeam`` over a simulated
``controller_manager`` table, mirroring the two facts about a real
``JointTrajectoryController`` that the lifecycle e-stop relies on (verified
against ``ros2_controllers`` jazzy): a deactivated controller **drops** every
trajectory it is sent (its subscriber goes inactive), and it writes nothing
until re-activated. So ``deactivate_controllers`` flips a controller to
``inactive`` and every later ``publish`` on that controller's topic is recorded
in ``dropped_calls`` instead of moving the simulated joints — which is what
lets the unit lane prove "an e-stopped HAL cannot move the robot" on the same
HAL code path production runs.

Example:
    >>> from openral_hal.sim_transport import SimTransport
    >>> transport = SimTransport(n_joints=3)
    >>> transport.publish(
    ...     "/ctrl/joint_trajectory",
    ...     {
    ...         "joint_targets": [[1.0, 2.0, 3.0]],
    ...         "control_mode": "joint_position",
    ...         "horizon": 1,
    ...         "stamp_ns": 0,
    ...     },
    ... )
    >>> transport.state()["position"]
    [1.0, 2.0, 3.0]
    >>> transport.call_count
    1
    >>> transport.deactivate_controllers(["ctrl"], timeout_s=1.0).ok
    True
    >>> transport.publish("/ctrl/joint_trajectory", {"joint_targets": [[0.0, 0.0, 0.0]]})
    >>> transport.state()["position"]  # inactive controller: command dropped, not applied
    [1.0, 2.0, 3.0]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from openral_core.exceptions import ROSConfigError

from openral_hal.ros_control import ControllerSwitchReport, TriggerReport

__all__ = ["SimTorqueSeam", "SimTransport"]


def _controller_of(topic: str) -> str:
    """The ``controller_manager`` name a ``/<controller>/<command>`` topic belongs to."""
    return topic.strip("/").split("/")[0]


class SimTransport:
    """In-memory transport simulating a ros2_control joint trajectory controller.

    Published joint trajectory commands update internal joint positions, which
    are then returned by ``state``.  All published messages are recorded
    for assertion in tests.

    Args:
        n_joints: Number of joints to simulate.  Initial positions, velocities,
            and efforts are all ``0.0``.
        controllers: The ``controller_manager`` table. When given, every
            switch is checked against it the way a ``STRICT`` switch is — a
            name not in the table is refused, exactly like an unloaded
            controller. When omitted, controllers are registered ``active``
            the first time a topic or switch names them.
        trigger_responses: Per-service ``(success, message)`` a
            ``call_trigger`` returns, to simulate e.g. a UR dashboard whose
            ``stop`` refuses. Unlisted services succeed.
        switch_faults: Per-operation (``"deactivate"`` / ``"activate"``)
            exception the switch **raises** instead of answering — the
            rclpy-shaped fault (executor or context shut down, a dead
            client handle) a real ``controller_manager`` call can die with.
            The HAL must contain it in its stop report.

    Example:
        >>> transport = SimTransport(n_joints=2)
        >>> transport.publish(
        ...     "/ctrl/traj",
        ...     {
        ...         "joint_targets": [[0.5, -0.5]],
        ...         "control_mode": "joint_position",
        ...         "horizon": 1,
        ...         "stamp_ns": 0,
        ...     },
        ... )
        >>> tuple(transport.state()["position"])
        (0.5, -0.5)
        >>> transport.call_count
        1
        >>> topic, msg = transport.last_call  # type: ignore[misc]
        >>> topic
        '/ctrl/traj'
    """

    def __init__(
        self,
        n_joints: int,
        *,
        controllers: Sequence[str] | None = None,
        trigger_responses: Mapping[str, tuple[bool, str]] | None = None,
        switch_faults: Mapping[str, Exception] | None = None,
    ) -> None:
        """Initialise zeroed joint state for *n_joints* joints."""
        self._n_joints = n_joints
        self._positions: list[float] = [0.0] * n_joints
        self._velocities: list[float] = [0.0] * n_joints
        self._efforts: list[float] = [0.0] * n_joints
        self._published: list[tuple[str, dict[str, object]]] = []
        self._dropped: list[tuple[str, dict[str, object]]] = []
        self._strict = controllers is not None
        self._controller_states: dict[str, str] = {name: "active" for name in controllers or ()}
        self._switches: list[tuple[str, tuple[str, ...]]] = []
        self._trigger_responses = dict(trigger_responses or {})
        self._switch_faults = dict(switch_faults or {})
        self._triggers: list[str] = []
        self._empties: list[str] = []

    # -- Transport callables (injected into RosControlHAL) --------------------

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Record *msg* and apply ``joint_targets`` to internal state.

        If ``msg["joint_targets"]`` is a list of trajectory steps, the **last
        step** is applied to positions -- mirroring how a real joint trajectory
        controller would reach the final waypoint. A command on a topic whose
        controller is ``inactive`` is recorded in ``dropped_calls`` and **not**
        applied — what a deactivated ``JointTrajectoryController`` does with a
        late trajectory.

        Args:
            topic: The ROS 2 topic name.
            msg: The published message dict.
        """
        if self._state_of(_controller_of(topic)) != "active":
            self._dropped.append((topic, msg))
            return
        self._published.append((topic, msg))
        targets = msg.get("joint_targets")
        if isinstance(targets, list) and targets:
            last_step = targets[-1]
            if isinstance(last_step, list):
                self._positions = [float(v) for v in last_step]

    def state(self) -> dict[str, object]:
        """Return the current simulated joint state.

        Returns:
            Dict with ``"position"``, ``"velocity"``, and ``"effort"`` keys,
            each containing a ``list[float]`` of length ``n_joints``.
        """
        return {
            "position": list(self._positions),
            "velocity": list(self._velocities),
            "effort": list(self._efforts),
        }

    # -- ControllerStopSeam ---------------------------------------------------

    def deactivate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Flip every named controller to ``inactive`` (STRICT: unknown names refuse)."""
        return self._switch("deactivate", names, "inactive")

    def activate_controllers(
        self, names: Sequence[str], *, timeout_s: float
    ) -> ControllerSwitchReport:
        """Flip every named controller to ``active`` (STRICT: unknown names refuse)."""
        return self._switch("activate", names, "active")

    def call_trigger(self, service: str, *, timeout_s: float) -> TriggerReport:
        """Record the call and answer from ``trigger_responses`` (default: success)."""
        self._triggers.append(service)
        success, message = self._trigger_responses.get(service, (True, "ok"))
        return TriggerReport(success=success, message=message)

    def publish_empty(self, topic: str) -> None:
        """Record one ``std_msgs/Empty`` publish."""
        self._empties.append(topic)

    def controller_state(self, name: str) -> str:
        """``active`` / ``inactive`` / ``unloaded`` for one controller."""
        return self._controller_states.get(name, "unloaded" if self._strict else "active")

    # -- Introspection (replaces MagicMock assertions) ------------------------

    @property
    def call_count(self) -> int:
        """Number of times ``publish`` delivered a command to an active controller."""
        return len(self._published)

    @property
    def last_call(self) -> tuple[str, dict[str, object]] | None:
        """The most recent delivered ``(topic, msg)`` pair, or ``None``."""
        return self._published[-1] if self._published else None

    @property
    def calls(self) -> list[tuple[str, dict[str, object]]]:
        """All delivered ``(topic, msg)`` pairs in chronological order."""
        return list(self._published)

    @property
    def dropped_calls(self) -> list[tuple[str, dict[str, object]]]:
        """Every command that reached an ``inactive`` controller and was dropped."""
        return list(self._dropped)

    @property
    def switch_calls(self) -> list[tuple[str, tuple[str, ...]]]:
        """Every ``("deactivate" | "activate", names)`` switch, in order."""
        return list(self._switches)

    @property
    def trigger_calls(self) -> list[str]:
        """Every ``std_srvs/Trigger`` service called, in order."""
        return list(self._triggers)

    @property
    def empty_publishes(self) -> list[str]:
        """Every ``std_msgs/Empty`` topic published on, in order."""
        return list(self._empties)

    # -- Internals ------------------------------------------------------------

    def _state_of(self, name: str) -> str:
        """Return a controller's table state, registering it ``active`` unless STRICT."""
        if name not in self._controller_states:
            if self._strict:
                return "unloaded"
            self._controller_states[name] = "active"
        return self._controller_states[name]

    def _switch(self, op: str, names: Sequence[str], target: str) -> ControllerSwitchReport:
        """Flip ``names`` to ``target``; STRICT refuses unloaded names; may raise the fault."""
        wanted = tuple(names)
        if not wanted:
            raise ROSConfigError(f"SimTransport.{op}_controllers(): no controller names given.")
        self._switches.append((op, wanted))
        if op in self._switch_faults:
            raise self._switch_faults[op]
        unknown = [n for n in wanted if self._state_of(n) == "unloaded"]
        if unknown:
            return ControllerSwitchReport(
                ok=False,
                controllers=wanted,
                states={n: self._state_of(n) for n in wanted},
                detail=f"controller(s) not loaded: {unknown}",
            )
        for n in wanted:
            self._controller_states[n] = target
        return ControllerSwitchReport(
            ok=True,
            controllers=wanted,
            states={n: target for n in wanted},
            detail=f"{op}d {list(wanted)}",
        )


class SimTorqueSeam:
    """In-memory ``InterbotixStopSeam`` — a simulated ``xs_sdk`` torque table.

    Mirrors what ``/<robot_name>/torque_enable`` does on a real Interbotix XS
    arm: flips the named group's torque, and (like a real service) refuses an
    arm that is not running. Every call is recorded for assertion.

    Args:
        arms: The ``xs_sdk`` namespaces that exist. A call on any other name
            reports ``success=False`` — an absent service, not a stop.
        failing: Arms whose ``torque_enable`` refuses, to simulate one side's
            SDK node being down at e-stop time.
        faults: Per-arm exception ``torque_enable`` **raises** instead of
            answering — the rclpy-shaped fault a real service call can die
            with. The HAL must contain it and still stop the other arms.

    Example:
        >>> from openral_hal.sim_transport import SimTorqueSeam
        >>> seam = SimTorqueSeam(arms=["follower_left", "follower_right"])
        >>> seam.torque_enable("follower_left", group="all", enable=False, timeout_s=1.0).success
        True
        >>> seam.torque("follower_left")
        False
        >>> seam.torque_enable("puppet", group="all", enable=False, timeout_s=1.0).success
        False
    """

    def __init__(
        self,
        *,
        arms: Sequence[str],
        failing: Sequence[str] = (),
        faults: Mapping[str, Exception] | None = None,
    ) -> None:
        """Every listed arm starts torqued on."""
        self._torque: dict[str, bool] = {arm: True for arm in arms}
        self._failing = set(failing)
        self._faults = dict(faults or {})
        self._calls: list[tuple[str, str, bool]] = []

    def torque_enable(
        self, robot_name: str, *, group: str, enable: bool, timeout_s: float
    ) -> TriggerReport:
        """Record the call and flip the arm's torque unless it is absent or failing."""
        self._calls.append((robot_name, group, enable))
        if robot_name in self._faults:
            raise self._faults[robot_name]
        if robot_name not in self._torque:
            return TriggerReport(
                success=False, message=f"/{robot_name}/torque_enable not available"
            )
        if robot_name in self._failing:
            return TriggerReport(success=False, message=f"/{robot_name}/torque_enable refused")
        self._torque[robot_name] = enable
        return TriggerReport(success=True, message=f"/{robot_name}/torque_enable {group}={enable}")

    def torque(self, robot_name: str) -> bool:
        """Whether *robot_name* is currently torqued on."""
        return self._torque[robot_name]

    @property
    def calls(self) -> list[tuple[str, str, bool]]:
        """Every ``(robot_name, group, enable)`` call, in order."""
        return list(self._calls)
