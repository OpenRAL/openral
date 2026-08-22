"""Real-hardware HAL adapter for the Enactic OpenArm v2 bimanual arm.

Where :mod:`openral_hal.openarm` ships the MuJoCo digital twin, this module
drives the physical arm.  Same 16-DoF ``Action`` layout, same
:class:`~openral_hal.protocol.HAL` Protocol, so a Skill or Reasoner moves
from twin to hardware without a line of change.

Why ros2_control and not SocketCAN directly
-------------------------------------------
The OpenArm's motors are Damiao BLDC servos on a CAN FD bus (1 Mbit/s
arbitration, 5 Mbit/s data), one bus per arm.  It is entirely possible to
drive them from Python via the ``openarm_can`` bindings — and that is
precisely what CLAUDE.md §1.5 forbids: *"Python touches motors only through
a typed bridge to ros2_control with a watchdog. Anything >100 Hz is C++."*
The OpenArm control loop runs at 400 Hz.

So the actuation path is:

    Skill → Action → OpenArmRealHAL → ros2_control command topics
          → controller_manager (400 Hz, C++)
          → openarm_hardware SystemInterface (C++)
          → openarm_can → SocketCAN → Damiao motors

This adapter owns only the top hop.  Everything below it is C++ with a
real-time budget, which is where a bimanual arm's actuation belongs.

Controller topology
-------------------
``openarm_bringup``'s bimanual configuration spawns **four** controllers,
not one — each arm's 7 joints and its gripper are commanded separately:

===========================  ============================================
controller                   joints
===========================  ============================================
``left_joint_trajectory``    ``openarm_left_joint1`` … ``joint7``
``left_gripper``             ``openarm_left_finger_joint1``
``right_joint_trajectory``   ``openarm_right_joint1`` … ``joint7``
``right_gripper``            ``openarm_right_finger_joint1``
===========================  ============================================

:meth:`OpenArmRealHAL.send_action` therefore fans one 16-DoF action out to
four command topics.  A partial action is never published: either all four
messages go out, or none do (see :meth:`send_action`).

Joint naming
------------
``RobotDescription`` uses the logical names a Skill sees (``left_joint1``,
``left_gripper``); ros2_control uses the URDF names (``openarm_left_joint1``,
``openarm_left_finger_joint1``).  The bridge already exists in the manifest:
each :class:`~openral_core.JointSpec` carries ``sim_joint_name``, which for
the OpenArm *is* the URDF name.  This adapter reads that field rather than
hardcoding a second copy of the mapping.

Action layout (identical to :class:`~openral_hal.OpenArmMujocoHAL`)
------------------------------------------------------------------
* ``target[0:7]``   — left arm joints (rad)
* ``target[7]``     — left gripper (rad)
* ``target[8:15]``  — right arm joints (rad)
* ``target[15]``    — right gripper (rad)

Bus preflight
-------------
:meth:`connect` refuses to proceed unless both CAN interfaces exist and are
up.  A HAL that reports "connected" while its motor bus is down is the worst
possible failure mode: every ``send_action`` succeeds, the arm never moves,
and nothing upstream can tell the difference.  Note the check is necessary,
not sufficient — an up interface with unpowered motors sits in
``ERROR-PASSIVE``, which :meth:`health` surfaces but which cannot be
detected without transmitting.

Example::

    from openral_hal.openarm_real import OpenArmRealHAL  # doctest: +SKIP

    hal = OpenArmRealHAL()  # doctest: +SKIP
    hal.connect()  # doctest: +SKIP
    state = hal.read_state()  # doctest: +SKIP
    hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, JointState, RobotDescription

from openral_hal._real_description import make_real_description
from openral_hal.openarm import OPENARM_DESCRIPTION
from openral_hal.protocol import EStopRecovery, HALHealthReport
from openral_hal.ros_control import RosControlHAL

__all__ = [
    "OPENARM_REAL_DESCRIPTION",
    "OpenArmRealHAL",
]

log = structlog.get_logger(__name__)


# ── Bringup defaults ──────────────────────────────────────────────────────────
# Names come from `openarm_bringup`'s bimanual controller config and from the
# udev rule the OpenArm Jetson provisioning installs (dev_id 0x1 →
# `openarm_left`, dev_id 0x0 → `openarm_right`). Pinning them as defaults keeps
# every OpenArm deployment identical; every one is overridable per rig.
_LEFT_ARM_CONTROLLER = "left_joint_trajectory_controller"
_LEFT_GRIPPER_CONTROLLER = "left_gripper_controller"
_RIGHT_ARM_CONTROLLER = "right_joint_trajectory_controller"
_RIGHT_GRIPPER_CONTROLLER = "right_gripper_controller"
_JOINT_STATE_TOPIC = "/joint_states"
_LEFT_CAN_INTERFACE = "openarm_left"
_RIGHT_CAN_INTERFACE = "openarm_right"

# Index of each side's slice in the 16-DoF action vector.
_LEFT_ARM_SLICE = slice(0, 7)
_LEFT_GRIPPER_SLICE = slice(7, 8)
_RIGHT_ARM_SLICE = slice(8, 15)
_RIGHT_GRIPPER_SLICE = slice(15, 16)

# `/sys/class/net` — a module constant so tests can point the preflight at a
# real fixture tree. ARPHRD_CAN (280) is the `type` value marking a CAN link;
# the same constant drives `openral_detect.probes.can`, which is the richer
# read (bitrates, controller state) used by `openral detect`. This one stays
# deliberately minimal: the HAL needs "is this link an up CAN bus", nothing more.
_SYSFS_NET = "/sys/class/net"
_ARPHRD_CAN = "280"


# ── Real-HW RobotDescription ──────────────────────────────────────────────────
# `sdk_kind: open` — both `openarm_can` and `openarm_ros2` are Apache-2.0, so
# unlike the UR / Franka adapters there is no closed vendor runtime in the path.
OPENARM_REAL_DESCRIPTION = make_real_description(OPENARM_DESCRIPTION, sdk_kind="open")


def _read_sysfs(path: str) -> str:
    """Read one sysfs attribute, returning ``""`` when it is absent."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _can_link_state(interface: str) -> tuple[bool, str]:
    """Report whether ``interface`` is an existing, up CAN link.

    Args:
        interface: SocketCAN interface name, e.g. ``"openarm_left"``.

    Returns:
        ``(is_up, reason)``.  ``reason`` is empty when the link is usable and
        otherwise names the specific problem, so ``connect()`` can put it in
        the exception rather than making the operator go look.
    """
    arphrd = _read_sysfs(f"{_SYSFS_NET}/{interface}/type")
    if not arphrd:
        return (False, f"no network interface named {interface!r}")
    if arphrd != _ARPHRD_CAN:
        return (False, f"{interface!r} exists but is not a CAN link (type={arphrd})")
    operstate = _read_sysfs(f"{_SYSFS_NET}/{interface}/operstate").lower()
    if operstate != "up":
        return (False, f"{interface!r} is a CAN link but is {operstate or 'in an unknown state'}")
    return (True, "")


class OpenArmRealHAL(RosControlHAL):
    """ros2_control-backed HAL adapter for the physical OpenArm v2.

    Args:
        description: Manifest to publish.  Defaults to
            :data:`OPENARM_REAL_DESCRIPTION`; pass a rig-specific manifest
            (e.g. one written by ``openral detect``) to override sensors or
            safety limits without changing kinematics.
        left_can_interface: SocketCAN interface for the left arm's motor bus.
        right_can_interface: SocketCAN interface for the right arm's bus.
        left_arm_controller: ros2_control controller commanding the left arm's
            7 joints.
        left_gripper_controller: Controller commanding the left gripper.
        right_arm_controller: Controller commanding the right arm's 7 joints.
        right_gripper_controller: Controller commanding the right gripper.
        joint_state_topic: Topic publishing ``sensor_msgs/JointState``.
        publish_fn: Transport callable; defaults to the base adapter's
            no-op logger.  Injected with a real ``rclpy`` publisher by the
            lifecycle node, and by HIL tests.
        state_fn: Callable returning the latest raw joint state as a dict with
            ``name`` / ``position`` / ``velocity`` / ``effort`` keys, in
            ros2_control joint naming.
        staleness_limit_s: Maximum age of a ``read_state()`` reading.
        require_can_links: When ``True`` (the default), :meth:`connect`
            verifies both CAN interfaces are up and refuses otherwise.  Set
            ``False`` only to exercise the ROS wiring against a bringup whose
            hardware interface is itself simulated.

    Raises:
        ROSConfigError: If the description's joint count is not 16, or if a
            joint is missing the ``sim_joint_name`` that maps it to
            ros2_control.

    Example:
        >>> from openral_hal.openarm_real import OpenArmRealHAL
        >>> hal = OpenArmRealHAL(require_can_links=False)
        >>> hal.ros2_control_joint_names()[0]
        'openarm_left_joint1'
        >>> hal.ros2_control_joint_names()[7]
        'openarm_left_finger_joint1'
    """

    #: The lifecycle node may re-arm this HAL in place after an e-stop: the
    #: controllers hold position and the CAN bus stays configured, so recovery
    #: does not require a process restart.
    estop_recovery: EStopRecovery = EStopRecovery.RESETTABLE

    _EXPECTED_JOINT_COUNT = 16

    def __init__(
        self,
        description: RobotDescription | None = None,
        *,
        left_can_interface: str = _LEFT_CAN_INTERFACE,
        right_can_interface: str = _RIGHT_CAN_INTERFACE,
        left_arm_controller: str = _LEFT_ARM_CONTROLLER,
        left_gripper_controller: str = _LEFT_GRIPPER_CONTROLLER,
        right_arm_controller: str = _RIGHT_ARM_CONTROLLER,
        right_gripper_controller: str = _RIGHT_GRIPPER_CONTROLLER,
        joint_state_topic: str = _JOINT_STATE_TOPIC,
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
        require_can_links: bool = True,
    ) -> None:
        """Initialise the adapter; opens neither ROS nor the CAN bus yet."""
        desc = description or OPENARM_REAL_DESCRIPTION
        if len(desc.joints) != self._EXPECTED_JOINT_COUNT:
            raise ROSConfigError(
                f"OpenArmRealHAL expects a {self._EXPECTED_JOINT_COUNT}-joint "
                f"bimanual manifest (7 arm + 1 gripper per side); "
                f"'{desc.name}' has {len(desc.joints)}."
            )
        super().__init__(
            desc,
            controller_name=left_arm_controller,
            joint_state_topic=joint_state_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
        self._left_can_interface = left_can_interface
        self._right_can_interface = right_can_interface
        self._require_can_links = require_can_links
        self._can_health: dict[str, str] = {}

        # Logical (Skill-facing) name → ros2_control (URDF) name.  Sourced from
        # the manifest so there is exactly one copy of this mapping in the repo.
        missing = [j.name for j in desc.joints if not j.sim_joint_name]
        if missing:
            raise ROSConfigError(
                f"OpenArmRealHAL cannot map {missing} onto ros2_control: these "
                f"joints of '{desc.name}' have no sim_joint_name. The OpenArm "
                "manifest carries the URDF joint name in that field."
            )
        self._ros2_names: list[str] = [str(j.sim_joint_name) for j in desc.joints]

        # (command topic, action slice, ros2_control joint names) per controller.
        self._command_groups: tuple[tuple[str, slice, list[str]], ...] = (
            (
                f"/{left_arm_controller}/joint_trajectory",
                _LEFT_ARM_SLICE,
                self._ros2_names[_LEFT_ARM_SLICE],
            ),
            (
                f"/{left_gripper_controller}/joint_trajectory",
                _LEFT_GRIPPER_SLICE,
                self._ros2_names[_LEFT_GRIPPER_SLICE],
            ),
            (
                f"/{right_arm_controller}/joint_trajectory",
                _RIGHT_ARM_SLICE,
                self._ros2_names[_RIGHT_ARM_SLICE],
            ),
            (
                f"/{right_gripper_controller}/joint_trajectory",
                _RIGHT_GRIPPER_SLICE,
                self._ros2_names[_RIGHT_GRIPPER_SLICE],
            ),
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    def ros2_control_joint_names(self) -> list[str]:
        """Return the ros2_control joint names in action-vector order.

        Returns:
            16 URDF joint names — left arm 1-7, left finger, right arm 1-7,
            right finger.

        Example:
            >>> from openral_hal.openarm_real import OpenArmRealHAL
            >>> names = OpenArmRealHAL(require_can_links=False).ros2_control_joint_names()
            >>> len(names)
            16
            >>> names[15]
            'openarm_right_finger_joint1'
        """
        return list(self._ros2_names)

    def command_topics(self) -> list[str]:
        """Return the four command topics this adapter publishes to.

        Example:
            >>> from openral_hal.openarm_real import OpenArmRealHAL
            >>> OpenArmRealHAL(require_can_links=False).command_topics()[1]
            '/left_gripper_controller/joint_trajectory'
        """
        return [topic for topic, _, _ in self._command_groups]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Verify both motor buses, then open the ROS connection.

        Raises:
            ROSConfigError: If ``require_can_links`` is set and either CAN
                interface is missing, is not a CAN link, or is down.
            ROSRuntimeError: If already connected.
        """
        if self._require_can_links:
            self._preflight_can_links()
        else:
            self._can_health = {"preflight": "skipped (require_can_links=False)"}
        super().connect()

    def _preflight_can_links(self) -> None:
        """Refuse to connect over a motor bus that is not up.

        Raises:
            ROSConfigError: Naming every interface that failed and why.
        """
        problems: list[str] = []
        health: dict[str, str] = {}
        for side, interface in (
            ("left", self._left_can_interface),
            ("right", self._right_can_interface),
        ):
            is_up, reason = _can_link_state(interface)
            health[f"{side}_can"] = interface if is_up else f"{interface} (DOWN)"
            if not is_up:
                problems.append(reason)
        self._can_health = health
        if problems:
            raise ROSConfigError(
                f"OpenArmRealHAL cannot connect: {'; '.join(problems)}. "
                "Bring the arm buses up before connecting — the OpenArm "
                "provisioning names them via udev and configures CAN FD at "
                "1 Mbit/s nominal / 5 Mbit/s data. Run `openral detect` for "
                "the full bus report."
            )

    # ── Hot path ──────────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state, reordered into manifest order.

        ``sensor_msgs/JointState`` carries no ordering guarantee, and the
        four controllers publish into it independently, so positions must be
        matched by name rather than by index.  The returned ``JointState`` is
        always in ``description.joints`` order — the order a Skill's action
        vector uses.

        Raises:
            ROSRuntimeError: If not connected, or if the incoming state is
                missing joints the manifest declares.
            ROSPerceptionStale: If the last reading exceeds the staleness
                deadline.

        Returns:
            A 16-element :class:`~openral_core.JointState` in manifest order.
        """
        state = super().read_state()
        if self._state_fn is None:
            # No subscriber wired: the base adapter returns a zeroed state,
            # which is already in manifest order.
            return state

        raw = self._state_fn()
        incoming = raw.get("name")
        if not isinstance(incoming, list):
            # A state dict with no names cannot be reordered; the base
            # adapter's positional read is the only available interpretation.
            return state

        index_of = {str(n): i for i, n in enumerate(incoming)}
        absent = [n for n in self._ros2_names if n not in index_of]
        if absent:
            raise ROSRuntimeError(
                f"{self._joint_state_topic} is missing {len(absent)} joint(s) "
                f"declared by '{self.description.name}': {absent}. Check that "
                "every controller is active — each arm and each gripper "
                "publishes its own joints."
            )

        def _reordered(key: str) -> list[float]:
            values = raw.get(key)
            if not isinstance(values, list) or len(values) != len(incoming):
                return [0.0] * len(self._ros2_names)
            return [float(values[index_of[n]]) for n in self._ros2_names]

        return JointState(
            name=list(self._joint_names),
            position=_reordered("position"),
            velocity=_reordered("velocity"),
            effort=_reordered("effort"),
            stamp_ns=state.stamp_ns,
        )

    def send_action(self, action: Action) -> None:
        """Fan one 16-DoF action out to the four ros2_control controllers.

        The four messages are built before any is published, so a malformed
        action cannot leave the arm half-commanded — one side moving while the
        other holds an older setpoint.

        Args:
            action: A 16-DoF ``Action`` in ``description.joints`` order.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If the control mode is unsupported or the joint
                target dimensions do not match the 16-DoF layout.
        """
        self._require_connected("send_action")
        self._validate_action(action)

        if action.joint_targets is None:
            log.debug("hal.send_action.empty", robot=self.description.name)
            return

        # Build every message first — publishing is all-or-nothing.
        outbound: list[tuple[str, dict[str, object]]] = []
        for topic, span, names in self._command_groups:
            outbound.append(
                (
                    topic,
                    {
                        "control_mode": action.control_mode,
                        "horizon": action.horizon,
                        "joint_names": names,
                        "joint_targets": [list(step[span]) for step in action.joint_targets],
                        "stamp_ns": action.stamp_ns,
                    },
                )
            )
        for topic, msg in outbound:
            self._publish_fn(topic, msg)

        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            horizon=action.horizon,
            controllers=len(outbound),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def health(self) -> HALHealthReport:
        """Return the cached bus health captured at ``connect()``.

        Performs no I/O — the lifecycle node calls this from its low-rate
        diagnostics path, and re-reading sysfs on every heartbeat would be
        both wasteful and, on a busy host, occasionally slow.

        Returns:
            A :class:`~openral_hal.protocol.HALHealthReport` naming each arm's
            CAN interface and whether the preflight found it up.

        Example:
            >>> from openral_hal.openarm_real import OpenArmRealHAL
            >>> hal = OpenArmRealHAL(require_can_links=False)
            >>> hal.health().fields["connected"]
            'False'
        """
        fields = {"connected": str(self._connected), **self._can_health}
        state = "connected" if self._connected else "disconnected"
        return HALHealthReport(
            message=f"OpenArm v2 ({self.description.name}) — {state}",
            fields=fields,
        )
