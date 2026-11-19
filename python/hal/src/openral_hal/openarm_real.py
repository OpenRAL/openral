"""Real-hardware HAL adapter for the Enactic OpenArm v2 bimanual arm.

Same 16-DoF ``Action`` layout and ``HAL`` Protocol as the
``openral_hal.openarm`` MuJoCo twin, so a Skill or Reasoner moves from
twin to hardware unchanged.

Actuation path (CLAUDE.md §1.5: motors only through a typed bridge to
ros2_control with a watchdog; anything >100 Hz is C++). The OpenArm's
Damiao BLDC servos run one CAN FD bus per arm (1 Mbit/s arbitration,
5 Mbit/s data) at 750 Hz:

    Skill → Action → OpenArmRealHAL → ros2_control command topics
          → controller_manager (750 Hz, C++)
          → openarm_hardware SystemInterface (C++)
          → openarm_can → SocketCAN → Damiao motors

This adapter owns only the top hop; everything below is C++ on a
real-time budget.

Controller topology: ``openarm_bringup``'s bimanual config spawns four
controllers — each side's 7 joints and its gripper commanded separately:

===========================  ============================================
controller                   joints
===========================  ============================================
``left_joint_trajectory``    ``openarm_left_joint1`` … ``joint7``
``left_gripper``             ``openarm_left_finger_joint1``
``right_joint_trajectory``   ``openarm_right_joint1`` … ``joint7``
``right_gripper``            ``openarm_right_finger_joint1``
===========================  ============================================

``send_action`` fans one 16-DoF action out to all four topics
atomically — either all four messages publish, or none do.

Joint naming: ``RobotDescription`` uses the logical names a Skill sees
(``left_joint1``, ``left_gripper``); ros2_control uses the URDF names
(``openarm_left_joint1``, ``openarm_left_finger_joint1``). Each
``JointSpec.sim_joint_name`` already carries that URDF name — read it
rather than hardcoding a second copy of the mapping.

Action layout (identical to ``OpenArmMujocoHAL``):
* ``target[0:7]``   — left arm joints (rad)
* ``target[7]``     — left gripper (rad)
* ``target[8:15]``  — right arm joints (rad)
* ``target[15]``    — right gripper (rad)

Bus preflight: ``connect`` refuses unless both CAN interfaces exist and
are up. A HAL reporting "connected" while its motor bus is down is the
worst failure mode — every ``send_action`` succeeds, the arm never
moves, and nothing upstream can tell. The check is necessary but not
sufficient: an up interface with unpowered motors sits in
``ERROR-PASSIVE``, which ``health`` surfaces but cannot be detected
without transmitting.

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
from openral_core.can import preflight_can_links
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import RobotDescription

from openral_hal._real_description import make_real_description
from openral_hal.openarm import OPENARM_DESCRIPTION
from openral_hal.protocol import EStopRecovery, HALHealthReport
from openral_hal.ros_control import GroupedRosControlHAL

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

# `/sys/class/net` — a module constant so tests can point the preflight at a
# real fixture tree. ARPHRD_CAN (280) is the `type` value marking a CAN link;
# the same constant drives `openral_detect.probes.can`, which is the richer
# read (bitrates, controller state) used by `openral detect`. This one stays
# deliberately minimal: the HAL needs "is this link an up CAN bus", nothing more.


# ── Real-HW RobotDescription ──────────────────────────────────────────────────
# `sdk_kind: open` — both `openarm_can` and `openarm_ros2` are Apache-2.0, so
# unlike the UR / Franka adapters there is no closed vendor runtime in the path.
OPENARM_REAL_DESCRIPTION = make_real_description(OPENARM_DESCRIPTION, sdk_kind="open")


class OpenArmRealHAL(GroupedRosControlHAL):
    """ros2_control-backed HAL adapter for the physical OpenArm v2.

    Args:
        description: Manifest to publish.  Defaults to
            ``OPENARM_REAL_DESCRIPTION``; pass a rig-specific manifest
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
            ``None`` (default) reads the manifest's
            ``safety.joint_state_staleness_limit_s``.
        require_can_links: When ``True`` (the default), ``connect``
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
    #: four deactivated controllers hold position and the CAN bus stays
    #: configured, so ``reset_estop`` re-activates them through the same
    #: ``controller_manager`` seam the stop used and reconnects only once
    #: every one is confirmed ``active``.
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
        staleness_limit_s: float | None = None,
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
        # Logical (Skill-facing) names differ from ros2_control's URDF names here
        # (`left_joint1` vs `openarm_left_joint1`); the manifest carries the URDF
        # name in `sim_joint_name`, so there is exactly one copy of the mapping.
        missing = [j.name for j in desc.joints if not j.sim_joint_name]
        if missing:
            raise ROSConfigError(
                f"OpenArmRealHAL cannot map {missing} onto ros2_control: these "
                f"joints of '{desc.name}' have no sim_joint_name. The OpenArm "
                "manifest carries the URDF joint name in that field."
            )
        super().__init__(
            desc,
            groups=[
                (left_arm_controller, 7),
                (left_gripper_controller, 1),
                (right_arm_controller, 7),
                (right_gripper_controller, 1),
            ],
            joint_state_topic=joint_state_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
        self._left_can_interface = left_can_interface
        self._right_can_interface = right_can_interface
        self._require_can_links = require_can_links
        self._can_health: dict[str, str] = {}

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
        self._can_health = preflight_can_links(
            {"left": self._left_can_interface, "right": self._right_can_interface},
            hal_label="OpenArmRealHAL",
            remedy=(
                "Bring the arm buses up before connecting — the OpenArm "
                "provisioning names them via udev and configures CAN FD at "
                "1 Mbit/s nominal / 5 Mbit/s data. Run `openral detect` for "
                "the full bus report."
            ),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def health(self) -> HALHealthReport:
        """Return the cached bus health captured at ``connect()``.

        Performs no I/O — the lifecycle node calls this from its low-rate
        diagnostics path, and re-reading sysfs on every heartbeat would be
        both wasteful and, on a busy host, occasionally slow.

        Returns:
            A ``HALHealthReport`` naming each arm's
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
