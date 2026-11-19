"""Real-hardware HAL adapter for two Flexiv Rizon 4s arms driven as one robot.

Actuation path (CLAUDE.md §1.5)::

    Skill → Action → RizonDualRealHAL → two JointTrajectory topics
          → controller_manager (1 kHz, C++)
          → 2 x flexiv_hardware/FlexivHardwareInterface (upstream flexiv_ros2)
          → flexiv_rdk (NRT joint position) → Ethernet → each arm's controller

Everything below this adapter is upstream Flexiv code, started by
``openral_hal_rizon``'s ``real_bringup.launch.py``. One
``FlexivHardwareInterface`` per arm, not upstream's ``FlexivDualHardwareInterface``:
the dual interface needs ``flexiv_drdk``, every tag of which requires an RDK newer
than the 1.9.0 the arms' v3.11 software accepts.

The action is 14-D: ``target[0:7]`` the left arm, ``target[7:14]`` the right. The
jaws on the reference rig (Damiao servos on CAN) have no upstream ros2_control
driver, so they are not part of this embodiment yet.

Example:
    >>> from openral_core.schemas import RobotDescription
    >>> from openral_hal.flexiv_rizon_real import RizonDualRealHAL
    >>> desc = RobotDescription.from_yaml("robots/rizon_dual/robot.yaml")  # doctest: +SKIP
    >>> RizonDualRealHAL(desc).controller_names()  # doctest: +SKIP
    ['left_arm_controller', 'right_arm_controller']
"""

from __future__ import annotations

from collections.abc import Callable

from openral_core.schemas import RobotDescription

from openral_hal.ros_control import GroupedRosControlHAL

__all__ = ["RizonDualRealHAL"]

# Controller names `openral_hal_rizon/config/rizon_dual_controllers.yaml` declares.
_LEFT_ARM_CONTROLLER = "left_arm_controller"
_RIGHT_ARM_CONTROLLER = "right_arm_controller"


class RizonDualRealHAL(GroupedRosControlHAL):
    """ros2_control HAL for the dual Rizon 4s: one trajectory controller per arm.

    ``estop`` deactivates both controllers (the base stop). With no controller
    writing, ``FlexivHardwareInterface::write`` sends the RDK nothing and each arm
    holds its last NRT target. Recovery stays ``RESTART_REQUIRED`` (the base
    default) until an in-place re-arm has been exercised on the rig.

    Args:
        description: The ``robots/rizon_dual`` manifest (threaded by ``build_hal``).
        left_arm_controller / right_arm_controller: Controller names.
        joint_state_topic / publish_fn / state_fn / staleness_limit_s: As
            ``GroupedRosControlHAL``.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        left_arm_controller: str = _LEFT_ARM_CONTROLLER,
        right_arm_controller: str = _RIGHT_ARM_CONTROLLER,
        joint_state_topic: str = "/joint_states",
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float | None = None,
    ) -> None:
        """Initialise the two-controller fan-out; opens no connection."""
        super().__init__(
            description,
            groups=[(left_arm_controller, 7), (right_arm_controller, 7)],
            joint_state_topic=joint_state_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
