"""Real-hardware HAL adapters for the Universal Robots UR5e and UR10e.

Where :mod:`openral_hal.ur` ships MuJoCo-backed adapters for sim, this
module wraps the Universal Robots ``ur_robot_driver`` (URCap / RTDE) under
the same :class:`openral_hal.protocol.HAL` Protocol so a real arm is
reachable from upper layers without changing any Skill or Reasoner code.

The real-hardware path is :class:`~openral_hal.ros_control.RosControlHAL`
plus a ``ros2_control`` controller manager driven by
``ur_robot_driver``.  The driver:

* runs as a ROS 2 node alongside ``ros2_control`` and exposes the standard
  ``/joint_states`` topic and a ``scaled_joint_trajectory_controller``
  command channel (RFC §5.1 control QoS), and
* speaks RTDE to the URCap on the teach pendant (URCap installs the
  ``external_control`` program that streams setpoints back to the driver).

Why a wrapper class instead of using :class:`RosControlHAL` directly?

* Defaults are pinned for the UR series (the controller name and the
  joint-trajectory topic differ from a generic ``ros2_control`` deployment).
* The sub-class advertises a typed :data:`UR5e_REAL_DESCRIPTION` /
  :data:`UR10e_REAL_DESCRIPTION` constant so the eval-layer manifest
  (``robots/ur5e/robot.yaml`` / ``robots/ur10e/robot.yaml``) has a single
  ``sdk_kind`` / ``hal`` block it pins to.
* It surfaces the deadman / E-stop subscription contract the safety
  supervisor expects.

License posture (CLAUDE.md §7.4)
--------------------------------
``ur_robot_driver`` is BSD-3 (open).  We mark the YAML manifest's
``sdk_kind`` as ``closed`` to flag that the **runtime path requires a real
arm + URCap**, not because the Python adapter or the driver carry a
restrictive license.  The full license string lives in the manifest
metadata so the loader can surface it.

Example::

    from openral_hal.ur_real import UR5eRealHAL  # doctest: +SKIP

    hal = UR5eRealHAL(robot_ip="192.168.1.42")  # doctest: +SKIP
    hal.connect()  # doctest: +SKIP
    state = hal.read_state()  # doctest: +SKIP
    hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable

from openral_core.schemas import RobotDescription

from openral_hal._real_description import make_real_description
from openral_hal.ros_control import RosControlHAL
from openral_hal.ur import UR5e_DESCRIPTION, UR10e_DESCRIPTION

__all__ = [
    "UR5eRealHAL",
    "UR5e_REAL_DESCRIPTION",
    "UR10eRealHAL",
    "UR10e_REAL_DESCRIPTION",
]


# ── Driver defaults ───────────────────────────────────────────────────────────
# ``ur_robot_driver`` ships these names in its launch files and tutorials;
# pinning them as defaults keeps every UR deployment identical.
_UR_CONTROLLER_NAME = "scaled_joint_trajectory_controller"
_UR_JOINT_STATE_TOPIC = "/joint_states"
_UR_DEADMAN_TOPIC = "/io_and_status_controller/safety_mode"


# ── Real-HW RobotDescription constants ───────────────────────────────────────
# Shared with the Franka / Sawyer / ALOHA real-HW adapters via the
# ``make_real_description`` helper — see ``_real_description.py``.

UR5e_REAL_DESCRIPTION = make_real_description(
    UR5e_DESCRIPTION,
    sdk_kind="closed",
)

UR10e_REAL_DESCRIPTION = make_real_description(
    UR10e_DESCRIPTION,
    sdk_kind="closed",
)


class _URRealHAL(RosControlHAL):
    """Shared implementation for the UR5e and UR10e real-hardware adapters.

    Wraps :class:`RosControlHAL` with UR-driver-specific defaults: the
    ``scaled_joint_trajectory_controller`` controller name and the standard
    ``/joint_states`` topic that ``ur_robot_driver`` publishes to.

    The deadman subscription contract is left to the safety supervisor (the
    HAL records the topic it expects via :attr:`deadman_topic`).  Per
    CLAUDE.md §7.7 the HAL never silences a ``ROSEStopRequested`` raised by
    :meth:`estop`.

    Args:
        description: One of :data:`UR5e_REAL_DESCRIPTION` /
            :data:`UR10e_REAL_DESCRIPTION` (or any UR-shaped
            :class:`RobotDescription`).
        robot_ip: Static IP of the UR controller (the URCap ``external_control``
            program connects back to the driver here).  Recorded for
            observability / launch-file generation; the driver is expected to
            have been started with this IP already.
        controller_name: ``ros2_control`` controller name; defaults to the
            ``scaled_joint_trajectory_controller`` shipped by
            ``ur_robot_driver``.
        joint_state_topic: ROS 2 topic the driver publishes joint state on.
        publish_fn / state_fn: Inject a :class:`~openral_hal.sim_transport.SimTransport`
            (or any callable pair) at construction time to drive the adapter
            in unit / integration tests without a live ROS 2 stack.
        staleness_limit_s: Maximum age of a ``read_state()`` reading before
            ``ROSPerceptionStale`` is raised (per the HAL Protocol).
        deadman_topic: Topic the safety supervisor subscribes to in order to
            cut motor power when a deadman / E-stop is released.  Defaults to
            ``/io_and_status_controller/safety_mode`` (the topic the UR
            driver publishes ``ur_msgs/msg/SafetyMode`` on).
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        robot_ip: str | None = None,
        controller_name: str = _UR_CONTROLLER_NAME,
        joint_state_topic: str = _UR_JOINT_STATE_TOPIC,
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
        deadman_topic: str = _UR_DEADMAN_TOPIC,
    ) -> None:
        super().__init__(
            description,
            controller_name=controller_name,
            joint_state_topic=joint_state_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
        self.robot_ip = robot_ip
        self.deadman_topic = deadman_topic


class UR5eRealHAL(_URRealHAL):
    """Real-hardware HAL adapter for the Universal Robots UR5e.

    Drives a real UR5e via ``ros2_control`` + ``ur_robot_driver`` (URCap +
    RTDE).  The Python adapter itself contains no ``rclpy`` import — the
    transport is injected, so the same class drives the real arm in
    production and a :class:`~openral_hal.sim_transport.SimTransport` in
    unit tests.

    Args:
        robot_ip: Static IP of the UR5e controller, e.g. ``"192.168.1.42"``.
            Recorded for observability; the driver itself must have been
            launched against this IP separately.
        publish_fn: Optional injected publish callable for tests.
        state_fn: Optional injected state-read callable for tests.
        staleness_limit_s: Maximum age of a cached state.

    Example::

        >>> from openral_hal.sim_transport import SimTransport
        >>> from openral_hal.ur_real import UR5eRealHAL
        >>> transport = SimTransport(n_joints=6)
        >>> hal = UR5eRealHAL(
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.description.name
        'ur5e'
        >>> hal.description.sdk_kind
        'closed'
        >>> hal.description.hal.real
        'openral_hal.ur_real:UR5eRealHAL'
    """

    def __init__(
        self,
        *,
        robot_ip: str | None = None,
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the UR5e real-HW HAL; transport defaults match ``ur_robot_driver``."""
        super().__init__(
            UR5e_REAL_DESCRIPTION,
            robot_ip=robot_ip,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )


class UR10eRealHAL(_URRealHAL):
    """Real-hardware HAL adapter for the Universal Robots UR10e.

    Identical to :class:`UR5eRealHAL` except for the wrapped
    :class:`RobotDescription` (12.5 kg payload, 1.30 m reach, larger
    torques) — ``ur_robot_driver`` itself is the same binary for both arms,
    differentiated by the URDF and per-joint envelope.

    Args mirror :class:`UR5eRealHAL`.

    Example::

        >>> from openral_hal.sim_transport import SimTransport
        >>> from openral_hal.ur_real import UR10eRealHAL
        >>> transport = SimTransport(n_joints=6)
        >>> hal = UR10eRealHAL(
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.description.name
        'ur10e'
        >>> hal.description.hal.real
        'openral_hal.ur_real:UR10eRealHAL'
    """

    def __init__(
        self,
        *,
        robot_ip: str | None = None,
        publish_fn: Callable[[str, dict[str, object]], None] | None = None,
        state_fn: Callable[[], dict[str, object]] | None = None,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the UR10e real-HW HAL; transport defaults match ``ur_robot_driver``."""
        super().__init__(
            UR10e_REAL_DESCRIPTION,
            robot_ip=robot_ip,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )
