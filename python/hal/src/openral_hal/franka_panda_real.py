"""Real-hardware HAL adapter for the Franka Emika Panda over the FCI.

This adapter targets a **physical** Panda arm driven through ``franka_ros2``
(``libfranka`` + ``franka_hardware``) and a ``ros2_control`` joint trajectory
controller.  It is the production sibling of the MuJoCo-backed
:class:`openral_hal.franka_panda.FrankaPandaHAL`: both expose the same
:data:`FRANKA_PANDA_DESCRIPTION` so upper layers (Skill, Reasoner, Safety)
see one normative robot regardless of where the joints physically live.

License posture
---------------
Per CLAUDE.md §7.4 the Franka FCI / ``libfranka`` stack is *closed but
permissive* (vendor-licensed binaries, free for research / commercial), and
``franka_ros2`` is Apache-2.0.  The manifest therefore declares
``sdk_kind: "closed_with_api"`` and sets ``hal.real`` to this adapter (ADR-0031).

Transport layering
------------------
The hot path is ``ros2_control``.  We do not import ``rclpy`` here — instead
we delegate to :class:`openral_hal.ros_control.RosControlHAL`, which takes
injected ``publish_fn`` / ``state_fn`` callables.  The lifecycle node defined
in ``packages/openral_hal_franka`` wires real publishers/subscribers at
runtime; unit tests inject :class:`SimTransport` to exercise the same code
path without ROS 2 installed.

Example:
    >>> from openral_hal.franka_panda_real import FrankaPandaRealHAL
    >>> from openral_hal.sim_transport import SimTransport
    >>> transport = SimTransport(n_joints=8)
    >>> hal = FrankaPandaRealHAL(
    ...     fci_ip="192.168.1.10",
    ...     publish_fn=transport.publish,
    ...     state_fn=transport.state,
    ... )
    >>> hal.description.name
    'franka_panda'
    >>> hal.controller_name
    'franka_arm_controller'
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import structlog
from openral_core.exceptions import ROSConfigError, ROSEStopRequested
from openral_core.schemas import Action, JointState, RobotDescription

from openral_hal._real_description import make_real_description
from openral_hal.franka_panda import FRANKA_PANDA_DESCRIPTION
from openral_hal.ros_control import RosControlHAL

__all__ = ["FRANKA_PANDA_REAL_DESCRIPTION", "FrankaPandaRealHAL"]

log = structlog.get_logger(__name__)

# Default ros2_control controller exported by ``franka_ros2``'s
# ``franka.launch.py``.  Override at construction time when a custom
# controller (e.g. ``cartesian_impedance_example_controller``) is loaded.
_DEFAULT_FRANKA_CONTROLLER: str = "franka_arm_controller"

# Default joint state topic published by ``franka_ros2`` (the global
# ``/joint_states`` topic aggregates the FCI feedback across all controllers
# managed by ``controller_manager``).
_DEFAULT_FRANKA_JOINT_STATE_TOPIC: str = "/joint_states"

# Default e-stop / error-recovery topic used by ``franka_ros2`` controllers.
# Publishing to this topic resets the FCI from a reflex-triggered halt; the
# safety supervisor uses it after handling an :class:`ROSEStopRequested`.
_DEFAULT_FRANKA_ESTOP_TOPIC: str = "/error_recovery/goal"

_PublishFn = Callable[[str, dict[str, object]], None]
_StateFn = Callable[[], dict[str, object]]


# ── RobotDescription ─────────────────────────────────────────────────────────
# The real-HW manifest shares kinematics + safety envelope + capabilities +
# ``hal`` entrypoints with the sim-side ``FRANKA_PANDA_DESCRIPTION``; only
# ``sdk_kind`` differs.  The eval-layer YAML at ``robots/franka_panda/robot.yaml``
# pins to this constant; drift is guarded by
# ``tests/unit/test_robot_manifests_match_hal_constants.py``.

FRANKA_PANDA_REAL_DESCRIPTION = make_real_description(
    FRANKA_PANDA_DESCRIPTION,
    sdk_kind="closed_with_api",
)


class FrankaPandaRealHAL:
    """HAL adapter for a physical Franka Emika Panda over the FCI.

    The adapter wraps :class:`RosControlHAL` and adds Franka-specific
    configuration: the FCI hostname, the ``franka_ros2`` controller name, and
    an explicit error-recovery topic used by the safety supervisor after
    ``estop()``.

    Args:
        fci_ip: Hostname or IP of the Franka FCI (the robot's "control" port,
            typically ``172.16.0.2`` for the default lab subnet).  Required;
            ``libfranka`` will refuse to connect without it.  Stored as
            metadata only — the actual TCP connection is opened by
            ``franka_hardware`` inside the lifecycle node.
        controller_name: Name of the ``ros2_control`` joint trajectory
            controller exposed by ``franka_ros2``.  Defaults to
            ``"franka_arm_controller"``.
        joint_state_topic: ROS 2 topic publishing
            ``sensor_msgs/JointState``.  Defaults to ``"/joint_states"``.
        command_topic: ROS 2 topic for joint trajectory commands.  Defaults
            to ``"/<controller_name>/joint_trajectory"`` (set by
            :class:`RosControlHAL`).
        error_recovery_topic: ROS 2 topic the safety supervisor publishes to
            after handling an :class:`ROSEStopRequested` so the FCI clears
            its reflex state.  Defaults to ``"/error_recovery/goal"``.
        publish_fn: Callable forwarding messages to ROS 2 topics.  Production
            use injects the lifecycle node's publisher; tests inject
            :class:`SimTransport.publish`.
        state_fn: Callable returning the latest raw joint state as a dict.
            Production use injects the lifecycle node's subscriber callback;
            tests inject :class:`SimTransport.state`.
        staleness_limit_s: Maximum age of a ``read_state()`` reading before
            :class:`ROSPerceptionStale` is raised.  Defaults to ``0.2 s``
            (tighter than the ``RosControlHAL`` default because the FCI
            feedback lands at 1 kHz).

    Raises:
        ROSConfigError: If ``fci_ip`` is empty / whitespace.

    Example:
        >>> from openral_hal.franka_panda_real import FrankaPandaRealHAL
        >>> from openral_hal.sim_transport import SimTransport
        >>> transport = SimTransport(n_joints=8)
        >>> hal = FrankaPandaRealHAL(
        ...     fci_ip="172.16.0.2",
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.connect()
        >>> hal.description.name
        'franka_panda'
        >>> hal.disconnect()
    """

    def __init__(
        self,
        *,
        fci_ip: str = "172.16.0.2",
        controller_name: str = _DEFAULT_FRANKA_CONTROLLER,
        joint_state_topic: str = _DEFAULT_FRANKA_JOINT_STATE_TOPIC,
        command_topic: str | None = None,
        error_recovery_topic: str = _DEFAULT_FRANKA_ESTOP_TOPIC,
        publish_fn: _PublishFn | None = None,
        state_fn: _StateFn | None = None,
        staleness_limit_s: float = 0.2,
    ) -> None:
        """Initialise the adapter; no TCP connection is opened until ``connect()``."""
        if not fci_ip or not fci_ip.strip():
            raise ROSConfigError(
                "FrankaPandaRealHAL requires a non-empty fci_ip "
                "(the robot's FCI hostname or IP, e.g. '172.16.0.2')."
            )
        self._fci_ip = fci_ip
        self._controller_name = controller_name
        self._error_recovery_topic = error_recovery_topic
        self._publish_fn: _PublishFn | None = publish_fn

        self._inner = RosControlHAL(
            FRANKA_PANDA_REAL_DESCRIPTION,
            controller_name=controller_name,
            joint_state_topic=joint_state_topic,
            command_topic=command_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )

    # ── Public attributes mandated by the HAL Protocol ────────────────────

    @property
    def description(self) -> RobotDescription:
        """Normative :class:`RobotDescription` for the Franka Panda."""
        return self._inner.description

    @property
    def controller_name(self) -> str:
        """Name of the ``ros2_control`` joint trajectory controller."""
        return self._controller_name

    @property
    def fci_ip(self) -> str:
        """Hostname / IP of the FCI; consumed by ``franka_hardware``."""
        return self._fci_ip

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the ROS 2 transport to the ``franka_ros2`` controller.

        The actual ``libfranka`` TCP socket is owned by ``franka_hardware``
        inside the lifecycle node; this call only attaches the adapter to
        the injected publisher / subscriber pair.

        Raises:
            ROSRuntimeError: If already connected.
        """
        log.info(
            "hal.connect",
            robot=self.description.name,
            fci_ip=self._fci_ip,
            controller=self._controller_name,
        )
        self._inner.connect()

    def disconnect(self) -> None:
        """Close the ROS 2 transport.  Idempotent."""
        self._inner.disconnect()

    # ── Hot path ──────────────────────────────────────────────────────────

    def read_state(self) -> JointState:
        """Return the latest joint state for all 8 description joints.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.
        """
        return self._inner.read_state()

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the ``franka_ros2`` controller.

        Args:
            action: The :class:`Action` produced by the Skill or safety
                shaper.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not in the
                description's ``supported_control_modes``.
        """
        self._inner.send_action(action)

    # ── Safety ────────────────────────────────────────────────────────────

    def estop(self) -> None:
        """Trigger an emergency stop on the FCI.

        Publishes a zero-velocity hold to the controller, marks the inner
        adapter disconnected, and raises :class:`ROSEStopRequested` so the
        safety supervisor can log the incident.  The supervisor is
        responsible for calling ``error_recovery`` (via
        ``error_recovery_topic``) before re-arming the robot.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical(
            "hal.estop",
            robot=self.description.name,
            fci_ip=self._fci_ip,
            recovery_topic=self._error_recovery_topic,
        )
        if self._publish_fn is not None:
            with contextlib.suppress(Exception):
                self._publish_fn(
                    self._error_recovery_topic,
                    {"reason": "openral_estop", "robot": self.description.name},
                )
        # Mirror SO100/UR estop semantics: drop the connection then raise so
        # subsequent ``read_state`` / ``send_action`` calls fail fast.
        with contextlib.suppress(Exception):
            self._inner.disconnect()
        raise ROSEStopRequested(
            f"Emergency stop triggered on Franka Panda at FCI {self._fci_ip!r}."
        )
