"""Real-hardware HAL adapter for the Rethink Robotics Sawyer 7-DoF arm.

The physical Sawyer is the real robot behind MetaWorld's MT50 / ML45
benchmarks; gym-MetaWorld uses a MuJoCo replica of the same kinematics for
sim. This module wires the **real-hardware** Layer-0 path, while the
MetaWorld sim path is owned by the eval-layer scene adapter.

Software / driver landscape
---------------------------
Rethink Robotics dissolved in 2018 (assets acquired by Hahn Group); the
``intera_sdk`` repository (BSD-3) is unmaintained upstream.  Active
maintenance lives in community forks — the most current public ROS 2
target is `RethinkRobotics-opensource/sawyer_robot`_, which provides URDFs,
gripper drivers, and a ``ros2_control`` shim built on top of Rethink's
original SDK.  The shim exposes a joint trajectory controller (default
name: ``"sawyer_arm_controller"``) and the ``/robot/joint_states`` topic
(legacy intera_sdk topic name preserved by the fork).

Per CLAUDE.md §7.4 ``intera_sdk`` is BSD-3 — fully compatible — so the
manifest declares ``sdk_kind: "closed_with_api"`` (the original company is
gone but the SDK + the forks need an explicit lineage tag) and sets
``hal.real`` to this adapter (Sawyer has no sim HAL, so ``hal.sim`` is None;
ADR-0031).

.. _RethinkRobotics-opensource/sawyer_robot:
   https://github.com/RethinkRobotics-opensource/sawyer_robot

Example:
    >>> from openral_hal.sawyer_real import SAWYER_DESCRIPTION
    >>> SAWYER_DESCRIPTION.name
    'sawyer'
    >>> [j.name for j in SAWYER_DESCRIPTION.joints][:3]
    ['right_j0', 'right_j1', 'right_j2']
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import structlog
from openral_core.exceptions import ROSConfigError, ROSEStopRequested
from openral_core.schemas import (
    Action,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    HalEntrypoints,
    Hand,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)

from openral_hal._real_description import make_real_description
from openral_hal.ros_control import RosControlHAL

__all__ = ["SAWYER_DESCRIPTION", "SAWYER_REAL_DESCRIPTION", "SawyerRealHAL"]

log = structlog.get_logger(__name__)

# ── Canonical joint order (matches sawyer_robot URDF + intera_sdk topics) ─────

_SAWYER_JOINT_NAMES: list[str] = [
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
]

# Position / velocity / effort limits taken from the public Rethink Sawyer
# data sheet.  The MetaWorld MJCF and the sawyer_robot URDF both mirror
# these numbers.
_SAWYER_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "right_j0": (-3.0503, 3.0503),
    "right_j1": (-3.8095, 2.2842),
    "right_j2": (-3.0426, 3.0426),
    "right_j3": (-3.0439, 3.0439),
    "right_j4": (-2.9761, 2.9761),
    "right_j5": (-2.9761, 2.9761),
    "right_j6": (-4.7124, 4.7124),
}

_SAWYER_VELOCITY_LIMITS: dict[str, float] = {
    "right_j0": 1.74,
    "right_j1": 1.328,
    "right_j2": 1.957,
    "right_j3": 1.957,
    "right_j4": 3.485,
    "right_j5": 3.485,
    "right_j6": 4.545,
}

_SAWYER_EFFORT_LIMITS: dict[str, float] = {
    "right_j0": 80.0,
    "right_j1": 80.0,
    "right_j2": 40.0,
    "right_j3": 40.0,
    "right_j4": 9.0,
    "right_j5": 9.0,
    "right_j6": 9.0,
}


def _sawyer_joint_specs() -> list[JointSpec]:
    parents = ["base", "right_l0", "right_l1", "right_l2", "right_l3", "right_l4", "right_l5"]
    children = ["right_l0", "right_l1", "right_l2", "right_l3", "right_l4", "right_l5", "right_l6"]
    arm = [
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            position_limits=_SAWYER_POSITION_LIMITS[name],
            velocity_limit=_SAWYER_VELOCITY_LIMITS[name],
            effort_limit=_SAWYER_EFFORT_LIMITS[name],
            has_torque_sensor=True,
            actuator_kind="bldc",
            role="arm",
        )
        for name, parent, child in zip(_SAWYER_JOINT_NAMES, parents, children, strict=True)
    ]
    # ADR-0028a — Rethink Electric Parallel Gripper as a 1-DoF prismatic
    # abstraction over the per-finger mimic. Matches
    # ``robots/sawyer/robot.yaml``; drift guarded by
    # ``tests/unit/test_robot_manifests_match_hal_constants.py``.
    gripper = JointSpec(
        name="right_gripper",
        joint_type=JointType.PRISMATIC,
        parent_link="right_l6",
        child_link="right_finger_pair",
        position_limits=(0.0, 0.041),
        velocity_limit=0.15,
        effort_limit=35.0,
        actuator_kind="servo",
        role="gripper",
    )
    return [*arm, gripper]


# ── RobotDescription (sim baseline) ──────────────────────────────────────────
# Sim/kinematic baseline.  Sawyer has no MuJoCo HAL adapter today (the
# MetaWorld scene adapter drives sim directly), so ``hal.sim`` is ``None``
# until a sim HAL lands; ``hal.real`` points at the intera_sdk adapter. The
# real-HW manifest below derives kinematics + safety from this constant via
# ``make_real_description`` (inheriting the same ``hal``).

SAWYER_DESCRIPTION = RobotDescription(
    name="sawyer",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="base",
    joints=_sawyer_joint_specs(),
    end_effectors=[
        EndEffectorSpec(
            name="right_hand",
            kind="parallel_gripper",
            hand=Hand.RIGHT,
            n_dof=1,
            max_grip_force_n=35.0,
            max_payload_kg=4.0,
            workspace_radius_m=1.26,
        )
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=4.0,
        has_force_control=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["sawyer"],
        embodiment_tags=["sawyer", "rethink"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=80.0,
        max_torque_nm=80.0,
        deadman_required=True,
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim=None, real="openral_hal.sawyer_real:SawyerRealHAL"),
    assets=AssetRefs(mjcf="rd:sawyer_mj_description"),
)


# ── RobotDescription (real-HW) ───────────────────────────────────────────────
# Pinned by ``robots/sawyer/robot.yaml``; drift guarded by
# ``tests/unit/test_robot_manifests_match_hal_constants.py``.

SAWYER_REAL_DESCRIPTION = make_real_description(
    SAWYER_DESCRIPTION,
    sdk_kind="closed_with_api",
)


# ── HAL ──────────────────────────────────────────────────────────────────────

# Default ros2_control controller exported by sawyer_robot's bring-up
# launch (mirrors the URDF + intera_sdk's joint trajectory controller).
_DEFAULT_SAWYER_CONTROLLER: str = "sawyer_arm_controller"

# Default joint state topic.  The sawyer_robot fork preserves the legacy
# intera_sdk topic name so existing tooling keeps working; new deployments
# can override at construction time.
_DEFAULT_SAWYER_JOINT_STATE_TOPIC: str = "/robot/joint_states"

# Rethink's "halt" topic — publishing to it stops the arm and clears any
# pending trajectory; analogous to franka_ros2's error_recovery.
_DEFAULT_SAWYER_ESTOP_TOPIC: str = "/robot/set_super_stop"

_PublishFn = Callable[[str, dict[str, object]], None]
_StateFn = Callable[[], dict[str, object]]


class SawyerRealHAL:
    """HAL adapter for a physical Rethink Sawyer over ``intera_sdk`` / ROS 2.

    Args:
        hostname: Hostname of the Sawyer's onboard PC, typically
            ``"sawyer.local"`` on a lab subnet.  Required; the upstream
            ``intera_sdk`` (and ``sawyer_robot`` fork) refuses to connect
            without it.  Stored as metadata only — the actual TCP
            connection is opened by the lifecycle node.
        controller_name: Name of the ``ros2_control`` joint trajectory
            controller exported by ``sawyer_robot``.  Defaults to
            ``"sawyer_arm_controller"``.
        joint_state_topic: ROS 2 topic publishing
            ``sensor_msgs/JointState``.  Defaults to
            ``"/robot/joint_states"`` (the legacy intera_sdk topic name).
        command_topic: ROS 2 topic for joint trajectory commands.  Defaults
            to ``"/<controller_name>/joint_trajectory"``.
        estop_topic: ROS 2 topic the safety supervisor publishes to on
            ``estop()``.  Defaults to ``"/robot/set_super_stop"``.
        publish_fn: Callable forwarding messages to ROS 2 topics.
            Production use injects the lifecycle node's publisher; tests
            inject :class:`SimTransport.publish`.
        state_fn: Callable returning the latest raw joint state as a dict.
            Production use injects the lifecycle node's subscriber
            callback; tests inject :class:`SimTransport.state`.
        staleness_limit_s: Maximum age of a ``read_state()`` reading
            before :class:`ROSPerceptionStale` is raised.  Defaults to
            ``0.2 s`` (Sawyer's intera_sdk feedback rate is ~100 Hz).

    Raises:
        ROSConfigError: If ``hostname`` is empty / whitespace.

    Example:
        >>> from openral_hal.sawyer_real import SawyerRealHAL
        >>> from openral_hal.sim_transport import SimTransport
        >>> transport = SimTransport(n_joints=7)
        >>> hal = SawyerRealHAL(
        ...     hostname="sawyer.local",
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.connect()
        >>> hal.description.name
        'sawyer'
        >>> hal.disconnect()
    """

    def __init__(
        self,
        *,
        hostname: str = "sawyer.local",
        controller_name: str = _DEFAULT_SAWYER_CONTROLLER,
        joint_state_topic: str = _DEFAULT_SAWYER_JOINT_STATE_TOPIC,
        command_topic: str | None = None,
        estop_topic: str = _DEFAULT_SAWYER_ESTOP_TOPIC,
        publish_fn: _PublishFn | None = None,
        state_fn: _StateFn | None = None,
        staleness_limit_s: float = 0.2,
    ) -> None:
        """Initialise the adapter; no TCP connection is opened until ``connect()``."""
        if not hostname or not hostname.strip():
            raise ROSConfigError(
                "SawyerRealHAL requires a non-empty hostname "
                "(e.g. 'sawyer.local' or the robot's IP)."
            )
        self._hostname = hostname
        self._controller_name = controller_name
        self._estop_topic = estop_topic
        self._publish_fn: _PublishFn | None = publish_fn

        self._inner = RosControlHAL(
            SAWYER_REAL_DESCRIPTION,
            controller_name=controller_name,
            joint_state_topic=joint_state_topic,
            command_topic=command_topic,
            publish_fn=publish_fn,
            state_fn=state_fn,
            staleness_limit_s=staleness_limit_s,
        )

    # ── HAL Protocol ──────────────────────────────────────────────────────

    @property
    def description(self) -> RobotDescription:
        """Normative :class:`RobotDescription` for the Sawyer."""
        return self._inner.description

    @property
    def hostname(self) -> str:
        """Hostname / IP of the Sawyer's onboard PC."""
        return self._hostname

    @property
    def controller_name(self) -> str:
        """Name of the ``ros2_control`` joint trajectory controller."""
        return self._controller_name

    def connect(self) -> None:
        """Open the ROS 2 transport to the ``sawyer_robot`` controller.

        Raises:
            ROSRuntimeError: If already connected.
        """
        log.info(
            "hal.connect",
            robot=self.description.name,
            hostname=self._hostname,
            controller=self._controller_name,
        )
        self._inner.connect()

    def disconnect(self) -> None:
        """Close the ROS 2 transport.  Idempotent."""
        self._inner.disconnect()

    def read_state(self) -> JointState:
        """Return the latest joint state for all 7 description joints.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.
        """
        return self._inner.read_state()

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the ``sawyer_robot`` controller.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not in the
                description's ``supported_control_modes``.
        """
        self._inner.send_action(action)

    def estop(self) -> None:
        """Trigger an emergency stop on the Sawyer.

        Publishes to the legacy intera_sdk halt topic, marks the inner
        adapter disconnected, and raises :class:`ROSEStopRequested`.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical(
            "hal.estop",
            robot=self.description.name,
            hostname=self._hostname,
            estop_topic=self._estop_topic,
        )
        if self._publish_fn is not None:
            with contextlib.suppress(Exception):
                self._publish_fn(
                    self._estop_topic,
                    {"reason": "openral_estop", "robot": self.description.name},
                )
        with contextlib.suppress(Exception):
            self._inner.disconnect()
        raise ROSEStopRequested(f"Emergency stop triggered on Sawyer at host {self._hostname!r}.")
