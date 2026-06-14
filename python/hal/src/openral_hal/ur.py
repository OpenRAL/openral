"""HAL adapters for the Universal Robots UR5e and UR10e arms.

The two arms share kinematic structure (6-DoF revolute, identical joint
naming) but differ in payload, reach, and per-joint velocity / effort limits.
We expose them as separate :class:`RobotDescription` constants and HAL
factories so safety envelopes and embodiment tags stay distinct.

Both adapters drive a MuJoCo simulation of the arm via
:class:`openral_hal._mujoco_arm.MujocoArmHAL`.  The MJCF is sourced from
``robot_descriptions`` (DeepMind ``mujoco_menagerie``).  Production deployment
on real hardware is expected to wrap the same ``RobotDescription`` with a
``ros2_control`` HAL talking to the ``ur_robot_driver`` (URCap / RTDE), but
that adapter lives in a separate package and is not part of this milestone.

Example:
    >>> from openral_hal import UR5e_DESCRIPTION
    >>> UR5e_DESCRIPTION.name
    'ur5e'
    >>> [j.name for j in UR5e_DESCRIPTION.joints]  # doctest: +NORMALIZE_WHITESPACE
    ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
     'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
"""

from __future__ import annotations

import math

from openral_core.schemas import (
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    HalEntrypoints,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
)

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._sensor_wiring import with_sensors

__all__ = [
    "UR5eHAL",
    "UR5e_DESCRIPTION",
    "UR10eHAL",
    "UR10e_DESCRIPTION",
    "ur5e_with_sensors",
    "ur10e_with_sensors",
]

# ── Canonical joint order ─────────────────────────────────────────────────────
# Matches the ``mujoco_menagerie`` UR MJCFs (same naming for UR5e and UR10e).

_UR_JOINT_NAMES: list[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Range of the elbow joint is half that of the others on real UR hardware
# (mechanical stop), reflected in the MJCF as ``[-pi, pi]``.
_UR5E_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan_joint": (-2 * math.pi, 2 * math.pi),
    "shoulder_lift_joint": (-2 * math.pi, 2 * math.pi),
    "elbow_joint": (-math.pi, math.pi),
    "wrist_1_joint": (-2 * math.pi, 2 * math.pi),
    "wrist_2_joint": (-2 * math.pi, 2 * math.pi),
    "wrist_3_joint": (-2 * math.pi, 2 * math.pi),
}

_UR5E_VELOCITY_LIMITS: dict[str, float] = {
    "shoulder_pan_joint": math.pi,
    "shoulder_lift_joint": math.pi,
    "elbow_joint": math.pi,
    "wrist_1_joint": math.pi,
    "wrist_2_joint": math.pi,
    "wrist_3_joint": math.pi,
}

# UR5e datasheet torque limits (Nm).
_UR5E_EFFORT_LIMITS: dict[str, float] = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

# UR10e: shoulder/elbow are slower (120°/s) and stronger; wrists keep the same
# velocity limits as the UR5e.
_UR10E_VELOCITY_LIMITS: dict[str, float] = {
    "shoulder_pan_joint": 2.094,  # 120°/s
    "shoulder_lift_joint": 2.094,
    "elbow_joint": 3.142,  # 180°/s
    "wrist_1_joint": math.pi,
    "wrist_2_joint": math.pi,
    "wrist_3_joint": math.pi,
}

# UR10e datasheet torque limits (Nm).
_UR10E_EFFORT_LIMITS: dict[str, float] = {
    "shoulder_pan_joint": 330.0,
    "shoulder_lift_joint": 330.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 56.0,
    "wrist_2_joint": 56.0,
    "wrist_3_joint": 56.0,
}


def _ur_joint_specs(
    velocity_limits: dict[str, float],
    effort_limits: dict[str, float],
) -> list[JointSpec]:
    parents = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
    ]
    children = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    return [
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            position_limits=_UR5E_POSITION_LIMITS[name],
            velocity_limit=velocity_limits[name],
            effort_limit=effort_limits[name],
            has_torque_sensor=True,  # UR provides motor torque feedback
            actuator_kind="bldc",
        )
        for name, parent, child in zip(_UR_JOINT_NAMES, parents, children, strict=True)
    ]


# ── UR5e ──────────────────────────────────────────────────────────────────────

UR5e_DESCRIPTION = RobotDescription(
    name="ur5e",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="ur5e_base_link",
    joints=_ur_joint_specs(_UR5E_VELOCITY_LIMITS, _UR5E_EFFORT_LIMITS),
    end_effectors=[
        EndEffectorSpec(
            name="tool0",
            kind="tool",
            n_dof=0,
            max_payload_kg=5.0,
            workspace_radius_m=0.85,
        )
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=5.0,
        has_force_control=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        embodiment_tags=["ur5e", "ur"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=150.0,
        max_torque_nm=150.0,
        deadman_required=True,
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim="openral_hal.ur:UR5eHAL", real="openral_hal.ur_real:UR5eRealHAL"),
    sim=SimDescription(mjcf_uri="robot_descriptions:ur5e_mj_description"),
)


# ── UR10e ─────────────────────────────────────────────────────────────────────

UR10e_DESCRIPTION = RobotDescription(
    name="ur10e",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="ur10e_base_link",
    joints=_ur_joint_specs(_UR10E_VELOCITY_LIMITS, _UR10E_EFFORT_LIMITS),
    end_effectors=[
        EndEffectorSpec(
            name="tool0",
            kind="tool",
            n_dof=0,
            max_payload_kg=12.5,
            workspace_radius_m=1.30,
        )
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=12.5,
        has_force_control=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        embodiment_tags=["ur10e", "ur"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=330.0,
        max_torque_nm=330.0,
        deadman_required=True,
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim="openral_hal.ur:UR10eHAL", real="openral_hal.ur_real:UR10eRealHAL"),
    sim=SimDescription(mjcf_uri="robot_descriptions:ur10e_mj_description"),
)


# ── Sensor-wired description factories (issue #23) ────────────────────────────


def ur5e_with_sensors(
    catalog_ids: list[str] | None = None,
) -> RobotDescription:
    """Return a copy of :data:`UR5e_DESCRIPTION` with catalog sensors attached.

    The reference UR5e research setup uses a flange-mounted RealSense D415
    plus a Robotiq FT-300S wrist sensor; pass ``None`` to get that default.

    Args:
        catalog_ids: Catalog ids to resolve, or ``None`` for the UR5e
            reference loadout (``["intel/realsense_d415", "robotiq/ft_300s"]``).

    Returns:
        A new :class:`RobotDescription` with ``sensors`` / ``sensor_bundles``
        populated.

    Example:
        >>> desc = ur5e_with_sensors()
        >>> desc.sensors[0].vendor
        'Robotiq'
    """
    if catalog_ids is None:
        catalog_ids = ["intel/realsense_d415", "robotiq/ft_300s"]
    return with_sensors(UR5e_DESCRIPTION, catalog_ids)


def ur10e_with_sensors(
    catalog_ids: list[str] | None = None,
) -> RobotDescription:
    """Return a copy of :data:`UR10e_DESCRIPTION` with catalog sensors attached.

    Default loadout matches :func:`ur5e_with_sensors` (D435 + FT-300S); the
    UR10e shares the same flange interface and is usually paired with the
    higher-range D435 instead of the D415.

    Args:
        catalog_ids: Catalog ids to resolve, or ``None`` for the UR10e
            reference loadout (``["intel/realsense_d435", "robotiq/ft_300s"]``).

    Returns:
        A new :class:`RobotDescription` with ``sensors`` / ``sensor_bundles``
        populated.

    Example:
        >>> desc = ur10e_with_sensors()
        >>> desc.sensor_bundles[0].sensors[0].vendor
        'Intel'
    """
    if catalog_ids is None:
        catalog_ids = ["intel/realsense_d435", "robotiq/ft_300s"]
    return with_sensors(UR10e_DESCRIPTION, catalog_ids)


# ── HAL factories ─────────────────────────────────────────────────────────────


class UR5eHAL(MujocoArmHAL):
    """HAL adapter for the Universal Robots UR5e (MuJoCo-backed simulation).

    Thin manifest-driven wrapper — every wiring constant lives in
    :data:`UR5e_DESCRIPTION.sim` (ADR-0023).

    Args:
        mjcf_path: Optional override for the MJCF file path.  When ``None``,
            ``UR5e_DESCRIPTION.sim.mjcf_uri`` is resolved at construction
            time (``robot_descriptions:ur5e_mj_description``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.  Defaults to 1.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect()``
            time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import UR5eHAL  # doctest: +SKIP
        >>> hal = UR5eHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> hal.disconnect()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int = 1,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the UR5e HAL; no MuJoCo state is created until ``connect()``."""
        self._init_from_description(
            UR5e_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )


class UR10eHAL(MujocoArmHAL):
    """HAL adapter for the Universal Robots UR10e (MuJoCo-backed simulation).

    Args mirror :class:`UR5eHAL`; the only difference is the underlying MJCF
    and ``RobotDescription`` (different velocity / effort envelopes), both of
    which now live in :data:`UR10e_DESCRIPTION` (ADR-0023).

    Example:
        >>> from openral_hal import UR10eHAL  # doctest: +SKIP
        >>> hal = UR10eHAL(gravity_enabled=False)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int = 1,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the UR10e HAL; no MuJoCo state is created until ``connect()``."""
        self._init_from_description(
            UR10e_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
