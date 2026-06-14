"""HAL adapter for the Franka Emika Panda (FR3 predecessor) 7-DoF arm.

The Panda has 7 revolute joints plus a parallel gripper (two prismatic finger
joints driven by a single tendon-based actuator).  We expose **8 joints** to
upper layers — the 7 arm joints plus a synthetic gripper joint reported in
``[0, 1]`` — and convert the gripper command on the fly inside the HAL.

The simulation is driven by the ``mujoco_menagerie`` MJCF
(``franka_emika_panda/panda.xml``).  Production deployment over the Franka
Control Interface (FCI) is expected to wrap the same ``RobotDescription`` with
a ``ros2_control`` HAL talking to ``franka_ros2``; that adapter is out of
scope here.

Example:
    >>> from openral_hal import FRANKA_PANDA_DESCRIPTION
    >>> FRANKA_PANDA_DESCRIPTION.name
    'franka_panda'
    >>> [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]  # doctest: +NORMALIZE_WHITESPACE
    ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
     'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_gripper']
"""

from __future__ import annotations

from openral_core.schemas import (
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    HalEntrypoints,
    Hand,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
    SimGripperDescription,
)

from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._sensor_wiring import with_sensors

__all__ = ["FRANKA_PANDA_DESCRIPTION", "FrankaPandaHAL", "franka_panda_with_sensors"]

# ── Canonical joint order ─────────────────────────────────────────────────────
# Arm joints follow the upstream ``panda_jointN`` naming; the gripper is a
# synthetic 1-DoF channel exposed as ``panda_gripper`` to match the SO-100's
# normalised gripper convention.

_PANDA_ARM_JOINT_NAMES: list[str] = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

_PANDA_GRIPPER_JOINT_NAME = "panda_gripper"

_PANDA_JOINT_NAMES: list[str] = [*_PANDA_ARM_JOINT_NAMES, _PANDA_GRIPPER_JOINT_NAME]

# MJCF qpos / actuator wiring lives in FRANKA_PANDA_DESCRIPTION.sim
# (ADR-0023); the Python constants previously defined here have moved
# into the manifest.

# Native MuJoCo MJCF joint names (mujoco_menagerie franka_emika_panda/panda.xml).
# Used by SimAttachedHAL.read_state to resolve joints by name in both the
# native MjSpec scene (joint1..joint7) and robosuite/LIBERO scenes where the
# prefix-strip fallback maps robot0_joint1 → joint1 (ADR-0034).
_PANDA_SIM_JOINT_NAMES: dict[str, str] = {
    "panda_joint1": "joint1",
    "panda_joint2": "joint2",
    "panda_joint3": "joint3",
    "panda_joint4": "joint4",
    "panda_joint5": "joint5",
    "panda_joint6": "joint6",
    "panda_joint7": "joint7",
    "panda_gripper": "finger_joint1",
}

# Per-joint position limits from the Franka FR3 / Panda data sheet (rad).
# The MJCF mirrors these.
_PANDA_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}

# Velocity limits (rad/s) from the Franka data sheet.
_PANDA_VELOCITY_LIMITS: dict[str, float] = {
    "panda_joint1": 2.175,
    "panda_joint2": 2.175,
    "panda_joint3": 2.175,
    "panda_joint4": 2.175,
    "panda_joint5": 2.610,
    "panda_joint6": 2.610,
    "panda_joint7": 2.610,
}

# Torque limits (Nm) from the Franka data sheet.
_PANDA_EFFORT_LIMITS: dict[str, float] = {
    "panda_joint1": 87.0,
    "panda_joint2": 87.0,
    "panda_joint3": 87.0,
    "panda_joint4": 87.0,
    "panda_joint5": 12.0,
    "panda_joint6": 12.0,
    "panda_joint7": 12.0,
}

# ── Joint specs ──────────────────────────────────────────────────────────────


def _panda_joint_specs() -> list[JointSpec]:
    arm_parents = [
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
    ]
    arm_children = [
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
    ]
    arm = [
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            position_limits=_PANDA_POSITION_LIMITS[name],
            velocity_limit=_PANDA_VELOCITY_LIMITS[name],
            effort_limit=_PANDA_EFFORT_LIMITS[name],
            has_torque_sensor=True,  # Panda has joint torque sensors
            actuator_kind="bldc",
            sim_joint_name=_PANDA_SIM_JOINT_NAMES[name],
        )
        for name, parent, child in zip(
            _PANDA_ARM_JOINT_NAMES, arm_parents, arm_children, strict=True
        )
    ]
    gripper = JointSpec(
        name=_PANDA_GRIPPER_JOINT_NAME,
        joint_type=JointType.PRISMATIC,
        parent_link="panda_hand",
        child_link="panda_finger_pair",
        # Synthetic normalised channel: 0 = closed, 1 = fully open.
        position_limits=(0.0, 1.0),
        velocity_limit=0.1,  # 0.1 m/s combined finger speed
        effort_limit=70.0,  # 70 N max grip force per Franka data sheet
        has_torque_sensor=False,
        actuator_kind="servo",
        sim_joint_name=_PANDA_SIM_JOINT_NAMES[_PANDA_GRIPPER_JOINT_NAME],
    )
    return [*arm, gripper]


# ── RobotDescription ─────────────────────────────────────────────────────────

FRANKA_PANDA_DESCRIPTION = RobotDescription(
    name="franka_panda",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="panda_link0",
    joints=_panda_joint_specs(),
    end_effectors=[
        EndEffectorSpec(
            name="panda_hand",
            kind="parallel_gripper",
            hand=Hand.NA,
            n_dof=1,
            max_grip_force_n=70.0,
            max_payload_kg=3.0,
            workspace_radius_m=0.855,
        )
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=3.0,
        has_force_control=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        embodiment_tags=["franka_panda", "franka", "panda", "libero"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=100.0,
        max_torque_nm=87.0,
        deadman_required=True,
    ),
    # ADR-0031 — the shared ``hal`` block names both the sim HAL
    # (``FrankaPandaHAL``) and the real-HW HAL (``FrankaPandaRealHAL``);
    # ``build_hal(mode=...)`` picks one. ``FRANKA_PANDA_REAL_DESCRIPTION`` in
    # ``franka_panda_real.py`` derives from this baseline via
    # ``make_real_description`` (it inherits this same ``hal``, flipping only
    # ``sdk_kind``). ``robots/franka_panda/robot.yaml`` mirrors the real one.
    sdk_kind="open",
    hal=HalEntrypoints(
        sim="openral_hal.franka_panda:FrankaPandaHAL",
        real="openral_hal.franka_panda_real:FrankaPandaRealHAL",
    ),
    sim=SimDescription(
        mjcf_uri="robot_descriptions:panda_mj_description",
        grippers=[
            SimGripperDescription(
                joint="panda_gripper",
                ctrl_range=(0.0, 255.0),
                qpos_addrs=(7, 8),
                qpos_scale=0.08,  # 2 * 0.04 m max finger extent
            ),
        ],
    ),
)


# ── Sensor-wired description factory (issue #23) ──────────────────────────────


def franka_panda_with_sensors(
    catalog_ids: list[str] | None = None,
) -> RobotDescription:
    """Return a copy of :data:`FRANKA_PANDA_DESCRIPTION` with catalog sensors attached.

    The Franka Panda research community reference setup uses a wrist-mounted
    RealSense D435i; pass ``None`` to get that default, or override.

    Args:
        catalog_ids: Catalog ids to resolve, or ``None`` for the Franka
            reference loadout (``["intel/realsense_d435i"]``).

    Returns:
        A new :class:`RobotDescription` with ``sensors`` / ``sensor_bundles``
        populated.

    Example:
        >>> desc = franka_panda_with_sensors()
        >>> desc.sensor_bundles[0].sensors[0].vendor
        'Intel'
    """
    if catalog_ids is None:
        catalog_ids = ["intel/realsense_d435i"]
    return with_sensors(FRANKA_PANDA_DESCRIPTION, catalog_ids)


# ── HAL ──────────────────────────────────────────────────────────────────────


class FrankaPandaHAL(MujocoArmHAL):
    """HAL adapter for the Franka Emika Panda (MuJoCo-backed simulation).

    The HAL exposes 8 joints to upper layers — the 7 arm joints plus a
    synthetic ``panda_gripper`` channel reported in ``[0, 1]`` (0 = closed,
    1 = fully open).  Internally the gripper command is mapped to the MJCF
    tendon actuator's ``[0, 255]`` range, and the reported gripper position
    is the sum of the two finger ``qpos`` values normalised by ``0.08`` m.

    Args:
        mjcf_path: Optional override for the MJCF file path.  When ``None``,
            the file is fetched lazily from ``robot_descriptions``
            (``mujoco_menagerie/franka_emika_panda/panda.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.
        gravity_enabled: When ``False``, gravity is zeroed at ``connect()``
            time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import FrankaPandaHAL  # doctest: +SKIP
        >>> hal = FrankaPandaHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 7 arm + 1 gripper  # doctest: +SKIP
        8
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
        """Initialise the Panda HAL; no MuJoCo state is created until ``connect()``.

        All wiring (MJCF URI, joint indices, gripper config) lives in
        :data:`FRANKA_PANDA_DESCRIPTION.sim` (ADR-0023).
        """
        self._init_from_description(
            FRANKA_PANDA_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
