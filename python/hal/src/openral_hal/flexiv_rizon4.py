"""HAL adapter for the Flexiv Rizon 4 (MuJoCo digital twin).

The Flexiv Rizon 4 is a 7-DoF collaborative arm with whole-body force
sensitivity (0.1 N resolution) and a 4 kg payload at 780 mm reach.
The MuJoCo digital twin loads the upstream DeepMind
``mujoco_menagerie`` Rizon 4 MJCF (``flexiv_rizon4/flexiv_rizon4.xml``,
vendored via ``robot_descriptions.rizon4_mj_description``) and drives
the 7 position-controlled actuators directly via the shared
:class:`openral_hal._mujoco_arm.MujocoArmHAL` base.

This adapter is **structurally identical** to the UR / Franka HAL twins
(single-arm, no gripper, no floating base, position actuators).  The
only differences from those siblings are the MJCF path, the joint name
table, and the published-spec safety envelope.

Joint inventory
---------------
The menagerie MJCF declares 7 hinge joints in canonical order
(``joint1`` ... ``joint7``).  Position limits, velocity limits, and
effort limits come from the Flexiv-published spec sheet — the
menagerie MJCF pins them verbatim.

The Rizon 4 has no gripper, no floating base, and a single keyframe
that is **not** applied at ``connect()`` (every joint defaults to
qpos=0 which is inside every joint range — same convention as
:class:`openral_hal.UR5eHAL` / :class:`openral_hal.FrankaPandaHAL`).

Example:
    >>> from openral_hal import Rizon4MujocoHAL, RIZON4_DESCRIPTION
    >>> hal = Rizon4MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> state = hal.read_state()  # doctest: +SKIP
    >>> len(state.position) == len(RIZON4_DESCRIPTION.joints) == 7  # doctest: +SKIP
    True
    >>> hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

from openral_core.schemas import (
    AssetRefs,
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
    UrdfAsset,
)

from openral_hal._mujoco_arm import MujocoArmHAL

__all__ = ["RIZON4_DESCRIPTION", "Rizon4MujocoHAL"]


# ── Canonical joint order ─────────────────────────────────────────────────────
# Matches the menagerie MJCF: 7 hinge joints named ``joint1``..``joint7``.
# Verified at import time by
# ``tests/sim/test_rizon4_hal_mujoco.py::TestMenagerieSchema``.

_RIZON4_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)


# ── Joint limits ─────────────────────────────────────────────────────────────
# Position limits (rad) come from the menagerie MJCF, verbatim — the
# menagerie pins them to the published Flexiv Rizon 4 spec sheet.

_RIZON4_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "joint1": (-2.88, 2.88),
    "joint2": (-2.356, 2.356),
    "joint3": (-3.054, 3.054),
    "joint4": (-1.955, 2.775),
    "joint5": (-3.054, 3.054),
    "joint6": (-1.484, 4.625),
    "joint7": (-3.054, 3.054),
}

# Velocity limits (rad/s) from the Flexiv Rizon 4 spec sheet.  All
# joints are rated at the same nominal peak — the conservative value
# below is the safety-envelope-side limit (the production envelope
# lives in the safety supervisor, not the HAL).
_RIZON4_VELOCITY_LIMIT: float = 2.0

# Effort limits (N·m) from the spec sheet: joints 1-4 are rated at
# 123 N·m, joints 5-7 at 39 N·m.  The MJCF's ctrlrange isn't a torque
# limit (it's the position-actuator setpoint range), so these come
# from the data sheet directly.
_RIZON4_EFFORT_LIMITS: dict[str, float] = {
    "joint1": 123.0,
    "joint2": 123.0,
    "joint3": 64.0,
    "joint4": 64.0,
    "joint5": 39.0,
    "joint6": 39.0,
    "joint7": 39.0,
}


def _rizon4_joint_specs() -> list[JointSpec]:
    parents = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
    children = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
    return [
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            position_limits=_RIZON4_POSITION_LIMITS[name],
            velocity_limit=_RIZON4_VELOCITY_LIMIT,
            effort_limit=_RIZON4_EFFORT_LIMITS[name],
            has_torque_sensor=True,  # Rizon's whole-body force sensitivity
            actuator_kind="bldc",
        )
        for name, parent, child in zip(_RIZON4_JOINT_NAMES, parents, children, strict=True)
    ]


# ── RobotDescription ─────────────────────────────────────────────────────────

RIZON4_DESCRIPTION = RobotDescription(
    name="rizon4",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="base_link",
    joints=_rizon4_joint_specs(),
    end_effectors=[
        EndEffectorSpec(
            name="flange",
            kind="tool",
            n_dof=0,
            max_payload_kg=4.0,
            workspace_radius_m=0.78,
        )
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=4.0,
        has_force_control=True,  # Rizon 4's defining feature
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["rizon4"],
        embodiment_tags=["rizon4", "flexiv"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=150.0,
        max_torque_nm=123.0,  # peak rated torque of the proximal joints
        deadman_required=True,
    ),
    sdk_kind="open",
    # The Rizon 4 sim HAL is the only execution path today; a future
    # ``Rizon4RealHAL`` wrapping ``flexiv_rdk`` (closed-with-api per
    # CLAUDE.md §7.4 — Flexiv's RDK is BSD-style but vendor-distributed)
    # will derive its ``RobotDescription`` from this sim baseline via
    # ``make_real_description``, matching the UR / Franka / Sawyer /
    # ALOHA pattern.
    hal=HalEntrypoints(sim="openral_hal.flexiv_rizon4:Rizon4MujocoHAL", real=None),
    assets=AssetRefs(
        urdf=UrdfAsset(ref="file:rizon4.urdf"),
        mjcf="rd:rizon4_mj_description",
        srdf="file:rizon4.srdf",
    ),
    sim=SimDescription(),
)


# ── HAL ──────────────────────────────────────────────────────────────────────


class Rizon4MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Flexiv Rizon 4 (MuJoCo-backed simulation).

    Drives the 7 position-controlled actuators of the menagerie
    ``flexiv_rizon4`` MJCF through :class:`MujocoArmHAL`.  Exposes a
    7-D :class:`openral_core.Action` matching the joint order in
    :data:`RIZON4_DESCRIPTION` (``joint1`` ... ``joint7``).

    Args:
        mjcf_path: Optional override for the MJCF file path.  When
            ``None``, the file is fetched lazily from
            ``robot_descriptions``
            (``mujoco_menagerie/flexiv_rizon4/flexiv_rizon4.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.  Defaults to ``1``; raise it in tests
            that assert the arm has settled at the commanded pose.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import Rizon4MujocoHAL  # doctest: +SKIP
        >>> hal = Rizon4MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 7 arm joints  # doctest: +SKIP
        7
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
        """Initialise the Rizon 4 HAL; no MuJoCo state is created until ``connect()``.

        All wiring lives in :data:`RIZON4_DESCRIPTION.sim` (ADR-0023).
        """
        self._init_from_description(
            RIZON4_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
