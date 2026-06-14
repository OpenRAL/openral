"""HAL adapter for the Unitree G1 humanoid (MuJoCo digital twin).

This module wraps the upstream DeepMind ``mujoco_menagerie`` G1 MJCF
(``unitree_g1/g1.xml``, vendored via ``robot_descriptions``) as a
:class:`openral_hal.HAL` Protocol implementation, extending the
:class:`openral_hal.UR5eHAL` / :class:`openral_hal.FrankaPandaHAL` /
:class:`openral_hal.SO100MujocoHAL` pattern to a 29-DoF bipedal
humanoid.

What this is — and what it isn't
--------------------------------
This HAL is a **digital-twin contract validator**, not a useful
humanoid sim. The G1 has a floating base and no S0 cerebellar
controller; left to its own devices it falls over under gravity and
the closed-loop convergence tests therefore run with
``gravity_enabled=False``. The point of the suite is the same as for
the SO-100 / ALOHA twins (CLAUDE.md §1.11):

* the 29-DoF joint-position action layout,
* the lifecycle wiring (``connect → read_state → send_action → estop``),
* the joint indexing,
* the ``RobotDescription`` round-trip,
* and the embodiment / VLA tag plumbing

all behave the same way the future ``G1RealHAL`` will see when the
physical robot is plugged in. Balance, walking, and any actually
useful humanoid control still live in CLAUDE.md §6.2 territory — the
C++ S0 cerebellum tracked under the M2 milestone — and are explicitly
out of scope here. See `docs/architecture/repo-state-map.html` for
the "HAL · Unitree G1 (real-HW)" planned block.

Joint inventory
---------------
The menagerie MJCF has 30 joints (29 actuated + 1 floating base) and
29 position actuators in a fixed order. The ``floating_base_joint``
is the free joint for the pelvis pose and is *not* exposed on the
public ``RobotDescription`` — it is implicit world state, not
something a Skill commands. The 29 actuated joints are, in order:

    legs  : 2 x (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    waist : yaw, roll, pitch
    arms  : 2 x (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                 wrist_roll, wrist_pitch, wrist_yaw)

i.e. 12 + 3 + 14 = 29.  qpos addresses for the actuated joints are
``7..35`` (the first 7 qpos slots belong to the floating base);
actuator indices are ``0..28`` and align 1:1 with the joint name
order above.

The wrist endpoint is a bare joint — this menagerie variant does NOT
ship hand actuators, so there is no gripper to map. A future
``g1_with_hands`` variant would need a hand-aware subclass.

Example:
    >>> from openral_hal import G1MujocoHAL, G1_DESCRIPTION
    >>> hal = G1MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> state = hal.read_state()  # doctest: +SKIP
    >>> len(state.position) == len(G1_DESCRIPTION.joints) == 29  # doctest: +SKIP
    True
    >>> hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    ControlMode,
    EmbodimentKind,
    HalEntrypoints,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
)

from openral_hal._mujoco_arm import MujocoArmHAL

__all__ = ["G1_DESCRIPTION", "G1MujocoHAL"]


# ── Canonical joint order ─────────────────────────────────────────────────────
# Matches the menagerie MJCF: 6 (left leg) + 6 (right leg) + 3 (waist)
# + 7 (left arm) + 7 (right arm) = 29 actuated joints.  Verified at
# import time by ``tests/sim/test_g1_hal_mujoco.py::TestMenagerieSchema``.

_G1_LEFT_LEG_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
_G1_RIGHT_LEG_JOINTS: tuple[str, ...] = (
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_G1_WAIST_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
_G1_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
_G1_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_JOINT_NAMES: tuple[str, ...] = (
    *_G1_LEFT_LEG_JOINTS,
    *_G1_RIGHT_LEG_JOINTS,
    *_G1_WAIST_JOINTS,
    *_G1_LEFT_ARM_JOINTS,
    *_G1_RIGHT_ARM_JOINTS,
)


# ── Joint limits ─────────────────────────────────────────────────────────────
# Position limits (rad) come from the menagerie MJCF, verbatim — the
# menagerie pins them to the upstream Unitree G1 URDF / data sheet.
# Velocity + effort are conservative published-spec values: the G1
# Edition 6f spec sheet lists ~30 N·m peak for the small-actuator
# joints (wrist / waist) and ~88 N·m for the large ones (hip / knee).
# These show up in ``JointSpec.effort_limit`` / ``velocity_limit`` for
# capability matching only — MuJoCo's ``mj_step`` does not honour them
# (its actuators use ``ctrlrange`` from the MJCF instead).

_G1_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}

# Conservative velocity limits — the G1 spec lists ~12 rad/s peak for
# hip / knee, ~20 rad/s for wrists.  Halved here for the safety
# envelope (CLAUDE.md §1.1 — refuse a request before it can damage the
# hardware).
_G1_VELOCITY_LIMITS_BY_GROUP: dict[str, float] = {
    "hip": 6.0,
    "knee": 6.0,
    "ankle": 6.0,
    "waist": 6.0,
    "shoulder": 6.0,
    "elbow": 6.0,
    "wrist": 10.0,
}
_G1_EFFORT_LIMITS_BY_GROUP: dict[str, float] = {
    "hip": 88.0,
    "knee": 88.0,
    "ankle": 30.0,
    "waist": 88.0,
    "shoulder": 30.0,
    "elbow": 30.0,
    "wrist": 30.0,
}


def _g1_group(joint_name: str) -> str:
    """Return the kinematic group for *joint_name* (``hip`` / ``knee`` / ``wrist`` / …)."""
    for token in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"):
        if token in joint_name:
            return token
    raise ROSConfigError(f"Unknown G1 joint group for joint {joint_name!r}.")


def _g1_parent_child(joint_name: str) -> tuple[str, str]:
    """Return ``(parent_link, child_link)`` for *joint_name*.

    The names follow the menagerie URDF convention — ``pelvis`` for the
    floating-base root, then ``<side>_<segment>_link`` for the body
    above each joint.  Strict accuracy isn't necessary (these fields
    are descriptive metadata for the JointSpec, not used by the HAL
    contract), but using stable names keeps :class:`RobotDescription`
    diffs readable.
    """
    # The "previous" link in each chain.  ``waist_*`` lifts off the pelvis;
    # the arms / shoulders lift off the upper-torso end of the waist chain.
    prev_link: dict[str, str] = {
        "left_hip_pitch_joint": "pelvis",
        "left_hip_roll_joint": "left_hip_pitch_link",
        "left_hip_yaw_joint": "left_hip_roll_link",
        "left_knee_joint": "left_hip_yaw_link",
        "left_ankle_pitch_joint": "left_knee_link",
        "left_ankle_roll_joint": "left_ankle_pitch_link",
        "right_hip_pitch_joint": "pelvis",
        "right_hip_roll_joint": "right_hip_pitch_link",
        "right_hip_yaw_joint": "right_hip_roll_link",
        "right_knee_joint": "right_hip_yaw_link",
        "right_ankle_pitch_joint": "right_knee_link",
        "right_ankle_roll_joint": "right_ankle_pitch_link",
        "waist_yaw_joint": "pelvis",
        "waist_roll_joint": "waist_yaw_link",
        "waist_pitch_joint": "waist_roll_link",
        "left_shoulder_pitch_joint": "torso_link",
        "left_shoulder_roll_joint": "left_shoulder_pitch_link",
        "left_shoulder_yaw_joint": "left_shoulder_roll_link",
        "left_elbow_joint": "left_shoulder_yaw_link",
        "left_wrist_roll_joint": "left_elbow_link",
        "left_wrist_pitch_joint": "left_wrist_roll_link",
        "left_wrist_yaw_joint": "left_wrist_pitch_link",
        "right_shoulder_pitch_joint": "torso_link",
        "right_shoulder_roll_joint": "right_shoulder_pitch_link",
        "right_shoulder_yaw_joint": "right_shoulder_roll_link",
        "right_elbow_joint": "right_shoulder_yaw_link",
        "right_wrist_roll_joint": "right_elbow_link",
        "right_wrist_pitch_joint": "right_wrist_roll_link",
        "right_wrist_yaw_joint": "right_wrist_pitch_link",
    }
    parent = prev_link[joint_name]
    # The child link of the joint is named after the joint itself, dropping
    # the ``_joint`` suffix and appending ``_link``.
    child = joint_name.removesuffix("_joint") + "_link"
    return parent, child


def _g1_joint_specs() -> list[JointSpec]:
    specs: list[JointSpec] = []
    for name in _G1_JOINT_NAMES:
        group = _g1_group(name)
        parent, child = _g1_parent_child(name)
        specs.append(
            JointSpec(
                name=name,
                joint_type=JointType.REVOLUTE,
                parent_link=parent,
                child_link=child,
                position_limits=_G1_POSITION_LIMITS[name],
                velocity_limit=_G1_VELOCITY_LIMITS_BY_GROUP[group],
                effort_limit=_G1_EFFORT_LIMITS_BY_GROUP[group],
                has_torque_sensor=True,  # G1 publishes joint torque feedback
                actuator_kind="bldc",
            )
        )
    return specs


# ── RobotDescription ─────────────────────────────────────────────────────────
# ``EmbodimentKind.HUMANOID`` because the G1 is bipedal; this triggers
# the eval / dispatcher pathways that recognise a humanoid embodiment
# (GR00T N1.x heads, ``humanoid_everyday_g1`` policies, etc.).  The
# floating-base joint is intentionally NOT enumerated — it's implicit
# world state, not something a Skill commands.

G1_DESCRIPTION = RobotDescription(
    name="g1",
    embodiment_kind=EmbodimentKind.HUMANOID,
    base_frame="pelvis",
    joints=_g1_joint_specs(),
    # No end-effector in this menagerie variant — the wrist is a bare
    # joint chain.  Future ``g1_with_hands`` revs would add Inspire / Dex-3
    # hands as ``EndEffectorSpec`` entries here.
    end_effectors=[],
    capabilities=RobotCapabilities(
        locomotion=["bipedal"],
        can_lift_kg=3.0,
        has_dexterous_hands=False,
        has_force_control=True,
        has_vision=False,  # built-in vision lives on the head; this manifest
        # is the base robot only — sensors are wired by the bring-up YAML.
        bimanual=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["g1", "humanoid_everyday_g1"],
        embodiment_tags=["g1", "unitree_g1", "humanoid"],
    ),
    safety=SafetyEnvelope(
        # Generous workspace envelope — the G1 stands ≈1.3 m tall and
        # reaches ≈0.7 m horizontally.  The S0 cerebellum (planned, M2)
        # is the real enforcer; this envelope is a Python-side belt
        # for the contract-validation path.
        max_ee_speed_m_s=1.5,
        max_joint_speed_factor=0.5,
        max_force_n=100.0,
        max_torque_nm=88.0,
        deadman_required=True,
    ),
    sdk_kind="open",
    # The G1 sim HAL is the only execution path today; a future
    # ``G1RealHAL`` over ``unitree_sdk2`` will derive its
    # ``RobotDescription`` from this sim baseline via
    # ``make_real_description`` (matches the UR / Franka / Sawyer / ALOHA
    # pattern) and will be tracked by a follow-up issue.  Until then the
    # manifest under ``robots/g1/robot.yaml`` mirrors this sim baseline.
    hal=HalEntrypoints(sim="openral_hal.g1:G1MujocoHAL", real=None),
    # Floating-base humanoid — qpos is offset by 7 (3 position + 4 quaternion);
    # qvel is offset by 6 (3 linear vel + 3 angular vel).  ``MujocoArmHAL``
    # derives both offsets from ``floating_base=True`` (ADR-0023).
    sim=SimDescription(
        mjcf_uri="robot_descriptions:g1_mj_description",
        floating_base=True,
    ),
)


# ── HAL ──────────────────────────────────────────────────────────────────────


class G1MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Unitree G1 humanoid (MuJoCo digital twin).

    Drives the 29 actuated joints of the menagerie ``unitree_g1`` MJCF
    through MuJoCo's position-controlled actuators.  Exposes a
    29-D :class:`openral_core.Action` matching the joint order in
    :data:`G1_DESCRIPTION` (left leg 6 → right leg 6 → waist 3 → left
    arm 7 → right arm 7).

    .. warning::

       This HAL does **not** provide balance.  Without an S0
       cerebellar controller (CLAUDE.md §6.2, M2 milestone) the robot
       will fall over under gravity.  The closed-loop convergence
       tests run with ``gravity_enabled=False`` for that reason.  Use
       this HAL to validate the action contract / joint indexing /
       lifecycle, **not** to roll out a humanoid policy.

    Args:
        mjcf_path: Optional override for the MJCF file path.  When
            ``None``, the file is fetched lazily from
            ``robot_descriptions``
            (``mujoco_menagerie/unitree_g1/g1.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.  Defaults to ``1``; raise it in tests
            that assert the body has converged at the commanded pose.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time — required for the contract-validation
            tests because the floating base falls otherwise.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import G1MujocoHAL  # doctest: +SKIP
        >>> hal = G1MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 29 actuated joints  # doctest: +SKIP
        29
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
        """Initialise the G1 HAL; no MuJoCo state is created until ``connect()``.

        All wiring (MJCF URI, floating-base offsets) lives in
        :data:`G1_DESCRIPTION.sim` (ADR-0023).
        """
        self._init_from_description(
            G1_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
