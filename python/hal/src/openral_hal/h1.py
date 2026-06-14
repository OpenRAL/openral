"""HAL adapter for the Unitree H1 humanoid (MuJoCo digital twin).

This module wraps the upstream DeepMind ``mujoco_menagerie`` H1 MJCF
(``unitree_h1/h1.xml``, vendored via ``robot_descriptions``) as a
:class:`openral_hal.HAL` Protocol implementation, following the same
pattern as :class:`openral_hal.G1MujocoHAL` — the H1's bigger,
earlier sibling.

What this is — and what it isn't
--------------------------------
Like :class:`G1MujocoHAL`, this HAL is a **digital-twin contract
validator**, not a useful humanoid sim.  The H1 has a floating base
and no S0 cerebellar controller; left to its own devices it falls
over under gravity and the closed-loop convergence tests therefore
run with ``gravity_enabled=False``.  The point of the suite is the
same as for the SO-100 / ALOHA / G1 twins (CLAUDE.md §1.11):

* the 19-DoF joint-position action layout,
* the lifecycle wiring (``connect → read_state → send_action → estop``),
* the joint indexing,
* the ``RobotDescription`` round-trip,
* and the embodiment / VLA tag plumbing

all behave the same way the future ``H1RealHAL`` will see when the
physical robot is plugged in.  Balance, walking, and any actually
useful humanoid control live in CLAUDE.md §6.2 territory — the C++
S0 cerebellum tracked under the M2 milestone — and are explicitly
out of scope here.

Joint inventory
---------------
The menagerie MJCF has 20 joints (19 actuated + 1 floating base) and
19 position actuators in a fixed order.  The floating base is the
free joint for the pelvis pose and is *not* exposed on the public
``RobotDescription`` — it is implicit world state, not something a
Skill commands.  The 19 actuated joints, in order:

    legs  : 2 x (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    torso : 1 (yaw only)
    arms  : 2 x (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)

i.e. 10 + 1 + 8 = 19.  qpos addresses for the actuated joints are
``7..25`` (the first 7 qpos slots belong to the floating base);
actuator indices are ``0..18`` and align 1:1 with the joint name
order above.

Differences from the G1 ``g1.py``
---------------------------------
* **19 DoF vs 29 DoF** — H1 is a coarser-DoF predecessor: each leg
  is 5-DoF (no separate hip yaw / yaw split, single-DoF ankle), the
  waist is 1-DoF (torso yaw only, no waist roll / pitch), and each
  arm stops at the elbow (no wrists).
* **No keyframe** — the menagerie H1 MJCF ships ``nkey=0``; the
  rest pose is every actuated joint at ``qpos=0``, which is already
  the upright neutral pose.  Unlike the ALOHA twin we don't need
  ``mj_resetDataKeyframe`` in ``connect()``.
* **Joint names** follow the menagerie convention without the
  ``_joint`` suffix (``left_hip_yaw`` not ``left_hip_yaw_joint``) —
  a stylistic difference between the two menagerie packages.
* **No hands** — wrists aren't actuated.  A future
  ``h1_with_hands`` variant would extend the joint set.

Example:
    >>> from openral_hal import H1MujocoHAL, H1_DESCRIPTION
    >>> hal = H1MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> state = hal.read_state()  # doctest: +SKIP
    >>> len(state.position) == len(H1_DESCRIPTION.joints) == 19  # doctest: +SKIP
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

__all__ = ["H1_DESCRIPTION", "H1MujocoHAL"]


# ── Canonical joint order ─────────────────────────────────────────────────────
# Matches the menagerie MJCF: 5 (left leg) + 5 (right leg) + 1 (torso)
# + 4 (left arm) + 4 (right arm) = 19 actuated joints.  Verified at
# import time by ``tests/sim/test_h1_hal_mujoco.py::TestMenagerieSchema``.
# The H1 menagerie does NOT use the ``_joint`` suffix convention — we
# mirror the upstream names verbatim so the drift guard catches any
# rename.

_H1_LEFT_LEG_JOINTS: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
)
_H1_RIGHT_LEG_JOINTS: tuple[str, ...] = (
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
_H1_TORSO_JOINTS: tuple[str, ...] = ("torso",)
_H1_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
)
_H1_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)
_H1_JOINT_NAMES: tuple[str, ...] = (
    *_H1_LEFT_LEG_JOINTS,
    *_H1_RIGHT_LEG_JOINTS,
    *_H1_TORSO_JOINTS,
    *_H1_LEFT_ARM_JOINTS,
    *_H1_RIGHT_ARM_JOINTS,
)


# ── Joint limits ─────────────────────────────────────────────────────────────
# Position limits (rad) come from the menagerie MJCF, verbatim.  Effort
# limits use the MJCF's ``ctrlrange`` directly because the H1 menagerie
# explicitly publishes them; this is unlike the G1 MJCF which leaves
# forcerange unset and required us to use published-spec values.

_H1_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "left_hip_yaw": (-0.43, 0.43),
    "left_hip_roll": (-0.43, 0.43),
    "left_hip_pitch": (-1.57, 1.57),
    "left_knee": (-0.26, 2.05),
    "left_ankle": (-0.87, 0.52),
    "right_hip_yaw": (-0.43, 0.43),
    "right_hip_roll": (-0.43, 0.43),
    "right_hip_pitch": (-1.57, 1.57),
    "right_knee": (-0.26, 2.05),
    "right_ankle": (-0.87, 0.52),
    "torso": (-2.35, 2.35),
    "left_shoulder_pitch": (-2.87, 2.87),
    "left_shoulder_roll": (-0.34, 3.11),
    "left_shoulder_yaw": (-1.3, 4.45),
    "left_elbow": (-1.25, 2.61),
    "right_shoulder_pitch": (-2.87, 2.87),
    "right_shoulder_roll": (-3.11, 0.34),
    "right_shoulder_yaw": (-4.45, 1.3),
    "right_elbow": (-1.25, 2.61),
}

# Effort limits straight from the menagerie ctrlrange — the H1
# datasheet's published peak torques (hip 360, knee 360, ankle 40,
# shoulder 40, elbow 18, torso 200 N·m) round to these and the MJCF
# is the authoritative numbers per CLAUDE.md §1.2 ("truth over
# plausibility").
_H1_EFFORT_LIMITS: dict[str, float] = {
    "left_hip_yaw": 200.0,
    "left_hip_roll": 200.0,
    "left_hip_pitch": 200.0,
    "left_knee": 300.0,
    "left_ankle": 40.0,
    "right_hip_yaw": 200.0,
    "right_hip_roll": 200.0,
    "right_hip_pitch": 200.0,
    "right_knee": 300.0,
    "right_ankle": 40.0,
    "torso": 200.0,
    "left_shoulder_pitch": 40.0,
    "left_shoulder_roll": 40.0,
    "left_shoulder_yaw": 18.0,
    "left_elbow": 18.0,
    "right_shoulder_pitch": 40.0,
    "right_shoulder_roll": 40.0,
    "right_shoulder_yaw": 18.0,
    "right_elbow": 18.0,
}

# Velocity limits — the H1 spec sheet lists ~12-15 rad/s for the legs
# and ~18-22 rad/s for the arms.  We halve the spec-sheet peak for
# the SafetyEnvelope-side limit (CLAUDE.md §1.1 — refuse a request
# before it can damage the hardware), grouping by joint kind to keep
# the table compact.
_H1_VELOCITY_LIMITS_BY_GROUP: dict[str, float] = {
    "hip": 6.0,
    "knee": 6.0,
    "ankle": 6.0,
    "torso": 6.0,
    "shoulder": 10.0,
    "elbow": 10.0,
}


def _h1_group(joint_name: str) -> str:
    """Return the kinematic group for *joint_name*."""
    for token in ("hip", "knee", "ankle", "torso", "shoulder", "elbow"):
        if token in joint_name:
            return token
    raise ROSConfigError(f"Unknown H1 joint group for joint {joint_name!r}.")


def _h1_parent_child(joint_name: str) -> tuple[str, str]:
    """Return ``(parent_link, child_link)`` for *joint_name*.

    The names follow the menagerie URDF convention — ``pelvis`` for the
    floating-base root, then ``<side>_<segment>_link`` for the body
    above each joint.  H1 has a simpler skeleton than the G1: each leg
    is a 5-link chain (no separate hip-yaw + hip-roll + hip-yaw
    structure), the waist is one link (``torso_link``), and each arm
    is a 4-link chain.
    """
    prev_link: dict[str, str] = {
        "left_hip_yaw": "pelvis",
        "left_hip_roll": "left_hip_yaw_link",
        "left_hip_pitch": "left_hip_roll_link",
        "left_knee": "left_hip_pitch_link",
        "left_ankle": "left_knee_link",
        "right_hip_yaw": "pelvis",
        "right_hip_roll": "right_hip_yaw_link",
        "right_hip_pitch": "right_hip_roll_link",
        "right_knee": "right_hip_pitch_link",
        "right_ankle": "right_knee_link",
        "torso": "pelvis",
        "left_shoulder_pitch": "torso_link",
        "left_shoulder_roll": "left_shoulder_pitch_link",
        "left_shoulder_yaw": "left_shoulder_roll_link",
        "left_elbow": "left_shoulder_yaw_link",
        "right_shoulder_pitch": "torso_link",
        "right_shoulder_roll": "right_shoulder_pitch_link",
        "right_shoulder_yaw": "right_shoulder_roll_link",
        "right_elbow": "right_shoulder_yaw_link",
    }
    parent = prev_link[joint_name]
    child = f"{joint_name}_link"
    return parent, child


def _h1_joint_specs() -> list[JointSpec]:
    specs: list[JointSpec] = []
    for name in _H1_JOINT_NAMES:
        group = _h1_group(name)
        parent, child = _h1_parent_child(name)
        specs.append(
            JointSpec(
                name=name,
                joint_type=JointType.REVOLUTE,
                parent_link=parent,
                child_link=child,
                position_limits=_H1_POSITION_LIMITS[name],
                velocity_limit=_H1_VELOCITY_LIMITS_BY_GROUP[group],
                effort_limit=_H1_EFFORT_LIMITS[name],
                has_torque_sensor=True,
                actuator_kind="bldc",
            )
        )
    return specs


# ── RobotDescription ─────────────────────────────────────────────────────────

H1_DESCRIPTION = RobotDescription(
    name="h1",
    embodiment_kind=EmbodimentKind.HUMANOID,
    base_frame="pelvis",
    joints=_h1_joint_specs(),
    # No end-effector — wrists aren't actuated on this menagerie variant.
    end_effectors=[],
    capabilities=RobotCapabilities(
        locomotion=["bipedal"],
        can_lift_kg=3.0,
        has_dexterous_hands=False,
        has_force_control=True,
        has_vision=False,
        bimanual=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["h1", "humanoid_everyday_h1"],
        embodiment_tags=["h1", "unitree_h1", "humanoid"],
    ),
    safety=SafetyEnvelope(
        # The H1 is ~1.8 m tall and reaches ~0.8 m horizontally — slightly
        # larger envelope than the G1.  Real enforcement lives in S0 (M2);
        # this is the Python-side belt for contract validation.
        max_ee_speed_m_s=1.5,
        max_joint_speed_factor=0.5,
        max_force_n=150.0,
        max_torque_nm=300.0,  # H1 knees are 300 N·m peak
        deadman_required=True,
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim="openral_hal.h1:H1MujocoHAL", real=None),
    # Floating-base humanoid — same offset arithmetic as G1 (qpos +7,
    # qvel +6).  ``MujocoArmHAL`` derives both from ``floating_base=True``
    # (ADR-0023).
    sim=SimDescription(
        mjcf_uri="robot_descriptions:h1_mj_description",
        floating_base=True,
    ),
)


# ── HAL ──────────────────────────────────────────────────────────────────────


# ── PD gains for the position loop ───────────────────────────────────────────
# The menagerie H1 ships **torque actuators** (``motor`` with
# ``gain=1, bias=0``): writing ``ctrl[i] = x`` applies ``x`` N·m
# directly, NOT "drive joint i to position x".  This is unlike the G1
# / UR / Franka MJCFs which ship ``position`` actuators with an
# internal PD law.  To preserve the HAL contract — every
# ``MujocoArmHAL`` subclass takes position targets — :class:`H1MujocoHAL`
# runs a P + D position loop in software and writes the resulting
# torque to ``ctrl``.  This mirrors what the real ``unitree_sdk2``
# driver does on hardware: the motor-level interface is torque; the
# user-facing interface is position.
#
# Gains are sized so a 1-rad position error saturates the actuator at
# roughly its ``ctrlrange`` limit, with critical-ish damping
# (kv = 0.05 * kp).  These are not the production-quality balance
# gains the S0 cerebellum will eventually use — they are "track a
# joint target with gravity off" gains for contract validation.
_H1_KP_BY_GROUP: dict[str, float] = {
    "hip": 200.0,
    "knee": 300.0,
    "ankle": 40.0,
    "torso": 200.0,
    "shoulder": 40.0,
    "elbow": 18.0,
}
_H1_KV_BY_GROUP: dict[str, float] = {group: 0.05 * kp for group, kp in _H1_KP_BY_GROUP.items()}


def _h1_pd_gains() -> dict[str, tuple[float, float]]:
    """Per-joint ``(kp, kv)`` gains keyed by the canonical joint name."""
    gains: dict[str, tuple[float, float]] = {}
    for name in _H1_JOINT_NAMES:
        group = _h1_group(name)
        gains[name] = (_H1_KP_BY_GROUP[group], _H1_KV_BY_GROUP[group])
    return gains


class H1MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Unitree H1 humanoid (MuJoCo digital twin).

    Drives the 19 actuated joints of the menagerie ``unitree_h1`` MJCF
    through MuJoCo's position-controlled actuators.  Exposes a 19-D
    :class:`openral_core.Action` matching the joint order in
    :data:`H1_DESCRIPTION` (left leg 5 → right leg 5 → torso 1 → left
    arm 4 → right arm 4).

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
            ``robot_descriptions`` (``mujoco_menagerie/unitree_h1/h1.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.  Defaults to ``1``; raise it in tests
            that assert the body has converged at the commanded pose.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time — required for the contract-validation
            tests because the floating base falls otherwise.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import H1MujocoHAL  # doctest: +SKIP
        >>> hal = H1MujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 19 actuated joints  # doctest: +SKIP
        19
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
        """Initialise the H1 HAL; no MuJoCo state is created until ``connect()``.

        All MuJoCo wiring (MJCF URI, floating-base offsets) lives in
        :data:`H1_DESCRIPTION.sim` (ADR-0023).  The software PD gains
        stay here because they are H1-specific cerebellar substitute
        behavior, not arm-data.
        """
        self._init_from_description(
            H1_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
        self._pd_gains: dict[str, tuple[float, float]] = _h1_pd_gains()

    def _per_step_update(self, targets: list[float]) -> None:
        """Run a software PD position loop every ``mj_step``.

        Overrides the base no-op because the H1 MJCF uses ``motor``
        actuators (torque-controlled): writing ``ctrl[i] = x`` applies
        ``x`` N·m directly, not "drive joint i to position x".  We
        recompute the torque every physics step from current state —
        ``tau = kp * (target - q) - kv * dq`` clamped to the
        actuator's ``ctrlrange`` — so the public
        ``Action.joint_targets`` contract stays "position targets in
        radians", the same as every other ``MujocoArmHAL`` subclass,
        even though the underlying actuator is torque-mode.

        This mirrors how the real ``unitree_sdk2`` driver works on
        hardware: motors are torque-controlled at the bus, but the
        user-facing API takes position targets and runs Kp/Kd in the
        driver layer.
        """
        assert self._data is not None
        assert self._model is not None
        for name, target in zip(self._joint_names, targets, strict=True):
            act_idx = self._actuator_index.get(name)
            if act_idx is None:
                continue
            kp, kv = self._pd_gains[name]
            qpos_idx = self._joint_qpos_addr[name]
            qvel_idx = self._joint_qvel_addr[name]
            q = float(self._data.qpos[qpos_idx])
            dq = float(self._data.qvel[qvel_idx])
            tau = kp * (float(target) - q) - kv * dq
            low, high = self._model.actuator_ctrlrange[act_idx]
            self._data.ctrl[act_idx] = max(float(low), min(float(high), tau))

    # The base ``_apply_arm_targets`` (called once before the settle
    # loop) writes the raw position target to ``ctrl``, which is wrong
    # for torque actuators.  We no-op that call and do the real work in
    # ``_per_step_update`` so MuJoCo sees a sensible torque on every
    # step rather than a stale position-as-torque value on step 0.
    def _apply_arm_targets(self, targets: list[float]) -> None:
        """No-op for H1; the per-step PD loop drives the actuators instead."""
        del targets
