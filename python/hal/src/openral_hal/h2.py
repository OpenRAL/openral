"""HAL adapter for the Unitree H2 humanoid (MuJoCo digital twin).

Wraps Unitree's official H2 MJCF (``unitreerobotics/unitree_mujoco`` @
``1eb6642e``, ``unitree_robots/h2/h2_mujoco.xml``, BSD-3-Clause) as an
``openral_hal.HAL`` Protocol implementation. The H2 is in neither
``mujoco_menagerie`` nor ``robot_descriptions``, so the MJCF + its meshes are
fetched at the pinned commit with per-file SHA-256 by
``openral_hal._unitree_h2_assets`` (asset ref ``unitree:h2``).

By default (``s0_controller="none"``) this is a **digital-twin contract
validator** like ``G1MujocoHAL``: no S0 balance controller, so the free joint is
**pinned upright** every physics step (ADR-0087 kinematic glide); a ``BODY_TWIST``
slides the pinned base and ``JOINT_POSITION`` drives the 31 joints.
``s0_controller="agile_velocity"`` instead runs NVIDIA's WBC-AGILE velocity policy
(the real H2's own S0 export, ``config/h2/policies/agile_velocity.yaml``) as a
**sim-only** S0 on the unpinned free base: it balances and walks under gravity,
``BODY_TWIST`` is its velocity command, and S1 keeps waist yaw, arms and head.
Real hardware is ``openral_hal.unitree_real.UnitreeHgRealHAL`` (its description
``H2_REAL_DESCRIPTION`` is this one with ``sdk_kind: closed_with_api``), whose S0
is the same WBC-AGILE policy in C++.

Joint inventory — 31 actuated joints in **unitree_sdk2 motor-index order**
(``example/h2/low_level/h2_ankle_swing_example.cpp``, ``H2_NUM_MOTOR = 31``),
which is also the MJCF actuator order:

    0-5   left  hip_pitch, hip_roll, hip_yaw, knee, ankle_roll, ankle_pitch
    6-11  right (same)
    12-14 waist_yaw, waist_roll, waist_pitch
    15-21 left  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                wrist_roll, wrist_pitch, wrist_yaw
    22-28 right (same)
    29-30 head_pitch, head_yaw

Note the ankle **roll before pitch** (the G1 is pitch-then-roll) and that the
MJCF *tree* order differs: the head joints sit between waist and arms, so the
qpos / qvel addresses are NOT the default ``7 + i`` / ``6 + i`` — the ``sim``
block carries explicit ``joint_qpos_addr`` / ``joint_qvel_addr`` maps (guarded
against the MJCF by ``tests/sim/test_h2_hal_mujoco.py``). Actuator order is SDK
order, so the default 1:1 actuator map holds. The ankles and waist
roll/pitch are parallel linkages on hardware; the twin models them as serial
joints, which is what the SDK's ``Mode::PR`` (series pitch/roll) exposes.

The MJCF ships ``motor`` (torque) actuators, so ``H2MujocoHAL`` runs a software
PD loop every ``mj_step`` (``mujoco_twin._SoftwarePDTwin``) to keep the
position-target ``Action`` contract — also during ``idle_step``, so a held pose
does not sag under gravity.

Example:
    >>> from openral_hal import H2MujocoHAL, H2_DESCRIPTION
    >>> hal = H2MujocoHAL()  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> len(hal.read_state().position) == len(H2_DESCRIPTION.joints) == 31  # doctest: +SKIP
    True
    >>> hal.disconnect()  # doctest: +SKIP
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, Literal, Unpack, get_args

import numpy as np
import structlog
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError
from openral_core.schemas import (
    Action,
    ActionSpec,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    HalEntrypoints,
    HalParameters,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
    UrdfAsset,
)
from openral_core.time_base import physics_steps_per_tick

from openral_hal._floating_base import capture_base_pin, write_base_pin
from openral_hal._h2_agile_velocity import (
    AGILE_LICENSE_POSTURE,
    LeappVelocityPolicy,
    h2_agile_velocity_config,
)
from openral_hal._mujoco_arm import MujocoArmHAL, _kinematic_group
from openral_hal._slot_group import joint_position_command, slot_group_targets
from openral_hal.mujoco_twin import (
    MujocoTwinRollout,
    TwinKwargs,
    _GlideTwin,
    _S0PolicyTwin,
    _SoftwarePDTwin,
)

log = structlog.get_logger(__name__)

__all__ = ["H2_DESCRIPTION", "H2MujocoHAL", "H2S0Controller"]

#: Sim-only S0 (cerebellum) controller choices for ``H2MujocoHAL``.
H2S0Controller = Literal["none", "agile_velocity"]


# ── Canonical joint order (unitree_sdk2 motor index) ─────────────────────────


def _side(prefix: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}_{n}_joint" for n in names)


_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")
_ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
_H2_JOINT_NAMES: tuple[str, ...] = (
    *_side("left", _LEG),
    *_side("right", _LEG),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *_side("left", _ARM),
    *_side("right", _ARM),
    "head_pitch_joint",
    "head_yaw_joint",
)


# ── Joint limits (verbatim from the official URDF) ───────────────────────────
# Source: unitreerobotics/unitree_ros @ 41dec58c7a3ccb3050243ef16ac2ca74b2b0f9ca,
# robots/h2_description/H2.urdf, ``<limit lower upper effort velocity>``. The
# MJCF ``range`` / ``ctrlrange`` agree (rounded); velocity and effort are the
# URDF's hardware peaks — the SafetyEnvelope's ``max_joint_speed_factor`` is
# where the conservative derating lives.

_H2_LIMITS: dict[str, tuple[float, float, float, float]] = {
    # name: (lower rad, upper rad, effort N·m, velocity rad/s)
    "left_hip_pitch_joint": (-2.4525975, 2.7754225, 360.0, 20.0),
    "left_hip_roll_joint": (-0.467441, 2.1688605, 360.0, 20.0),
    "left_hip_yaw_joint": (-2.827, 2.827, 360.0, 20.0),
    "left_knee_joint": (-0.08725, 2.53025, 360.0, 20.0),
    "left_ankle_roll_joint": (-0.3490658503, 0.296705972783, 19.0, 100.7),
    "left_ankle_pitch_joint": (-1.13446401358, 0.61086523808, 66.88, 28.61),
    "right_hip_pitch_joint": (-2.4525975, 2.7754225, 360.0, 20.0),
    "right_hip_roll_joint": (-2.1688605, 0.467441, 360.0, 20.0),
    "right_hip_yaw_joint": (-2.827, 2.827, 360.0, 20.0),
    "right_knee_joint": (-0.08725, 2.53025, 360.0, 20.0),
    "right_ankle_roll_joint": (-0.296705972783, 0.3490658503, 19.0, 100.7),
    "right_ankle_pitch_joint": (-1.13446401358, 0.61086523808, 66.88, 28.61),
    "waist_yaw_joint": (-1.7453, 1.7453, 120.0, 28.375),
    "waist_roll_joint": (-0.5236, 0.5236, 180.0, 28.375),
    "waist_pitch_joint": (-0.43633, 0.5236, 180.0, 28.375),
    "left_shoulder_pitch_joint": (-2.618, 1.833, 130.0, 21.9),
    "left_shoulder_roll_joint": (-0.517, 2.494, 60.0, 18.7),
    "left_shoulder_yaw_joint": (-2.618, 2.618, 60.0, 18.7),
    "left_elbow_joint": (-0.986, 3.071, 60.0, 18.7),
    "left_wrist_roll_joint": (-2.618, 2.618, 60.0, 18.7),
    "left_wrist_pitch_joint": (-0.576, 0.576, 10.0, 37.7),
    "left_wrist_yaw_joint": (-1.22, 1.22, 10.0, 37.7),
    "right_shoulder_pitch_joint": (-2.618, 1.833, 130.0, 21.9),
    "right_shoulder_roll_joint": (-2.494, 0.517, 60.0, 18.7),
    "right_shoulder_yaw_joint": (-2.618, 2.618, 60.0, 18.7),
    "right_elbow_joint": (-0.986, 3.071, 60.0, 18.7),
    "right_wrist_roll_joint": (-2.618, 2.618, 60.0, 18.7),
    "right_wrist_pitch_joint": (-0.576, 0.576, 10.0, 37.7),
    "right_wrist_yaw_joint": (-1.22, 1.22, 10.0, 37.7),
    "head_pitch_joint": (-0.5236, 0.83775, 50.0, 10.0),
    "head_yaw_joint": (-1.7453, 1.7453, 50.0, 10.0),
}

# URDF tree: (parent_link, child_link) per joint. ``waist_pitch_joint``'s child
# is ``torso_link`` (no ``waist_pitch_link``); the head hangs off the torso.
_H2_PARENT_CHILD: dict[str, tuple[str, str]] = {
    "left_hip_pitch_joint": ("pelvis", "left_hip_pitch_link"),
    "left_hip_roll_joint": ("left_hip_pitch_link", "left_hip_roll_link"),
    "left_hip_yaw_joint": ("left_hip_roll_link", "left_hip_yaw_link"),
    "left_knee_joint": ("left_hip_yaw_link", "left_knee_link"),
    "left_ankle_roll_joint": ("left_knee_link", "left_ankle_roll_link"),
    "left_ankle_pitch_joint": ("left_ankle_roll_link", "left_ankle_pitch_link"),
    "right_hip_pitch_joint": ("pelvis", "right_hip_pitch_link"),
    "right_hip_roll_joint": ("right_hip_pitch_link", "right_hip_roll_link"),
    "right_hip_yaw_joint": ("right_hip_roll_link", "right_hip_yaw_link"),
    "right_knee_joint": ("right_hip_yaw_link", "right_knee_link"),
    "right_ankle_roll_joint": ("right_knee_link", "right_ankle_roll_link"),
    "right_ankle_pitch_joint": ("right_ankle_roll_link", "right_ankle_pitch_link"),
    "waist_yaw_joint": ("pelvis", "waist_yaw_link"),
    "waist_roll_joint": ("waist_yaw_link", "waist_roll_link"),
    "waist_pitch_joint": ("waist_roll_link", "torso_link"),
    "left_shoulder_pitch_joint": ("torso_link", "left_shoulder_pitch_link"),
    "left_shoulder_roll_joint": ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    "left_shoulder_yaw_joint": ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    "left_elbow_joint": ("left_shoulder_yaw_link", "left_elbow_link"),
    "left_wrist_roll_joint": ("left_elbow_link", "left_wrist_roll_link"),
    "left_wrist_pitch_joint": ("left_wrist_roll_link", "left_wrist_pitch_link"),
    "left_wrist_yaw_joint": ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    "right_shoulder_pitch_joint": ("torso_link", "right_shoulder_pitch_link"),
    "right_shoulder_roll_joint": ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    "right_shoulder_yaw_joint": ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    "right_elbow_joint": ("right_shoulder_yaw_link", "right_elbow_link"),
    "right_wrist_roll_joint": ("right_elbow_link", "right_wrist_roll_link"),
    "right_wrist_pitch_joint": ("right_wrist_roll_link", "right_wrist_pitch_link"),
    "right_wrist_yaw_joint": ("right_wrist_pitch_link", "right_wrist_yaw_link"),
    "head_pitch_joint": ("torso_link", "head_pitch_link"),
    "head_yaw_joint": ("head_pitch_link", "head_yaw_link"),
}


def _h2_joint_specs() -> list[JointSpec]:
    specs: list[JointSpec] = []
    for name in _H2_JOINT_NAMES:
        lower, upper, effort, velocity = _H2_LIMITS[name]
        parent, child = _H2_PARENT_CHILD[name]
        specs.append(
            JointSpec(
                name=name,
                joint_type=JointType.REVOLUTE,
                parent_link=parent,
                child_link=child,
                position_limits=(lower, upper),
                velocity_limit=velocity,
                effort_limit=effort,
                has_torque_sensor=False,  # tau_est is current-derived, not a joint torque sensor
                actuator_kind="bldc",
            )
        )
    return specs


# ── MJCF addressing ──────────────────────────────────────────────────────────
# ``h2_mujoco.xml`` tree order: free joint (qpos 0-6), legs 7-18, waist 19-21,
# head 22-23, left arm 24-30, right arm 31-37 — the head sits between waist and
# arms, so description index i (SDK order) does not map to qpos 7 + i. qvel is
# qpos - 1 for every hinge (the free joint has 7 qpos but 6 qvel).
_H2_MJCF_TREE_ORDER: tuple[str, ...] = (
    *_H2_JOINT_NAMES[:15],  # legs + waist
    *_H2_JOINT_NAMES[29:],  # head
    *_H2_JOINT_NAMES[15:29],  # left arm, right arm
)
_H2_QPOS_ADDR: dict[str, int] = {name: 7 + i for i, name in enumerate(_H2_MJCF_TREE_ORDER)}


# ── RobotDescription ─────────────────────────────────────────────────────────

H2_DESCRIPTION = RobotDescription(
    name="h2",
    embodiment_kind=EmbodimentKind.HUMANOID,
    base_frame="pelvis",
    joints=_h2_joint_specs(),
    # The official assets ship bare wrists (``*_hand_link`` is a fixed frame,
    # not an actuated hand), so there is no gripper to map.
    end_effectors=[],
    # No sensors declared: the assets prove only frames (URDF ``camera``,
    # ``imu_in_torso``, ``imu_in_pelvis``; MJCF sites ``imu`` / ``secondary_imu``)
    # — the camera / LiDAR model, intrinsics and rates are unverified.
    capabilities=RobotCapabilities(
        locomotion=["bipedal"],
        has_dexterous_hands=False,
        has_force_control=True,
        has_vision=False,
        bimanual=True,
        # BODY_TWIST: the ADR-0087 kinematic glide in sim; on real hardware the S0
        # AGILE velocity policy's velocity command.
        supported_control_modes=[ControlMode.JOINT_POSITION, ControlMode.BODY_TWIST],
        embodiment_tags=["h2", "unitree_h2", "humanoid", "mobile_base"],
    ),
    safety=SafetyEnvelope(
        # Conservative Python-side belt for the contract-validation twin; the
        # S0 controller (future) is the real enforcer. See robots/h2/robot.yaml.
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.3,
        max_force_n=150.0,
        max_torque_nm=360.0,
        deadman_required=True,
        max_base_linear_speed_m_s=0.5,
        max_base_angular_speed_rad_s=0.5,
        # Provisional real-path values (the G1's; not measured on an H2) — see the manifest.
        max_ee_accel_m_s2=1.0,
        contact_force_threshold_n=30.0,
        self_collision_margin_m=0.0,
        starting_pose_max_joint_speed_rad_s=0.5,
        starting_pose_tolerance_rad=0.05,
        # = unitree_hg_ros2_control's FreshStateTracker kMaximumStateAge (isaac_ros_robots
        # @ 520aa94f) and H2MujocoHAL's default.
        joint_state_staleness_limit_s=0.5,
    ),
    sdk_kind="open",
    # Sim: the MuJoCo twin. Real: ``UnitreeHgRealHAL`` over NVIDIA's ros2_control stack;
    # its ``RobotDescription`` is ``H2_REAL_DESCRIPTION`` (``closed_with_api``), which
    # ``robots/h2/robot.yaml`` mirrors.
    hal=HalEntrypoints(
        sim="openral_hal.h2:H2MujocoHAL",
        real="openral_hal.unitree_real:UnitreeHgRealHAL",
        real_bringup="openral_hal_unitree:real_bringup.launch.py",
        # build_hal drops keys a constructor does not take, so the twin never sees them.
        parameters=HalParameters(defaults={"robot_model": "h2", "s0_policy": "agile_velocity"}),
    ),
    # 50 Hz: the S0 AGILE policy's rate (manifest `action_spec`).
    action_spec=ActionSpec(control_freq_hz=50.0),
    assets=AssetRefs(urdf=UrdfAsset(ref="unitree:h2"), mjcf="unitree:h2"),
    sim=SimDescription(
        floating_base=True,
        joint_qpos_addr=_H2_QPOS_ADDR,
        joint_qvel_addr={name: addr - 1 for name, addr in _H2_QPOS_ADDR.items()},
    ),
)


# ── Software PD gains ────────────────────────────────────────────────────────
# Body joints: unitree_rl_mjlab @ 1425b15f73bd4095f0df53709d7c389c3eb9e790
# (Apache-2.0), src/assets/robots/unitree_h2/h2_constants.py — the
# ``BuiltinPositionActuatorCfg`` stiffness / damping its Unitree-H2 velocity
# tasks are trained with on an H2 MJCF with the same joint set. That config
# covers 29 joints (no head), so the two head joints take Kp=30 / Kd=1 from
# unitree_sdk2 @ 8e1b927586712edd6a4c29d50dd52dff1e163439 (BSD-3-Clause),
# example/h2/low_level/h2_ankle_swing_example.cpp (motor indices 29-30).
_H2_PD_BY_GROUP: dict[str, tuple[float, float]] = {
    "hip": (200.0, 4.0),  # H2_ACTUATOR_LEG
    "knee": (200.0, 4.0),  # H2_ACTUATOR_LEG
    "ankle": (40.0, 2.0),  # H2_ACTUATOR_ANKLE_ROLL / _ANKLE_PITCH
    "waist": (150.0, 3.0),  # H2_ACTUATOR_WAIST
    "shoulder": (40.0, 2.0),  # H2_ACTUATOR_ARM
    "elbow": (40.0, 2.0),  # H2_ACTUATOR_ARM
    "head": (30.0, 1.0),  # unitree_sdk2 Kp / Kd, indices 29-30
}
_H2_WRIST_PD: dict[str, tuple[float, float]] = {
    "wrist_roll": (40.0, 2.0),  # H2_ACTUATOR_ARM covers ``.*_wrist_roll.*``
    "wrist_pitch": (20.0, 1.0),  # H2_ACTUATOR_WRIST
    "wrist_yaw": (20.0, 1.0),  # H2_ACTUATOR_WRIST
}
_H2_GROUPS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist", "head")


def _h2_pd_gains() -> dict[str, tuple[float, float]]:
    """Per-joint ``(kp, kd)`` keyed by the canonical joint name."""
    gains: dict[str, tuple[float, float]] = {}
    for name in _H2_JOINT_NAMES:
        group = _kinematic_group(name, _H2_GROUPS, robot="H2")
        if group == "wrist":
            axis = next(k for k in _H2_WRIST_PD if k in name)
            gains[name] = _H2_WRIST_PD[axis]
        else:
            gains[name] = _H2_PD_BY_GROUP[group]
    return gains


# Bodies whose contact spheres define the stand height (MJCF foot spheres).
_FOOT_BODIES = ("left_ankle_pitch_link", "right_ankle_pitch_link")


# ── HAL ──────────────────────────────────────────────────────────────────────


class _H2Twin(_GlideTwin, _SoftwarePDTwin):
    """The H2 twin: the ADR-0087 glide pin and the software PD, before every physics step.

    The MJCF ships ``motor`` (torque) actuators, so a command's target row becomes the
    PD hold target (``_SoftwarePDTwin``; the loaded pose is held from ``load`` on) and
    every physics step — actions and idle slices alike — re-pins the base, then
    recomputes the PD torques. A ``BODY_TWIST`` slides the pin (``_GlideTwin``). The
    pin is lowered at load so the feet stand on ``z = 0`` and, as on the G1, kept across
    a ``reset_to_pose`` (which moves no free joint).
    """

    hold_pose_on_load = True

    def _on_load(self) -> None:
        """Hold the default pose and lower the base onto ``z = 0``.

        The upstream MJCF spawns the pelvis at 1.2 m with the legs straight,
        leaving the foot spheres ~0.16 m in the air; the pin height is lowered
        so the lowest foot-sphere surface touches the ``z = 0`` floor plane.

        Raises:
            ROSConfigError: The MJCF has no foot contact sphere.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        super()._on_load()
        model, data = self.model, self.data
        assert model is not None and data is not None
        x, y, yaw, z = capture_base_pin(data.qpos)
        self._base_pin = (x, y, yaw, z - _lowest_foot_z(model, data))
        write_base_pin(data, self._base_pin)
        data.ctrl[:] = 0.0
        mj.mj_forward(model, data)


def _lowest_foot_z(model: Any, data: Any) -> float:  # noqa: ANN401  # reason: mujoco handles
    """World z of the lowest foot contact-sphere surface (after ``mj_forward``).

    Raises:
        ROSConfigError: The MJCF has no foot contact sphere.
    """
    import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

    foot_bottoms = [
        float(data.geom_xpos[g][2]) - float(model.geom_size[g][0])
        for g in range(model.ngeom)
        if int(model.geom_type[g]) == int(mj.mjtGeom.mjGEOM_SPHERE)
        and mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) in _FOOT_BODIES
    ]
    if not foot_bottoms:
        raise ROSConfigError(
            f"H2 MJCF has no foot contact sphere on {sorted(_FOOT_BODIES)}; "
            "cannot set the standing height."
        )
    return min(foot_bottoms)


# The AGILE fall guard: the export's training terminates below DEFAULT_PELVIS_HEIGHT
# (0.915 m) - 0.3 (WBC-AGILE h2/velocity_history_env_cfg.py); the twin stands at ~0.93 m.
_AGILE_FALL_PELVIS_HEIGHT_M = 0.6


class _H2AgileTwin(_S0PolicyTwin, _SoftwarePDTwin):
    """NVIDIA WBC-AGILE velocity policy (sim-only S0) on the free-base H2 twin, no pin.

    The policy owns 14 joints (12 legs + waist roll/pitch) at its 50 Hz: every policy
    tick replaces their software-PD targets **and** kp/kd with the policy's outputs
    (targets clamped to the MJCF joint ranges, each clamp counted and its first one
    per joint logged); the other 17 (waist yaw, arms, head) track S1 joint targets
    with the gains the policy was trained with. ``BODY_TWIST`` sets the held velocity
    command (joint actions leave it, an idle slice zeroes it); ``JOINT_POSITION`` must
    name only non-policy joints. A fall (pelvis < 0.6 m or torso tilt > 45°) cuts every
    torque and refuses commands until ``reset_to_pose`` / reconnect.
    """

    s0_name = "agile_velocity"
    fall_pelvis_height_m = _AGILE_FALL_PELVIS_HEIGHT_M
    hold_pose_on_load = True

    def __init__(
        self,
        description: RobotDescription,
        *,
        policy: LeappVelocityPolicy,
        **kwargs: Unpack[TwinKwargs],
    ) -> None:
        """Bind the twin; ``kwargs`` are ``MujocoTwinRollout``'s."""
        gains = {
            name: (float(policy.default_kp[name]), float(policy.default_kd[name]))
            for name in _H2_JOINT_NAMES
        }
        super().__init__(description, pd_gains=gains, **kwargs)
        self._policy = policy
        self._decimation = 1
        self._counter = 0
        #: Policy ticks whose target for a joint was clamped to its MJCF range, by joint.
        self.clamp_counts: dict[str, int] = {}

    def _on_load(self) -> None:
        """Stand the twin in the policy's default pose on a colliding floor and bind the policy.

        Raises:
            ROSConfigError: The policy rate does not divide the physics timestep, a policy
                joint is missing from the twin, or there is no foot sphere / torso body.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        model, data = self.model, self.data
        assert model is not None and data is not None
        timestep = float(model.opt.timestep)
        decimation, exact_dt = physics_steps_per_tick(self._policy.frequency_hz, timestep)
        if not math.isclose(exact_dt, timestep, rel_tol=1e-9):
            raise ROSConfigError(
                f"H2 agile_velocity runs at {self._policy.frequency_hz:g} Hz; physics "
                f"timestep {timestep} s does not divide its period."
            )
        self._decimation = decimation
        missing = set(self._policy.obs_joint_names) - set(self.joint_names)
        if missing:
            raise ROSConfigError(f"H2 agile_velocity: twin lacks policy joints {sorted(missing)}.")
        self._enable_ground_contact()
        self._bind_torso()
        self._stand_default_pose()
        super()._on_load()  # _SoftwarePDTwin: index arrays; holds the default pose
        driven = [self.joint_names[r] for r in self._pd_rows]
        self._s0_idx = np.array(
            [driven.index(n) for n in self._policy.action_joint_names], dtype=np.intp
        )
        self._obs_qpos = np.array(
            [self.joint_qpos_addr[n] for n in self._policy.obs_joint_names], dtype=np.intp
        )
        self._obs_qvel = np.array(
            [self.joint_qvel_addr[n] for n in self._policy.obs_joint_names], dtype=np.intp
        )
        jids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in self._policy.action_joint_names
        ]
        self._s0_low = np.asarray(model.jnt_range, dtype=np.float64)[jids, 0]
        self._s0_high = np.asarray(model.jnt_range, dtype=np.float64)[jids, 1]
        self._reset_state()
        log.info(
            "hal.h2.s0_controller.connected",
            s0_controller=self.s0_name,
            control_rate_hz=self._policy.frequency_hz,
            physics_dt_s=timestep,
            decimation=decimation,
            weights_license=AGILE_LICENSE_POSTURE.value,
        )

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints, re-seat the base upright on the floor, re-arm the policy.

        The base keeps its ``x, y, yaw``; the pose's non-policy entries become their PD
        targets, the policy's own are replaced at its next tick; the policy history, the
        velocity command and the fall latch clear.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        super().reset_to_pose(pose)
        model, data = self.model, self.data
        assert model is not None and data is not None
        x, y, yaw = self.read_base_pose()
        write_base_pin(data, (x, y, yaw, float(data.qpos[2])))
        mj.mj_forward(model, data)
        data.qpos[2] -= _lowest_foot_z(model, data)
        mj.mj_forward(model, data)
        self._reset_state()

    def hold_present_pose(self) -> None:
        """Hold the present pose (``_SoftwarePDTwin``) and stop the velocity command."""
        super().hold_present_pose()
        self._s0_command = (0.0, 0.0, 0.0)

    def _compose(
        self, actions: list[Action]
    ) -> tuple[list[float] | None, tuple[float, float, float] | None, Action]:
        """Split one tick into its ``(vx, vy, wz)`` twist and a full row of PD targets.

        Raises:
            ROSCapabilityMismatch: A policy joint is commanded, or a joint action is
                whole-vector.
            ROSConfigError: An unsupported mode, a malformed payload, a non-finite target.
        """
        twist, joint_actions = self._split_twist(actions)
        for slot in joint_actions:
            self._validate(slot)
            if not slot.joint_names:
                raise ROSCapabilityMismatch(
                    "H2 agile_velocity: a whole-vector JOINT_POSITION action commands all 31 "
                    "joints, but legs + waist roll/pitch belong to the S0 policy. Name the "
                    "joints the action drives with Action.joint_names (ADR-0102)."
                )
        targets = slot_group_targets(joint_actions, self.joint_names) if joint_actions else []
        s0 = set(self._policy.action_joint_names)
        owned = [
            self.joint_names[i]
            for i, v in enumerate(targets)
            if v is not None and self.joint_names[i] in s0
        ]
        if owned:
            raise ROSCapabilityMismatch(
                f"H2 agile_velocity: joints {owned} are owned by the S0 policy (legs + waist "
                "roll/pitch); S1 may command only waist yaw, the arms and the head."
            )
        assert self._pd_targets is not None
        row = np.zeros(len(self.joint_names))
        row[self._pd_rows] = self._pd_targets
        for i, value in enumerate(targets):
            if value is not None:
                row[i] = value
        if not np.isfinite(row).all():
            raise ROSConfigError("H2 agile_velocity: non-finite joint target.")
        applied = (
            joint_position_command(row.tolist(), actions[0]) if len(actions) > 1 else actions[0]
        )
        return row.tolist(), twist, applied

    def _command(
        self, targets: list[float] | None, twist: tuple[float, float, float] | None
    ) -> None:
        if targets is not None:
            self._apply_targets(targets)
        if twist is not None:
            self._s0_command = twist

    def _per_step(self) -> None:
        """Fall check, the 50 Hz policy tick (targets + gains), then the software PD.

        After a fall every torque stays at zero and the policy no longer runs.
        """
        data = self.data
        assert data is not None and self._pd_targets is not None
        self._latch_if_fallen()
        if self.fallen:
            data.ctrl[:] = 0.0
            return
        if self._counter % self._decimation == 0:
            self._policy_tick()
        self._counter += 1
        super()._per_step()

    def _policy_tick(self) -> None:
        data = self.data
        assert data is not None and self._pd_targets is not None
        w, x, y, z = (float(v) for v in data.qpos[3:7])
        raw, kp, kd = self._policy.step(
            ang_vel_b=np.asarray(data.qvel[3:6]),  # free-joint angular velocity: body frame
            quat_xyzw=np.array([x, y, z, w]),
            command=self._s0_command,
            joint_pos=np.asarray(data.qpos[self._obs_qpos]),
            joint_vel=np.asarray(data.qvel[self._obs_qvel]),
        )
        target = np.clip(raw, self._s0_low, self._s0_high)
        for j in np.flatnonzero(target != raw):
            name = self._policy.action_joint_names[j]
            self.clamp_counts[name] = self.clamp_counts.get(name, 0) + 1
            if self.clamp_counts[name] == 1:
                log.warning(
                    "hal.h2.s0_controller.target_clamped",
                    joint=name,
                    target_rad=round(float(raw[j]), 4),
                    limit_rad=round(float(target[j]), 4),
                    note="policy target past the MJCF joint range; clamped (counted per tick)",
                )
        self._pd_targets[self._s0_idx] = target
        self._pd_kp[self._s0_idx] = kp
        self._pd_kd[self._s0_idx] = kd

    def _stand_default_pose(self) -> None:
        """Every joint at the policy's default, base upright with the feet on ``z = 0``."""
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        model, data = self.model, self.data
        assert model is not None and data is not None
        for name in self.joint_names:
            data.qpos[self.joint_qpos_addr[name]] = float(self._policy.default_joint_pos[name])
        data.qvel[:] = 0.0
        x, y, yaw, z = capture_base_pin(data.qpos)
        write_base_pin(data, (x, y, yaw, z))
        mj.mj_forward(model, data)
        data.qpos[2] -= _lowest_foot_z(model, data)
        data.ctrl[:] = 0.0
        mj.mj_forward(model, data)

    def _reset_state(self) -> None:
        self._policy.reset()
        self._s0_command = (0.0, 0.0, 0.0)
        self._counter = 0
        self.fallen = False
        self.clamp_counts = {}


class H2MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Unitree H2 humanoid (MuJoCo digital twin).

    Exposes a 31-D ``JOINT_POSITION`` action in unitree_sdk2 motor order (left
    leg 6 → right leg 6 → waist 3 → left arm 7 → right arm 7 → head 2) and a
    planar ``BODY_TWIST``. ``s0_controller`` picks how the gravity-on twin stays up:

    * ``"none"`` (default) — ``_H2Twin``: the free joint is pinned upright at the
      stand height (feet on ``z = 0``) every physics step and a ``BODY_TWIST``
      slides the pin (ADR-0087). No balance: a contract validator.
    * ``"agile_velocity"`` — ``_H2AgileTwin``: NVIDIA WBC-AGILE (the pinned,
      SHA-256-checked Hub export; NVIDIA Open Model License weights) owns legs +
      waist roll/pitch at 50 Hz with **no pin**; its per-joint targets (clamped to
      the MJCF ranges) and kp/kd drive the software PD of those 14 joints.
      ``BODY_TWIST`` sets its ``(vx, vy, wz)`` command (zero = stand), named
      ``JOINT_POSITION`` actions drive waist yaw / arms / head with the policy's
      training gains, and a command on a policy joint raises
      ``ROSCapabilityMismatch``. A fall (pelvis < 0.6 m or torso tilt > 45°) cuts
      every torque and raises ``ROSSafetyViolation`` until ``reset_to_pose`` /
      reconnect. Simulation only — on hardware the same export runs in C++.

    Args:
        mjcf_path: Optional MJCF override; ``None`` resolves ``assets.mjcf``
            (``unitree:h2`` → the pinned official MJCF).
        settle_steps: Physics steps per ``send_action``; ``None`` derives one
            control period from ``description.control_rate_hz``.
        gravity_enabled: ``False`` zeroes gravity at ``connect()``.
        staleness_limit_s: Maximum age of a cached state.
        body_twist_dt_s: Seconds one BODY_TWIST action integrates the glide by.
            ``None`` (default) is one control period (``MujocoTwinRollout``): 20 ms at
            the manifest's 50 Hz.
        s0_controller: Sim-only S0 controller (see above); needs ``gravity_enabled``.
        description: The loaded ``robots/h2/robot.yaml`` (threaded by
            ``build_hal``); ``None`` uses the in-code ``H2_DESCRIPTION`` mirror.

    Example:
        >>> from openral_hal import H2MujocoHAL  # doctest: +SKIP
        >>> hal = H2MujocoHAL()  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> len(hal.read_state().position)  # doctest: +SKIP
        31
        >>> hal.disconnect()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        body_twist_dt_s: float | None = None,
        s0_controller: H2S0Controller = "none",
        description: RobotDescription | None = None,
    ) -> None:
        """Initialise the H2 HAL; no MuJoCo state is created until ``connect()``.

        Raises:
            ROSConfigError: A non-positive ``body_twist_dt_s``, an unknown
                ``s0_controller``, an S0 controller with gravity off, or the AGILE bundle
                is not downloaded / fails its SHA-256 pins.
        """
        if body_twist_dt_s is not None and not body_twist_dt_s > 0.0:  # NaN too
            raise ROSConfigError("H2MujocoHAL body_twist_dt_s must be > 0.")
        if s0_controller not in get_args(H2S0Controller):
            raise ROSConfigError(
                f"H2 s0_controller={s0_controller!r} is not one of {get_args(H2S0Controller)}."
            )
        if s0_controller != "none" and not gravity_enabled:
            raise ROSConfigError(
                f"H2 s0_controller={s0_controller!r} requires gravity_enabled=True."
            )
        twin: Callable[..., MujocoTwinRollout]
        if s0_controller == "agile_velocity":
            leapp = h2_agile_velocity_config()
            log.info(
                "hal.h2.s0_controller",
                s0_controller=s0_controller,
                bundle=str(leapp.parent),
                code_license="apache-2.0",
                weights_license=AGILE_LICENSE_POSTURE.value,
                sim_only=True,
            )
            twin = functools.partial(_H2AgileTwin, policy=LeappVelocityPolicy(leapp))
        else:
            twin = functools.partial(_H2Twin, pd_gains=_h2_pd_gains())
        self._s0_kind: H2S0Controller = s0_controller
        self._init_from_description(
            description or H2_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            body_twist_dt_s=body_twist_dt_s,
            twin=twin,
        )

    @property
    def s0_controller(self) -> H2S0Controller:
        """The sim-only S0 controller this HAL runs (``"none"`` = ADR-0087 glide)."""
        return self._s0_kind

    @property
    def s0_fallen(self) -> bool:
        """``True`` once the AGILE fall detector latched (torques are cut)."""
        return isinstance(self._twin, _H2AgileTwin) and self._twin.fallen

    @property
    def s0_clamp_counts(self) -> dict[str, int]:
        """Policy ticks whose target for each joint was clamped to its MJCF range."""
        return dict(self._twin.clamp_counts) if isinstance(self._twin, _H2AgileTwin) else {}
