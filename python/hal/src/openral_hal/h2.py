"""HAL adapter for the Unitree H2 humanoid (MuJoCo digital twin).

Wraps Unitree's official H2 MJCF (``unitreerobotics/unitree_mujoco`` @
``1eb6642e``, ``unitree_robots/h2/h2_mujoco.xml``, BSD-3-Clause) as an
``openral_hal.HAL`` Protocol implementation. The H2 is in neither
``mujoco_menagerie`` nor ``robot_descriptions``, so the MJCF + its meshes are
fetched at the pinned commit with per-file SHA-256 by
``openral_hal._unitree_h2_assets`` (asset ref ``unitree:h2``).

Like ``G1MujocoHAL`` this is a **digital-twin contract validator**: there is no
S0 balance controller and no public H2 locomotion checkpoint, so the free joint
is **pinned upright** every physics step (ADR-0087 kinematic glide). A
``BODY_TWIST`` action slides the pinned base; ``JOINT_POSITION`` drives the 31
joints. Balance, walking and real hardware are out of scope (``hal.real`` is
``null``; the C++ S0 controller and ``unitree_sdk2`` real HAL are future work).

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
PD loop every ``mj_step`` (``_floating_base.apply_software_pd``) to keep the
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

import time

import structlog
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    Action,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    HalEntrypoints,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
    UrdfAsset,
)

from openral_hal._floating_base import (
    BasePin,
    apply_software_pd,
    capture_base_pin,
    integrate_planar_twist,
    validate_planar_body_twist,
    write_base_pin,
)
from openral_hal._mujoco_arm import _IDLE_STEP_CAP, MujocoArmHAL, _kinematic_group

log = structlog.get_logger(__name__)

__all__ = ["H2_DESCRIPTION", "H2MujocoHAL"]


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
        # BODY_TWIST rides the ADR-0087 kinematic glide (sim-only); there is no
        # public H2 walking checkpoint.
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
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim="openral_hal.h2:H2MujocoHAL", real=None),
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


class H2MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Unitree H2 humanoid (MuJoCo digital twin).

    Exposes a 31-D ``JOINT_POSITION`` action in unitree_sdk2 motor order (left
    leg 6 → right leg 6 → waist 3 → left arm 7 → right arm 7 → head 2) and a
    planar ``BODY_TWIST`` that slides the pinned base (ADR-0087).

    .. warning::

       No balance: the free joint is pinned upright at the stand height (feet
       on ``z = 0``) every physics step. This validates the action contract,
       joint indexing and lifecycle — it is not a humanoid locomotion sim.

    Args:
        mjcf_path: Optional MJCF override; ``None`` resolves ``assets.mjcf``
            (``unitree:h2`` → the pinned official MJCF).
        settle_steps: Physics steps per ``send_action``; ``None`` derives one
            control period from ``description.control_rate_hz``.
        gravity_enabled: ``False`` zeroes gravity at ``connect()``.
        staleness_limit_s: Maximum age of a cached state.
        body_twist_dt_s: Seconds one BODY_TWIST action integrates the glide by.
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

    _step_while_active = False

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        body_twist_dt_s: float = 0.05,
        description: RobotDescription | None = None,
    ) -> None:
        """Initialise the H2 HAL; no MuJoCo state is created until ``connect()``."""
        self._init_from_description(
            description or H2_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
        if body_twist_dt_s <= 0.0:
            raise ROSConfigError("H2MujocoHAL body_twist_dt_s must be > 0.")
        self._body_twist_dt_s = body_twist_dt_s
        self._pd_gains = _h2_pd_gains()
        # Position targets the software PD tracks on every physics step,
        # including idle steps; captured from qpos at connect.
        self._hold_targets: dict[str, float] = {}
        self._base_pin: BasePin | None = None
        self._last_body_twist: tuple[float, float, float, float, float, float] = (0.0,) * 6

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Load the MJCF, lower the base onto ``z = 0`` and hold the default pose.

        The upstream MJCF spawns the pelvis at 1.2 m with the legs straight,
        leaving the foot spheres ~0.16 m in the air; the pin height is lowered
        so the lowest foot-sphere surface touches the ``z = 0`` floor plane.
        """
        super().connect()
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        assert self._model is not None and self._data is not None
        foot_bottoms = [
            float(self._data.geom_xpos[g][2]) - float(self._model.geom_size[g][0])
            for g in range(self._model.ngeom)
            if int(self._model.geom_type[g]) == int(mj.mjtGeom.mjGEOM_SPHERE)
            and mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, self._model.geom_bodyid[g])
            in _FOOT_BODIES
        ]
        if not foot_bottoms:
            super().disconnect()
            raise ROSConfigError(
                f"H2 MJCF has no foot contact sphere on {sorted(_FOOT_BODIES)}; "
                "cannot set the standing height."
            )
        lowest = min(foot_bottoms)
        x, y, yaw, z = capture_base_pin(self._data.qpos)
        self._base_pin = (x, y, yaw, z - lowest)
        write_base_pin(self._data, self._base_pin)
        self._data.ctrl[:] = 0.0
        mj.mj_forward(self._model, self._data)
        self._capture_hold_targets()

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints to ``pose`` and make it the PD hold target."""
        super().reset_to_pose(pose)
        assert self._data is not None
        # The base re-seeds ``ctrl`` from qpos (a position-actuator idiom); on a
        # torque actuator that would apply qpos as N·m, so clear it.
        self._data.ctrl[:] = 0.0
        self._capture_hold_targets()

    # ── Read accessors ───────────────────────────────────────────────────────

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """Current pinned base ``(x, y, yaw)``."""
        self._require_connected("base_pose")
        assert self._base_pin is not None
        x, y, yaw, _z0 = self._base_pin
        return (x, y, yaw)

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """Last commanded base twist (``base_link`` frame); zeroed on non-twist actions."""
        return self._last_body_twist

    # ── Actions ──────────────────────────────────────────────────────────────

    def send_action(self, action: Action) -> None:
        """Route BODY_TWIST to the glide base; JOINT_POSITION to the PD joint path.

        BODY_TWIST integrates ``(vx, vy, wz)`` into the pin over
        ``body_twist_dt_s``, then steps physics with the joints holding their
        last targets. JOINT_POSITION makes the action's last waypoint the hold
        target and steps ``settle_steps`` with PD + pin every step.
        """
        if action.control_mode != ControlMode.BODY_TWIST.value:
            self._last_body_twist = (0.0,) * 6
            super().send_action(action)
            return
        self._require_connected("send_action")
        vx, vy, wz = validate_planar_body_twist(action, type(self).__name__)
        supported = self.description.capabilities.supported_control_modes
        if action.control_mode not in supported:
            raise ROSConfigError(
                f"Action control_mode '{action.control_mode}' is not in "
                f"supported_control_modes {supported} for robot '{self.description.name}'."
            )
        assert self._base_pin is not None
        self._last_body_twist = (vx, vy, 0.0, 0.0, 0.0, wz)
        self._base_pin = integrate_planar_twist(self._base_pin, vx, vy, wz, self._body_twist_dt_s)
        self._step(self._settle_steps)
        self._last_state_time = time.monotonic()
        self._last_action_ns = time.monotonic_ns()
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            control_mode=action.control_mode,
            body_twist=self._last_body_twist,
        )

    def idle_step(self, wall_dt_s: float | None = None) -> bool:
        """HOLD-step with the base pinned and the PD tracking the held pose.

        The base ``idle_step`` only calls ``mj_step``; on torque actuators that
        would replay the last torque open-loop, so this recomputes PD + pin on
        every step. Returns ``False`` (no-op) once disconnected / e-stopped.
        """
        if not self._connected or self._data is None or self._model is None:
            return False
        self._last_body_twist = (0.0,) * 6
        steps = 1
        if wall_dt_s is not None and wall_dt_s > 0:
            timestep = float(self._model.opt.timestep)
            if timestep > 0:
                steps = max(1, min(_IDLE_STEP_CAP, round(wall_dt_s / timestep)))
        self._step(steps)
        self._last_state_time = time.monotonic()
        return True

    # ── MujocoArmHAL hooks ───────────────────────────────────────────────────

    def _validate_action(self, action: Action) -> None:
        if action.control_mode == ControlMode.BODY_TWIST.value:
            validate_planar_body_twist(action, type(self).__name__)
            return
        super()._validate_action(action)

    def _apply_arm_targets(self, targets: list[float]) -> None:
        """Make ``targets`` the PD hold target (``ctrl`` is torque, never a position)."""
        self._hold_targets = dict(zip(self._joint_names, (float(t) for t in targets), strict=True))

    def _per_step_update(self, targets: list[float]) -> None:
        """Re-pin the base and recompute the PD torques before every ``mj_step``."""
        del targets  # ``_apply_arm_targets`` already stored them as the hold target
        self._pin_and_pd()

    # ── Internals ────────────────────────────────────────────────────────────

    def _capture_hold_targets(self) -> None:
        assert self._data is not None
        self._hold_targets = {
            name: float(self._data.qpos[self._joint_qpos_addr[name]]) for name in self._joint_names
        }

    def _pin_and_pd(self) -> None:
        assert self._data is not None and self._model is not None
        assert self._base_pin is not None
        write_base_pin(self._data, self._base_pin)
        apply_software_pd(
            self._model,
            self._data,
            self._hold_targets,
            self._pd_gains,
            self._actuator_index,
            self._joint_qpos_addr,
            self._joint_qvel_addr,
        )

    def _step(self, n: int) -> None:
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        assert self._data is not None and self._model is not None
        for _ in range(n):
            self._pin_and_pd()
            mj.mj_step(self._model, self._data)
