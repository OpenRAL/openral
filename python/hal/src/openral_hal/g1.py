"""HAL adapter for the Unitree G1 humanoid (MuJoCo digital twin).

Wraps the upstream DeepMind ``mujoco_menagerie`` G1 MJCF
(``unitree_g1/g1.xml``, vendored via ``robot_descriptions``) as an
``openral_hal.HAL`` Protocol implementation, extending the
``openral_hal.UR5eHAL`` / ``openral_hal.FrankaPandaHAL`` /
``openral_hal.SO100MujocoHAL`` pattern to a 29-DoF bipedal humanoid.

The default HAL is a **digital-twin contract validator** with the
ADR-0087 kinematic glide. ``s0_controller`` selects a sim-only S0
(cerebellum) controller instead: ``"playground_walk"`` (also
``walking_enabled=True``) is ADR-0089's pinned MuJoCo Playground ONNX
controller on its own policy-tuned MJCF; ``"gr00t_wbc"`` is NVIDIA's GR00T
Decoupled WBC lower-body policy on the stock Menagerie MJCF, which balances
and walks the free-floating twin under gravity while S1 keeps the 14 arm
joints. Both are simulation-only Python policies; the production C++ S0
controller is future work — see `docs/architecture/repo-state-map.html` for
the planned production S0 block; neither may be reused by a real HAL. The
suite validates the 29-DoF joint-position action layout, lifecycle
wiring (``connect → read_state → send_action → estop``), joint
indexing, ``RobotDescription`` round-trip, embodiment/VLA tag plumbing,
and the optional BODY_TWIST-to-walking controller path.

Joint inventory: the menagerie MJCF has 30 joints (29 actuated + 1
floating base, not exposed on ``RobotDescription`` — implicit world
state, not Skill-commanded) and 29 position actuators in a fixed order:

    legs  : 2 x (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    waist : yaw, roll, pitch
    arms  : 2 x (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                 wrist_roll, wrist_pitch, wrist_yaw)

i.e. 12 + 3 + 14 = 29. qpos addresses for the actuated joints are
``7..35`` (first 7 qpos slots are the floating base); actuator indices
are ``0..28``, aligned 1:1 with the joint order above.

The wrist endpoint is a bare joint — this menagerie variant ships no
hand actuators, so there is no gripper to map. A future
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

import functools
import math
from collections.abc import Callable
from typing import Literal, Unpack, get_args

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core.exceptions import ROSCapabilityMismatch, ROSConfigError, ROSSafetyViolation
from openral_core.schemas import (
    Action,
    ActionSpec,
    AssetRefs,
    CameraSimPlacement,
    ControlMode,
    EmbodimentKind,
    HalEntrypoints,
    HalParameters,
    IntrinsicsPinhole,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SensorModality,
    SensorSpec,
    SimDescription,
    UrdfAsset,
)
from openral_core.time_base import physics_steps_per_tick

from openral_hal._floating_base import write_base_pin
from openral_hal._g1_gr00t_wbc import (
    ARM_KD,
    ARM_KP,
    CONTROL_RATE_HZ,
    GR00T_WBC_LICENSE_POSTURE,
    GR00T_WBC_PINNED_SHA,
    S0_DEFAULT_ANGLES,
    S0_JOINT_COUNT,
    S0_KD,
    S0_KP,
    Gr00tWbcAssets,
    Gr00tWbcController,
    ensure_gr00t_wbc_assets,
    position_actuators_to_torque_motors,
)
from openral_hal._g1_walking import G1WalkingController, ensure_g1_walking_assets
from openral_hal._mujoco_arm import MujocoArmHAL, _kinematic_group
from openral_hal._slot_group import joint_position_command, slot_group_targets
from openral_hal.mujoco_twin import (
    MujocoTwinRollout,
    TwinKwargs,
    TwinStepResult,
    _FreeBaseTwin,
    _GlideTwin,
)

log = structlog.get_logger(__name__)

__all__ = ["G1_DESCRIPTION", "G1MujocoHAL", "G1S0Controller"]

#: Sim-only S0 (cerebellum) controller choices for ``G1MujocoHAL``.
G1S0Controller = Literal["none", "playground_walk", "gr00t_wbc"]

# GR00T WBC fall detector: a pelvis below this height (m, world z) or a torso
# tilted further than this from vertical latches the S0 off (bake-off protocol).
_FALL_PELVIS_HEIGHT_M = 0.5
_FALL_TILT_RAD = math.radians(45.0)


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
# Effort limits are the vendor's per-joint ratings from Unitree's own
# ``g1_29dof.urdf`` (github.com/unitreerobotics/unitree_ros @ 267182b8,
# robots/g1_description, ``<limit effort=…>``). The menagerie MJCF
# (unitree_g1/g1.xml @ feadf76d) rates hip_roll at 139 and ankle / waist
# roll+pitch at 50 N·m via ``actuatorfrcrange``; where the two disagree the
# vendor URDF (the lower value) is used, and where the vendor rating is above
# the previous OpenRAL cap (knee 139 -> 88, ankles 35 -> 30) the previous cap
# is kept: a limits correction must never loosen the safety envelope without a
# safety-WG review (CLAUDE.md §3 Safety). They reach the safety kernel as
# per-joint torque caps (``envelope_loader``), further capped by
# ``safety.max_torque_nm``. MuJoCo's ``mj_step`` does not honour them (its
# actuators use ``ctrlrange`` / ``actuatorfrcrange`` from the MJCF instead).

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

# Conservative velocity limits, deliberately far below the vendor ratings
# (Unitree ``g1_29dof.urdf`` @ 267182b8 ``<limit velocity=…>``: hip / waist
# yaw 32, knee 20, ankle / waist roll+pitch 30, shoulder / elbow / wrist roll
# 37, wrist pitch+yaw 22 rad/s) for the safety envelope (CLAUDE.md §1.1 —
# refuse a request before it can damage the hardware).
_G1_VELOCITY_LIMITS_BY_GROUP: dict[str, float] = {
    "hip": 6.0,
    "knee": 6.0,
    "ankle": 6.0,
    "waist": 6.0,
    "shoulder": 6.0,
    "elbow": 6.0,
    "wrist": 10.0,
}
# Per joint (not per group): waist yaw vs roll/pitch and wrist roll vs
# pitch/yaw use different actuators. Keyed by the side-less joint name.
_G1_EFFORT_LIMITS: dict[str, float] = {
    "hip_pitch_joint": 88.0,
    "hip_roll_joint": 88.0,
    "hip_yaw_joint": 88.0,
    "knee_joint": 88.0,
    "ankle_pitch_joint": 30.0,
    "ankle_roll_joint": 30.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 35.0,
    "waist_pitch_joint": 35.0,
    "shoulder_pitch_joint": 25.0,
    "shoulder_roll_joint": 25.0,
    "shoulder_yaw_joint": 25.0,
    "elbow_joint": 25.0,
    "wrist_roll_joint": 25.0,
    "wrist_pitch_joint": 5.0,
    "wrist_yaw_joint": 5.0,
}


_G1_GROUPS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")


def _g1_group(joint_name: str) -> str:
    """Return the kinematic group for *joint_name* (``hip`` / ``knee`` / ``wrist`` / …)."""
    return _kinematic_group(joint_name, _G1_GROUPS, robot="G1")


def _g1_parent_child(joint_name: str) -> tuple[str, str]:
    """Return ``(parent_link, child_link)`` for *joint_name*.

    The names follow the menagerie URDF convention — ``pelvis`` for the
    floating-base root, then ``<side>_<segment>_link`` for the body
    above each joint.  Strict accuracy isn't necessary (these fields
    are descriptive metadata for the JointSpec, not used by the HAL
    contract), but using stable names keeps ``RobotDescription``
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
                effort_limit=_G1_EFFORT_LIMITS[name.removeprefix("left_").removeprefix("right_")],
                has_torque_sensor=False,  # tau_est is current-derived, not a joint torque sensor
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
    # Forward egocentric head camera (ADR-0087).  The menagerie MJCF ships
    # no <camera>, so the generic HAL camera rig splices this onto
    # ``torso_link`` at head height (~1.2 m world when standing), looking
    # forward with a slight downward pitch. ``vla_feature_key`` matches the
    # InternVLA-N1 VLN rSkill's ``sensors_required`` so the skill loader
    # auto-wires it as the policy's egocentric view.
    sensors=[
        SensorSpec(
            name="head",
            modality=SensorModality.RGB,
            frame_id="head_camera",
            parent_frame="torso_link",
            rate_hz=15.0,
            intrinsics=IntrinsicsPinhole(
                width=640, height=480, fx=343.0, fy=343.0, cx=320.0, cy=240.0
            ),
            encoding="rgb8",
            vla_feature_key="observation.images.head",
            sim_placement=CameraSimPlacement(
                parent_body="torso_link",
                pos=(0.06, 0.0, 0.40),
                target=(1.06, 0.0, 0.33),
            ),
        )
    ],
    capabilities=RobotCapabilities(
        locomotion=["bipedal"],
        can_lift_kg=3.0,
        has_dexterous_hands=False,
        has_force_control=True,
        has_vision=True,  # head camera above (rigged into the sim MJCF)
        bimanual=True,
        # BODY_TWIST uses ADR-0087's kinematic glide by default and ADR-0089's
        # pinned walking controller when the deploy scene opts in.
        supported_control_modes=[ControlMode.JOINT_POSITION, ControlMode.BODY_TWIST],
        supported_vla_embodiments=["g1", "humanoid_everyday_g1"],
        # `mobile_base` is the canonical class tag BODY_TWIST nav skills
        # match on (InternVLA-N1 VLN, nav2-navigate-to-pose).
        embodiment_tags=["g1", "unitree_g1", "humanoid", "mobile_base"],
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
        # Walking-policy trained command box; see robots/g1/robot.yaml.
        max_base_linear_speed_m_s=0.5,
        max_base_angular_speed_rad_s=1.0,
        # Provisional (former schema defaults / runner defaults; robots/g1/robot.yaml).
        max_ee_accel_m_s2=1.0,
        contact_force_threshold_n=30.0,
        self_collision_margin_m=0.0,
        starting_pose_max_joint_speed_rad_s=0.5,
        starting_pose_tolerance_rad=0.05,
        # = unitree_hg_ros2_control's own state-freshness bound (FreshStateTracker
        # kMaximumStateAge, isaac_ros_robots @ 520aa94f) and G1MujocoHAL's default.
        joint_state_staleness_limit_s=0.5,
    ),
    sdk_kind="open",
    # Sim: the MuJoCo twin. Real: ``UnitreeHgRealHAL`` over NVIDIA's isaac_ros_robots /
    # isaac_ros_deploy ros2_control stack (S0 whole-body policy in C++); its
    # ``RobotDescription`` is ``G1_REAL_DESCRIPTION`` (this one, ``closed_with_api``),
    # which ``robots/g1/robot.yaml`` mirrors.
    hal=HalEntrypoints(
        sim="openral_hal.g1:G1MujocoHAL",
        real="openral_hal.unitree_real:UnitreeHgRealHAL",
        real_bringup="openral_hal_unitree:real_bringup.launch.py",
        # build_hal drops keys a constructor does not take, so the twin never sees it.
        parameters=HalParameters(
            defaults={"robot_model": "g1", "s0_policy": "gr00t_decoupled_wbc"}
        ),
    ),
    # Floating-base humanoid — qpos is offset by 7 (3 position + 4 quaternion);
    # qvel is offset by 6 (3 linear vel + 3 angular vel).  ``MujocoArmHAL``
    # derives both offsets from ``floating_base=True``.
    assets=AssetRefs(
        urdf=UrdfAsset(ref="rd:g1_description"),
        mjcf="rd:g1_mj_description",
    ),
    sim=SimDescription(
        floating_base=True,
    ),
    # 50 Hz: the rate both sim S0 controllers (Playground walking, GR00T WBC)
    # run their policy at, so one send_action is one S0 control period.
    action_spec=ActionSpec(control_freq_hz=CONTROL_RATE_HZ),
)


# ── HAL ──────────────────────────────────────────────────────────────────────


class _G1WalkTwin(_FreeBaseTwin, _GlideTwin):
    """ADR-0089: the pinned MuJoCo Playground ONNX walker on its policy-tuned MJCF.

    A ``BODY_TWIST`` runs the walking policy for ``body_twist_dt_s`` of physics (one
    control period, ``MujocoTwinRollout``; the base is free while it walks); joint
    commands and idle slices keep the glide pin, recaptured where the walk left the
    base. A tick cannot both walk and carry joint targets: the walking policy writes
    all 29 actuators, so the joint slots would be acknowledged and never applied.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        walking_policy_path: str,
        **kwargs: Unpack[TwinKwargs],
    ) -> None:
        """Bind the twin; ``kwargs`` are ``MujocoTwinRollout``'s."""
        super().__init__(description, **kwargs)
        self._walking_policy_path = walking_policy_path
        self._controller: G1WalkingController | None = None
        # The ``(vx, vy, wz)`` the current command walks, ``None`` when it does not.
        self._walk: tuple[float, float, float] | None = None

    def _on_load(self) -> None:
        """Start from the walker's ``knees_bent`` keyframe with a fresh policy state.

        Raises:
            ROSConfigError: The MJCF lacks the ``knees_bent`` keyframe.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        super()._on_load()
        model, data = self.model, self.data
        assert model is not None and data is not None
        keyframe_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "knees_bent")
        if keyframe_id < 0:
            raise ROSConfigError("G1 walking MJCF is missing the 'knees_bent' keyframe.")
        mj.mj_resetDataKeyframe(model, data, keyframe_id)
        mj.mj_forward(model, data)
        self._controller = G1WalkingController(self._walking_policy_path)
        self._controller.initialize(data)

    def close(self) -> None:
        """Drop the policy with the model."""
        super().close()
        self._controller = None

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints and clear the policy history."""
        super().reset_to_pose(pose)
        assert self._controller is not None
        self._controller.reset_state()

    def _compose(
        self, actions: list[Action]
    ) -> tuple[list[float] | None, tuple[float, float, float] | None, Action]:
        """``MujocoTwinRollout._compose``, refusing a walking tick that carries joint targets.

        Raises:
            ROSConfigError: A tick with a ``BODY_TWIST`` and joint / gripper slots, before
                anything moves (see the class docstring); else as the base.
        """
        kinds = {a.control_mode is ControlMode.BODY_TWIST for a in actions}
        if kinds == {True, False}:
            raise ROSConfigError(
                f"{self.backend}: a playground_walk tick cannot carry a BODY_TWIST and joint "
                "targets: the walking policy drives all 29 actuators while it walks. Send "
                "the twist and the joint command in separate ticks."
            )
        return super()._compose(actions)

    def _command(
        self, targets: list[float] | None, twist: tuple[float, float, float] | None
    ) -> None:
        self._walk = None
        super()._command(targets, twist)

    def _drive_base(self, vx: float, vy: float, wz: float) -> None:
        self._base_pin = None
        self._walk = (vx, vy, wz)

    def _action_steps(self, twist: tuple[float, float, float] | None) -> int:
        """Physics steps of one command: ``body_twist_dt_s`` for a walk, else a period.

        ``body_twist_dt_s`` is the control period unless a wall-paced twin pinned another.
        """
        if twist is None:
            return self.settle_steps
        assert self.model is not None and self.body_twist_dt_s is not None
        return max(1, round(self.body_twist_dt_s / float(self.model.opt.timestep)))

    def _per_step(self) -> None:
        if self._walk is None:
            super()._per_step()
            return
        assert self._controller is not None
        self._controller.apply(self.model, self.data, self._walk)

    def _on_idle(self) -> None:
        # ponytail: pin only while idle; replace with a zero-command
        # standing policy when one is published and validated.
        # The pin is kept: re-capturing it each slice would read qpos one step's
        # sag below it, so a held stand sinks. ``_drive_base`` clears it to walk.
        self._walk = None
        assert self._controller is not None
        self._controller.reset_state()


class _G1Gr00tTwin(_FreeBaseTwin):
    """NVIDIA GR00T Decoupled WBC (sim-only S0) on the stock Menagerie MJCF, no pin.

    The Balance + Walk ONNX policy owns legs + waist (joints 0-14) at 50 Hz; every
    physics step closes a 29-joint PD over torque motors toward the policy's targets
    and the arm targets. ``BODY_TWIST`` sets the held ``(vx, vy, wz)`` command (arm
    commands leave it standing; an idle slice zeroes it); ``JOINT_POSITION`` must name
    only arm joints. A fall (pelvis < 0.5 m or torso tilt > 45°) cuts every torque and
    raises ``ROSSafetyViolation`` until ``reset_to_pose`` / reconnect.
    """

    def __init__(
        self, description: RobotDescription, *, assets: Gr00tWbcAssets, **kwargs: Unpack[TwinKwargs]
    ) -> None:
        """Bind the twin; ``kwargs`` are ``MujocoTwinRollout``'s."""
        super().__init__(description, **kwargs)
        self._assets = assets
        self._policy: Gr00tWbcController | None = None
        n = len(_G1_JOINT_NAMES)
        # Runtime state, reset on load / reset_to_pose.
        self._s0_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._s0_targets: NDArray[np.float64] = S0_DEFAULT_ANGLES.copy()
        self._arm_targets: NDArray[np.float64] = np.zeros(n - S0_JOINT_COUNT)
        self._kp: NDArray[np.float64] = np.concatenate([S0_KP, ARM_KP])
        self._kd: NDArray[np.float64] = np.concatenate([S0_KD, ARM_KD])
        self._decimation = 1
        self._counter = 0
        self.fallen = False
        self._torso_body_id = -1

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _on_load(self) -> None:
        """Turn the loaded twin into a free-floating, torque-driven GR00T WBC robot.

        Raises:
            ROSConfigError: The MJCF's joint / actuator layout is not the
                Menagerie 29-DoF order the policy was trained on, the S0 rate
                does not divide the physics timestep, or there is no ground.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        model, data = self.model, self.data
        assert model is not None and data is not None
        n = len(_G1_JOINT_NAMES)
        layout_ok = (
            model.nu == n
            and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
            and [self.joint_qpos_addr[j] for j in _G1_JOINT_NAMES] == list(range(7, 7 + n))
            and [self.joint_qvel_addr[j] for j in _G1_JOINT_NAMES] == list(range(6, 6 + n))
            and [self.actuator_index[j] for j in _G1_JOINT_NAMES] == list(range(n))
            and all(model.jnt_qposadr[model.actuator_trnid[i, 0]] == 7 + i for i in range(n))
        )
        if not layout_ok:
            raise ROSConfigError(
                "G1 gr00t_wbc needs the Menagerie layout: a free base joint, then the 29 "
                "joints at qpos[7:36] / qvel[6:35] driven by actuators 0..28 in "
                "G1_DESCRIPTION order."
            )
        timestep = float(model.opt.timestep)
        decimation, exact_dt = physics_steps_per_tick(CONTROL_RATE_HZ, timestep)
        if not math.isclose(exact_dt, timestep, rel_tol=1e-9):
            raise ROSConfigError(
                f"G1 gr00t_wbc runs at {CONTROL_RATE_HZ:g} Hz; physics timestep "
                f"{timestep} s does not divide its period."
            )
        self._decimation = decimation
        position_actuators_to_torque_motors(model)
        self._enable_ground_contact()
        self._torso_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso_link")
        if self._torso_body_id < 0:
            raise ROSConfigError("G1 gr00t_wbc MJCF has no 'torso_link' body for fall detection.")
        self._policy = Gr00tWbcController(self._assets)
        self._arm_targets = np.zeros(n - S0_JOINT_COUNT)
        self._reset_state()
        mj.mj_forward(model, data)
        log.info(
            "hal.g1.s0_controller.connected",
            s0_controller="gr00t_wbc",
            control_rate_hz=CONTROL_RATE_HZ,
            physics_dt_s=timestep,
            decimation=decimation,
            integrator=int(model.opt.integrator),
            weights_license=GR00T_WBC_LICENSE_POSTURE.value,
        )

    def close(self) -> None:
        """Drop the policy with the model."""
        super().close()
        self._policy = None

    def reset_to_pose(self, pose: list[float]) -> None:
        """Snap the joints, re-seat the base upright and hand legs + waist back to the policy.

        The free base keeps its ``x, y, yaw`` at the MJCF standing height, the pose's
        arm entries become the arm targets, and the policy history and the fall latch
        clear.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        super().reset_to_pose(pose)
        model, data = self.model, self.data
        assert model is not None and data is not None
        x, y, yaw = self.read_base_pose()
        write_base_pin(data, (x, y, yaw, float(model.qpos0[2])))
        mj.mj_forward(model, data)
        self._arm_targets = np.asarray(pose[S0_JOINT_COUNT:], dtype=np.float64).copy()
        self._reset_state()

    # ── Reads ────────────────────────────────────────────────────────────────

    def read_base_twist(self) -> tuple[float, float, float, float, float, float]:
        """The held velocity command: arm actions leave it, an idle slice zeroes it."""
        vx, vy, wz = self._s0_command
        return (vx, vy, 0.0, 0.0, 0.0, wz)

    # ── Commands ─────────────────────────────────────────────────────────────

    def step_action_group(self, actions: list[Action]) -> TwinStepResult:
        """One S1 tick under the S0, then one control period.

        Raises:
            ROSSafetyViolation: The fall detector has latched (before or during
                this period).
            ROSCapabilityMismatch: The tick commands a leg / waist joint, or is a
                whole-vector joint action (which commands all 29).
            ROSConfigError: An unsupported control mode, a malformed twist /
                joint payload, or a non-finite arm target.
        """
        self._raise_if_fallen()
        result = super().step_action_group(actions)
        self._latch_if_fallen()  # the state the last mj_step left, before success
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            s0_controller="gr00t_wbc",
            s0_command=self._s0_command,
            s0_walk_policy=self._policy.used_walk if self._policy is not None else False,
        )
        self._raise_if_fallen()
        return result

    def _compose(
        self, actions: list[Action]
    ) -> tuple[list[float] | None, tuple[float, float, float] | None, Action]:
        """Split one tick into its ``(vx, vy, wz)`` twist and the new arm targets.

        Raises:
            ROSCapabilityMismatch: A leg / waist joint is commanded, or a joint
                action is whole-vector.
            ROSConfigError: See ``step_action_group``.
        """
        twist, joint_actions = self._split_twist(actions)
        for slot in joint_actions:
            self._validate(slot)  # supported mode + 29-wide row checks
            if not slot.joint_names:
                raise ROSCapabilityMismatch(
                    "G1 gr00t_wbc: a whole-vector JOINT_POSITION action commands all 29 "
                    "joints, but legs + waist are owned by the S0 whole-body controller. "
                    "Name the arm joints the action drives with Action.joint_names "
                    "(ADR-0102)."
                )
        targets = slot_group_targets(joint_actions, self.joint_names) if joint_actions else []
        owned = [
            self.joint_names[i] for i, v in enumerate(targets[:S0_JOINT_COUNT]) if v is not None
        ]
        if owned:
            raise ROSCapabilityMismatch(
                f"G1 gr00t_wbc: joints {owned} are owned by the S0 whole-body controller "
                "(legs + waist); S1 may command only the 14 arm joints."
            )
        arm_targets = self._arm_targets.copy()
        for i, value in enumerate(targets[S0_JOINT_COUNT:]):
            if value is not None:
                arm_targets[i] = value
        if not np.isfinite(arm_targets).all():
            raise ROSConfigError("G1 gr00t_wbc: non-finite arm joint target.")
        row = [*self._s0_targets.tolist(), *arm_targets.tolist()]
        # A slot group applied the whole S0 + arm row; an ungrouped action is its own record.
        applied = joint_position_command(row, actions[0]) if len(actions) > 1 else actions[0]
        return row, twist, applied

    def _command(
        self, targets: list[float] | None, twist: tuple[float, float, float] | None
    ) -> None:
        assert targets is not None
        self._arm_targets = np.asarray(targets[S0_JOINT_COUNT:], dtype=np.float64)
        if twist is not None:
            self._s0_command = twist

    def _on_idle(self) -> None:
        """Idle: zero the velocity command; the Balance policy keeps running."""
        self._s0_command = (0.0, 0.0, 0.0)

    def _per_step(self) -> None:
        """Fall check, 50 Hz policy update and the 29-joint PD torque, before each ``mj_step``.

        The fall is sampled every physics step, not on the policy cadence. After a
        fall every torque is held at zero (no actuation) and the policy no longer runs.
        """
        model, data, policy = self.model, self.data, self._policy
        assert model is not None and data is not None and policy is not None
        self._latch_if_fallen()
        if not self.fallen and self._counter % self._decimation == 0:
            self._s0_targets = policy.step(data, command=self._s0_command)
        self._counter += 1
        if self.fallen:
            data.ctrl[:] = 0.0
        else:
            target = np.concatenate([self._s0_targets, self._arm_targets])
            n = len(target)
            data.ctrl[:] = (
                self._kp * (target - data.qpos[7 : 7 + n]) - self._kd * (data.qvel[6 : 6 + n])
            )

    # ── Internals ────────────────────────────────────────────────────────────

    def _enable_ground_contact(self) -> None:
        """Make the twin's ground plane collide (the free base needs a floor to stand on).

        A composed scene that already brings a colliding world plane is used
        as-is. On the bare Menagerie twin the only plane is the camera rig's
        visual-only ``camrig_floor``; its collision bits are switched on in the
        loaded model (the MJCF on disk is unchanged).

        Raises:
            ROSConfigError: The MJCF has no world plane at all.
        """
        import mujoco as mj  # noqa: PLC0415  # reason: optional sim-only dep

        model = self.model
        assert model is not None
        planes = [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[g] == 0
        ]
        if any(model.geom_contype[g] or model.geom_conaffinity[g] for g in planes):
            return
        if not planes:
            raise ROSConfigError(
                "G1 gr00t_wbc needs a ground plane: the MJCF has none (the camera rig adds "
                "one only when the manifest declares a camera)."
            )
        for g in planes:
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1
        # The broad phase prunes by the per-body OR of its geoms' bits, compiled
        # in; without these the world body still reads "never collides".
        model.body_contype[0] |= 1
        model.body_conaffinity[0] |= 1
        log.info(
            "hal.g1.s0_controller.ground_contact_enabled",
            geoms=[mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in planes],
        )

    def _reset_state(self) -> None:
        assert self._policy is not None
        self._policy.reset()
        self._s0_command = (0.0, 0.0, 0.0)
        self._s0_targets = S0_DEFAULT_ANGLES.copy()
        self._counter = 0
        self.fallen = False

    def _torso_tilt_rad(self) -> float:
        assert self.data is not None
        r22 = float(self.data.xmat[self._torso_body_id][8])
        return math.acos(max(-1.0, min(1.0, r22)))

    def _detect_fall(self) -> bool:
        assert self.data is not None
        return (
            float(self.data.qpos[2]) < _FALL_PELVIS_HEIGHT_M
            or self._torso_tilt_rad() > _FALL_TILT_RAD
        )

    def _latch_if_fallen(self) -> None:
        """Latch a fall the current state shows and cut every torque at once."""
        if self.fallen or not self._detect_fall():
            return
        assert self.data is not None
        self.fallen = True
        self.data.ctrl[:] = 0.0
        log.critical(
            "hal.g1.s0_controller.fall_detected",
            robot=self.description.name,
            pelvis_height_m=round(float(self.data.qpos[2]), 3),
            torso_tilt_deg=round(math.degrees(self._torso_tilt_rad()), 1),
            action="all joint torques cut; reset_to_pose or reconnect to re-arm",
        )

    def _raise_if_fallen(self) -> None:
        if self.fallen:
            raise ROSSafetyViolation(
                f"G1 '{self.description.name}' fell under the gr00t_wbc S0 controller "
                f"(pelvis < {_FALL_PELVIS_HEIGHT_M} m or torso tilt > 45°); joint torques "
                "are cut. reset_to_pose or reconnect to re-arm."
            )


class G1MujocoHAL(MujocoArmHAL):
    """HAL adapter for the Unitree G1 humanoid (MuJoCo digital twin).

    Drives the 29 actuated joints of the menagerie ``unitree_g1`` MJCF
    through MuJoCo's position-controlled actuators.  Exposes a
    29-D ``openral_core.Action`` matching the joint order in
    ``G1_DESCRIPTION`` (left leg 6 → right leg 6 → waist 3 → left
    arm 7 → right arm 7).

    ``s0_controller`` picks how the gravity-on twin stays up:

    * ``"none"`` (default) — ADR-0087's upright kinematic pin (no balance).
    * ``"playground_walk"`` (``walking_enabled=True``) — ADR-0089's pinned
      MuJoCo Playground ONNX walker on its own MJCF; pinned while idle.
    * ``"gr00t_wbc"`` — NVIDIA GR00T Decoupled WBC (Balance + Walk ONNX,
      NVIDIA Open Model License weights) owns legs + waist (joints 0-14) on
      the stock Menagerie MJCF with **no base pin**: ``BODY_TWIST`` sets its
      ``(vx, vy, wz)`` command, ``JOINT_POSITION`` actions that name only arm
      joints (``Action.joint_names``, ADR-0102) are tracked by a joint PD loop,
      and any command on a leg / waist joint is refused with
      ``ROSCapabilityMismatch``. ``idle_step`` keeps the Balance policy running.
      A fall (pelvis < 0.5 m or torso tilt > 45°) cuts every joint torque and
      raises ``ROSSafetyViolation`` until ``reset_to_pose`` / reconnect.

    .. warning::

       Every mode is simulation-only. The S0 policies are not the production
       C++ S0 controller required for hardware and must never drive a real G1.

    Args:
        mjcf_path: Optional override for the MJCF file path.  When
            ``None``, the file is fetched lazily from
            ``robot_descriptions``
            (``mujoco_menagerie/unitree_g1/g1.xml``).
        settle_steps: Number of MuJoCo physics steps performed in
            ``send_action``. ``None`` (default) derives one control period of
            ``description.control_rate_hz`` at ``connect`` (issue #358); an explicit
            count is a pinned, logged override.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time — the joint-convergence contract tests use
            it for determinism; the default glide pin keeps the
            gravity-on twin upright.
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
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
        body_twist_dt_s: float | None = None,
        walking_enabled: bool = False,
        s0_controller: G1S0Controller | None = None,
        description: RobotDescription | None = None,
    ) -> None:
        """Initialise the G1 HAL; no MuJoCo state is created until ``connect()``.

        All wiring (MJCF URI, floating-base offsets) lives in
        ``G1_DESCRIPTION.sim``.

        Args:
            mjcf_path: Optional override for the MJCF file path.
            settle_steps: MuJoCo physics steps per ``send_action``.
            gravity_enabled: When ``False``, gravity is zeroed at connect.
            staleness_limit_s: Maximum age of a cached state.
            body_twist_dt_s: Seconds one BODY_TWIST action integrates the glide
                base by and the Playground walker steps physics for. ``None`` =
                one control period of the twin (0.02 s at the manifest's 50 Hz),
                so a commanded twist advances the base at its commanded speed;
                an explicit value is refused at ``connect`` unless it is that
                period or the twin is wall-paced (``MujocoTwinRollout``).
            walking_enabled: Legacy alias for ``s0_controller="playground_walk"``.
            s0_controller: Sim-only S0 controller (see the class docstring).
                ``None`` = ``"playground_walk"`` when ``walking_enabled`` else
                ``"none"``.
            description: The loaded ``robots/g1/robot.yaml`` manifest
                (threaded by ``build_hal``); ``None`` falls back to the
                in-code ``G1_DESCRIPTION`` mirror.

        Raises:
            ROSConfigError: An unknown ``s0_controller``, one contradicting
                ``walking_enabled``, an S0 controller with gravity off, or the
                pinned S0 assets cannot be fetched / verified.
        """
        if s0_controller is None:
            s0_controller = "playground_walk" if walking_enabled else "none"
        elif walking_enabled and s0_controller != "playground_walk":
            raise ROSConfigError(
                f"G1 walking_enabled=True selects s0_controller='playground_walk' but "
                f"s0_controller={s0_controller!r} was also given; set only one."
            )
        if s0_controller not in get_args(G1S0Controller):
            raise ROSConfigError(
                f"G1 s0_controller={s0_controller!r} is not one of {get_args(G1S0Controller)}."
            )
        if s0_controller != "none" and not gravity_enabled:
            raise ROSConfigError(
                f"G1 s0_controller={s0_controller!r} requires gravity_enabled=True."
            )
        desc = description if description is not None else G1_DESCRIPTION
        if body_twist_dt_s is not None and body_twist_dt_s <= 0.0:
            raise ROSConfigError("G1MujocoHAL body_twist_dt_s must be > 0.")
        twin: Callable[..., MujocoTwinRollout]
        if s0_controller == "playground_walk":
            if mjcf_path is not None:
                raise ROSConfigError(
                    "G1 walking uses its pinned policy-tuned MJCF; mjcf_path cannot override it."
                )
            mjcf_path, walking_policy_path = ensure_g1_walking_assets()
            twin = functools.partial(_G1WalkTwin, walking_policy_path=walking_policy_path)
        elif s0_controller == "gr00t_wbc":
            gr00t_assets = ensure_gr00t_wbc_assets()
            log.info(
                "hal.g1.s0_controller",
                s0_controller=s0_controller,
                upstream=f"NVlabs/GR00T-WholeBodyControl@{GR00T_WBC_PINNED_SHA}",
                code_license="apache-2.0",
                weights_license=GR00T_WBC_LICENSE_POSTURE.value,
                license_text=str(gr00t_assets.license_text),
                sim_only=True,
            )
            twin = functools.partial(_G1Gr00tTwin, assets=gr00t_assets)
        else:
            twin = _GlideTwin
        self._init_from_description(
            desc,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
            body_twist_dt_s=body_twist_dt_s,
            twin=twin,
        )
        self._s0_kind: G1S0Controller = s0_controller

    @property
    def s0_controller(self) -> G1S0Controller:
        """The sim-only S0 controller this HAL runs (``"none"`` = ADR-0087 glide)."""
        return self._s0_kind

    @property
    def s0_fallen(self) -> bool:
        """``True`` once the GR00T WBC fall detector latched (torques are cut)."""
        return isinstance(self._twin, _G1Gr00tTwin) and self._twin.fallen

    _step_while_active = False
