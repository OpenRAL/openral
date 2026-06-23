"""HAL adapter for the Enactic OpenArm v2 bimanual humanoid arm (MuJoCo digital twin).

The Enactic OpenArm is an 8-DoF (7 revolute arm + 1 parallel-jaw
gripper) per-side open-hardware humanoid arm.  This HAL drives the
upstream ``enactic/openarm_mujoco`` **v2** bimanual MJCF (see
:mod:`openral_hal._openarm_v2_assets`); v2 replaces v1's
torque-mode arm motors with native ``<position>`` actuators carrying
per-class PD gains, fixes v1's asymmetric LEFT-finger gain bug, and
collapses the two-finger-per-side gripper to a single driven joint
with a kinematic equality constraint coupling the follower finger.

That upstream cleanup deletes a meaningful pile of HAL workaround
code that the v1 adapter required (software PD loop sized from
``forcerange``, ``ctrllimited`` override, asymmetric-gain
compensation, two-finger averaging) — see git history for context.
This adapter is therefore a thin :class:`HALBase` subclass: read /
write the 16-element action vector directly into MuJoCo's 16
position-actuator ``ctrl`` slots and let the MJCF's own PD law
handle dynamics.

What this is — and what it isn't
--------------------------------
Like the SO-100 / ALOHA / G1 / H1 / Rizon-4 twins, this HAL is a
**digital-twin contract validator** (CLAUDE.md §1.11): if the sim
tests pass, the 16-DoF action layout, lifecycle, joint indexing,
and ``RobotDescription`` round-trip are guaranteed to match what a
future ``OpenArmRealHAL`` (most likely wrapping lerobot's upstream
OpenArm driver — see ``huggingface.co/docs/lerobot/openarm``) will
see when the physical arm arrives.  The only remaining failure
surfaces are at the lerobot / CAN-FD driver layer (HIL territory).

Action layout
-------------
16-DoF :class:`openral_core.Action` with the same shape as
:class:`AlohaHAL` (just one extra arm joint per side, plus a
hinge-mode gripper instead of a prismatic one):

* ``target[0:7]``   — left arm joints (radians)
* ``target[7]``     — left gripper position (rad, ``[0, 0.7854]``)
* ``target[8:15]``  — right arm joints (radians)
* ``target[15]``    — right gripper position (rad, ``[-0.7854, 0]``)

The asymmetric gripper ctrlranges (left positive, right negative)
come from the v2 MJCF — the mechanism mirrors physically and the
upstream definitions reflect that.  Each gripper command drives
*one* finger actuator; the second finger per side follows via the
MJCF's ``<equality>`` constraint and does not need a separate
command.

Example:
    >>> from openral_hal import OpenArmMujocoHAL, OPENARM_DESCRIPTION
    >>> hal = OpenArmMujocoHAL(gravity_enabled=False)  # doctest: +SKIP
    >>> hal.connect()  # doctest: +SKIP
    >>> state = hal.read_state()  # doctest: +SKIP
    >>> len(state.position) == len(OPENARM_DESCRIPTION.joints) == 16  # doctest: +SKIP
    True
    >>> hal.disconnect()  # doctest: +SKIP

.. note::

   Once ``robot_descriptions`` bumps its ``enactic/openarm_mujoco``
   pin past v2's introduction (PR #19), this module should drop
   :func:`openral_hal._openarm_v2_assets.ensure_openarm_v2_mjcf`
   and resolve the MJCF the same way every other sim HAL does
   (``from robot_descriptions import openarm_v2_mj_description``).
   Tracked at the call site in :func:`_openarm_mjcf_path`.
"""

from __future__ import annotations

from openral_core.schemas import (
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    GripperReadMode,
    GripperWriteMode,
    HalEntrypoints,
    HalParameters,
    Hand,
    IntrinsicsPinhole,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SensorModality,
    SensorSpec,
    SimDescription,
    SimGripperDescription,
    UrdfAsset,
)

from openral_hal._mujoco_arm import MujocoArmHAL

__all__ = ["OPENARM_DESCRIPTION", "OpenArmMujocoHAL"]


# ── Joint inventory ──────────────────────────────────────────────────────────
# 16-DoF public surface: 7 arm + 1 gripper per side.  Joint names use
# the menagerie's ``openarm_<side>_*`` convention shortened to the
# logical names a Skill would use (no `openarm_` prefix in the
# description — that's an internal MJCF naming detail).

_OPENARM_LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"left_joint{i}" for i in range(1, 8))
_OPENARM_RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"right_joint{i}" for i in range(1, 8))
_OPENARM_LEFT_GRIPPER_JOINT: str = "left_gripper"
_OPENARM_RIGHT_GRIPPER_JOINT: str = "right_gripper"

_OPENARM_JOINT_NAMES: tuple[str, ...] = (
    *_OPENARM_LEFT_ARM_JOINTS,
    _OPENARM_LEFT_GRIPPER_JOINT,
    *_OPENARM_RIGHT_ARM_JOINTS,
    _OPENARM_RIGHT_GRIPPER_JOINT,
)


# ── Joint limits (from v2 MJCF ctrlrange) ────────────────────────────────────
# Position limits (rad) come from the v2 ``ctrlrange`` block, verbatim.

_OPENARM_LEFT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "left_joint1": (-3.49066, 1.39626),
    "left_joint2": (-3.31613, 0.174533),
    "left_joint3": (-1.5708, 1.5708),
    "left_joint4": (0.0, 2.44346),
    "left_joint5": (-1.5708, 1.5708),
    "left_joint6": (-0.785398, 0.785398),
    "left_joint7": (-1.5708, 1.5708),
}
_OPENARM_RIGHT_ARM_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "right_joint1": (-1.39626, 3.49066),  # mirrored
    "right_joint2": (-0.174533, 3.31613),  # mirrored
    "right_joint3": (-1.5708, 1.5708),
    "right_joint4": (0.0, 2.44346),
    "right_joint5": (-1.5708, 1.5708),
    "right_joint6": (-0.785398, 0.785398),
    "right_joint7": (-1.5708, 1.5708),
}
# Grippers are revolute (hinge) in v2 — not prismatic like v1.
# Left jaw rotates in [0, 0.7854] rad (closed → open); right
# mirrors with the negative half [-0.7854, 0].
_OPENARM_LEFT_GRIPPER_POSITION_LIMITS: tuple[float, float] = (0.0, 0.7854)
_OPENARM_RIGHT_GRIPPER_POSITION_LIMITS: tuple[float, float] = (-0.7854, 0.0)


# v2 actuator force limits (N·m) per joint class — these match the
# DM8009 / DM4340 / DM4310 motor families documented in the upstream
# MJCF defaults.
_OPENARM_ARM_EFFORT_LIMITS: dict[str, float] = {
    "left_joint1": 40.0,
    "left_joint2": 40.0,
    "left_joint3": 27.0,
    "left_joint4": 27.0,
    "left_joint5": 7.0,
    "left_joint6": 7.0,
    "left_joint7": 7.0,
    "right_joint1": 40.0,
    "right_joint2": 40.0,
    "right_joint3": 27.0,
    "right_joint4": 27.0,
    "right_joint5": 7.0,
    "right_joint6": 7.0,
    "right_joint7": 7.0,
}
_OPENARM_GRIPPER_EFFORT_LIMIT: float = 333.0  # v2 finger forcerange [-333, 333] N·m

# Conservative velocity limits (half the published peak).
_OPENARM_ARM_VELOCITY_LIMIT: float = 4.0
_OPENARM_GRIPPER_VELOCITY_LIMIT: float = 4.0


def _openarm_arm_joint_specs(
    names: tuple[str, ...],
    position_limits: dict[str, tuple[float, float]],
    side: str,
) -> list[JointSpec]:
    parents = [f"openarm_{side}_link{i}" for i in range(7)]  # link0..link6
    children = [f"openarm_{side}_link{i + 1}" for i in range(7)]  # link1..link7
    return [
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link=parent,
            child_link=child,
            position_limits=position_limits[name],
            velocity_limit=_OPENARM_ARM_VELOCITY_LIMIT,
            effort_limit=_OPENARM_ARM_EFFORT_LIMITS[name],
            has_torque_sensor=True,
            actuator_kind="bldc",
            # Upstream MJCF qualifies the logical joint with the `openarm_`
            # prefix; the openarm_robosuite scene reads this to resolve the
            # per-arm actuators (must match robots/openarm/robot.yaml).
            sim_joint_name=f"openarm_{name}",
        )
        for name, parent, child in zip(names, parents, children, strict=True)
    ]


def _openarm_gripper_joint_spec(
    name: str, side: str, position_limits: tuple[float, float]
) -> JointSpec:
    return JointSpec(
        name=name,
        joint_type=JointType.REVOLUTE,  # v2 grippers are hinge joints
        parent_link=f"openarm_{side}_link7",
        child_link=f"openarm_{side}_finger_pair",
        position_limits=position_limits,
        velocity_limit=_OPENARM_GRIPPER_VELOCITY_LIMIT,
        effort_limit=_OPENARM_GRIPPER_EFFORT_LIMIT,
        has_torque_sensor=False,
        actuator_kind="servo",
    )


def _openarm_joint_specs() -> list[JointSpec]:
    return [
        *_openarm_arm_joint_specs(
            _OPENARM_LEFT_ARM_JOINTS, _OPENARM_LEFT_ARM_POSITION_LIMITS, "left"
        ),
        _openarm_gripper_joint_spec(
            _OPENARM_LEFT_GRIPPER_JOINT, "left", _OPENARM_LEFT_GRIPPER_POSITION_LIMITS
        ),
        *_openarm_arm_joint_specs(
            _OPENARM_RIGHT_ARM_JOINTS, _OPENARM_RIGHT_ARM_POSITION_LIMITS, "right"
        ),
        _openarm_gripper_joint_spec(
            _OPENARM_RIGHT_GRIPPER_JOINT, "right", _OPENARM_RIGHT_GRIPPER_POSITION_LIMITS
        ),
    ]


# ── RobotDescription ─────────────────────────────────────────────────────────

OPENARM_DESCRIPTION = RobotDescription(
    name="openarm_v2",
    embodiment_kind=EmbodimentKind.BIMANUAL,
    base_frame="openarm_base",
    joints=_openarm_joint_specs(),
    end_effectors=[
        EndEffectorSpec(
            name="left_gripper",
            kind="parallel_gripper",
            hand=Hand.LEFT,
            n_dof=1,
            max_grip_force_n=_OPENARM_GRIPPER_EFFORT_LIMIT,
            max_payload_kg=2.0,
            workspace_radius_m=0.7,
        ),
        EndEffectorSpec(
            name="right_gripper",
            kind="parallel_gripper",
            hand=Hand.RIGHT,
            n_dof=1,
            max_grip_force_n=_OPENARM_GRIPPER_EFFORT_LIMIT,
            max_payload_kg=2.0,
            workspace_radius_m=0.7,
        ),
    ],
    # RGB cameras (issue #191 Phase 3b): the manifest-driven node's
    # SimSensorBridge publishes these via MujocoArmHAL.read_images, which renders
    # the MJCF camera `sim_camera_name or name`. Kept in sync with
    # robots/openarm/robot.yaml. The MJCF overview camera is named "top" and the
    # canonical sensor name (per ADR-0070) is also "top", so sim_camera_name is
    # no longer set explicitly. vla_feature_key values are checkpoint-frozen.
    sensors=[
        SensorSpec(
            name="top",
            modality=SensorModality.RGB,
            frame_id="world",
            rate_hz=10.0,
            intrinsics=IntrinsicsPinhole(
                width=640, height=480, fx=640.0, fy=640.0, cx=320.0, cy=240.0
            ),
            encoding="rgb8",
            vla_feature_key="observation.images.base",
            vendor="sim",
            model="mujoco_top",
        ),
        SensorSpec(
            name="wrist_left",
            modality=SensorModality.RGB,
            frame_id="openarm_left_ee_base_link",
            rate_hz=10.0,
            intrinsics=IntrinsicsPinhole(
                width=640, height=480, fx=640.0, fy=640.0, cx=320.0, cy=240.0
            ),
            encoding="rgb8",
            vla_feature_key="observation.images.wrist_left",
            vendor="sim",
            model="mujoco_wrist",
        ),
        SensorSpec(
            name="wrist_right",
            modality=SensorModality.RGB,
            frame_id="openarm_right_ee_base_link",
            rate_hz=10.0,
            intrinsics=IntrinsicsPinhole(
                width=640, height=480, fx=640.0, fy=640.0, cx=320.0, cy=240.0
            ),
            encoding="rgb8",
            vla_feature_key="observation.images.wrist_right",
            vendor="sim",
            model="mujoco_wrist",
        ),
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=2.0,
        has_force_control=True,
        bimanual=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["openarm_v2", "openarm"],
        embodiment_tags=["openarm", "openarm_v2", "enactic", "bimanual"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        max_force_n=40.0,
        max_torque_nm=40.0,
        deadman_required=True,
    ),
    sdk_kind="open",
    hal=HalEntrypoints(
        sim="openral_hal.openarm:OpenArmMujocoHAL",
        real=None,
        # issue #191 Phase 3b — kept in sync with robots/openarm/robot.yaml so the
        # manifest-driven node threads these OpenArmMujocoHAL kwargs via build_hal.
        parameters=HalParameters(
            defaults={"settle_steps": 4, "gravity_enabled": False, "staleness_limit_s": 0.5}
        ),
    ),
    # MuJoCo wiring — v2 bimanual MJCF fetched by ``ensure_openarm_v2_mjcf``.
    # Each gripper is a single revolute jaw actuator (no mirror — the
    # passive follower finger tracks via the MJCF's <equality> constraint).
    # ``seed_ctrl_from_qpos=True`` is required because v2's <position>
    # actuators with per-class PD gains would drive every joint to ctrl=0
    # otherwise.  qpos[8] is the left follower finger (passive); qpos[17]
    # is the right follower finger — we skip both.  See ADR-0023.
    assets=AssetRefs(
        urdf=UrdfAsset(ref="file:openarm.urdf"),
        mjcf="openarm:bimanual",
    ),
    sim=SimDescription(
        joint_qpos_addr={
            "left_joint1": 0,
            "left_joint2": 1,
            "left_joint3": 2,
            "left_joint4": 3,
            "left_joint5": 4,
            "left_joint6": 5,
            "left_joint7": 6,
            "left_gripper": 7,
            "right_joint1": 9,
            "right_joint2": 10,
            "right_joint3": 11,
            "right_joint4": 12,
            "right_joint5": 13,
            "right_joint6": 14,
            "right_joint7": 15,
            "right_gripper": 16,
        },
        grippers=[
            SimGripperDescription(
                joint="left_gripper",
                ctrl_range=(0.0, 0.7854),
                qpos_addrs=(7,),
                qpos_scale=0.7854,
                read_mode=GripperReadMode.PASSTHROUGH,
                write_mode=GripperWriteMode.PASSTHROUGH,
            ),
            SimGripperDescription(
                joint="right_gripper",
                ctrl_range=(-0.7854, 0.0),
                qpos_addrs=(16,),
                qpos_scale=0.7854,
                read_mode=GripperReadMode.PASSTHROUGH,
                write_mode=GripperWriteMode.PASSTHROUGH,
            ),
        ],
        seed_ctrl_from_qpos=True,
    ),
    # ADR-0066 — the tabletop arena composition + overview-camera pose are NOT
    # robot properties; they live on the scenes that own them (the deploy scene's
    # `composition:` and the sim scene's `backend_options.top_camera_*`). This
    # constant (and `robots/openarm/robot.yaml`, drift-checked equal) therefore
    # carries no `scene_defaults`. The robot / scene / rSkill are separate.
)


# ── HAL ──────────────────────────────────────────────────────────────────────
# Post-ADR-0023, OpenArmMujocoHAL is a thin :class:`MujocoArmHAL` subclass.
# v2 has 18 qpos (7 arm + 2 finger per side) but only 16 actuators — the
# follower finger tracks via an MJCF ``<equality>`` constraint, so we
# skip qpos 8 / qpos 17 via the explicit ``joint_qpos_addr`` on
# :data:`OPENARM_DESCRIPTION.sim`.  ``seed_ctrl_from_qpos=True`` on the
# manifest replaces the per-class ``connect()`` seeding loop the old
# bespoke class used to do.


class OpenArmMujocoHAL(MujocoArmHAL):
    """HAL adapter for the Enactic OpenArm v2 (MuJoCo digital twin).

    Thin manifest-driven wrapper around :class:`MujocoArmHAL`; all wiring
    (MJCF URI via the ``openarm_v2:`` scheme, joint→qpos map that skips
    the passive follower fingers, two ``PASSTHROUGH`` grippers,
    ``seed_ctrl_from_qpos`` to hold the initial pose under the v2 PD
    actuators) lives in :data:`OPENARM_DESCRIPTION.sim` (ADR-0023).

    Public 16-DoF surface (7 arm + 1 gripper per side, left then right)
    matches what a future ``OpenArmRealHAL`` wrapping the LeRobot OpenArm
    driver will accept.  Gripper commands are passthrough radians
    (left jaw ``[0, 0.7854]``, right jaw ``[-0.7854, 0]``).

    Args:
        mjcf_path: Optional override for the MJCF file path.  When
            ``None``, the v2 bimanual MJCF is fetched lazily through
            :func:`openral_hal._openarm_v2_assets.ensure_openarm_v2_mjcf`
            via the ``openarm_v2:bimanual`` URI scheme.
        settle_steps: Number of MuJoCo physics steps performed in
            :meth:`send_action`.
        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import OpenArmMujocoHAL  # doctest: +SKIP
        >>> hal = OpenArmMujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 7 + 1 + 7 + 1  # doctest: +SKIP
        16
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
        """Initialise the adapter; the MJCF is not loaded until ``connect()``.

        OpenArm has no per-robot ``connect()`` override: any starting pose
        a Skill needs is carried by the rSkill manifest's
        ``starting_pose:`` and applied by ``rskill_runner_node`` via
        :meth:`MujocoArmHAL.reset_to_pose` before the first inference
        tick (ADR-0023 bimanual amendment).
        """
        self._init_from_description(
            OPENARM_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
