"""HAL adapter for the Trossen ALOHA bimanual setup.

The physical ALOHA is two ViperX 300 6-DoF arms with parallel grippers
mounted side-by-side, exposing a 14-DoF joint-position action space
(2 * (6 arm + 1 gripper)). The gym-aloha simulator uses an MJCF
mirroring the same kinematics and scene layout, so the manifest and the
in-code ``ALOHA_DESCRIPTION`` describe both the real robot and the
simulator one-to-one (CLAUDE.md robot/sim split).

This module wires the **real-hardware** Layer-0 path; the gym-aloha sim
path is owned by ``openral_sim.backends.aloha`` and invokes the gym
``MjModel`` directly.

Driver landscape: ``Interbotix/aloha``'s ``aloha_bringup.launch.py``
starts the ``interbotix_xs_sdk`` ``xs_sdk`` node — a direct
Dynamixel/U2D2 driver — per arm. There is **no** ``controller_manager``
and no ``JointTrajectoryController`` anywhere in it; a ros2_control
bridge exists upstream (``interbotix_xs_ros_control``) but bring-up does
not start it. The real node subscribes on
``/<robot_name>/commands/joint_group`` and ``/<robot_name>/commands/joint_single``
(``interbotix_xs_msgs``) and publishes ``/<robot_name>/joint_states`` under
its own namespace, with default namespaces ``follower_left`` /
``follower_right``.

``AlohaHAL`` emits that contract (#250): each arm's 6 joints as a
``JointGroupCommand`` for the ``arm`` group, each gripper as a
``JointSingleCommand`` for the ``gripper`` motor (stock Interbotix ALOHA has
no ``gripper`` *group* — upstream ``follower_modes_*.yaml`` lists it under
``singles:``). ``AlohaHAL`` stays on ``HALBase`` — it is a vendor-driver robot
like ``SO100FollowerHAL``, not a ``RosControlHAL``.

.. warning::
   Never run against a rig. Under ``hal_mode:=real`` the lifecycle node
   attaches the production ``InterbotixXSTransport`` (commands, merged
   per-namespace joint states, torque stop) before ``connect()``; built any
   other way, ``connect()`` refuses (``ROSConfigError``) until a transport is
   attached. The defaults are stock-Interbotix values, **not** read off a
   cell: #250 still needs the rig's namespaces, ``motor_config`` and the
   gripper command range confirmed before first motion. The operating modes
   are checked in code: ``connect()`` reads them back and refuses unless the
   gripper is in ``current_based_position``.

   Like ``SO100FollowerHAL`` this is a vendor-driver robot, outside
   CLAUDE.md §1.5's ``ros2_control`` bridge: Python publishes straight to
   ``xs_sdk``, and there is no command-age watchdog. If Python stalls, each
   Dynamixel holds its last position target; a torque-cutting watchdog
   would drop both unbraked arms instead, so none is added. Recorded as a
   known deviation under #250 (hazard-log entry pending).

ALOHA bring-up launches two robot namespaces; we expose a single 14-DoF
action by interleaving left arm + left gripper, then right arm + right
gripper, in the same order as ``ALOHA_DESCRIPTION.joints``.

Per CLAUDE.md §1.9 the Trossen Interbotix XS SDK is BSD-3 / Apache-2.0
(fully compatible) but ships as vendor-distributed packages, so the
real-hardware manifest (``ALOHA_REAL_DESCRIPTION``, derived from
``ALOHA_DESCRIPTION`` via ``make_real_description``) declares
``sdk_kind: "closed_with_api"``. Both share the same ``hal`` block:
``hal.sim = "openral_hal.aloha:AlohaMujocoHAL"`` and
``hal.real = "openral_hal.aloha:AlohaHAL"``; the sim baseline keeps
``sdk_kind: "open"``. ``deploy sim`` / ``deploy run`` pick the HAL via
``build_hal(mode=...)``.

.. _Interbotix/interbotix_ros_manipulators:
   https://github.com/Interbotix/interbotix_ros_manipulators

Example:
    >>> from openral_hal.aloha import ALOHA_DESCRIPTION
    >>> ALOHA_DESCRIPTION.name
    'aloha_bimanual'
    >>> len(ALOHA_DESCRIPTION.joints)
    14
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    ActionRepresentation,
    ActionSpec,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    GripperReadMode,
    GripperWriteMode,
    HalEntrypoints,
    HalParameters,
    Hand,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SimDescription,
    SimGripperDescription,
)

from openral_hal._base import HALBase, _raw_floats, resolve_staleness_limit_s
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._real_description import make_real_description
from openral_hal.protocol import EStopRecovery
from openral_hal.ros_control import DownstreamStopReport, TriggerReport

__all__ = [
    "ALOHA_DESCRIPTION",
    "ALOHA_REAL_DESCRIPTION",
    "AlohaHAL",
    "AlohaMujocoHAL",
    "InterbotixStopSeam",
    "InterbotixStoppable",
]

log = structlog.get_logger(__name__)

# xs_sdk command contract (#250). Stock Interbotix ALOHA
# (Interbotix/aloha config/follower_modes_*.yaml) defines one motor group,
# ``arm`` (6 joints) -> interbotix_xs_msgs/JointGroupCommand on
# /<ns>/commands/joint_group, and the gripper as a *single* motor ``gripper``
# -> interbotix_xs_msgs/JointSingleCommand on /<ns>/commands/joint_single.
# A real ALOHA is bimanual: exactly two follower arms, each its own xs_sdk
# namespace. Defaults are stock values, not read off a cell (#250).
_ALOHA_ARM_COUNT: int = 2
_DEFAULT_ARM_GROUP: str = "arm"
_DEFAULT_GRIPPER_NAME: str = "gripper"
_DEFAULT_ALOHA_ARM_NAMESPACES: tuple[str, str] = ("follower_left", "follower_right")

# Gripper command range (closed, open) the normalised [0, 1] action maps onto.
# Calibration, so a constructor parameter. Default = upstream Interbotix
# aloha/constants.py FOLLOWER_GRIPPER_JOINT_{CLOSE,OPEN}: what upstream
# real_env.set_gripper_pose publishes on JointSingleCommand after
# setup_follower_bot switches the gripper to ``current_based_position``
# (radians). Under the bring-up yaml's ``linear_position`` mode xs_sdk reads
# ``cmd`` as finger separation in metres, which is why the HAL refuses any
# gripper mode but ``current_based_position`` (below). Confirm on the cell (#250).
_DEFAULT_GRIPPER_CMD_RANGE: tuple[float, float] = (0.6197, 1.6214)

# Gripper read-back range (closed, open), metres of the ``left_finger`` joint.
# Upstream reads the gripper as the finger *position*, not the motor angle:
# real_env.get_qpos normalises qpos_raw[7] (left_finger) with
# FOLLOWER_GRIPPER_POSITION_{CLOSE,OPEN}, and the sim twin's channel is a
# normalised finger position too. Command and read-back therefore use
# different ranges, as upstream does (the linkage is nonlinear), so a policy
# trained on upstream data sees the observation it was trained on.
_DEFAULT_GRIPPER_POSITION_RANGE: tuple[float, float] = (0.0440, 0.0579)

# Operating modes the commands above are only valid in. The arm takes joint
# radians in ``position``; ``gripper_cmd_range`` is radians, which xs_sdk
# reads as such only in ``current_based_position``. The stock bring-up yaml
# (follower_modes_*.yaml) starts the gripper in ``linear_position`` — there
# ``cmd`` is finger separation in metres, and 0.62 to 1.62 "m" puts xs_driver's
# convert_linear_position_to_radian outside acos's domain, so a NaN reaches
# goalPosition. Upstream real_env switches the mode in setup_follower_bot;
# this HAL never changes a mode itself (§1.4): it reads it back over
# ``get_robot_info`` in ``connect()`` and refuses on a mismatch.
_ARM_OPERATING_MODE: str = "position"
_GRIPPER_OPERATING_MODE: str = "current_based_position"

# ── Joint inventory ──────────────────────────────────────────────────────────
# Order matches the gym-aloha 14-D action vector and the YAML manifest:
# left arm 6 + left gripper 1 + right arm 6 + right gripper 1.

_ALOHA_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
)
_ALOHA_LEFT_GRIPPER_JOINT: str = "left_gripper"
_ALOHA_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
)
_ALOHA_RIGHT_GRIPPER_JOINT: str = "right_gripper"

_ALOHA_JOINT_NAMES: tuple[str, ...] = (
    *_ALOHA_LEFT_ARM_JOINTS,
    _ALOHA_LEFT_GRIPPER_JOINT,
    *_ALOHA_RIGHT_ARM_JOINTS,
    _ALOHA_RIGHT_GRIPPER_JOINT,
)

# Per-joint position limits — one block per arm, mirrored across left/right.
# Numbers come from the ViperX 300 data sheet and match the YAML manifest
# verbatim (the YAML uses the truncated 3.14159 form for π so the in-code
# constant uses the same string-equal value to keep the manifest-vs-HAL
# drift guard happy).
_PI: float = 3.14159
_ALOHA_ARM_POSITION_LIMITS: dict[str, tuple[float, float]] = {
    "waist": (-_PI, _PI),
    "shoulder": (-1.85, 1.25),
    "elbow": (-1.76, 1.6),
    "forearm_roll": (-_PI, _PI),
    "wrist_angle": (-1.8, 2.2),
    "wrist_rotate": (-_PI, _PI),
}
# The gripper channel is normalized: 0 = closed, 1 = open (gym_aloha and the
# act-aloha checkpoints). The twin maps it onto the MJCF finger range
# [0.021, 0.057] m. The former (0.0, 0.041) m had no source and clamped every
# normalized command to nearly closed.
_ALOHA_GRIPPER_POSITION_LIMITS: tuple[float, float] = (0.0, 1.0)
_ALOHA_ARM_AXIS: dict[str, tuple[float, float, float]] = {
    "waist": (0.0, 0.0, 1.0),
    "shoulder": (0.0, 1.0, 0.0),
    "elbow": (0.0, 1.0, 0.0),
    "forearm_roll": (1.0, 0.0, 0.0),
    "wrist_angle": (0.0, 1.0, 0.0),
    "wrist_rotate": (1.0, 0.0, 0.0),
}


def _aloha_joint_specs() -> list[JointSpec]:
    specs: list[JointSpec] = []
    for side in ("left", "right"):
        # Six arm joints driven from the side root link.
        prev_link = f"{side}_link0"  # virtual root above the waist
        for idx, suffix in enumerate(
            ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
            start=1,
        ):
            child_link = f"{side}_link{idx}"
            parent_link = f"vx300s_{side}" if suffix == "waist" else prev_link
            specs.append(
                JointSpec(
                    name=f"{side}_{suffix}",
                    joint_type=JointType.REVOLUTE,
                    parent_link=parent_link,
                    child_link=child_link,
                    axis_xyz=_ALOHA_ARM_AXIS[suffix],
                    position_limits=_ALOHA_ARM_POSITION_LIMITS[suffix],
                    velocity_limit=3.0,
                    effort_limit=4.0,
                    actuator_kind="servo",
                    sim_joint_name=f"vx300s_{side}/{suffix}",
                )
            )
            prev_link = child_link
        # Gripper: normalized [0, 1] channel (see _ALOHA_GRIPPER_POSITION_LIMITS).
        specs.append(
            JointSpec(
                name=f"{side}_gripper",
                joint_type=JointType.PRISMATIC,
                parent_link=f"{side}_link6",
                child_link=f"{side}_finger",
                axis_xyz=(1.0, 0.0, 0.0),
                position_limits=_ALOHA_GRIPPER_POSITION_LIMITS,
                velocity_limit=0.5,
                effort_limit=4.0,
                actuator_kind="servo",
                # The twin's driven finger; the other finger is its mirror actuator.
                sim_joint_name=f"vx300s_{side}/left_finger",
            )
        )
    return specs


# ── RobotDescription ─────────────────────────────────────────────────────────

ALOHA_DESCRIPTION = RobotDescription(
    name="aloha_bimanual",
    embodiment_kind=EmbodimentKind.BIMANUAL,
    base_frame="world",
    joints=_aloha_joint_specs(),
    end_effectors=[
        EndEffectorSpec(
            name="left_gripper",
            kind="parallel_gripper",
            hand=Hand.LEFT,
            n_dof=1,
            max_grip_force_n=4.0,
            max_payload_kg=0.5,
            workspace_radius_m=0.6,
            command_convention="normalized_open_unit",
        ),
        EndEffectorSpec(
            name="right_gripper",
            kind="parallel_gripper",
            hand=Hand.RIGHT,
            n_dof=1,
            max_grip_force_n=4.0,
            max_payload_kg=0.5,
            workspace_radius_m=0.6,
            command_convention="normalized_open_unit",
        ),
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=0.5,
        has_vision=True,
        bimanual=True,
        supported_control_modes=[ControlMode.JOINT_POSITION],
        supported_vla_embodiments=["aloha", "lerobot"],
        embodiment_tags=["aloha", "lerobot"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=1.0,
        max_joint_speed_factor=0.5,
        deadman_required=False,
        # provisional: former schema default, not measured on this rig — see issue #303
        max_force_n=50.0,
        max_torque_nm=10.0,
        max_ee_accel_m_s2=1.0,
        contact_force_threshold_n=30.0,
        self_collision_margin_m=0.0,
        # runner ramp to starting_pose — the former defaults, declared (issue #303)
        starting_pose_max_joint_speed_rad_s=0.5,
        starting_pose_tolerance_rad=0.05,
        starting_pose_max_joint_speed_m_s=0.025,  # prismatic gripper; mirrors the YAML
        starting_pose_tolerance_m=0.0025,
        joint_state_staleness_limit_s=0.2,  # provisional, mirrors the YAML
    ),
    sdk_kind="open",
    # Control rate: the runner ticks at it, the HAL node publishes proprio at
    # it, and the recorder stamps it as fps (issue #303). Mirrors the YAML.
    action_spec=ActionSpec(
        dim=14,
        representation=ActionRepresentation.JOINT_POSITIONS,
        control_freq_hz=50.0,
    ),
    hal=HalEntrypoints(
        sim=None,
        real="openral_hal.aloha:AlohaHAL",
        # xs_sdk wire contract (#250) — mirrors hal.parameters.defaults in
        # robots/aloha_bimanual/robot.yaml (test_robot_manifests_match_hal_
        # constants keeps the two in lockstep) and is built from the module
        # constants, so the code holds one copy.
        parameters=HalParameters(
            defaults={
                "arm_namespaces": list(_DEFAULT_ALOHA_ARM_NAMESPACES),
                "arm_group": _DEFAULT_ARM_GROUP,
                "gripper_name": _DEFAULT_GRIPPER_NAME,
                "gripper_cmd_range": list(_DEFAULT_GRIPPER_CMD_RANGE),
                "gripper_position_range": list(_DEFAULT_GRIPPER_POSITION_RANGE),
            }
        ),
    ),
    # MuJoCo wiring for the gym-aloha sim twin.  Two normalised grippers
    # with mirror_actuator_index (positive finger + mirror to negative
    # finger).  keyframe_index=0 seeds the fingers inside their
    # ctrlrange — gym-aloha's reset does the same.
    assets=AssetRefs(mjcf="gym_aloha:bimanual_viperx_transfer_cube"),
    sim=SimDescription(
        joint_qpos_addr={
            "left_waist": 0,
            "left_shoulder": 1,
            "left_elbow": 2,
            "left_forearm_roll": 3,
            "left_wrist_angle": 4,
            "left_wrist_rotate": 5,
            "left_gripper": 6,
            "right_waist": 8,
            "right_shoulder": 9,
            "right_elbow": 10,
            "right_forearm_roll": 11,
            "right_wrist_angle": 12,
            "right_wrist_rotate": 13,
            "right_gripper": 14,
        },
        actuator_index={
            "left_waist": 0,
            "left_shoulder": 1,
            "left_elbow": 2,
            "left_forearm_roll": 3,
            "left_wrist_angle": 4,
            "left_wrist_rotate": 5,
            "left_gripper": 6,
            "right_waist": 8,
            "right_shoulder": 9,
            "right_elbow": 10,
            "right_forearm_roll": 11,
            "right_wrist_angle": 12,
            "right_wrist_rotate": 13,
            "right_gripper": 14,
        },
        grippers=[
            SimGripperDescription(
                joint="left_gripper",
                ctrl_range=(0.021, 0.057),
                qpos_addrs=(6,),
                qpos_scale=0.036,
                read_mode=GripperReadMode.AFFINE_LOW_HIGH,
                write_mode=GripperWriteMode.NORMALISED,
                actuator_index=6,
                mirror_actuator_index=7,
            ),
            SimGripperDescription(
                joint="right_gripper",
                ctrl_range=(0.021, 0.057),
                qpos_addrs=(14,),
                qpos_scale=0.036,
                read_mode=GripperReadMode.AFFINE_LOW_HIGH,
                write_mode=GripperWriteMode.NORMALISED,
                actuator_index=14,
                mirror_actuator_index=15,
            ),
        ],
        keyframe_index=0,
    ),
)


# ── RobotDescription (real-HW) ───────────────────────────────────────────────
# Pinned by ``robots/aloha_bimanual/robot.yaml``; drift guarded by
# ``tests/unit/test_robot_manifests_match_hal_constants.py``.  The eval-layer
# gym-aloha scene is still constructed directly by ``openral_sim.backends.aloha``
# from ``ALOHA_DESCRIPTION``; ``deploy sim`` builds the sim HAL via
# ``build_hal(mode="sim")`` → ``hal.sim`` (AlohaMujocoHAL).

ALOHA_REAL_DESCRIPTION = make_real_description(
    ALOHA_DESCRIPTION,
    sdk_kind="closed_with_api",
)


# ── HAL ──────────────────────────────────────────────────────────────────────

# Joint-state source the lifecycle node is told about. xs_sdk publishes
# /<ns>/joint_states per arm with unprefixed joint names and NO aggregated
# topic; ``InterbotixXSTransport`` merges them into the 14-DoF manifest order.
_DEFAULT_ALOHA_JOINT_STATE_TOPIC: str = "/joint_states"

# Interbotix XS exposes one ``torque_enable`` service per arm
# (``/<robot_name>/torque_enable``, ``interbotix_xs_msgs/srv/TorqueEnable``:
# ``cmd_type='group'``, ``name='all'``, ``enable=false`` cuts torque to every
# motor of that arm). That is the downstream stop: with torque off the
# ``xs_sdk`` node cannot move the arm, whatever trajectory it still holds.
# The two follower arms of a real ALOHA run under these namespaces (#250).
_TORQUE_GROUP_ALL = "all"


def _closed_open(name: str, value: Sequence[float]) -> tuple[float, float]:
    """Validate a ``(closed, open)`` calibration pair: two distinct finite values.

    Raises:
        ROSConfigError: Otherwise.
    """
    if (
        isinstance(value, str)
        or len(value) != 2  # noqa: PLR2004
        or not all(math.isfinite(float(v)) for v in value)
        or float(value[0]) == float(value[1])
    ):
        raise ROSConfigError(
            f"AlohaHAL {name} must be two distinct finite values (closed, open); got {value!r}."
        )
    return (float(value[0]), float(value[1]))


_PublishFn = Callable[[str, dict[str, object]], None]
_StateFn = Callable[[], dict[str, object]]


@runtime_checkable
class InterbotixStopSeam(Protocol):
    """What a transport must expose for ``AlohaHAL`` to cut torque and check modes.

    The production implementation is ``InterbotixXSTransport`` (real
    ``interbotix_xs_msgs/srv/TorqueEnable`` and ``RobotInfo`` clients on a
    live graph); unit tests use ``SimTorqueSeam`` from
    ``openral_hal.sim_transport``.
    """

    def torque_enable(
        self, robot_name: str, *, group: str, enable: bool, timeout_s: float
    ) -> TriggerReport:
        """Call ``/<robot_name>/torque_enable`` for one joint group; report the ack."""
        ...

    def operating_mode(self, robot_name: str, *, cmd_type: str, name: str, timeout_s: float) -> str:
        """Read one group's / single motor's mode off ``/<robot_name>/get_robot_info``.

        Read-only. Raises ``ROSRuntimeError`` if the service is absent or
        does not answer within ``timeout_s``.
        """
        ...


@runtime_checkable
class InterbotixStoppable(Protocol):
    """HALs the lifecycle node wires to an ``InterbotixXSTransport`` under ``hal_mode:=real``.

    The transport is both the drive (``attach_transport``) and the
    ``InterbotixStopSeam`` (``attach_torque_stop``); both are attached before
    ``connect()``.
    """

    @property
    def gripper_name(self) -> str:
        """The ``xs_sdk`` single motor the gripper is commanded through."""
        ...

    @property
    def gripper_cmd_range(self) -> tuple[float, float]:
        """``(closed, open)`` the normalised gripper is mapped onto."""
        ...

    @property
    def gripper_position_range(self) -> tuple[float, float]:
        """``(closed, open)`` finger position the gripper read-back is normalised with."""
        ...

    def arm_namespaces(self) -> list[str]:
        """Every ``xs_sdk`` robot namespace whose torque the stop cuts."""
        ...

    def joint_names(self) -> list[str]:
        """The HAL's joint names in manifest order."""
        ...

    def attach_transport(
        self,
        publish_fn: _PublishFn,
        state_fn: _StateFn,
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind the live command/state transport after construction."""
        ...

    def attach_torque_stop(self, seam: InterbotixStopSeam) -> None:
        """Bind the live torque seam after construction."""
        ...


class AlohaHAL(HALBase):
    """HAL adapter for a physical Trossen ALOHA bimanual setup.

    The 14-DoF action vector is split internally:

    * indices ``0:6``  → ``JointGroupCommand{arm}``  on ``/<left_ns>/commands/joint_group``
    * index   ``6``    → ``JointSingleCommand{gripper}`` on ``/<left_ns>/commands/joint_single``
    * indices ``7:13`` → ``JointGroupCommand{arm}``  on ``/<right_ns>/commands/joint_group``
    * index   ``13``   → ``JointSingleCommand{gripper}`` on ``/<right_ns>/commands/joint_single``

    Both messages hold one target, so only the chunk's first step (the next
    target) is sent. ``connect()`` reads each arm's ``arm`` group and
    ``gripper`` motor mode back over the attached ``InterbotixStopSeam``
    (``get_robot_info``) and refuses unless they are ``position`` /
    ``current_based_position`` — the only modes ``gripper_cmd_range``
    (radians) is valid in. Under the lifecycle node that runs at configure,
    off the control loop. It never sets a mode itself; the stock bring-up
    yaml's ``linear_position`` gripper must be switched by the operator
    (upstream ``setup_follower_bot``). If the joint-state stream goes stale
    after it was fresh (an ``xs_sdk`` relaunch comes back in the bring-up
    modes), the check is void: ``send_action`` refuses, without service
    calls, until the HAL reconnects (lifecycle cleanup + configure).

    The split keeps each side addressable independently while upstream
    layers see one unified 14-DoF action. Under ``hal_mode:=real`` the
    lifecycle node attaches the production ``InterbotixXSTransport``; never
    run against a rig (#250).

    Args:
        arm_group: ``xs_sdk`` motor group receiving each arm's 6 joints
            (``JointGroupCommand.name``). Defaults to ``"arm"`` (stock
            Interbotix ``motor_config``; not confirmed on a cell, #250).
        gripper_name: ``xs_sdk`` single motor receiving the gripper
            (``JointSingleCommand.name``). Defaults to ``"gripper"``.
        gripper_cmd_range: ``(closed, open)`` gripper motor angles in radians
            the normalised ``[0, 1]`` gripper action maps onto affinely —
            radians because ``connect()`` refuses any gripper mode but
            ``current_based_position``. Calibration; defaults to upstream
            ALOHA's ``FOLLOWER_GRIPPER_JOINT_{CLOSE,OPEN}``.
        gripper_position_range: ``(closed, open)`` ``left_finger`` position in
            metres the gripper read-back is normalised with (upstream
            ``FOLLOWER_GRIPPER_POSITION_{CLOSE,OPEN}``, the observation
            convention upstream data was recorded in). Calibration.
        joint_state_topic: Reported to the lifecycle node only. No real ALOHA
            publishes an aggregated topic — each arm publishes
            ``/<ns>/joint_states`` — and ``InterbotixXSTransport`` merges them.
        arm_namespaces: The two distinct ``xs_sdk`` robot namespaces of the
            follower arms (left, right); commands go to
            ``/<namespace>/commands/joint_{group,single}`` and ``estop()``
            calls ``/<namespace>/torque_enable`` on each.
            Defaults to ``("follower_left", "follower_right")``.
        stop_timeout_s: How long ``estop`` waits for each torque-off call, and
            ``connect()`` for each ``get_robot_info`` mode read.
        publish_fn: Callable forwarding messages to ROS 2 topics. Required
            before ``connect()``; tests inject ``SimTransport.publish``.
        state_fn: Callable returning the latest raw joint state as a dict.
            Required before ``connect()``; tests inject ``SimTransport.state``.
        stamp_fn: Returns the ``time.monotonic()`` arrival time of the newest
            joint state that covers *both* arms (the older side's stamp, so
            one silent arm reads stale); ``0.0`` before both have reported,
            which reads as infinitely stale. A real-rig transport must supply
            it. ``None`` falls back to the ``connect()`` time — a liveness
            floor, not a freshness check (same contract as
            ``RosControlHAL.attach_transport``).
        staleness_limit_s: Maximum age of a ``read_state()`` reading
            before ``ROSPerceptionStale`` is raised. ``None`` (default) reads the manifest's
            ``safety.joint_state_staleness_limit_s``.
        description: The loaded ``robots/<id>/robot.yaml`` manifest
            (threaded by ``build_hal``). ``None`` falls back to the
            in-code ``ALOHA_REAL_DESCRIPTION`` mirror.

    **Lifecycle e-stop.** ``estop()`` calls ``torque_enable(enable=false)``
    for the ``all`` group of every namespace in ``arm_namespaces`` through
    the attached ``InterbotixStopSeam`` and records a
    ``DownstreamStopReport`` — ``stopped`` only when every arm acknowledged.
    Torque off means the ViperX arms go limp (Dynamixels have no brakes), so
    they will settle under gravity; that is the same outcome as the rig's
    hardware e-stop and is the only stop ``xs_sdk`` offers. Recovery is
    ``RESTART_REQUIRED``: re-torque and re-home the arms with the Interbotix
    tooling, then restart the HAL lifecycle node.

    Example:
        >>> from openral_hal.aloha import AlohaHAL
        >>> from openral_hal.sim_transport import SimTorqueSeam, SimTransport
        >>> transport = SimTransport(n_joints=14)
        >>> hal = AlohaHAL(
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.attach_torque_stop(SimTorqueSeam(arms=hal.arm_namespaces()))
        >>> hal.connect()
        >>> hal.description.name
        'aloha_bimanual'
        >>> hal.arm_namespaces()
        ['follower_left', 'follower_right']
        >>> hal.disconnect()
    """

    #: Torque-off leaves the arms limp; only an operator can re-torque and
    #: re-home them, so the lifecycle node must not re-arm in process.
    estop_recovery: EStopRecovery = EStopRecovery.RESTART_REQUIRED

    def __init__(
        self,
        *,
        arm_group: str = _DEFAULT_ARM_GROUP,
        gripper_name: str = _DEFAULT_GRIPPER_NAME,
        gripper_cmd_range: Sequence[float] = _DEFAULT_GRIPPER_CMD_RANGE,
        gripper_position_range: Sequence[float] = _DEFAULT_GRIPPER_POSITION_RANGE,
        joint_state_topic: str = _DEFAULT_ALOHA_JOINT_STATE_TOPIC,
        arm_namespaces: Sequence[str] = _DEFAULT_ALOHA_ARM_NAMESPACES,
        publish_fn: _PublishFn | None = None,
        state_fn: _StateFn | None = None,
        stamp_fn: Callable[[], float] | None = None,
        staleness_limit_s: float | None = None,
        stop_timeout_s: float = 5.0,
        description: RobotDescription | None = None,
    ) -> None:
        """Initialise the adapter; no transport is opened until ``connect()``."""
        # Bimanual contract (#250): the two follower arms must be distinct
        # xs_sdk namespaces — with one namespace both arms would publish
        # different targets to the same /<ns>/commands/joint_group topic and
        # the later publish would silently replace the earlier.
        # A bare str is a Sequence[str] too: "lr" would pass as two namespaces.
        # Stored stripped: "/follower_left" and "follower_left" are one
        # namespace, and a slash left on would build "//ns/..." topic names.
        namespaces = (
            [] if isinstance(arm_namespaces, str) else [str(ns).strip("/") for ns in arm_namespaces]
        )
        if (
            isinstance(arm_namespaces, str)
            or len(namespaces) != _ALOHA_ARM_COUNT
            or len(set(namespaces)) != _ALOHA_ARM_COUNT
            or not all(namespaces)
        ):
            raise ROSConfigError(
                "AlohaHAL needs exactly two distinct arm namespaces "
                f"(follower_left/follower_right); got {list(arm_namespaces)!r}."
            )
        self.description: RobotDescription = description or ALOHA_REAL_DESCRIPTION
        self._gripper_closed, self._gripper_open = _closed_open(
            "gripper_cmd_range", gripper_cmd_range
        )
        self._gripper_position_range = _closed_open(
            "gripper_position_range", gripper_position_range
        )
        self._arm_group = arm_group
        self._gripper_name = gripper_name
        self._joint_state_topic = joint_state_topic
        self._arm_namespaces = namespaces
        self._publish_fn: _PublishFn | None = publish_fn
        self._state_fn: _StateFn | None = state_fn
        self._stamp_fn = stamp_fn
        self._staleness_limit_s = resolve_staleness_limit_s(self.description, staleness_limit_s)
        self._stop_timeout_s = stop_timeout_s
        self._torque_seam: InterbotixStopSeam | None = None
        self._last_stop_report: DownstreamStopReport | None = None
        # Set by connect() once get_robot_info confirmed both arms' modes;
        # voided when the stream goes stale after being fresh (an xs_sdk
        # relaunch comes back in the bring-up modes).
        self._modes_verified: bool = False
        self._stream_was_fresh: bool = False

        self._connected: bool = False
        self._last_state_time: float = 0.0
        self._joint_names: list[str] = list(_ALOHA_JOINT_NAMES)

    # ── Lifecycle e-stop wiring ───────────────────────────────────────────

    @property
    def gripper_name(self) -> str:
        """The ``xs_sdk`` single motor the gripper is commanded through."""
        return self._gripper_name

    @property
    def gripper_cmd_range(self) -> tuple[float, float]:
        """``(closed, open)`` gripper motor radians the normalised command maps onto."""
        return (self._gripper_closed, self._gripper_open)

    @property
    def gripper_position_range(self) -> tuple[float, float]:
        """``(closed, open)`` ``left_finger`` metres the read-back is normalised with."""
        return self._gripper_position_range

    def arm_namespaces(self) -> list[str]:
        """The ``xs_sdk`` namespaces ``estop`` cuts torque on."""
        return list(self._arm_namespaces)

    def joint_names(self) -> list[str]:
        """The 14 joint names in manifest order (``left_waist`` … ``right_gripper``)."""
        return list(self._joint_names)

    def attach_transport(
        self,
        publish_fn: _PublishFn,
        state_fn: _StateFn,
        stamp_fn: Callable[[], float] | None = None,
    ) -> None:
        """Bind the live ``xs_sdk`` transport (the lifecycle node, before ``connect()``).

        ``build_hal`` constructs the HAL before any ROS node exists, so a real
        deployment cannot pass the callables to ``__init__``; same contract as
        ``RosControlHAL.attach_transport``.

        Raises:
            ROSRuntimeError: If already connected — swapping the transport
                under a live HAL would skip the operating-mode check.
            ROSConfigError: If any argument is not callable.
        """
        if self._connected:
            raise ROSRuntimeError("AlohaHAL.attach_transport() must run before connect().")
        for name, fn in (("publish_fn", publish_fn), ("state_fn", state_fn)):
            if not callable(fn):
                raise ROSConfigError(f"attach_transport({name}=...) needs a callable.")
        if stamp_fn is not None and not callable(stamp_fn):
            raise ROSConfigError(
                "attach_transport(stamp_fn=...) needs a callable returning the newest "
                "sample's time.monotonic() timestamp, e.g. `transport.last_arrival`."
            )
        self._publish_fn = publish_fn
        self._state_fn = state_fn
        self._stamp_fn = stamp_fn

    def attach_torque_stop(self, seam: InterbotixStopSeam) -> None:
        """Bind the live torque seam (the lifecycle node under ``hal_mode:=real``).

        Raises:
            ROSConfigError: If ``seam`` does not satisfy ``InterbotixStopSeam``.
        """
        if not isinstance(seam, InterbotixStopSeam):
            raise ROSConfigError(
                f"attach_torque_stop() needs an InterbotixStopSeam (torque_enable + "
                f"operating_mode); "
                f"got {type(seam).__name__}."
            )
        self._torque_seam = seam

    @property
    def last_stop_report(self) -> DownstreamStopReport | None:
        """Report of the most recent ``estop()``; ``None`` before it."""
        return self._last_stop_report

    # ── HAL Protocol ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Confirm both arms' operating modes, then mark the HAL live.

        The mode check runs here — at configure under the lifecycle node, off
        the control loop — so a mismatch or an unreachable ``xs_sdk`` fails
        the configure instead of blocking command ticks.

        Raises:
            ROSRuntimeError: If already connected, or ``get_robot_info`` is
                absent or does not answer.
            ROSConfigError: If ``publish_fn`` or ``state_fn`` was neither
                injected nor attached (``attach_transport``) — commands would
                never leave the process and state would read zeros — if no
                ``InterbotixStopSeam`` is attached, or an arm's modes are not
                ``position`` / ``current_based_position``.
        """
        if self._connected:
            raise ROSRuntimeError("AlohaHAL is already connected.")
        if self._publish_fn is None or self._state_fn is None:
            missing = [
                name
                for name, fn in (("publish_fn", self._publish_fn), ("state_fn", self._state_fn))
                if fn is None
            ]
            raise ROSConfigError(
                f"AlohaHAL has no {' / '.join(missing)}: commands would go nowhere and state "
                "would read zeros. Inject both, or attach_transport() an "
                "InterbotixXSTransport (the lifecycle node does under hal_mode:=real), "
                "before connect()."
            )
        self._modes_verified = False
        self._verify_operating_modes()
        log.info(
            "hal.connect",
            robot=self.description.name,
            arm_namespaces=list(self._arm_namespaces),
            arm_group=self._arm_group,
            gripper_name=self._gripper_name,
        )
        self._connected = True
        self._stream_was_fresh = False
        self._last_state_time = time.monotonic()

    def read_state(self) -> JointState:
        """Return the latest joint state for all 14 description joints.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.
        """
        self._require_connected("read_state")
        age = self._stream_age()
        sample_age_s = age if self._stamp_fn is not None else 0.0
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        assert self._state_fn is not None  # connect() refuses without one
        raw: dict[str, object] = self._state_fn()

        return JointState(
            name=list(self._joint_names),
            position=_raw_floats(raw, "position", n),
            velocity=_raw_floats(raw, "velocity", n),
            effort=_raw_floats(raw, "effort", n),
            # When the sample arrived, not when it was read (as RosControlHAL).
            stamp_ns=int(time.time_ns()) - int(max(sample_age_s, 0.0) * 1e9),
        )

    def send_action(self, action: Action) -> None:
        """Send the chunk's next target to both arms and both grippers over xs_sdk.

        ``JointGroupCommand`` / ``JointSingleCommand`` hold one target, so
        only ``joint_targets[0]`` is sent — jumping to a later step would cut
        through the path the policy planned between. Each arm's 6 joints go
        to ``/<ns>/commands/joint_group`` (group ``arm_group``); each gripper,
        mapped from ``[0, 1]`` onto ``gripper_cmd_range``, to
        ``/<ns>/commands/joint_single`` (motor ``gripper_name``).

        Raises:
            ROSRuntimeError: If not connected, or the joint-state stream went
                stale since ``connect()`` (the mode check ``connect()`` made
                no longer holds; reconnect). No service call is made here.
            ROSConfigError: If ``action.control_mode`` is not
                ``JOINT_POSITION``, the joint count is wrong, or the sent
                step is not finite. The gripper value is clamped to ``[0, 1]``.
        """
        self._require_connected("send_action")
        if action.control_mode != ControlMode.JOINT_POSITION:
            raise ROSConfigError(
                f"AlohaHAL only supports JOINT_POSITION; got {action.control_mode!r}."
            )
        n = len(self._joint_names)
        if not action.joint_targets:
            raise ROSConfigError("AlohaHAL.send_action requires non-empty joint_targets.")
        for step_idx, step in enumerate(action.joint_targets):
            if len(step) != n:
                raise ROSConfigError(
                    f"action.joint_targets[{step_idx}] has {len(step)} values "
                    f"but ALOHA exposes {n} joints."
                )

        # Index layout — see class docstring.
        step = action.joint_targets[0]
        if not all(math.isfinite(v) for v in step):
            raise ROSConfigError(f"AlohaHAL.send_action got a non-finite target: {list(step)!r}.")
        self._stream_age()  # voids the mode check if the stream went stale
        if not self._modes_verified:
            raise ROSRuntimeError(
                "AlohaHAL refuses to command: the xs_sdk joint-state stream went stale "
                "since connect(), and a relaunched xs_sdk comes back in the bring-up "
                "operating modes. Reconnect (lifecycle cleanup + configure) to re-read them."
            )
        span = self._gripper_open - self._gripper_closed
        publish = self._publish_fn
        assert publish is not None  # connect() refuses without one
        left_ns, right_ns = self._arm_namespaces  # validated distinct pair
        for ns, arm, grip in ((left_ns, step[0:6], step[6]), (right_ns, step[7:13], step[13])):
            publish(f"/{ns}/commands/joint_group", {"name": self._arm_group, "cmd": list(arm)})
            publish(
                f"/{ns}/commands/joint_single",
                # Clamped: an overshoot past [0, 1] must not drive past the calibrated stops.
                {
                    "name": self._gripper_name,
                    "cmd": self._gripper_closed + min(max(grip, 0.0), 1.0) * span,
                },
            )
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            horizon=action.horizon,
        )

    def _stream_age(self) -> float:
        """Age of the newest sample covering both arms; latch a stale-after-fresh stream.

        Nothing received yet (``stamp_fn`` → 0.0) is infinitely stale, never
        "fresh at connect()". Once a ``stamp_fn`` stream has been fresh, going
        stale voids the operating-mode check (the lifecycle node polls
        ``read_state`` on its joint-state timer, so a relaunch is seen).
        """
        if self._stamp_fn is not None:
            last = self._stamp_fn()
            age = float("inf") if last <= 0.0 else time.monotonic() - last
        else:
            age = time.monotonic() - self._last_state_time
        if age <= self._staleness_limit_s:
            self._stream_was_fresh = True
        elif self._stamp_fn is not None and self._stream_was_fresh and self._modes_verified:
            self._modes_verified = False
            log.warning(
                "hal.aloha.modes_unverified",
                robot=self.description.name,
                age_s=age,
                reason="joint-state stream went stale; commands refused until reconnect",
            )
        return age

    def _verify_operating_modes(self) -> None:
        """Refuse to command unless every arm runs the modes the payloads assume.

        Read-only (``get_robot_info``); a mode is never set from here. Runs in
        ``connect()``.

        Raises:
            ROSConfigError: No seam attached, or a mode mismatch.
            ROSRuntimeError: The service is absent or does not answer.
        """
        seam = self._torque_seam
        if seam is None:
            raise ROSConfigError(
                "AlohaHAL cannot confirm the arm/gripper operating modes: no "
                "InterbotixStopSeam is attached (attach_torque_stop). Refusing to command — "
                "gripper_cmd_range is radians and is only valid in current_based_position."
            )
        expected = (
            ("group", self._arm_group, _ARM_OPERATING_MODE),
            ("single", self._gripper_name, _GRIPPER_OPERATING_MODE),
        )
        wrong: list[str] = []
        for ns in self._arm_namespaces:
            for cmd_type, name, want in expected:
                got = seam.operating_mode(
                    ns, cmd_type=cmd_type, name=name, timeout_s=self._stop_timeout_s
                )
                if got != want:
                    wrong.append(f"/{ns} {cmd_type} {name!r} is {got!r}, needs {want!r}")
        if wrong:
            raise ROSConfigError(
                "AlohaHAL refuses to command: "
                + "; ".join(wrong)
                + ". Switch it on the rig (upstream setup_follower_bot, or "
                "`ros2 service call /<ns>/set_operating_modes "
                "interbotix_xs_msgs/srv/OperatingModes "
                '"{cmd_type: single, name: gripper, mode: current_based_position}"`).'
            )
        log.info(
            "hal.aloha.operating_modes_verified",
            robot=self.description.name,
            arm_namespaces=list(self._arm_namespaces),
            arm_mode=_ARM_OPERATING_MODE,
            gripper_mode=_GRIPPER_OPERATING_MODE,
        )
        self._modes_verified = True

    def estop(self) -> None:
        """Trigger an emergency stop: drop the connection, cut torque on every arm, raise.

        The connection flag drops first so nothing more leaves this HAL, then
        ``torque_enable(enable=false)`` is called for the ``all`` group of
        every arm namespace. Every arm is attempted even if an earlier one
        failed, and the report names each acknowledgement. With no seam
        attached the report says so and ``stopped`` is ``False`` — a local
        latch is never reported as a downstream stop.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical("hal.estop", robot=self.description.name, arms=self._arm_namespaces)
        self._connected = False
        report = self._cut_torque()
        self._last_stop_report = report
        verdict = "acknowledged" if report.stopped else "NOT acknowledged"
        raise ROSEStopRequested(
            f"Emergency stop triggered on ALOHA bimanual ('{self.description.name}') "
            f"(downstream stop {verdict}: {report.detail})."
        )

    def _cut_torque(self) -> DownstreamStopReport:
        """Call ``torque_enable(false)`` on every arm and report per-arm acknowledgement."""
        arms = tuple(self._arm_namespaces)
        seam = self._torque_seam
        if seam is None:
            log.critical(
                "hal.estop.unproven",
                robot=self.description.name,
                reason="no InterbotixStopSeam attached; only the local latch holds",
            )
            return DownstreamStopReport(
                stopped=False,
                controllers=arms,
                detail="no torque stop seam attached; only the local latch holds",
            )
        states: dict[str, str] = {}
        problems: list[str] = []
        for arm in arms:
            try:
                ack = seam.torque_enable(
                    arm, group=_TORQUE_GROUP_ALL, enable=False, timeout_s=self._stop_timeout_s
                )
            except Exception as exc:  # reason: an rclpy fault on one arm must not skip the others
                ack = TriggerReport(success=False, message=f"{type(exc).__name__}: {exc}")
            states[arm] = "torque_off" if ack.success else "torque_unknown"
            if not ack.success:
                problems.append(f"{arm}: {ack.message or 'torque_enable not acknowledged'}")
        report = DownstreamStopReport(
            stopped=not problems,
            controllers=arms,
            controller_states=states,
            vendor_stop="torque_enable",
            detail="; ".join(problems) if problems else "torque off on every arm",
        )
        log.critical(
            "hal.estop.downstream_stopped"
            if report.stopped
            else "hal.estop.downstream_unacknowledged",
            robot=self.description.name,
            **report.fields(),
        )
        return report


# ── MuJoCo HAL (digital twin) ────────────────────────────────────────────────
# The gym-aloha bimanual sim twin is a thin
# ``MujocoArmHAL`` subclass — all wiring (MJCF URI, joint→qpos/
# actuator maps, two normalised grippers with mirror_actuator_index,
# keyframe seeding) lives in ``ALOHA_DESCRIPTION.sim``.


class AlohaMujocoHAL(MujocoArmHAL):
    """HAL adapter for the Trossen ALOHA bimanual setup (MuJoCo digital twin).

    Thin manifest-driven wrapper around ``MujocoArmHAL``; all wiring
    (MJCF URI, joint→qpos/actuator maps, two ``NORMALISED`` grippers with
    ``mirror_actuator_index`` for the antisymmetric finger pair, keyframe
    seeding) lives in ``ALOHA_DESCRIPTION.sim``.

    Public surface mirrors ``AlohaHAL``: a 14-DoF
    ``openral_core.Action`` with the
    ``left arm 6 + left gripper 1 + right arm 6 + right gripper 1``
    layout.  Gripper values are normalized ``[0, 1]`` (0 closed, 1 open),
    mapped affinely onto the finger range ``[0.021, 0.057]`` m and read
    back the same way.

    Args:
        mjcf_path: Optional override for the MJCF file.  When ``None``,
            the file is resolved through the ``gym_aloha:`` URI scheme
            from ``ALOHA_DESCRIPTION.assets.mjcf``.
        settle_steps: Number of MuJoCo physics steps per
            ``send_action`` call. ``None`` (default) derives one control period of
            ``description.control_rate_hz`` at ``connect`` (issue #358); an explicit
            count is a pinned, logged override.

        gravity_enabled: When ``False``, gravity is zeroed at
            ``connect()`` time for deterministic closed-loop tests.
        staleness_limit_s: Maximum age of a cached state.

    Example:
        >>> from openral_hal import AlohaMujocoHAL  # doctest: +SKIP
        >>> hal = AlohaMujocoHAL(gravity_enabled=False)  # doctest: +SKIP
        >>> hal.connect()  # doctest: +SKIP
        >>> state = hal.read_state()  # doctest: +SKIP
        >>> len(state.position)  # 14 joints  # doctest: +SKIP
        14
        >>> hal.disconnect()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        mjcf_path: str | None = None,
        settle_steps: int | None = None,
        gravity_enabled: bool = True,
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Initialise the adapter; the MJCF is not loaded until ``connect()``."""
        self._init_from_description(
            ALOHA_DESCRIPTION,
            mjcf_path=mjcf_path,
            settle_steps=settle_steps,
            gravity_enabled=gravity_enabled,
            staleness_limit_s=staleness_limit_s,
        )
