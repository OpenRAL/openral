"""HAL adapter for the Trossen ALOHA bimanual setup.

The physical ALOHA is two ViperX 300 6-DoF arms with parallel grippers
mounted side-by-side, exposing a 14-DoF joint-position action space
(2 * (6 arm + 1 gripper)).  The gym-aloha simulator uses an MJCF that mirrors
the same kinematics and scene layout, so the manifest and the in-code
:data:`ALOHA_DESCRIPTION` describe both the real robot and the simulator
one-to-one (CLAUDE.md / ADR-0007 robot/sim split).

This module wires the **real-hardware** Layer-0 path; the gym-aloha sim
path is owned by ``openral_sim.backends.aloha`` and invokes the
gym ``MjModel`` directly.

Driver landscape
----------------
The reference ROS 2 driver is
`Interbotix/interbotix_ros_manipulators`_, which provides a
``ros2_control`` joint trajectory controller (default name
``"arm_controller"``) per arm and a gripper position controller per
gripper.  ALOHA bring-up launches two robot namespaces (``"left_arm"`` /
``"right_arm"``) and we expose a single 14-DoF action by interleaving
left arm + left gripper, then right arm + right gripper, in the same
order as :data:`ALOHA_DESCRIPTION.joints`.

Per CLAUDE.md §7.4 the Trossen Interbotix XS SDK is BSD-3 / Apache-2.0
(fully compatible) but ships as vendor-distributed packages, so the
real-hardware manifest (:data:`ALOHA_REAL_DESCRIPTION`, derived from
:data:`ALOHA_DESCRIPTION` via :func:`make_real_description`) declares
``sdk_kind: "closed_with_api"``.  Both share the same ``hal`` block (ADR-0031):
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

import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    AssetRefs,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    GripperReadMode,
    GripperWriteMode,
    HalEntrypoints,
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

from openral_hal._base import HALBase
from openral_hal._mujoco_arm import MujocoArmHAL
from openral_hal._real_description import make_real_description

if TYPE_CHECKING:
    pass

__all__ = [
    "ALOHA_DESCRIPTION",
    "ALOHA_REAL_DESCRIPTION",
    "AlohaHAL",
    "AlohaMujocoHAL",
]

log = structlog.get_logger(__name__)

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
_ALOHA_GRIPPER_POSITION_LIMITS: tuple[float, float] = (0.0, 0.041)
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
            parent_link = "world" if suffix == "waist" else prev_link
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
                )
            )
            prev_link = child_link
        # Gripper: prismatic [0, 0.041] m per finger (Interbotix gripper spec).
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
        ),
        EndEffectorSpec(
            name="right_gripper",
            kind="parallel_gripper",
            hand=Hand.RIGHT,
            n_dof=1,
            max_grip_force_n=4.0,
            max_payload_kg=0.5,
            workspace_radius_m=0.6,
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
    ),
    sdk_kind="open",
    hal=HalEntrypoints(sim="openral_hal.aloha:AlohaMujocoHAL", real="openral_hal.aloha:AlohaHAL"),
    # MuJoCo wiring for the gym-aloha sim twin.  Two passthrough grippers
    # with mirror_actuator_index (positive finger + mirror to negative
    # finger).  keyframe_index=0 seeds the fingers inside their
    # ctrlrange — gym-aloha's reset does the same.  See ADR-0023.
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
                read_mode=GripperReadMode.PASSTHROUGH,
                write_mode=GripperWriteMode.PASSTHROUGH,
                actuator_index=6,
                mirror_actuator_index=7,
            ),
            SimGripperDescription(
                joint="right_gripper",
                ctrl_range=(0.021, 0.057),
                qpos_addrs=(14,),
                qpos_scale=0.036,
                read_mode=GripperReadMode.PASSTHROUGH,
                write_mode=GripperWriteMode.PASSTHROUGH,
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

# Default ros2_control controllers exported by the Interbotix XS launch
# files for ALOHA.  Each arm is its own controller_manager namespace, and
# the gripper is a separate position controller per arm.
_DEFAULT_LEFT_ARM_CONTROLLER: str = "left_arm/arm_controller"
_DEFAULT_RIGHT_ARM_CONTROLLER: str = "right_arm/arm_controller"
_DEFAULT_LEFT_GRIPPER_CONTROLLER: str = "left_arm/gripper_controller"
_DEFAULT_RIGHT_GRIPPER_CONTROLLER: str = "right_arm/gripper_controller"

# Default joint state topic — Interbotix XS aggregates each arm's joint
# states under its own namespace, then a top-level remap gathers them.
_DEFAULT_ALOHA_JOINT_STATE_TOPIC: str = "/joint_states"

# Trossen Interbotix XS exposes a torque-disable service per arm; the
# safety supervisor calls it on E-stop.  We publish a typed estop message
# to a shared topic so a watchdog node can react to either side.
_DEFAULT_ALOHA_ESTOP_TOPIC: str = "/aloha/estop"

_PublishFn = Callable[[str, dict[str, object]], None]
_StateFn = Callable[[], dict[str, object]]


class AlohaHAL(HALBase):
    """HAL adapter for a physical Trossen ALOHA bimanual setup.

    The 14-DoF action vector is split internally:

    * indices ``0:6``  → left arm joint trajectory  (left_arm controller)
    * index   ``6``    → left gripper position       (left gripper controller)
    * indices ``7:13`` → right arm joint trajectory  (right_arm controller)
    * index   ``13``   → right gripper position      (right gripper controller)

    The split allows the per-arm Interbotix controllers to handle each side
    independently (trajectory smoothing, gravity comp), while openral
    upstream layers see one unified 14-DoF action.

    Args:
        left_arm_controller: ``ros2_control`` joint trajectory controller
            for the left ViperX.  Defaults to ``"left_arm/arm_controller"``.
        right_arm_controller: same for the right arm.
        left_gripper_controller: gripper position controller for the left
            gripper.  Defaults to ``"left_arm/gripper_controller"``.
        right_gripper_controller: same for the right gripper.
        joint_state_topic: ROS 2 topic publishing aggregated joint state.
        estop_topic: ROS 2 topic the safety supervisor publishes to on
            ``estop()``.  A watchdog node downstream is expected to call
            the per-arm torque-disable service.
        publish_fn: Callable forwarding messages to ROS 2 topics.
            Production use injects the lifecycle node's publisher; tests
            inject :class:`SimTransport.publish`.
        state_fn: Callable returning the latest raw joint state as a dict.
            Production use injects the lifecycle node's subscriber
            callback; tests inject :class:`SimTransport.state`.
        staleness_limit_s: Maximum age of a ``read_state()`` reading
            before :class:`ROSPerceptionStale` is raised.

    Example:
        >>> from openral_hal.aloha import AlohaHAL
        >>> from openral_hal.sim_transport import SimTransport
        >>> transport = SimTransport(n_joints=14)
        >>> hal = AlohaHAL(
        ...     publish_fn=transport.publish,
        ...     state_fn=transport.state,
        ... )
        >>> hal.connect()
        >>> hal.description.name
        'aloha_bimanual'
        >>> hal.disconnect()
    """

    def __init__(
        self,
        *,
        left_arm_controller: str = _DEFAULT_LEFT_ARM_CONTROLLER,
        right_arm_controller: str = _DEFAULT_RIGHT_ARM_CONTROLLER,
        left_gripper_controller: str = _DEFAULT_LEFT_GRIPPER_CONTROLLER,
        right_gripper_controller: str = _DEFAULT_RIGHT_GRIPPER_CONTROLLER,
        joint_state_topic: str = _DEFAULT_ALOHA_JOINT_STATE_TOPIC,
        estop_topic: str = _DEFAULT_ALOHA_ESTOP_TOPIC,
        publish_fn: _PublishFn | None = None,
        state_fn: _StateFn | None = None,
        staleness_limit_s: float = 0.2,
    ) -> None:
        """Initialise the adapter; no transport is opened until ``connect()``."""
        self.description: RobotDescription = ALOHA_REAL_DESCRIPTION
        self._left_arm_controller = left_arm_controller
        self._right_arm_controller = right_arm_controller
        self._left_gripper_controller = left_gripper_controller
        self._right_gripper_controller = right_gripper_controller
        self._joint_state_topic = joint_state_topic
        self._estop_topic = estop_topic
        self._publish_fn: _PublishFn = publish_fn or _default_publish
        self._state_fn: _StateFn | None = state_fn
        self._staleness_limit_s = staleness_limit_s

        self._connected: bool = False
        self._last_state_time: float = 0.0
        self._joint_names: list[str] = list(_ALOHA_JOINT_NAMES)

    # ── HAL Protocol ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the transport to the four Interbotix ros2_control controllers.

        Raises:
            ROSRuntimeError: If already connected.
        """
        if self._connected:
            raise ROSRuntimeError("AlohaHAL is already connected.")
        log.info(
            "hal.connect",
            robot=self.description.name,
            left_arm=self._left_arm_controller,
            right_arm=self._right_arm_controller,
        )
        self._connected = True
        self._last_state_time = time.monotonic()

    def disconnect(self) -> None:
        """Close the transport.  Idempotent."""
        if not self._connected:
            return
        log.info("hal.disconnect", robot=self.description.name)
        self._connected = False

    def read_state(self) -> JointState:
        """Return the latest joint state for all 14 description joints.

        Raises:
            ROSRuntimeError: If not connected.
            ROSPerceptionStale: If the last reading is older than
                ``staleness_limit_s``.
        """
        self._require_connected("read_state")
        age = time.monotonic() - self._last_state_time
        if age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"Joint state is {age:.3f} s old (limit {self._staleness_limit_s} s)."
            )

        n = len(self._joint_names)
        raw: dict[str, object] = {} if self._state_fn is None else self._state_fn()

        def _floats(key: str) -> list[float]:
            val = raw.get(key)
            if isinstance(val, list):
                return [float(v) for v in val]
            return [0.0] * n

        return JointState(
            name=list(self._joint_names),
            position=_floats("position"),
            velocity=_floats("velocity"),
            effort=_floats("effort"),
            stamp_ns=int(time.time_ns()),
        )

    def send_action(self, action: Action) -> None:
        """Forward an action chunk to the four Interbotix controllers.

        The 14-D action is split four ways: each arm gets a 6-D joint
        trajectory and each gripper gets a 1-D position command.

        Raises:
            ROSRuntimeError: If not connected.
            ROSConfigError: If ``action.control_mode`` is not
                ``JOINT_POSITION`` or the joint count is wrong.
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
        left_arm_chunk = [step[0:6] for step in action.joint_targets]
        right_arm_chunk = [step[7:13] for step in action.joint_targets]
        left_gripper_targets = [step[6] for step in action.joint_targets]
        right_gripper_targets = [step[13] for step in action.joint_targets]

        self._publish_fn(
            f"/{self._left_arm_controller}/joint_trajectory",
            {
                "control_mode": action.control_mode,
                "horizon": action.horizon,
                "joint_targets": left_arm_chunk,
                "stamp_ns": action.stamp_ns,
            },
        )
        self._publish_fn(
            f"/{self._right_arm_controller}/joint_trajectory",
            {
                "control_mode": action.control_mode,
                "horizon": action.horizon,
                "joint_targets": right_arm_chunk,
                "stamp_ns": action.stamp_ns,
            },
        )
        self._publish_fn(
            f"/{self._left_gripper_controller}/command",
            {"position": left_gripper_targets[-1], "stamp_ns": action.stamp_ns},
        )
        self._publish_fn(
            f"/{self._right_gripper_controller}/command",
            {"position": right_gripper_targets[-1], "stamp_ns": action.stamp_ns},
        )
        log.debug(
            "hal.send_action",
            robot=self.description.name,
            horizon=action.horizon,
        )

    def estop(self) -> None:
        """Trigger an emergency stop on both arms.

        Publishes a structured estop message; downstream watchdog node
        calls the per-arm Interbotix torque-disable service.

        Raises:
            ROSEStopRequested: Always.
        """
        log.critical(
            "hal.estop",
            robot=self.description.name,
            estop_topic=self._estop_topic,
        )
        with contextlib.suppress(Exception):
            self._publish_fn(
                self._estop_topic,
                {"reason": "openral_estop", "robot": self.description.name},
            )
        self._connected = False
        raise ROSEStopRequested(
            f"Emergency stop triggered on ALOHA bimanual ('{self.description.name}')."
        )


def _default_publish(topic: str, msg: dict[str, object]) -> None:  # pragma: no cover
    """No-op publish used when no real ROS 2 node is wired in."""
    log.debug("hal.publish", topic=topic, fields=list(msg.keys()))


# ── MuJoCo HAL (digital twin) ────────────────────────────────────────────────
# Post-ADR-0023, the gym-aloha bimanual sim twin is a thin
# :class:`MujocoArmHAL` subclass — all wiring (MJCF URI, joint→qpos/
# actuator maps, two passthrough grippers with mirror_actuator_index,
# keyframe seeding) lives in :data:`ALOHA_DESCRIPTION.sim`.


class AlohaMujocoHAL(MujocoArmHAL):
    """HAL adapter for the Trossen ALOHA bimanual setup (MuJoCo digital twin).

    Thin manifest-driven wrapper around :class:`MujocoArmHAL`; all wiring
    (MJCF URI, joint→qpos/actuator maps, two ``PASSTHROUGH`` grippers with
    ``mirror_actuator_index`` for the antisymmetric finger pair, keyframe
    seeding) lives in :data:`ALOHA_DESCRIPTION.sim` (ADR-0023).

    Public surface mirrors :class:`AlohaHAL`: a 14-DoF
    :class:`openral_core.Action` with the
    ``left arm 6 + left gripper 1 + right arm 6 + right gripper 1``
    layout.  Gripper values are positive-finger metres in
    ``[0.021, 0.057]`` (passthrough); MuJoCo's ``ctrlrange`` clips
    out-of-range commands.

    Args:
        mjcf_path: Optional override for the MJCF file.  When ``None``,
            the file is resolved through the ``gym_aloha:`` URI scheme
            from :data:`ALOHA_DESCRIPTION.assets.mjcf`.
        settle_steps: Number of MuJoCo physics steps per
            :meth:`send_action` call.
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
        settle_steps: int = 1,
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
