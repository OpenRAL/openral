"""SimAttachedHAL — wrap any ``openral_sim.SimRollout`` as a HAL adapter.

Generic bridge between ``openral_sim.SimRollout`` and the
``openral_hal.HAL`` Protocol the ROS lifecycle nodes consume. ``SimAttachedHAL`` is
the one sim HAL shell (send path and slot-group staging, idle stepper, clock and
time base, estop latch, state / image / depth reads, scene export, step
bookkeeping, the attachment API, ``reset_episode``, per-goal placement grading),
for a scene backend and a bare twin alike: the MuJoCo twin, ``MujocoArmHAL``, is a
``SimAttachedHAL`` over a ``MujocoTwinRollout``. Runs the
HAL and the env in one process — a split HAL/env process pair left ``/scan``
unable to ray-cast live MJCF geometry (no handle to ``MjModel``/``MjData``) —
so the lifecycle node's ``mujoco_handle_provider`` can bind to
``self._hal.mujoco_handles()``.

``read_state`` walks ``description.joints``, resolving each via
``JointSpec.sim_joint_name`` (or a matching env name) against the MJCF;
``send_action`` calls ``env.step(...)`` through a per-robot
``ActionPacker`` (see ``pack_action_for_env``), so the HAL is
generic across any robot whose manifest declares the joint mapping.

Also owns the **task-success signal**: ``deploy sim`` suppresses the
backend's own per-step task evaluation, so ``task_success`` reads the
backend's predicate and ``_observe_task_success`` logs
``sim.task_success`` on every change plus a terminal
``sim.task_success_final`` at ``disconnect`` — reached on a
signal-driven teardown via
``openral_hal.lifecycle.HALLifecycleNodeBase.shutdown_hal`` (SIGINT runs
no lifecycle transition). Observability only (CLAUDE.md §1.4): never
termination, reset, reward, or the action path.

CLAUDE.md §1.5/§3: ``read_state``/``send_action`` must fit the robot's
control cycle — ``env.step`` is the slowest call (robocasa ~5-15 ms on the
reference host, within the 50 ms/20 Hz Nav2 cmd_vel budget). This is a HAL
adapter, not a safety shim: every action here has already been clamped by
the C++ kernel.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    BODY_TWIST_DIM,
    SIM_EXECUTABLE_CONTROL_MODES,
    AttachedCollisionObject,
    ClockAuthority,
    RobotDescription,
    check_time_base,
)
from openral_core.exceptions import (
    ROSConfigError,
    ROSError,
    ROSRuntimeError,
    ROSSafetyViolation,
)
from openral_core.schemas import Action, ControlMode, JointState

from openral_hal._base import require_connected
from openral_hal._floating_base import integrate_planar_twist, validate_planar_body_twist
from openral_hal._slot_group import (
    SlotGroupStager,
    compose_slot_group_action,
    slot_group_targets,
)
from openral_hal.mobile_base_bridge import describes_mobile_base
from openral_hal.mujoco_twin import joint_actuator_effort

log = structlog.get_logger(__name__)

_ZERO_TWIST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Dotted under ``openral.`` (not this module's own name): stdlib logger
# ancestry is dotted and openral_observability's structlog bridge attaches
# its OTel handler to ``logging.getLogger("openral")``; a sibling name like
# ``openral_hal.sim_attached`` never propagates to it, so the transition
# lines would be missing from `openral monitor` and the dashboard event ring.
_TASK_SUCCESS_LOGGER = "openral.sim.task_success"

# structlog event keys for the task-success signal. Grep targets: a
# validation run's artifacts are searched for these exact strings to
# decide whether the scene's task was actually completed.
_EVENT_TASK_SUCCESS = "sim.task_success"
_EVENT_TASK_SUCCESS_FINAL = "sim.task_success_final"
_EVENT_TASK_SUCCESS_PROBE_FAILED = "sim.task_success_probe_failed"
# Graded placement of the dispatched goal's prompt (``set_active_task``): each
# grade change, the goal's closing grade, and "this prompt is not graded".
_EVENT_PLACEMENT_GRADE = "sim.placement_grade"
_EVENT_PLACEMENT_GRADE_FINAL = "sim.placement_grade_final"
_EVENT_PLACEMENT_GRADE_UNAVAILABLE = "sim.placement_grade_unavailable"
# The time base a deploy-sim session runs on (one line per connect): what one
# env.step advances the sim clock by, against the manifest's control rate.
# Dotted under ``openral.`` for the same OTel-bridge reason as the task-success
# logger, but its own name so a logger filter does not mix the two.
_TIME_BASE_LOGGER = "openral.sim.time_base"
_EVENT_TIME_BASE = "sim.time_base"

if TYPE_CHECKING:
    from openral_core import IntrinsicsPinhole

    from openral_hal.sim_scene import SceneCache


class SimEnv(Protocol):
    """What ``SimAttachedHAL`` calls on its env directly; every other hook is ``getattr``-gated.

    An ``openral_sim.SimRollout`` satisfies it, and so does a ``MujocoTwinRollout``.
    """

    def reset(self, seed: int | None = ...) -> Any:  # noqa: ANN401  # reason: an Observation dict, or a backend's own
        """Reset the simulator; its initial observation."""

    def step(self, action: Any) -> Any:  # noqa: ANN401  # reason: a backend's action vector / StepResult
        """Apply one action; the transition."""

    def close(self) -> None:
        """Release the backend."""


def _task_success_logger() -> Any:  # noqa: ANN401  # reason: structlog's BoundLogger is untyped
    """Bind the task-success structlog logger.

    Bound per call rather than once at import: the unit suite reconfigures
    structlog's global processor pipeline per test, and a logger cached at
    import time would keep emitting through the pipeline that existed then.
    """
    return structlog.get_logger(_TASK_SUCCESS_LOGGER)


def _spec_id(spec: object) -> str | None:
    """Read ``.id`` off a SceneSpec / TaskSpec, or ``None`` when absent."""
    value = getattr(spec, "id", None)
    return None if value is None else str(value)


# The sim packers below (``pack_action_for_env`` and
# ``SimAttachedHAL._pack_with_composite_split``) collectively implement
# exactly this canonical set, plus the BODY_TWIST direct-qpos path in
# ``SimAttachedHAL.send_action``. Re-exported here so the provenance of the
# reasoner's sim palette gate is one import away from the packers it gates;
# the lockstep (both directions) is enforced by
# ``tests/unit/test_sim_executable_modes_match_packers.py``.
__all__ = [
    "SIM_EXECUTABLE_CONTROL_MODES",
    "ActionPacker",
    "SimAttachedHAL",
    "SimEnv",
    "gripper_targets_action",
    "normalized_joint_index",
    "pack_action_for_env",
]


def gripper_targets_action(group: list[Action], description: RobotDescription) -> Action | None:
    """What one committed slot group set for every gripper joint, as a compact action.

    The gripper-subset form of ``compose_slot_group_action`` for a group that does not
    compose into one whole joint-position command (a BODY_TWIST or Cartesian slot,
    arm joints left uncommanded): ``slot_group_targets`` places the group with the
    composer's own strictness, and the ``role: "gripper"`` joints are read out as a
    one-row ``JOINT_POSITION`` action (``joint_names`` = every gripper, manifest
    order) — the compact row ``VisionAttachmentBridge.observe_command`` reads
    positionally. Never a command to apply: the record of what the jaws were told.

    All or nothing: a partial record would leave the uncovered jaw's previous command
    standing in the grasp trigger (a phantom attach on the twin), so any gripper the
    group left uncommanded, or any slot the composer cannot read, gives ``None``,
    which clears every jaw's command.

    Args:
        group: Every slot action of the committed tick.
        description: The robot manifest (gripper roles and joint order).

    Returns:
        Every gripper's target, or ``None`` when the manifest has no gripper, the
        group left one uncommanded, or ``slot_group_targets`` refused the group
        (an unreadable or overlapping slot: which value the jaw got is unknown).

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/franka_panda/robot.yaml")
        >>> twist = Action(control_mode=ControlMode.BODY_TWIST, body_twist=[[0.1] + [0.0] * 5])
        >>> grip = Action(
        ...     control_mode=ControlMode.GRIPPER_POSITION, gripper=[0.02], ee_name="panda_gripper"
        ... )
        >>> applied = gripper_targets_action([twist, grip], d)
        >>> applied.joint_names, applied.joint_targets
        (['panda_gripper'], [[0.02]])
        >>> gripper_targets_action([twist], d) is None
        True
    """
    names = [joint.name for joint in description.joints]
    try:
        targets = slot_group_targets(group, names)
    except ROSConfigError:
        return None
    jaws: dict[str, float] = {}
    for joint, target in zip(description.joints, targets, strict=True):
        if joint.role == "gripper":
            if target is None:
                return None
            jaws[joint.name] = target
    if not jaws:
        return None
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[list(jaws.values())],
        joint_names=list(jaws),
        stamp_ns=group[0].stamp_ns,
        confidence=group[0].confidence,
    )


_ROBOSUITE_GROUP_PREFIX = re.compile(r"^[a-z]+[0-9]+_")  # robot0_ / gripper0_ / mobilebase0_


def normalized_joint_index(model_joint_names: list[str]) -> dict[str, int]:
    """Map MJCF joint names (exact + robosuite-prefix-stripped) to model index.

    Robosuite prefixes every joint with ``<class>N_``
    (``robot0_joint1``); native MJCFs do not. Exact names always win; a
    stripped name (``robot0_joint1`` -> ``joint1``) is added only when it
    neither shadows an exact name nor collides with another stripped name
    (a bimanual ``robot0_``/``robot1_`` model strips ambiguously -> keep
    explicit, require ``sim_joint_name``).

    Example:
        >>> normalized_joint_index(["robot0_joint1", "gripper0_finger_joint1"])
        {'robot0_joint1': 0, 'gripper0_finger_joint1': 1, 'joint1': 0, 'finger_joint1': 1}
    """
    exact: dict[str, int] = {name: i for i, name in enumerate(model_joint_names)}
    index: dict[str, int] = dict(exact)
    seen_stripped: dict[str, int] = {}
    ambiguous: set[str] = set()
    for i, name in enumerate(model_joint_names):
        stripped = _ROBOSUITE_GROUP_PREFIX.sub("", name)
        if stripped == name or stripped in exact:
            continue
        if stripped in seen_stripped:
            ambiguous.add(stripped)
            continue
        seen_stripped[stripped] = i
    for stripped, i in seen_stripped.items():
        if stripped not in ambiguous:
            index[stripped] = i
    return index


# How long a /cmd_vel-style BODY_TWIST setpoint keeps driving a kinematic base
# with no new command: diff_drive_controller's cmd_vel_timeout default (0.5 s), in
# sim time when the backend has a clock (_twist_hold_clock_ns). A publisher that
# dies stops the base within this window.
_BASE_TWIST_HOLD_NS = 500_000_000


# Atomic action-group fail-loud bounds: after this many CONSECUTIVE
# incomplete-group drops send_action raises (the sim has not stepped once —
# a slot-count contract mismatch or a persistently-rejected slot), and a
# pending partial group older than this releases the idle stepper (a skill
# that died mid-tick must not freeze scene physics/cameras forever). Inference
# ticks are budgeted <= ~1.5 s, so 5 s is unambiguously a dead tick.
_PENDING_GROUP_STALE_NS = 5_000_000_000


# ActionPacker is the per-composition translation between an OpenRAL
# `Action` chunk and the env's flat action vector. We model it as a
# plain callable for testability; the default factory below handles
# the BASIC-composite layout. The trailing ``prev`` arg (previous
# env-frame command, or None) is optional so a packer can carry an
# untouched slot — e.g. the gripper while the arm steps — across the
# two typed Actions one policy step splits into on a non-composite env.
ActionPacker = Callable[..., "np.ndarray[Any, np.dtype[np.float32]]"]


def _seed_from_prev(
    prev: np.ndarray[Any, np.dtype[np.float32]] | None, env_action_dim: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """A working env-vector seeded from the previous command, or zeros.

    Returns a fresh ``(env_action_dim,)`` zero vector when ``prev`` is missing
    or the wrong width (episode boundary / dim change); otherwise a copy of
    ``prev`` so untouched slots (e.g. the gripper while the arm steps) hold
    their last commanded value.
    """
    if prev is not None and prev.shape[0] == env_action_dim:
        return prev.copy()
    return np.zeros(env_action_dim, dtype=np.float32)


def pack_action_for_env(  # noqa: PLR0912  # reason: one branch per supported control mode; flat dispatch reads clearer than nested helpers
    action: Action,
    description: RobotDescription,
    env_action_dim: int,
    prev: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Default packer: translate an OpenRAL Action into the env action vector.

    Modes: ``CARTESIAN_DELTA`` (OSC arm delta into slots
    ``[base_dim:base_dim+6]``), ``GRIPPER_POSITION`` (last slot),
    ``BODY_TWIST`` (vx/vy/wz into slots 0-2), ``JOINT_POSITION`` (an
    arm-only row fills ``[base_dim:base_dim+arm_dim]``; a full
    ``base_dim+arm_dim`` row fills ``[0:base_dim+arm_dim]``). Other modes
    raise ``ROSConfigError`` — the lifecycle node enforces the supported set
    via ``RobotDescription.capabilities.supported_control_modes``. A
    caller needing richer translation (whole-body humanoid, dexterous-hand)
    passes its own ``ActionPacker``.

    Args:
        action: The chunk to pack; only the first row is consumed
            (chunk_size=1 is invariant on the wrapped-ROS dispatch path).
        description: Robot manifest; sizes the arm portion of the env vector.
        env_action_dim: The env's declared action_dim (typically 11 for the
            robosuite BASIC composite + gripper).
        prev: Env-frame action vector from the previous ``send_action`` in
            the same policy tick, or ``None``. A single policy step on a
            non-composite env (e.g. LIBERO OSC_POSE) arrives as two typed
            Actions — CARTESIAN_DELTA then GRIPPER_POSITION — each stepping
            the env separately; seeding from ``prev`` carries the last
            gripper command through the arm step instead of zeroing it
            (mirrors ``_pack_with_composite_split``'s merge) — without it
            the arm and gripper zero each other out across the two Actions.

    Returns:
        ``(env_action_dim,)`` float32 — the env-frame action vector.

    Raises:
        ROSConfigError: when `control_mode` is unsupported or its row width
            doesn't match the implied slot layout.
    """
    # Each mode reads its own Action field (not joint_targets[0] for all).
    # This packer's modes (CARTESIAN_DELTA, GRIPPER_POSITION, BODY_TWIST,
    # JOINT_POSITION) plus `_pack_with_composite_split` plus the BODY_TWIST
    # direct-qpos path union to exactly
    # `openral_core.SIM_EXECUTABLE_CONTROL_MODES` (lockstep test enforced).
    out = np.zeros(env_action_dim, dtype=np.float32)
    # Slot layout derives from the robot manifest (single source of
    # truth) rather than hardcoded panda dims: base width from
    # ``base_joints``; arm width = the joints that are neither base nor
    # gripper (the gripper parks in the LAST slot). Deriving the arm by
    # exclusion (rather than ``role == "arm"``) keeps it correct for
    # descriptions that don't annotate every joint's role. Generic
    # across mobile manipulators sharing the robosuite BASIC composite
    # (base slots first, then arm, gripper last).
    base_names = set(description.base_joints or [])
    base_dim = len(base_names)
    arm_dim = sum(1 for j in description.joints if j.name not in base_names and j.role != "gripper")
    if action.control_mode is ControlMode.CARTESIAN_DELTA:
        # RoboCasa PandaMobile env action layout (BASIC composite + OSC
        # arm + gripper, after the dim-12→11 skew adjustment in
        # ``openral_sim.backends.robocasa``):
        #   slots 0-2 = base (vx, vy, wz)
        #   slots 3-8 = arm OSC delta (xyz + axis-angle)
        #   slot   9  = robosuite torso (always -1 placeholder; we
        #               leave it 0 here — the backend's skew adapter
        #               appends the -1 when it sees env_action_dim=12)
        #   slot  10  = gripper width
        if not action.cartesian_delta:
            raise ROSConfigError("pack_action_for_env: empty Action.cartesian_delta")
        delta = list(action.cartesian_delta[0])
        cartesian_dim = 6
        if len(delta) != cartesian_dim:
            raise ROSConfigError(
                f"pack_action_for_env: CARTESIAN_DELTA row width must be "
                f"{cartesian_dim}, got {len(delta)}."
            )
        arm_base = base_dim
        if env_action_dim < arm_base + cartesian_dim:
            raise ROSConfigError(
                f"pack_action_for_env: env_action_dim={env_action_dim} "
                f"can't hold {arm_base + cartesian_dim} base+arm slots."
            )
        # Seed from the previous commanded vector so the last gripper (and
        # any base) command holds while the arm steps; only the arm's OSC
        # slots are rewritten. The OSC delta is per-step, so the arm slots
        # are zeroed first (a stale prev delta must not accumulate).
        out = _seed_from_prev(prev, env_action_dim)
        out[arm_base : arm_base + cartesian_dim] = 0.0
        for i, v in enumerate(delta):
            out[arm_base + i] = float(v)
        return out
    if action.control_mode is ControlMode.GRIPPER_POSITION:
        if not action.gripper:
            raise ROSConfigError("pack_action_for_env: empty Action.gripper")
        # Gripper parks at the LAST slot. Seed from prev (holding any base
        # command), zero everything between the base prefix and the gripper
        # slot so the arm HOLDS on the gripper step (no double-applied delta),
        # then set the gripper. Width-agnostic: holds whatever arm DOF the env
        # has (6-D OSC franka/widowx today, any future width) rather than a
        # fixed 6. Pairs with the CARTESIAN_DELTA branch so one policy step =
        # [arm,grip] then [hold,grip] — the arm advances once, the gripper is
        # always commanded.
        out = _seed_from_prev(prev, env_action_dim)
        out[base_dim : env_action_dim - 1] = 0.0
        out[-1] = float(action.gripper[0])
        return out
    if action.control_mode is ControlMode.BODY_TWIST:
        # body_twist now comes through the typed
        # ``Action.body_twist`` field, not joint_targets. The Nav2
        # cmd_vel bridge + the slot dispatcher both publish through
        # this field.
        if not action.body_twist:
            raise ROSConfigError("pack_action_for_env: empty Action.body_twist for BODY_TWIST")
        twist = list(action.body_twist[0])
        if len(twist) != BODY_TWIST_DIM:
            raise ROSConfigError(
                f"pack_action_for_env: BODY_TWIST row width must be "
                f"{BODY_TWIST_DIM}, got {len(twist)}."
            )
        if env_action_dim < base_dim:
            raise ROSConfigError(
                f"pack_action_for_env: env_action_dim={env_action_dim} is "
                f"too small for BODY_TWIST; need ≥ {base_dim} base slots "
                f"for (vx, vy, wz)."
            )
        out[0] = float(twist[0])  # vx
        out[1] = float(twist[1])  # vy
        # robosuite mobile-base BASIC composite uses slot 2 for yaw
        # velocity; BODY_TWIST's index 5 is wz.
        out[2] = float(twist[5])
        return out
    if not action.joint_targets:
        raise ROSConfigError("pack_action_for_env: empty Action.joint_targets")
    row = list(action.joint_targets[0])
    if action.control_mode is ControlMode.JOINT_POSITION:
        if len(row) == arm_dim:
            # Arm-only: drop into slots [base_dim:base_dim+arm_dim].
            if env_action_dim < base_dim + arm_dim:
                raise ROSConfigError(
                    f"pack_action_for_env: env_action_dim={env_action_dim} "
                    f"can't hold {base_dim + arm_dim} arm+base slots."
                )
            for i in range(arm_dim):
                out[base_dim + i] = float(row[i])
            return out
        full = base_dim + arm_dim
        if len(row) == full:
            for i in range(full):
                if i < env_action_dim:
                    out[i] = float(row[i])
            return out
        raise ROSConfigError(
            f"pack_action_for_env: JOINT_POSITION row width must be "
            f"{arm_dim} (arm-only) or {full} (base+arm); got {len(row)}."
        )
    raise ROSConfigError(
        f"pack_action_for_env: unsupported control_mode {action.control_mode!r}; "
        f"expected JOINT_POSITION / BODY_TWIST / CARTESIAN_DELTA / GRIPPER_POSITION."
    )


def _group_route(env: object) -> tuple[Callable[[list[Action]], Any] | None, int, bool]:
    """How ``SimAttachedHAL`` sends to a backend that takes typed commands (``step_action_group``).

    Decided from what the env declares (see ``openral_sim.rollout.SimRollout``):
    ``action_group_size`` > 0 groups every run of that many ungrouped slots (BEHAVIOR);
    ``steps_every_command_as_group`` takes an ungrouped action as a group of one (the
    MuJoCo twin); with neither, only multi-slot groups reach it and an ungrouped action
    is refused at send.

    Returns:
        ``(step_action_group or None, the fixed group size or 0, every command a group)``.
    """
    group_step = getattr(env, "step_action_group", None)
    if not callable(group_step):
        return None, 0, False
    size = int(getattr(env, "action_group_size", 0) or 0)
    if size > 0:
        return group_step, size, False
    return group_step, 0, bool(getattr(env, "steps_every_command_as_group", False))


def _group_twist_row(actions: list[Action]) -> list[float] | None:
    """The first ``BODY_TWIST`` slot's row of a command, ``None`` when it carries none."""
    return next(
        (
            list(a.body_twist[0])
            for a in actions
            if a.control_mode is ControlMode.BODY_TWIST and a.body_twist
        ),
        None,
    )


class SimAttachedHAL:
    """HAL Protocol adapter that wraps an in-process ``SimRollout``.

    Generic over robot embodiment. The active simulator is the source
    of truth for state; actions flow through ``pack_action_for_env``
    (or a caller-supplied ``ActionPacker``) into ``env.step()``.

    Args:
        env: The live simulator. Must implement
            ``openral_sim.rollout.SimRollout`` (``reset/step/
            mujoco_handles`` at minimum).
        description: Normative robot manifest. Used to populate
            ``JointState.name`` and to feed the
            ``HALLifecycleNodeBase`` OTel attributes.
        action_packer: Optional override for the default packer.
            Defaults to ``pack_action_for_env``.
        env_reset_seed: Seed forwarded to ``env.reset(seed=...)`` on
            ``connect``. ``None`` means "use the env's own
            default" (typically ``0`` or non-deterministic).
        env_action_dim: The env's flat action dimensionality. When
            ``None`` the HAL probes ``env.action_dim`` or
            ``env._env.action_dim`` on connect; if neither is available
            it raises ``ROSConfigError`` naming the backend (it
            never guesses a width). Pass this only for an env whose
            action space genuinely isn't introspectable.
    """

    description: RobotDescription
    # Frames on demand (``set_frame_period_s``): sim period between rendered steps and
    # the sim time of the last one; ``None`` = render every step. Immutable defaults.
    _frame_period_ns: int | None = None
    _last_frame_sim_ns: int | None = None
    # ``release_env`` ran (the kept-alive sidecar was handed over).
    _env_released: bool = False
    # Velocity setpoint of a kinematic base stepped through env.step (the Isaac
    # base): an ungrouped BODY_TWIST sets it and every idle step integrates it
    # until _held_twist_until_ns (see _apply_body_twist_via_env_step). Immutable
    # defaults; instances rebind them.
    _held_twist: tuple[float, float, float] | None = None
    _held_twist_until_ns: int = 0
    #: The Sim scene export per scene source (``export_sim_scene``), made on first use.
    _scene_cache: SceneCache | None = None
    #: A step finished and its post-step observers have not run yet (``_run_due_observers``).
    _observers_due: bool = False
    #: ``reset_episode`` can start a fresh episode (the HAL node advertises the service).
    episode_reset_supported: bool = True

    def __init__(
        self,
        env: SimEnv,
        description: RobotDescription,
        *,
        action_packer: ActionPacker | None = None,
        env_reset_seed: int | None = None,
        env_action_dim: int | None = None,
        body_twist_dt_s: float = 0.05,
    ) -> None:
        """Bind the env + description; no env interaction until ``connect``.

        Args:
            env: A ``SimRollout`` providing
                ``reset`` / ``step`` and (optionally) ``mujoco_handles``.
            description: The host ``RobotDescription`` — joint
                ordering, ``base_joints`` + ``sim_joint_name`` map.
            action_packer: Per-composition Action-to-env-vec translator;
                defaults to the env's own ``pack_action(action, prev)`` when it
                defines one (a backend that knows its slot layout — the Isaac
                manifest scene), else ``pack_action_for_env``.
            env_reset_seed: Optional seed forwarded to ``env.reset``.
            env_action_dim: Override the auto-probed env action width;
                useful for envs whose action space isn't introspectable.
            body_twist_dt_s: Logical control timestep used when a
                BODY_TWIST action is applied via direct base-qpos write.
                Mirrors ``PandaMobileHAL._dt_s`` (default 0.05 — 20 Hz
                nav control rate). In deploy-sim this is the action tick,
                not a render cadence or wall-clock sleep: each tick advances
                the base by ``velocity * dt`` and advances MuJoCo elapsed sim
                time by the same ``dt`` so `/clock`, odom, TF, and Nav2
                deadlines share one timestep definition.
        """
        self._env = env
        # Episodic backends (LIBERO) re-randomise the scene the instant
        # a task succeeds (lerobot's LiberoEnv.step resets inline). In a continuous
        # deploy twin the reasoner/mission own episode boundaries, not the env, so
        # ask the env to run continuously when it supports the hook (no-op for
        # backends without it).
        enable_continuous = getattr(env, "enable_continuous", None)
        if callable(enable_continuous):
            enable_continuous()
        self.description = description
        # A backend that declares its own layout packs by name (env hook, like
        # ``idle_action``); the default packer assumes the robosuite layout
        # (base first, ONE gripper in the last slot, per-step deltas) and would
        # misroute such an env's slots — an arm command into its base twist.
        env_pack = getattr(env, "pack_action", None)
        self._env_packs = action_packer is None and callable(env_pack)
        if action_packer is not None:
            self._action_packer: ActionPacker = action_packer
        elif callable(env_pack):
            self._action_packer = lambda action, _description, _dim, prev: env_pack(action, prev)
        else:
            self._action_packer = pack_action_for_env
        self._reset_seed = env_reset_seed
        # The send route for a backend that takes typed commands, decided once.
        self._group_step, self._fixed_group_size, self._every_command_is_group = _group_route(env)
        # One "dropped while latched" line per latch episode (estop .. reset_estop).
        self._latched_drop_logged = False
        # Episodes started by `reset_episode` since construction (its seed offset).
        self._episode_index = 0
        self._env_action_dim: int | None = env_action_dim
        self._connected: bool = False
        self._estop_latched: bool = False
        self._last_state_ns: int = 0
        # Monotonic timestamp (added by the 2026-06-04 idle-stepper amendment)
        # of the last command ``send_action`` accepted — staged or applied (the
        # single choke point both ``_on_safe_action`` and ``_on_cmd_vel``
        # reach). The sim-only free-running idle stepper reads this to yield
        # the env to an active skill: it skips an idle tick whenever a real
        # action arrived within the idle-hold window. A refused or latch-dropped
        # command, and a base velocity setpoint the idle stepper itself drives,
        # leave it. ``0`` until the first one (so an idle scene starts stepping).
        self._last_action_ns: int = 0
        # Cached observation from the most recent ``reset`` / ``step``.
        # Used by ``read_images`` so the lifecycle node's camera
        # publisher can republish whatever the env rendered without
        # re-stepping the simulator. ``None`` until ``connect``.
        self._last_obs: dict[str, Any] | None = None
        # Newest registered colour + depth frames (``read_depth_frames``): a backend
        # ships them on some steps only, so they outlive the step that carried them.
        self._last_depth_frames: dict[str, dict[str, Any]] = {}
        # name -> (the depth frame it came from, its base-frame cloud, the cloud's colours):
        # each frame is deprojected once, however often its cloud is read.
        self._depth_cloud_cache: dict[str, tuple[Any, NDArray[np.float32], Any]] = {}
        self._body_twist_dt_s: float = body_twist_dt_s
        # Sim seconds one env.step advances, when the backend declares it
        # (``env.sim_dt_per_tick_s``, the Isaac sidecar); checked against the
        # manifest control rate in ``connect``. ``None`` for a backend that
        # does not declare its tick.
        self._sim_dt_per_tick_s: float | None = None
        # Built once per env on first read_state; reset on connect.
        self._joint_index: dict[str, int] | None = None
        # Last action vector applied via composite-split
        # packing. Held across send_action calls so a per-mode chunk
        # that only fills one slot (e.g. CARTESIAN_DELTA arm) doesn't
        # silently zero out the other slots (e.g. gripper position),
        # which on the openral RoboCasa slot-dispatch path used to
        # cause the gripper to flick open between every arm tick. The
        # arm's OSC-delta slot is RE-ZEROED right before each step so
        # the policy's per-step delta is applied once, not accumulated.
        self._last_env_action: NDArray[np.float32] | None = None
        # Atomic slot-group staging, the one machine every HAL that reassembles a
        # tick uses (``SlotGroupStager``, shared with the real ros2_control HALs):
        # a tick's safety-approved slots, keyed by (runner_session_id, tick_index),
        # step the env exactly once when the group is complete; the committed
        # watermark refuses replays of grouped and ungrouped ticked actions alike,
        # and ``last_applied_action`` is the command each commit applied. Backends
        # that expose ``step_action_group(actions)`` receive the whole group.
        self._stager = SlotGroupStager()
        # Wall-clock stamp of the newest staged slot so a skill that dies
        # mid-tick cannot block ``idle_step`` forever.
        self._pending_since_ns: int = 0
        # send_action tick counter for the throttled diagnostic log.
        self._send_log_tick: int = 0
        # Complete atomic attachment snapshot plus exact MuJoCo body ids masked
        # from world perception while the same geometry remains collision-active.
        self._attached_objects: dict[str, AttachedCollisionObject] = {}
        self._attached_body_ids: frozenset[int] = frozenset()
        self._post_step_observers: list[Callable[[], None]] = []
        # Last commanded base body twist (vx, vy, vz, wx, wy, wz) in the
        # base_link frame. The base moves by exact Euler integration of
        # this command, so it IS the base velocity — the panda_mobile
        # lifecycle node publishes it as the ``/odom`` twist (REP-105:
        # twist is in the child frame). Zeroed when a non-BODY_TWIST
        # action is sent (the base is no longer velocity-commanded).
        self._last_body_twist: tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # Cross-reset sim-time offset. The wrapped
        # SimRollout's ``sim_time_ns`` reports time *within the current
        # episode*, and backends like robocasa rewind ``MjData.time`` to 0 on
        # every ``env.reset``. A ``/clock`` publisher must never see time go
        # backwards, so we accumulate each finished episode's elapsed sim-time
        # into this offset right before each reset (in
        # ``_accumulate_sim_time_before_reset``) and add it to the live
        # per-episode reading in ``sim_time_ns``. Stays ``0`` for a
        # clock-less wrapped rollout (whose every read is ``None``, so
        # ``sim_time_ns`` returns ``None`` and the offset is never used).
        self._sim_time_offset_ns: int = 0
        # ── Task-success witness (observability only) ────────────────────
        # ``deploy sim`` suppresses the backend's own per-step task
        # evaluation (``enable_continuous`` above), so nothing in the deploy
        # stack used to say whether the scene's task (e.g. "cup in the sink")
        # was ever completed — a validation run could prove the arm moved and
        # the object attached, but not that it was PLACED. These four slots
        # are the whole witness: last observed verdict, when it first read
        # True, how many times it flipped, and a latch that disables polling
        # after a backend's predicate raises. They gate logging only: nothing
        # here feeds termination, reset, reward, or the action path.
        self._task_success: bool | None = None
        self._task_success_first_ns: int | None = None
        self._task_success_transitions: int = 0
        self._task_success_probe_failed: bool = False
        # Count of ``env.step`` calls since ``connect`` — the step ordinal
        # reported alongside each task-success transition.
        self._step_count: int = 0
        # Graded placement (``set_active_task``; logging only): the dispatched goal's
        # prompt ("" between goals), the prompt being graded and its grade -> score
        # (None = ungraded), the grade the last step reported, the best one the goal
        # reached, and a latch for a backend whose ``set_task`` raised.
        self._active_task: str = ""
        self._placement_prompt: str = ""
        self._placement_scores: dict[str, float] | None = None
        self._placement_grade: str = "fail"
        self._placement_best: str = "fail"
        self._set_task_failed: bool = False

    # ── HAL Protocol surface ────────────────────────────────────────────

    def connect(self) -> None:
        """Reset the env and cache the initial obs as the seed for read_state.

        Idempotent: a second `connect` re-resets the env at the same seed
        (the lifecycle node calls `connect` on each `configure` → `cleanup`
        cycle, so the contract must tolerate repeated calls).
        """
        # Fold any elapsed sim-time into the cross-reset
        # offset BEFORE the reset rewinds the backend clock, so a re-connect
        # (the lifecycle node re-resets on each configure→cleanup cycle) never
        # makes the published ``/clock`` jump backwards. On the very first
        # connect a freshly built env reads ~0, so this is a no-op there.
        # The estop latch survives a reconnect: only ``reset_estop`` clears it, so a
        # configure → cleanup → configure cycle never re-arms a stopped robot.
        self._reset_env(self._reset_seed, "connect")
        try:
            if self._env_action_dim is None:
                self._env_action_dim = self._probe_env_action_dim()
            self._check_time_base()
        except BaseException:
            self._connected = False  # a refused connect leaves the HAL disconnected
            raise

    def export_sim_scene(self) -> dict[str, object] | None:
        """``/openral/viz/sim_scene``'s payload for the live scene, or ``None`` (no exporter).

        The Isaac sidecar's ``scene_mesh``, else ``mujoco_scene_mesh`` on the env's live
        model in the frame ``odom_from_base`` names, through a ``SceneCache``: exported on
        the first call (the lifecycle node latches the topic at configure), again after a
        reset that composed a new ``MjModel`` (an unpinned RoboCasa kitchen) or a failed
        export. Never raises but ``ROSSafetyViolation`` (``write_sim_scene``).
        """
        from openral_hal.sim_scene import (  # noqa: PLC0415  # reason: sim only
            SceneCache,
            mujoco_scene_mesh,
            odom_from_base,
        )

        if self._scene_cache is None:
            self._scene_cache = SceneCache()
        isaac = getattr(self._env, "scene_mesh", None)
        if callable(isaac):
            return self._scene_cache.get(self._env, isaac)
        handles = self._mujoco_handles()
        if handles is None:
            return None
        model, data = handles
        return self._scene_cache.get(
            model,
            lambda limit: mujoco_scene_mesh(
                model, data, self.description, limit, odom_from_base(self)
            ),
        )

    def _check_time_base(self) -> None:
        """Refuse an env whose step is not one control period of the robot.

        Only a backend that declares ``sim_dt_per_tick_s`` (the Isaac sidecar,
        RoboCasa, LIBERO, a ``MujocoTwinRollout`` held to its control rate) is
        checked: the runner ticks at ``action_spec.control_freq_hz`` and steps the env
        once per tick,
        so a step that advances the sim by any other interval traverses every
        trajectory at the wrong speed and every kernel velocity / tracking
        check reads the wrong clock (issues #355, #358). The rule itself is
        ``openral_core.check_time_base``. Logs ``sim.time_base`` once per
        connect so a trace records the time base it was captured on, with the
        backend's ``time_base_fields`` (the twin's HAL class as ``backend``, its physics
        dt and steps per tick).

        Raises:
            ROSConfigError: the manifest declares no control rate, or the env's
                tick differs from ``1 / control_freq_hz`` by more than
                ``TIME_BASE_TOL`` (1 %).
        """
        dt = getattr(self._env, "sim_dt_per_tick_s", None)
        if dt is None:
            self._sim_dt_per_tick_s = None
            return
        dt = float(dt)
        backend = type(self._env).__name__
        rate = self.description.control_rate_hz
        check_time_base(
            dt, rate, backend=f"{type(self).__name__}[{backend}]", robot=self.description.name
        )
        self._sim_dt_per_tick_s = dt
        fields = dict(getattr(self._env, "time_base_fields", None) or {})
        # The checked values win: a backend field of the same name (#700 item 13) is
        # dropped rather than passed twice (a raw TypeError out of connect()).
        record: dict[str, object] = {
            **fields,
            "backend": fields.get("backend", backend),
            "sim_dt_per_tick_s": round(dt, 6),
            "control_freq_hz": rate,
            "robot": self.description.name,
        }
        structlog.get_logger(_TIME_BASE_LOGGER).info(_EVENT_TIME_BASE, **record)

    @property
    def sim_dt_per_tick_s(self) -> float | None:
        """Sim seconds one command advances, once held to the control rate (``connect``).

        ``None`` before connect, or for a backend that declares no tick (a twin with a
        pinned ``settle_steps`` or a manifest without a rate is wall-paced).
        """
        return self._sim_dt_per_tick_s

    def _reset_env(self, seed: int | None, caller: str) -> None:
        """``env.reset(seed)`` and reset the HAL's per-episode state."""
        # Read before the reset, folded into the offset only once it succeeded: a
        # refused reset leaves the clock where it was (and a retry counts nothing twice).
        elapsed = self._rollout_sim_time_ns()
        try:
            obs = self._env.reset(seed=seed)
        except ROSError:
            raise  # a typed refusal (a twin's MJCF that does not load) keeps its type
        except Exception as exc:
            raise ROSRuntimeError(
                f"{type(self).__name__}.{caller}: env.reset failed: {exc}"
            ) from exc
        if elapsed is not None:
            self._sim_time_offset_ns += elapsed
        self._last_obs = dict(obs) if isinstance(obs, dict) else None
        self._keep_depth_frames(self._last_obs)
        self._last_state_ns = time.time_ns()
        self._connected = True
        # A half-staged tick never outlives a reset; the watermark is cleared by
        # ``disconnect`` / ``reset_episode`` only (a stop is not a renumbering).
        self._stager.discard()
        # The reset re-seeded the scene: commands from before it must not ride
        # along as the next pack's ``prev`` (an absolute-target env would move
        # the arm back to its pre-reset pose on the first base twist).
        self._last_env_action = None
        # Nor may a held base setpoint drive the new episode, or /odom report its twist.
        self._held_twist = None
        self._last_body_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Every object is back at its spawn pose: nothing is held, nothing masked.
        self._attached_objects = {}
        self._attached_body_ids = frozenset()
        self._joint_index = None  # rebuilt on next read_state (model identity stable per env)
        # A reset re-randomises the scene, so the previous episode's success
        # verdict no longer describes anything live. Re-seed the witness from
        # the freshly reset env (normally False) WITHOUT logging a transition:
        # the drop from a finished episode's True back to the new episode's
        # False is not a task being undone.
        self._task_success = self._probe_task_success()
        self._task_success_first_ns = None
        self._task_success_transitions = 0
        self._step_count = 0

    def _require_connected(self, operation: str) -> None:
        """Refuse ``operation`` before ``connect`` / after ``disconnect``.

        Raises:
            ROSRuntimeError: The HAL is not connected (``_base.require_connected``).
        """
        require_connected(self._connected, type(self).__name__, operation)

    def _probe_env_action_dim(self) -> int:
        """Return the env's flat action dimensionality, or raise.

        Delegates to ``openral_sim.rollout.env_action_dim`` (the one probe
        the sim runner's mock policies also size from): ``env.action_dim``,
        then ``env._env.action_dim``, then a wrapped gym ``Box`` shape.

        If neither resolves and no ``env_action_dim`` override was supplied
        to the constructor, raises ``ROSConfigError`` naming the
        backend — a loud boot-time failure beats a wrong-width mid-run
        E-stop; this method never guesses a width (e.g. falling back to 11,
        the robosuite BASIC composite width).

        Raises:
            ROSConfigError: the env exposes no introspectable ``action_dim``
                and no ``env_action_dim`` override was supplied.
        """
        from openral_sim.rollout import env_action_dim  # noqa: PLC0415  # reason: optional dep

        dim = env_action_dim(self._env)
        if dim is not None:
            return dim
        backend = type(self._env).__name__
        raise ROSConfigError(
            f"{type(self).__name__}: cannot resolve the env action width — backend "
            f"{backend!r} exposes no `action_dim` (nor does its inner `_env`), "
            "and no `env_action_dim` override was supplied to the constructor. "
            "Add an `action_dim` property to the backend's rollout (reporting "
            "its true `step` width) or pass `env_action_dim` explicitly. Refusing "
            "to guess a width — a wrong guess E-stops mid-run on the first env.step."
        )

    def disconnect(self) -> None:
        """Idempotent — release the env handle (we don't own its lifetime).

        Emits the terminal ``sim.task_success_final`` line first (see
        ``task_success``), so a deploy-sim session against a backend
        that HAS a task-success predicate always closes with one greppable
        statement of whether the scene's task ended completed. Idempotent
        because the line only fires while still connected.

        Reached on BOTH teardown paths: the lifecycle ``cleanup`` /
        ``shutdown`` transition, and — since ``rclpy`` answers SIGINT by
        shutting the context without running any transition —
        ``openral_hal.lifecycle.HALLifecycleNodeBase.shutdown_hal`` in the
        node ``main()``'s ``finally``. The signal path is the one every real
        ``openral deploy sim`` run takes; before it existed the verdict was
        emitted by no field run at all.
        """
        if self._connected:
            self._emit_task_success_final()
            if self._active_task:
                self._emit_placement_grade_final()
        self._connected = False
        # The tick numbering restarts with the next connection.
        self._stager.reset()

    def release_env(self) -> None:
        """Hand a ``keep_alive`` backend's sidecar over to the next launch (process teardown).

        Called once by the lifecycle node as its process exits, never on a cleanup that a
        configure may follow. Only a backend the scene opted in (``backend_options.keep_alive``)
        is closed: its ``close`` detaches the sidecar from this process's exit reaper, so the
        next deploy sim of the same scene adopts it instead of booting Kit again. Any other
        env is left to its owner, as before. Idempotent.
        """
        env = self._env
        if getattr(env, "keep_alive", False) is not True or self._env_released:
            return
        self._env_released = True
        env.close()

    def read_state(self) -> JointState:
        """Read live joint state from the env's MJCF qpos via the description.

        Walks `description.joints` and looks up each joint's
        `sim_joint_name` (falling back to `name`) in the env's MJCF.
        Returns the canonical ``JointState`` the safety supervisor
        + the world_state aggregator consume.

        ``effort`` is ``joint_actuator_effort`` per joint (MuJoCo
        ``qfrc_actuator``; N·m or N) — what a current-sensing drive reports,
        gravity load included. A non-MuJoCo backend fills it from
        ``obs["joint_efforts"]``. All or nothing: if any joint is unmeasured (absent
        from the MJCF, no ``joint_efforts`` key, or a short vector) ``effort`` is
        empty — never a zero a torque observer would read as a free joint.

        A backend with no MJCF handle is read from its observation
        (``_read_state_from_obs``), and refuses rather than publish zeros.

        Raises:
            ROSRuntimeError: when called before `connect`, or when a backend with
                no MJCF handle reports no measured position for some joint.
        """
        self._require_connected("read_state")
        # A backend that reads its own joint state (``read_joint_state``, the
        # bare MuJoCo twin: manifest addresses, grippers in public units) owns it.
        read_joint_state = getattr(self._env, "read_joint_state", None)
        if callable(read_joint_state):
            state: JointState = read_joint_state()
            return state
        positions: list[float] = []
        velocities: list[float] = []
        handles = self._mujoco_handles()
        if handles is None:
            return self._read_state_from_obs()
        model, data = handles
        import mujoco  # noqa: PLC0415  # reason: optional dep guarded by caller

        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        efforts: list[float] = []
        measured = True
        if self._joint_index is None:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(int(model.njnt))
            ]
            self._joint_index = normalized_joint_index([n for n in names if n])
        for joint in self.description.joints:
            sim_name = joint.sim_joint_name or joint.name
            jid = self._joint_index.get(sim_name, -1)
            if jid < 0:
                # Joint not in this MJCF (maybe excluded composition);
                # contribute a zero so the position vector length stays
                # right.
                positions.append(0.0)
                velocities.append(0.0)
                measured = False
                continue
            dof = int(model.jnt_dofadr[jid])
            positions.append(float(qpos[int(model.jnt_qposadr[jid])]))
            velocities.append(float(qvel[dof]))
            efforts.append(joint_actuator_effort(data, dof))
        self._last_state_ns = time.time_ns()
        return JointState(
            name=[j.name for j in self.description.joints],
            position=positions,
            velocity=velocities,
            effort=efforts if measured else [],
            stamp_ns=self._last_state_ns,
        )

    def _read_state_from_obs(self) -> JointState:
        """``read_state`` for a backend with no MJCF handle (Isaac, BEHAVIOR, RoboTwin, ...).

        Reads ``obs["joint_positions"]`` (and ``joint_velocities`` / ``joint_efforts``).
        With ``obs["joint_names"]`` the vectors are keyed by name and each description
        joint is looked up by ``name``, then ``sim_joint_name``; without it they must be
        in description-joint order and exactly one value per joint.

        Position is all or refuse: a missing key, a short or over-long vector, or a
        joint the backend does not report raises, because the safety kernel, the
        self-collision check and world state read this as the measured pose (a
        fabricated zero pose is a pose the robot is not in). Velocity and effort are
        all or nothing: unmeasured means an empty list, never zeros.

        Raises:
            ROSRuntimeError: The backend reports no joint positions for some joint.
                Runtime, not ``ROSPerceptionStale``: the backend's observation never
                carries them, so waiting for a fresher one cannot recover.
        """
        joints = self.description.joints
        obs = self._last_obs or {}
        raw_names = obs.get("joint_names")
        index: dict[str, int] | None = None
        n_reported = 0
        if raw_names is not None:
            n_reported = len(raw_names)
            reported = {str(n): i for i, n in enumerate(raw_names)}
            index = {}
            for j in joints:
                for key in (j.name, j.sim_joint_name):
                    if key is not None and key in reported:
                        index[j.name] = reported[key]
                        break

        def _vector(key: str) -> list[float] | None:
            raw = obs.get(key)
            if raw is None:
                return None
            vec = np.asarray(raw, dtype=np.float64).reshape(-1)
            if index is not None:
                if len(index) != len(joints) or vec.shape[0] != n_reported:
                    return None
                return [float(vec[index[j.name]]) for j in joints]
            if vec.shape[0] != len(joints):
                return None
            return [float(v) for v in vec]

        position = _vector("joint_positions")
        if position is None:
            missing = [j.name for j in joints if index is not None and j.name not in index]
            raise ROSRuntimeError(
                f"{type(self._env).__name__} reports no measured joint positions for "
                f"robot {self.description.name!r} (obs keys {sorted(obs)}; "
                f"joint_positions width {np.asarray(obs.get('joint_positions', [])).size}, "
                f"{len(joints)} description joints"
                + (f"; unreported joints {missing}" if missing else "")
                + "). Refusing to publish a fabricated /joint_states: the backend's "
                "SimRollout must emit obs['joint_positions'] for every description joint."
            )
        return JointState(
            name=[j.name for j in joints],
            position=position,
            velocity=_vector("joint_velocities") or [],
            effort=_vector("joint_efforts") or [],
            stamp_ns=time.time_ns(),
        )

    def add_post_step_observer(self, observer: Callable[[], None]) -> None:
        """Register a synchronous observer invoked after each simulator step."""
        if observer not in self._post_step_observers:
            self._post_step_observers.append(observer)

    def send_action(self, action: Action) -> None:
        """Step the env with the packed action vector.

        The post-step observers run once the command is committed, so they read its
        applied action and tick watermark, and one that raises cannot leave a stepped
        tick uncommitted (a resend would step it again).

        Args:
            action: Per-tick chunk from the safety supervisor. Already
                envelope-clamped; this HAL forwards verbatim.

        Raises:
            ROSConfigError: when the action's `control_mode` isn't one
                the default packer accepts (or whatever the
                caller-supplied `action_packer` rejects).
            ROSRuntimeError: when called before `connect` or when
                `env.step` raises.
        """
        try:
            self._send_action(action)
        finally:
            self._run_due_observers()

    def _send_action(self, action: Action) -> None:
        """``send_action`` up to the commit; observers excluded."""
        self._require_connected("send_action")
        if self._estop_latched:
            self._log_latched_drop(action)
            return
        if self._env_action_dim is None:
            raise ROSRuntimeError(
                f"{type(self).__name__}.send_action: env_action_dim resolved to None; "
                "re-connect or pass it explicitly to the constructor."
            )
        if action.tick_group_size > 1 or self._fixed_group_size > 0:
            self._stage_action_group(action)
            return
        group_step = self._group_step
        if group_step is not None and not self._every_command_is_group:
            raise ROSConfigError(
                f"{type(self).__name__}: {type(self._env).__name__} steps typed slot groups "
                "(step_action_group) but declares neither action_group_size nor "
                "steps_every_command_as_group: an ungrouped action needs a positive group size."
            )
        # Ungrouped ticked actions (starting-pose ramp, approach) share the
        # watermark, so a replay is refused and a restarted runner is seen here too.
        self._stager.admit(action)
        self._check_twists([action])
        if group_step is not None:
            self._step_env_group([action], group_step)
            self._commit_ungrouped(action)
            return
        # BODY_TWIST direct-qpos path: robosuite's BASIC controller doesn't
        # interpret ``pack_action_for_env``'s slots 0-2 as OmronMobileBase
        # planar velocities, so mirror ``PandaMobileHAL._apply_body_twist``
        # instead — rotate the body-frame twist into world frame, Euler-
        # integrate by ``body_twist_dt_s``, write the base qpos slots
        # directly, and skip ``env.step()`` so the arm dynamics don't churn.
        # JOINT_POSITION still flows through ``env.step()``.
        if action.control_mode is ControlMode.BODY_TWIST:
            assert action.body_twist  # _check_twists
            row = [float(v) for v in action.body_twist[0]]
            # MuJoCo integrates the base by direct qpos write (skips env.step so the
            # arm dynamics don't churn). A non-MuJoCo backend (Isaac kinematic base)
            # has no qpos handle — it integrates the base inside env.step, so the
            # twist becomes a held velocity setpoint the idle stepper drives: it does
            # not mark the HAL active (that would stop the very stepper moving it).
            if self._mujoco_handles() is not None:
                self._apply_body_twist_to_qpos(row)
                self._commit_ungrouped(action)
            else:
                self._apply_body_twist_via_env_step(row)
                self._stager.commit_tick(action)
            return
        # Robosuite composite controllers expose part-name
        # → action-slot mapping via ``cc._action_split_indexes``. The
        # legacy ``pack_action_for_env`` hardcoded the PandaMobile+BASIC
        # layout (action_dim=11, arm at slots [3:9]); switching the
        # robot to PandaOmron+HybridMobileBase (action_dim=12, arm at
        # slots [0:6]) silently rerouted per-mode bytes into wrong slots
        # so the env stepped with garbage. Prefer the composite's own
        # split when available — mirrors the upstream
        # ``RoboCasaGymEnv.step + unmap_action`` flow.
        if self._has_composite_split():
            # The composite packer maintains ``_last_env_action`` internally.
            env_action = self._pack_with_composite_split(action)
        else:
            # Non-composite env (e.g. LIBERO OSC_POSE): the legacy packer is
            # stateless, so thread the previous command through ``prev`` and
            # latch the result here. This carries the gripper across the arm
            # step the way the composite path does — without it the arm and
            # gripper zero each other out across the two typed Actions a
            # single policy step splits into.
            env_action = self._action_packer(
                action, self.description, self._env_action_dim, self._last_env_action
            )
            self._last_env_action = env_action.copy()
        # One stdout line per first chunk + every 50th — gives us the
        # smoke trail we needed to diagnose the prior "arm doesn't move"
        # silent failure without spamming the hot path. The previous
        # `_on_safe_action` lie (every chunk → JOINT_POSITION) was
        # invisible because nothing logged at this layer; that's how it
        # cost a debug session. ``print`` (not stdlib logging) is used
        # so the line survives ros2 launch's subprocess stdout capture —
        # Python's stdlib logging is unconfigured in those subprocesses,
        # so a stdlib ``logging`` call would vanish.
        self._send_log_tick += 1
        if self._send_log_tick == 1 or self._send_log_tick % 50 == 0:
            head = ",".join(f"{float(v):+.3f}" for v in env_action[:8])
            print(
                f"[sim_attached.send_action] tick={self._send_log_tick} "
                f"mode={action.control_mode.value} env_dim={len(env_action)} "
                f"head=[{head}]",
                flush=True,
            )
        # A non-BODY_TWIST action leaves the base still, unless a held /cmd_vel
        # setpoint is live (arm actions suppress the idle stepper that applies it).
        env_action = self._drive_held_twist(env_action)
        self._step_and_cache(env_action, source="send_action")
        self._commit_ungrouped(action)

    def _commit_ungrouped(self, action: Action) -> None:
        """An ungrouped command was applied: it is the record, and the HAL is active."""
        self._stager.commit_tick(action)
        self._mark_active()

    def _mark_active(self) -> None:
        """Stamp ``last_action_ns``: a command was accepted (staged or applied)."""
        self._last_action_ns = time.monotonic_ns()

    def _log_latched_drop(self, action: Action) -> None:
        """Say once per latch episode that commands are being dropped (CLAUDE.md §1.4)."""
        if self._latched_drop_logged:
            return
        self._latched_drop_logged = True
        log.warning(
            "hal.send_action.dropped_estop_latched",
            backend=type(self).__name__,
            robot=self.description.name,
            control_mode=action.control_mode.value,
            tick_index=int(action.tick_index),
            hint="the estop latch drops every command until reset_estop()",
        )

    def _check_twists(self, actions: list[Action] | tuple[Action, ...]) -> None:
        """Refuse every ``BODY_TWIST`` a planar sim base cannot execute, before any steps.

        ``_floating_base.validate_planar_body_twist`` (the one rule): a 6-float row, no
        linear-z / angular-x / angular-y, and no lateral ``vy`` on a manifest-declared
        differential base — a sim base would slide sideways where the robot cannot.

        Raises:
            ROSConfigError: A twist the rule refuses.
        """
        differential = self.description.base_kinematics == "differential"
        for action in actions:
            if action.control_mode is ControlMode.BODY_TWIST:
                validate_planar_body_twist(action, type(self).__name__, differential=differential)

    def discard_staged_slots(self) -> int:
        """Drop a half-staged action group, keeping the committed watermark.

        Called by the HAL lifecycle node at every ``/openral/estop`` latch and
        ``/openral/estop_cleared`` (``_discard_staged_slots``), as for the MuJoCo and
        real HALs: a tick cut short by a kernel stop never outlives it into the next
        goal, whose first slot would otherwise report an incomplete group. A pre-stop
        tick replayed afterwards is still refused as stale.

        Returns:
            How many staged slots were dropped.
        """
        return self._stager.discard()

    def _stage(self, action: Action, group_size: int | None = None) -> list[Action] | None:
        """Stage one slot in ``SlotGroupStager``; the whole tick once complete, else ``None``.

        The stager's rules apply (replays refused, a lost slot reported once and the
        next tick opened from the slot that exposed it, a size change discarding the
        tick, a superseded runner's partial tick dropped and logged). A slot the stager
        took — staged, adopted, or completing its tick — marks the HAL active; one left
        staged also restarts the stale-group timer ``idle_step`` reads. A refused slot
        does neither, so it cannot keep a dead half-staged tick alive.

        Raises:
            ROSConfigError: The action carries no ``tick_index`` (``SlotGroupStager.stage``).
            ROSRuntimeError: A replay, an incomplete, overrun or resized tick.
        """
        try:
            group = self._stager.stage(action, group_size=group_size)
        finally:
            staged = self._stager.staged
            if staged and staged[-1] is action:
                self._pending_since_ns = time.monotonic_ns()
                self._mark_active()
        if group is not None:
            self._mark_active()
        return group

    def _stage_action_group(self, action: Action) -> None:
        """Commit one simulator step after every slot in an inference tick is safe."""
        actions = self._stage(
            action, self._fixed_group_size if action.tick_group_size <= 1 else None
        )
        if actions is None:
            return
        # Refused before either the env packer or ``step_action_group`` sees it.
        self._check_twists(actions)
        if self._group_step is None:
            self._step_packed_action_group(actions)
            self._commit_group(actions)
            return
        self._commit_group(actions, self._step_env_group(actions, self._group_step))

    def _step_boundary(self, step: Callable[[], Any], source: str) -> Any:  # noqa: ANN401  # reason: the backend's own transition (StepResult, TwinStepResult, bool)
        """One simulator step inside the shell's step boundary, for every route.

        Frames on demand (``_request_frames``) before it; after it the step
        bookkeeping (``_finish_step``) unless ``step`` returned ``False`` (an idle hook
        that did not advance). A typed ``ROSError`` (a refused command, a twin's
        ``ROSSafetyViolation``) propagates as raised; anything else is a backend
        failure, ``ROSRuntimeError``.

        Returns:
            What ``step`` returned.
        """
        rendered = self._request_frames()
        try:
            result = step()
        except ROSError:
            raise
        except Exception as exc:
            raise ROSRuntimeError(
                f"{type(self).__name__}.{source}: env.step failed: {exc}"
            ) from exc
        if result is not False:
            self._finish_step(result, rendered)
        return result

    def _step_env_group(
        self, actions: list[Action], group_step: Callable[[list[Action]], Any]
    ) -> Action | None:
        """One ``step_action_group`` call inside the step boundary; latches ``base_twist``.

        Returns:
            The command the backend reports it applied (``info["applied_action"]``),
            or ``None`` when it reports none.
        """
        step_result = self._step_boundary(lambda: group_step(actions), "send_action_group")
        # The group path bypasses the per-mode branches that maintain the twist.
        self._latch_twist(_group_twist_row(actions))
        info = getattr(step_result, "info", None)
        applied = info.get("applied_action") if isinstance(info, dict) else None
        return applied if isinstance(applied, Action) else None

    def _latch_twist(self, row: list[float] | tuple[float, ...] | None) -> None:
        """``base_twist`` (the /odom twist) := ``(vx, vy, 0, 0, 0, wz)`` of ``row``, else zero."""
        self._last_body_twist = (
            (float(row[0]), float(row[1]), 0.0, 0.0, 0.0, float(row[5]))
            if row is not None and len(row) >= BODY_TWIST_DIM
            else _ZERO_TWIST
        )

    def _commit_group(self, actions: list[Action], applied: Action | None = None) -> None:
        """Move the watermark and record the group's command as one full-dof action.

        The command is what the backend reports it applied (``applied``), else
        ``compose_slot_group_action`` over the manifest joints, as the real HAL composes
        it. A group that is not one whole joint-position command (a BODY_TWIST or
        Cartesian slot, joints it leaves uncommanded) records what it set
        for the gripper joints instead (``gripper_targets_action``), so the grasp trigger
        still reads the jaw command on a mobile manipulator; ``None`` — command unknown,
        every jaw's command in the trigger is cleared — unless every gripper was commanded
        readably (never a partial record that leaves one jaw's stale command standing).
        """
        if applied is None:
            try:
                applied = compose_slot_group_action(
                    actions, [joint.name for joint in self.description.joints]
                )
            except ROSConfigError:
                applied = gripper_targets_action(actions, self.description)
        self._stager.commit(actions, applied=applied)

    def _step_packed_action_group(self, actions: list[Action]) -> None:
        """Pack every safe slot, then execute exactly one simulator step."""
        has_twist = any(action.control_mode is ControlMode.BODY_TWIST for action in actions)
        if has_twist and not self._env_packs:
            raise ROSConfigError(
                f"{type(self).__name__}: BODY_TWIST cannot share a packed env-step group; "
                "use the backend's step_action_group implementation."
            )
        env_action: NDArray[np.float32] | None = None
        if self._env_packs:
            # The env packs by name; slots after the first carry the twist an
            # earlier slot of this same step packed (a mobile-manipulation tick).
            env_pack = self._env.pack_action  # type: ignore[attr-defined]  # reason: duck-typed env hook, checked in __init__
            for i, action in enumerate(actions):
                env_action = env_pack(
                    action, self._last_env_action if i == 0 else env_action, carry_twist=i > 0
                )
            if env_action is not None:
                self._last_env_action = env_action.copy()
            self._latch_twist(_group_twist_row(actions))
        elif self._has_composite_split():
            for action in actions:
                env_action = self._pack_with_composite_split(action)
        else:
            # Nothing commanded yet: an env that packs by name gets None (its
            # own HOLD); the default packer's robosuite envs read 0 as hold.
            previous: NDArray[np.float32] | None = (
                self._last_env_action.copy()
                if self._last_env_action is not None
                else None
                if self._env_packs
                else np.zeros(int(self._env_action_dim or 0), dtype=np.float32)
            )
            for action in actions:
                if (
                    previous is not None  # always, unless the env packs by name
                    and not self._env_packs
                    and action.control_mode is ControlMode.GRIPPER_POSITION
                    and action.gripper
                ):
                    env_action = previous.copy()
                    env_action[-1] = float(action.gripper[0])
                else:
                    env_action = self._action_packer(
                        action,
                        self.description,
                        int(self._env_action_dim or 0),
                        previous,
                    )
                previous = env_action
            if env_action is not None:
                self._last_env_action = env_action.copy()
        if env_action is None:
            raise ROSConfigError(
                f"{type(self).__name__}: packed action group produced no env action."
            )
        # A twist-free group keeps a live /cmd_vel setpoint driving.
        env_action = self._drive_held_twist(env_action, twist_commanded=has_twist)
        self._step_and_cache(env_action, source="send_action_group")

    def _step_and_cache(self, env_action: NDArray[np.float32], *, source: str) -> bool:
        """Step the persistent deploy-sim environment and cache its observation.

        ``deploy sim`` is a long-lived deployment twin, not an evaluator. It
        never interprets terminal/success signals and never resets after
        startup. Backends that refuse post-terminal commands surface their
        error explicitly instead of silently starting a new episode.

        The backend's task-success predicate IS read here, once per step,
        and logged on every change (``_observe_task_success``). That is
        an observation, not an interpretation: the verdict reaches the log
        and nothing else — not termination, not reset, not the action path.

        Args:
            env_action: The flat env-frame action vector to step with.
            source: Caller tag (``"send_action"`` / ``"idle_step"``) used in
                the diagnostic log + the wrapped ``ROSRuntimeError`` messages.

        Returns:
            ``True`` once the env has been stepped (callers that may early-out
            before reaching here return ``False`` themselves).

        Raises:
            ROSError: ``env.step`` raised a typed error (it propagates as raised).
            ROSRuntimeError: ``env.step`` raised anything else.
        """
        if source == "idle_step":
            # A hold step: in a shared sidecar the other robots never wait for it
            # (openral-pro#99); a backend without the hook steps as before.
            mark_idle = getattr(self._env, "set_idle_next_step", None)
            if callable(mark_idle):
                mark_idle()
        self._step_boundary(lambda: self._env.step(env_action), source)
        return True

    def _finish_step(self, step_result: object, rendered: bool) -> None:
        """The bookkeeping after every simulator step, once per step.

        Caches the transition, counts the step, polls the task-success witness and
        marks the post-step observers due (``send_action`` / ``idle_step`` run them
        after the commit). ``rendered`` is what ``_request_frames`` said
        before the step.
        """
        if rendered:
            self._last_frame_sim_ns = self._rollout_sim_time_ns()
        # ``StepResult.observation`` carries the rendered camera frames
        # (see ``openral_sim.rollout.StepResult``). Cache so the
        # lifecycle node's camera publisher can serve them without
        # re-stepping the simulator. A dict-shaped Protocol fallback
        # ``getattr(..., 'observation', None)`` is enough because every
        # in-tree backend returns a ``StepResult`` with this attribute.
        self._cache_step_result(step_result)
        self._step_count += 1
        self._observe_task_success()
        self._observers_due = True

    def _run_due_observers(self) -> None:
        """Run the post-step observers once after a step, from the public entry point."""
        if not self._observers_due:
            return
        self._observers_due = False
        for observer in self._post_step_observers:
            try:
                observer()
            except ROSSafetyViolation:
                raise  # never caught here (CLAUDE.md §5)
            except Exception as exc:  # reason: the tick is already committed and applied
                # Raising would report an applied tick as unapplied: the node would
                # skip its /openral/action_applied ack, and a resend is refused as stale.
                log.error(
                    "hal.post_step_observer_failed",
                    robot=self.description.name,
                    observer=getattr(observer, "__qualname__", repr(observer)),
                    tick=self.last_committed_tick,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    def set_frame_period_s(self, period_s: float | None) -> None:
        """Render camera frames only every ``period_s`` of sim time (``None``: every step).

        For a backend that renders on demand (``env.set_frames_next_step``, Isaac):
        the sensor bridge passes its fastest camera/depth publish period, so the scene
        renders ``"fresh"`` frames (showing that step's state) once per publish
        instead of a lagging frame every control tick. Other backends ignore it.

        Example:
            >>> SimAttachedHAL.set_frame_period_s.__name__
            'set_frame_period_s'
        """
        self._frame_period_ns = None if period_s is None else int(period_s * 1e9)

    def refresh_frames(self) -> bool:
        """Render the current state's frames now, without stepping the simulation.

        For a backend that renders on demand (``env.refresh_frames``, Isaac): a
        waiter that holds the runner (the attachment perception barrier) gets a
        fresh frame instead of waiting for the next stepped one. Returns ``False``
        (nothing done) for a backend without it.

        Example:
            >>> SimAttachedHAL.refresh_frames.__name__
            'refresh_frames'
        """
        refresh = getattr(self._env, "refresh_frames", None)
        if not callable(refresh):
            return False
        obs = refresh()
        if isinstance(obs, dict):
            self._last_obs = dict(obs)
            self._keep_depth_frames(self._last_obs)
            self._last_frame_sim_ns = self._rollout_sim_time_ns()
        return True

    def _request_frames(self) -> bool:
        """Pick the next step's frame mode on an on-demand backend; ``True`` if it renders."""
        set_frames = getattr(self._env, "set_frames_next_step", None)
        if not callable(set_frames) or self._frame_period_ns is None:
            return True
        now = self._rollout_sim_time_ns()
        dt_ns = int((self._sim_dt_per_tick_s or 0.0) * 1e9)
        last = self._last_frame_sim_ns
        # Due when a period has passed by the end of this step (half a tick early so a
        # period that is a whole number of ticks does not slip one), or the sim clock
        # went back (an episode reset).
        due = (
            now is None
            or last is None
            or now < last
            or now + dt_ns - last >= self._frame_period_ns - dt_ns // 2
        )
        set_frames("fresh" if due else "none")
        return due

    def _cache_step_result(self, step_result: object) -> None:
        """Cache one backend transition without evaluating or resetting it."""
        obs = getattr(step_result, "observation", None)
        if isinstance(obs, dict):
            self._last_obs = dict(obs)
            self._keep_depth_frames(self._last_obs)
        info = getattr(step_result, "info", None)
        if self._placement_scores is not None and isinstance(info, dict):
            self._observe_placement_grade(info.get("placement_grade"))

    def _keep_depth_frames(self, obs: dict[str, Any] | None) -> None:
        """Keep the newest ``"depth_frames"`` an observation carried (see ``read_depth_frames``)."""
        frames = obs.get("depth_frames") if obs is not None else None
        if isinstance(frames, dict) and frames:
            self._last_depth_frames = {str(k): v for k, v in frames.items() if isinstance(v, dict)}

    # ── Task-success signal (observability only) ────────────────────────

    def task_success(self) -> bool | None:
        """Return the wrapped backend's own task-success verdict, or ``None``.

        Introspection accessor, same shape as ``mujoco_handles``: the
        ``task_success`` extension is optional on
        ``SimRollout``, so this forwards when the
        backend defines it and returns ``None`` otherwise.

        The value is the simulator's ground truth — for RoboCasa, the task
        class's own ``_check_success()`` against live MuJoCo state ("is the
        cup inside the sink basin") — not an estimate and not the reward
        monitor's opinion. It is a live read, NOT a latch: a task that
        succeeds and is then undone reads ``False`` again. Use the
        ``sim.task_success`` log lines (which carry ``first_success``) to
        recover "did it ever succeed".

        ``None`` means "this backend exposes no task-success predicate", or
        that a previous read raised and polling latched off. It must never
        be read as failure.

        Returns:
            The backend's verdict, or ``None`` when unavailable.
        """
        return self._probe_task_success()

    def _probe_task_success(self) -> bool | None:
        """Read the backend predicate defensively, latching off on failure.

        A backend's ``_check_success`` runs task-specific geometry against
        live MuJoCo state; if it raises (an env mid-reset, a task whose
        predicate assumes an object the scene did not spawn) that must not
        take down the actuation path this is polled from. So the first
        failure is reported once at WARNING and disables further polling
        for the life of this HAL — never a silent swallow, never a repeated
        per-step log flood.
        """
        if self._task_success_probe_failed:
            return None
        read = getattr(self._env, "task_success", None)
        if not callable(read):
            return None
        try:
            value = read()
        except Exception as exc:  # reason: an observability read must never break actuation
            self._task_success_probe_failed = True
            # Mirrored to stdout for the same reason the signal itself is
            # (see ``_emit_task_success``): "the signal is missing" and
            # "the signal says no" must never look alike in a run's log.
            self._emit_task_success(
                _EVENT_TASK_SUCCESS_PROBE_FAILED,
                level="warning",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        return None if value is None else bool(value)

    def _observe_task_success(self) -> None:
        """Poll the predicate once and log ``sim.task_success`` on any change.

        Called once per ``env.step``. Emits on BOTH edges, not just
        ``False -> True``: RoboCasa success is not latched (an object can be
        knocked back out of the sink), so a run that logged only the rising
        edge would read as a success it no longer holds. Each line carries
        ``success`` (the new verdict) and ``first_success`` (whether this is
        the first ``True`` of the session), so "did it ever complete" and
        "was it complete at the end" are both answerable from the log.
        """
        value = self._probe_task_success()
        if value is None or value == self._task_success:
            return
        previous = self._task_success
        self._task_success = value
        self._task_success_transitions += 1
        first_success = value and self._task_success_first_ns is None
        sim_time_ns = self.sim_time_ns()
        if first_success:
            self._task_success_first_ns = sim_time_ns if sim_time_ns is not None else -1
        self._emit_task_success(
            _EVENT_TASK_SUCCESS,
            success=value,
            previous=previous,
            first_success=bool(first_success),
            sim_time_ns=sim_time_ns,
            sim_time_s=None if sim_time_ns is None else round(sim_time_ns / 1e9, 3),
            step=self._step_count,
        )

    def _emit_task_success_final(self) -> None:
        """Log the terminal ``sim.task_success_final`` verdict for the session.

        Re-reads the predicate rather than trusting the last polled value:
        the MuJoCo ``BODY_TWIST`` path advances the sim by writing base qpos
        directly instead of through ``_step_and_cache``, so the verdict can
        move without a poll. Falls back to the last observed value when the
        live read is unavailable (probe latched off, env already torn down).

        No-op when neither is available — a line claiming ``success=False``
        for an env that never had a success criterion would be a fabricated
        verdict, not an observation.
        """
        live = self._probe_task_success()
        final = self._task_success if live is None else live
        if final is None:
            return
        first_ns = self._task_success_first_ns
        self._emit_task_success(
            _EVENT_TASK_SUCCESS_FINAL,
            success=final,
            ever_succeeded=first_ns is not None,
            first_success_sim_time_ns=None if first_ns is None or first_ns < 0 else first_ns,
            transitions=self._task_success_transitions,
            sim_time_ns=self.sim_time_ns(),
            steps=self._step_count,
        )

    # ── Graded placement of the dispatched goal (observability only) ─────

    def _observe_placement_grade(self, grade: object) -> None:
        """Log ``sim.placement_grade`` when the graded prompt's grade changes."""
        scores = self._placement_scores
        if scores is None or not isinstance(grade, str) or grade == self._placement_grade:
            return
        previous = self._placement_grade
        self._placement_grade = grade
        if scores.get(grade, 0.0) > scores.get(self._placement_best, 0.0):
            self._placement_best = grade
        sim_time_ns = self.sim_time_ns()
        self._emit_task_success(
            _EVENT_PLACEMENT_GRADE,
            prompt=self._placement_prompt,
            grade=grade,
            score=scores.get(grade, 0.0),
            previous=previous,
            goal_active=self._active_task == self._placement_prompt,
            sim_time_ns=sim_time_ns,
            sim_time_s=None if sim_time_ns is None else round(sim_time_ns / 1e9, 3),
            step=self._step_count,
        )

    def _emit_placement_grade_final(self) -> None:
        """Log the ending goal's ``sim.placement_grade_final`` (no-op when it is ungraded)."""
        scores = self._placement_scores
        if scores is None or self._placement_prompt != self._active_task:
            return
        self._emit_task_success(
            _EVENT_PLACEMENT_GRADE_FINAL,
            prompt=self._placement_prompt,
            grade=self._placement_grade,
            score=scores.get(self._placement_grade, 0.0),
            best_grade=self._placement_best,
            best_score=scores.get(self._placement_best, 0.0),
            sim_time_ns=self.sim_time_ns(),
            steps=self._step_count,
        )

    def _emit_task_success(self, event: str, *, level: str = "info", **fields: object) -> None:
        """Emit one task-success line on both paths a deploy run is read from.

        1. ``structlog`` under ``openral.sim.task_success`` — the structured
           record, which ``install_structlog_bridge`` ships as OTLP so the
           line lands in the monitor dashboard's event log.
        2. ``print`` — because that structlog record does NOT reach the
           ``ros2 launch`` console. The bridge attaches its OTel handler to
           the ``openral`` stdlib logger and nothing in a launched HAL
           subprocess configures a stdout handler, so an INFO record is
           exported and never printed. The same reasoning already forces the
           ``send_action`` diagnostic above onto ``print``. A verdict that
           only exists inside the collector is not recoverable from a
           validation run's artifacts, which is the entire point of this
           signal — so it is mirrored as ``<event> <json>``, matching the
           ``sim.estop_ground_truth_snapshot`` line the sibling
           ``openral_hal.sim_sensor_bridge`` writes into the same log.
        """
        payload: dict[str, object] = {
            "scene_id": self._scene_id(),
            "task_id": self._task_id(),
            **fields,
        }
        getattr(_task_success_logger(), level)(event, **payload)
        print(f"{event} {json.dumps(payload, sort_keys=True)}", flush=True)

    def _scene_id(self) -> str | None:
        """The wrapped rollout's scene id (``robocasa/PickPlaceCounterToSink``)."""
        return _spec_id(getattr(self._env, "scene", None))

    def _task_id(self) -> str | None:
        """The wrapped rollout's task id.

        On the ``deploy sim`` path this is the synthesised
        ``<scene_id>/_hal_deploy_noop`` (see
        ``openral_hal.sim_bringup.build_sim_env_from_yaml``) — the scene
        id is the identifying half there, which is why both are logged.
        """
        return _spec_id(getattr(self._env, "task", None))

    def step_estop_hold(self) -> bool:
        """One tick while the e-stop is latched (the node's latch or this HAL's own).

        ``True`` when the backend's stop was stepped (one robot of a shared world,
        ADR-0137), ``False`` when the robot freezes (a world of its own). The post-step
        observers run after it, as after ``idle_step``.
        """
        try:
            return self._step_estop_hold()
        finally:
            self._run_due_observers()

    def _step_estop_hold(self) -> bool:
        """While latched, one robot of a shared world steps its backend's stop (ADR-0137).

        The other robots keep stepping its world, so freezing only its observation would
        leave its ``/joint_states`` stale for every peer while its world moves on.
        ``estop_hold_action`` keeps every joint target and stops the base; ``False``
        (freeze) when the backend offers none, as for a world of its own.
        """
        hold_getter = getattr(self._env, "estop_hold_action", None)
        hold = hold_getter(self._last_obs) if callable(hold_getter) else None
        if hold is None or self._env_action_dim is None:
            return False
        hold = np.asarray(hold, dtype=np.float32).reshape(-1)
        if hold.shape != (self._env_action_dim,):
            raise ROSRuntimeError(
                f"{type(self).__name__}.idle_step: backend estop_hold_action width "
                f"{hold.shape[0]} != env_action_dim {self._env_action_dim}."
            )
        stepped = self._step_and_cache(hold, source="idle_step")
        if stepped:
            self._last_body_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return stepped

    def idle_step(self, wall_dt_s: float | None = None) -> bool:
        """Advance the wrapped sim one tick with a zero/HOLD action when idle (``_idle_step``).

        The post-step observers run after it, as after ``send_action``.
        """
        try:
            return self._idle_step(wall_dt_s)
        finally:
            self._run_due_observers()

    def _idle_step(self, wall_dt_s: float | None = None) -> bool:
        """Advance the wrapped sim one tick with a zero/HOLD action when idle.

        SIM-ONLY. Refreshes ``_last_obs`` (camera frames + state) so
        perception sees a live scene between skill ticks — without this the
        env only steps on ``/openral/safe_action`` receipt, so an idle scene
        freezes physics and cameras go stale.

        Safety: defined ONLY on the sim shell (``SimAttachedHAL``) — real HALs don't
        define it, and ``SimSensorBridge``
        gates its idle timer on ``callable(getattr(hal, "idle_step", None))``,
        so the timer is never created against a real HAL. That method-only
        exclusion is the actual safety guarantee, not "zero is harmless": a
        zero vector is a HOLD for the sim's velocity/OSC-delta controllers but
        would command "drive every joint to 0 rad" on a real absolute-position
        arm. Does not touch ``_last_env_action`` — an idle HOLD is orthogonal
        to the last skill command. A held base setpoint (an ungrouped
        ``BODY_TWIST`` on a base the env integrates, the Isaac kinematic base) is
        packed into the twist slots and reported as ``base_twist`` until
        ``_BASE_TWIST_HOLD_NS`` after its last command; otherwise the base is
        still and ``base_twist`` is zero.

        Caveat: zero is a true HOLD only for velocity/OSC-delta/composite
        controllers. A position-controlled native backend (``so101_box``)
        drives joints toward 0 rad instead of holding — acceptable here since
        the goal is to keep the scene rendering, not freeze the arm. A backend
        can opt out via ``idle_action()`` returning its own hold vector (the
        BEHAVIOR-1K backend returns current joint targets).

        ``_env_action_dim`` is resolved by ``_probe_env_action_dim`` from
        the backend's own ``action_dim``; a backend with no width and no
        override fails loudly at ``connect``, so the idle tick never builds a
        wrong-width zero vector.

        A backend with its own ``idle_step(wall_dt_s)`` hook (the bare MuJoCo
        twin: a wall-time slice of physics, ``ctrl`` untouched) is advanced by it
        instead of an ``idle_action`` step, through the same step bookkeeping.

        Args:
            wall_dt_s: Wall-clock seconds this tick represents. Handed to the
                backend's ``idle_step`` hook; unused otherwise — a wrapped
                ``SimRollout`` owns its own per-``step`` sim-time stride.

        Returns:
            ``True`` if the env was stepped, ``False`` if suppressed (not
            connected, estop latched, or action dim unresolved).
            Backend-agnostic — no MuJoCo-handle gate.
        """
        if not self._connected:
            return False
        if self._estop_latched:
            # An estopped HAL freezes — honoring the estop contract is correct;
            # the frozen scene is the intended, safe behaviour here. Except where
            # the world does not freeze with it (ADR-0137, ``_step_estop_hold``).
            return self._step_estop_hold()
        if self._env_action_dim is None:
            return False
        staged = self._stager.staged
        if staged:
            # A tick's slots are mid-flight — never interleave a HOLD step
            # inside an atomic group. But a group whose slots stopped arriving
            # (skill died mid-tick) must not freeze the scene forever: past
            # the staleness bound, discard it loudly and resume idle stepping.
            if time.monotonic_ns() - self._pending_since_ns < _PENDING_GROUP_STALE_NS:
                return False
            print(
                f"[sim_attached.idle_step] ERROR discarding stale pending action group "
                f"(session, tick)={(staged[0].runner_session_id, staged[0].tick_index)} "
                f"slots={len(staged)} (no new slot for "
                f"{_PENDING_GROUP_STALE_NS / 1e9:.0f}s); resuming idle stepping",
                flush=True,
            )
            self._stager.discard()
        env_idle_step = getattr(self._env, "idle_step", None)
        if callable(env_idle_step):
            stepped = self._step_boundary(lambda: bool(env_idle_step(wall_dt_s)), "idle_step")
            if stepped:
                self._latch_twist(None)  # a HOLD: the base is still
            return bool(stepped)
        # No MuJoCo-handle gate: valid for ANY wrapped SimRollout, including
        # non-MuJoCo backends (Isaac Sim sidecar, ManiSkill3). Steps via the
        # same env.step idiom as send_action's tail (not robocasa.refresh_obs,
        # which re-renders without stepping), so `_step_and_cache` keeps
        # observation caching identical across command and idle paths.
        idle_action_getter = getattr(self._env, "idle_action", None)
        idle_action: NDArray[np.float32] = (
            np.asarray(idle_action_getter(), dtype=np.float32).reshape(-1)
            if callable(idle_action_getter)
            else np.zeros(self._env_action_dim, dtype=np.float32)
        )
        if idle_action.shape != (self._env_action_dim,):
            raise ROSRuntimeError(
                f"{type(self).__name__}.idle_step: backend idle_action width "
                f"{idle_action.shape[0]} != env_action_dim {self._env_action_dim}."
            )
        held = self._live_held_twist()
        if held is not None:  # a held /cmd_vel setpoint keeps driving the base
            idle_action = self._with_twist(idle_action, held)
        stepped = self._step_and_cache(idle_action, source="idle_step")
        if stepped:
            vx, vy, wz = held if held is not None else (0.0, 0.0, 0.0)
            self._last_body_twist = (vx, vy, 0.0, 0.0, 0.0, wz)
        return stepped

    # ── Per-mode → composite-controller slot mapping ──────────
    def _composite_controller(self) -> Any:  # noqa: ANN401  # reason: robosuite composite controller is an untyped third-party object
        """Return the (single) robot's composite controller, or None.

        Peels through openral SimRollout wrappers (e.g.
        ``openral_sim.backends.robocasa._RoboCasaSim``) that hold the
        actual robosuite env at ``self._env._env``. Without this the
        composite-split path is dormant in deploy_sim and chunks fall
        back to the legacy BASIC-composite ``pack_action_for_env``,
        which mis-slots HybridMobileBase actions.
        """
        # Try the bound env, then walk one level of wrapping.
        for candidate in (self._env, getattr(self._env, "_env", None)):
            if candidate is None:
                continue
            robots = getattr(candidate, "robots", None)
            if robots:
                return getattr(robots[0], "composite_controller", None)
        return None

    def _has_composite_split(self) -> bool:
        """True iff the bound env exposes the robosuite composite split."""
        cc = self._composite_controller()
        return cc is not None and hasattr(cc, "_action_split_indexes")

    def _part_slot(self, part: str) -> tuple[int, int] | None:
        """Return ``(start, end)`` slot range for a composite part, or None."""
        cc = self._composite_controller()
        if cc is None:
            return None
        split = getattr(cc, "_action_split_indexes", None)
        if not split:
            return None
        rng = split.get(part)
        if rng is None:
            return None
        return (int(rng[0]), int(rng[1]))

    def _pack_with_composite_split(  # noqa: PLR0912, PLR0915  # reason: one branch per composite slot; mirrors the upstream gym wrapper's flat packer
        self, action: Action
    ) -> NDArray[np.float32]:
        """Pack a typed Action into the env action vector via composite slots.

        Uses the composite controller's authoritative ``cc._action_split_indexes``
        slot map, mirroring upstream ``RoboCasaGymEnv.step`` /
        ``PandaOmronKeyConverter.unmap_action``. Layout-agnostic: works for
        PandaMobile+BASIC (action_dim=11) and PandaOmron+HybridMobileBase
        (action_dim=12) without per-layout branching.

        Handles CARTESIAN_DELTA (arm), GRIPPER_POSITION (finger),
        JOINT_VELOCITY (base), COMPOSITE_MODE, and delegates JOINT_POSITION to
        ``pack_action_for_env`` — the full
        ``openral_core.SIM_EXECUTABLE_CONTROL_MODES`` set except BODY_TWIST
        (own direct-qpos path). Any other mode raises (lockstep-test-enforced
        drift guard).

        Slots from previous ``send_action`` calls in the same policy tick
        persist in ``self._last_env_action`` so multi-Action dispatch (arm +
        gripper) doesn't zero each other out; the arm's OSC-delta slot is
        re-zeroed each call so the per-step delta applies once, not
        accumulated.
        """
        env_dim = int(self._env_action_dim or 0)
        if self._last_env_action is None or self._last_env_action.shape[0] != env_dim:
            self._last_env_action = np.zeros(env_dim, dtype=np.float32)
        out = self._last_env_action.copy()
        cc = self._composite_controller()
        from robosuite.controllers.composite.composite_controller import (  # noqa: PLC0415
            HybridMobileBase,
        )

        if action.control_mode is ControlMode.CARTESIAN_DELTA:
            if not action.cartesian_delta:
                raise ROSConfigError("_pack_with_composite_split: empty Action.cartesian_delta")
            delta = list(action.cartesian_delta[0])
            slot = self._part_slot("right")
            if slot is None:
                raise ROSConfigError(
                    "_pack_with_composite_split: composite has no 'right' part — "
                    f"split keys: {list(getattr(cc, '_action_split_indexes', {}).keys())}",
                )
            lo, hi = slot
            width = hi - lo
            if len(delta) != width:
                raise ROSConfigError(
                    f"_pack_with_composite_split: CARTESIAN_DELTA row width {len(delta)} "
                    f"does not match arm slot width {width} (slot=[{lo},{hi}]).",
                )
            # OSC delta is per-step — zero arm slots first so the value
            # from the previous tick doesn't accumulate.
            for i in range(width):
                out[lo + i] = 0.0
            for i, v in enumerate(delta):
                out[lo + i] = float(v)
        elif action.control_mode is ControlMode.GRIPPER_POSITION:
            if not action.gripper:
                raise ROSConfigError("_pack_with_composite_split: empty Action.gripper")
            slot = self._part_slot("right_gripper")
            if slot is None:
                raise ROSConfigError(
                    "_pack_with_composite_split: composite has no 'right_gripper' part — "
                    f"split keys: {list(getattr(cc, '_action_split_indexes', {}).keys())}",
                )
            lo, _hi = slot
            out[lo] = float(action.gripper[0])
        elif action.control_mode is ControlMode.JOINT_VELOCITY:
            # Route a JOINT_VELOCITY chunk to the
            # HybridMobileBase composite's 'base' part. The chunk
            # arrives padded to the robot's full n_dof (so the C++
            # safety kernel's n_dof check passes); we extract the
            # base-joint values via description.base_joints and
            # write them to the env_action vector at _part_slot('base').
            if not action.joint_velocities:
                raise ROSConfigError("_pack_with_composite_split: empty Action.joint_velocities")
            full = list(action.joint_velocities[0])
            base_joint_names = list(self.description.base_joints or [])
            if not base_joint_names:
                raise ROSConfigError(
                    "_pack_with_composite_split: JOINT_VELOCITY requires "
                    "description.base_joints to be declared (got empty list)"
                )
            name_to_idx = {j.name: i for i, j in enumerate(self.description.joints)}
            try:
                base_vels = [full[name_to_idx[n]] for n in base_joint_names]
            except KeyError as exc:
                raise ROSConfigError(
                    "_pack_with_composite_split: base_joints reference unknown "
                    f"joint {exc.args[0]!r} (description has: "
                    f"{sorted(name_to_idx.keys())})"
                ) from None
            slot = self._part_slot("base")
            if slot is None:
                raise ROSConfigError(
                    "_pack_with_composite_split: composite has no 'base' part — "
                    f"split keys: {list(getattr(cc, '_action_split_indexes', {}).keys())}",
                )
            lo, hi = slot
            width = hi - lo
            if len(base_vels) != width:
                raise ROSConfigError(
                    f"_pack_with_composite_split: base_joints len "
                    f"{len(base_vels)} does not match composite 'base' slot "
                    f"width {width} (slot=[{lo},{hi})).",
                )
            for i, v in enumerate(base_vels):
                out[lo + i] = float(v)
        elif action.control_mode is ControlMode.COMPOSITE_MODE:
            # Sim-only multiplexer flag. Write the policy's
            # raw value to the env_action vector's LAST slot, which
            # HybridMobileBase.set_goal reads as ``all_action[-1]`` to
            # select arm-active ("desired" goal_update_mode, value > 0)
            # vs base-active ("achieved" goal_update_mode, value <= 0).
            # Without this passthrough the HAL hardcoded -1.0 below,
            # which froze the arm OSC in "achieved" mode and let only
            # the base move.
            if not action.composite_mode:
                raise ROSConfigError("_pack_with_composite_split: empty Action.composite_mode")
            out[-1] = float(action.composite_mode[0])
            # Persist + return early — skip the trailing
            # ``out[-1] = -1.0`` override below.
            self._last_env_action = out.copy()
            return out
        elif action.control_mode is ControlMode.JOINT_POSITION:
            # Fall back to the legacy free-function packer for joint
            # mode — it already knows the arm-only vs base+arm widths.
            return self._action_packer(action, self.description, env_dim)
        else:
            raise ROSConfigError(
                f"_pack_with_composite_split: unsupported control_mode {action.control_mode!r} "
                "(BODY_TWIST has its own direct-qpos path; CARTESIAN_DELTA / "
                "GRIPPER_POSITION / JOINT_VELOCITY / JOINT_POSITION are the only "
                "slot-dispatch modes this helper handles).",
            )

        # HybridMobileBase reserves the LAST slot for the composite
        # multiplexer flag (``action[-1] > 0`` = arm OSC tracks the
        # commanded delta; ``<= 0`` = arm OSC tracks the achieved pose
        # i.e. arm is frozen). This is promoted to a first-class
        # ``COMPOSITE_MODE`` ControlMode handled in the branch above;
        # when the manifest doesn't declare a COMPOSITE_MODE slot, the
        # persisted value from the previous tick stays in place (so the
        # mode flag is not reset spuriously between heterogeneous
        # chunks within the same policy tick).
        _ = cc  # reason: HybridMobileBase import kept for future per-composite branches
        _ = HybridMobileBase
        # Persist for the next send_action so unspecified slots
        # (e.g. gripper between two arm ticks) carry their previous
        # commanded value instead of falling to zero.
        self._last_env_action = out.copy()
        return out

    def _apply_body_twist_to_qpos(self, row: list[float]) -> None:
        """Euler-integrate a validated 6-vec body twist directly into MuJoCo base qpos.

        Mirrors ``PandaMobileHAL._apply_body_twist`` through the one planar integrator
        (``_floating_base.integrate_planar_twist``): the body-frame ``(vx, vy)`` is
        rotated into world by the current ``base_yaw`` and ``velocity * dt`` added to
        each base joint's qpos slot; yaw wraps to ``[-π, π)``. ``send_action`` validated
        the row (``_check_twists``).

        Raises:
            ROSConfigError: The manifest declares no planar base (``base_joints``
                ``[x, y, yaw]``) or a base joint is missing from the MJCF.
        """
        handles = self._mujoco_handles()
        assert handles is not None  # the caller's route
        model, data = handles
        import mujoco  # noqa: PLC0415  # reason: optional dep, guarded by handles check

        addrs = self._base_qpos_addrs(model, mujoco, strict=True)
        vx_body, vy_body, _vz, _wx, _wy, wz = row
        # Latch the commanded twist for the /odom publisher (base_link frame).
        self._latch_twist(row)
        dt = self._body_twist_dt_s
        x, y, yaw, _ = integrate_planar_twist(
            (
                float(data.qpos[addrs[0]]),
                float(data.qpos[addrs[1]]),
                float(data.qpos[addrs[2]]),
                0.0,
            ),
            vx_body,
            vy_body,
            wz,
            dt,
        )
        data.qpos[addrs[0]] = x
        data.qpos[addrs[1]] = y
        data.qpos[addrs[2]] = yaw
        # Propagate the qpos change into derived state (body xforms +
        # sensor positions) so the live MuJoCo viewer + ``/scan``
        # ray-cast see the new base pose on the same tick. No physics
        # step — just the kinematic update.
        mujoco.mj_forward(model, data)
        # Re-render camera observations against the new model+data — env.step
        # is the only path that normally refreshes ``_last_obs`` and we skip
        # it here, so without this the dashboard shows the connect-time frame
        # for every BODY_TWIST command. Backends without ``refresh_obs``
        # (in-process digital twins, gymnasium-wrapped envs) keep the cached
        # frame (documented visual-lag tradeoff).
        #
        # ``refresh_obs`` MUST NOT step physics (see ``_RoboCasaSim.
        # refresh_obs``): a zero-action ``env.step`` there let robosuite's
        # mobile-base controller regulate the base back to its held setpoint,
        # erasing the qpos write above — measured on the baguette scene, of
        # 1.0000 m written over 40 commands only 0.0396 m (4.0%) survived,
        # starving Nav2's ``SimpleProgressChecker`` ("Failed to make progress").
        refresh = getattr(self._env, "refresh_obs", None)
        if refresh is not None:
            refreshed = refresh()
            if refreshed is not None:
                self._merge_refreshed_obs(refreshed)
        # This path deliberately runs no ``env.step``, but it is still a
        # simulation timestep: advance the clock by the same interval used for
        # the kinematic integration above so ``SimRollout.sim_time_ns`` (and
        # therefore deploy-sim ``/clock``) cannot freeze while Nav2 streams
        # BODY_TWIST commands. Exactly one advance per command — pairing this
        # with a stepping ``refresh_obs`` double-counted it, making the clock
        # run at 2x the rate of the motion it was timing.
        data.time = float(data.time) + dt

    def _apply_body_twist_via_env_step(self, row: list[float]) -> None:
        """Hold a validated BODY_TWIST as the base setpoint ``env.step`` integrates.

        For a backend without a qpos handle (the Isaac kinematic base)
        the planar base lives inside the env: the scene integrates ``(vx, vy, wz)``
        and teleports its root each ``env.step``. An env with its own
        ``pack_action`` packs the twist itself (holding every joint target);
        otherwise we pack the body-frame twist into the **final three** slots and
        leave the rest at zero. The scene integrates by its own command-interval
        dt, so we pass the velocity raw. ``send_action`` validated the row.
        """
        if self._env_action_dim is None or self._env_action_dim < 3:  # noqa: PLR2004  # reason: vx/vy/wz triple
            raise ROSRuntimeError(
                f"{type(self).__name__}: BODY_TWIST on a non-MuJoCo backend needs an env "
                f"action vector of at least 3 slots; got {self._env_action_dim}. "
                "Does this robot declare a planar base (base_joints)?"
            )
        vx_body, vy_body, _vz, _wx, _wy, wz = row
        # A velocity setpoint, as a real base controller takes /cmd_vel: held and
        # integrated by every idle step (idle_step) until _BASE_TWIST_HOLD_NS after
        # the last command. Stepping the env once per message instead tied the
        # distance driven, and the sim clock, to the message rate: at 20 Hz the
        # base covered 2/3 of the commanded distance per wall second, /odom's twist
        # flickered to zero on the idle steps between, and Nav2 (on sim time,
        # whose clock only its own commands advanced) drove past its goals.
        self._held_twist = (float(vx_body), float(vy_body), float(wz))
        self._held_twist_until_ns = self._twist_hold_clock_ns() + _BASE_TWIST_HOLD_NS
        self._latch_twist(row)

    def _drive_held_twist(
        self, env_action: NDArray[np.float32], *, twist_commanded: bool = False
    ) -> NDArray[np.float32]:
        """``env_action`` with a live held setpoint in its twist slots, else the base still.

        Sets ``_last_body_twist`` to what the step drives. A copy when it writes, so the
        setpoint never latches into the packer's ``prev``. A step that commands its own
        twist (``twist_commanded``) is returned untouched.
        """
        if twist_commanded:
            # The step's own twist supersedes the setpoint: an older /cmd_vel must not
            # resume over it (a skill's stop) on the next idle or arm step.
            self._held_twist = None
            return env_action
        held = self._live_held_twist()
        if held is None:
            self._last_body_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return env_action
        self._last_body_twist = (held[0], held[1], 0.0, 0.0, 0.0, held[2])
        return self._with_twist(env_action, held)

    def _with_twist(
        self, env_action: NDArray[np.float32], twist: tuple[float, float, float]
    ) -> NDArray[np.float32]:
        """A copy of ``env_action`` with ``(vx, vy, wz)`` in the base twist slots.

        An env that packs by name places it in its own layout; otherwise the default
        packer's convention holds (the final three slots). Never writes in place, so
        neither the packer's ``prev`` nor a backend's own hold vector latches it.
        """
        if self._env_packs:
            vx, vy, wz = twist
            return self._env.pack_action(  # type: ignore[attr-defined,no-any-return]  # reason: duck-typed env hook, checked in __init__
                Action(
                    control_mode=ControlMode.BODY_TWIST,
                    body_twist=[(vx, vy, 0.0, 0.0, 0.0, wz)],
                ),
                env_action.copy(),
            )
        out = env_action.copy()
        out[-3:] = twist
        return out

    def _live_held_twist(self) -> tuple[float, float, float] | None:
        """The held base setpoint, or ``None`` once it has timed out (the base stops)."""
        if (
            self._held_twist is not None
            and self._twist_hold_clock_ns() >= self._held_twist_until_ns
        ):
            self._held_twist = None
        return self._held_twist

    def _twist_hold_clock_ns(self) -> int:
        """The clock the held setpoint expires on: sim time when the backend has one.

        Nav2 publishes /cmd_vel on sim time. Below real time a wall-clock hold lapsed
        between its commands, so the base stopped and odom's twist dropped to zero
        between them; on sim time the hold spans the same sim interval as on hardware.
        A clock-less backend falls back to the monotonic wall clock.
        """
        sim_ns = self.sim_time_ns()
        return time.monotonic_ns() if sim_ns is None else sim_ns

    def _merge_refreshed_obs(self, refreshed: Any) -> None:  # noqa: ANN401  # reason: Observation is dict[str, Any]
        """Merge a post-BODY_TWIST ``refresh_obs`` result into ``_last_obs``.

        MERGE (not replace): ``refresh_obs`` may return an Observation
        with ``images=None`` on intermittent ticks (robocasa's observable
        cycle has rate gates); replacing would drop the ``images`` key
        entirely → ``read_images()`` returns ``{}`` → the lifecycle node
        stops publishing ``/openral/cameras/`` and the dashboard
        PERCEPTION cards go blank for the BODY_TWIST burst. So we keep
        the last non-empty value per key.

        Crucially this includes ``raw_proprio`` — the ``/odom`` /
        ``odom → base_link`` source via ``base_pose_6dof``. Dropping
        it (the original bug) left ``robot0_base_pos`` / ``robot0_base_quat``
        frozen at the connect-time pose: the base physically moved (we
        just wrote its qpos) but ``/odom`` reported it standing still, so
        Nav2's control loop never saw progress toward the goal and kept
        issuing corrections — the robot drove in circles on a "move
        backwards" command. ``refresh_obs`` recomputes ``raw_proprio``
        from the live sim after our qpos write, so merging it here makes
        odom track the base again.
        """
        if self._last_obs is None:
            self._last_obs = {}
        # ``Observation`` is ``dict[str, Any]`` (openral_sim.rollout) —
        # use mapping access, not attribute access.
        refreshed_state = refreshed.get("state")
        if refreshed_state is not None:
            self._last_obs["state"] = refreshed_state
        refreshed_images = refreshed.get("images")
        if refreshed_images:
            self._last_obs["images"] = dict(refreshed_images)
        refreshed_proprio = refreshed.get("raw_proprio")
        if refreshed_proprio:
            self._last_obs["raw_proprio"] = dict(refreshed_proprio)

    def estop(self) -> None:
        """Latch e-stop: commands are dropped (logged once) and idle does not step.

        The latch survives a reconnect; only ``reset_estop`` clears it. The base stops
        and a half-staged slot group is dropped (the watermark is kept).
        """
        self._estop_latched = True
        self._latched_drop_logged = False
        self._held_twist = None
        self._latch_twist(None)  # /odom: the base is stopped
        self._stager.discard()

    # ── Helpers exposed to the lifecycle node ──────────────────────────

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Forward the underlying env's MJCF handles, or None.

        Used by the panda_mobile ROS lifecycle node to bind its
        ``mujoco_handle_provider`` so ``/scan`` ray-casts against the
        live env instead of the no-hit fallback. Generic across
        robots — any ``SimRollout`` that exposes
        ``mujoco_handles()`` works.
        """
        return self._mujoco_handles()

    def camera_intrinsics(self) -> dict[str, IntrinsicsPinhole]:
        """The env's rendered RGB camera intrinsics (name -> ``IntrinsicsPinhole``), or ``{}``.

        For a backend that renders each camera from a known K but exposes no MuJoCo
        handles (Isaac): ``SimSensorBridge`` publishes its ``CameraInfo`` from it.
        """
        getter = getattr(self._env, "camera_intrinsics", None)
        return dict(getter()) if getter is not None else {}

    @property
    def last_applied_action(self) -> Action | None:
        """The command the simulator was last stepped with (``None`` = none, or unreadable).

        For a slot group, the composed full-dof ``JOINT_POSITION`` action, set in step with
        ``last_committed_tick`` — or, when the group is not one joint-position command,
        its gripper targets as a compact action (``gripper_targets_action``; ``None``
        when it set none); for an ungrouped action, the action itself. The lifecycle node
        folds it into the grasp trigger.
        """
        return self._stager.last_applied_action

    @property
    def last_committed_tick(self) -> int:
        """Most recent ticked action or slot group committed to the simulator (0 = none)."""
        return self._stager.last_committed_tick

    @property
    def last_committed_session(self) -> int:
        """``runner_session_id`` of ``last_committed_tick`` (0 = none / session-less)."""
        return self._stager.last_committed_session

    def _rollout_sim_time_ns(self) -> int | None:
        """Read the wrapped rollout's per-episode sim time, or ``None``.

        ``sim_time_ns`` is an OPTIONAL duck-typed extension of the
        ``SimRollout`` protocol —
        clock-less adapters (PushT, the Isaac Sim sidecar) do not implement it.
        ``getattr`` narrows the missing-attribute case to ``None`` without
        catching exceptions; a backend that DOES implement it is trusted to
        honour the contract (``int | None``, no raise).
        """
        getter = getattr(self._env, "sim_time_ns", None)
        if getter is None:
            return None
        value = getter()
        return None if value is None else int(value)

    def _accumulate_sim_time_before_reset(self) -> None:
        """Fold the current episode's elapsed sim-time into the cross-reset offset.

        Called immediately before lifecycle-driven ``connect`` resets.
        Backends such as robocasa rewind
        ``MjData.time`` to 0 on reset, so without this the published value
        would jump backwards on each new episode. Reads the live per-episode
        sim time and, when the backend has a clock, adds it to
        ``_sim_time_offset_ns``. A clock-less backend (``None``) leaves the
        offset untouched — ``sim_time_ns`` then also returns ``None``.
        """
        elapsed = self._rollout_sim_time_ns()
        if elapsed is not None:
            self._sim_time_offset_ns += elapsed

    def sim_time_ns(self) -> int | None:
        """Cross-reset-monotonic elapsed simulation time in ns, or ``None``.

        The value a sim ``/clock`` publisher reads so the
        deploy-sim ROS graph runs on simulation time. Returns the wrapped
        ``SimRollout``'s per-episode sim time plus the
        accumulated offset from prior lifecycle reconnects (``connect``
        folds elapsed time into the offset before the backend rewinds its
        clock). The result is therefore
        **monotonic non-decreasing across ``env.reset``**, unlike the raw
        backend clock (robocasa rewinds ``MjData.time`` to 0 on reset).

        Returns ``None`` when the wrapped rollout has no sim clock — a
        clock-less backend (PushT) or an out-of-process sidecar (Isaac Sim).
        In that case the consumer falls back to wall time.

        Returns:
            Cross-reset-monotonic elapsed sim time in nanoseconds, or ``None``
            when the wrapped rollout exposes no sim clock.
        """
        current = self._rollout_sim_time_ns()
        if current is None:
            return None
        return self._sim_time_offset_ns + current

    def clock_authority(self) -> ClockAuthority:
        """Return the timestamp authority this HAL contributes to the graph.

        A sim-attached HAL with a live backend clock is the simulation-time
        authority and may be projected onto ROS ``/clock`` by the lifecycle
        node. A clock-less wrapped rollout falls back to host wall time; callers
        must then keep the graph on the host-wall clock authority. The
        ``timestep_s`` is the env's declared sim seconds per step when it
        reports one (checked in ``connect``), else its physics timestep
        (``physics_dt_s``, a wall-paced twin), else the body-twist interval. The clock
        id is the env's ``clock_id`` (``"mujoco"`` for the twin), else its class name.
        """
        if self.sim_time_ns() is None:
            return ClockAuthority.host_wall()
        return ClockAuthority.simulation(
            getattr(self._env, "clock_id", None) or type(self._env).__name__,
            timestep_s=self._sim_dt_per_tick_s
            or getattr(self._env, "physics_dt_s", None)
            or self._body_twist_dt_s,
            publishes_ros_clock=True,
        )

    def _mujoco_handles(self) -> tuple[Any, Any] | None:
        """Return the env's MJCF (model, data) tuple, or None.

        The ``SimRollout`` protocol declares ``mujoco_handles()``
        as optional — backends that don't run on MuJoCo (PushT,
        SimplerEnv on Bridge) don't implement it. ``getattr`` with a
        default narrows the missing-attribute case to ``None`` without
        catching exceptions. If the env *does* implement
        ``mujoco_handles`` we trust its contract: a clean
        ``(model, data) | None`` return, no exceptions in the hot path.
        """
        getter = getattr(self._env, "mujoco_handles", None)
        if getter is None:
            return None
        return getter()  # type: ignore[no-any-return]  # reason: mujoco_handles contract is duck-typed; runtime-verified by SimAttachedHAL callers

    @property
    def env(self) -> SimEnv:
        """Direct access to the wrapped sim env (for tests / advanced wiring)."""
        return self._env

    @property
    def estop_latched(self) -> bool:
        """``True`` while the estop latch is set."""
        return self._estop_latched

    @property
    def last_action_ns(self) -> int:
        """Monotonic ns timestamp of the last real action seen by ``send_action``.

        ``0`` until the first ``send_action``. The sim-only idle stepper reads
        this (via ``should_idle_step``) to
        yield the env to an active skill — it skips an idle tick whenever a
        real action arrived within the idle-hold window.
        """
        return self._last_action_ns

    def read_images(self) -> dict[str, Any]:
        """Return the latest rendered camera frames keyed by camera name.

        The wrapped ``SimRollout`` returns rendered images on each
        ``reset`` / ``step`` under the ``"images"`` slot of its
        ``Observation`` dict (per ``openral_sim.rollout`` schema).
        The HAL caches that slot so the panda_mobile lifecycle node's
        camera publisher can republish the frames as
        ``sensor_msgs/Image`` on ``/openral/cameras/<name>/image`` at
        the configured rate — that's the topic WorldState subscribes to
        and the path the rldx / pi05 / smolvla adapters consume via
        ``observation.images.<name>``. Frame keys match the canonical
        ``camera1`` / ``camera2`` / ``camera3`` aliases the robocasa
        adapter exposes in ``openral_sim.backends.robocasa._RoboCasaSim._wrap_obs``
        plus the raw robosuite keys
        (``robot0_agentview_left_image`` etc.); the caller chooses
        which subset to forward.

        Returns an empty dict when no observation has been cached yet
        (e.g. before ``connect``) or when the observation has no
        ``"images"`` slot (non-image backends). Never raises.

        A backend that renders on demand (``read_images()`` hook, the bare MuJoCo
        twin) is asked instead: its frames show the current state, so
        ``read_image_times`` stays empty for it.
        """
        render = getattr(self._env, "read_images", None)
        if callable(render):
            return dict(render())
        if self._last_obs is None:
            return {}
        images = self._last_obs.get("images")
        if not isinstance(images, dict):
            return {}
        return dict(images)

    def read_image_times(self) -> dict[str, int]:
        """Sim time (ns) each cached camera frame shows, keyed like ``read_images``.

        A backend whose renderer delivers frames behind the physics (the Isaac
        sidecar: ~4 steps, issue #361) reports per camera the sim time its frame
        shows under the ``"image_time_ns"`` obs slot. Returned in the
        ``sim_time_ns`` domain (the cross-reset offset added), so
        ``sim_time_ns() - read_image_times()[k]`` is that frame's age; depth
        cameras are keyed by sensor name, as in ``read_depth_clouds``.
        ``SimSensorBridge`` stamps each frame with it. Empty when the backend
        reports none (its frames show the current step). Never raises.
        """
        if self._last_obs is None:
            return {}
        times = self._last_obs.get("image_time_ns")
        if not isinstance(times, dict):
            return {}
        return {str(k): self._sim_time_offset_ns + int(v) for k, v in times.items()}

    def read_depth_clouds(self) -> dict[str, NDArray[np.float32]]:
        """Return per-depth-sensor point clouds ``{name: (N, 3) base_frame}``.

        Derived from the newest registered RGB-D frame of each depth camera
        (``read_depth_frames``: metric depth, its K and the optical pose in the base
        frame, as a non-MuJoCo backend such as the Isaac scene ships them) by
        ``depth_cloud.cloud_from_depth_frame``, kept within the sensor's
        ``range_max_m`` in the base plane. Each frame is deprojected once and its
        cloud returned as the same array until a newer frame arrives:
        ``SimSensorBridge`` publishes each cloud once, by identity.
        ``read_depth_cloud_colours`` has the matching per-point colours. Empty when
        the backend renders no depth (MuJoCo backends ray-cast instead). Never raises.
        """
        # Deferred: only a depth-rendering backend pays the import.
        from openral_hal.depth_cloud import cloud_from_depth_frame  # noqa: PLC0415

        out: dict[str, NDArray[np.float32]] = {}
        for name, frame in self._last_depth_frames.items():
            cached = self._depth_cloud_cache.get(name)
            if cached is None or cached[0] is not frame:
                spec = next((s for s in self.description.sensors if s.name == name), None)
                pts, colours = cloud_from_depth_frame(
                    frame["depth"],
                    frame["rgb"],
                    frame["k"],
                    frame["optical_in_base"],
                    max_planar_range_m=getattr(spec, "range_max_m", None) or None,
                )
                cached = (frame, pts, colours)
                self._depth_cloud_cache[name] = cached
            out[name] = cached[1]
        return out

    def read_depth_cloud_colours(self) -> dict[str, NDArray[np.uint8]]:
        """Per-point RGB of the clouds ``read_depth_clouds`` last returned, by sensor.

        Each colour is the pixel its point was back-projected from. A sensor whose
        frame carries no matching colour raster is absent. Never raises.
        """
        return {name: c for name, (_f, _p, c) in self._depth_cloud_cache.items() if c is not None}

    def read_depth_frames(self) -> dict[str, dict[str, Any]]:
        """Return the newest registered colour + depth frame per depth sensor.

        A non-MuJoCo backend (the Isaac manifest scene) renders each depth camera's
        metric depth AND its RGB in one render and surfaces them under the
        ``"depth_frames"`` obs slot as ``{name: {"depth": (H, W) float32 m,
        "rgb": (H, W, 3) uint8, "k": [fx, fy, cx, cy], "optical_in_base": 4x4}}``.
        The backend ships them on some steps only, so the newest frame is kept
        until the next one replaces it; ``SimSensorBridge`` publishes each frame
        once (by identity). Empty when the backend renders none (MuJoCo backends
        ray-cast their depth instead). Never raises.
        """
        return dict(self._last_depth_frames)

    def read_scan(self) -> NDArray[np.float32] | None:
        """Return the 2-D LaserScan range fan, or ``None``.

        A non-MuJoCo backend that ray-casts a 2-D lidar (the Isaac scene)
        surfaces the per-beam ranges (the lidar's own frame, ``angle_min=-π`` →
        ``angle_max=+π``, the bridge's convention) under the ``"scan"`` obs slot.
        The lifecycle node captures it with each observation (``ProprioFrame.scan``)
        and ``SimSensorBridge.publish_scan_from_frame`` publishes it on ``/scan``. ``None``
        when the backend renders no lidar (MuJoCo backends ray-cast instead).
        """
        if self._last_obs is None:
            return None
        scan = self._last_obs.get("scan")
        if scan is None:
            return None
        return np.asarray(scan, dtype=np.float32).reshape(-1)

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """Current base ``(x, y, yaw)`` read from MJCF qpos.

        In order: a backend that reads its own base pose (``read_base_pose()``: the
        twins' pinned or free base, the OpenAMRobot's planar joints) is asked; a robot
        whose manifest declares no planar base (``describes_mobile_base``) reads
        ``(0.0, 0.0, 0.0)``; a non-MuJoCo backend that drives a planar base (the Isaac
        kinematic base) reports ``obs["base_pose"] = (x, y, yaw)``, else
        ``(0.0, 0.0, 0.0)``; a MuJoCo scene reads the qpos of the three joints named in
        ``description.base_joints`` (``[base_x, base_y, base_yaw]``, by
        ``sim_joint_name``; one absent from the MJCF reads ``0.0``), as
        ``PandaMobileHAL.base_pose`` does, so the ``/odom`` publisher works whichever
        HAL is wired in.

        Raises:
            ROSConfigError: A MuJoCo scene whose manifest declares one or two
                ``base_joints`` (no ``[x, y, yaw]`` triple).
        """
        read_base_pose = getattr(self._env, "read_base_pose", None)
        if callable(read_base_pose):
            x, y, yaw = read_base_pose()
            return (float(x), float(y), float(yaw))
        if not describes_mobile_base(self.description):
            return (0.0, 0.0, 0.0)
        handles = self._mujoco_handles()
        if handles is None:
            # Non-MuJoCo planar base: read the (x, y, yaw) the rollout surfaces.
            if self._last_obs is not None:
                bp = self._last_obs.get("base_pose")
                if bp is not None:
                    arr = np.asarray(bp, dtype=np.float64).reshape(-1)
                    if arr.shape[0] >= 3:  # noqa: PLR2004  # reason: x/y/yaw triple
                        return (float(arr[0]), float(arr[1]), float(arr[2]))
            return (0.0, 0.0, 0.0)
        model, data = handles
        import mujoco  # noqa: PLC0415  # reason: optional dep, guarded by handles check

        qpos = np.asarray(data.qpos, dtype=np.float64)
        return cast(
            "tuple[float, float, float]",
            tuple(
                0.0 if addr is None else float(qpos[addr])
                for addr in self._base_qpos_addrs(model, mujoco, strict=False)
            ),
        )

    def _base_qpos_addrs(
        self,
        model: Any,  # noqa: ANN401  # reason: mujoco.MjModel
        mujoco: Any,  # noqa: ANN401  # reason: the lazily imported mujoco module
        *,
        strict: bool,
    ) -> list[int | None]:
        """The qpos address of each planar base joint ``base_joints[:3]`` (``sim_joint_name``).

        Args:
            model: The live ``MjModel``.
            mujoco: The ``mujoco`` module.
            strict: Raise for a joint missing from the MJCF (a twist to apply), instead
                of ``None`` (a pose to read).

        Raises:
            ROSConfigError: The manifest declares fewer than three ``base_joints`` (no
                planar base to read or move), or (``strict``) one is not in the MJCF.
        """
        base = list(self.description.base_joints or [])
        if len(base) < 3:  # noqa: PLR2004  # reason: x/y/yaw triple
            raise ROSConfigError(
                f"{type(self).__name__}: robot '{self.description.name}' declares no planar "
                f"base (base_joints [x, y, yaw], got {base}); a BODY_TWIST cannot move it."
            )
        joints_by_name = {j.name: j for j in self.description.joints}
        addrs: list[int | None] = []
        for name in base[:3]:
            sim_name = joints_by_name[name].sim_joint_name or name
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if jid < 0:
                if strict:
                    raise ROSConfigError(
                        f"{type(self).__name__}: base joint {sim_name!r} not found in MJCF; "
                        "cannot apply BODY_TWIST. Check robot.yaml base_joints + "
                        "sim_joint_name."
                    )
                addrs.append(None)
                continue
            addrs.append(int(model.jnt_qposadr[jid]))
        return addrs

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """Last commanded base body twist ``(vx, vy, vz, wx, wy, wz)``.

        In the ``base_link`` frame. The base advances by exact Euler
        integration of this command, so it is the base's velocity — the
        panda_mobile lifecycle node publishes it as the ``/odom`` twist.
        Zeroed once a non-BODY_TWIST action is sent or an idle HOLD step runs.
        A backend that holds its own velocity setpoint (``read_base_twist()``: the
        G1 GR00T WBC command, which arm actions leave standing) reports it instead.
        """
        read_base_twist = getattr(self._env, "read_base_twist", None)
        if callable(read_base_twist):
            vx, vy, vz, wx, wy, wz = read_base_twist()
            return (float(vx), float(vy), float(vz), float(wx), float(wy), float(wz))
        return self._last_body_twist

    def base_pose_6dof(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        """Full 6-DoF base pose ``(xyz, quat_xyzw)`` from the cached robocasa obs.

        The planar ``base_pose`` (``x, y, yaw``) sets ``z=0``/
        ``roll=pitch=0``, dropping the ~0.70 m platform height RoboCasa
        proprio (``robot0_base_pos[2]``) reports — the rldx/pi05 state
        assemblers read the base's full 6-DoF pose via
        ``tf("odom", "base_link")``, and a planar-only pose there previously
        produced ``world_to_base.position.z == 0.0`` instead of ``0.70``,
        putting the arm 0.49 m off in the base frame ("moves but grabs at the
        wrong position").

        Source of truth: ``_last_obs["raw_proprio"]["robot0_base_pos"]`` +
        ``["robot0_base_quat"]`` — the same values ``openral sim run`` feeds
        the policy in the 16-D ``human300`` state vector, keeping the
        deploy_sim ``odom → base_link`` TF byte-identical to sim_run's state
        assembly. NOT ``data.xpos[robot0_base]`` (raw MJCF body position):
        robosuite's observable applies a robot-specific offset
        (``robot.robot_model.base_xpos_offset``) so the reported base
        position matches a stable mount reference, not the MJCF body anchor.

        Returns ``None`` when the wrapped obs has no ``raw_proprio`` slot
        (non-RoboCasa backend) or the keys are missing — the caller falls
        back to the planar ``base_pose``.
        """
        if self._last_obs is None:
            return None
        proprio = self._last_obs.get("raw_proprio")
        if not isinstance(proprio, dict):
            return None
        pos = proprio.get("robot0_base_pos")
        quat = proprio.get("robot0_base_quat")
        if pos is None or quat is None:
            return None
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        if pos_arr.shape[0] < 3 or quat_arr.shape[0] < 4:  # noqa: PLR2004
            return None
        # RoboCasa / robosuite emit quaternions as ``(x, y, z, w)`` —
        # same convention TF + the human300_16d assembler expect, no
        # permutation needed.
        return (
            (float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2])),
            (
                float(quat_arr[0]),
                float(quat_arr[1]),
                float(quat_arr[2]),
                float(quat_arr[3]),
            ),
        )

    def reset_estop(self) -> None:
        """Clear the estop latch. Caller asserts the cause has been resolved.

        The robot comes back holding its present pose (``stop_stale_motion``), the
        ``ResettableLifecycleEStopHAL`` contract.
        """
        self._estop_latched = False
        self._latched_drop_logged = False
        self.stop_stale_motion()

    def stop_stale_motion(self) -> None:
        """Drop every setpoint still in flight: the held base twist and the arm's targets.

        Called by the HAL lifecycle node at its ``/openral/estop`` latch and at
        ``/openral/estop_cleared`` (openral#700 item 1). The node's latch alone froze
        stepping but kept a held ``/cmd_vel`` setpoint and the twin's ctrl targets, so
        the base drove on and the arm finished its last move after the clear. A backend
        that exposes ``hold_present_pose`` (the MuJoCo twins) holds its joints where they
        are; the base twist is dropped and ``/odom`` reads still.
        """
        self._held_twist = None
        self._latch_twist(None)
        # A target the stop cut short must not ride along as the next pack's ``prev``:
        # an absolute-target env (RoboTwin) would resume it on the next gripper or twist
        # command. Unset slots then take the measured pose, as after a reset.
        self._last_env_action = None
        hold = getattr(self._env, "hold_present_pose", None)
        if callable(hold) and self._connected:
            hold()

    def read_policy_state(self) -> list[float] | None:
        """Return the simulator-native policy state, when the backend exposes one.

        Only an explicit ``obs["policy_state"]`` counts — the generic
        ``obs["state"]`` every sim backend populates is NOT a policy state, and
        falling back to it would silently publish `/openral/policy_state` for
        robots whose manifests never opted in (WorldState.policy_state is
        documented as never inferred).
        """
        if self._last_obs is None:
            return None
        raw = self._last_obs.get("policy_state")
        if raw is None:
            return None
        return [float(value) for value in np.asarray(raw, dtype=np.float32).reshape(-1)]

    def reset_episode(self) -> int:
        """Start a fresh episode on the live env without rebuilding it (sim only).

        Re-runs exactly what ``connect`` does to the scene (``env.reset`` plus the
        HAL's per-episode bookkeeping) with the next seed of a deterministic
        sequence: the base reset seed plus the episode index (1, 2, ...). A scene
        with per-reset randomisation (Isaac ``pose_noise``) therefore draws a new
        layout each episode, reproducible from the returned seed. Keeps the env,
        its sidecar and every ROS node up: a multi-episode harness pays the boot
        and policy load once. Call between goals, never during one.

        Returns:
            The seed the env was reset with.

        Raises:
            ROSRuntimeError: Not connected, or ``env.reset`` raised an untyped error.
            ROSError: ``env.reset`` raised a typed refusal (e.g. ``ROSConfigError`` for a
                twin's MJCF that does not load); it keeps its type.

        """
        self._require_connected("reset_episode")
        self._episode_index += 1
        seed = (self._reset_seed or 0) + self._episode_index
        self._reset_env(seed, "reset_episode")
        self._stager.reset()  # a new episode restarts the tick numbering
        return seed

    def update_attached_objects(self, objects: list[AttachedCollisionObject]) -> None:
        """Atomically replace attached objects and resolve their MuJoCo bodies.

        Sim evidence uses ``evidence_ref="mujoco_body:<body-name>"``. Every
        descendant body is masked too, so multipart payloads cannot leak into
        depth/OctoMap through child geoms. A backend without MuJoCo handles (Isaac)
        records the set and masks nothing, as on real hardware: only a tracker-bound
        payload is cleared from the map (by identity), a geometry-only one stays.
        On a MuJoCo backend an object whose evidence names no MuJoCo body (the
        vision leg's) is recorded the same way: kept, masking nothing.

        Args:
            objects: Complete current attachment set.

        Raises:
            ROSConfigError: On duplicate ids or an unknown ``mujoco_body:`` name.
                The previous snapshot is preserved on failure.
        """
        by_id = {obj.object_id: obj for obj in objects}
        if len(by_id) != len(objects):
            raise ROSConfigError("Attached collision object ids must be unique.")
        handles = self._mujoco_handles()
        if handles is None:
            # No MuJoCo model to mask payload bodies in (Isaac and other sidecar
            # backends): the attachment is recorded and nothing is masked, so the
            # payload stays in the simulated depth exactly as it stays in a real
            # camera's — the real-hardware path. The octomap bridge clears its cells
            # only for a tracker-bound attachment (``OctomapVoxelBridge::clear_by_identity``,
            # ``object_tracking: tracking | on``); a geometry-only one stays occupied and the kernel
            # may stop on it: never looser than masking.
            self._attached_objects = by_id
            self._attached_body_ids = frozenset()
            return

        import mujoco  # noqa: PLC0415  # reason: optional sim dependency guarded by handles

        model, _data = handles
        roots: set[int] = set()
        for obj in objects:
            prefix = "mujoco_body:"
            if obj.evidence_ref is None or not obj.evidence_ref.startswith(prefix):
                # Evidence that names no MuJoCo body (the vision attachment leg's
                # ``segmenter_mask:…``) is recorded and masks nothing, as on Isaac
                # and real hardware above. Refusing it left the kernel-accepted
                # revision pending, which held every following action ack.
                continue
            body_name = obj.evidence_ref.removeprefix(prefix)
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id < 0:
                raise ROSConfigError(
                    f"Attached object {obj.object_id!r} references unknown "
                    f"MuJoCo body {body_name!r}."
                )
            roots.add(body_id)

        body_ids = set(roots)
        for body_id in range(1, int(model.nbody)):
            if int(model.body_parentid[body_id]) in body_ids:
                body_ids.add(body_id)
        self._attached_objects = by_id
        self._attached_body_ids = frozenset(body_ids)

    def read_attached_objects(self) -> list[AttachedCollisionObject]:
        """Return the current attachment snapshot in stable object-id order."""
        return [self._attached_objects[key] for key in sorted(self._attached_objects)]

    def read_attached_body_ids(self) -> frozenset[int]:
        """Return exact MuJoCo payload body ids excluded from world perception."""
        return self._attached_body_ids

    def set_active_task(self, prompt: str) -> None:
        """Grade the dispatched goal's ``prompt``; ``""`` ends the goal.

        ``deploy sim``'s HAL node calls this from ``/openral/reward/active_task``
        (the rSkill runner publishes the goal's ``ExecuteRskill`` prompt on accept
        and ``""`` when it ends). Forwards to the backend's ``set_task`` (the Isaac
        scene's ``placement_success``), which activates that prompt's rubric; a
        prompt the scene does not grade logs ``sim.placement_grade_unavailable``
        once and grades nothing. Each grade change then logs ``sim.placement_grade``
        and the goal's end ``sim.placement_grade_final``. The rubric keeps grading
        after the goal ends (an object released at the end still has to settle),
        until the next goal's prompt replaces it. Logging only: never termination,
        reset or the action path. No-op for a backend without ``set_task``.

        Args:
            prompt: The executing instruction, verbatim; ``""`` when none is.
        """
        if prompt == self._active_task:
            return
        set_task = getattr(self._env, "set_task", None)
        if self._active_task:
            self._emit_placement_grade_final()
        self._active_task = prompt
        if not prompt or not callable(set_task) or self._set_task_failed:
            return
        self._placement_prompt = prompt
        self._placement_grade = self._placement_best = "fail"
        try:
            reply = set_task(prompt)
        except ROSError as exc:  # reason: an observability call must never break actuation
            self._set_task_failed = True
            self._placement_scores = None
            self._emit_task_success(
                _EVENT_PLACEMENT_GRADE_UNAVAILABLE,
                level="warning",
                prompt=prompt,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return
        scores = reply.get("grade_scores") or {}
        self._placement_scores = (
            {str(k): float(v) for k, v in scores.items()} if reply.get("graded") else None
        )
        if self._placement_scores is None and reply.get("prompts"):
            self._emit_task_success(
                _EVENT_PLACEMENT_GRADE_UNAVAILABLE,
                prompt=prompt,
                graded_prompts=list(reply["prompts"]),
            )
