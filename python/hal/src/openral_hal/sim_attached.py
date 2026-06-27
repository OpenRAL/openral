"""SimAttachedHAL — wrap any ``openral_sim.SimRollout`` as a HAL adapter.

ADR-0025 Stage 3 — the generic, robot-agnostic bridge between the
:class:`openral_sim.SimRollout` simulator interface and the
:class:`openral_hal.HAL` Protocol the ROS lifecycle nodes consume.

The motivating problem: a HAL ROS lifecycle node + a robocasa env
historically lived in separate processes (the launch tree spawns the
HAL; ``openral_sim.SimRunner`` spawns the env). With the two split,
``/scan`` couldn't ray-cast against live MJCF geometry because the
lifecycle node had no handle to ``MjModel``/``MjData``. The bringup
script we used for the navigate-look-pick recordings co-located them
in one process by hand. This module is the in-process composition
made first-class.

Architecture
------------

* :class:`SimAttachedHAL` implements the structural
  :class:`openral_hal.HAL` Protocol against any
  :class:`openral_sim.rollout.SimRollout` instance. Reading state
  walks ``description.joints`` and looks up live qpos / qvel from
  the env's MJCF handles; sending an action calls ``env.step(...)``.

* The lifecycle node uses :class:`SimAttachedHAL` in place of the
  per-robot digital-twin HAL whenever its
  ``sim_env_yaml`` ROS param is set, then binds its
  ``mujoco_handle_provider`` to ``self._hal.mujoco_handles()`` so the
  ray-cast ``/scan`` generator sees the live MJCF.

* Generic across embodiments: any robot whose
  :class:`~openral_core.RobotDescription` declares the joint mapping
  (``JointSpec.sim_joint_name`` or matching ``name``s the env emits)
  works. The translation between the HAL :class:`Action` and the
  env's flat action vector is delegated to a small per-robot
  ``ActionPacker`` (see :func:`pack_action_for_env`).

CLAUDE.md §1.5 / §3
-------------------
The hot path (``read_state`` / ``send_action``) must complete within
the robot's control cycle. ``env.step`` is the slowest call —
robocasa steps ~5-15 ms per call on the reference host, well inside
the 50 ms (20 Hz) Nav2 cmd_vel budget.

This is a HAL adapter, not a safety shim. The safety supervisor's
contract holds: every action passing through here has already been
clamped by the C++ kernel.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    BODY_TWIST_DIM,
    SIM_EXECUTABLE_CONTROL_MODES,
    ClockAuthority,
    RobotDescription,
)
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, JointState

# Module logger — the HAL lifecycle wrapper logs through self.get_logger();
# this falls through to stderr via the rclpy logging bridge when running
# inside a ROS node, otherwise just stdlib logging.
_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openral_sim.rollout import SimRollout

# ADR-0036 — the sim packers below (``pack_action_for_env`` and
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
    "normalized_joint_index",
    "pack_action_for_env",
]

_ROBOSUITE_GROUP_PREFIX = re.compile(r"^[a-z]+[0-9]+_")  # robot0_ / gripper0_ / mobilebase0_

# robosuite's post-terminal step guard (``environments/base.py``) raises
# ``ValueError("executing action in terminated episode")`` whenever ``step`` is
# called after the episode ended and ``ignore_done=False`` — the configuration
# raw robosuite-backed adapters can use.
# Matched by message substring (stable across robosuite releases) so
# ``_step_and_cache`` can recover by resetting; see ADR-0036.
_TERMINATED_EPISODE_MARKER = "terminated episode"


def is_terminated_episode_error(exc: BaseException) -> bool:
    """True iff ``exc`` is robosuite's "executing action in terminated episode" guard.

    Raw-robosuite backends (``ignore_done=False``) HARD-RAISE this on a step
    taken after the episode ended, instead of returning a terminal ``StepResult``.
    :meth:`SimAttachedHAL._step_and_cache` treats that as a recoverable terminal
    (reset + re-step) so deploy-sim's continuous twin keeps driving; any other
    ``step`` failure (bad action, NaN, …) is a real fault and must propagate.

    Args:
        exc: The exception raised by ``env.step``.

    Returns:
        ``True`` when the message identifies the robosuite terminal-episode guard.
    """
    return _TERMINATED_EPISODE_MARKER in str(exc).lower()


def normalized_joint_index(model_joint_names: list[str]) -> dict[str, int]:
    """Map MJCF joint names (exact + robosuite-prefix-stripped) to model index.

    ADR-0034 §3.6 — robosuite prefixes every joint with ``<class>N_``
    (``robot0_joint1``); native MJCFs do not. Exact names always win; a
    stripped name (``robot0_joint1`` -> ``joint1``) is added only when it
    neither shadows an exact name nor collides with another stripped name
    (a bimanual ``robot0_``/``robot1_`` model strips ambiguously -> keep
    explicit, require ``sim_joint_name``).

    Example:
        >>> normalized_joint_index(["robot0_joint1", "gripper0_finger_joint1"])
        {'robot0_joint1': 0, 'joint1': 0, 'gripper0_finger_joint1': 1, 'finger_joint1': 1}
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


# Default mapping for robosuite BASIC-controller compositions that
# expose an 11-D action vector: ``[base_x, base_y, base_yaw,
# arm_j1..arm_j7, gripper]``. Each entry maps a row of
# ``Action.joint_targets`` (URDF-ordered) into the env action slot.
# Override per-robot via the ``action_packer`` factory passed to
# :class:`SimAttachedHAL`.

# Tolerance for non-planar twist components (vz / wx / wy) — anything
# above this is rejected with ROSConfigError. Mirror of
# ``openral_hal.panda_mobile._PLANAR_TWIST_EPS``.
_PLANAR_TWIST_EPS = 1e-6


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

    Handles the two control modes panda_mobile (and any future
    mobile-manipulator with the same robosuite BASIC composite)
    actually use:

    * ``BODY_TWIST`` — the first row's first three slots (vx, vy, wz)
      become the env's first three action dims; the remaining slots
      (arm + gripper) are zeroed.
    * ``JOINT_POSITION`` — the row directly populates the env's
      joint slots. An arm-only row (``arm_dim`` floats) fills slots
      ``[base_dim:base_dim + arm_dim]``; a full ``base_dim + arm_dim``
      row fills slots ``[0:base_dim + arm_dim]``; otherwise rejected.
      ``base_dim`` / ``arm_dim`` derive from the description.

    Other modes raise ``ROSConfigError`` — the dispatching lifecycle
    node enforces the supported set via
    :attr:`RobotDescription.capabilities.supported_control_modes`.
    Callers that need richer translations (whole-body humanoid
    actions, dexterous-hand chunks) pass their own
    :class:`ActionPacker` to the HAL constructor.

    Args:
        action: The chunk to pack. Only the first row is consumed
            (chunk_size=1 is invariant for the wrapped-ROS dispatch
            path; per-row safety-supervisor commit is what makes that
            valid).
        description: The robot manifest. Used to size the arm portion
            of the env vector when arm joints are present.
        env_action_dim: The env's declared action_dim (typically 11
            for the robosuite BASIC composite + gripper).
        prev: The env-frame action vector from the previous ``send_action``
            within the same policy tick, or ``None``. A single policy step
            on a non-composite env (e.g. LIBERO OSC_POSE) arrives as TWO
            typed Actions — CARTESIAN_DELTA (arm) then GRIPPER_POSITION
            (finger) — and each one drives a separate ``env.step``. Seeding the arm pack
            from ``prev`` carries the last commanded gripper through the arm
            step (instead of zeroing it to a half-open neutral), so the
            policy's gripper command holds while the arm moves — mirroring
            the ``_pack_with_composite_split`` merge. Without it the arm
            steps with gripper=0 and the gripper steps with arm=0, so the
            arm only advances every other env step with a flickering gripper
            and never coordinates a grasp.

    Returns:
        ``(env_action_dim,)`` float32 — the env-frame action vector.

    Raises:
        ROSConfigError: when the chunk's `control_mode` isn't one of
            the supported modes or its row width doesn't match the
            implied slot layout.
    """
    # ADR-0028c — payload field depends on control_mode. The pre-0028c
    # code path lifted ``joint_targets[0]`` for every mode, which was
    # the lie that conflated cartesian / gripper bytes into a
    # joint-targets row. Now each mode reads from its own field.
    #
    # ADR-0036 — the modes this packer handles (CARTESIAN_DELTA,
    # GRIPPER_POSITION, BODY_TWIST, JOINT_POSITION) are a subset of
    # ``openral_core.SIM_EXECUTABLE_CONTROL_MODES``; the union of this
    # packer + ``_pack_with_composite_split`` + the BODY_TWIST direct-qpos
    # path equals that constant exactly (lockstep test enforces it).
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
        # ADR-0028c — body_twist now comes through the typed
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


class SimAttachedHAL:
    """HAL Protocol adapter that wraps an in-process :class:`SimRollout`.

    Generic over robot embodiment. The active simulator is the source
    of truth for state; actions flow through :func:`pack_action_for_env`
    (or a caller-supplied :class:`ActionPacker`) into ``env.step()``.

    Args:
        env: The live simulator. Must implement
            :class:`openral_sim.rollout.SimRollout` (``reset/step/
            mujoco_handles`` at minimum).
        description: Normative robot manifest. Used to populate
            :attr:`JointState.name` and to feed the
            ``HALLifecycleNodeBase`` OTel attributes.
        action_packer: Optional override for the default packer.
            Defaults to :func:`pack_action_for_env`.
        env_reset_seed: Seed forwarded to ``env.reset(seed=...)`` on
            :meth:`connect`. ``None`` means "use the env's own
            default" (typically ``0`` or non-deterministic).
        env_action_dim: The env's flat action dimensionality. When
            ``None`` the HAL probes ``env.action_dim`` or
            ``env._env.action_dim`` on connect; if neither is available
            it raises :class:`ROSConfigError` naming the backend (it
            never guesses a width). Pass this only for an env whose
            action space genuinely isn't introspectable.
    """

    description: RobotDescription

    def __init__(
        self,
        env: SimRollout,
        description: RobotDescription,
        *,
        action_packer: ActionPacker | None = None,
        env_reset_seed: int | None = None,
        env_action_dim: int | None = None,
        body_twist_dt_s: float = 0.05,
    ) -> None:
        """Bind the env + description; no env interaction until :meth:`connect`.

        Args:
            env: A :class:`~openral_sim.rollout.SimRollout` providing
                ``reset`` / ``step`` and (optionally) ``mujoco_handles``.
            description: The host :class:`RobotDescription` — joint
                ordering, ``base_joints`` + ``sim_joint_name`` map.
            action_packer: Per-composition Action-to-env-vec translator;
                defaults to :func:`pack_action_for_env`.
            env_reset_seed: Optional seed forwarded to ``env.reset``.
            env_action_dim: Override the auto-probed env action width;
                useful for envs whose action space isn't introspectable.
            body_twist_dt_s: Logical control timestep used when a
                BODY_TWIST action is applied via direct base-qpos write.
                Mirrors :attr:`PandaMobileHAL._dt_s` (default 0.05 — 20 Hz
                nav control rate). In deploy-sim this is the action tick,
                not a render cadence or wall-clock sleep: each tick advances
                the base by ``velocity * dt`` and advances MuJoCo elapsed sim
                time by the same ``dt`` so `/clock`, odom, TF, and Nav2
                deadlines share one timestep definition.
        """
        self._env = env
        # ADR-0036 — episodic backends (LIBERO) re-randomise the scene the instant
        # a task succeeds (lerobot's LiberoEnv.step resets inline). In a continuous
        # deploy twin the reasoner/mission own episode boundaries, not the env, so
        # ask the env to run continuously when it supports the hook (no-op for
        # backends without it, e.g. robocasa which already runs ignore_done).
        enable_continuous = getattr(env, "enable_continuous", None)
        if callable(enable_continuous):
            enable_continuous()
        self.description = description
        self._action_packer = action_packer if action_packer is not None else pack_action_for_env
        self._reset_seed = env_reset_seed
        self._env_action_dim: int | None = env_action_dim
        self._connected: bool = False
        self._estop_latched: bool = False
        self._last_state_ns: int = 0
        # ADR-0034 (2026-06-04 idle-stepper amendment) — monotonic timestamp
        # of the last real actuation that passed through ``send_action`` (the
        # single choke point both ``_on_safe_action`` and ``_on_cmd_vel``
        # reach). The sim-only free-running idle stepper reads this to yield
        # the env to an active skill: it skips an idle tick whenever a real
        # action arrived within the idle-hold window. ``0`` until the first
        # ``send_action`` (always "stale" so an idle scene starts stepping).
        self._last_action_ns: int = 0
        # Cached observation from the most recent ``reset`` / ``step``.
        # Used by :meth:`read_images` so the lifecycle node's camera
        # publisher can republish whatever the env rendered without
        # re-stepping the simulator. ``None`` until :meth:`connect`.
        self._last_obs: dict[str, Any] | None = None
        self._body_twist_dt_s: float = body_twist_dt_s
        # ADR-0034 §3.6 — built once per env on first read_state; reset on connect.
        self._joint_index: dict[str, int] | None = None
        # ADR-0028c — last action vector applied via composite-split
        # packing. Held across send_action calls so a per-mode chunk
        # that only fills one slot (e.g. CARTESIAN_DELTA arm) doesn't
        # silently zero out the other slots (e.g. gripper position),
        # which on the openral RoboCasa slot-dispatch path used to
        # cause the gripper to flick open between every arm tick. The
        # arm's OSC-delta slot is RE-ZEROED right before each step so
        # the policy's per-step delta is applied once, not accumulated.
        self._last_env_action: NDArray[np.float32] | None = None
        # ADR-0036 — latched when the last env.step reported terminal
        # (terminated/truncated). The next send_action resets the env
        # before stepping so episodic backends (LIBERO) never step a
        # terminated episode; robocasa runs ignore_done and never latches.
        self._episode_done: bool = False
        # send_action tick counter for the throttled diagnostic log.
        self._send_log_tick: int = 0
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
        # ADR-0048 Phase 1 — cross-reset sim-time offset. The wrapped
        # SimRollout's ``sim_time_ns`` reports time *within the current
        # episode*, and backends like robocasa rewind ``MjData.time`` to 0 on
        # every ``env.reset``. A ``/clock`` publisher must never see time go
        # backwards, so we accumulate each finished episode's elapsed sim-time
        # into this offset right before each reset (in
        # :meth:`_accumulate_sim_time_before_reset`) and add it to the live
        # per-episode reading in :meth:`sim_time_ns`. Stays ``0`` for a
        # clock-less wrapped rollout (whose every read is ``None``, so
        # :meth:`sim_time_ns` returns ``None`` and the offset is never used).
        self._sim_time_offset_ns: int = 0

    # ── HAL Protocol surface ────────────────────────────────────────────

    def connect(self) -> None:
        """Reset the env and cache the initial obs as the seed for read_state.

        Idempotent: a second `connect` re-resets the env at the same seed
        (the lifecycle node calls `connect` on each `configure` → `cleanup`
        cycle, so the contract must tolerate repeated calls).
        """
        # ADR-0048 Phase 1 — fold any elapsed sim-time into the cross-reset
        # offset BEFORE the reset rewinds the backend clock, so a re-connect
        # (the lifecycle node re-resets on each configure→cleanup cycle) never
        # makes the published ``/clock`` jump backwards. On the very first
        # connect a freshly built env reads ~0, so this is a no-op there.
        self._accumulate_sim_time_before_reset()
        try:
            obs = self._env.reset(seed=self._reset_seed)
        except Exception as exc:
            raise ROSRuntimeError(f"SimAttachedHAL.connect: env.reset failed: {exc}") from exc
        self._last_obs = dict(obs) if isinstance(obs, dict) else None
        self._last_state_ns = time.time_ns()
        self._connected = True
        self._episode_done = False  # fresh episode after (re)connect
        self._joint_index = None  # rebuilt on next read_state (model identity stable per env)
        if self._env_action_dim is None:
            self._env_action_dim = self._probe_env_action_dim()

    def _probe_env_action_dim(self) -> int:
        """Return the env's flat action dimensionality, or raise.

        Two probe paths, in order:

        1. ``self._env.action_dim`` — the direct attribute every backend
           exposes (robosuite/robocasa carry it natively; the native MuJoCo
           backends — ``so101_box``, ``tabletop_push``, ``openarm_tabletop_pnp``
           — expose it as a property reporting their true ``step`` width per
           ADR-0034's probe-gap fix).
        2. ``self._env._env.action_dim`` — robocasa wraps the raw robosuite
           env on ``_env`` for gymnasium-shaped envs and on a sibling
           attribute on the kitchen path; the inner env is what carries
           ``action_dim``.

        If neither path resolves AND no ``env_action_dim`` override was
        supplied to the constructor, this raises :class:`ROSConfigError`
        naming the backend — a loud boot-time failure beats a wrong-width
        mid-run E-stop. (Previously this silently fell back to ``11``, the
        robosuite BASIC composite width, so a native backend whose ``step``
        required a different width — ``so101_box`` → 6 — raised a width
        mismatch on the next ``env.step``, including the idle stepper firing
        autonomously on the bridge timer.) The fix is single-source-of-truth:
        every backend reports its own ``action_dim``; this method only
        introspects it, never guesses.

        Splitting this off keeps :meth:`connect` linear.

        Raises:
            ROSConfigError: the env exposes no introspectable ``action_dim``
                and no ``env_action_dim`` override was supplied.
        """
        if hasattr(self._env, "action_dim"):
            return int(self._env.action_dim)
        inner = getattr(self._env, "_env", None)
        if inner is not None and hasattr(inner, "action_dim"):
            return int(inner.action_dim)
        backend = type(self._env).__name__
        raise ROSConfigError(
            f"SimAttachedHAL: cannot resolve the env action width — backend "
            f"{backend!r} exposes no `action_dim` (nor does its inner `_env`), "
            "and no `env_action_dim` override was supplied to the constructor. "
            "Add an `action_dim` property to the backend's rollout (reporting "
            "its true `step` width) or pass `env_action_dim` explicitly. Refusing "
            "to guess a width — a wrong guess E-stops mid-run on the first env.step."
        )

    def disconnect(self) -> None:
        """Idempotent — release the env handle (we don't own its lifetime)."""
        self._connected = False

    def read_state(self) -> JointState:
        """Read live joint state from the env's MJCF qpos via the description.

        Walks `description.joints` and looks up each joint's
        `sim_joint_name` (falling back to `name`) in the env's MJCF.
        Returns the canonical :class:`JointState` the safety supervisor
        + the world_state aggregator consume.

        Raises:
            ROSRuntimeError: when called before `connect` or when the
                env hasn't been reset (no obs cached).
        """
        if not self._connected:
            raise ROSRuntimeError("SimAttachedHAL.read_state called before connect().")
        positions: list[float] = []
        velocities: list[float] = []
        handles = self._mujoco_handles()
        if handles is None:
            # No MJCF handle (a non-MuJoCo backend — e.g. the Isaac Sim sidecar,
            # ManiSkill3 / SimplerEnv). Prefer the real joint angles the
            # SimRollout surfaces as obs["joint_positions"] (in description-joint
            # order) so /joint_states carries live values; fall back to zeros
            # (shape-correct) when the backend provides none. Never reached on
            # the MuJoCo path. ADR-0034 amendment (non-MuJoCo joint-state).
            names = [j.name for j in self.description.joints]
            njoints = len(names)

            def _from_obs(key: str) -> list[float]:
                """Read a per-joint vector from the cached obs, padded/truncated to njoints."""
                raw = self._last_obs.get(key) if self._last_obs is not None else None
                if raw is None:
                    return [0.0] * njoints
                vec = np.asarray(raw, dtype=np.float64).reshape(-1)
                return [float(vec[i]) if i < vec.shape[0] else 0.0 for i in range(njoints)]

            return JointState(
                name=names,
                position=_from_obs("joint_positions"),
                velocity=_from_obs("joint_velocities"),
                effort=[0.0] * njoints,
                stamp_ns=time.time_ns(),
            )
        model, data = handles
        import mujoco  # noqa: PLC0415  # reason: optional dep guarded by caller

        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
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
                continue
            positions.append(float(qpos[int(model.jnt_qposadr[jid])]))
            velocities.append(float(qvel[int(model.jnt_dofadr[jid])]))
        self._last_state_ns = time.time_ns()
        return JointState(
            name=[j.name for j in self.description.joints],
            position=positions,
            velocity=velocities,
            effort=[0.0] * len(self.description.joints),
            stamp_ns=self._last_state_ns,
        )

    def send_action(self, action: Action) -> None:
        """Step the env with the packed action vector.

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
        # Stamp the real-actuation clock FIRST — before any early return — so
        # the idle stepper yields even on a dropped (estop) or rejected tick:
        # a skill is still actively trying to drive, and the idle stepper must
        # not race its env.step. This is the single choke point every real
        # action passes (both _on_safe_action and _on_cmd_vel reach here).
        self._last_action_ns = time.monotonic_ns()
        if not self._connected:
            raise ROSRuntimeError("SimAttachedHAL.send_action called before connect().")
        if self._estop_latched:
            return
        if self._env_action_dim is None:
            raise ROSRuntimeError(
                "SimAttachedHAL.send_action: env_action_dim resolved to None; "
                "re-connect or pass it explicitly to the constructor."
            )
        # BODY_TWIST direct-qpos path. The default ``pack_action_for_env``
        # packs ``[vx, vy, wz]`` into slots 0-2 of robocasa's composite-
        # controller action vector, but robocasa's BASIC controller
        # doesn't interpret those slots as planar velocities for the
        # OmronMobileBase — the base doesn't move when commanded that
        # way. Mirror :meth:`PandaMobileHAL._apply_body_twist` instead:
        # rotate the body-frame twist into world frame by the current
        # yaw, Euler-integrate by ``body_twist_dt_s``, and write the
        # three base qpos slots directly. Skips ``env.step()`` so the
        # arm dynamics don't churn (the navigate-kitchen use case wants
        # the base to translate; the arm stays in its starting pose).
        # JOINT_POSITION continues to flow through ``env.step()`` so
        # robosuite's per-joint controllers run as before.
        if action.control_mode is ControlMode.BODY_TWIST:
            # ADR-0028c — body_twist payload moved from joint_targets
            # (legacy lie) to the typed Action.body_twist field.
            if not action.body_twist:
                raise ROSConfigError(
                    "SimAttachedHAL.send_action: empty Action.body_twist for BODY_TWIST."
                )
            row = list(action.body_twist[0])
            # MuJoCo integrates the base by direct qpos write (skips env.step so the
            # arm dynamics don't churn). A non-MuJoCo backend (Isaac kinematic base,
            # ADR-0045) has no qpos handle — it integrates the base inside env.step,
            # so route the twist through the env action vector instead.
            if self._mujoco_handles() is not None:
                self._apply_body_twist_to_qpos(row)
            else:
                self._apply_body_twist_via_env_step(row)
            return
        # A non-BODY_TWIST action means the base is no longer being
        # velocity-commanded — clear the latched twist so /odom doesn't
        # report a stale base velocity through an arm-only step.
        self._last_body_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # ADR-0028c — robosuite composite controllers expose part-name
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
            # single policy step splits into (ADR-0036).
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
        # so an ``_log.info`` call would vanish.
        self._send_log_tick += 1
        if self._send_log_tick == 1 or self._send_log_tick % 50 == 0:
            head = ",".join(f"{float(v):+.3f}" for v in env_action[:8])
            print(
                f"[sim_attached.send_action] tick={self._send_log_tick} "
                f"mode={action.control_mode.value} env_dim={len(env_action)} "
                f"head=[{head}]",
                flush=True,
            )
        # If the prior step latched terminal, the commanded-slot merge state
        # belongs to the now-finished episode — clear it BEFORE the shared
        # step helper resets the env, so a fresh episode starts from a clean
        # slot vector. This one reset is send_action-only: ``idle_step`` must
        # NOT touch ``_last_env_action`` (an idle HOLD is orthogonal to the
        # commanded slots), so it stays out of ``_step_and_cache``.
        if self._episode_done:
            self._last_env_action = None  # stale: belongs to the prior episode
        self._step_and_cache(env_action, source="send_action")

    def _step_and_cache(self, env_action: NDArray[np.float32], *, source: str) -> bool:
        """Deferred-reset → ``env.step`` → re-cache ``_last_obs`` → re-latch terminal.

        The shared core of :meth:`send_action` and :meth:`idle_step` (ADR-0036)
        so the subtle terminal-reset path cannot drift between the two callers.

        ADR-0036 — auto-reset on episode termination. Native episodic backends
        (LIBERO) set ``StepResult.terminated`` / ``truncated`` when the task
        ends (success / failure / horizon); robocasa runs with ``ignore_done``
        and never does. Without a reset the next ``env.step`` raises "executing
        action in terminated episode" and the deploy-sim continuous-control
        loop spams failures. Resetting here makes every episodic backend behave
        like robocasa's ``ignore_done`` (continuous operation) — safe in a
        digital twin (no real motor), the arm simply starts a fresh episode.
        The reset is deferred to the NEXT step (not done eagerly when the
        terminal flag is set) so the terminal frame is still served to
        read_state / read_images before the env re-randomises.

        Does NOT touch ``_last_env_action`` — the send_action-only stale-slot
        reset stays in :meth:`send_action` so an idle HOLD does not perturb the
        commanded-slot merge state.

        Args:
            env_action: The flat env-frame action vector to step with.
            source: Caller tag (``"send_action"`` / ``"idle_step"``) used in
                the diagnostic log + the wrapped ``ROSRuntimeError`` messages.

        Returns:
            ``True`` once the env has been stepped (callers that may early-out
            before reaching here return ``False`` themselves).

        Raises:
            ROSRuntimeError: when ``env.reset`` (after termination) or
                ``env.step`` raises.
        """
        if self._episode_done:
            self._reset_terminated_episode(source, trigger="returned-terminal")
        try:
            step_result = self._env.step(env_action)
        except Exception as exc:
            # ADR-0036 (amended) — raw robosuite-backed adapters can HARD-RAISE
            # "executing action in terminated episode" on a post-terminal step instead of
            # returning a terminal ``StepResult``, so the returned-flag latch
            # above never fired and ``_episode_done`` is still False. Treat a
            # *raised* terminal exactly like a *returned* one: reset once and
            # re-step so deploy-sim's continuous twin keeps driving instead of
            # spamming the failure with a frozen arm. robosuite's ``reset`` clears
            # ``done`` (``environments/base.py``), so the re-step cannot re-raise
            # the same guard. Gymnasium / native backends never raise this, so the
            # branch is inert for them; any non-terminal failure propagates.
            if not is_terminated_episode_error(exc):
                raise ROSRuntimeError(f"SimAttachedHAL.{source}: env.step failed: {exc}") from exc
            self._reset_terminated_episode(source, trigger="raised-terminal")
            try:
                step_result = self._env.step(env_action)
            except Exception as exc2:
                raise ROSRuntimeError(
                    f"SimAttachedHAL.{source}: env.step failed after terminal-episode reset: {exc2}"
                ) from exc2
        # ``StepResult.observation`` carries the rendered camera frames
        # (see ``openral_sim.rollout.StepResult``). Cache so the
        # lifecycle node's camera publisher can serve them without
        # re-stepping the simulator. A dict-shaped Protocol fallback
        # ``getattr(..., 'observation', None)`` is enough because every
        # in-tree backend returns a ``StepResult`` with this attribute.
        obs = getattr(step_result, "observation", None)
        if isinstance(obs, dict):
            self._last_obs = dict(obs)
        # Latch terminal so the NEXT step resets before stepping.
        self._episode_done = bool(
            getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False)
        )
        return True

    def _reset_terminated_episode(self, source: str, *, trigger: str) -> None:
        """Reset the env after an episode terminal and clear the terminal latch.

        Shared by both terminal paths in :meth:`_step_and_cache` (ADR-0036):
        the *returned*-terminal latch (``_episode_done`` set by the prior step's
        ``StepResult``) and the *raised*-terminal recovery (raw-robosuite
        ``ignore_done=False`` backends that throw
        :func:`is_terminated_episode_error` instead of returning a terminal).
        Re-caches ``_last_obs`` from the reset observation and drops
        ``_joint_index`` so it is rebuilt against the fresh episode.

        Args:
            source: Caller tag (``"send_action"`` / ``"idle_step"``) for the
                diagnostic log + the wrapped ``ROSRuntimeError`` message.
            trigger: ``"returned-terminal"`` or ``"raised-terminal"`` — surfaced
                in the log so the two paths are distinguishable in deploy-sim
                output (ADR-0034 §1.4 observability).

        Raises:
            ROSRuntimeError: when ``env.reset`` itself fails.
        """
        # ADR-0048 Phase 1 — accumulate the finished episode's elapsed sim-time
        # into the cross-reset offset BEFORE the backend rewinds its clock, so
        # ``sim_time_ns`` (and the ``/clock`` publisher reading it) stays
        # monotonic non-decreasing across the auto-reset. Covers BOTH terminal
        # paths (returned-terminal latch + raised-terminal recovery) since both
        # funnel through this helper.
        self._accumulate_sim_time_before_reset()
        try:
            reset_obs = self._env.reset(seed=self._reset_seed)
        except Exception as exc:
            raise ROSRuntimeError(
                f"SimAttachedHAL.{source}: env.reset after episode termination failed: {exc}"
            ) from exc
        self._last_obs = dict(reset_obs) if isinstance(reset_obs, dict) else None
        self._episode_done = False
        self._joint_index = None  # rebuilt on next read_state
        print(
            f"[sim_attached.{source}] episode terminated ({trigger}); auto-reset "
            f"(robot={self.description.name}) — continuous deploy-sim (ADR-0036)",
            flush=True,
        )

    def idle_step(self) -> bool:
        """Advance the wrapped sim one tick with a zero/HOLD action when idle.

        SIM-ONLY. This refreshes ``_last_obs`` (camera frames + state) so the
        ADR-0035 perception / object-detector bus sees a live scene even when
        no skill is executing — without this the env only steps on
        ``/openral/safe_action`` receipt, so an idle scene freezes physics and
        cameras go stale.

        Safety: this method is defined ONLY on :class:`SimAttachedHAL`. Real
        HALs (Franka FCI, ros2_control bridges, lerobot followers) do NOT
        define it, and the :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`
        gates its idle timer on ``callable(getattr(hal, "idle_step", None))``,
        so the timer is never even created against a real HAL. This is the real
        guarantee — NOT "zero is harmless": a zero vector is a HOLD for the
        sim's velocity / OSC-delta controllers, but on a real absolute-position
        arm it would command "drive every joint to 0 rad" (violent). The
        method-only-on-sim exclusion is what makes the zero action safe.

        Does NOT touch the commanded-slot state (``_last_env_action``) nor the
        latched base twist (``_last_body_twist``) — an idle HOLD is orthogonal
        to whatever a skill last commanded; the next ``send_action`` resumes
        from its own merged state.

        Caveat — zero is a true HOLD only for the sim's velocity / OSC-delta /
        robosuite composite controllers. For a **position-controlled native
        backend** (e.g. ``so101_box``, whose ``step`` consumes joint-position
        targets) a zero vector commands the joints *toward 0 rad* rather than
        holding the current pose. That is acceptable here — the goal is to keep
        the scene physically live so cameras render, not to freeze the arm in
        place — but it is NOT a literal hold for those backends.

        Action-dim note: ``_env_action_dim`` is the env's authoritative width,
        resolved by :meth:`_probe_env_action_dim` from the backend's own
        ``action_dim`` (every native backend — ``so101_box`` → 6,
        ``tabletop_push`` → actuator count, ``openarm_tabletop_pnp`` →
        ``state_dim`` — now reports it). A backend that exposes no width and
        carries no override fails loudly at ``connect`` time, so the idle tick
        never builds a wrong-width zero vector. (The bridge's catch-once-and-
        disable guard around :meth:`idle_step` remains as defence in depth.)

        Returns:
            ``True`` if the env was stepped, ``False`` if suppressed (not
            connected, estop latched, or action dim unresolved). Backend-agnostic
            since the ADR-0034 amendment — no MuJoCo-handle gate.
        """
        if not self._connected:
            return False
        if self._estop_latched:
            # An estopped HAL freezes — honoring the estop contract is correct;
            # the frozen scene is the intended, safe behaviour here.
            return False
        if self._env_action_dim is None:
            return False
        # No MuJoCo-handle gate: idle-stepping is valid for ANY wrapped
        # SimRollout — a zero action is a HOLD for the sim's velocity / OSC-delta
        # controllers, and the method-only-on-SimAttachedHAL exclusion (real HALs
        # never define idle_step) is the real safety guarantee. Non-MuJoCo
        # backends (Isaac Sim sidecar, ManiSkill3) step via env.step(zeros) the
        # same as MJCF ones. ADR-0034 amendment.
        # Zero-action step — the same env.step(zeros) idiom as ADR-0036's
        # deferred-reset path / send_action's tail (NOT robocasa.refresh_obs,
        # which re-renders WITHOUT stepping). The shared ``_step_and_cache``
        # does the deferred reset → step → obs re-cache → terminal re-latch so
        # this path cannot drift from send_action. Never build a non-zero
        # vector here. ``_step_and_cache`` deliberately leaves
        # ``_last_env_action`` / ``_last_body_twist`` untouched.
        zero_action = np.zeros(self._env_action_dim, dtype=np.float32)
        return self._step_and_cache(zero_action, source="idle_step")

    # ── ADR-0028c per-mode → composite-controller slot mapping ──────────
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

        Uses the composite controller's authoritative slot indexes.

        Mirrors the upstream ``robocasa.wrappers.gym_wrapper.RoboCasaGymEnv.
        step`` / ``PandaOmronKeyConverter.unmap_action`` pattern. Handles
        the four slot kinds the openral RoboCasa rSkill manifests
        actually emit today (CARTESIAN_DELTA arm, GRIPPER_POSITION
        finger, JOINT_VELOCITY base, JOINT_POSITION fallback). BODY_TWIST
        has its own direct qpos path and never reaches this helper.

        Layout-agnostic: works for PandaMobile+BASIC (action_dim=11) and
        PandaOmron+HybridMobileBase (action_dim=12) without per-layout
        branching, because ``cc._action_split_indexes`` carries the
        truth for whichever composite was instantiated.

        Slots from previous send_action calls within the same policy
        tick persist in ``self._last_env_action`` so multi-Action
        dispatch (arm + gripper, etc.) doesn't zero each other out —
        the arm's OSC-delta slot is RE-ZEROED on every call so the
        per-step delta is applied once, not accumulated.

        ADR-0036 — handles CARTESIAN_DELTA, GRIPPER_POSITION,
        JOINT_VELOCITY, COMPOSITE_MODE, and delegates JOINT_POSITION to
        ``pack_action_for_env``; all members of
        ``openral_core.SIM_EXECUTABLE_CONTROL_MODES``. Any other mode hits
        the closing ``else`` and raises (the drift guard the lockstep test
        proves).
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
            # ADR-0028d — route a JOINT_VELOCITY chunk to the
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
            # ADR-0028d — sim-only multiplexer flag. Write the policy's
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
        # i.e. arm is frozen). ADR-0028d promotes this to a first-class
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
        """Euler-integrate a 6-vec body twist directly into MuJoCo base qpos.

        Mirrors :meth:`PandaMobileHAL._apply_body_twist`. Rotates the
        body-frame velocity ``(vx, vy)`` into world frame by the
        current ``base_yaw``, then adds ``velocity * dt`` to each base
        joint's qpos slot. Yaw wraps to ``[-π, π]``.

        Raises:
            ROSConfigError: when ``row`` is not a 6-vec or its
                non-planar components (vz / wx / wy) are non-zero.
            ROSRuntimeError: when no MuJoCo handles are bound (the
                env isn't MJCF-backed — non-applicable backends should
                never see a BODY_TWIST action).
        """
        import math  # noqa: PLC0415  # reason: stdlib defer

        import mujoco  # noqa: PLC0415  # reason: optional dep, guarded by handles check

        if len(row) != BODY_TWIST_DIM:
            raise ROSConfigError(
                f"SimAttachedHAL: BODY_TWIST action expects {BODY_TWIST_DIM} "
                f"floats per row (vx, vy, vz, wx, wy, wz); got {len(row)}."
            )
        if any(abs(row[i]) > _PLANAR_TWIST_EPS for i in (2, 3, 4)):
            raise ROSConfigError(
                "SimAttachedHAL: BODY_TWIST row carries non-zero linear-z / "
                "angular-x / angular-y components; the panda_mobile base is "
                "holonomic planar — only vx, vy, wz are actuated."
            )
        handles = self._mujoco_handles()
        if handles is None:
            raise ROSRuntimeError(
                "SimAttachedHAL: BODY_TWIST received but no MuJoCo handles "
                "on the bound env — this backend cannot integrate base velocity."
            )
        model, data = handles
        # Resolve the three planar base joint names via the same lookup
        # chain :attr:`base_pose` uses (description.base_joints +
        # sim_joint_name override, fallback to first three joints).
        bj = self.description.base_joints
        if bj is not None and len(bj) >= 3:  # noqa: PLR2004  # reason: x/y/yaw triple
            joints_by_name = {j.name: j for j in self.description.joints}
            base_joint_names = tuple(
                (joints_by_name[name].sim_joint_name or name) for name in bj[:3]
            )
        else:
            base_joint_names = tuple(
                (j.sim_joint_name or j.name) for j in self.description.joints[:3]
            )
        addrs: list[int] = []
        for sim_name in base_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if jid < 0:
                raise ROSRuntimeError(
                    f"SimAttachedHAL: base joint {sim_name!r} not found in MJCF; "
                    "cannot apply BODY_TWIST. Check robot.yaml base_joints + sim_joint_name."
                )
            addrs.append(int(model.jnt_qposadr[jid]))

        vx_body, vy_body, _vz, _wx, _wy, wz = row
        # Latch the commanded twist for the /odom publisher (base_link frame).
        self._last_body_twist = (vx_body, vy_body, 0.0, 0.0, 0.0, wz)
        yaw = float(data.qpos[addrs[2]])
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx_world = cy * vx_body - sy * vy_body
        vy_world = sy * vx_body + cy * vy_body
        dt = self._body_twist_dt_s
        data.qpos[addrs[0]] = float(data.qpos[addrs[0]]) + vx_world * dt
        data.qpos[addrs[1]] = float(data.qpos[addrs[1]]) + vy_world * dt
        new_yaw = yaw + wz * dt
        # Wrap to [-π, π] so long sessions don't accumulate drift.
        new_yaw = (new_yaw + math.pi) % (2.0 * math.pi) - math.pi
        data.qpos[addrs[2]] = new_yaw
        # The direct-qpos path intentionally bypasses env.step, but it is still
        # a simulation timestep. Advance MuJoCo's elapsed time by the same
        # command interval used for kinematic integration so SimRollout
        # sim_time_ns() (and therefore deploy-sim /clock) cannot freeze while
        # Nav2 is actively streaming BODY_TWIST commands.
        data.time = float(data.time) + dt
        # Propagate the qpos change into derived state (body xforms +
        # sensor positions) so the live MuJoCo viewer + ``/scan``
        # ray-cast see the new base pose on the same tick. No physics
        # step — just the kinematic update.
        mujoco.mj_forward(model, data)
        # Re-render camera observations against the new model+data.
        # Without this the WorldState aggregator + dashboard show the
        # connect-time frame for every BODY_TWIST command (env.step is
        # the only path that normally refreshes ``_last_obs``, and we
        # skip it here). Backends without the ``refresh_obs`` method
        # (in-process digital twins, gymnasium-wrapped envs) silently
        # keep the cached frame — visual lag is the documented
        # tradeoff for those paths.
        refresh = getattr(self._env, "refresh_obs", None)
        if refresh is not None:
            refreshed = refresh()
            if refreshed is not None:
                self._merge_refreshed_obs(refreshed)

    def _apply_body_twist_via_env_step(self, row: list[float]) -> None:
        """Integrate a BODY_TWIST through ``env.step`` (non-MuJoCo planar base).

        For a backend without a qpos handle (the Isaac kinematic base, ADR-0045)
        the planar base lives inside the env: the scene integrates ``(vx, vy, wz)``
        and teleports its root each ``env.step``. We pack the body-frame twist into
        the **final three** slots of the env action vector — the convention the
        manifest scene uses (``[arm…, gripper, vx, vy, wz]``) — and leave the
        arm/gripper slots at zero so a pure base move holds the arm. The scene
        integrates by its own command-interval dt, so we pass the velocity raw.
        """
        if any(abs(row[i]) > _PLANAR_TWIST_EPS for i in (2, 3, 4)):
            raise ROSConfigError(
                "SimAttachedHAL: BODY_TWIST row carries non-zero linear-z / "
                "angular-x / angular-y components; a holonomic planar base only "
                "actuates vx, vy, wz."
            )
        if self._env_action_dim is None or self._env_action_dim < 3:  # noqa: PLR2004  # reason: vx/vy/wz triple
            raise ROSRuntimeError(
                "SimAttachedHAL: BODY_TWIST on a non-MuJoCo backend needs an env "
                f"action vector of at least 3 slots; got {self._env_action_dim}. "
                "Does this robot declare a planar base (base_joints)?"
            )
        vx_body, vy_body, _vz, _wx, _wy, wz = row
        # Latch the commanded twist for the /odom publisher (base_link frame).
        self._last_body_twist = (vx_body, vy_body, 0.0, 0.0, 0.0, wz)
        env_action = np.zeros(self._env_action_dim, dtype=np.float32)
        env_action[-3:] = (vx_body, vy_body, wz)
        self._step_and_cache(env_action, source="cmd_vel")

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
        ``odom → base_link`` source via :meth:`base_pose_6dof`. Dropping
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
        """Latch e-stop. Subsequent send_action calls are dropped."""
        self._estop_latched = True

    # ── Helpers exposed to the lifecycle node ──────────────────────────

    def mujoco_handles(self) -> tuple[Any, Any] | None:
        """Forward the underlying env's MJCF handles, or None.

        Used by the panda_mobile ROS lifecycle node to bind its
        ``mujoco_handle_provider`` so ``/scan`` ray-casts against the
        live env instead of the no-hit fallback. Generic across
        robots — any :class:`SimRollout` that exposes
        ``mujoco_handles()`` works.
        """
        return self._mujoco_handles()

    def _rollout_sim_time_ns(self) -> int | None:
        """Read the wrapped rollout's per-episode sim time, or ``None``.

        ``sim_time_ns`` is an OPTIONAL duck-typed extension of the
        :class:`~openral_sim.rollout.SimRollout` protocol (ADR-0048 Phase 1) —
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

        Called immediately BEFORE every ``env.reset`` (connect + both
        auto-reset terminal paths). Backends such as robocasa rewind
        ``MjData.time`` to 0 on reset, so without this the published value
        would jump backwards on each new episode. Reads the live per-episode
        sim time and, when the backend has a clock, adds it to
        :attr:`_sim_time_offset_ns`. A clock-less backend (``None``) leaves the
        offset untouched — :meth:`sim_time_ns` then also returns ``None``.
        """
        elapsed = self._rollout_sim_time_ns()
        if elapsed is not None:
            self._sim_time_offset_ns += elapsed

    def sim_time_ns(self) -> int | None:
        """Cross-reset-monotonic elapsed simulation time in ns, or ``None``.

        ADR-0048 Phase 1 — the value a sim ``/clock`` publisher reads so the
        deploy-sim ROS graph runs on simulation time. Returns the wrapped
        :class:`~openral_sim.rollout.SimRollout`'s per-episode sim time plus the
        accumulated offset from all prior episodes (:meth:`connect` and the
        ADR-0036 auto-resets fold each finished episode's elapsed time into the
        offset before the backend rewinds its clock). The result is therefore
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
        must then keep the graph on the host-wall clock authority.
        """
        if self.sim_time_ns() is None:
            return ClockAuthority.host_wall()
        return ClockAuthority.simulation(
            type(self._env).__name__,
            timestep_s=self._body_twist_dt_s,
            publishes_ros_clock=True,
        )

    def _mujoco_handles(self) -> tuple[Any, Any] | None:
        """Return the env's MJCF (model, data) tuple, or None.

        The :class:`SimRollout` protocol declares ``mujoco_handles()``
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
    def env(self) -> SimRollout:
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
        this (via :func:`~openral_hal.sim_sensor_bridge.should_idle_step`) to
        yield the env to an active skill — it skips an idle tick whenever a
        real action arrived within the idle-hold window.
        """
        return self._last_action_ns

    def read_images(self) -> dict[str, Any]:
        """Return the latest rendered camera frames keyed by camera name.

        The wrapped :class:`SimRollout` returns rendered images on each
        ``reset`` / ``step`` under the ``"images"`` slot of its
        :class:`Observation` dict (per ``openral_sim.rollout`` schema).
        The HAL caches that slot so the panda_mobile lifecycle node's
        camera publisher can republish the frames as
        ``sensor_msgs/Image`` on ``/openral/cameras/<name>/image`` at
        the configured rate — that's the topic WorldState subscribes to
        and the path the rldx / pi05 / smolvla adapters consume via
        ``observation.images.<name>``. Frame keys match the canonical
        ``camera1`` / ``camera2`` / ``camera3`` aliases the robocasa
        adapter exposes in :meth:`openral_sim.backends.robocasa.
        _RoboCasaSim._wrap_obs` plus the raw robosuite keys
        (``robot0_agentview_left_image`` etc.); the caller chooses
        which subset to forward.

        Returns an empty dict when no observation has been cached yet
        (e.g. before :meth:`connect`) or when the observation has no
        ``"images"`` slot (non-image backends). Never raises.
        """
        if self._last_obs is None:
            return {}
        images = self._last_obs.get("images")
        if not isinstance(images, dict):
            return {}
        return dict(images)

    def read_depth_clouds(self) -> dict[str, NDArray[np.float32]]:
        """Return per-depth-sensor point clouds ``{name: (N, 3) base_link}``.

        A non-MuJoCo backend that renders depth (the Isaac manifest scene,
        ADR-0045) surfaces clouds already deprojected to ``base_link`` under the
        ``"depth_points"`` obs slot — Isaac's ``Camera.get_pointcloud`` owns the
        camera convention, so the HAL never re-derives geometry. ``SimSensorBridge``
        publishes them as ``PointCloud2`` for octomap. Empty when the backend
        renders no depth (MuJoCo backends use the ray-cast path instead). Never
        raises.
        """
        if self._last_obs is None:
            return {}
        clouds = self._last_obs.get("depth_points")
        if not isinstance(clouds, dict):
            return {}
        return {str(k): np.asarray(v, dtype=np.float32).reshape(-1, 3) for k, v in clouds.items()}

    def read_scan(self) -> NDArray[np.float32] | None:
        """Return the 2-D LaserScan range fan, or ``None``.

        A non-MuJoCo backend that ray-casts a 2-D lidar (the Isaac scene, ADR-0045)
        surfaces the per-beam ranges (``base_link`` frame, ``angle_min=-π`` →
        ``angle_max=+π``, the bridge's convention) under the ``"scan"`` obs slot.
        ``SimSensorBridge._compute_scan_ranges`` reads it for ``/scan``. ``None``
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

        Mirrors :attr:`PandaMobileHAL.base_pose` so the panda_mobile
        lifecycle node's ``/odom`` publisher works regardless of which
        HAL is wired in. Reads the three joint positions named in
        ``description.base_joints`` (typically
        ``[base_x, base_y, base_yaw]``) — falls back to the first three
        ``description.joints`` if ``base_joints`` is unset.

        For a non-MuJoCo backend that drives a planar base (e.g. the Isaac
        kinematic base, ADR-0045) the pose comes from ``obs["base_pose"]``
        ``= (x, y, yaw)`` the SimRollout surfaces; this is what feeds the
        ``/odom`` publisher there. Falls back to ``(0.0, 0.0, 0.0)`` when the
        backend reports neither (non-mobile backends like PushT) — the same
        neutral pose the lifecycle node's ``getattr(..., (0.0, 0.0, 0.0))``
        fallback would supply.
        """
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

        base_joint_names: tuple[str, ...]
        bj = self.description.base_joints
        if bj is not None and len(bj) >= 3:  # noqa: PLR2004  # reason: x/y/yaw triple
            joints_by_name = {j.name: j for j in self.description.joints}
            base_joint_names = tuple(
                (joints_by_name[name].sim_joint_name or name) for name in bj[:3]
            )
        else:
            base_joint_names = tuple(
                (j.sim_joint_name or j.name) for j in self.description.joints[:3]
            )
        qpos = np.asarray(data.qpos, dtype=np.float64)
        out: list[float] = []
        for sim_name in base_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if jid < 0:
                out.append(0.0)
                continue
            out.append(float(qpos[int(model.jnt_qposadr[jid])]))
        return (out[0], out[1], out[2])

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """Last commanded base body twist ``(vx, vy, vz, wx, wy, wz)``.

        In the ``base_link`` frame. The base advances by exact Euler
        integration of this command, so it is the base's velocity — the
        panda_mobile lifecycle node publishes it as the ``/odom`` twist.
        Zeroed once a non-BODY_TWIST action is sent.
        """
        return self._last_body_twist

    def base_pose_6dof(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        """Full 6-DoF base pose ``(xyz, quat_xyzw)`` from the cached robocasa obs.

        ``base_pose`` (planar ``x, y, yaw``) is what the panda_mobile
        HAL was originally designed around — Nav2 / SLAM consume that
        as a 2-D twist + yaw. But the rldx / pi05 state assemblers
        (ADR-0027) read the base's *full* 6-DoF pose via
        ``tf("odom", "base_link")``, and the planar version sets
        ``z = 0.0`` + ``roll = pitch = 0``, which silently drops the
        ~0.70 m platform height that RoboCasa proprio
        (``robot0_base_pos[2]``) reports. Symptom in ``openral deploy sim``:
        the assembled ``world_to_base.position.z`` is ``0.0`` instead
        of ``0.70``, so the policy sees the arm 0.49 m higher in the
        base frame than at training time and tries to reach a target
        that's at the wrong height — the "robot moves but grabs at
        the wrong position" failure mode the dump diff exposed.

        Source of truth: ``_last_obs["raw_proprio"]["robot0_base_pos"]``
        + ``["robot0_base_quat"]`` — the values the RoboCasa env's
        robosuite observable system computes (the SAME values
        ``openral sim run`` feeds to the policy via
        ``robot0_base_pos`` in the 16-D ``human300`` state vector).
        Reading them here keeps the deploy_sim ``odom → base_link``
        TF byte-identical to what sim_run's state assembly produces.

        Note: ``data.xpos[robot0_base]`` (raw MJCF body world-frame
        position) is NOT the same as ``robot0_base_pos`` — robosuite's
        observable wraps the body lookup with a robot-specific offset
        (``robot.robot_model.base_xpos_offset``) so the publicly
        reported "base position" matches a stable robot-mount
        reference, not the MJCF body's own anchor. We MUST use the
        observable's output, not the raw xpos, or the policy sees a
        translated world.

        Returns ``None`` when the wrapped obs has no ``raw_proprio``
        slot (non-RoboCasa backend) or the keys are missing — the
        caller falls back to the planar :attr:`base_pose`.
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
        """Clear the estop latch. Caller asserts the cause has been resolved."""
        self._estop_latched = False
