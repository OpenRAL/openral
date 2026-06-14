"""HAL stub for the panda_mobile mobile-manipulator (ADR-0025).

In-process digital-twin HAL for the ``panda_mobile`` embodiment: a
Franka 7-DoF arm mounted on a holonomic three-DoF planar base. The
real robosuite/robocasa-backed sim adapter that drives MuJoCo physics
lives in :mod:`openral_sim.backends.robocasa`; this module provides the
*HAL Protocol* surface (``connect`` / ``disconnect`` /
``read_state`` / ``send_action`` / ``estop``) so the higher layers
(safety supervisor, ``RskillRunnerNode``, dashboard) can be exercised
end-to-end without a robosuite / MuJoCo install.

The HAL maintains 10-DoF integrator state:

* Base: ``base_x`` (m), ``base_y`` (m), ``base_yaw`` (rad). Driven by
  :attr:`~openral_core.ControlMode.BODY_TWIST` actions whose
  ``joint_targets`` carry six floats — only the first three (linear x,
  linear y, angular z) are honoured; the others are zeroed and warned
  on if non-zero, per the planar-base convention documented on
  :class:`~openral_core.schemas.ControlMode.BODY_TWIST`.
* Arm: ``panda_joint1..7``. Driven by
  :attr:`~openral_core.ControlMode.JOINT_POSITION` actions whose
  ``joint_targets`` carry seven floats. The gripper joint is not
  modelled here — its motion is locally trivial and the existing
  Franka HAL already covers it; Nav2 + SLAM exercise does not need
  the gripper.

The follow-up ADR-0025 implementation steps are documented in the
plan: a real ament-python ``packages/openral_hal_panda_mobile/`` ROS
lifecycle node that subscribes ``/openral/safe_action`` and publishes
``/joint_states`` + ``/odom`` + a MuJoCo-ray-cast-derived ``/scan``;
plus the matching ``robocasa.py`` adapter changes to expose base
velocity and synthesise the laser scan.

Example:
    >>> from openral_hal.panda_mobile import PandaMobileHAL
    >>> hal = PandaMobileHAL()
    >>> hal.connect()
    >>> state = hal.read_state()
    >>> len(state.position)
    10
"""

from __future__ import annotations

from pathlib import Path

from openral_core import BODY_TWIST_DIM, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import (
    Action,
    ControlMode,
    JointState,
)

__all__ = [
    "PANDA_MOBILE_BASE_JOINT_NAMES",
    "PANDA_MOBILE_DESCRIPTION",
    "PANDA_MOBILE_JOINT_NAMES",
    "PandaMobileHAL",
]


def _load_panda_mobile_description() -> RobotDescription:
    """Load the canonical ``robots/panda_mobile/robot.yaml`` once at import.

    The :class:`HALLifecycleNodeBase` reads ``self._hal.description`` to
    populate the per-joint limit attributes on ``hal.read_state`` OTel
    spans + to size the ``JointState`` message. We load the same YAML
    that the robot registry / Reasoner palette consults so the
    description is the single source of truth — never a hand-coded
    duplicate.

    Returns:
        :class:`~openral_core.RobotDescription` for ``panda_mobile``.

    Raises:
        :class:`~openral_core.exceptions.ROSConfigError`: when the YAML
            is missing or malformed.
    """
    here = Path(__file__).resolve()
    # python/hal/src/openral_hal/panda_mobile.py → repo root.
    repo_root = here.parents[4]
    yaml_path = repo_root / "robots" / "panda_mobile" / "robot.yaml"
    if not yaml_path.is_file():
        raise ROSConfigError(
            f"PandaMobileHAL: canonical robot.yaml missing at {yaml_path}. "
            "The HAL needs the description for OTel span attributes + "
            "JointState publication sizing."
        )
    return RobotDescription.from_yaml(str(yaml_path))


PANDA_MOBILE_DESCRIPTION: RobotDescription = _load_panda_mobile_description()
"""Canonical ``RobotDescription`` for ``panda_mobile`` — single source
of truth from ``robots/panda_mobile/robot.yaml``."""

# Joint names are derived from the canonical description (single source
# of truth) rather than hardcoded: base from ``base_joints``, arm +
# gripper from the per-joint ``role`` annotations in robot.yaml.
PANDA_MOBILE_BASE_JOINT_NAMES: list[str] = list(PANDA_MOBILE_DESCRIPTION.base_joints or [])
"""Base joints (forward, side, yaw), in order — from
``robots/panda_mobile/robot.yaml`` ``base_joints``."""

_PANDA_MOBILE_ARM_JOINT_NAMES: list[str] = [
    j.name for j in PANDA_MOBILE_DESCRIPTION.joints if j.role == "arm"
]

# ADR-0028a — the parallel gripper is declared as a 1-DoF joint
# (role ``"gripper"``); the digital-twin HAL tracks it as the trailing
# qpos slot so the published JointState aligns with the robot.yaml
# inventory.
_PANDA_MOBILE_GRIPPER_JOINT_NAME: str = next(
    j.name for j in PANDA_MOBILE_DESCRIPTION.joints if j.role == "gripper"
)

PANDA_MOBILE_JOINT_NAMES: list[str] = [
    *PANDA_MOBILE_BASE_JOINT_NAMES,
    *_PANDA_MOBILE_ARM_JOINT_NAMES,
    _PANDA_MOBILE_GRIPPER_JOINT_NAME,
]
"""Full 11-DoF joint order: base (3) + arm (7) + gripper (1).
Matches ``robots/panda_mobile/robot.yaml`` after ADR-0028a."""

# `BODY_TWIST` is the canonical 6-vec velocity command (linear xyz +
# angular xyz; width ``openral_core.BODY_TWIST_DIM``). The planar base
# only honours indices 0, 1, 5 (vx, vy, wz); other indices must be zero.

# Tolerance for "must be zero" non-planar twist components.
_PLANAR_TWIST_EPS = 1e-9


class PandaMobileHAL:
    """In-process digital-twin HAL for the panda_mobile embodiment.

    Maintains 10-DoF qpos state in memory. Routing per
    :attr:`Action.control_mode`:

    * :attr:`ControlMode.BODY_TWIST` — Euler-integrates base pose using
      the planar components of the twist (linear x, linear y,
      angular z). Each ``send_action`` call advances by
      ``dt_s`` seconds (default ``0.05`` — 20 Hz nav control rate).
    * :attr:`ControlMode.JOINT_POSITION` — sets the seven arm joints
      directly when the action carries seven targets; sets all ten
      slots when ten targets are supplied (base position + arm).

    Args:
        initial_pose: Optional override of the start state. Length
            must be either ``3`` (base only) or ``10`` (base + arm).
            Defaults to all-zero.
        dt_s: Integration timestep for body-twist commands. Defaults
            to 0.05 s (20 Hz). The Nav2 controller's default
            ``cmd_vel`` rate.
    """

    def __init__(
        self,
        *,
        initial_pose: list[float] | None = None,
        dt_s: float = 0.05,
    ) -> None:
        """Latch initial 10-DoF state. No I/O until :meth:`connect`."""
        if initial_pose is None:
            self._qpos: list[float] = [0.0] * len(PANDA_MOBILE_JOINT_NAMES)
        elif len(initial_pose) == len(PANDA_MOBILE_BASE_JOINT_NAMES):
            self._qpos = [*initial_pose, *([0.0] * len(_PANDA_MOBILE_ARM_JOINT_NAMES))]
        elif len(initial_pose) == len(PANDA_MOBILE_JOINT_NAMES):
            self._qpos = list(initial_pose)
        else:
            raise ROSConfigError(
                f"PandaMobileHAL.initial_pose must have length 3 (base only) "
                f"or 10 (base+arm); got {len(initial_pose)}."
            )
        self._dt_s = float(dt_s)
        self._connected: bool = False
        self._estop_latched: bool = False
        # Last commanded base body twist (vx, vy, vz, wx, wy, wz) in the
        # base_link frame — published as the /odom twist by the lifecycle
        # node. The base integrates this exactly, so it is the velocity.
        # Zeroed on any non-BODY_TWIST action.
        self._last_body_twist: tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # `HALLifecycleNodeBase._publish_joint_state` reads
        # `self._hal.description` for OTel span attributes + per-joint
        # limit population. Bind the canonical RobotDescription so
        # downstream observability surfaces the right names / limits
        # without a separate registry lookup.
        self.description: RobotDescription = PANDA_MOBILE_DESCRIPTION

    # ── HAL Protocol surface ────────────────────────────────────────────

    def connect(self) -> None:
        """Idempotent — flips the connected flag."""
        self._connected = True

    def disconnect(self) -> None:
        """Idempotent — clears the connected flag."""
        self._connected = False

    def read_state(self) -> JointState:
        """Return a fresh 10-DoF :class:`JointState` snapshot."""
        if not self._connected:
            raise ROSConfigError("PandaMobileHAL.read_state called before connect().")
        import time  # noqa: PLC0415

        return JointState(
            name=list(PANDA_MOBILE_JOINT_NAMES),
            position=list(self._qpos),
            velocity=[0.0] * len(PANDA_MOBILE_JOINT_NAMES),
            effort=[0.0] * len(PANDA_MOBILE_JOINT_NAMES),
            stamp_ns=time.time_ns(),
        )

    def send_action(self, action: Action) -> None:
        """Apply the action to the in-memory state. Branches on control_mode.

        ADR-0028c — accepts the four surfaces the slot dispatcher
        emits: ``JOINT_POSITION``, ``BODY_TWIST``, ``CARTESIAN_DELTA``,
        ``GRIPPER_POSITION``. Each reads its mode-specific payload
        from the matching :class:`Action` field (joint_targets,
        body_twist, cartesian_delta, gripper) — NOT joint_targets for
        every mode, which was the pre-0028c lie that conflated all
        surfaces onto one field.

        Raises:
            ROSConfigError: If the action's ``control_mode`` is not in
                the accepted set, or the mode-specific payload field
                is missing / mis-shaped.
        """
        if not self._connected:
            raise ROSConfigError("PandaMobileHAL.send_action called before connect().")
        if self._estop_latched:
            # ADR-0025 / CLAUDE.md §1.1 — don't silently honour actions
            # after an estop; the supervisor reset path must clear the
            # latch explicitly.
            return

        mode = action.control_mode
        # Default: base not velocity-commanded. BODY_TWIST overwrites via
        # _apply_body_twist below; every other mode leaves it zeroed so
        # /odom doesn't report a stale base velocity.
        self._last_body_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if mode is ControlMode.JOINT_POSITION:
            if not action.joint_targets:
                raise ROSConfigError(
                    "PandaMobileHAL.send_action: empty Action.joint_targets for JOINT_POSITION."
                )
            self._apply_joint_position(list(action.joint_targets[0]))
        elif mode is ControlMode.BODY_TWIST:
            if not action.body_twist:
                raise ROSConfigError(
                    "PandaMobileHAL.send_action: empty Action.body_twist for BODY_TWIST."
                )
            self._apply_body_twist(list(action.body_twist[0]))
        elif mode is ControlMode.CARTESIAN_DELTA:
            # ADR-0028c — apply OSC delta to the cached arm joint
            # vector via a Jacobian-free approximation: treat the
            # cartesian delta as an additive bias on the gripper
            # frame's qpos snapshot. Real motion lives in the
            # sim-attached path (robosuite OSC). The digital-twin
            # tracks the delta for dashboard observability and so
            # tests can verify the chunk reached the HAL inbox.
            if not action.cartesian_delta:
                raise ROSConfigError("PandaMobileHAL.send_action: empty Action.cartesian_delta.")
            self._apply_cartesian_delta(list(action.cartesian_delta[0]))
        elif mode is ControlMode.GRIPPER_POSITION:
            if not action.gripper:
                raise ROSConfigError(
                    "PandaMobileHAL.send_action: empty Action.gripper for GRIPPER_POSITION."
                )
            self._apply_gripper_position(float(action.gripper[0]))
        else:
            raise ROSConfigError(
                f"PandaMobileHAL.send_action: unsupported control_mode "
                f"{mode!r}; expected JOINT_POSITION / BODY_TWIST / "
                f"CARTESIAN_DELTA / GRIPPER_POSITION (ADR-0028c)."
            )

    def estop(self) -> None:
        """Latch the estop flag. Subsequent ``send_action`` calls no-op.

        The latch can only be cleared by calling :meth:`reset_estop`,
        mirroring the supervisor's recovery contract: estops never
        auto-clear.
        """
        self._estop_latched = True

    # ── Convenience APIs ────────────────────────────────────────────────

    def reset_estop(self) -> None:
        """Clear the estop latch. Caller asserts the cause has been resolved."""
        self._estop_latched = False

    @property
    def estop_latched(self) -> bool:
        """``True`` while the estop latch is set."""
        return self._estop_latched

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """Current base ``(x, y, yaw)`` for tests / odom publishers."""
        return (self._qpos[0], self._qpos[1], self._qpos[2])

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """Last commanded base body twist ``(vx, vy, vz, wx, wy, wz)``.

        In the ``base_link`` frame — published as the ``/odom`` twist.
        Zeroed once a non-BODY_TWIST action is sent.
        """
        return self._last_body_twist

    # ── Internals ───────────────────────────────────────────────────────

    def _apply_body_twist(self, row: list[float]) -> None:
        """Euler-integrate the planar components of a 6-vec twist.

        The body-frame twist `(vx, vy, wz)` rotates into the world
        frame by the current ``base_yaw`` before integration so the
        operator's "forward" stays the robot's forward as it turns.
        """
        if len(row) != BODY_TWIST_DIM:
            raise ROSConfigError(
                f"PandaMobileHAL: BODY_TWIST action expects {BODY_TWIST_DIM} "
                f"floats per row (vx, vy, vz, wx, wy, wz); got {len(row)}."
            )
        # Indices not actuated by a planar holonomic base.
        if any(abs(row[i]) > _PLANAR_TWIST_EPS for i in (2, 3, 4)):
            raise ROSConfigError(
                "PandaMobileHAL: BODY_TWIST row carries non-zero linear-z / "
                "angular-x / angular-y components; the panda_mobile base is "
                "holonomic planar — only vx, vy, wz are actuated."
            )
        vx_body, vy_body, _vz, _wx, _wy, wz = row
        # Latch the commanded twist for the /odom publisher (base_link frame).
        self._last_body_twist = (vx_body, vy_body, 0.0, 0.0, 0.0, wz)
        import math  # noqa: PLC0415  # reason: stdlib defer

        yaw = self._qpos[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        # Rotate body-frame velocity into world frame.
        vx_world = cy * vx_body - sy * vy_body
        vy_world = sy * vx_body + cy * vy_body

        self._qpos[0] += vx_world * self._dt_s
        self._qpos[1] += vy_world * self._dt_s
        self._qpos[2] += wz * self._dt_s
        # Wrap yaw to [-π, π] so long sessions don't accumulate drift.
        self._qpos[2] = (self._qpos[2] + math.pi) % (2.0 * math.pi) - math.pi

    def _apply_joint_position(self, row: list[float]) -> None:
        """Set arm (or arm+base, or arm+base+gripper) joints to absolute targets.

        Three accepted widths:

        * ``len(row) == 7`` — arm-only; the seven joints map onto
          ``panda_joint1..7``, the base + gripper stay where they were.
        * ``len(row) == 10`` — base (3) + arm (7); the gripper stays
          where it was. Pre-ADR-0028a state-replay shape preserved
          for legacy callers / MoveIt trajectory replay.
        * ``len(row) == 11`` — full chain: base (3) + arm (7) +
          gripper (1). The post-ADR-0028a canonical width matching
          ``robots/panda_mobile/robot.yaml``.
        """
        n_arm = len(_PANDA_MOBILE_ARM_JOINT_NAMES)
        n_base = len(PANDA_MOBILE_BASE_JOINT_NAMES)
        n_all = len(PANDA_MOBILE_JOINT_NAMES)
        if len(row) == n_arm:
            for i, v in enumerate(row):
                self._qpos[n_base + i] = float(v)
        elif len(row) == n_base + n_arm or len(row) == n_all:
            for i, v in enumerate(row):
                self._qpos[i] = float(v)
        else:
            raise ROSConfigError(
                f"PandaMobileHAL: JOINT_POSITION action expects {n_arm} "
                f"(arm-only), {n_base + n_arm} (base+arm), or {n_all} "
                f"(base+arm+gripper) floats per row; got {len(row)}."
            )

    def _apply_cartesian_delta(self, row: list[float]) -> None:
        """Track the OSC delta for dashboard observability (ADR-0028c).

        The digital-twin HAL has no Jacobian / kinematic chain to
        translate ``[dx, dy, dz, drx, dry, drz]`` into joint motion
        — real motion lives in the sim-attached path
        (:class:`openral_hal.sim_attached.SimAttachedHAL` →
        robosuite OSC). Here we just stamp the latest commanded
        delta onto ``self._last_cartesian_delta`` so the lifecycle
        node's diagnostics + the dashboard's command-vs-reality
        overlay can read it; ``_qpos`` is unchanged (motion =
        zero in the digital twin).

        Raises:
            ROSConfigError: if ``row`` is not a 6-vec.
        """
        cartesian_dim = 6
        if len(row) != cartesian_dim:
            raise ROSConfigError(
                f"PandaMobileHAL: CARTESIAN_DELTA expects {cartesian_dim} "
                f"floats per row (dx, dy, dz, drx, dry, drz); got {len(row)}."
            )
        self._last_cartesian_delta: tuple[float, ...] = tuple(float(v) for v in row)

    def _apply_gripper_position(self, width: float) -> None:
        """Set the gripper qpos slot to ``width`` (ADR-0028c).

        The slot lives at ``len(PANDA_MOBILE_JOINT_NAMES) - 1`` —
        index 10 today (3 base + 7 arm + 1 gripper). Mirror the
        joint's declared ``position_limits`` from
        ``robots/panda_mobile/robot.yaml`` (``[0.0, 1.0]``); the
        safety supervisor's per-mode envelope (ADR-0028b step 5)
        already clamps via ``gripper_min`` / ``gripper_max`` when
        configured, so the HAL trusts its input.
        """
        gripper_idx = len(PANDA_MOBILE_JOINT_NAMES) - 1
        self._qpos[gripper_idx] = float(width)
