"""Compose one whole-robot command from a tick's slot-dispatched actions.

ADR-0102. ``rskill_runner_node._dispatch_slots`` splits a policy's flat action
vector into **one typed** ``Action`` **per
non-discard slot** — four for the OpenArm v2 bimanual contract (left
arm, left gripper, right arm, right gripper) — each carrying the tick's shared
``tick_index`` / ``tick_group_size`` and each published separately through the
safety kernel.

A HAL that commands its robot as one vector therefore has to reassemble the
tick. Two properties matter, and both are why this is a shared helper rather
than an inline branch:

* **Atomicity.** A group whose slot the safety kernel rejected must never
  commit its surviving slots — that would leave one arm driven from the new
  chunk while the other holds an older setpoint. Mirrors
  ``openral_hal.sim_attached.SimAttachedHAL._stage_action_group``, which
  established this contract for the simulator side.
* **Addressing by name, not position.** ``JOINT_*`` slots are zero-padded to
  full dof (the C++ kernel enforces ``chunk.n_dof == envelope.n_dof``), so a
  padded row cannot say which joints it owns — ``0.0`` is a legal joint
  target. ADR-0102 carries the slot's ``joint_names`` onto the action and
  across the wire for exactly this; ``ee_name`` already did the same job for
  gripper slots. Ordering is *not* used: reordering a manifest's ``slots:``
  block must not silently reroute bytes to the wrong joints.
"""

from __future__ import annotations

from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode

__all__ = [
    "GRIPPER_MODES",
    "SlotGroupStager",
    "compose_slot_group",
    "compose_slot_group_action",
    "refuse_stale_tick",
]

GRIPPER_MODES = (ControlMode.GRIPPER_POSITION, ControlMode.GRIPPER_BINARY)


def compose_slot_group(
    actions: list[Action],
    joint_names: list[str],
) -> list[float]:
    """Merge one tick's slot actions into a single full-dof target row.

    Args:
        actions: Every slot action of one inference tick, in any order.
        joint_names: The robot's actuated joint names, in HAL/action order.

    Returns:
        One target value per entry of ``joint_names``.

    Raises:
        ROSConfigError: A slot carries no payload, addresses an unknown joint
            or end-effector, is a mode this composer cannot place, or the
            group leaves a joint uncommanded / commands one twice.

    Example:
        >>> from openral_core.schemas import Action, ControlMode
        >>> arm = Action(
        ...     control_mode=ControlMode.JOINT_POSITION,
        ...     horizon=1,
        ...     joint_targets=[[0.5, 0.0]],
        ...     joint_names=["j1"],
        ... )
        >>> grip = Action(
        ...     control_mode=ControlMode.GRIPPER_POSITION,
        ...     horizon=1,
        ...     gripper=[0.3],
        ...     ee_name="grip",
        ... )
        >>> compose_slot_group([arm, grip], ["j1", "grip"])
        [0.5, 0.3]
    """
    index_of = {name: i for i, name in enumerate(joint_names)}
    targets: list[float | None] = [None] * len(joint_names)

    def _claim(idx: int, value: float, who: str) -> None:
        if targets[idx] is not None:
            raise ROSConfigError(
                f"slot group commands joint {joint_names[idx]!r} twice "
                f"(second writer: {who}). Two slots overlap — the rSkill's "
                "action_contract ranges are supposed to be disjoint."
            )
        targets[idx] = value

    for action in actions:
        mode = action.control_mode
        if mode in GRIPPER_MODES:
            if not action.gripper:
                raise ROSConfigError(
                    f"slot group has a {mode.value} action with an empty Action.gripper payload."
                )
            ee = action.ee_name
            if not ee:
                raise ROSConfigError(
                    f"slot group has a {mode.value} action with no ee_name; "
                    "a gripper slot must name the end-effector it drives."
                )
            if ee not in index_of:
                raise ROSConfigError(
                    f"slot group addresses end-effector {ee!r}, which is not one "
                    f"of this robot's joints {joint_names}."
                )
            _claim(index_of[ee], float(action.gripper[-1]), f"gripper {ee!r}")
            continue
        if mode is not ControlMode.JOINT_POSITION:
            raise ROSConfigError(
                f"slot group carries control_mode {mode.value!r}, which this "
                "composer cannot place into a joint-position command."
            )
        if not action.joint_targets:
            raise ROSConfigError("slot group has a joint_position action with no joint_targets.")
        names = action.joint_names
        if not names:
            raise ROSConfigError(
                "slot group has a joint_position action with no joint_names "
                "(ADR-0102). A padded sub-slot chunk carries zeros at the joints "
                "it does not own, so without the names it cannot be placed — and "
                "guessing from the zeros is wrong, 0.0 being a legal target. "
                "Declare joint_names on the rSkill's action_contract slots."
            )
        row = list(action.joint_targets[-1])
        for name in names:
            idx = index_of.get(name)
            if idx is None:
                raise ROSConfigError(
                    f"slot group addresses joint {name!r}, which is not one of "
                    f"this robot's joints {joint_names}."
                )
            if idx >= len(row):
                raise ROSConfigError(
                    f"slot group's joint_position row has {len(row)} values but "
                    f"joint {name!r} sits at index {idx}; the chunk was not padded "
                    "to full dof."
                )
            _claim(idx, float(row[idx]), f"joint {name!r}")

    missing = [joint_names[i] for i, v in enumerate(targets) if v is None]
    if missing:
        raise ROSConfigError(
            f"slot group leaves {len(missing)} joint(s) uncommanded: {missing}. "
            "Publishing a partial command would drive the covered joints and "
            "leave the rest on a stale setpoint, so the whole tick is refused."
        )
    return [float(v) for v in targets]  # type: ignore[arg-type]  # reason: `missing` proved no None remains


def compose_slot_group_action(group: list[Action], joint_names: list[str]) -> Action:
    """Compose one tick's slot actions into a single full-dof ``JOINT_POSITION`` action.

    Args:
        group: Every slot action of one inference tick.
        joint_names: The robot's actuated joint names, in HAL/action order.

    Returns:
        A one-step ``JOINT_POSITION`` action carrying the first slot's stamp
        and confidence.

    Raises:
        ROSConfigError: The group is unplaceable (see ``compose_slot_group``).
    """
    first = group[0]
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[compose_slot_group(group, joint_names)],
        stamp_ns=first.stamp_ns,
        confidence=first.confidence,
    )


def refuse_stale_tick(tick: int, last_committed: int) -> int:
    """Refuse a replayed slot tick; return the watermark to stage against.

    A staging buffer only guards the tick in flight; a whole group of an
    already-committed tick would otherwise replay stale targets, and the
    lifecycle node's monotonic acknowledgement would hide it. Runner ticks are
    process-monotonic from 1, so a tick at or below the watermark is a replay
    and is refused, with ONE exception: ``tick == 1`` while the watermark is
    above 1 is a restarted runner's fresh numbering (the HAL stayed up). It is
    adopted by returning a reset watermark (``0``); refusing it would wedge the
    HAL for the rest of its node's life. A replay of tick 1 itself (watermark
    1) is still refused. The rule is hazard-log Entry 036's.

    Args:
        tick: The slot's ``Action.tick_index``.
        last_committed: The last tick the HAL applied (``0`` = none).

    Returns:
        ``last_committed`` unchanged, or ``0`` when a restart was adopted.

    Raises:
        ROSRuntimeError: ``0 < tick <= last_committed``, except the adoption.

    Example:
        >>> from openral_core.exceptions import ROSRuntimeError
        >>> refuse_stale_tick(3, 2)
        2
        >>> refuse_stale_tick(1, 7)  # a restarted runner
        0
        >>> for tick, last in ((2, 2), (5, 7), (1, 1)):
        ...     try:
        ...         refuse_stale_tick(tick, last)
        ...     except ROSRuntimeError:
        ...         print("refused")
        refused
        refused
        refused
    """
    if tick == 1 and last_committed > 1:
        return 0
    if 0 < tick <= last_committed:
        raise ROSRuntimeError(
            f"stale slot group: tick {tick} is not after the last committed "
            f"tick {last_committed}; refusing to replay it (only tick 1, a "
            "restarted runner, may renumber)."
        )
    return last_committed


class SlotGroupStager:
    """Buffer one inference tick's slot actions until every slot has arrived.

    Holds at most one tick. A tick change mid-group means the safety kernel
    dropped or rejected a slot, which is a hard error rather than a silent
    partial commit (see the module docstring). The error costs exactly the one
    incomplete tick: the slot that exposed it opens the new tick, so a single
    rejected slot does not wedge the HAL for the rest of the run.

    It also holds the **committed watermark**: the HAL calls ``commit`` once a
    released group has actually been applied, and ``stage`` then refuses any
    slot of a tick at or below it (``refuse_stale_tick``), so a replayed group
    can never re-command the robot; tick 1 above a watermark of 1 is a
    restarted runner and resets the watermark instead. ``discard`` (E-stop)
    keeps the watermark, ``reset`` (disconnect) clears it. ``last_committed_tick`` is what the HAL
    lifecycle node acknowledges on ``/openral/action_applied``.

    Example:
        >>> stager = SlotGroupStager()
        >>> stager.pending, stager.last_committed_tick
        (0, 0)
    """

    def __init__(self) -> None:
        """Start with no tick staged and nothing committed."""
        self._actions: list[Action] = []
        self._tick: int | None = None
        self._last_committed = 0

    @property
    def pending(self) -> int:
        """Number of slots staged for the tick in flight."""
        return len(self._actions)

    @property
    def last_committed_tick(self) -> int:
        """Tick of the last group the HAL applied (``0`` = none since ``reset``)."""
        return self._last_committed

    def commit(self, group: list[Action]) -> None:
        """Record that a group ``stage`` released was applied to the robot.

        Call only after the apply succeeded: the lifecycle node acknowledges
        ``last_committed_tick``, so committing a group that failed would ack a
        tick the robot never received.
        """
        self._last_committed = int(group[0].tick_index)

    def discard(self) -> None:
        """Drop the half-staged tick but keep the committed watermark (estop).

        A stop is not a renumbering — the runner's ticks keep increasing — so
        a pre-stop tick replayed after the stop must still be refused.
        """
        self._actions.clear()
        self._tick = None

    def reset(self) -> None:
        """Drop the staged tick AND the watermark (disconnect: numbering restarts)."""
        self.discard()
        self._last_committed = 0

    def stage(self, action: Action) -> list[Action] | None:
        """Add one slot; return the whole group once complete, else ``None``.

        Args:
            action: One slot action carrying ``tick_index`` /
                ``tick_group_size``.

        Returns:
            Every action of the tick when the last slot lands, else ``None``.

        Raises:
            ROSConfigError: The action carries no usable tick index.
            ROSRuntimeError: The tick is at or below ``last_committed_tick``
                (a replay; nothing is staged — except tick 1 above a watermark
                of 1, a restarted runner, which resets the watermark), the
                staged tick changed before completing, or the group overran
                its declared size — the last two mean a slot was lost.
        """
        group_size = int(action.tick_group_size)
        tick = int(action.tick_index)
        if tick <= 0:
            raise ROSConfigError(
                "slot-group staging requires Action.tick_index > 0; got "
                f"{tick}. The runner sets it on every slot of a multi-slot tick."
            )
        self._last_committed = refuse_stale_tick(tick, self._last_committed)
        if self._tick is not None and tick != self._tick:
            dropped = [a.control_mode.value for a in self._actions]
            staged, expected = len(self._actions), self._tick
            # Discard the abandoned tick but ADOPT the slot that exposed it.
            # Dropping it as well would leave the new tick permanently one slot
            # short, so it could never complete either — and every following
            # tick would lose its own first slot the same way, so one rejected
            # slot would stop the HAL publishing for the rest of the run. The
            # raise reports the tick that was lost; it is not a reason to lose
            # the next one too.
            self._actions.clear()
            self._tick = tick
            self._actions.append(action)
            raise ROSRuntimeError(
                f"incomplete slot group: tick {expected} had {staged}/{group_size} "
                f"slots (modes={dropped}) when tick {tick} started. The safety "
                "supervisor dropped a slot, or the rSkill declared the wrong "
                "group size; the partial tick is discarded rather than committed. "
                f"Tick {tick} is staged from this slot, so the next complete tick "
                "commits normally."
            )
        self._tick = tick
        self._actions.append(action)
        if len(self._actions) < group_size:
            return None
        if len(self._actions) > group_size:
            staged = len(self._actions)
            self.discard()
            raise ROSRuntimeError(
                f"slot group for tick {tick} received {staged} slots but declared {group_size}."
            )
        group = list(self._actions)
        self.discard()
        return group
