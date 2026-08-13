"""Gripper-effort grasp trigger — when to ask perception "what is in the jaws?".

:mod:`~openral_hal._vision_attachment_evidence` is *told* that a grasp happened;
it does not decide it. This module is that decision, and only that: a debounced
state machine over the gripper joint's effort channel in the typed
:class:`~openral_core.JointState` the HAL already reads every tick. It emits
:class:`GraspEvent` s — ATTACH, REGRASP, DETACH — which the vision attachment
bridge turns into ``SegmentInView`` calls.

Pure: no ROS, no numpy, no I/O, no clock. The caller supplies the state
snapshots, so the whole state machine is unit-testable against real manifests.

Why effort, and why N consecutive ticks
---------------------------------------

A commanded jaw *position* says nothing about whether anything is between the
jaws — a closed-on-nothing gripper reaches the same position as one holding a
thin object. Effort is what distinguishes them: closing onto an object stalls
the servo and the effort readback rises. A single tick over the threshold is not
enough, though: effort spikes transiently on every jaw acceleration, and a
one-tick spike would fire a segmentation (and a payload attachment) for a grasp
that never happened. :attr:`GraspTriggerConfig.consecutive_ticks` is the
debounce, and the thresholds carry hysteresis so a payload held right at the
boundary does not chatter attach/detach.

.. warning::

   **The SO-101 (Feetech STS3215) effort readback is UNVERIFIED as a grasp
   signal.** Nothing in the design work behind this module measured whether
   those servos report a usable present-load value through
   ``lerobot``'s Feetech bus, or whether the channel is quantised, sign-flipped,
   latched, or simply zero. Until that is measured, this trigger MUST NOT be
   trusted as the sole attach signal on SO-101 hardware.

   This is deliberately left as a falsifiable measurement rather than a guessed
   constant. :func:`assess_effort_readback` is the test hook: record a real
   open → close-on-object → hold → open sequence on the arm, feed the
   :class:`~openral_core.JointState` snapshots in, and read the verdict. A
   channel that is absent, all-zero, or constant across a sequence that
   *physically* loaded the jaws is unusable, and the attach trigger then needs a
   different signal entirely (tactile, current sense, or an explicit skill-level
   attach command) — which would change this module's interface, so it is the
   cheapest thing to falsify now and the most expensive to discover late.

Every threshold below is a **calibration point, not a measured constant**
(CLAUDE.md §1.2). They are expressed as fractions of the gripper joint's own
``effort_limit`` from the manifest, so at least the scale comes from the robot
rather than from a number typed into this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import JointState, RobotDescription

__all__ = [
    "EffortReadbackHealth",
    "GraspEvent",
    "GraspTriggerConfig",
    "GripperEffortTrigger",
    "assess_effort_readback",
]


class GraspEvent(str, Enum):
    """A transition the vision attachment producer must react to.

    Attributes:
        ATTACH: The jaws went from empty to loaded. Segment, gate, attach.
        REGRASP: The jaws stayed loaded but moved materially — the payload was
            re-seated, so the fitted geometry is stale. Segment again.
        DETACH: The jaws went from loaded to empty. Publish an empty attachment
            set; no segmentation needed.

    Example:
        >>> GraspEvent.ATTACH.value
        'attach'
    """

    ATTACH = "attach"
    REGRASP = "regrasp"
    DETACH = "detach"


@dataclass(frozen=True)
class GraspTriggerConfig:
    """Debounce and threshold configuration for :class:`GripperEffortTrigger`.

    .. warning::

       **None of these values is benchmarked.** Each is a conservative starting
       point and a calibration point that must be tuned per robot — and on
       SO-101 specifically, the underlying effort channel itself is unverified
       (see the module docstring and :func:`assess_effort_readback`).

    Attributes:
        attach_effort_fraction: Fraction of the gripper joint's manifest
            ``effort_limit`` at or above which the jaws count as loaded.
            *Calibration point*, ``0.30``.
        release_effort_fraction: Fraction at or below which a loaded gripper
            counts as released. Strictly below
            :attr:`attach_effort_fraction` — the gap is the hysteresis band
            that stops a payload held near the threshold from chattering
            attach/detach every tick. *Calibration point*, ``0.10``.
        consecutive_ticks: How many consecutive samples must agree before a
            transition is emitted. *Calibration point*, ``3``. At the SO-101's
            ~30 Hz read rate that is ~100 ms of agreement, which is the same
            order as the deferred-ack barrier this feeds.
        regrasp_position_delta: Jaw-channel movement (in the gripper joint's own
            units — a normalised [0, 1] fraction on SO-101, radians elsewhere)
            away from the position held at attach that counts as a re-seat while
            still loaded. *Calibration point*, ``0.15``.
    """

    attach_effort_fraction: float = 0.30
    release_effort_fraction: float = 0.10
    consecutive_ticks: int = 3
    regrasp_position_delta: float = 0.15


@dataclass(frozen=True)
class EffortReadbackHealth:
    """Verdict on whether a recorded effort channel can drive a grasp trigger.

    The output of :func:`assess_effort_readback` — the falsification hook for
    the open question in this module's docstring.

    Attributes:
        samples: Snapshots inspected.
        present_samples: Snapshots that carried an effort value for the joint at
            all. A driver that publishes an empty ``effort`` list scores ``0``.
        nonzero_samples: Snapshots whose effort was non-zero.
        distinct_values: Distinct effort readings seen. ``1`` over a sequence
            that physically loaded and unloaded the jaws means the channel is
            latched or synthesised, not measured.
        span: ``max - min`` of the observed efforts.
        usable: Whether the channel varied enough to threshold on.
        reason: Why not, when ``usable`` is False; empty otherwise.
    """

    samples: int
    present_samples: int
    nonzero_samples: int
    distinct_values: int
    span: float
    usable: bool
    reason: str


# A recorded open→close-on-object→open sequence whose effort span is below this
# is flat: nothing a threshold could separate. Expressed as a fraction of the
# joint's ``effort_limit`` so it scales with the robot. CALIBRATION POINT.
_MIN_HEALTHY_SPAN_FRACTION = 0.05


def gripper_joint(description: RobotDescription) -> object:
    """Return the single ``role: "gripper"`` joint spec of a manifest.

    Args:
        description: The robot manifest.

    Returns:
        The gripper :class:`~openral_core.JointSpec`.

    Raises:
        ROSConfigError: If the manifest declares no gripper-role joint, or more
            than one — this module drives a single parallel jaw, and guessing
            which of several is "the" gripper is exactly the kind of implicit
            choice that must not be buried in a safety-adjacent trigger.
    """
    joints = [joint for joint in description.joints if joint.role == "gripper"]
    if len(joints) != 1:
        raise ROSConfigError(
            f"grasp trigger needs exactly one role='gripper' joint on "
            f"{description.name!r}, found {len(joints)}."
        )
    return joints[0]


def assess_effort_readback(
    states: Sequence[JointState],
    *,
    description: RobotDescription,
) -> EffortReadbackHealth:
    """Judge a recorded effort trace as a grasp signal — the SO-101 test hook.

    Feed this the :class:`~openral_core.JointState` snapshots captured while a
    real arm opens, closes onto an object, holds, and releases. If the verdict
    is not ``usable``, :class:`GripperEffortTrigger` cannot be trusted on that
    robot and the attach trigger needs a different signal — see the module
    docstring. This function measures; it does not assume.

    Args:
        states: Recorded snapshots, in order.
        description: The robot manifest, for the gripper joint and its
            ``effort_limit`` (the scale the span is judged against).

    Returns:
        The :class:`EffortReadbackHealth` verdict.

    Raises:
        ROSConfigError: If the manifest has no single gripper joint, or that
            joint declares no ``effort_limit`` (there is then no scale to judge
            the span against, and a fraction-based threshold is undefined).

    Example:
        >>> from openral_core import JointState, RobotDescription
        >>> d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> flat = [
        ...     JointState(name=["gripper"], position=[0.5], effort=[0.0], stamp_ns=i)
        ...     for i in range(4)
        ... ]
        >>> assess_effort_readback(flat, description=d).usable
        False
    """
    spec = gripper_joint(description)
    name = str(spec.name)  # type: ignore[attr-defined]  # reason: JointSpec is typed only under TYPE_CHECKING here
    limit = spec.effort_limit  # type: ignore[attr-defined]  # reason: as above
    if limit is None or limit <= 0.0:
        raise ROSConfigError(
            f"grasp trigger needs a positive effort_limit on joint {name!r} of "
            f"{description.name!r} to scale its thresholds; the manifest declares {limit!r}."
        )
    efforts = [value for state in states if (value := _effort_of(state, name)) is not None]
    span = (max(efforts) - min(efforts)) if efforts else 0.0
    distinct = len(set(efforts))
    nonzero = sum(1 for value in efforts if value != 0.0)
    reason = ""
    if not efforts:
        reason = "no effort value for the gripper joint in any sample (empty effort channel)"
    elif nonzero == 0:
        reason = "effort readback is identically zero across the whole trace"
    elif span < _MIN_HEALTHY_SPAN_FRACTION * float(limit):
        reason = (
            f"effort span {span:.4f} is under {_MIN_HEALTHY_SPAN_FRACTION:.0%} of the joint's "
            f"effort_limit {float(limit):.4f} — nothing a threshold could separate"
        )
    return EffortReadbackHealth(
        samples=len(states),
        present_samples=len(efforts),
        nonzero_samples=nonzero,
        distinct_values=distinct,
        span=span,
        usable=not reason,
        reason=reason,
    )


def _effort_of(state: JointState, joint_name: str) -> float | None:
    """Absolute effort of one joint in a snapshot, or ``None`` when absent.

    Absolute because sign convention is per-driver and a grasp loads the jaw in
    whichever direction that driver calls positive; the trigger only cares about
    magnitude.
    """
    try:
        index = state.name.index(joint_name)
    except ValueError:
        return None
    if index >= len(state.effort):
        return None
    return abs(float(state.effort[index]))


class GripperEffortTrigger:
    """Debounced attach / regrasp / detach events from gripper effort.

    Fed one :class:`~openral_core.JointState` per HAL read tick; returns a
    :class:`GraspEvent` on the tick a transition is confirmed, and ``None``
    otherwise. Confirmation needs :attr:`GraspTriggerConfig.consecutive_ticks`
    agreeing samples, so a transient effort spike never fires an attachment.

    Ticks where the gripper carries **no** effort value at all are not silently
    ignored: they are counted in :attr:`missing_effort_ticks` so the caller can
    surface a driver that publishes an empty effort channel instead of quietly
    never attaching (CLAUDE.md §1.4).

    Args:
        description: The robot manifest — supplies the gripper joint and the
            ``effort_limit`` the fractional thresholds scale against.
        config: Thresholds and debounce. Every field is a calibration point.

    Raises:
        ROSConfigError: If the manifest has no single gripper joint, that joint
            declares no positive ``effort_limit``, or the config's hysteresis
            band is inverted / its debounce is under one tick.

    Example:
        >>> from openral_core import JointState, RobotDescription
        >>> d = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
        >>> t = GripperEffortTrigger(d, config=GraspTriggerConfig(consecutive_ticks=2))
        >>> loaded = JointState(name=["gripper"], position=[0.2], effort=[3.0], stamp_ns=0)
        >>> t.update(loaded) is None
        True
        >>> t.update(loaded)
        <GraspEvent.ATTACH: 'attach'>
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        config: GraspTriggerConfig | None = None,
    ) -> None:
        """Resolve the gripper joint and turn fractional thresholds into absolutes."""
        self._config = config or GraspTriggerConfig()
        if self._config.consecutive_ticks < 1:
            raise ROSConfigError(
                f"GraspTriggerConfig.consecutive_ticks must be >= 1, "
                f"got {self._config.consecutive_ticks}."
            )
        if self._config.release_effort_fraction >= self._config.attach_effort_fraction:
            raise ROSConfigError(
                "GraspTriggerConfig needs release_effort_fraction < attach_effort_fraction "
                f"(got {self._config.release_effort_fraction} >= "
                f"{self._config.attach_effort_fraction}); without the gap there is no "
                "hysteresis and a payload held at the threshold chatters."
            )
        spec = gripper_joint(description)
        limit = spec.effort_limit  # type: ignore[attr-defined]  # reason: JointSpec typed under TYPE_CHECKING
        if limit is None or limit <= 0.0:
            raise ROSConfigError(
                f"GripperEffortTrigger needs a positive effort_limit on joint "
                f"{spec.name!r} of {description.name!r}; the manifest declares {limit!r}."  # type: ignore[attr-defined]
            )
        self._joint_name = str(spec.name)  # type: ignore[attr-defined]
        self._attach_effort = self._config.attach_effort_fraction * float(limit)
        self._release_effort = self._config.release_effort_fraction * float(limit)
        self._loaded = False
        self._streak = 0
        self._streak_kind: GraspEvent | None = None
        self._attach_position: float | None = None
        self.missing_effort_ticks = 0

    @property
    def joint_name(self) -> str:
        """Name of the gripper joint this trigger reads."""
        return self._joint_name

    @property
    def attached(self) -> bool:
        """Whether the trigger currently believes the jaws are loaded."""
        return self._loaded

    @property
    def thresholds_n(self) -> tuple[float, float]:
        """The absolute ``(attach, release)`` effort thresholds actually in force."""
        return (self._attach_effort, self._release_effort)

    def update(self, state: JointState) -> GraspEvent | None:
        """Fold one state snapshot into the state machine.

        Args:
            state: The tick's joint state, as read from the HAL.

        Returns:
            The confirmed :class:`GraspEvent`, or ``None`` when this tick
            confirms nothing.
        """
        effort = _effort_of(state, self._joint_name)
        if effort is None:
            # A driver with no effort channel can never trigger. Counted, not
            # swallowed, so the caller can say so out loud.
            self.missing_effort_ticks += 1
            self._streak = 0
            self._streak_kind = None
            return None
        candidate = self._classify(state, effort)
        if candidate is None or candidate is not self._streak_kind:
            self._streak_kind = candidate
            self._streak = 1 if candidate is not None else 0
        else:
            self._streak += 1
        if candidate is None or self._streak < self._config.consecutive_ticks:
            return None
        self._commit(candidate, state)
        return candidate

    def _classify(self, state: JointState, effort: float) -> GraspEvent | None:
        """Which transition this single sample argues for, before debouncing."""
        if not self._loaded:
            return GraspEvent.ATTACH if effort >= self._attach_effort else None
        if effort <= self._release_effort:
            return GraspEvent.DETACH
        position = self._position_of(state)
        if (
            position is not None
            and self._attach_position is not None
            and abs(position - self._attach_position) > self._config.regrasp_position_delta
        ):
            return GraspEvent.REGRASP
        return None

    def _commit(self, event: GraspEvent, state: JointState) -> None:
        """Apply a confirmed transition and reset the debounce."""
        self._loaded = event is not GraspEvent.DETACH
        self._attach_position = None if event is GraspEvent.DETACH else self._position_of(state)
        self._streak = 0
        self._streak_kind = None

    def _position_of(self, state: JointState) -> float | None:
        """Gripper channel position in this snapshot, or ``None`` when absent."""
        try:
            index = state.name.index(self._joint_name)
        except ValueError:
            return None
        if index >= len(state.position):
            return None
        return float(state.position[index])
