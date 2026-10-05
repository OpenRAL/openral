"""Position-stall grasp trigger — when to ask perception "what is in the jaws?".

``_vision_attachment_evidence`` is *told* that a grasp happened; it does
not decide it. This module is that decision, and only that: a debounced
state machine over the gripper joint's commanded target and measured
*position*. It emits ``GraspEvent`` s — ATTACH, REGRASP, DETACH — which the
vision attachment bridge turns into ``SegmentInView`` calls.

Pure: no ROS, no numpy, no I/O, no clock. The caller supplies the state
snapshots and the commanded targets, so the whole state machine is
unit-testable against real manifests.

Why position and not effort: the real OpenArm gripper reports **no effort**
— the vendor ``openarm_ros2`` hardware interface (4e837e1,
``openarm_simple_hardware.cpp`` L268-279) hard-codes effort and velocity to
``0.0`` every tick, so a full list of zeros reaches the HAL and an effort
threshold silently never fires. No in-tree gripper declares a torque sensor
(``JointSpec.has_torque_sensor``), so the effort trigger this replaced had no
robot it could honestly arm on and was removed.

What position *does* say, from real 30 fps OpenArm teleop
recordings: a position-controlled jaw
commanded closed on an object **stalls** 0.18-0.29 rad short of the command
and stays flat to ~1e-3 rad; closed on nothing it reaches within ~0.02 rad;
free-motion steady-state error is 0.006-0.025 rad. So:

* **ATTACH** — the command is near closed, the jaw is settled (its position
  span over the last ``PositionStallConfig.settle_s`` seconds is within the joint's
  ``settle_tolerance``), and it sits further from closed than the command by
  more than ``closed_rest_offset + stall_gap``; agreed for
  ``consecutive_s`` seconds.

Every window is in **seconds of sample time** (``JointState.stamp_ns``), never in
ticks: a tick count tuned at the 30 Hz action rate is 15x shorter on a 500 Hz joint
state stream, where a slowly closing jaw moves less than ``settle_tolerance`` in five
samples and would read as stalled. A sample whose stamp is not later than the last one
by more than ``min_sample_interval_s`` is a repeat — a cached joint state read twice —
and is dropped (counted in ``repeated_samples``), so it can neither settle the jaw nor
extend a debounce. Time alone is not enough either: a stream of late ticks spans a window
on fewer samples than it was tuned for, so each window also needs a minimum **sample
count** (``consecutive_samples`` / ``settle_samples``, the original 3 and 5 ticks), and a
gap between accepted samples longer than ``max_gap_s`` restarts both — a sparse stream
that happens to catch a chattering jaw at the same phase each time is not a settled jaw.
* **DETACH** — while still commanded closed, the gap collapses (the object
  slipped out, the jaw closed onto nothing); or the command opened and the jaw
  opened past the position it held at attach. Both use half the stall gap as
  hysteresis so a jaw held at a threshold does not chatter.
* **REGRASP** — still stalled, but settled at a position materially (half the
  stall gap) away from the one held at attach: re-seated, the fitted geometry
  is stale.

Known limits, each a Safety-WG hazard entry rather than a constant to tune
away: an object thinner than ``stall_gap`` (in jaw angle) does not stall far
enough to read as held (false negative); a jaw obstructed by something other
than the target — the table, the other hand — stalls exactly like a grasp
(false positive). Vision confirmation at ATTACH and the release window bound
what either costs; see ``vision_attachment_bridge``.

The thresholds come from the manifest (``JointSpec.closure_calibration``),
never from this file: a gripper joint with no calibration refuses to arm.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import JointSpec, JointState, RobotDescription

__all__ = [
    "GAP_PERIODS",
    "GraspEvent",
    "PositionStallConfig",
    "PositionStallTrigger",
    "gripper_joint",
    "gripper_joints",
]


#: How many sample periods ``PositionStallConfig.for_rate`` tolerates between accepted samples
#: before a window restarts: three missed samples (one executor stall of a few ticks) are
#: jitter, a fourth is a gap. Calibration point, not a benchmark.
GAP_PERIODS = 4.0


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
class PositionStallConfig:
    """Robot-independent debounce for ``PositionStallTrigger``.

    The physical thresholds are the gripper joint's ``closure_calibration`` in the
    manifest; only time windows live here, in seconds of sample time so they mean
    the same at any joint-state rate. Calibration points, not benchmarks: the
    defaults are the windows the original 30 Hz tick counts spanned (3 and 5
    samples).

    Every window must hold on BOTH its duration and its sample count, and a gap longer than
    ``max_gap_s`` between accepted samples restarts every window.

    Attributes:
        consecutive_s: How long samples must keep agreeing — from the first agreeing
            sample's stamp to the confirming one's — before a transition is emitted.
            ``0.06`` (3 samples at 30 Hz). ``0`` confirms on the first sample.
        settle_s: The window the jaw's position span is measured over to call it
            stationary; the samples kept must cover all of it. ``0.13`` (5 samples at
            30 Hz).
        consecutive_samples: The fewest agreeing samples (the confirming one included)
            that confirm a transition. ``3``.
        settle_samples: The fewest samples the settle window must hold. ``5``.
        max_gap_s: A sample stamped more than this after the previous accepted one
            restarts the settle history and the debounce: what the jaw did in the gap is
            unknown. ``0.1`` (3 periods at 30 Hz).
        min_sample_interval_s: A sample stamped no later than the previous accepted
            one plus this is a repeat and is dropped. ``5e-4``: under the 1 ms period
            of a 1 kHz joint-state stream, above the microseconds of jitter a re-read
            cached sample's reconstructed stamp carries.
    """

    consecutive_s: float = 0.06
    settle_s: float = 0.13
    consecutive_samples: int = 3
    settle_samples: int = 5
    max_gap_s: float = 0.1
    min_sample_interval_s: float = 5e-4

    def validate(self) -> None:
        """Refuse a degenerate window.

        Raises:
            ROSConfigError: A non-finite window, a negative one, a zero ``settle_s``
                (a span needs two samples apart in time), ``consecutive_samples < 1``,
                ``settle_samples < 2``, or a ``max_gap_s`` not above
                ``min_sample_interval_s``.
        """
        windows = (self.consecutive_s, self.settle_s, self.max_gap_s, self.min_sample_interval_s)
        if (
            not all(math.isfinite(w) and w >= 0.0 for w in windows)
            or self.settle_s <= 0.0
            or self.max_gap_s <= self.min_sample_interval_s
            or self.consecutive_samples < 1
            or self.settle_samples < 2  # noqa: PLR2004  # reason: a span needs two samples
        ):
            raise ROSConfigError(
                "PositionStallConfig needs finite consecutive_s >= 0, settle_s > 0, "
                "max_gap_s > min_sample_interval_s >= 0, consecutive_samples >= 1 and "
                f"settle_samples >= 2, got {self!r}."
            )

    @classmethod
    def for_rate(cls, rate_hz: float, *, max_gap_s: float | None = None) -> PositionStallConfig:
        """The windows for a joint-state stream sampled at ``rate_hz``.

        The defaults were tuned on a 30 Hz stream, and only ``max_gap_s`` is rate-bound:
        a fixed 0.1 s is one period at 10 Hz, so every jittered sample read as a gap and
        nothing (trigger nor heartbeat evidence) ever confirmed. It becomes
        ``GAP_PERIODS`` periods, never below the 0.1 s default. ``consecutive_s`` /
        ``settle_s`` stay: their sample-count floors already stretch them to
        ``(count - 1)`` periods at a slow rate, and at a fast one the seconds hold.

        Args:
            rate_hz: The rate the trigger is fed — the HAL node's joint-state rate.
            max_gap_s: Explicit override of the derived gap (an operator calibration).

        Returns:
            A validated config.

        Raises:
            ROSConfigError: A non-finite or non-positive ``rate_hz``, or a degenerate
                override.

        Example:
            >>> [round(PositionStallConfig.for_rate(hz).max_gap_s, 3) for hz in (10, 30, 500)]
            [0.4, 0.133, 0.1]
        """
        if not (math.isfinite(rate_hz) and rate_hz > 0.0):
            raise ROSConfigError(f"PositionStallConfig.for_rate needs rate_hz > 0, got {rate_hz}.")
        default = cls()
        gap = max(default.max_gap_s, GAP_PERIODS / rate_hz) if max_gap_s is None else max_gap_s
        config = dataclasses.replace(default, max_gap_s=gap)
        config.validate()
        return config

    def is_gap(self, stamp_ns: int, last_ns: int | None) -> bool:
        """Whether ``stamp_ns`` follows the last accepted sample by more than ``max_gap_s``.

        Example:
            >>> PositionStallConfig().is_gap(200_000_000, 0)
            True
            >>> PositionStallConfig().is_gap(33_333_333, 0)
            False
        """
        return last_ns is not None and stamp_ns - last_ns > self.max_gap_s * 1e9

    def is_repeat(self, stamp_ns: int, last_ns: int | None) -> bool:
        """Whether a sample stamped ``stamp_ns`` repeats (or predates) the last accepted one.

        Example:
            >>> PositionStallConfig().is_repeat(1_000_100, 1_000_000)
            True
            >>> PositionStallConfig().is_repeat(2_000_000, 1_000_000)
            False
        """
        return last_ns is not None and stamp_ns - last_ns <= self.min_sample_interval_s * 1e9


def gripper_joints(description: RobotDescription) -> list[JointSpec]:
    """Return every ``role: "gripper"`` joint spec of a manifest, in manifest order.

    A bimanual robot has one per hand; each one drives its own trigger.

    Args:
        description: The robot manifest.

    Returns:
        The gripper ``JointSpec`` s.

    Raises:
        ROSConfigError: If the manifest declares no gripper-role joint.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> [joint.name for joint in gripper_joints(d)]
        ['left_gripper', 'right_gripper']
    """
    joints = [joint for joint in description.joints if joint.role == "gripper"]
    if not joints:
        raise ROSConfigError(f"no role='gripper' joint on {description.name!r}.")
    return joints


def gripper_joint(description: RobotDescription, joint_name: str | None = None) -> JointSpec:
    """Return one ``role: "gripper"`` joint spec of a manifest.

    Args:
        description: The robot manifest.
        joint_name: Which gripper joint. ``None`` is the single-gripper
            convenience and requires the manifest to declare exactly one.

    Returns:
        The gripper ``JointSpec``.

    Raises:
        ROSConfigError: If the manifest declares no gripper-role joint; if
            ``joint_name`` is ``None`` and it declares more than one — guessing
            which of several is "the" gripper is exactly the kind of implicit
            choice that must not be buried in a safety-adjacent trigger; or if
            ``joint_name`` names no gripper-role joint.

    Example:
        >>> from openral_core import RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> gripper_joint(d, "right_gripper").child_link
        'openarm_right_finger_pair'
    """
    joints = gripper_joints(description)
    if joint_name is None:
        if len(joints) != 1:
            raise ROSConfigError(
                f"grasp trigger needs exactly one role='gripper' joint on "
                f"{description.name!r}, found {len(joints)}; name one with joint_name."
            )
        return joints[0]
    for joint in joints:
        if joint.name == joint_name:
            return joint
    raise ROSConfigError(
        f"{joint_name!r} is not a role='gripper' joint of {description.name!r}; "
        f"gripper joints are {[joint.name for joint in joints]}."
    )


class PositionStallTrigger:
    """Debounced attach / regrasp / detach events from a jaw stalling short of its command.

    Fed the commanded target (``command``, whenever the HAL applies one) and one
    ``JointState`` per HAL read tick (``update``); ``update`` returns a ``GraspEvent``
    on the tick a transition is confirmed and ``None`` otherwise.

    Nothing is silently ignored (CLAUDE.md §1.4): a tick with no finite position for the
    joint counts in ``missing_position_ticks`` (the bridge's heartbeat liveness reads it),
    a tick before any command has been seen counts in ``uncommanded_ticks`` — without a
    command there is no "short of it", so the trigger cannot attach — a sample whose
    stamp repeats the last one counts in ``repeated_samples``, and a gap longer than
    ``max_gap_s`` that restarted every window counts in ``gap_resets``. The last
    classified sample's ``last_short_of_command`` and ``last_settle_spread`` say how near
    it came; the bridge logs them per close (``closing``).

    Args:
        description: The robot manifest — supplies the gripper joint and its
            ``closure_calibration``.
        joint_name: The gripper joint to watch. ``None`` requires the manifest to
            declare exactly one; a bimanual robot builds one trigger per hand.
        config: Debounce. Every field is a calibration point.

    Raises:
        ROSConfigError: If the gripper joint cannot be resolved; it declares no
            ``closure_calibration``, or one whose ``closed_position`` is not an end of
            its ``position_limits``; it has no position sensor; or a window is degenerate.

    Example:
        >>> from openral_core import JointState, RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> t = PositionStallTrigger(d, joint_name="left_gripper")
        >>> t.command(0.0)  # close
        >>> ticks = [  # 30 Hz samples of a jaw stalled 0.2 rad short
        ...     JointState(name=["left_gripper"], position=[0.2], stamp_ns=i * 33_333_333)
        ...     for i in range(7)
        ... ]
        >>> [t.update(held) for held in ticks][-1]
        <GraspEvent.ATTACH: 'attach'>
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        joint_name: str | None = None,
        config: PositionStallConfig | None = None,
    ) -> None:
        """Resolve the gripper joint and its calibration; refuse an uncalibrated one."""
        self._config = config or PositionStallConfig()
        self._config.validate()
        self._consecutive_ns = self._config.consecutive_s * 1e9
        self._settle_ns = self._config.settle_s * 1e9
        spec = gripper_joint(description, joint_name)
        calibration = spec.closure_calibration
        if calibration is None:
            raise ROSConfigError(
                f"gripper joint {spec.name!r} of {description.name!r} declares no "
                "closure_calibration; the position-stall grasp trigger will not guess its "
                "closed position, rest offset, stall gap or settle tolerance."
            )
        if not spec.has_position_sensor:
            raise ROSConfigError(
                f"gripper joint {spec.name!r} of {description.name!r} has no position sensor; "
                "a position-stall trigger cannot read it."
            )
        limits = spec.position_limits
        if limits is not None and not any(
            math.isclose(calibration.closed_position, end, abs_tol=1e-9) for end in limits
        ):
            raise ROSConfigError(
                f"closure_calibration.closed_position {calibration.closed_position} of "
                f"{spec.name!r} is not an end of its position_limits {limits}; the trigger "
                "reads 'closer to closed' as the distance to that end."
            )
        self._joint_name = spec.name
        self._closed = calibration.closed_position
        self._rest = calibration.closed_rest_offset
        self._gap = calibration.stall_gap
        self._settle = calibration.settle_tolerance
        # (stamp_ns, position), oldest first, pruned to just cover ``settle_s``.
        self._history: deque[tuple[int, float]] = deque()
        self._last_stamp_ns: int | None = None
        self._command: float | None = None
        self._loaded = False
        self._hold: float | None = None
        self._streak_since_ns: int | None = None
        self._streak_kind: GraspEvent | None = None
        self._streak_samples = 0
        self.missing_position_ticks = 0
        self.uncommanded_ticks = 0
        self.repeated_samples = 0
        self.gap_resets = 0
        self.last_short_of_command: float | None = None
        self.last_settle_spread: float | None = None

    @property
    def joint_name(self) -> str:
        """Name of the gripper joint this trigger reads."""
        return self._joint_name

    @property
    def attached(self) -> bool:
        """Whether the trigger currently believes the jaws are loaded."""
        return self._loaded

    @property
    def last_command(self) -> float | None:
        """The last commanded target folded in, or ``None`` before the first."""
        return self._command

    @property
    def closing(self) -> bool:
        """Whether the last command is within ``stall_gap`` of closed — the band ATTACH reads."""
        return self._command is not None and abs(self._command - self._closed) <= self._gap

    @property
    def thresholds(self) -> tuple[float, float, float, float]:
        """``(closed_position, closed_rest_offset, stall_gap, settle_tolerance)`` in force."""
        return (self._closed, self._rest, self._gap, self._settle)

    def command(self, target: float) -> None:
        """Fold in the target the HAL just commanded this jaw to (joint units).

        A non-finite target is ignored: the safety kernel never approves one, and it
        cannot be a reference for "short of the command".
        """
        if math.isfinite(target):
            self._command = float(target)

    def clear_command(self) -> None:
        """Forget the commanded target: what this jaw was last told is no longer known.

        Until the next ``command`` the trigger cannot ATTACH, REGRASP or DETACH (each tick
        counts in ``uncommanded_ticks``); a held payload stays held. Measuring "short of
        the command" against a stale one is how an open, stationary jaw reads as a grasp.
        """
        self._command = None

    def update(self, state: JointState) -> GraspEvent | None:
        """Fold one state snapshot into the state machine.

        Args:
            state: The tick's joint state, as read from the HAL.

        Returns:
            The confirmed ``GraspEvent``, or ``None`` when this tick confirms nothing.
        """
        stamp_ns = int(state.stamp_ns)
        if self._config.is_repeat(stamp_ns, self._last_stamp_ns):
            # A cached sample read again says nothing new about the jaw.
            self.repeated_samples += 1
            return None
        if self._config.is_gap(stamp_ns, self._last_stamp_ns):
            # What the jaw did in the gap is unknown: no window spans it.
            self.gap_resets += 1
            self._history.clear()
            self._reset_streak()
        self._last_stamp_ns = stamp_ns
        position = self._position_of(state)
        if position is None:
            # A dead position channel can never trigger and must not look settled.
            self.missing_position_ticks += 1
            self._history.clear()
            self._reset_streak()
            return None
        self._history.append((stamp_ns, position))
        while (
            len(self._history) > self._config.settle_samples
            and self._history[1][0] <= stamp_ns - self._settle_ns
        ):
            self._history.popleft()
        if self._command is None:
            self.uncommanded_ticks += 1
        candidate = self._classify(position, stamp_ns)
        if candidate is None or candidate is not self._streak_kind:
            self._streak_kind = candidate
            self._streak_since_ns = stamp_ns if candidate is not None else None
            self._streak_samples = 0
        self._streak_samples += candidate is not None
        if (
            candidate is None
            or self._streak_since_ns is None
            or stamp_ns - self._streak_since_ns < self._consecutive_ns
            or self._streak_samples < self._config.consecutive_samples
        ):
            return None
        self._loaded = candidate is not GraspEvent.DETACH
        self._hold = position if self._loaded else None
        self._reset_streak()
        return candidate

    def _classify(self, position: float, stamp_ns: int) -> GraspEvent | None:
        """Which transition this single sample argues for, before debouncing."""
        if self._command is None:
            return None
        from_closed = abs(position - self._closed)
        command_from_closed = abs(self._command - self._closed)
        closing = command_from_closed <= self._gap
        short_of_command = from_closed - command_from_closed
        positions = [q for _, q in self._history]
        spread = max(positions) - min(positions)
        self.last_short_of_command, self.last_settle_spread = short_of_command, spread
        settled = (
            len(positions) >= self._config.settle_samples
            and self._history[0][0] <= stamp_ns - self._settle_ns
            and spread <= self._settle
        )
        stalled = closing and settled and short_of_command > self._rest + self._gap
        if not self._loaded:
            return GraspEvent.ATTACH if stalled else None
        half = 0.5 * self._gap
        hold = self._hold if self._hold is not None else position
        if closing and short_of_command <= self._rest + half:
            return GraspEvent.DETACH  # slipped out / closed onto nothing
        if not closing and from_closed > abs(hold - self._closed) + half:
            return GraspEvent.DETACH  # commanded open and the jaw opened past the hold
        if stalled and abs(position - hold) > half:
            return GraspEvent.REGRASP
        return None

    def _reset_streak(self) -> None:
        self._streak_since_ns = None
        self._streak_kind = None
        self._streak_samples = 0

    def _position_of(self, state: JointState) -> float | None:
        """Finite gripper position in this snapshot, or ``None`` when absent / non-finite."""
        try:
            index = state.name.index(self._joint_name)
        except ValueError:
            return None
        if index >= len(state.position):
            return None
        value = float(state.position[index])
        return value if math.isfinite(value) else None
