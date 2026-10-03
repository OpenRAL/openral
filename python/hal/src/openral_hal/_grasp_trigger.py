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
(``qualiadev/openarm-canonical-and-fabians-vr``): a position-controlled jaw
commanded closed on an object **stalls** 0.18-0.29 rad short of the command
and stays flat to ~1e-3 rad; closed on nothing it reaches within ~0.02 rad;
free-motion steady-state error is 0.006-0.025 rad. So:

* **ATTACH** — the command is near closed, the jaw is settled (its position
  span over ``PositionStallConfig.settle_ticks`` is within the joint's
  ``settle_tolerance``), and it sits further from closed than the command by
  more than ``closed_rest_offset + stall_gap``; agreed for
  ``consecutive_ticks`` ticks.
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

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import JointSpec, JointState, RobotDescription

__all__ = [
    "GraspEvent",
    "PositionStallConfig",
    "PositionStallTrigger",
    "gripper_joint",
    "gripper_joints",
]


#: A span needs two samples; one sample is always "settled".
_MIN_SETTLE_TICKS = 2


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
    manifest; only tick counts live here. Calibration points, not benchmarks.

    Attributes:
        consecutive_ticks: How many consecutive samples must agree before a transition
            is emitted. ``3`` — ~100 ms at the OpenArm's 30 Hz action rate, the order
            of the deferred-ack barrier this feeds.
        settle_ticks: Samples the jaw's position span is measured over to call it
            stationary. ``5``.
    """

    consecutive_ticks: int = 3
    settle_ticks: int = 5


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
    and a tick before any command has been seen counts in ``uncommanded_ticks`` — without a
    command there is no "short of it", so the trigger cannot attach.

    Args:
        description: The robot manifest — supplies the gripper joint and its
            ``closure_calibration``.
        joint_name: The gripper joint to watch. ``None`` requires the manifest to
            declare exactly one; a bimanual robot builds one trigger per hand.
        config: Debounce. Every field is a calibration point.

    Raises:
        ROSConfigError: If the gripper joint cannot be resolved; it declares no
            ``closure_calibration``, or one whose ``closed_position`` is not an end of
            its ``position_limits``; it has no position sensor; or the debounce is under
            one tick.

    Example:
        >>> from openral_core import JointState, RobotDescription
        >>> d = RobotDescription.from_yaml("robots/openarm/robot.yaml")
        >>> t = PositionStallTrigger(d, joint_name="left_gripper")
        >>> t.command(0.0)  # close
        >>> held = JointState(name=["left_gripper"], position=[0.2], stamp_ns=0)
        >>> [t.update(held) for _ in range(7)][-1]
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
        if self._config.consecutive_ticks < 1 or self._config.settle_ticks < _MIN_SETTLE_TICKS:
            raise ROSConfigError(
                "PositionStallConfig needs consecutive_ticks >= 1 and settle_ticks >= 2, got "
                f"{self._config.consecutive_ticks} and {self._config.settle_ticks}."
            )
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
        self._history: deque[float] = deque(maxlen=self._config.settle_ticks)
        self._command: float | None = None
        self._loaded = False
        self._hold: float | None = None
        self._streak = 0
        self._streak_kind: GraspEvent | None = None
        self.missing_position_ticks = 0
        self.uncommanded_ticks = 0

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

    def update(self, state: JointState) -> GraspEvent | None:
        """Fold one state snapshot into the state machine.

        Args:
            state: The tick's joint state, as read from the HAL.

        Returns:
            The confirmed ``GraspEvent``, or ``None`` when this tick confirms nothing.
        """
        position = self._position_of(state)
        if position is None:
            # A dead position channel can never trigger and must not look settled.
            self.missing_position_ticks += 1
            self._history.clear()
            self._reset_streak()
            return None
        self._history.append(position)
        if self._command is None:
            self.uncommanded_ticks += 1
        candidate = self._classify(position)
        if candidate is None or candidate is not self._streak_kind:
            self._streak_kind = candidate
            self._streak = 1 if candidate is not None else 0
        else:
            self._streak += 1
        if candidate is None or self._streak < self._config.consecutive_ticks:
            return None
        self._loaded = candidate is not GraspEvent.DETACH
        self._hold = position if self._loaded else None
        self._reset_streak()
        return candidate

    def _classify(self, position: float) -> GraspEvent | None:
        """Which transition this single sample argues for, before debouncing."""
        if self._command is None:
            return None
        from_closed = abs(position - self._closed)
        command_from_closed = abs(self._command - self._closed)
        closing = command_from_closed <= self._gap
        short_of_command = from_closed - command_from_closed
        settled = (
            len(self._history) == self._history.maxlen
            and max(self._history) - min(self._history) <= self._settle
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
        self._streak = 0
        self._streak_kind = None

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
