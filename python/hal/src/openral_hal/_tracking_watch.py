"""Tracking watch: a position-commanded joint that does not follow its command.

Nothing downstream of the safety kernel used to check that the arm actually moved.
On the Orin cell (2026-10-16) the OpenArm drives stopped producing torque about 4 s
after bringup (|tau| ~0 on every joint, J4 included, which holds ~0.65 Nm against
gravity when driven) while CAN feedback kept arriving: the kernel passed every
chunk, the HAL handed every target to the trajectory controllers, the goal ran to
its deadline, and the arms hung limp with 0.05-0.34 rad between command and pose.

A watched joint (revolute / prismatic / continuous, not ``role: gripper`` — a jaw
stalled on an object is the grasp working) faults when its error to the applied
target stays above ``tolerance_rad`` for ``window_s`` while its progress toward that
target over the window is below ``min_progress`` of the error. Progress is signed: a
limp joint sagging away from its target makes negative progress. That covers drives
not producing torque and a joint blocked by something the map does not hold.

A held target with no usable measurement (a non-finite position, or ``read_state``
failing) for ``window_s`` faults too: feedback lost while commanding is not "fine".
Continuous joints are compared on the circle; prismatic joints use ``tolerance_m``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openral_core.schemas import ControlMode, JointType, RobotDescription

_WATCHED_TYPES = frozenset({JointType.REVOLUTE, JointType.PRISMATIC, JointType.CONTINUOUS})


def joint_row_targets(
    row: Sequence[float], names: Sequence[str] | None, joint_order: Sequence[str]
) -> dict[str, float] | None:
    """An ungrouped ``JOINT_POSITION`` row by joint name; ``None`` when unplaceable.

    A full-dof row is padded (ADR-0102): each owned joint is read at its manifest
    index, as the HAL's slot composition reads it — also when ``names`` lists every
    joint. A row as long as ``names`` is compact: ``names[i]`` owns ``row[i]``. Any
    other length places nothing.

    Example:
        >>> joint_row_targets([0.1, 0.2], ["b"], ["a", "b"])
        {'b': 0.2}
        >>> joint_row_targets([0.3], ["b"], ["a", "b"])
        {'b': 0.3}
    """
    if len(row) == len(joint_order):
        owned = set(names or joint_order)
        return {name: float(row[i]) for i, name in enumerate(joint_order) if name in owned}
    if names and len(row) == len(names):
        return {name: float(value) for name, value in zip(names, row, strict=True)}
    return None


@dataclass(frozen=True)
class TrackingFault:
    """One joint that did not follow its command.

    Attributes:
        joint: Manifest joint name.
        target: The applied target (rad or m).
        measured: The measured position at the fault.
        error: ``|target - measured|`` (``inf`` when no usable measurement arrived).
        progress: Signed motion toward the target over the window.
        window_s: The window the error persisted over.
    """

    joint: str
    target: float
    measured: float
    error: float
    progress: float
    window_s: float


class TrackingWatch:
    """Watches applied ``JOINT_POSITION`` targets against measured joint state.

    Args:
        description: The robot manifest (joint order, types and roles).
        tolerance_rad: Error below which a revolute/continuous joint is tracking (rad).
        window_s: How long the error must persist without progress to fault.
        min_progress: Fraction of the error the joint must close per window.
        tolerance_m: Error below which a prismatic joint is tracking (m).

    Raises:
        ValueError: A non-positive or non-finite tolerance or window, or progress
            outside (0, 1].

    Example:
        >>> from openral_core import RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/openarm/robot.yaml")  # doctest: +SKIP
        >>> watch = TrackingWatch(desc, tolerance_rad=0.05, window_s=1.0)  # doctest: +SKIP
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        tolerance_rad: float,
        window_s: float,
        min_progress: float = 0.2,
        tolerance_m: float = 0.01,
    ) -> None:
        finite = all(map(math.isfinite, (tolerance_rad, window_s, tolerance_m)))
        if not (finite and tolerance_rad > 0 and tolerance_m > 0 and window_s > 0) or not (
            0 < min_progress <= 1
        ):
            raise ValueError(
                "TrackingWatch needs finite tolerance_rad > 0, tolerance_m > 0, window_s > 0 "
                f"and 0 < min_progress <= 1; got {tolerance_rad}, {tolerance_m}, {window_s}, "
                f"{min_progress}"
            )
        self._order = [j.name for j in description.joints]
        self.watched = frozenset(
            j.name
            for j in description.joints
            if j.joint_type in _WATCHED_TYPES and j.role != "gripper"
        )
        self._continuous = frozenset(
            j.name for j in description.joints if j.joint_type is JointType.CONTINUOUS
        )
        self._tolerance = {
            j.name: tolerance_m if j.joint_type is JointType.PRISMATIC else tolerance_rad
            for j in description.joints
        }
        self.window_s = window_s
        self.min_progress = min_progress
        self._target: dict[str, float] = {}
        # Per joint: (window start, position at start, sign of target - start position).
        self._since: dict[str, tuple[float, float, float]] = {}
        self._unread_since: float | None = None

    def _delta(self, name: str, a: float, b: float) -> float:
        """``a - b``, wrapped to (-pi, pi] on a continuous joint."""
        d = a - b
        if name in self._continuous:
            d = math.remainder(d, math.tau)
        return d

    def observe_command(self, action: Any) -> None:  # noqa: ANN401  # reason: typed Action lives in openral_core; duck-typed so the ROS path needs no import
        """Fold in the command the HAL applied (ungrouped, or a slot group's composed action).

        Only ``JOINT_POSITION`` actions carry targets; the chunk's last row is the goal
        the trajectory controller is handed. Any other applied command (another control
        mode, or a row that places nothing) supersedes the old targets without stating
        new ones, so they are forgotten: the arm is no longer being driven to them.
        """
        targets = None
        if action.control_mode is ControlMode.JOINT_POSITION and action.joint_targets:
            targets = joint_row_targets(action.joint_targets[-1], action.joint_names, self._order)
        if targets is None:
            self.clear_command()
            return
        for name, value in targets.items():
            if name in self.watched and math.isfinite(value):
                self._target[name] = value

    def clear(self) -> None:
        """Forget every target and window (e-stop latched or cleared, HAL reset)."""
        self._target.clear()
        self._since.clear()
        self._unread_since = None

    def clear_command(self) -> None:
        """The HAL applied a command it cannot state: forget the targets (never trip on a guess)."""
        self.clear()

    def observe_state(
        self, names: Sequence[str], positions: Sequence[float], now_s: float
    ) -> TrackingFault | None:
        """Judge one measured state; the first fault, or ``None``."""
        self._unread_since = None
        for name, q in zip(names, positions, strict=False):
            target = self._target.get(name)
            if target is None:
                self._since.pop(name, None)
                continue
            start = self._since.get(name)
            if not math.isfinite(q):
                # No usable measurement: keep (or open) the window; it faults if no
                # finite sample arrives within it.
                if start is None:
                    self._since[name] = (now_s, q, 0.0)
                elif now_s - start[0] >= self.window_s:
                    return TrackingFault(name, target, q, math.inf, 0.0, now_s - start[0])
                continue
            error = abs(self._delta(name, target, q))
            if error <= self._tolerance[name]:
                self._since.pop(name, None)
                continue
            heading = math.copysign(1.0, self._delta(name, target, q))
            if start is not None and not math.isfinite(start[1]):
                # The window opened on an unusable sample: keep its clock (alternating
                # NaN must not hold it open forever), start measuring progress from here.
                self._since[name] = (start[0], q, heading)
                start = self._since[name]
                if now_s - start[0] < self.window_s:
                    continue
            if start is None or math.copysign(1.0, self._delta(name, target, start[1])) != start[2]:
                # A new window: none yet, or the target moved to the other side of the
                # window's start (a reversal is a new goal, not a failure to progress
                # toward the old one).
                self._since[name] = (now_s, q, heading)
                continue
            t0, q0, direction = start
            if now_s - t0 < self.window_s:
                continue
            progress = direction * self._delta(name, q, q0)
            if progress < self.min_progress * error:
                return TrackingFault(name, target, q, error, progress, now_s - t0)
            self._since[name] = (now_s, q, heading)
        return None

    def observe_read_failure(self, now_s: float) -> TrackingFault | None:
        """``read_state`` failed: a fault once reads fail for a whole window with a target held."""
        if not self._target:
            self._unread_since = None
            return None
        if self._unread_since is None:
            self._unread_since = now_s
            return None
        elapsed = now_s - self._unread_since
        if elapsed < self.window_s:
            return None
        name = sorted(self._target)[0]
        return TrackingFault(name, self._target[name], math.nan, math.inf, 0.0, elapsed)
