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
        error: ``|target - measured|``.
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
        tolerance_rad: Error below which a joint is tracking (rad, or m for prismatic).
        window_s: How long the error must persist without progress to fault.
        min_progress: Fraction of the error the joint must close per window.

    Raises:
        ValueError: A non-positive tolerance or window, or progress outside (0, 1].

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
    ) -> None:
        if not (tolerance_rad > 0 and window_s > 0 and 0 < min_progress <= 1):
            raise ValueError(
                "TrackingWatch needs tolerance_rad > 0, window_s > 0, 0 < min_progress <= 1; "
                f"got {tolerance_rad}, {window_s}, {min_progress}"
            )
        self._order = [j.name for j in description.joints]
        self.watched = frozenset(
            j.name
            for j in description.joints
            if j.joint_type in _WATCHED_TYPES and j.role != "gripper"
        )
        self.tolerance = tolerance_rad
        self.window_s = window_s
        self.min_progress = min_progress
        self._target: dict[str, float] = {}
        self._since: dict[str, tuple[float, float]] = {}

    def observe_command(self, action: Any) -> None:  # noqa: ANN401  # reason: typed Action lives in openral_core; duck-typed so the ROS path needs no import
        """Fold in the command the HAL applied (ungrouped, or a slot group's composed action).

        Only ``JOINT_POSITION`` actions carry targets; the chunk's last row is the goal
        the trajectory controller is handed. Other modes leave the targets as they are.
        """
        if action.control_mode is not ControlMode.JOINT_POSITION or not action.joint_targets:
            return
        targets = joint_row_targets(action.joint_targets[-1], action.joint_names, self._order)
        for name, value in (targets or {}).items():
            if name in self.watched and math.isfinite(value):
                self._target[name] = value

    def clear(self) -> None:
        """Forget every target and window (e-stop latched or cleared, HAL reset)."""
        self._target.clear()
        self._since.clear()

    def clear_command(self) -> None:
        """The HAL applied a command it cannot state: forget the targets (never trip on a guess)."""
        self.clear()

    def observe_state(
        self, names: Sequence[str], positions: Sequence[float], now_s: float
    ) -> TrackingFault | None:
        """Judge one measured state; the first fault, or ``None``."""
        for name, q in zip(names, positions, strict=False):
            target = self._target.get(name)
            if target is None or not math.isfinite(q):
                self._since.pop(name, None)
                continue
            error = abs(target - q)
            if error <= self.tolerance:
                self._since.pop(name, None)
                continue
            start = self._since.get(name)
            if start is None:
                self._since[name] = (now_s, q)
                continue
            t0, q0 = start
            if now_s - t0 < self.window_s:
                continue
            progress = math.copysign(1.0, target - q0) * (q - q0)
            if progress < self.min_progress * error:
                return TrackingFault(name, target, q, error, progress, now_s - t0)
            self._since[name] = (now_s, q)
        return None
