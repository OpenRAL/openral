"""The DROID Robotiq's sim-measured grasp trigger: it fires on a banana hold, never on nothing.

``robots/droid_franka_robotiq/robot.yaml`` declares ``finger_joint``'s
``closure_calibration`` from deploy-sim trial t5 (2026-10-16, 15 Hz joint states, sim
time): π0.5 held the banana at commands 0.093-0.105 with the jaw stopped at
0.469-0.696 and a 0.267 s span of 0.0028-0.0185; a close on nothing sat at the
command (gap <= 0.0073, span <= 0.0163). Each case below replays those measured
numbers through the real ``PositionStallTrigger`` on the real manifest
(CLAUDE.md §1.11): the extremes of each phase, not invented values.
"""

from __future__ import annotations

import pytest
from openral_core import JointState, RobotDescription
from openral_hal._grasp_trigger import GraspEvent, PositionStallConfig, PositionStallTrigger

_ROBOT = "robots/droid_franka_robotiq/robot.yaml"
_TICK_NS = 66_666_667  # 15 Hz, the twin's joint-state rate


@pytest.fixture(scope="module")
def droid() -> RobotDescription:
    return RobotDescription.from_yaml(_ROBOT)


def _replay(
    droid: RobotDescription, command: float, jaw: list[float]
) -> tuple[list[GraspEvent], PositionStallTrigger]:
    trigger = PositionStallTrigger(droid, config=PositionStallConfig.for_rate(15.0))
    trigger.command(command)
    names = [j.name for j in droid.joints]
    events: list[GraspEvent] = []
    for k, f in enumerate(jaw):
        state = JointState(
            name=names, position=[0.0] * (len(names) - 1) + [f], stamp_ns=10**9 + k * _TICK_NS
        )
        event = trigger.update(state)
        if event is not None:
            events.append(event)
    return events, trigger


def _settled(centre: float, span: float, n: int = 12) -> list[float]:
    """``n`` samples alternating over ``span`` around ``centre`` (the measured spread)."""
    return [centre + span / 2 * (1 if k % 2 else -1) for k in range(n)]


def test_the_calibration_is_the_measured_one(droid: RobotDescription) -> None:
    (jaw,) = [j for j in droid.joints if j.role == "gripper"]
    cal = jaw.closure_calibration
    assert cal is not None
    assert (cal.closed_position, cal.closed_rest_offset) == (0.0, 0.0)
    assert (cal.stall_gap, cal.settle_tolerance, cal.close_band) == (0.15, 0.02, 0.20)
    assert cal.command_band == 0.20


@pytest.mark.parametrize(
    ("command", "jaw_centre", "span"),
    [
        (0.105, 0.469, 0.0185),  # the smallest hold gap (0.366) with the largest span
        (0.093, 0.696, 0.0028),
        (0.15, 0.469, 0.0185),  # a hold commanded at pi0.5's loosest close (0.15)
    ],
)
def test_a_measured_banana_hold_attaches(
    droid: RobotDescription, command: float, jaw_centre: float, span: float
) -> None:
    events, _ = _replay(droid, command, _settled(jaw_centre, span))
    assert events == [GraspEvent.ATTACH]


@pytest.mark.parametrize(
    ("command", "jaw_offset", "span"),
    [
        (0.096, 0.0073, 0.0163),  # closed on nothing in t5: the largest gap and span
        (0.096, -0.0008, 0.0018),
        (0.0, 0.0, 0.0),  # the twin's empty close
    ],
)
def test_a_close_on_nothing_never_attaches(
    droid: RobotDescription, command: float, jaw_offset: float, span: float
) -> None:
    events, _ = _replay(droid, command, _settled(command + jaw_offset, span))
    assert events == []


def test_free_motion_is_not_a_close_command(droid: RobotDescription) -> None:
    """Open commands (> close_band) never arm, whatever the jaw does."""
    events, _ = _replay(droid, 0.99, _settled(0.47, 0.0196))
    assert events == []


def test_a_closing_jaw_is_not_settled(droid: RobotDescription) -> None:
    """The twin's close moves ~0.4 per tick; the trigger waits for it to stop."""
    events, _ = _replay(droid, 0.1, [1.0, 0.575, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert events == []
