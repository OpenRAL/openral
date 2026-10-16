"""The grasp trigger on a sim slower than real time: fed on the sim clock, it settles.

Reproduces the LeIsaac PickOrange deploy sim (2026-10-16, ~0.13x real time): 1/60 s
physics steps arrive ~0.13 s apart in wall time — past ``max_gap_s`` — and the HAL
node's 60 Hz publish timer re-reads each post-step frame several times. Wall-stamped,
every step restarts the trigger's history and the held jaw never reads as settled;
re-stamped with the frame's sim time (``openral_hal.lifecycle.grasp_trigger_state``)
the same samples ATTACH. Real ``robots/so101_follower/robot.yaml`` manifest and its
``closure_calibration`` (CLAUDE.md §1.11).
"""

from __future__ import annotations

import pytest
from openral_core import JointState, RobotDescription
from openral_hal._grasp_trigger import GraspEvent, PositionStallConfig, PositionStallTrigger
from openral_hal.lifecycle import grasp_trigger_state

_SO101 = "robots/so101_follower/robot.yaml"
_SIM_STEP_NS = 16_666_667  # one 60 Hz physics step
_WALL_STEP_NS = 130_000_000  # what that step took in wall time at ~0.13x real time
_REREADS = 7  # the 60 Hz wall publish timer re-reads each frame this often


@pytest.fixture(scope="module")
def so101() -> RobotDescription:
    return RobotDescription.from_yaml(_SO101)


def _hold(
    so101: RobotDescription, *, sim_clock: bool
) -> tuple[list[GraspEvent], PositionStallTrigger]:
    """A policy-shaped hold (command 0.20, jaw stopped at ~0.30) over 30 slow sim steps."""
    trigger = PositionStallTrigger(so101, config=PositionStallConfig.for_rate(60.0))
    trigger.command(0.20)
    names = [j.name for j in so101.joints]
    events: list[GraspEvent] = []
    wall0 = 1_700_000_000_000_000_000
    for step in range(30):
        jaw = 0.30 + 0.002 * ((step % 3) - 1)
        state = JointState(
            name=names,
            position=[0.0] * (len(names) - 1) + [jaw],
            stamp_ns=wall0 + step * _WALL_STEP_NS,
        )
        sim_ns = 5_000_000_000 + step * _SIM_STEP_NS
        for _ in range(_REREADS):
            fed = grasp_trigger_state(state, sim_ns if sim_clock else None)
            event = trigger.update(fed)
            if event is not None:
                events.append(event)
    return events, trigger


def test_wall_stamps_on_a_slow_sim_never_settle(so101: RobotDescription) -> None:
    """The g6 failure: every 0.13 s wall step is a gap, so the jaw never reads as settled."""
    events, trigger = _hold(so101, sim_clock=False)
    assert events == []
    assert trigger.gap_resets >= 25
    assert trigger.repeated_samples == 30 * (_REREADS - 1)


def test_sim_clock_stamps_settle_and_attach(so101: RobotDescription) -> None:
    events, trigger = _hold(so101, sim_clock=True)
    assert events == [GraspEvent.ATTACH]
    assert trigger.gap_resets == 0
    assert trigger.repeated_samples == 30 * (_REREADS - 1), "a re-read is still a repeat"


def test_a_real_hal_frame_is_untouched(so101: RobotDescription) -> None:
    """No sim clock (a real HAL): the trigger reads the HAL's own wall stamp."""
    state = JointState(
        name=[j.name for j in so101.joints], position=[0.0] * 6, stamp_ns=1_700_000_000_000_000_001
    )
    assert grasp_trigger_state(state, None) is state
    restamped = grasp_trigger_state(state, 42)
    assert restamped.stamp_ns == 42
    assert state.stamp_ns == 1_700_000_000_000_000_001, "the HAL's snapshot is not mutated"
    assert restamped.position == state.position
