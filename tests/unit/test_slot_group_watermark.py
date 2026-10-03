"""The (runner_session_id, tick_index) replay watermark (hazard log Entry 036, Amendment 3).

``TickWatermark`` is the rule every slot-group HAL shares; ``SlotGroupStager``
is how the MuJoCo twins and ``OpenArmRealHAL`` apply it. Real ``Action`` models
throughout — the watermark reads nothing else.
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import Action, ControlMode
from openral_hal._slot_group import SlotGroupStager, TickWatermark, compose_slot_group_action

_A = 0xA11CE
_B = 0xB0B
_C = 0xC0FFEE


def _slots(tick: int, session: int) -> list[Action]:
    """A two-slot tick (arm + gripper) as the runner dispatches it."""
    return [
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[[0.1, 0.0]],
            joint_names=["j1"],
            tick_index=tick,
            tick_group_size=2,
            runner_session_id=session,
        ),
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[0.5],
            ee_name="grip",
            tick_index=tick,
            tick_group_size=2,
            runner_session_id=session,
        ),
    ]


def _apply(stager: SlotGroupStager, tick: int, session: int) -> None:
    group = None
    for slot in _slots(tick, session):
        group = stager.stage(slot)
    assert group is not None
    stager.commit(group, applied=compose_slot_group_action(group, ["j1", "grip"]))


def test_same_session_refuses_every_tick_at_or_below_the_watermark_including_one() -> None:
    mark = TickWatermark()
    mark.commit(5, session=_A)
    for tick in (5, 2, 1):
        with pytest.raises(ROSRuntimeError, match="stale slot group"):
            mark.check(tick, session=_A)
    mark.check(6, session=_A)


def test_a_new_session_is_admitted_but_adopted_only_on_commit() -> None:
    stager = SlotGroupStager()
    _apply(stager, 40, _A)
    assert stager.stage(_slots(1, _B)[0]) is None  # a stray slot
    assert (stager.last_committed_tick, stager.last_committed_session) == (40, _A)
    # The old runner is still current until B commits: its next tick is fine,
    # and it displaces B's half-staged tick instead of merging with it — a
    # different session's partial is dropped, not reported as this tick's loss.
    assert stager.stage(_slots(41, _A)[0]) is None
    group = stager.stage(_slots(41, _A)[1])
    assert group is not None
    stager.commit(group, applied=compose_slot_group_action(group, ["j1", "grip"]))
    _apply(stager, 1, _B)
    assert (stager.last_committed_tick, stager.last_committed_session) == (1, _B)


def test_retired_sessions_are_refused_after_adoption_whatever_their_tick() -> None:
    stager = SlotGroupStager()
    _apply(stager, 40, _A)
    _apply(stager, 1, _B)
    _apply(stager, 1, _C)
    for session in (_A, _B):
        with pytest.raises(ROSRuntimeError, match="superseded"):
            stager.stage(_slots(99, session)[0])
        with pytest.raises(ROSRuntimeError, match="superseded"):
            stager.admit(_slots(99, session)[0])
    assert stager.pending == 0
    assert (stager.last_committed_tick, stager.last_committed_session) == (1, _C)


def test_ungrouped_admit_and_commit_tick_share_the_session_rule() -> None:
    stager = SlotGroupStager()
    ramp = Action(
        control_mode=ControlMode.JOINT_POSITION,
        joint_targets=[[0.0, 0.0]],
        tick_index=3,
        runner_session_id=_A,
    )
    stager.admit(ramp)
    stager.commit_tick(ramp)
    with pytest.raises(ROSRuntimeError, match="stale slot group"):
        stager.admit(ramp)
    restarted = ramp.model_copy(update={"tick_index": 1, "runner_session_id": _B})
    stager.admit(restarted)
    assert stager.last_committed_session == _A  # admit never moves it
    stager.commit_tick(restarted)
    with pytest.raises(ROSRuntimeError, match="superseded"):
        stager.admit(ramp.model_copy(update={"tick_index": 50}))


def test_reset_clears_the_watermark_but_a_dead_runner_stays_dead() -> None:
    stager = SlotGroupStager()
    _apply(stager, 4, _A)
    _apply(stager, 1, _B)
    stager.reset()
    assert (stager.last_committed_tick, stager.last_committed_session) == (0, 0)
    with pytest.raises(ROSRuntimeError, match="superseded"):
        stager.stage(_slots(1, _A)[0])
    _apply(stager, 2, _B)  # the live runner carries on after a reconnect


def test_legacy_actions_keep_the_entry_036_heuristic_exactly() -> None:
    stager = SlotGroupStager()
    _apply(stager, 7, 0)
    for stale in (7, 5):
        with pytest.raises(ROSRuntimeError, match="stale slot group"):
            stager.stage(_slots(stale, 0)[0])
    _apply(stager, 1, 0)  # tick 1 above a watermark of 1: a restarted runner
    assert stager.last_committed_tick == 1
    with pytest.raises(ROSRuntimeError, match="stale slot group"):
        stager.stage(_slots(1, 0)[0])


def test_the_stager_holds_the_command_each_commit_applied_until_reset() -> None:
    """``last_applied_action``: the composed group, then an ungrouped action, then nothing.

    One accessor every stager-backed HAL reads, so the grasp trigger sees what the robot
    was actually told — the composed row for a group, never a staged slot.
    """
    stager = SlotGroupStager()
    slots = _slots(1, _A)
    assert stager.stage(slots[0]) is None
    assert stager.last_applied_action is None, "a staged slot applied nothing"
    group = stager.stage(slots[1])
    assert group is not None
    composed = compose_slot_group_action(group, ["j1", "grip"])
    stager.commit(group, applied=composed)
    assert stager.last_applied_action is composed
    assert composed.joint_targets == [[0.1, 0.5]]
    ramp = Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0, 0.0]])
    stager.commit_tick(ramp)  # unticked: the watermark stays, the command moves
    assert stager.last_applied_action is ramp
    assert stager.last_committed_tick == 1
    stager.discard()
    assert stager.last_applied_action is ramp, "an estop drop is not a new command"
    stager.reset()
    assert stager.last_applied_action is None, "after a disconnect nothing is commanded"


def test_a_new_runner_session_drops_the_old_sessions_partial_tick_without_raising() -> None:
    # Goal 2 on a restarted runner must start clean: the dead run's half tick is
    # nobody's loss, so it is dropped (logged) rather than raised as incomplete.
    stager = SlotGroupStager()
    assert stager.stage(_slots(195, _A)[0]) is None
    assert stager.stage(_slots(1, _B)[0]) is None
    assert stager.pending == 1
    group = stager.stage(_slots(1, _B)[1])
    assert group is not None
    assert {a.runner_session_id for a in group} == {_B}


def test_the_same_session_changing_tick_mid_group_still_raises() -> None:
    # A lost slot within one run is still a reported loss.
    stager = SlotGroupStager()
    stager.stage(_slots(7, _A)[0])
    with pytest.raises(ROSRuntimeError, match="incomplete slot group"):
        stager.stage(_slots(8, _A)[0])


def test_discard_reports_what_it_dropped_and_keeps_the_watermark() -> None:
    stager = SlotGroupStager()
    _apply(stager, 5, _A)
    stager.stage(_slots(6, _A)[0])
    assert stager.discard() == 1
    assert stager.discard() == 0
    assert (stager.last_committed_tick, stager.last_committed_session) == (5, _A)
