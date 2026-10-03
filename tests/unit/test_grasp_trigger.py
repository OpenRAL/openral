"""Position-stall grasp trigger: stall, empty close, slip, release, re-seat, noise.

Snapshots are real ``openral_core.JointState`` over the real bimanual
``robots/openarm/robot.yaml`` manifest — joint names, ``role: "gripper"`` resolution
and every threshold (``closure_calibration``) come from the shipped robot
(CLAUDE.md §1.11).

The position traces are **shaped on real data** rather than recorded: 30 fps OpenArm
teleop (``qualiadev/openarm-canonical-and-fabians-vr``) shows a jaw commanded to 0.0
stalling 0.18-0.29 rad short on an object and staying flat to ~1e-3 rad, reaching
<= 0.02 rad on nothing, and a free-motion steady-state error of 0.006-0.025 rad. The
traces below reproduce those numbers with the real encoder's LSB (3.815e-4 rad) of
stationary noise; replaying a recorded episode is the attended measurement in design
§5, not a unit test.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import pytest
from openral_core import GripperClosureCalibration, JointState, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_trigger import (
    GraspEvent,
    PositionStallConfig,
    PositionStallTrigger,
    gripper_joint,
    gripper_joints,
)

_OPENARM = "robots/openarm/robot.yaml"
_LSB = 3.815e-4  # DM4310 position LSB, rad
_OPEN = 0.70  # a commanded-open left jaw, rad


@pytest.fixture(scope="module")
def openarm() -> RobotDescription:
    """The real bimanual OpenArm v2 manifest."""
    return RobotDescription.from_yaml(_OPENARM)


def _state(openarm: RobotDescription, **jaws: float) -> JointState:
    """A full 16-joint snapshot; ``jaws`` sets gripper positions, effort all zeros (real)."""
    names = [joint.name for joint in openarm.joints]
    position = [jaws.get(name, 0.0) for name in names]
    return JointState(name=names, position=position, effort=[0.0] * len(names), stamp_ns=0)


def _close(start: float, stop: float, *, step: float = 0.08) -> list[float]:
    """A jaw driven from ``start`` toward ``stop`` at ``step`` rad per 30 Hz tick."""
    out, q = [], start
    while abs(q - stop) > step:
        q += step if stop > q else -step
        out.append(q)
    return out


def _flat(at: float, ticks: int) -> list[float]:
    """Stationary jaw with +-1 LSB encoder noise."""
    return [at + _LSB * ((i % 3) - 1) for i in range(ticks)]


def _run(
    trigger: PositionStallTrigger, openarm: RobotDescription, trace: Iterable[float]
) -> list[GraspEvent]:
    events = []
    for q in trace:
        event = trigger.update(_state(openarm, **{trigger.joint_name: q}))
        if event is not None:
            events.append(event)
    return events


def _left(openarm: RobotDescription) -> PositionStallTrigger:
    trigger = PositionStallTrigger(openarm, joint_name="left_gripper")
    trigger.command(_OPEN)
    return trigger


def test_the_manifest_calibration_is_what_the_trigger_uses(openarm: RobotDescription) -> None:
    left = PositionStallTrigger(openarm, joint_name="left_gripper")
    right = PositionStallTrigger(openarm, joint_name="right_gripper")
    assert left.thresholds == (0.0, 0.0086, 0.08, 0.001)
    assert right.thresholds == (0.0, 0.0116, 0.08, 0.001)


@pytest.mark.parametrize("stall_at", [0.18, 0.29])
def test_closing_on_an_object_attaches_once_settled(
    openarm: RobotDescription, stall_at: float
) -> None:
    """The real stall band: 0.18-0.29 rad short of a 0.0 command, flat to ~1e-3."""
    trigger = _left(openarm)
    assert _run(trigger, openarm, _flat(_OPEN, 10)) == []
    trigger.command(0.0)
    assert _run(trigger, openarm, _close(_OPEN, stall_at)) == [], "moving is not stalled"
    events = _run(trigger, openarm, _flat(stall_at, 12))
    assert events == [GraspEvent.ATTACH]
    assert trigger.attached


def test_attach_waits_for_settle_and_debounce(openarm: RobotDescription) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    flat = _flat(0.2, 12)
    # settle_ticks=5 samples, then 3 agreeing ticks: the 7th flat sample attaches.
    assert _run(trigger, openarm, flat[:6]) == []
    assert _run(trigger, openarm, flat[6:7]) == [GraspEvent.ATTACH]


def test_closing_on_nothing_never_attaches(openarm: RobotDescription) -> None:
    """The jaw reaches its rest offset (<= 0.02 rad) — that is an empty close."""
    trigger = _left(openarm)
    trigger.command(0.0)
    assert _run(trigger, openarm, [*_close(_OPEN, 0.02), *_flat(0.02, 30)]) == []
    assert _run(trigger, openarm, _flat(0.0086, 30)) == []
    assert not trigger.attached


def test_free_motion_tracking_error_never_attaches(openarm: RobotDescription) -> None:
    """Steady-state error 0.006-0.025 rad around a partly-closed command is not a stall."""
    trigger = _left(openarm)
    trigger.command(0.05)  # inside the close band
    assert _run(trigger, openarm, _flat(0.05 + 0.025, 30)) == []


def test_a_jaw_held_open_is_not_a_grasp(openarm: RobotDescription) -> None:
    """Far from the command but commanded OPEN: no close command, no grasp."""
    trigger = _left(openarm)
    trigger.command(0.5)
    assert _run(trigger, openarm, _flat(0.3, 30)) == []


def test_noise_on_a_held_jaw_never_chatters(openarm: RobotDescription) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    events = _run(trigger, openarm, _flat(0.2, 200))
    assert events == [GraspEvent.ATTACH], "one attach, then nothing across 200 noisy ticks"


def test_a_slip_detaches_while_still_commanded_closed(openarm: RobotDescription) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    _run(trigger, openarm, _flat(0.2, 10))
    assert trigger.attached
    events = _run(trigger, openarm, [*_close(0.2, 0.01), *_flat(0.01, 5)])
    assert events == [GraspEvent.DETACH]
    assert not trigger.attached


def test_release_detaches_only_once_the_jaw_opens_past_the_hold(
    openarm: RobotDescription,
) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    _run(trigger, openarm, _flat(0.2, 10))
    trigger.command(_OPEN)
    # Commanded open but the jaw has not moved yet: still held (conservative).
    assert _run(trigger, openarm, _flat(0.2, 10)) == []
    assert trigger.attached
    assert _run(trigger, openarm, _close(0.2, _OPEN)) == [GraspEvent.DETACH]


def test_a_reseat_while_stalled_is_a_regrasp(openarm: RobotDescription) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    _run(trigger, openarm, _flat(0.29, 10))
    events = _run(trigger, openarm, [0.25, 0.22, *_flat(0.20, 10)])
    assert events == [GraspEvent.REGRASP]
    assert trigger.attached


def test_no_command_means_no_attach_and_is_counted(openarm: RobotDescription) -> None:
    trigger = PositionStallTrigger(openarm, joint_name="left_gripper")
    assert _run(trigger, openarm, _flat(0.2, 20)) == []
    assert trigger.uncommanded_ticks == 20
    assert trigger.last_command is None


def test_a_dead_position_channel_is_counted_and_breaks_settle(
    openarm: RobotDescription,
) -> None:
    trigger = _left(openarm)
    trigger.command(0.0)
    names = [joint.name for joint in openarm.joints if joint.name != "left_gripper"]
    gap = JointState(name=names, position=[0.0] * len(names), stamp_ns=0)
    for q in _flat(0.2, 6):
        trigger.update(_state(openarm, left_gripper=q))
        assert trigger.update(gap) is None
    assert trigger.missing_position_ticks == 6
    assert not trigger.attached, "interleaved gaps never accumulate a settle window"
    assert trigger.update(_state(openarm, left_gripper=float("nan"))) is None
    assert trigger.missing_position_ticks == 7


def test_the_right_hand_mirrors_through_its_closed_end(openarm: RobotDescription) -> None:
    """Right jaw range is [-0.7854, 0]: an object stalls at -0.2, nothing at -0.0116."""
    trigger = PositionStallTrigger(openarm, joint_name="right_gripper")
    trigger.command(-_OPEN)
    trigger.command(0.0)
    assert _run(trigger, openarm, _flat(-0.0116, 30)) == []
    assert _run(trigger, openarm, _flat(-0.2, 10)) == [GraspEvent.ATTACH]


def test_a_left_grasp_fires_only_the_left_trigger(openarm: RobotDescription) -> None:
    left = _left(openarm)
    right = PositionStallTrigger(openarm, joint_name="right_gripper")
    left.command(0.0)
    right.command(0.0)
    fired = []
    for q in _flat(0.2, 10):
        state = _state(openarm, left_gripper=q, right_gripper=-0.0116)
        fired.extend(e for e in (left.update(state), right.update(state)) if e)
    assert fired == [GraspEvent.ATTACH]
    assert left.attached and not right.attached


def test_an_uncalibrated_gripper_refuses_to_arm() -> None:
    so101 = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    with pytest.raises(ROSConfigError, match="closure_calibration"):
        PositionStallTrigger(so101)


def test_a_closed_position_off_the_range_end_is_refused(openarm: RobotDescription) -> None:
    bad = GripperClosureCalibration(
        closed_position=0.3, closed_rest_offset=0.0, stall_gap=0.08, settle_tolerance=0.001
    )
    joints = [
        j.model_copy(update={"closure_calibration": bad}) if j.name == "left_gripper" else j
        for j in openarm.joints
    ]
    with pytest.raises(ROSConfigError, match="not an end"):
        PositionStallTrigger(
            openarm.model_copy(update={"joints": joints}), joint_name="left_gripper"
        )


@pytest.mark.parametrize(("ticks", "settle"), [(0, 5), (3, 1)])
def test_a_degenerate_debounce_is_refused(
    openarm: RobotDescription, ticks: int, settle: int
) -> None:
    with pytest.raises(ROSConfigError, match="PositionStallConfig"):
        PositionStallTrigger(
            openarm,
            joint_name="left_gripper",
            config=PositionStallConfig(consecutive_ticks=ticks, settle_ticks=settle),
        )


def test_a_non_finite_command_is_ignored(openarm: RobotDescription) -> None:
    trigger = _left(openarm)
    trigger.command(float("nan"))
    assert trigger.last_command == _OPEN


def test_gripper_joints_lists_both_openarm_hands(openarm: RobotDescription) -> None:
    assert [j.name for j in gripper_joints(openarm)] == ["left_gripper", "right_gripper"]


def test_an_unnamed_trigger_refuses_to_guess_between_two_grippers(
    openarm: RobotDescription,
) -> None:
    with pytest.raises(ROSConfigError, match="exactly one"):
        gripper_joint(openarm)


def test_a_non_gripper_joint_name_is_rejected(openarm: RobotDescription) -> None:
    with pytest.raises(ROSConfigError, match="not a role='gripper'"):
        gripper_joint(openarm, "left_joint1")


def test_gripper_joints_raises_only_on_zero(openarm: RobotDescription) -> None:
    joints = [j.model_copy(update={"role": "arm"}) for j in openarm.joints]
    with pytest.raises(ROSConfigError, match="no role='gripper'"):
        gripper_joints(openarm.model_copy(update={"joints": joints}))


def test_the_event_sequence_is_attach_then_detach_over_a_full_cycle(
    openarm: RobotDescription,
) -> None:
    """open -> close on object -> hold -> open: exactly ATTACH, DETACH."""
    trigger = _left(openarm)
    events = _run(trigger, openarm, _flat(_OPEN, 10))
    trigger.command(0.0)
    events += _run(trigger, openarm, itertools.chain(_close(_OPEN, 0.22), _flat(0.22, 30)))
    trigger.command(_OPEN)
    events += _run(trigger, openarm, itertools.chain(_close(0.22, _OPEN), _flat(_OPEN, 10)))
    assert events == [GraspEvent.ATTACH, GraspEvent.DETACH]
