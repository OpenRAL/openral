"""Gripper-effort grasp trigger: debounce, hysteresis, regrasp, effort-channel health.

Every state snapshot here is a real :class:`openral_core.JointState` over the
real ``robots/so101_follower/robot.yaml`` manifest, so the joint names, the
``role: "gripper"`` resolution and the ``effort_limit`` the fractional
thresholds scale against all come from the shipped robot rather than from
placeholders (CLAUDE.md §1.11).

The effort *values* are chosen by these tests, and deliberately so: this is the
component that decides when to believe a grasp happened, and the point of the
suite is to pin what it does at, above and below its own thresholds. What no
test here can establish — and what the module says out loud — is whether an
SO-101's Feetech servos actually report those numbers. That is hardware
validation, and :func:`~openral_hal._grasp_trigger.assess_effort_readback` is the
hook for it; the last test in this file exercises the hook itself.
"""

from __future__ import annotations

import pytest
from openral_core import JointState, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal._grasp_trigger import (
    GraspEvent,
    GraspTriggerConfig,
    GripperEffortTrigger,
    assess_effort_readback,
    gripper_joint,
)

_ROBOT_YAML = "robots/so101_follower/robot.yaml"


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    """The real SO-101 follower manifest."""
    return RobotDescription.from_yaml(_ROBOT_YAML)


def _state(
    description: RobotDescription,
    *,
    effort: float,
    position: float = 0.2,
    stamp_ns: int = 0,
    with_effort: bool = True,
) -> JointState:
    """A full-arm snapshot with the gripper carrying ``effort`` / ``position``."""
    names = [joint.name for joint in description.joints]
    positions = [0.0] * len(names)
    efforts = [0.0] * len(names)
    index = names.index("gripper")
    positions[index] = position
    efforts[index] = effort
    return JointState(
        name=names,
        position=positions,
        effort=efforts if with_effort else [],
        stamp_ns=stamp_ns,
    )


def test_gripper_joint_resolves_the_single_manifest_gripper(
    description: RobotDescription,
) -> None:
    """The SO-101's one role='gripper' joint is found, with its effort limit."""
    spec = gripper_joint(description)
    assert spec.name == "gripper"
    assert spec.effort_limit == pytest.approx(3.35)


def test_thresholds_scale_off_the_manifest_effort_limit(
    description: RobotDescription,
) -> None:
    """Fractional config becomes absolute thresholds from the robot's own limit."""
    trigger = GripperEffortTrigger(description)
    attach, release = trigger.thresholds_n
    assert attach == pytest.approx(0.30 * 3.35)
    assert release == pytest.approx(0.10 * 3.35)
    assert release < attach, "the hysteresis band must be non-empty"


def test_attach_needs_n_consecutive_ticks(description: RobotDescription) -> None:
    """A loaded gripper fires ATTACH only once the debounce is satisfied."""
    trigger = GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=3))
    loaded = _state(description, effort=2.0)
    assert trigger.update(loaded) is None
    assert trigger.update(loaded) is None
    assert trigger.update(loaded) is GraspEvent.ATTACH
    assert trigger.attached is True


def test_a_transient_effort_spike_never_attaches(description: RobotDescription) -> None:
    """One tick over the threshold between quiet ticks is rejected as a spike.

    This is the whole reason the debounce exists: closing the jaws accelerates
    the servo and spikes the effort readback whether or not anything is between
    them, and a one-tick attach would hand the collision checker a payload for a
    grasp that never happened.
    """
    trigger = GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=3))
    for effort in (0.0, 3.0, 0.0, 3.0, 0.0):
        assert trigger.update(_state(description, effort=effort)) is None
    assert trigger.attached is False


def test_hysteresis_holds_the_attachment_between_the_thresholds(
    description: RobotDescription,
) -> None:
    """Effort in the band (release, attach) neither detaches nor re-attaches."""
    config = GraspTriggerConfig(consecutive_ticks=2)
    trigger = GripperEffortTrigger(description, config=config)
    loaded = _state(description, effort=2.0)
    trigger.update(loaded)
    assert trigger.update(loaded) is GraspEvent.ATTACH

    # 0.5 N.m sits between release (0.335) and attach (1.005).
    band = _state(description, effort=0.5)
    for _ in range(6):
        assert trigger.update(band) is None
    assert trigger.attached is True


def test_detach_needs_the_release_threshold_and_the_debounce(
    description: RobotDescription,
) -> None:
    """Dropping under the release threshold for N ticks detaches, and only then."""
    trigger = GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=2))
    loaded = _state(description, effort=2.0)
    trigger.update(loaded)
    assert trigger.update(loaded) is GraspEvent.ATTACH

    empty = _state(description, effort=0.0)
    assert trigger.update(empty) is None
    assert trigger.update(empty) is GraspEvent.DETACH
    assert trigger.attached is False


def test_regrasp_fires_when_the_jaw_moves_while_still_loaded(
    description: RobotDescription,
) -> None:
    """A re-seated payload invalidates the fitted geometry, so it re-segments."""
    config = GraspTriggerConfig(consecutive_ticks=2, regrasp_position_delta=0.15)
    trigger = GripperEffortTrigger(description, config=config)
    held = _state(description, effort=2.0, position=0.20)
    trigger.update(held)
    assert trigger.update(held) is GraspEvent.ATTACH

    # Still loaded, but the jaw has closed 0.25 of its normalised span.
    moved = _state(description, effort=2.0, position=0.45)
    assert trigger.update(moved) is None
    assert trigger.update(moved) is GraspEvent.REGRASP
    assert trigger.attached is True, "a regrasp stays attached; it does not detach first"

    # The regrasp re-baselines: holding at the NEW position is not another regrasp.
    for _ in range(4):
        assert trigger.update(moved) is None


def test_a_jaw_that_holds_still_never_regrasps(description: RobotDescription) -> None:
    """Sub-threshold jaw drift under load is not a re-seat."""
    config = GraspTriggerConfig(consecutive_ticks=2, regrasp_position_delta=0.15)
    trigger = GripperEffortTrigger(description, config=config)
    held = _state(description, effort=2.0, position=0.20)
    trigger.update(held)
    assert trigger.update(held) is GraspEvent.ATTACH
    for _ in range(6):
        assert trigger.update(_state(description, effort=2.0, position=0.28)) is None


def test_a_driver_with_no_effort_channel_is_counted_not_swallowed(
    description: RobotDescription,
) -> None:
    """No effort value means no attach — and a visible counter saying why."""
    trigger = GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=2))
    for _ in range(5):
        assert trigger.update(_state(description, effort=0.0, with_effort=False)) is None
    assert trigger.attached is False
    assert trigger.missing_effort_ticks == 5


def test_sign_convention_does_not_matter(description: RobotDescription) -> None:
    """A driver that reports the closing load as negative still attaches."""
    trigger = GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=2))
    loaded = _state(description, effort=-2.0)
    trigger.update(loaded)
    assert trigger.update(loaded) is GraspEvent.ATTACH


def test_inverted_hysteresis_is_rejected(description: RobotDescription) -> None:
    """A config with no hysteresis band is a configuration error, not a default."""
    with pytest.raises(ROSConfigError, match="hysteresis"):
        GripperEffortTrigger(
            description,
            config=GraspTriggerConfig(attach_effort_fraction=0.10, release_effort_fraction=0.10),
        )


def test_zero_debounce_is_rejected(description: RobotDescription) -> None:
    """consecutive_ticks < 1 would make every effort spike an attachment."""
    with pytest.raises(ROSConfigError, match="consecutive_ticks"):
        GripperEffortTrigger(description, config=GraspTriggerConfig(consecutive_ticks=0))


def test_effort_readback_health_rejects_a_flat_channel(
    description: RobotDescription,
) -> None:
    """The SO-101 falsification hook fails an all-zero effort trace.

    This is the shape of the *real* unverified question: if a recorded
    open → close-on-object → open sequence comes back flat, the servos are not
    reporting load and this trigger cannot be the attach signal.
    """
    flat = [_state(description, effort=0.0, stamp_ns=i) for i in range(20)]
    health = assess_effort_readback(flat, description=description)
    assert health.usable is False
    assert health.nonzero_samples == 0
    assert "identically zero" in health.reason
    assert health.present_samples == 20


def test_effort_readback_health_rejects_an_absent_channel(
    description: RobotDescription,
) -> None:
    """A driver publishing an empty effort list scores zero present samples."""
    absent = [_state(description, effort=0.0, with_effort=False, stamp_ns=i) for i in range(5)]
    health = assess_effort_readback(absent, description=description)
    assert health.usable is False
    assert health.present_samples == 0
    assert "empty effort channel" in health.reason


def test_effort_readback_health_rejects_a_channel_that_barely_moves(
    description: RobotDescription,
) -> None:
    """Quantisation noise is not a load signal: the span gate catches it."""
    noisy = [
        _state(description, effort=0.01 * (i % 3), stamp_ns=i)  # span 0.02 vs limit 3.35
        for i in range(30)
    ]
    health = assess_effort_readback(noisy, description=description)
    assert health.usable is False
    assert "effort span" in health.reason


def test_effort_readback_health_accepts_a_real_load_cycle(
    description: RobotDescription,
) -> None:
    """An open → close-on-object → open trace with real load passes."""
    cycle = [0.0, 0.0, 0.4, 1.6, 2.4, 2.5, 2.4, 1.1, 0.2, 0.0]
    states = [_state(description, effort=e, stamp_ns=i) for i, e in enumerate(cycle)]
    health = assess_effort_readback(states, description=description)
    assert health.usable is True
    assert health.reason == ""
    assert health.span == pytest.approx(2.5)
    assert health.samples == len(cycle)
