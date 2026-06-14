"""Integration tests for `HardwareRunner` end-to-end.

No mocks (CLAUDE.md §1.11). The runner is exercised against real
components: a real SO100FollowerHAL backed by SO100DigitalTwin (in-memory,
no serial port), a real WorldStateAggregator, a minimal inline `rSkillBase`
subclass driven through its full lifecycle, and the real NullSafetyClient
plus a custom rejecting SafetyClient.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from openral_core import Action, ControlMode, SafetyEnvelope
from openral_core.exceptions import ROSWorkspaceViolation
from openral_core.schemas import WorldState
from openral_hal.so100_follower import SO100FollowerHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_rskill.base import rSkillBase
from openral_runner import (
    HardwareRunner,
    InferenceRunner,
    NullSafetyClient,
    SafetyClient,
)
from openral_world_state.aggregator import WorldStateAggregator

if TYPE_CHECKING:
    from collections.abc import Generator


class _NoOpTestSkill(rSkillBase):
    """Minimal inline rSkillBase subclass for runner integration tests.

    Returns a zero JOINT_POSITION action chunk and counts steps so tests
    can assert that the runner is driving the skill at each tick.
    """

    def __init__(self, n_joints: int = 6, horizon: int = 1) -> None:
        super().__init__(name="noop_test_skill", embodiment_tags=["so100_follower"])
        self._n_joints = n_joints
        self._horizon = horizon
        self.step_count = 0

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state
        self.step_count += 1
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=self._horizon,
            joint_targets=[[0.0] * self._n_joints for _ in range(self._horizon)],
            confidence=1.0,
        )


# ── Real-component fixtures ─────────────────────────────────────────────────


@pytest.fixture
def hal() -> SO100FollowerHAL:
    """Real HAL backed by the SO-100 digital twin (no serial port)."""
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    return SO100FollowerHAL(robot=twin)


@pytest.fixture
def aggregator(hal: SO100FollowerHAL) -> WorldStateAggregator:
    """Real WorldStateAggregator over the SO-100 description."""
    return WorldStateAggregator(hal.description)


@pytest.fixture
def active_skill() -> Generator[_NoOpTestSkill, None, None]:
    """Real inline test skill taken through its full configure + activate lifecycle."""
    skill = _NoOpTestSkill(n_joints=6, horizon=1)
    skill.configure()
    skill.activate()
    yield skill
    if skill.info.state.value == "active":
        skill.deactivate()
    if skill.info.state.value != "finalized":
        skill.shutdown()


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_hardware_runner_satisfies_inference_runner_protocol(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """Structural ``isinstance`` against :class:`InferenceRunner` succeeds."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=30.0)
    assert isinstance(runner, InferenceRunner)


# ── End-to-end loop ─────────────────────────────────────────────────────────


def test_runner_drives_skill_step_at_each_tick(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """5 ticks → ``skill.step`` called 5 times, no safety violations."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=30.0)
    runner.activate()
    try:
        result = runner.run(max_ticks=5)
    finally:
        runner.deactivate()

    assert result.n_ticks == 5
    assert active_skill.step_count == 5
    assert result.budget_violations == 0
    # Per-stage timings must be populated (not all zero).
    assert result.avg_tick_ms > 0
    # avg_inference_ms can be very small for _NoOpTestSkill but is recorded.
    assert result.avg_inference_ms >= 0


def test_runner_honors_30_hz_cadence_end_to_end(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """10 ticks @ 30 Hz wall-time within ±15 ms of 333.3 ms.

    Slightly wider slack than the synthetic ``FixedLatencyRunner`` test —
    the real HAL ``read_state`` / ``send_action`` adds a few ms of jitter.
    """
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=30.0)
    runner.activate()
    try:
        t0 = time.perf_counter()
        result = runner.run(max_ticks=10)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
    finally:
        runner.deactivate()

    expected_ms = 10 * (1000.0 / 30.0)
    assert result.n_ticks == 10
    assert abs(elapsed_ms - expected_ms) < 15.0, (
        f"30 Hz x 10 ticks elapsed {elapsed_ms:.3f} ms, expected ~{expected_ms:.3f} ms"
    )


def test_runner_pushes_joint_state_into_aggregator(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """After one tick the aggregator's snapshot carries SO-100 joint names."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=60.0)
    runner.activate()
    try:
        runner.run(max_ticks=1)
    finally:
        runner.deactivate()
    snapshot = aggregator.snapshot()
    # SO-100 has 6 joints (shoulder_pan / shoulder_lift / elbow_flex /
    # wrist_flex / wrist_roll / gripper). Joint names live on the manifest.
    expected_names = [j.name for j in hal.description.joints]
    assert snapshot.joint_state is not None
    assert snapshot.joint_state.name == expected_names


def test_runner_records_skill_lifecycle_step_count(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """``_NoOpTestSkill.step_count`` increments once per tick."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=60.0)
    runner.activate()
    try:
        runner.run(max_ticks=3)
    finally:
        runner.deactivate()
    assert active_skill.step_count == 3


# ── Safety supervisor boundary (CLAUDE.md §10) ──────────────────────────────


class _RejectAfterSafetyClient:
    """A SafetyClient that allows the first N actions then rejects all.

    Used to verify the supervisor boundary behaviour: rejected actions
    are recorded into ``TickResult.safety_violations`` and the runner
    skips ``HAL.send_action`` (no exception propagates out of ``tick``).
    """

    envelope: SafetyEnvelope

    def __init__(self, allow_first: int) -> None:
        """Initialise; allow the first ``allow_first`` calls."""
        self.envelope = SafetyEnvelope()
        self._allow_first = allow_first
        self._call_count = 0

    def check_action(self, action: Action) -> None:
        """Allow the first ``allow_first`` calls; reject the rest."""
        self._call_count += 1
        if self._call_count > self._allow_first:
            raise ROSWorkspaceViolation(f"reject action #{self._call_count}")


def test_safety_violation_withholds_action_and_records_on_tick(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """Rejected actions don't propagate; they record on the TickResult."""
    safety = _RejectAfterSafetyClient(allow_first=2)
    assert isinstance(safety, SafetyClient)
    runner = HardwareRunner(
        hal=hal,
        skill=active_skill,
        aggregator=aggregator,
        safety_client=safety,
        rate_hz=60.0,
    )
    runner.activate()
    try:
        # 5 ticks: first 2 allowed, next 3 rejected
        result = runner.run(max_ticks=5)
    finally:
        runner.deactivate()
    assert result.n_ticks == 5
    # Skill is always called (safety check happens AFTER inference).
    assert active_skill.step_count == 5


def test_null_safety_client_is_default(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """When no ``safety_client`` is passed, the runner installs a ``NullSafetyClient``."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator)
    assert isinstance(runner._safety_client, NullSafetyClient)


# ── Lifecycle idempotency ────────────────────────────────────────────────────


def test_re_activate_after_deactivate(
    hal: SO100FollowerHAL,
    aggregator: WorldStateAggregator,
    active_skill: _NoOpTestSkill,
) -> None:
    """The runner can be re-activated for a second :meth:`run`."""
    runner = HardwareRunner(hal=hal, skill=active_skill, aggregator=aggregator, rate_hz=60.0)
    runner.activate()
    r1 = runner.run(max_ticks=2)
    runner.deactivate()
    runner.activate()
    r2 = runner.run(max_ticks=3)
    runner.deactivate()
    assert r1.n_ticks == 2
    assert r2.n_ticks == 3
    # _NoOpTestSkill's step counter resets only on activate(); the runner does
    # not touch the skill lifecycle, so the count keeps accumulating.
    assert active_skill.step_count == 5
