"""Skill ABC contract — parametrized contract test for every concrete Skill.

CLAUDE.md §5.1: *"types are the contract"*.  This file pins the runtime
contract :class:`openral_rskill.Skill` declares so a typo or signature
drift in any Skill subclass fails the unit lane immediately instead of
waiting for a sim or HIL run.

Why a parametrized test?  When a new Skill subclass is added (e.g. a new
VLA adapter), its author can verify it satisfies the Protocol by appending
a one-line entry to :data:`SKILL_BUILDERS`.  No new test file is needed.

Coverage (asserted against every Skill listed in :data:`SKILL_BUILDERS`)
----------------------------------------------------------------------
- Initial state is :class:`RSkillState.UNCONFIGURED`.
- ``configure()`` from ``unconfigured`` → ``inactive`` and sets
  ``info.weights_loaded`` and ``info.quantized``.
- ``activate()`` from ``inactive`` → ``active`` and sets ``info.warmed_up``.
- ``deactivate()`` from ``active`` → ``inactive``.
- ``shutdown()`` from any state → ``finalized``; idempotent if already
  ``finalized``.
- ``step()`` outside the ``active`` state raises :class:`ROSRuntimeError`.
- ``step()`` in ``active`` returns an :class:`Action`.
- Illegal transitions (``activate()`` from ``unconfigured``,
  ``deactivate()`` from ``inactive``) raise :class:`ROSRuntimeError`.
- Errors raised inside the lifecycle hooks move the Skill to the
  ``error`` state and re-raise.
- ``info`` returns a fresh copy — mutating it does not corrupt internal
  state.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable

import pytest
from openral_core.exceptions import ROSRuntimeError
from openral_core.schemas import (
    Action,
    JointState,
    RSkillState,
    WorldState,
)
from openral_rskill.base import rSkillBase

# ── World state fixture ──────────────────────────────────────────────────────


def _world_state(n_joints: int = 1) -> WorldState:
    return WorldState(
        stamp_ns=time.time_ns(),
        joint_state=JointState(
            name=[f"j{i}" for i in range(n_joints)],
            position=[0.0] * n_joints,
            stamp_ns=time.time_ns(),
        ),
    )


# ── Skill builders ───────────────────────────────────────────────────────────


SkillBuilder = Callable[[], rSkillBase]


class _MinimalSkill(rSkillBase):
    """Smallest possible custom subclass — proves the ABC's contract holds for new authors."""

    def __init__(self) -> None:
        super().__init__(
            name="minimal_skill",
            embodiment_tags=["any"],
            latency_budget_ms=None,
        )

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        from openral_core.schemas import (
            ControlMode,  # reason: keep imports lazy
        )

        del world_state
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0]],
            confidence=1.0,
        )


def _minimal_skill() -> rSkillBase:
    return _MinimalSkill()


SKILL_BUILDERS: dict[str, SkillBuilder] = {
    "_MinimalSkill": _minimal_skill,
}


# Optional: SmolVLAAdapter requires lerobot + torch.  Only add it when the
# heavy deps are present so the contract test stays in the unit lane.
if (
    importlib.util.find_spec("lerobot") is not None
    and importlib.util.find_spec("torch") is not None
):  # pragma: no cover  # reason: only takes the branch when heavy deps install
    pass  # SmolVLAAdapter requires real weights to construct; skip here.


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_initial_state_is_unconfigured(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    assert skill.state is RSkillState.UNCONFIGURED
    assert skill.info.weights_loaded is False
    assert skill.info.quantized is False
    assert skill.info.warmed_up is False


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_configure_transitions_to_inactive_and_sets_subflags(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    assert skill.state is RSkillState.INACTIVE
    assert skill.info.weights_loaded is True
    assert skill.info.quantized is True


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_activate_transitions_to_active_and_sets_warmed_up(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    skill.activate()
    assert skill.state is RSkillState.ACTIVE
    assert skill.info.warmed_up is True


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_deactivate_returns_to_inactive(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    skill.activate()
    skill.deactivate()
    assert skill.state is RSkillState.INACTIVE


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_shutdown_finalizes_from_any_state(skill_name: str) -> None:
    """Shutdown must reach FINALIZED from unconfigured / inactive / active alike."""
    for advance_to in (
        RSkillState.UNCONFIGURED,
        RSkillState.INACTIVE,
        RSkillState.ACTIVE,
    ):
        skill = SKILL_BUILDERS[skill_name]()
        if advance_to is not RSkillState.UNCONFIGURED:
            skill.configure()
        if advance_to is RSkillState.ACTIVE:
            skill.activate()
        skill.shutdown()
        assert skill.state is RSkillState.FINALIZED


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_shutdown_is_idempotent(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    skill.shutdown()
    # Second call is a no-op.
    skill.shutdown()
    assert skill.state is RSkillState.FINALIZED


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_step_outside_active_raises_rosruntimeerror(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    with pytest.raises(ROSRuntimeError, match="must be 'active'"):
        skill.step(_world_state())


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_step_in_active_returns_action(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    skill.activate()
    action = skill.step(_world_state())
    assert isinstance(action, Action)
    # last_inference_ms is recorded as a side-effect of step().
    assert skill.info.last_inference_ms is not None
    assert skill.info.last_inference_ms >= 0.0


# ── Illegal transitions ─────────────────────────────────────────────────────


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_activate_from_unconfigured_raises(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    with pytest.raises(ROSRuntimeError, match="cannot transition"):
        skill.activate()


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_deactivate_from_inactive_raises(skill_name: str) -> None:
    skill = SKILL_BUILDERS[skill_name]()
    skill.configure()
    with pytest.raises(ROSRuntimeError, match="requires 'active' state"):
        skill.deactivate()


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_configure_after_finalize_raises(skill_name: str) -> None:
    """Once finalized, no further transition is allowed."""
    skill = SKILL_BUILDERS[skill_name]()
    skill.shutdown()
    with pytest.raises(ROSRuntimeError, match="cannot transition"):
        skill.configure()


# ── info copy isolation ─────────────────────────────────────────────────────


@pytest.mark.parametrize("skill_name", list(SKILL_BUILDERS.keys()))
def test_skill_info_returns_fresh_copy(skill_name: str) -> None:
    """``skill.info`` is documented as a copy; mutating it must not corrupt internal state."""
    skill = SKILL_BUILDERS[skill_name]()
    snapshot1 = skill.info
    snapshot1.weights_loaded = True  # type: ignore[misc]  # reason: deliberately mutating the copy
    snapshot2 = skill.info
    assert snapshot2.weights_loaded is False  # internal state unchanged


# ── Error-state latching ────────────────────────────────────────────────────


class _ConfigureFailsSkill(rSkillBase):
    """Subclass whose ``_configure_impl`` always raises — pins the error path."""

    def __init__(self) -> None:
        super().__init__(name="configure_fails", embodiment_tags=["any"])

    def _configure_impl(self) -> None:
        raise RuntimeError("synthetic configure failure")

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        from openral_core.schemas import ControlMode

        del world_state
        return Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.0]])


def test_configure_failure_latches_error_state_and_records_message() -> None:
    skill = _ConfigureFailsSkill()
    with pytest.raises(RuntimeError, match="synthetic configure failure"):
        skill.configure()
    assert skill.state is RSkillState.ERROR
    assert skill.info.error_msg is not None
    assert "synthetic configure failure" in skill.info.error_msg


# ── Weight-unload contract (ADR-0050 single-resident-skill eviction) ─────────


class _WeightTrackingSkill(rSkillBase):
    """Counts ``on_load_weights`` / ``on_unload_weights`` — pins the ADR-0050 release contract."""

    def __init__(self) -> None:
        super().__init__(name="weight_tracking", embodiment_tags=["any"])
        self.load_calls = 0
        self.unload_calls = 0

    def on_load_weights(self) -> None:
        self.load_calls += 1

    def on_unload_weights(self) -> None:
        self.unload_calls += 1

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        from openral_core.schemas import ControlMode

        del world_state
        return Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.0]])


def test_shutdown_calls_on_unload_weights_and_clears_loaded_flag() -> None:
    """ADR-0050: shutdown() releases weights via on_unload_weights + clears weights_loaded."""
    skill = _WeightTrackingSkill()
    skill.configure()
    assert skill.info.weights_loaded is True
    assert skill.load_calls == 1

    skill.shutdown()

    assert skill.state is RSkillState.FINALIZED
    assert skill.unload_calls == 1
    assert skill.info.weights_loaded is False


def test_shutdown_unload_is_idempotent() -> None:
    """ADR-0050: a second shutdown() must not re-run on_unload_weights."""
    skill = _WeightTrackingSkill()
    skill.configure()
    skill.shutdown()
    skill.shutdown()
    assert skill.unload_calls == 1


def test_shutdown_without_loaded_weights_skips_unload_hook() -> None:
    """ADR-0050: shutting down before configure() must not call on_unload_weights."""
    skill = _WeightTrackingSkill()
    skill.shutdown()
    assert skill.unload_calls == 0
    assert skill.state is RSkillState.FINALIZED


def test_on_unload_weights_default_is_noop() -> None:
    """ADR-0050: the base hook defaults to a no-op so existing skills need no change."""
    skill = _MinimalSkill()
    skill.configure()
    skill.shutdown()  # must not raise
    assert skill.info.weights_loaded is False
