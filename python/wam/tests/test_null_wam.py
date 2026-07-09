"""Real-schema plumbing tests for :class:`NullWorldModel`.

Constructs **real** ``WorldState`` / ``Action`` instances from
``openral_core`` (per CLAUDE.md §1.11 — no mocks).
"""

from __future__ import annotations

import pytest
from openral_core import Action, ControlMode, JointState, WorldState
from openral_wam import NullWorldModel, Rollout, WorldModel


def _world_state() -> WorldState:
    """Minimal real ``WorldState`` — no placeholders, no mocks."""
    return WorldState(
        stamp_ns=0,
        joint_state=JointState(name=["j1"], position=[0.0], stamp_ns=0),
    )


def _action_chunk(horizon: int = 4) -> Action:
    """Minimal real ``Action`` chunk."""
    return Action(control_mode=ControlMode.JOINT_POSITION, horizon=horizon)


def test_null_wam_replays_input_state() -> None:
    wam = NullWorldModel(max_horizon=8)
    ws = _world_state()

    r = wam.rollout(ws, _action_chunk(horizon=4), horizon=4)

    assert isinstance(r, Rollout)
    assert r.horizon == 4
    assert len(r.predicted_states) == 4
    assert all(s == ws for s in r.predicted_states)
    assert r.predicted_rewards is None
    assert r.latency_ms == 0.0
    assert r.confidence == 1.0


def test_null_wam_satisfies_protocol() -> None:
    wam = NullWorldModel()
    assert isinstance(wam, WorldModel)
    assert wam.max_horizon == 16


def test_null_wam_rejects_zero_horizon() -> None:
    wam = NullWorldModel(max_horizon=8)
    with pytest.raises(ValueError, match="horizon must satisfy"):
        wam.rollout(_world_state(), _action_chunk(horizon=1), horizon=0)


def test_null_wam_rejects_horizon_above_max() -> None:
    wam = NullWorldModel(max_horizon=4)
    with pytest.raises(ValueError, match="horizon must satisfy"):
        wam.rollout(_world_state(), _action_chunk(horizon=8), horizon=8)


def test_null_wam_rejects_zero_max_horizon() -> None:
    with pytest.raises(ValueError, match="max_horizon must be > 0"):
        NullWorldModel(max_horizon=0)


def test_rollout_round_trips_through_json() -> None:
    """Schema round-trip per CLAUDE.md §5.4 (no mocks; real Pydantic v2)."""
    wam = NullWorldModel(max_horizon=4)
    original = wam.rollout(_world_state(), _action_chunk(horizon=2), horizon=2)
    restored = Rollout.model_validate_json(original.model_dump_json())
    assert restored == original
