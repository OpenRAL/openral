"""Unit tests for :class:`TickResult` v2 (ADR-0010 amendment 1).

The amendment added five optional sim-only fields
(``step_idx``, ``episode_idx``, ``reward``, ``terminated``, ``truncated``)
that default to ``None`` so a hardware tick serialises byte-identically
to v1 JSON under ``exclude_none=True``. These tests pin that promise
and exercise the new fields' validation.

CLAUDE.md §1.11 — real schemas; no mocks.
"""

from __future__ import annotations

import pytest
from openral_core import TickResult
from pydantic import ValidationError

# ── v1 backward compatibility ────────────────────────────────────────────────


def test_v1_shape_loads_unchanged() -> None:
    """A v1 hardware-shaped JSON loads cleanly and leaves new fields at None."""
    v1_json = {
        "stamp_ns": 123_000_000_000,
        "tick_idx": 0,
        "sensors_ms": 1.0,
        "world_state_ms": 0.5,
        "inference_ms": 12.0,
        "safety_ms": 0.3,
        "hal_ms": 0.7,
        "tick_ms": 33.0,
        "chunk_index": 0,
        "safety_violations": [],
        "action_applied": True,
    }
    tr = TickResult.model_validate(v1_json)
    assert tr.step_idx is None
    assert tr.episode_idx is None
    assert tr.reward is None
    assert tr.terminated is None
    assert tr.truncated is None
    # trace_context is also part of the optional-additive surface — v1
    # JSON omitting it must still validate, and the round-trip under
    # exclude_none drops it from the output.
    assert tr.trace_context is None
    dumped = tr.model_dump(exclude_none=True)
    assert "trace_context" not in dumped


def test_trace_context_round_trips() -> None:
    """A populated ``trace_context`` round-trips through JSON."""
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    tr = TickResult(stamp_ns=1, tick_idx=0, tick_ms=1.0, trace_context=tp)
    raw = tr.model_dump_json()
    loaded = TickResult.model_validate_json(raw)
    assert loaded.trace_context == tp


def test_hardware_tick_excludes_sim_fields_under_exclude_none() -> None:
    """``model_dump(exclude_none=True)`` drops the v2 fields when unset.

    This keeps the hardware OTel / trace payload byte-identical with the
    pre-bump shape so existing consumers (dashboards, JSON dumps) stay
    valid.
    """
    tr = TickResult(stamp_ns=1, tick_idx=0, tick_ms=33.0)
    dumped = tr.model_dump(exclude_none=True)
    for field in ("step_idx", "episode_idx", "reward", "terminated", "truncated"):
        assert field not in dumped, f"v2 field {field} leaked into hardware dump"


# ── Sim ticks populate the new fields ────────────────────────────────────────


def test_sim_step_tick_round_trip() -> None:
    """A SimRunner step-tick: all v2 fields populated; round-trips through JSON."""
    tr = TickResult(
        stamp_ns=1_000,
        tick_idx=5,
        inference_ms=12.0,
        tick_ms=15.0,
        action_applied=True,
        step_idx=4,
        episode_idx=0,
        reward=0.25,
        terminated=False,
        truncated=False,
    )
    raw = tr.model_dump_json()
    loaded = TickResult.model_validate_json(raw)
    assert loaded == tr
    assert loaded.step_idx == 4
    assert loaded.episode_idx == 0
    assert loaded.reward == 0.25
    assert loaded.terminated is False
    assert loaded.truncated is False


def test_sim_reset_tick_carries_only_episode_idx() -> None:
    """Reset-tick semantics: action_applied=False, step_idx=None, episode_idx set."""
    tr = TickResult(
        stamp_ns=1,
        tick_idx=0,
        tick_ms=0.5,
        action_applied=False,
        episode_idx=0,
    )
    assert tr.step_idx is None
    assert tr.reward is None
    assert tr.terminated is None
    assert tr.truncated is None
    assert tr.episode_idx == 0


# ── Validation ──────────────────────────────────────────────────────────────


def test_step_idx_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        TickResult(stamp_ns=1, tick_idx=0, tick_ms=1.0, step_idx=-1)


def test_episode_idx_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        TickResult(stamp_ns=1, tick_idx=0, tick_ms=1.0, episode_idx=-2)


def test_extra_fields_still_forbidden() -> None:
    """``extra='forbid'`` survives the v2 bump."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TickResult.model_validate(
            {
                "stamp_ns": 1,
                "tick_idx": 0,
                "tick_ms": 1.0,
                "future_field": 42,
            }
        )
