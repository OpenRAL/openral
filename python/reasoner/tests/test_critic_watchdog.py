"""Unit tests for the Tier-C critic progress-stall watchdog (audit P1 R3).

These tests exercise the **real** :class:`openral_core.CriticEvidence`
schema the watchdog emits (CLAUDE.md §1.11 — no mocks). The watchdog is
pure logic and import-safe, so the suite runs without ROS or a GPU.
"""

from __future__ import annotations

import pytest
from openral_core import CriticEvidence
from openral_reasoner import CriticWatchdog


def _watchdog(
    *,
    threshold: float = 0.8,
    stall_patience: int = 3,
    min_delta: float = 0.0,
) -> CriticWatchdog:
    """A watchdog with a stable Robometer-style id for assertions."""
    return CriticWatchdog(
        critic_id="OpenRAL/rskill-robometer-4b",
        threshold=threshold,
        stall_patience=stall_patience,
        min_delta=min_delta,
    )


def test_rejects_non_positive_stall_patience() -> None:
    with pytest.raises(ValueError):
        _watchdog(stall_patience=0)
    with pytest.raises(ValueError):
        _watchdog(stall_patience=-1)


def test_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError):
        _watchdog(min_delta=-0.01)


def test_no_fire_while_above_threshold() -> None:
    wd = _watchdog(threshold=0.5, stall_patience=2)
    # Steady, flat, but above threshold → never a stall.
    for _ in range(10):
        assert wd.observe(0.7) is None


def test_no_fire_while_improving_below_threshold() -> None:
    wd = _watchdog(threshold=0.9, stall_patience=2)
    # Below threshold the whole time, but each step improves → progress.
    for score in (0.1, 0.2, 0.3, 0.4, 0.5):
        assert wd.observe(score) is None


def test_fires_once_after_stall_patience() -> None:
    wd = _watchdog(threshold=0.8, stall_patience=3)
    # First below-threshold obs sets the running best (counts as a stall: 1).
    assert wd.observe(0.4) is None  # stall 1
    assert wd.observe(0.4) is None  # stall 2
    evidence = wd.observe(0.4)  # stall 3 → fire
    assert isinstance(evidence, CriticEvidence)
    assert evidence.kind == "critic"
    assert evidence.critic_id == "OpenRAL/rskill-robometer-4b"
    assert evidence.score == pytest.approx(0.4)
    assert evidence.threshold == pytest.approx(0.8)


def test_latches_after_firing() -> None:
    wd = _watchdog(threshold=0.8, stall_patience=2)
    assert wd.observe(0.3) is None
    assert isinstance(wd.observe(0.3), CriticEvidence)  # fires
    # Still stalled — must not spam the bus.
    assert wd.observe(0.3) is None
    assert wd.observe(0.2) is None
    assert wd.observe(0.3) is None


def test_recovery_above_threshold_unlatches_and_allows_refire() -> None:
    wd = _watchdog(threshold=0.8, stall_patience=2)
    assert wd.observe(0.3) is None
    assert isinstance(wd.observe(0.3), CriticEvidence)  # fire
    assert wd.observe(0.3) is None  # latched
    # Recovery above threshold clears the latch and the counter.
    assert wd.observe(0.95) is None
    # A fresh stall must be able to fire again.
    assert wd.observe(0.3) is None
    assert isinstance(wd.observe(0.3), CriticEvidence)  # fires again


def test_improvement_resets_stall_counter() -> None:
    wd = _watchdog(threshold=0.9, stall_patience=3)
    assert wd.observe(0.4) is None  # stall 1, best=0.4
    assert wd.observe(0.4) is None  # stall 2
    assert wd.observe(0.5) is None  # improvement → best=0.5, counter reset
    assert wd.observe(0.5) is None  # stall 1
    assert wd.observe(0.5) is None  # stall 2
    # Only on the 3rd consecutive stall after the reset does it fire.
    assert isinstance(wd.observe(0.5), CriticEvidence)


def test_min_delta_boundary() -> None:
    wd = _watchdog(threshold=0.9, stall_patience=2, min_delta=0.05)
    assert wd.observe(0.40) is None  # stall 1, best=0.40
    # +0.05 is NOT strictly greater than min_delta=0.05 → still a stall.
    fired = wd.observe(0.45)  # stall 2 → fire
    assert isinstance(fired, CriticEvidence)


def test_min_delta_genuine_improvement_resets() -> None:
    wd = _watchdog(threshold=0.9, stall_patience=2, min_delta=0.05)
    assert wd.observe(0.40) is None  # stall 1, best=0.40
    # +0.06 > min_delta → genuine progress, counter resets.
    assert wd.observe(0.46) is None
    assert wd.observe(0.46) is None  # stall 1 (no improvement on new best)
    assert isinstance(wd.observe(0.46), CriticEvidence)  # stall 2 → fire


def test_reset_clears_state() -> None:
    wd = _watchdog(threshold=0.8, stall_patience=2)
    assert wd.observe(0.3) is None  # stall 1
    wd.reset()
    # Counter and best are cleared; a single stall must not fire.
    assert wd.observe(0.3) is None  # stall 1 again
    assert isinstance(wd.observe(0.3), CriticEvidence)  # stall 2 → fire


def test_reset_clears_latch() -> None:
    wd = _watchdog(threshold=0.8, stall_patience=1)
    assert isinstance(wd.observe(0.3), CriticEvidence)  # fire (patience 1)
    assert wd.observe(0.3) is None  # latched
    wd.reset()
    # After reset the latch is gone → it can fire again immediately.
    assert isinstance(wd.observe(0.3), CriticEvidence)
