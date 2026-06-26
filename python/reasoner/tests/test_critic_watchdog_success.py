"""Tests for the one-shot success fire added to CriticWatchdog (ADR-0074).

The reward-watcher must wake the reasoner the moment an attempt is over —
stuck (stall, existing path) OR likely-done (score >= threshold).  These
tests verify the new success-fire semantics without duplicating the stall
tests in ``test_critic_watchdog.py``; a short stall regression is included
so a revert of the stall path breaks here too.

Run (from the worktree root)::

    WT=$(pwd)
    MYPYPATH=$(ls -d $WT/python/*/src | tr '\n' :)
    PYTHONPATH=$MYPYPATH /home/allopart/workspace/openral/.venv/bin/python \\
        -m pytest python/reasoner/tests/test_critic_watchdog_success.py -v
"""

from __future__ import annotations

import pytest
from openral_core import CriticEvidence
from openral_reasoner import CriticWatchdog, CriticWatchdogGroup

# ── helpers ────────────────────────────────────────────────────────────────────


def _wd(
    *,
    threshold: float = 0.8,
    stall_patience: int = 3,
    min_delta: float = 0.0,
) -> CriticWatchdog:
    """Return a watchdog with a stable id for assertions."""
    return CriticWatchdog(
        critic_id="OpenRAL/rskill-robometer-4b",
        threshold=threshold,
        stall_patience=stall_patience,
        min_delta=min_delta,
    )


# ── success-fire: one-shot latch ───────────────────────────────────────────────


def test_success_fires_on_first_crossing() -> None:
    """The very first sample at or above threshold fires a CriticEvidence."""
    wd = _wd(threshold=0.8, stall_patience=5)
    evidence = wd.observe(0.85)
    assert isinstance(evidence, CriticEvidence)
    assert evidence.kind == "critic"
    assert evidence.critic_id == "OpenRAL/rskill-robometer-4b"
    assert evidence.score == pytest.approx(0.85)
    assert evidence.threshold == pytest.approx(0.8)


def test_success_fires_exactly_at_threshold() -> None:
    """score == threshold (boundary) fires once."""
    wd = _wd(threshold=0.8, stall_patience=5)
    assert isinstance(wd.observe(0.8), CriticEvidence)


def test_success_latches_after_first_fire() -> None:
    """Subsequent samples at or above threshold return None (one-shot)."""
    wd = _wd(threshold=0.8, stall_patience=5)
    assert isinstance(wd.observe(0.85), CriticEvidence)  # fires
    # Same score — latched.
    assert wd.observe(0.85) is None
    # Even higher score — still latched.
    assert wd.observe(0.99) is None
    assert wd.observe(1.00) is None


def test_success_latch_clears_after_score_dips_below_threshold() -> None:
    """Once the score drops below threshold and rises again, success fires once more."""
    wd = _wd(threshold=0.8, stall_patience=5)
    assert isinstance(wd.observe(0.85), CriticEvidence)  # fires
    assert wd.observe(0.85) is None  # latched
    # Dip below threshold clears the latch.
    assert wd.observe(0.50) is None
    # Re-cross threshold → fires again.
    assert isinstance(wd.observe(0.90), CriticEvidence)
    assert wd.observe(0.90) is None  # latched again


def test_success_fires_on_first_sample_no_prior_history() -> None:
    """Success fire does not require a prior observation to set a running best."""
    wd = _wd(threshold=0.8, stall_patience=5)
    # The very first sample ever can be a success.
    assert isinstance(wd.observe(1.0), CriticEvidence)


def test_success_fires_after_stall_recovery() -> None:
    """A stall that fires and latches, then recovers above threshold, fires success."""
    wd = _wd(threshold=0.8, stall_patience=2)
    assert wd.observe(0.3) is None  # stall 1
    assert isinstance(wd.observe(0.3), CriticEvidence)  # stall fires
    assert wd.observe(0.3) is None  # stall-latched
    # Now score crosses success threshold → success fires.
    assert isinstance(wd.observe(0.9), CriticEvidence)
    assert wd.observe(0.9) is None  # success-latched


# ── stall regression — success path must not break stall path ──────────────────


def test_stall_still_fires_after_stall_patience() -> None:
    """Existing stall behavior: fires once after stall_patience sub-threshold observations."""
    wd = _wd(threshold=0.8, stall_patience=3)
    assert wd.observe(0.4) is None  # stall 1
    assert wd.observe(0.4) is None  # stall 2
    evidence = wd.observe(0.4)  # stall 3 → fire
    assert isinstance(evidence, CriticEvidence)
    assert evidence.score == pytest.approx(0.4)
    assert evidence.threshold == pytest.approx(0.8)
    # Stall-latched; no spam.
    assert wd.observe(0.4) is None


def test_stall_latch_independent_of_success_latch() -> None:
    """A success fire does not accidentally clear the stall latch or vice-versa."""
    wd = _wd(threshold=0.8, stall_patience=2)
    # Accumulate two stalls → stall fires.
    assert wd.observe(0.3) is None
    assert isinstance(wd.observe(0.3), CriticEvidence)
    # Stall latch is set; success latch is NOT set (score never crossed threshold).
    assert wd.observe(0.3) is None  # stall-latched, returns None
    # Score crosses threshold — stall latch should also clear, and success fires once.
    assert isinstance(wd.observe(0.9), CriticEvidence)  # success fire
    assert wd.observe(0.9) is None  # success-latched


# ── progress below threshold — neither path fires ──────────────────────────────


def test_progress_below_threshold_no_fire() -> None:
    """A steadily climbing but sub-threshold series fires neither stall nor success."""
    wd = _wd(threshold=0.9, stall_patience=2)
    for score in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        assert wd.observe(score) is None, f"unexpected fire at score={score}"


def test_reset_clears_success_latch() -> None:
    """After reset(), success fires again on the next above-threshold sample."""
    wd = _wd(threshold=0.8, stall_patience=5)
    assert isinstance(wd.observe(0.9), CriticEvidence)  # fires
    assert wd.observe(0.9) is None  # latched
    wd.reset()
    # Reset must clear the success latch.
    assert isinstance(wd.observe(0.9), CriticEvidence)


# ── CriticWatchdogGroup success wiring ─────────────────────────────────────────


def test_group_fires_success_through_observe() -> None:
    """Group.observe propagates the success fire from the underlying watchdog."""
    g = CriticWatchdogGroup(stall_patience=5)
    evidence = g.observe(critic_id="robometer", score=0.9, threshold=0.8)
    assert isinstance(evidence, CriticEvidence)
    assert evidence.critic_id == "robometer"
    # Latched — next sample at same score returns None.
    assert g.observe(critic_id="robometer", score=0.9, threshold=0.8) is None


def test_group_success_and_stall_independent_per_critic() -> None:
    """Success fire on one critic does not affect a different critic's stall state."""
    g = CriticWatchdogGroup(stall_patience=2)
    # sarm succeeds immediately.
    assert isinstance(g.observe(critic_id="sarm", score=0.95, threshold=0.9), CriticEvidence)
    # robometer is stalling independently.
    assert g.observe(critic_id="robometer", score=0.3, threshold=0.8) is None  # stall 1
    assert isinstance(
        g.observe(critic_id="robometer", score=0.3, threshold=0.8), CriticEvidence
    )  # stall 2 → fire
    # sarm still latched — extra score sample returns None.
    assert g.observe(critic_id="sarm", score=0.95, threshold=0.9) is None
