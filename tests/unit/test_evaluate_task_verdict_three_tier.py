"""Three-tier evaluate_task_verdict tests (ADR-0074 Decision 5).

Tests the auto-pass / vlm_check / attempts-ladder logic introduced by the
ADR-0074 three-tier gate. Complements the existing mission tests in
test_reasoner_mission.py — those cover the broader MissionState lifecycle;
these focus specifically on the new bands of evaluate_task_verdict.
"""

from __future__ import annotations

from openral_reasoner import evaluate_task_verdict

# ── Tier 1: auto-pass (success_now >= success_threshold) ──────────────────────


def test_tier1_auto_pass_at_threshold() -> None:
    """Exact threshold value is an auto-pass (boundary is inclusive)."""
    action, verdict = evaluate_task_verdict(
        ok=True, success_now=0.8, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "complete"
    assert verdict == "success=0.80"


def test_tier1_auto_pass_above_threshold() -> None:
    """A high score (0.91) auto-passes without needing VLM adjudication."""
    action, verdict = evaluate_task_verdict(
        ok=True, success_now=0.91, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "complete"
    assert "0.91" in verdict


def test_tier1_auto_pass_ignores_attempt_count() -> None:
    """Auto-pass fires regardless of how many attempts were made."""
    action, _ = evaluate_task_verdict(
        ok=True, success_now=0.95, success_threshold=0.8, check_floor=0.5, attempts=5
    )
    assert action == "complete"


# ── Tier 2: vlm_check (check_floor <= success_now < success_threshold) ────────


def test_tier2_vlm_check_in_middle_band() -> None:
    """score=0.65 sits in [0.5, 0.8) → vlm_check."""
    action, verdict = evaluate_task_verdict(
        ok=True, success_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"
    assert "0.65" in verdict
    assert "VLM adjudicates" in verdict


def test_tier2_vlm_check_at_floor_boundary() -> None:
    """Exact check_floor value falls into the ambiguous band (boundary is inclusive)."""
    action, _ = evaluate_task_verdict(
        ok=True, success_now=0.5, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"


def test_tier2_vlm_check_just_below_threshold() -> None:
    """One epsilon below success_threshold triggers vlm_check, not complete."""
    action, _ = evaluate_task_verdict(
        ok=True, success_now=0.799, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"


def test_tier2_vlm_check_ignores_attempt_count() -> None:
    """vlm_check fires regardless of attempts (the ladder is bypassed in tier 2)."""
    action, _ = evaluate_task_verdict(
        ok=True, success_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=10
    )
    assert action == "vlm_check"


# ── Tier 3: attempts ladder (success_now < check_floor) ───────────────────────


def test_tier3_retry_below_floor() -> None:
    """score=0.40 < check_floor → retry when attempts remain."""
    action, verdict = evaluate_task_verdict(
        ok=True, success_now=0.40, success_threshold=0.8,
        check_floor=0.5, attempts=1, max_attempts=3,
    )
    assert action == "retry"
    assert "attempt 1/3" in verdict


def test_tier3_abandon_below_floor_exhausted() -> None:
    """score=0.40 < check_floor → abandon when attempts exhausted."""
    action, verdict = evaluate_task_verdict(
        ok=True, success_now=0.40, success_threshold=0.8,
        check_floor=0.5, attempts=3, max_attempts=3,
    )
    assert action == "abandon"
    assert "after 3 attempt(s)" in verdict


def test_tier3_just_below_floor_boundary() -> None:
    """One epsilon below check_floor → ladder, not vlm_check."""
    action, _ = evaluate_task_verdict(
        ok=True, success_now=0.499, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "retry"


# ── ok=False path: never complete / vlm_check ─────────────────────────────────


def test_ok_false_never_complete_even_high_score() -> None:
    """When the reward is unavailable, a high score must not auto-pass."""
    action, _ = evaluate_task_verdict(
        ok=False, success_now=0.99, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action not in ("complete", "vlm_check")


def test_ok_false_never_vlm_check_even_in_band() -> None:
    """When the reward is unavailable, a middle-band score must not vlm_check."""
    action, _ = evaluate_task_verdict(
        ok=False, success_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "retry"


def test_ok_false_retry_until_exhausted_then_abandon() -> None:
    """ok=False falls to the attempts ladder: retry then abandon."""
    assert (
        evaluate_task_verdict(
            ok=False, success_now=0.0, success_threshold=0.8, check_floor=0.5, attempts=1
        )[0]
        == "retry"
    )
    assert (
        evaluate_task_verdict(
            ok=False, success_now=0.0, success_threshold=0.8, check_floor=0.5, attempts=3
        )[0]
        == "abandon"
    )
