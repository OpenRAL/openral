"""Three-tier evaluate_task_verdict tests (ADR-0074 Decision 5 + amendment).

Tests the auto-pass / vlm_check / attempts-ladder logic of the ADR-0074 gate.
The ADR-0074 amendment gates the band on the PROGRESS head (task closeness) — the
``success_threshold`` / ``check_floor`` bars were calibrated against progress, not
the compressed success head — and keeps ``success_now`` as a secondary
corroborating signal surfaced in the verdict text. Complements the broader
MissionState lifecycle tests in test_reasoner_mission.py; these focus on the
bands and the progress-vs-success gating.
"""

from __future__ import annotations

from openral_reasoner import evaluate_task_verdict

# ── Tier 1: auto-pass (progress_now >= success_threshold) ─────────────────────


def test_tier1_auto_pass_at_threshold() -> None:
    """Exact threshold value is an auto-pass (boundary is inclusive)."""
    action, verdict = evaluate_task_verdict(
        ok=True, progress_now=0.8, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "complete"
    assert verdict == "progress=0.80"


def test_tier1_auto_pass_above_threshold() -> None:
    """A high progress (0.91) auto-passes without needing VLM adjudication."""
    action, verdict = evaluate_task_verdict(
        ok=True, progress_now=0.91, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "complete"
    assert "0.91" in verdict


def test_tier1_auto_pass_ignores_attempt_count() -> None:
    """Auto-pass fires regardless of how many attempts were made."""
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.95, success_threshold=0.8, check_floor=0.5, attempts=5
    )
    assert action == "complete"


# ── Tier 2: vlm_check (check_floor <= progress_now < success_threshold) ───────


def test_tier2_vlm_check_in_middle_band() -> None:
    """progress=0.65 sits in [0.5, 0.8) → vlm_check."""
    action, verdict = evaluate_task_verdict(
        ok=True, progress_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"
    assert "0.65" in verdict
    assert "VLM adjudicates" in verdict


def test_tier2_vlm_check_at_floor_boundary() -> None:
    """Exact check_floor value falls into the ambiguous band (boundary is inclusive)."""
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.5, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"


def test_tier2_vlm_check_just_below_threshold() -> None:
    """One epsilon below success_threshold triggers vlm_check, not complete."""
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.799, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "vlm_check"


def test_tier2_vlm_check_ignores_attempt_count() -> None:
    """vlm_check fires regardless of attempts (the ladder is bypassed in tier 2)."""
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=10
    )
    assert action == "vlm_check"


# ── Tier 3: attempts ladder (progress_now < check_floor) ──────────────────────


def test_tier3_retry_below_floor() -> None:
    """progress=0.40 < check_floor → retry when attempts remain."""
    action, verdict = evaluate_task_verdict(
        ok=True,
        progress_now=0.40,
        success_threshold=0.8,
        check_floor=0.5,
        attempts=1,
        max_attempts=3,
    )
    assert action == "retry"
    assert "attempt 1/3" in verdict


def test_tier3_abandon_below_floor_exhausted() -> None:
    """progress=0.40 < check_floor → abandon when attempts exhausted."""
    action, verdict = evaluate_task_verdict(
        ok=True,
        progress_now=0.40,
        success_threshold=0.8,
        check_floor=0.5,
        attempts=3,
        max_attempts=3,
    )
    assert action == "abandon"
    assert "after 3 attempt(s)" in verdict


def test_tier3_just_below_floor_boundary() -> None:
    """One epsilon below check_floor → ladder, not vlm_check."""
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.499, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "retry"


# ── ok=False path: never complete / vlm_check ─────────────────────────────────


def test_ok_false_never_complete_even_high_score() -> None:
    """When the reward is unavailable, a high progress must not auto-pass."""
    action, _ = evaluate_task_verdict(
        ok=False, progress_now=0.99, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action not in ("complete", "vlm_check")


def test_ok_false_never_vlm_check_even_in_band() -> None:
    """When the reward is unavailable, a middle-band progress must not vlm_check."""
    action, _ = evaluate_task_verdict(
        ok=False, progress_now=0.65, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "retry"


def test_ok_false_retry_until_exhausted_then_abandon() -> None:
    """ok=False falls to the attempts ladder: retry then abandon."""
    assert (
        evaluate_task_verdict(
            ok=False, progress_now=0.0, success_threshold=0.8, check_floor=0.5, attempts=1
        )[0]
        == "retry"
    )
    assert (
        evaluate_task_verdict(
            ok=False, progress_now=0.0, success_threshold=0.8, check_floor=0.5, attempts=3
        )[0]
        == "abandon"
    )


# ── vlm_check degrade contract (the node bounds an unconfirmed ambiguous task) ─
# Tier 2 returns vlm_check regardless of attempts; the *node* adjudicates and, when
# the VLM cannot confirm completion, re-runs the verdict with ok=False to apply the
# attempts ladder. Without that, an ambiguous-band reward retries forever. These
# guard the exact composition the node uses so the bound can't silently regress.


def test_vlm_declined_in_band_degrades_to_retry_while_attempts_remain() -> None:
    # Ambiguous progress the VLM can't confirm, attempts not yet exhausted → retry.
    action, _ = evaluate_task_verdict(
        ok=False, progress_now=0.5, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert action == "retry"


def test_vlm_declined_in_band_abandons_once_attempts_exhausted() -> None:
    # Same ambiguous progress, attempts exhausted → abandon (not an infinite loop).
    action, _ = evaluate_task_verdict(
        ok=False, progress_now=0.5, success_threshold=0.8, check_floor=0.5, attempts=3
    )
    assert action == "abandon"


# ── ADR-0074 amendment: gate on PROGRESS, corroborate with SUCCESS ────────────
# These pin the amendment that fixed the libero_object sub-0.5 plateau: the band
# is gated on the progress head (which reaches ~0.85 on a real success), NOT the
# compressed success head (~0.56–0.79 even on a genuine success). success_now is
# kept only as a corroborating annotation.


def test_real_success_progress_auto_passes_despite_compressed_success() -> None:
    """The cached-rollout regime: a genuine success scores progress≈0.85 (auto-pass)
    while its success head sits at ≈0.63 — under the OLD success-gated logic this was
    a permanent vlm_check/ladder plateau; gating on progress now completes it."""
    action, verdict = evaluate_task_verdict(
        ok=True,
        progress_now=0.85,
        success_now=0.63,
        success_threshold=0.8,
        check_floor=0.5,
        attempts=1,
    )
    assert action == "complete"
    # Progress is primary in the verdict; success is the corroborating annotation.
    assert "progress=0.85" in verdict
    assert "success=0.63" in verdict


def test_gate_ignores_success_head_when_progress_is_low() -> None:
    """A high success head must NOT rescue a low-progress attempt (the band is the
    progress head's; success is advisory only)."""
    action, _ = evaluate_task_verdict(
        ok=True,
        progress_now=0.40,
        success_now=0.95,
        success_threshold=0.8,
        check_floor=0.5,
        attempts=1,
    )
    assert action == "retry"


def test_success_corroboration_optional_in_verdict_text() -> None:
    """Omitting success_now leaves the verdict text progress-only (no annotation)."""
    _, verdict = evaluate_task_verdict(
        ok=True, progress_now=0.85, success_threshold=0.8, check_floor=0.5, attempts=1
    )
    assert verdict == "progress=0.85"
    assert "success=" not in verdict
