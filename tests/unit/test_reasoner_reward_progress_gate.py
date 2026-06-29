"""ADR-0074 amendment — progress-gated verdict, two-head context, locate budget.

Pure-Python (no rclpy / no openral_msgs): exercises the reasoner-internal
``mission`` + ``context`` modules that the libero_object plateau fix lives in.

* the ``## REWARD`` context section renders BOTH heads, distinctly labelled, so
  the LLM uses progress for persist-vs-replan and success for done-ness;
* the per-task :class:`TaskLocateBudget` abandons the active subtask after N
  locate cycles without a skill dispatch, carrying a specific reason into the
  mission ledger so the next pick proceeds.
"""

from __future__ import annotations

from openral_reasoner.context import ContextRenderer, RewardStateRecord
from openral_reasoner.mission import (
    DEFAULT_MAX_TASK_LOCATE_ATTEMPTS,
    MissionState,
    TaskLocateBudget,
)

# ── ## REWARD: both heads, distinctly labelled ────────────────────────────────


def test_reward_section_renders_both_heads_labelled() -> None:
    """progress is labelled 'closeness', success 'done-confidence' — never blurred."""
    r = ContextRenderer()
    r.set_reward_state(
        RewardStateRecord(
            progress=0.81,
            success=0.45,
            progress_trend=0.04,
            success_trend=0.01,
            task="pick up the alphabet soup and place it in the basket",
            stamp_ns=7,
        )
    )
    out = r.render(world_state=None)
    assert "## REWARD" in out
    # Both heads present with their distinct meanings.
    assert "progress=0.81 (closeness" in out
    assert "success=0.45 (done-confidence" in out
    # Trends carried so the LLM can read persist (+) vs replan (flat/-).
    assert "trend +0.040/frame" in out
    # Explicit guidance that progress drives persist-vs-replan.
    assert "Gate on progress" in out


def test_reward_section_absent_until_set() -> None:
    """No reward assessment yet → no ## REWARD section (keeps the context tight)."""
    assert "## REWARD" not in ContextRenderer().render(world_state=None)


def test_set_reward_state_is_an_event_bumps_seq() -> None:
    """A fresh assessment wakes an otherwise-idle heartbeat (seq bumps)."""
    r = ContextRenderer()
    seq0 = r.seq
    r.set_reward_state(
        RewardStateRecord(
            progress=0.5, success=0.5, progress_trend=0.0, success_trend=0.0,
            task="t", stamp_ns=0,
        )
    )
    assert r.seq > seq0


def test_set_reward_state_none_clears_section() -> None:
    r = ContextRenderer()
    r.set_reward_state(
        RewardStateRecord(
            progress=0.5, success=0.5, progress_trend=0.0, success_trend=0.0,
            task="t", stamp_ns=0,
        )
    )
    r.set_reward_state(None)
    assert "## REWARD" not in r.render(world_state=None)


# ── per-task locate budget → abandon with a displayed reason ───────────────────


def test_locate_budget_default_is_three() -> None:
    assert DEFAULT_MAX_TASK_LOCATE_ATTEMPTS == 3


def test_locate_budget_charges_then_exhausts() -> None:
    """N=3: three locate cycles are allowed; the 4th exhausts the budget."""
    b = TaskLocateBudget(max_attempts=3)
    assert [b.charge("t1") for _ in range(4)] == [False, False, False, True]


def test_locate_budget_resets_on_task_change() -> None:
    """A different active task starts a fresh budget (advancing the queue resets)."""
    b = TaskLocateBudget(max_attempts=3)
    for _ in range(3):
        b.charge("t1")
    assert b.charge("t1") is True  # t1 exhausted
    assert b.charge("t2") is False  # t2 is fresh
    assert b.count == 1


def test_locate_budget_explicit_reset() -> None:
    """A skill dispatch resets the budget so locate cycles only count pre-dispatch."""
    b = TaskLocateBudget(max_attempts=3)
    for _ in range(3):
        b.charge("t1")
    b.reset()
    assert b.charge("t1") is False
    assert b.count == 1


def test_locate_budget_reason_names_object_and_count() -> None:
    """The abandonment reason is specific (the goal noun + the attempt count)."""
    b = TaskLocateBudget(max_attempts=3)
    for _ in range(4):
        b.charge("t1")  # 4th exhausts
    reason = b.reason("teapot")
    assert "'teapot'" in reason
    assert "3 locate attempts" in reason  # count-1 = the cycles actually spent
    assert "without a skill dispatch" in reason


def test_locate_budget_abandon_surfaces_reason_in_ledger() -> None:
    """The reason becomes the abandoned task's ✗ ledger verdict so the NEXT pick
    knows why and proceeds (the node calls advance_mission(done=False, reason))."""
    mission = MissionState(["find the teapot", "find the cup"])
    budget = TaskLocateBudget(max_attempts=3)
    active = mission.active()
    assert active is not None and active.task_id == "t1"

    # Simulate the node's locate-loop: charge until exhausted with no dispatch.
    abandoned = False
    for _ in range(4):
        if budget.charge(active.task_id):
            abandoned = True
            break
    assert abandoned
    reason = budget.reason("teapot")

    nxt = mission.abandon_active(reason)
    assert nxt is not None and nxt.text == "find the cup"

    ledger = mission.render()
    # The abandoned task renders ✗ with the specific reason as its verdict, and the
    # next task is now active (▶) — the next reasoner pick sees both.
    assert "✗ t1: find the teapot" in ledger
    assert "could not confirm 'teapot' in view" in ledger
    assert "▶ t2: find the cup" in ledger
