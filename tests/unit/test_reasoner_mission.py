"""Unit tests for :mod:`openral_reasoner.mission` (ADR-0073 §1).

The deterministic mission queue that fixes the multi-task deploy gap: an
operator goal carrying several ordered subtasks is split, sequenced one-active
at a time, and advanced only on an explicit verdict — instead of being handed to
the LLM as one opaque string and forgotten on the pull-once prompt drain.
"""

from __future__ import annotations

import pytest
from openral_reasoner import (
    MissionState,
    TaskState,
    evaluate_task_verdict,
    split_mission,
)


# ── split_mission ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The deploy CLI joins DeployScene.tasks with " | ".
        (
            "pick the black bowl from the stove | pick the butter from the cabinet",
            ["pick the black bowl from the stove", "pick the butter from the cabinet"],
        ),
        # The natural-language "…, then …" an operator types (the live test prompt).
        (
            "stack all the bowls and place them in the drawer, then put the plate on the heating box",
            [
                "stack all the bowls and place them in the drawer",
                "put the plate on the heating box",
            ],
        ),
        # " then " without a comma also separates.
        ("open the drawer then close it", ["open the drawer", "close it"]),
        # A single task is a one-element list.
        ("just pick the cup", ["just pick the cup"]),
        # Empty / whitespace yields no tasks.
        ("", []),
        ("   ", []),
        # We do NOT split on a bare "and" — one action keeps its "and".
        ("pick the bowl and place it", ["pick the bowl and place it"]),
        # Mixed separators + stray whitespace are normalised.
        ("  a  |  b , then c ", ["a", "b", "c"]),
    ],
)
def test_split_mission(text: str, expected: list[str]) -> None:
    assert split_mission(text) == expected


def test_split_mission_is_case_insensitive_on_then() -> None:
    assert split_mission("do A Then do B") == ["do A", "do B"]


# ── MissionState construction + activation ───────────────────────────────────


def test_first_task_is_active_rest_pending() -> None:
    m = MissionState.from_prompt("a | b | c")
    statuses = [(t.task_id, t.status) for t in m.tasks]
    assert statuses == [("t1", "active"), ("t2", "pending"), ("t3", "pending")]
    assert m.active().text == "a"
    assert len(m) == 3
    assert not m.is_complete()


def test_empty_mission() -> None:
    m = MissionState.from_prompt("   ")
    assert m.is_empty()
    assert m.active() is None
    # An empty mission is not "complete" — there was nothing to do.
    assert not m.is_complete()


# ── advancement: complete / abandon walk the queue in order ──────────────────


def test_complete_active_advances_to_next() -> None:
    m = MissionState.from_prompt("a | b")
    nxt = m.complete_active("success=0.91")
    assert nxt is not None and nxt.task_id == "t2" and nxt.status == "active"
    # t1 is terminal with its verdict recorded.
    t1 = m.tasks[0]
    assert t1.status == "done" and t1.last_verdict == "success=0.91"
    # active is now t2; mission not yet complete.
    assert m.active().task_id == "t2"
    assert not m.is_complete()


def test_completing_last_task_finishes_mission() -> None:
    m = MissionState.from_prompt("only task")
    assert m.complete_active("success=0.95") is None
    assert m.active() is None
    assert m.is_complete()


def test_abandon_active_advances_and_is_terminal() -> None:
    m = MissionState.from_prompt("hard task | easy task")
    nxt = m.abandon_active("ladder exhausted: stalled@0.73")
    assert nxt is not None and nxt.task_id == "t2"
    t1 = m.tasks[0]
    assert t1.status == "abandoned" and t1.last_verdict.startswith("ladder exhausted")
    # Abandoned is terminal — it is NOT silently re-queued.
    assert m.active().task_id == "t2"


def test_mission_complete_when_all_terminal_mixed() -> None:
    m = MissionState.from_prompt("a | b")
    m.abandon_active("unverifiable")  # t1 abandoned, t2 active
    assert m.complete_active("success=0.9") is None  # t2 done, none left
    assert m.is_complete()
    assert [t.status for t in m.tasks] == ["abandoned", "done"]


# ── attempts + verifying transition ──────────────────────────────────────────


def test_record_attempt_increments_and_tracks_ids() -> None:
    m = MissionState.from_prompt("a | b")
    m.record_attempt(rskill_id="OpenRAL/rskill-smolvla-libero", trace_id="abc123")
    m.record_attempt(rskill_id="OpenRAL/rskill-smolvla-libero")
    t1 = m.active()
    assert t1.attempts == 2
    assert t1.last_rskill_id == "OpenRAL/rskill-smolvla-libero"
    assert t1.last_trace_id == "abc123"  # preserved; second call passed no trace


def test_mark_verifying_keeps_task_active_for_active_lookup() -> None:
    m = MissionState.from_prompt("a | b")
    m.mark_verifying()
    assert m.tasks[0].status == "verifying"
    # `active()` still returns it while verifying (gating is in progress).
    assert m.active().task_id == "t1"


def test_record_attempt_and_complete_on_finished_mission_are_noops() -> None:
    m = MissionState.from_prompt("a")
    m.complete_active("success=0.9")  # mission finished
    # No active task: these must not raise and must not resurrect anything.
    m.record_attempt(rskill_id="x")
    m.mark_verifying()
    assert m.complete_active("again") is None
    assert m.abandon_active("again") is None
    assert m.is_complete()


# ── rendering (the ## MISSION ledger) ────────────────────────────────────────


def test_render_shows_active_done_and_pending() -> None:
    m = MissionState.from_prompt("a | b | c")
    m.complete_active("success=0.9")  # t1 done, t2 active, t3 pending
    m.record_attempt(rskill_id="r")
    out = m.render()
    assert "✓ t1: a" in out
    assert "▶ t2: b" in out
    assert "attempts=1" in out
    assert "1 pending task(s)" in out


def test_render_empty_mission() -> None:
    assert MissionState.from_prompt("").render() == "(no mission)"


def test_taskstate_defaults() -> None:
    t = TaskState(task_id="t1", text="x")
    assert t.status == "pending" and t.attempts == 0
    assert t.last_rskill_id is None and t.last_verdict is None


# ── evaluate_task_verdict (ADR-0073 §2 — the reward gate decision) ───────────


def test_verdict_complete_on_success() -> None:
    action, verdict = evaluate_task_verdict(
        ok=True, succeeded=True, success_now=0.91, attempts=1
    )
    assert action == "complete"
    assert verdict == "success=0.91"


def test_verdict_retry_when_below_threshold_and_attempts_remain() -> None:
    action, verdict = evaluate_task_verdict(
        ok=True, succeeded=False, success_now=0.40, attempts=1, max_attempts=3
    )
    assert action == "retry"
    assert "attempt 1/3" in verdict


def test_verdict_abandon_when_attempts_exhausted() -> None:
    action, verdict = evaluate_task_verdict(
        ok=True, succeeded=False, success_now=0.40, attempts=3, max_attempts=3
    )
    assert action == "abandon"
    assert "after 3 attempt(s)" in verdict


def test_verdict_never_completes_without_succeeded_flag() -> None:
    # Even with a high raw score, completion requires the reward monitor's own
    # succeeded flag (success_now >= its threshold). No fake success.
    action, _ = evaluate_task_verdict(
        ok=True, succeeded=False, success_now=0.79, attempts=1
    )
    assert action == "retry"


def test_verdict_not_ok_falls_through_to_retry_then_abandon() -> None:
    # ok=False (stale/errored reward) is treated as "not verified": retry until
    # the attempt cap, then abandon — never an accidental complete.
    assert evaluate_task_verdict(ok=False, succeeded=False, success_now=0.0, attempts=1)[0] == "retry"
    assert evaluate_task_verdict(ok=False, succeeded=False, success_now=0.0, attempts=3)[0] == "abandon"
