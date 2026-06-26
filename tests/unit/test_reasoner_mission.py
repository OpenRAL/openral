"""Unit tests for :mod:`openral_reasoner.mission` (ADR-0073 §1).

The deterministic mission queue that fixes the multi-task deploy gap: an
operator goal carrying several ordered subtasks is split, sequenced one-active
at a time, and advanced only on an explicit verdict — instead of being handed to
the LLM as one opaque string and forgotten on the pull-once prompt drain.
"""

from __future__ import annotations

from openral_reasoner import (
    MissionState,
    TaskState,
    evaluate_task_verdict,
)
from openral_reasoner.mission import DEFAULT_MAX_SUBDIVIDE_DEPTH

# ── MissionState.from_prompt — single-task seeding (ADR-0073 amendment) ──────


def test_from_prompt_seeds_a_single_task_verbatim() -> None:
    """The operator goal becomes ONE task; the LLM (not a regex) decomposes it."""
    m = MissionState.from_prompt("pick the milk, then the ketchup | and the soup")
    assert len(m) == 1
    assert m.tasks[0].text == "pick the milk, then the ketchup | and the soup"
    assert m.tasks[0].status == "active"


def test_from_prompt_empty_is_empty_mission() -> None:
    assert MissionState.from_prompt("   ").is_empty()


# ── MissionState construction + activation ───────────────────────────────────


def test_first_task_is_active_rest_pending() -> None:
    m = MissionState(["a", "b", "c"])
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
    m = MissionState(["a", "b"])
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
    m = MissionState(["hard task", "easy task"])
    nxt = m.abandon_active("ladder exhausted: stalled@0.73")
    assert nxt is not None and nxt.task_id == "t2"
    t1 = m.tasks[0]
    assert t1.status == "abandoned" and t1.last_verdict.startswith("ladder exhausted")
    # Abandoned is terminal — it is NOT silently re-queued.
    assert m.active().task_id == "t2"


def test_mission_complete_when_all_terminal_mixed() -> None:
    m = MissionState(["a", "b"])
    m.abandon_active("unverifiable")  # t1 abandoned, t2 active
    assert m.complete_active("success=0.9") is None  # t2 done, none left
    assert m.is_complete()
    assert [t.status for t in m.tasks] == ["abandoned", "done"]


# ── attempts + verifying transition ──────────────────────────────────────────


def test_record_attempt_increments_and_tracks_ids() -> None:
    m = MissionState(["a", "b"])
    m.record_attempt(rskill_id="OpenRAL/rskill-smolvla-libero", trace_id="abc123")
    m.record_attempt(rskill_id="OpenRAL/rskill-smolvla-libero")
    t1 = m.active()
    assert t1.attempts == 2
    assert t1.last_rskill_id == "OpenRAL/rskill-smolvla-libero"
    assert t1.last_trace_id == "abc123"  # preserved; second call passed no trace


def test_mark_verifying_keeps_task_active_for_active_lookup() -> None:
    m = MissionState(["a", "b"])
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
    m = MissionState(["a", "b", "c"])
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
    action, verdict = evaluate_task_verdict(ok=True, succeeded=True, success_now=0.91, attempts=1)
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
    action, _ = evaluate_task_verdict(ok=True, succeeded=False, success_now=0.79, attempts=1)
    assert action == "retry"


def test_verdict_not_ok_falls_through_to_retry_then_abandon() -> None:
    # ok=False (stale/errored reward) is treated as "not verified": retry until
    # the attempt cap, then abandon — never an accidental complete.
    assert (
        evaluate_task_verdict(ok=False, succeeded=False, success_now=0.0, attempts=1)[0] == "retry"
    )
    assert (
        evaluate_task_verdict(ok=False, succeeded=False, success_now=0.0, attempts=3)[0]
        == "abandon"
    )


# ── subdivide_active (ADR-0073 amendment / #123 — flat-splice subdivision) ────


def test_subdivide_active_splices_children_in_place() -> None:
    m = MissionState(["tidy the kitchen", "wipe the table"])
    child = m.subdivide_active(["clear the counter", "load the dishwasher"])
    assert child is not None
    # The blocked parent (t1) is *replaced* by its children, the pending tail (t2)
    # keeps its order — the queue stays flat (Option 1).
    assert [t.task_id for t in m.tasks] == ["t1.1", "t1.2", "t2"]
    # First child is active at depth+1; its siblings/tail stay pending.
    assert (child.task_id, child.text, child.depth, child.status) == (
        "t1.1",
        "clear the counter",
        1,
        "active",
    )
    assert [t.status for t in m.tasks] == ["active", "pending", "pending"]


def test_subdivide_then_advance_walks_children_then_tail() -> None:
    m = MissionState(["task one", "task two"])
    m.subdivide_active(["one-a", "one-b"])
    assert m.active().task_id == "t1.1"
    assert m.complete_active("ok").task_id == "t1.2"  # next child
    assert m.complete_active("ok").task_id == "t2"  # then the original tail
    assert m.complete_active("ok") is None
    assert m.is_complete()


def test_subdivide_respects_depth_bound_then_refuses() -> None:
    m = MissionState.from_prompt("root task")
    # depth 0 → 1 → 2 allowed; at DEFAULT_MAX_SUBDIVIDE_DEPTH (2) it is refused.
    assert m.subdivide_active(["a", "b"]).depth == 1
    assert m.subdivide_active(["c", "d"]).depth == DEFAULT_MAX_SUBDIVIDE_DEPTH
    assert m.subdivide_active(["e"]) is None  # would be depth 3 — refused → caller hands off
    # The refused call must not mutate the queue (the depth-1 sibling t1.2 trails).
    assert [t.task_id for t in m.tasks] == ["t1.1.1", "t1.1.2", "t1.2"]


def test_subdivide_custom_max_depth() -> None:
    m = MissionState.from_prompt("root")
    assert m.subdivide_active(["a"], max_depth=1).depth == 1
    assert m.subdivide_active(["b"], max_depth=1) is None  # depth 1 >= 1


def test_subdivide_empty_subtasks_is_noop() -> None:
    m = MissionState.from_prompt("only task")
    assert m.subdivide_active([]) is None
    assert m.subdivide_active(["  ", ""]) is None  # whitespace trims to empty
    assert [t.task_id for t in m.tasks] == ["t1"]
    assert m.active().task_id == "t1"


def test_subdivide_with_no_active_task_is_noop() -> None:
    m = MissionState.from_prompt("one task")
    m.complete_active("done")  # mission finished, nothing active
    assert m.subdivide_active(["a", "b"]) is None


def test_subdivide_render_indents_children() -> None:
    m = MissionState(["parent goal", "later goal"])
    m.subdivide_active(["first step", "second step"])
    rendered = m.render()
    # Children (depth 1) are indented; the pending tail (t1.2 + t2) count shows.
    assert "  ▶ t1.1: first step" in rendered
    assert "2 pending task(s)" in rendered


def test_subdivide_preserves_completed_prefix() -> None:
    m = MissionState(["a", "b", "c"])
    m.complete_active("ok")  # t1 done, t2 active
    m.subdivide_active(["b-1", "b-2"])
    # t1 (done) stays at the front; t2 is replaced by its children; t3 trails.
    assert [t.task_id for t in m.tasks] == ["t1", "t2.1", "t2.2", "t3"]
    assert m.tasks[0].status == "done"


def test_has_started_tracks_progress() -> None:
    m = MissionState(["a", "b"])
    assert not m.has_started()  # fresh — a populate replace is safe here
    m.record_attempt(rskill_id="x")
    assert m.has_started()  # the active task was attempted


def test_has_started_true_after_a_terminal_task() -> None:
    m = MissionState(["a", "b"])
    m.complete_active("ok")  # t1 done, t2 active untouched
    assert m.has_started()  # a wholesale replace would now drop t1's progress


def test_rearm_active_moves_verifying_back_to_active() -> None:
    m = MissionState.from_prompt("pick the milk")
    m.mark_verifying()
    assert m.active().status == "verifying"
    rearmed = m.rearm_active()
    assert rearmed is m.active() and rearmed.status == "active"
    # idempotent on an already-active task.
    assert m.rearm_active().status == "active"
