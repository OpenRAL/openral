"""Unit tests for ``openral_reasoner.mission``.

The deterministic mission queue that fixes the multi-task deploy gap: an
operator goal carrying several ordered subtasks is split, sequenced one-active
at a time, and advanced only on an explicit verdict — instead of being handed to
the LLM as one opaque string and forgotten on the pull-once prompt drain.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

from openral_reasoner import (
    MissionState,
    TaskState,
    evaluate_task_verdict,
)
from openral_reasoner.mission import DEFAULT_MAX_SUBDIVIDE_DEPTH

_T = TypeVar("_T")


def _some(value: _T | None) -> _T:
    assert value is not None
    return value


# ── MissionState.from_prompt — single-task seeding ────────────────────────────


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
    assert _some(m.active()).text == "a"
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
    assert _some(m.active()).task_id == "t2"
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
    assert t1.status == "abandoned" and _some(t1.last_verdict).startswith("ladder exhausted")
    # Abandoned is terminal — it is NOT silently re-queued.
    assert _some(m.active()).task_id == "t2"


def test_mission_complete_when_all_terminal_mixed() -> None:
    m = MissionState(["a", "b"])
    m.abandon_active("unverifiable")  # t1 abandoned, t2 active
    assert m.complete_active("success=0.9") is None  # t2 done, none left
    assert m.is_complete()
    assert [t.status for t in m.tasks] == ["abandoned", "done"]


# ── attempts + verifying transition ──────────────────────────────────────────


def test_record_attempt_increments_and_tracks_ids() -> None:
    skill_id = "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"
    m = MissionState(["a", "b"])
    m.record_attempt(rskill_id=skill_id, trace_id="abc123")
    m.record_attempt(rskill_id=skill_id)
    t1 = _some(m.active())
    assert t1.attempts == 2
    assert t1.last_rskill_id == "OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16"
    assert t1.last_trace_id == "abc123"  # preserved; second call passed no trace


def test_mark_verifying_keeps_task_active_for_active_lookup() -> None:
    m = MissionState(["a", "b"])
    m.mark_verifying()
    assert m.tasks[0].status == "verifying"
    # `active()` still returns it while verifying (gating is in progress).
    assert _some(m.active()).task_id == "t1"


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


# ── evaluate_task_verdict — the reward gate decision ──────────────────────────


def test_verdict_complete_on_success() -> None:
    # The band gates on the PROGRESS head.
    action, verdict = evaluate_task_verdict(
        ok=True, progress_now=0.91, success_threshold=0.8, check_floor=0.5, failures=1
    )
    assert action == "complete"
    assert verdict == "progress=0.91"


def test_verdict_ambiguous_band_returns_vlm_check() -> None:
    # progress_now=0.79 sits in the [check_floor, success_threshold) band — vlm_check.
    action, _ = evaluate_task_verdict(
        ok=True, progress_now=0.79, success_threshold=0.8, check_floor=0.5, failures=1
    )
    assert action == "vlm_check"


# test_verdict_not_ok_falls_through_to_retry_then_abandon moved to
# tests/unit/test_evaluate_task_verdict_three_tier.py
# ::test_ok_false_retry_until_exhausted_then_abandon (byte-identical body).


# ── subdivide_active (#123 — flat-splice subdivision) ─────────────────────────


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
    assert _some(m.active()).task_id == "t1.1"
    assert _some(m.complete_active("ok")).task_id == "t1.2"  # next child
    assert _some(m.complete_active("ok")).task_id == "t2"  # then the original tail
    assert m.complete_active("ok") is None
    assert m.is_complete()


def test_subdivide_respects_depth_bound_then_refuses() -> None:
    m = MissionState.from_prompt("root task")
    # depth 0 → 1 → 2 allowed; at DEFAULT_MAX_SUBDIVIDE_DEPTH (2) it is refused.
    assert _some(m.subdivide_active(["a", "b"])).depth == 1
    assert _some(m.subdivide_active(["c", "d"])).depth == DEFAULT_MAX_SUBDIVIDE_DEPTH
    assert m.subdivide_active(["e"]) is None  # would be depth 3 — refused → caller hands off
    # The refused call must not mutate the queue (the depth-1 sibling t1.2 trails).
    assert [t.task_id for t in m.tasks] == ["t1.1.1", "t1.1.2", "t1.2"]


def test_subdivide_custom_max_depth() -> None:
    m = MissionState.from_prompt("root")
    assert _some(m.subdivide_active(["a"], max_depth=1)).depth == 1
    assert m.subdivide_active(["b"], max_depth=1) is None  # depth 1 >= 1


def test_subdivide_empty_subtasks_is_noop() -> None:
    m = MissionState.from_prompt("only task")
    assert m.subdivide_active([]) is None
    assert m.subdivide_active(["  ", ""]) is None  # whitespace trims to empty
    assert [t.task_id for t in m.tasks] == ["t1"]
    assert _some(m.active()).task_id == "t1"


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
    assert _some(m.active()).status == "verifying"
    rearmed = _some(m.rearm_active())
    assert rearmed is m.active() and rearmed.status == "active"
    # idempotent on an already-active task.
    assert _some(m.rearm_active()).status == "active"


def _summary_tasks(m: MissionState) -> list[dict[str, Any]]:
    return cast("list[dict[str, Any]]", m.to_summary()["tasks"])


def test_steps_track_each_dispatch_and_agree_with_the_ladder() -> None:
    """Each dispatch is a step; an unreported one is interrupted; a counted failure wins."""
    m = MissionState.from_prompt("go to shelf S3 and restock lower slot 1")
    m.record_attempt(rskill_id="nav", prompt="go to S3")
    m.settle_step("ok")
    m.record_attempt(rskill_id="restock", prompt="restock slot 1")  # never reported back
    m.record_attempt(rskill_id="restock", prompt="restock slot 1")
    m.settle_step("ok")
    _some(m.active()).settle_step("failed", overwrite=True)  # reward gate: not done
    m.record_attempt(rskill_id="restock", prompt="restock slot 1")
    m.settle_step("cancelled", task_id="t1")
    m.settle_step("ok")  # a late result must not rewrite a settled step
    steps = _summary_tasks(m)[0]["steps"]
    assert [(s["rskill_id"], s["outcome"]) for s in steps] == [
        ("nav", "ok"),
        ("restock", "interrupted"),
        ("restock", "failed"),
        ("restock", "cancelled"),
    ]
    assert m.to_summary()["goal"] == "go to shelf S3 and restock lower slot 1"


def test_splits_keep_the_goal_and_the_parent_steps_at_every_depth() -> None:
    """Nested subdivisions keep each replaced task (text + steps) by id; a restart restores them."""
    goal = "go to shelf S3 and restock lower slot 1"
    m = MissionState.from_prompt(goal)
    m.record_attempt(rskill_id="nav", prompt="go to S3")
    m.subdivide_active(["navigate to shelf S3", "restock box x1 into lower slot 1"])
    m.record_attempt(rskill_id="nav", prompt="navigate")
    m.subdivide_active(["clear the aisle", "drive to the S3 dock"])
    summary = m.to_summary()
    assert [t["id"] for t in _summary_tasks(m)] == ["t1.1.1", "t1.1.2", "t1.2"]
    assert summary["splits"] == {
        "t1": {
            "text": goal,
            "steps": [{"rskill_id": "nav", "prompt": "go to S3", "outcome": "interrupted"}],
        },
        "t1.1": {
            "text": "navigate to shelf S3",
            "steps": [{"rskill_id": "nav", "prompt": "navigate", "outcome": "interrupted"}],
        },
    }
    m.record_attempt(rskill_id="nav", prompt="clear")  # in flight at the "crash"
    restored = MissionState.from_state_dict(m.to_state_dict())
    assert restored.to_summary()["splits"] == summary["splits"]
    assert restored.to_summary()["goal"] == goal
    assert _summary_tasks(restored)[0]["steps"][-1]["outcome"] == "interrupted"


def test_populated_queue_keeps_the_operator_goal() -> None:
    """decompose_mission populate builds a new queue; the goal it serves stays for the card."""
    seed = MissionState.from_prompt("go to shelf S3 and restock lower slot 1")
    m = MissionState(["navigate to shelf S3", "restock lower slot 1"], goal=seed.goal)
    assert m.to_summary()["goal"] == "go to shelf S3 and restock lower slot 1"
    assert [t["depth"] for t in _summary_tasks(m)] == [0, 0]


def test_forget_conversation_drops_the_old_goals_words_but_keeps_skill_outcomes() -> None:
    """#661: in the fleet spike a robot asked about the previous goal's "left block"."""
    from openral_reasoner.context import ContextRenderer, ExecutionEventRecord, PromptRecord

    r = ContextRenderer()
    r.append_execution(
        ExecutionEventRecord(
            rskill_id="OpenRAL/rskill-pick", outcome="ok", summary="picked", reflection=None,
            stamp_ns=0,
        )
    )  # fmt: skip
    r.append_execution(
        ExecutionEventRecord(
            rskill_id="(emit_prompt)", outcome="ok",
            summary="said to the operator: the left block or the right block?", reflection=None,
            stamp_ns=1,
        )
    )  # fmt: skip
    r.append_prompt(PromptRecord(text="the left one", metadata_json="", stamp_ns=2))
    new = r.append_prompt(PromptRecord(text="pick up the orange", metadata_json="", stamp_ns=3))
    r.awaiting_operator = "the left block or the right block?"
    r.forget_conversation(keep=new)
    text = r.render(world_state=None)
    assert "block" not in text and "the left one" not in text
    assert "pick up the orange" in text and "OpenRAL/rskill-pick" in text
    assert r.awaiting_operator is None


def test_append_prompt_returns_the_record_as_stored() -> None:
    from openral_reasoner.context import ContextRenderer, PromptRecord

    r = ContextRenderer()
    stored = r.append_prompt(PromptRecord(text="go", metadata_json='{"priority": 100}', stamp_ns=0))
    assert stored.priority == 100 and r.prompts[0] is stored


def test_forget_conversation_keeps_the_new_goal_when_a_full_buffer_evicted_it() -> None:
    from openral_reasoner.context import ContextRenderer, PromptRecord

    r = ContextRenderer(buffer_size=2)
    for i in range(2):
        r.append_prompt(
            PromptRecord(text=f"old {i}", metadata_json='{"priority": 100}', stamp_ns=i)
        )
    new = r.append_prompt(PromptRecord(text="pick up the orange", metadata_json="", stamp_ns=9))
    assert new not in r.prompts  # evicted on arrival: lower priority than a full buffer
    r.forget_conversation(keep=new)
    assert [p.text for p in r.prompts] == ["pick up the orange"]
