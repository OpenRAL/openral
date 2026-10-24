"""Tests for the bounded active-search frontier.

Pure-Python over the real home scene graph (CLAUDE.md §1.11) — exercises
candidate generation/ranking, the budget bound, the attempt counter, and the
LLM-readable frontier rendering.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from openral_core import (
    ObjectQueryEvidence,
    ObjectQueryObservation,
    ObjectRole,
    PlaceRegion,
    Pose6D,
    SceneGraph,
)
from openral_reasoner import (
    SearchBudget,
    SearchProgress,
    format_search_frontier,
    object_query_matches_invocation,
    object_query_step,
    object_query_task_is_current,
    plan_active_search,
)

_FIXTURE = Path("tests/unit/fixtures/home_scene_graph.json")


def _graph() -> SceneGraph:
    return SceneGraph.model_validate_json(_FIXTURE.read_text())


def test_frontier_ranks_occluding_containers_first() -> None:
    cands = plan_active_search(_graph(), target_text="a wine glass", budget=SearchBudget())
    assert cands, "the home graph has places/containers to search"
    # Occluding containers (fridge, cabinet) outrank open places.
    assert cands[0].open_container_id in {"fridge", "cabinet"}
    assert cands[1].open_container_id in {"fridge", "cabinet"}
    # Ranks are non-increasing.
    assert all(a.rank >= b.rank for a, b in pairwise(cands))
    # At least one open-place candidate (no container to open).
    assert any(c.open_container_id is None for c in cands)


def test_budget_caps_the_frontier() -> None:
    cands = plan_active_search(
        _graph(), target_text="a wine glass", budget=SearchBudget(max_candidates=2)
    )
    assert len(cands) == 2
    assert all(c.open_container_id in {"fridge", "cabinet"} for c in cands)


def test_empty_graph_yields_no_candidates() -> None:
    assert plan_active_search(SceneGraph(), target_text="anything", budget=SearchBudget()) == []


def test_search_progress_is_bounded_and_resettable() -> None:
    progress = SearchProgress(SearchBudget(max_attempts=2))
    assert progress.record_attempt() is True  # attempt 1, budget remains
    assert progress.attempts == 1
    assert progress.record_attempt() is False  # attempt 2, now exhausted
    assert progress.exhausted is True
    progress.reset()
    assert progress.attempts == 0
    assert progress.exhausted is False


def test_format_frontier_text() -> None:
    cands = plan_active_search(_graph(), target_text="a wine glass", budget=SearchBudget())
    text = format_search_frontier(cands, "a wine glass")
    assert "Candidate places to search" in text
    assert "a wine glass" in text
    assert "hand off to a human" in format_search_frontier([], "a wine glass")


def test_object_query_recovery_carries_reason_workspace_and_bounded_observations() -> None:
    workspace = PlaceRegion(
        frame_id="base",
        pose=Pose6D(xyz=(0.2, 0.1, 0.3), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        half_extents=(0.1, 0.1, 0.1),
    )
    evidence = ObjectQueryEvidence(
        escalation="exhausted",
        goal_id="grasp:obj:orange@17",
        query_id="q0",
        label="orange",
        role=ObjectRole.TARGET,
        waited_s=12.0,
        attempts=4,
        reason="workspace_mismatch",
        workspace=workspace,
        observations=[
            ObjectQueryObservation(camera=f"cam{i}", stamp_ns=i, reason="too_few_points")
            for i in range(8)
        ],
    )

    assert len(evidence.observations) == 8
    assert evidence.observations[0].camera == "cam0"
    step = object_query_step(evidence, attempts=1, max_attempts=3)
    assert step.rung == "recover"
    assert step.reason == "workspace_mismatch"
    assert "base bounds" in step.reflection
    assert "cam7@7:too_few_points" in step.reflection
    assert "re-ground" in step.reflection
    assert "handoff_prompt" not in step.model_dump()
    blocked = object_query_step(evidence, attempts=3, max_attempts=3)
    assert blocked.rung == "blocked"
    assert "new operator goal" in blocked.reflection


def test_unavailable_observation_recovery_states_visibility_is_unknown() -> None:
    evidence = ObjectQueryEvidence(
        escalation="exhausted",
        goal_id="grasp:obj:orange@17",
        query_id="q0",
        label="orange",
        role=ObjectRole.TARGET,
        waited_s=12.0,
        attempts=0,
        reason="no_camera_pose",
        observations=[
            ObjectQueryObservation(camera="top_depth", stamp_ns=12, reason="no_camera_pose")
        ],
    )

    step = object_query_step(evidence, attempts=0, max_attempts=3)

    assert step.reason == "no_camera_pose"
    assert "visibility is unknown" in step.reflection
    assert "absence is unverified" in step.reflection
    assert "top_depth@12:no_camera_pose" in step.reflection
    assert "object_not_found" not in step.reflection


def test_delayed_object_query_failure_must_match_live_invocation() -> None:
    evidence = ObjectQueryEvidence(
        escalation="unresolved",
        goal_id="grasp:obj:orange@17",
        query_id="q0",
        label="orange",
        role=ObjectRole.TARGET,
        waited_s=5.0,
        attempts=1,
        invocation_id="0123456789abcdef0123456789abcdef",
    )
    current = {
        "active_invocation_id": "0123456789abcdef0123456789abcdef",
    }
    assert object_query_matches_invocation(evidence, **current)
    assert not object_query_matches_invocation(
        evidence, **{**current, "active_invocation_id": "fedcba9876543210fedcba9876543210"}
    )
    assert not object_query_matches_invocation(evidence, active_invocation_id=None)
    # MissionState starts task ids at t1 for each operator goal. The generation
    # check distinguishes two different goals that happen to reuse the same id.
    assert not object_query_task_is_current(
        dispatch_task_id="t1",
        dispatch_mission_generation=4,
        current_task_id="t1",
        current_mission_generation=5,
    )
    assert object_query_task_is_current(
        dispatch_task_id="t1",
        dispatch_mission_generation=5,
        current_task_id="t1",
        current_mission_generation=5,
    )
