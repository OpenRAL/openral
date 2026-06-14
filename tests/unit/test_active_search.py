"""Tests for the bounded active-search frontier (ADR-0039 §3, Phase 4).

Pure-Python over the real home scene graph (CLAUDE.md §1.11) — exercises
candidate generation/ranking, the budget bound, the attempt counter, and the
LLM-readable frontier rendering.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from openral_core import SceneGraph
from openral_reasoner import (
    SearchBudget,
    SearchProgress,
    format_search_frontier,
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
