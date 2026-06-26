"""Unit slice for the #123 blocked-task subdivision offer (ADR-0073 amendment).

The pure node decision ``_should_offer_subdivision`` — does a just-abandoned task
get one chance to decompose before the abandon/handoff ladder runs? It is bounded
two ways (one offer per task id; below the depth cap) so a task that refuses to
decompose still terminates in human-handoff. The full dispatch + re-arm flow is
exercised live in ``tests/integration/test_reasoner_node_end_to_end.py`` and the
deploy-sim run; this guards the bound itself, the way ``test_reasoner_search_cascade``
guards the search-cascade bound.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_reasoner.mission import DEFAULT_MAX_SUBDIVIDE_DEPTH, TaskState
from openral_reasoner_ros.reasoner_node import _may_subdivide_active, _should_offer_subdivision

_MAX = DEFAULT_MAX_SUBDIVIDE_DEPTH


def test_offer_subdivision_for_a_fresh_blocked_task() -> None:
    task = TaskState(task_id="t2", text="pick the milk", status="verifying", depth=0)
    assert _should_offer_subdivision(task, set(), _MAX)


def test_no_second_offer_for_the_same_task() -> None:
    # Once offered, a second abandon of the SAME task falls through to handoff.
    task = TaskState(task_id="t2", text="pick the milk", status="verifying", depth=0)
    assert not _should_offer_subdivision(task, {"t2"}, _MAX)


def test_no_offer_at_the_depth_bound() -> None:
    # A task already split to the depth cap is handed off, not split again.
    deep = TaskState(task_id="t2.1.1", text="grip", status="verifying", depth=_MAX)
    assert not _should_offer_subdivision(deep, set(), _MAX)


def test_offer_still_allowed_one_below_the_bound() -> None:
    task = TaskState(task_id="t2.1", text="grip", status="verifying", depth=_MAX - 1)
    assert _should_offer_subdivision(task, set(), _MAX)


# ── _may_subdivide_active: don't let the LLM reset the reward-gate ladder ──────
# Root cause of the glm deploy loop: the LLM proactively re-subdivides an already
# dispatched active task; `subdivide_active` splices in fresh attempts=0 children,
# so the verify ladder evaluates attempts=0 forever, never reaching max_attempts →
# never abandons → never advances/hands off. The guard makes attempts monotonic.


def test_may_subdivide_a_not_yet_attempted_task() -> None:
    # attempts == 0: no ladder progress to discard, so splitting is harmless.
    task = TaskState(task_id="t1.1", text="pick the milk", status="active", depth=0)
    assert task.attempts == 0
    assert _may_subdivide_active(task, set())


def test_may_not_subdivide_an_attempted_task_that_was_not_offered() -> None:
    # The bug: re-subdividing an in-flight task (attempts>0) would reset its ladder.
    task = TaskState(task_id="t1.1", text="pick the milk", status="active", depth=0)
    task.attempts = 2
    assert not _may_subdivide_active(task, set())


def test_may_subdivide_an_attempted_task_once_offered() -> None:
    # The #123 post-abandon invite: an exhausted task is allowed ONE subdivision.
    task = TaskState(task_id="t1.1", text="pick the milk", status="active", depth=0)
    task.attempts = 3
    assert _may_subdivide_active(task, {"t1.1"})
