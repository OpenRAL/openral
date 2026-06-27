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
from openral_reasoner_ros.reasoner_node import _should_offer_subdivision

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
