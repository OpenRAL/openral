"""Regression guard for the active-search cascade bound.

The find→re-prompt cascade (recall_object → escalate locate_in_view → re-prompt)
is bounded by ``SearchProgress``: after ``max_attempts`` consecutive lookups the
reasoner hands off to a human instead of looping forever. The bound counts only
*consecutive search* dispatches — a non-search tool call resets the streak.

The bug this guards: ``locate_in_view`` was **not** in the exempt set, so a
directly-emitted ``LocateInViewTool`` reset the very budget meant to bound it.
Against an object the detector can't recognise (``found=False`` every time), the
``recall → locate → recall`` loop zeroed the counter each cycle and never handed
off — observed live as 127 consecutive locate attempts with the VLA never
dispatched (libero_object deploy scene).
"""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_core import (
    EmitPromptTool,
    ExecuteRskillTool,
    LocateInViewTool,
    QuerySceneTool,
    RecallObjectTool,
    ResolvePlaceTool,
    WaitTool,
)
from openral_reasoner.active_search import SearchBudget, SearchProgress
from openral_reasoner_ros.reasoner_node import _resets_search_episode


def test_search_actions_do_not_reset_the_cascade() -> None:
    # recall_object / resolve_place / locate_in_view are search steps — they must
    # NOT reset the budget, or the cascade can never exhaust.
    assert not _resets_search_episode(RecallObjectTool(query="milk"))
    assert not _resets_search_episode(ResolvePlaceTool(reference="the basket"))
    assert not _resets_search_episode(
        LocateInViewTool(query="milk", detector="omdet-turbo-locator")
    )
    # The wait no-op makes no progress on the search: if it reset the budget,
    # a locate -> wait -> locate alternation would zero the counter every
    # other tick and the cascade could never exhaust.
    assert not _resets_search_episode(WaitTool(rationale="observing"))


def test_non_search_actions_reset_the_cascade() -> None:
    # Anything that actually makes progress (or asks the scene VLM) ends the
    # search episode so the next streak starts fresh.
    assert _resets_search_episode(
        ExecuteRskillTool(
            rskill_id="OpenRAL/rskill-smolvla-franka_panda-libero_spatial-bf16",
            prompt="pick the milk",
        )
    )
    assert _resets_search_episode(
        EmitPromptTool(target_topic="/openral/prompt", text="handing off")
    )
    assert _resets_search_episode(QuerySceneTool(question="is the milk grasped?"))


def test_repeated_locate_misses_exhaust_the_budget_when_not_reset() -> None:
    # The invariant the fix restores: as long as locate does not reset the
    # budget, a fixed number of consecutive misses terminates the cascade.
    progress = SearchProgress(SearchBudget(max_attempts=5))
    # Four misses still within budget (re-prompt continues)…
    for _ in range(4):
        assert progress.record_attempt() is True
    # …the fifth exhausts it → handoff.
    assert progress.record_attempt() is False
    assert progress.exhausted
