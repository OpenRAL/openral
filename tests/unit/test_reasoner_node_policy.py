"""Unit tests for the pure prompt-handling policy (``openral_reasoner.node_policy``).

Covers the two ``_on_prompt`` decisions that used to live inline in the ROS
callback (and diverged — the search-bound reset excluded only
``"spatial_memory"`` while the cascade spans six sources, so every
``detector`` re-prompt reset the locate-miss budget it had just charged):

1. ``is_cascade_source`` — the single source of truth for "this prompt is the
   reasoner's own cascade, reset nothing".
2. ``should_rebuild_mission`` — new-goal vs mid-mission-reply discrimination.
"""

from __future__ import annotations

import pytest
from openral_reasoner import MissionState
from openral_reasoner.node_policy import (
    CASCADE_PROMPT_SOURCES,
    is_cascade_source,
    should_rebuild_mission,
)

# ── is_cascade_source ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    sorted(CASCADE_PROMPT_SOURCES),
)
def test_every_cascade_source_is_recognised(source: str) -> None:
    """Every cascade frame_id — including detector/reward_monitor/mission —
    is classified as cascade (the pre-fix reset guard matched only
    ``spatial_memory``)."""
    assert is_cascade_source(source)


@pytest.mark.parametrize("source", ["cli", "dashboard", "auto", "", "operator"])
def test_external_sources_are_not_cascade(source: str) -> None:
    assert not is_cascade_source(source)


def test_detector_is_a_cascade_source_regression() -> None:
    """Regression for the locate-miss budget reset: a ``detector`` re-prompt is
    cascade, so the node must NOT reset ``_spatial_search`` on it — otherwise
    the miss budget can never exceed 1 and an undetectable object loops
    locate→miss→reset forever."""
    assert "detector" in CASCADE_PROMPT_SOURCES
    assert is_cascade_source("detector")


# ── should_rebuild_mission ───────────────────────────────────────────────────


def _in_progress_mission() -> MissionState:
    mission = MissionState(["pick the bowl", "place the butter"])
    mission.record_attempt(rskill_id="openral/rskill-x")
    return mission


def test_external_prompt_with_no_mission_rebuilds() -> None:
    assert should_rebuild_mission("cli", "", None)


def test_external_prompt_with_empty_mission_rebuilds() -> None:
    assert should_rebuild_mission("cli", "", MissionState([]))


def test_external_prompt_with_finished_mission_rebuilds() -> None:
    mission = MissionState(["pick the bowl"])
    mission.complete_active("progress=0.91")
    assert mission.is_complete()
    assert should_rebuild_mission("dashboard", "", mission)


def test_resend_before_any_work_started_rebuilds() -> None:
    """An operator correcting/re-sending a goal before the robot begins must
    still replace the seed mission (the pre-work resend flow)."""
    mission = MissionState(["pick the bowl"])
    assert not mission.has_started()
    assert should_rebuild_mission("cli", "", mission)


def test_reply_mid_mission_keeps_the_queue() -> None:
    """An operator *reply* ("the red one") mid-mission must not clobber the
    in-flight task queue — the pre-fix behaviour replaced the whole mission
    with the reply text as a single task."""
    assert not should_rebuild_mission("cli", "", _in_progress_mission())


def test_explicit_new_goal_metadata_replaces_mid_mission() -> None:
    assert should_rebuild_mission("cli", '{"new_goal": true}', _in_progress_mission())


@pytest.mark.parametrize(
    "metadata",
    ['{"new_goal": false}', '{"new_goal": "yes"}', "not json", "[1,2]", ""],
)
def test_non_true_new_goal_metadata_does_not_replace(metadata: str) -> None:
    assert not should_rebuild_mission("cli", metadata, _in_progress_mission())


@pytest.mark.parametrize("source", sorted(CASCADE_PROMPT_SOURCES))
def test_cascade_prompts_never_rebuild(source: str) -> None:
    assert not should_rebuild_mission(source, '{"new_goal": true}', None)
    assert not should_rebuild_mission(source, "", _in_progress_mission())
