"""Unit tests for the pure prompt-handling policy (``openral_reasoner.node_policy``).

Covers the two ``_on_prompt`` decisions that used to live inline in the ROS
callback (and diverged — the search-bound reset excluded only
``"spatial_memory"`` while the cascade spans six sources, so every
``detector`` re-prompt reset the locate-miss budget it had just charged):

1. ``CASCADE_PROMPT_SOURCES`` — the single source of truth for "this prompt is
   the reasoner's own cascade, reset nothing".
2. ``should_rebuild_mission`` — new-goal vs mid-mission-reply discrimination.
"""

from __future__ import annotations

import pytest
from openral_reasoner import MissionState
from openral_reasoner.mission import DEFAULT_MAX_ATTEMPTS
from openral_reasoner.node_policy import (
    CASCADE_PROMPT_SOURCES,
    should_rebuild_mission,
    task_step_without_verifier,
)

# ── CASCADE_PROMPT_SOURCES ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    sorted(CASCADE_PROMPT_SOURCES),
)
def test_every_cascade_source_is_recognised(source: str) -> None:
    """Every cascade frame_id — including detector/reward_monitor/mission — is
    classified as cascade (the pre-fix reset guard matched only ``spatial_memory``)."""
    assert source in CASCADE_PROMPT_SOURCES


@pytest.mark.parametrize("source", ["cli", "dashboard", "auto", "", "operator"])
def test_external_sources_are_not_cascade(source: str) -> None:
    assert source not in CASCADE_PROMPT_SOURCES


def test_detector_is_a_cascade_source_regression() -> None:
    """Regression for the locate-miss budget reset: a ``detector`` re-prompt is
    cascade, so the node must NOT reset ``_spatial_search`` on it — otherwise
    the miss budget can never exceed 1 and an undetectable object loops
    locate→miss→reset forever."""
    assert "detector" in CASCADE_PROMPT_SOURCES


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


# ── finished-but-unverified tasks (2026-11-11 Nav2 run) ──────────────────────


def test_prompt_after_an_unverified_success_starts_a_new_goal() -> None:
    """ "drive back to start" after a Nav2 drive that succeeded with no verifier is a
    new goal, not an aside the first task resumes after; a reply after a FAILED
    attempt still keeps the queue."""
    mission = _in_progress_mission()
    assert should_rebuild_mission("dashboard", "", mission, active_succeeded_unverified=True)
    assert not should_rebuild_mission("dashboard", "", mission, active_succeeded_unverified=False)
    # Cascade prompts never rebuild, whatever the task's state.
    assert not should_rebuild_mission("mission", "", mission, active_succeeded_unverified=True)


def test_a_task_completed_on_action_success_is_rebuilt_by_the_next_prompt() -> None:
    """Once ``completes_task_on_success`` completed the only task, the mission is
    finished, so the operator's next prompt replaces it."""
    mission = MissionState(["drive to x=4,y=0 in map"])
    mission.record_attempt(rskill_id="OpenRAL/rskill-nav2-mobile_base-navigate_to_pose")
    step = task_step_without_verifier(
        succeeded=True, completes_task_on_success=True, reward_available=False, attempts=1
    )
    assert step is not None and step[0] == "complete"
    mission.complete_active(step[1])
    assert mission.is_complete()
    assert should_rebuild_mission("dashboard", "", mission)


@pytest.mark.parametrize("reward_available", [False, True])
def test_opted_in_success_completes_with_or_without_a_reward_monitor(
    reward_available: bool,
) -> None:
    step = task_step_without_verifier(
        succeeded=True,
        completes_task_on_success=True,
        reward_available=reward_available,
        attempts=1,
    )
    assert step is not None
    assert step[0] == "complete"
    assert "completes_task_on_success" in step[1]


def test_success_without_the_opt_in_is_not_completed() -> None:
    assert (
        task_step_without_verifier(
            succeeded=True, completes_task_on_success=False, reward_available=False, attempts=1
        )
        is None
    )


@pytest.mark.parametrize("succeeded", [False, True])
def test_attempt_cap_abandons_without_a_reward_monitor(succeeded: bool) -> None:
    """The per-task cap binds even when no reward gate runs (it used to live only there)."""
    below = task_step_without_verifier(
        succeeded=succeeded,
        completes_task_on_success=False,
        reward_available=False,
        attempts=DEFAULT_MAX_ATTEMPTS - 1,
    )
    assert below is None
    at_cap = task_step_without_verifier(
        succeeded=succeeded,
        completes_task_on_success=False,
        reward_available=False,
        attempts=DEFAULT_MAX_ATTEMPTS,
    )
    assert at_cap is not None
    assert at_cap[0] == "abandon"
    assert f"{DEFAULT_MAX_ATTEMPTS} attempt(s)" in at_cap[1]


def test_attempt_cap_is_left_to_the_reward_gate_when_one_is_available() -> None:
    assert (
        task_step_without_verifier(
            succeeded=False,
            completes_task_on_success=False,
            reward_available=True,
            attempts=DEFAULT_MAX_ATTEMPTS + 5,
        )
        is None
    )
