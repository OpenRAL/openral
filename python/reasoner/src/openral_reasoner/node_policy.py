"""Pure prompt-handling policy for the reasoner node (no rclpy).

``openral_reasoner_ros.reasoner_node._on_prompt`` decides, per inbound
``PromptStamped``, whether the prompt is a *new operator goal* (rebuild
the mission queue, reset the search bounds and call streak) or a
*cascade response* the reasoner's own tools produced (feed it to the
next tick, keep every bound accumulating). The cascade spans six
sources (``CASCADE_PROMPT_SOURCES``). This module makes the policy
pure, single-sourced, and unit-testable without a ROS install.
"""

from __future__ import annotations

import json
from typing import Literal

from openral_reasoner.mission import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TASK_DISPATCHES,
    MissionState,
)

__all__ = [
    "CASCADE_PROMPT_SOURCES",
    "should_rebuild_mission",
    "task_step_without_verifier",
]

# Prompt frame_ids the reasoner re-publishes onto /openral/prompt for its OWN
# cascade (advisory query responses + spatial-memory re-prompts + mission
# invites). These are not new operator goals: they must not (re)build the
# mission queue, must not reset the active-search bounds, and must not reset
# the retry-cap call streak. Self-emits (frame_id == the node name) are
# dropped before this policy runs.
CASCADE_PROMPT_SOURCES: frozenset[str] = frozenset(
    {"spatial_memory", "detector", "scene_vlm", "reward_monitor", "memory", "mission"}
)


def should_rebuild_mission(
    source: str,
    metadata_json: str,
    mission: MissionState | None,
    *,
    active_succeeded_unverified: bool = False,
) -> bool:
    """Whether an inbound prompt replaces the mission queue.

    A cascade prompt never does. A genuine external prompt does when no
    mission is *in progress*: none set, empty, finished (every task
    terminal), or **not yet started** (no attempt made, nothing terminal —
    the operator re-sending / correcting a goal before the robot begins is
    the normal flow and must keep working). While a mission is genuinely in
    progress, an external prompt only replaces it when its ``metadata_json``
    carries a top-level ``"new_goal": true`` (stamped by the CLI / dashboard
    when the human explicitly starts over): an operator *reply* to a
    reasoner question ("the red one") must not silently discard the
    in-flight task queue — it reaches the LLM through the PROMPTS section
    instead. Exception: ``active_succeeded_unverified`` (the active task's
    last dispatch succeeded and no reward monitor can judge it) with no task
    queued after it — the mission is as done as the reasoner can know, so the
    operator's next prompt is a new goal, not an aside the finished task resumes
    after. Queued tasks are never dropped this way, and the node clears the flag
    once the reasoner prompts the operator, so a reply to its question stays a
    reply.

    Example:
        >>> should_rebuild_mission("cli", "", None)
        True
        >>> m = MissionState(["pick the bowl", "place it"])
        >>> should_rebuild_mission("cli", "", m)  # not started yet: resend wins
        True
        >>> m.record_attempt(rskill_id="x")
        >>> should_rebuild_mission("cli", "", m)  # reply mid-mission: keep queue
        False
        >>> should_rebuild_mission("cli", '{"new_goal": true}', m)
        True
        >>> should_rebuild_mission("cli", "", m, active_succeeded_unverified=True)  # queued
        False
        >>> _ = m.abandon_active("x")
        >>> _ = m.abandon_active("x")
        >>> should_rebuild_mission("cli", "", m)  # mission finished: next goal
        True
        >>> should_rebuild_mission("detector", "", None)  # cascade: never
        False
    """
    if source in CASCADE_PROMPT_SOURCES:
        return False
    if mission is None or mission.is_empty() or mission.is_complete() or not mission.has_started():
        return True
    if active_succeeded_unverified and not any(t.status == "pending" for t in mission.tasks):
        return True
    if not metadata_json:
        return False
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("new_goal") is True


def task_step_without_verifier(
    *,
    succeeded: bool,
    completes_task_on_success: bool,
    reward_available: bool,
    attempts: int,
    failures: int = 0,
    final_step: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_dispatches: int = DEFAULT_MAX_TASK_DISPATCHES,
) -> tuple[Literal["complete", "step", "abandon"], str] | None:
    """Settle the active task after a dispatch returns, where no reward verdict decides it.

    * ``complete`` — the skill succeeded, its manifest opts in
      (``RosIntegration.completes_task_on_success``) and the goal was the task's
      ``final_step``: the action's own success is the task's success, reward
      monitor or not.
    * ``abandon`` — the task used ``max_dispatches`` dispatches of any outcome
      (the hard bound: a task that keeps succeeding but never completes still
      terminates), or no reward monitor is available and ``failures`` reached
      ``max_attempts``. A success nobody can judge is not a failure, so a
      multi-skill task whose steps all succeed is not abandoned by the attempt
      cap (#562). The node's abandon path offers subdivision first, then hands off.
    * ``step`` — an opted-in skill succeeded as one step of a longer task
      (``final_step=False``): the step is done, the task stays active and no
      reward verdict is asked of a half-done task.
    * ``None`` — the reward gate (when available) or the LLM decides.

    Example:
        >>> task_step_without_verifier(
        ...     succeeded=True, completes_task_on_success=True, reward_available=True, attempts=1
        ... )[0]
        'complete'
        >>> task_step_without_verifier(
        ...     succeeded=True,
        ...     completes_task_on_success=True,
        ...     reward_available=False,
        ...     attempts=1,
        ...     final_step=False,
        ... )[0]
        'step'
        >>> task_step_without_verifier(  # three successes: not three failures
        ...     succeeded=True, completes_task_on_success=False, reward_available=False, attempts=3
        ... ) is None
        True
        >>> task_step_without_verifier(
        ...     succeeded=False,
        ...     completes_task_on_success=False,
        ...     reward_available=False,
        ...     attempts=3,
        ...     failures=3,
        ... )[0]
        'abandon'
        >>> task_step_without_verifier(
        ...     succeeded=True, completes_task_on_success=False, reward_available=False, attempts=10
        ... )[0]
        'abandon'
    """
    if succeeded and completes_task_on_success and final_step:
        return "complete", "the skill's own success completes the task (completes_task_on_success)"
    if attempts >= max_dispatches:
        return "abandon", (
            f"not complete after {attempts} dispatch(es), the per-task bound ({failures} failed)"
        )
    if not reward_available and failures >= max_attempts:
        return "abandon", (
            f"{failures} of {attempts} attempt(s) failed; no reward monitor can judge the task"
        )
    if succeeded and completes_task_on_success:
        return "step", (
            "the skill's own success completes only this step (final_step=false); "
            "the task stays active"
        )
    return None
