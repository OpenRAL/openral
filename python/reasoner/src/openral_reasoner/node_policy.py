"""Pure prompt-handling policy for the reasoner node (no rclpy).

``openral_reasoner_ros.reasoner_node._on_prompt`` makes two consequential
decisions per inbound ``PromptStamped``: whether the prompt is a *new
operator goal* (rebuild the mission queue, reset the search bounds and call
streak) or a *cascade response* the reasoner's own tools produced (feed it
to the next tick, keep every bound accumulating). Both decisions were inline
in the ROS callback — and the search-bound reset compared against the single
literal ``"spatial_memory"`` while the cascade actually spans six sources,
so every ``detector`` / ``reward_monitor`` / ``mission`` / ``memory`` /
``scene_vlm`` re-prompt silently reset the very budget its dispatch had just
charged (the locate-miss budget could never exceed 1). This module makes the
policy pure, single-sourced, and unit-testable without a ROS install.
"""

from __future__ import annotations

import json

from openral_reasoner.mission import MissionState

__all__ = [
    "CASCADE_PROMPT_SOURCES",
    "should_rebuild_mission",
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
    instead.

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
    if not metadata_json:
        return False
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("new_goal") is True
