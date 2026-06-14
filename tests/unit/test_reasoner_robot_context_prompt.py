"""Unit tests for :func:`openral_reasoner.render_robot_context_prompt`.

Option B (ADR-0018 F4): the reasoner's system prompt carries a
``## THIS ROBOT`` block built from the active robot's
:class:`~openral_core.RobotCapabilities`. We validate against real
``robots/`` fixtures (CLAUDE.md §1.11) — ``panda_mobile`` (a wheeled
mobile manipulator) and ``so100_follower`` (a fixed-base arm) — to
exercise both locomotion branches with real capability data, never
``"foo"`` placeholders.
"""

from __future__ import annotations

import pathlib

import pytest
from openral_core import RobotCapabilities, RobotDescription
from openral_reasoner.tool_use import (
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_ENV_VAR,
    render_robot_context_prompt,
    resolve_reasoner_system_prompt,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _capabilities(robot_id: str) -> RobotCapabilities:
    """Load real ``RobotCapabilities`` from ``robots/<id>/robot.yaml``."""
    robot_yaml = _REPO_ROOT / "robots" / robot_id / "robot.yaml"
    if not robot_yaml.is_file():
        pytest.skip(f"robot fixture not present: {robot_yaml}")
    return RobotDescription.from_yaml(str(robot_yaml)).capabilities


def test_none_returns_base_prompt_unchanged() -> None:
    """With no robot wired the prompt is the robot-agnostic default."""
    assert render_robot_context_prompt(None) == DEFAULT_SYSTEM_PROMPT


def test_base_prompt_is_preserved_as_prefix() -> None:
    """The robot block is appended; the base prompt is not mutated."""
    caps = _capabilities("so100_follower")
    rendered = render_robot_context_prompt(caps)
    assert rendered.startswith(DEFAULT_SYSTEM_PROMPT.rstrip())
    assert "## THIS ROBOT" in rendered


def test_mobile_manipulator_advertises_navigation() -> None:
    """A wheeled base gets the 'you may navigate to approach' guidance."""
    caps = _capabilities("panda_mobile")
    rendered = render_robot_context_prompt(caps)
    # panda_mobile declares several tags (franka, mobile_base, panda, ...);
    # they render sorted on one line, so assert the line + the tag presence.
    assert "embodiment_tags: franka, mobile_base, panda, panda_mobile, robocasa" in rendered
    assert "locomotion: wheeled" in rendered
    assert "dispatch a navigation skill" in rendered
    # panda_mobile declares force control and a 3 kg payload.
    assert "force/impedance control" in rendered
    assert "payload: up to 3 kg" in rendered


def test_fixed_base_arm_disables_navigation() -> None:
    """A ``locomotion: [none]`` robot is told it cannot drive to approach."""
    caps = _capabilities("so100_follower")
    rendered = render_robot_context_prompt(caps)
    assert "embodiment_tags: lerobot, so100_follower" in rendered
    assert "locomotion: none" in rendered
    assert "no mobile base" in rendered
    assert "hand off to the operator" in rendered
    # No force control declared → not listed; single-arm is.
    assert "single-arm" in rendered
    assert "force/impedance control" not in rendered


def test_render_is_deterministic() -> None:
    """Same capabilities render byte-identically (reproducibility, §8)."""
    caps = _capabilities("panda_mobile")
    assert render_robot_context_prompt(caps) == render_robot_context_prompt(caps)


def test_embodiment_tags_are_sorted() -> None:
    """Tag order is stable regardless of declaration order."""
    caps = RobotCapabilities(embodiment_tags=["widowx", "aloha", "lerobot"])
    rendered = render_robot_context_prompt(caps)
    assert "embodiment_tags: aloha, lerobot, widowx" in rendered


# ── OPENRAL_REASONER_SYSTEM_PROMPT override ───────────────────────────────────


def test_env_unset_uses_default_brief() -> None:
    """No override → the default operating brief is the base."""
    caps = _capabilities("so100_follower")
    rendered = resolve_reasoner_system_prompt(caps, env={})
    assert rendered.startswith(DEFAULT_SYSTEM_PROMPT.rstrip())
    assert "## THIS ROBOT" in rendered


def test_env_override_replaces_base_brief() -> None:
    """A non-empty override replaces the brief; robot block still appended."""
    caps = _capabilities("so100_follower")
    rendered = resolve_reasoner_system_prompt(
        caps,
        env={SYSTEM_PROMPT_ENV_VAR: "Operate the lab arm carefully."},
    )
    assert rendered.startswith("Operate the lab arm carefully.")
    # The default brief is gone; the per-robot block remains.
    assert "You are the OpenRAL S2 reasoner" not in rendered
    assert "## THIS ROBOT" in rendered
    assert "embodiment_tags: lerobot, so100_follower" in rendered


def test_env_override_blank_falls_back_to_default() -> None:
    """A whitespace-only override is treated as unset (fail-safe)."""
    caps = _capabilities("so100_follower")
    rendered = resolve_reasoner_system_prompt(caps, env={SYSTEM_PROMPT_ENV_VAR: "   "})
    assert rendered.startswith(DEFAULT_SYSTEM_PROMPT.rstrip())


def test_env_override_without_robot_is_brief_only() -> None:
    """Override + no capabilities → just the custom brief, no robot block."""
    rendered = resolve_reasoner_system_prompt(
        None,
        env={SYSTEM_PROMPT_ENV_VAR: "Custom brief."},
    )
    assert rendered == "Custom brief."
    assert "## THIS ROBOT" not in rendered


# ── ADR-0044 Phase 4b — the go-see-then-act ladder in the base prompt ─────────


def test_default_prompt_carries_the_full_ladder() -> None:
    """The recall→navigate→look→verify→manipulate ladder is in the base prompt.

    The rungs are phrased conditionally on tool/skill availability (so the LLM
    is never told to call a tool that isn't in its palette), but the ladder
    structure and each rung's intent must be present.
    """
    p = DEFAULT_SYSTEM_PROMPT
    # The ladder is named and ordered.
    assert "ladder" in p
    assert "(1) recall" in p and "(2) navigate" in p
    assert "(3) aim a camera" in p and "(4) verify" in p and "(5) manipulate" in p
    # Rung 3 (look-at) is referenced generically — it's an execute_rskill rSkill,
    # like the nav skill — not a hardcoded tool name.
    assert "camera-aiming (look-at) skill" in p
    # Rung 4 (locate_in_view) is a named read-only tool, gated on its palette.
    assert "locate_in_view" in p
    assert "in the palette, call it to confirm" in p
    # The live-vs-remembered contrast is spelled out so the LLM picks the right one.
    assert "unlike recall_object" in p and "live perception" in p
    # Grasp only after confirmation; otherwise re-aim/re-approach.
    assert "Only dispatch a manipulation skill once the target is confirmed in view" in p
    assert "grasping blind" in p


def test_default_prompt_respects_grid_blocked_approach() -> None:
    """The recall rung tells the LLM to honour Phase-4a 'approach BLOCKED'."""
    p = DEFAULT_SYSTEM_PROMPT
    assert "approach BLOCKED" in p
    assert "do NOT navigate to it" in p


def test_ladder_rungs_stay_conditional_not_imperative() -> None:
    """Each optional rung is gated on its tool/skill being in the palette.

    Guards against advertising a tool the LLM cannot call (CLAUDE.md §1.4):
    recall_object, resolve_place, the look-at skill, and locate_in_view are all
    introduced with a 'when ... is in the palette' qualifier.
    """
    p = DEFAULT_SYSTEM_PROMPT
    assert "read-only recall_object tool is in the palette" in p
    assert "read-only resolve_place tool is in the palette" in p
    assert "camera-aiming (look-at) skill is in the palette" in p
    assert "read-only locate_in_view tool is in the palette" in p


def test_prompt_allows_manipulation_when_seen_but_not_lifted() -> None:
    """ADR-0043/0052 — a live in-view confirmation lets the reasoner attempt a
    manipulation skill even when recall_object cannot resolve a 3-D pose (the
    object is seen but not yet lifted into spatial memory). Without this, the
    reasoner stalls on the search ladder / hands off for a target the depth
    sensor can't lift (e.g. a distant object only the wide camera sees).
    """
    p = DEFAULT_SYSTEM_PROMPT.lower()
    assert "seen but not yet lifted" in p
    assert "locate_in_view" in p
    assert "human-handoff" in p
    assert "missing 3-d lift" in p
    assert "mobile-manipulator skill drives the base" in p
