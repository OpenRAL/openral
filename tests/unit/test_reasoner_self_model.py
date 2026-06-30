"""Tests for the reasoner robot self-model (ADR-0072 Decision 2.1).

Covers:
- :func:`~openral_reasoner.context.render_robot_self_model` against real
  ``robots/<id>/robot.yaml`` fixtures (no synthetic placeholders, CLAUDE.md §1.11).
- :class:`~openral_reasoner.context.ContextRenderer` surfacing the ``## ROBOT``
  section when (and only when) a self-model is supplied.

Run with:
    uv run pytest tests/unit/test_reasoner_self_model.py -v
"""

from __future__ import annotations

import pathlib

from openral_core import RobotDescription
from openral_reasoner.context import ContextRenderer, render_robot_self_model

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO_ROOT / "robots" / name / "robot.yaml"))


def test_self_model_summarises_arm_robot() -> None:
    """An arm robot's self-model names its embodiment, DOF, gripper, and cameras."""
    text = render_robot_self_model(_robot("so100_follower"))
    assert "name: so100_follower" in text
    assert "dof:" in text and "joints" in text
    # so100 has a parallel gripper end-effector and at least one camera.
    assert "end_effectors:" in text
    assert "capabilities:" in text and "vision" in text
    # Deterministic: same input → identical text.
    assert render_robot_self_model(_robot("so100_follower")) == text


def test_self_model_reports_mobile_base() -> None:
    """A mobile manipulator surfaces locomotion + payload so the LLM can judge reach."""
    text = render_robot_self_model(_robot("panda_mobile"))
    assert "name: panda_mobile" in text
    assert "locomotion:" in text  # wheeled base
    assert "control_modes:" in text


def test_context_renderer_includes_robot_section_when_supplied() -> None:
    """ContextRenderer renders ``## ROBOT`` iff a self-model is supplied."""
    model = render_robot_self_model(_robot("so100_follower"))
    with_model = ContextRenderer(robot_model=model).render(world_state=None)
    assert with_model.startswith("## ROBOT")
    assert "name: so100_follower" in with_model
    assert "## WORLD_STATE" in with_model  # other sections still present

    without = ContextRenderer().render(world_state=None)
    assert "## ROBOT" not in without
    assert without.startswith("## WORLD_STATE")
