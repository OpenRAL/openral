"""What each skill tool tells the LLM about grasp/place targets under object tracking.

With the deploy's ``object_tracking`` on, the runner refuses a manipulation goal that names
no object to grasp, so the tool of a skill that actuates an arm / gripper says
``grasp_target`` is required, and a base-only skill's says neither target applies. With
tracking off every description is exactly what it was. Real in-tree manifests and robots.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription, RSkillManifest
from openral_reasoner.palette import build_tool_palette
from openral_reasoner.tool_use import (
    _GRASP_TARGET_AND_HAND_REQUIRED,
    _GRASP_TARGET_REQUIRED,
    _OBJECT_TARGETS_NOT_APPLICABLE,
    _format_skill_tool_description,
)

_REPO = Path(__file__).resolve().parents[2]


def _descriptions(
    skill: str, robot: str, *, tracking: bool, hand_required: bool = False
) -> dict[str, str]:
    manifest = RSkillManifest.from_yaml(str(_REPO / "rskills" / skill / "rskill.yaml"))
    description = RobotDescription.from_yaml(str(_REPO / "robots" / robot / "robot.yaml"))
    palette = build_tool_palette(
        installed_skills=[manifest],
        robot_capabilities=description.capabilities,
        object_tracking=tracking,
        grasp_hand_required=hand_required,
    )
    assert palette.skills, f"{skill} does not match {robot}"
    return {e.rskill_id: _format_skill_tool_description(e) for e in palette.skills}


@pytest.mark.parametrize(
    ("skill", "robot", "hand_required", "sentence"),
    [
        ("pi05-metal-pick-place", "makermods_metal", False, _GRASP_TARGET_REQUIRED),
        ("pi05-metal-pick-place", "makermods_metal", True, _GRASP_TARGET_AND_HAND_REQUIRED),
        ("rskill-nav2-navigate-to-pose", "openamrobot_v2", False, _OBJECT_TARGETS_NOT_APPLICABLE),
        ("rskill-nav2-navigate-to-pose", "openamrobot_v2", True, _OBJECT_TARGETS_NOT_APPLICABLE),
    ],
)
def test_tracking_appends_the_target_rule_to_the_unchanged_description(
    skill: str, robot: str, hand_required: bool, sentence: str
) -> None:
    off = _descriptions(skill, robot, tracking=False, hand_required=hand_required)
    on = _descriptions(skill, robot, tracking=True, hand_required=hand_required)
    assert off.keys() == on.keys()
    for rskill_id, text in on.items():
        assert text == f"{off[rskill_id]} {sentence}"
        assert "Object tracking" not in off[rskill_id]


def test_the_manipulation_rule_says_required_and_names_a_detector_class() -> None:
    for rule in (_GRASP_TARGET_REQUIRED, _GRASP_TARGET_AND_HAND_REQUIRED):
        assert "grasp_target is REQUIRED" in rule
        assert "as a detector names it" in rule
    # The deployment supplies the hand: the LLM never guesses a learned policy's hand.
    assert "Leave its contact_links unset" in _GRASP_TARGET_REQUIRED
    assert "the deployment supplies the grasping hand" in _GRASP_TARGET_REQUIRED
    # Several hands and none declared: the call must name it.
    assert "contact_links to the grasping hand's links" in _GRASP_TARGET_AND_HAND_REQUIRED
    assert "grasp_target and place_target do not apply" in _OBJECT_TARGETS_NOT_APPLICABLE
