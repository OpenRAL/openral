"""ADR-0053 — the runner's pure starting-pose dispatch decision.

Pins the precedence (approach > reset > none) and the fatal-on-failure contract
that makes a failed collision-aware approach abort the ExecuteSkill goal while a
failed legacy snap only warns. Pure — no ROS, always runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the pure module directly: the package __init__ eagerly imports the
# lifecycle node, which needs a fully sourced ROS 2 install — unavailable in a
# plain unit-test env. The dispatch logic itself has no ROS dependency.
_MOD = Path(__file__).resolve().parents[1] / "openral_rskill_ros" / "_starting_pose.py"
_NAME = "openral_rskill_ros_starting_pose_adr0051"
_spec = importlib.util.spec_from_file_location(_NAME, _MOD)
assert _spec is not None and _spec.loader is not None
_starting_pose = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = _starting_pose  # slotted dataclass introspection needs this
_spec.loader.exec_module(_starting_pose)
resolve_starting_pose_action = _starting_pose.resolve_starting_pose_action
joint_names_from_goal_json = _starting_pose.joint_names_from_goal_json
moveit_joint_goal_override = _starting_pose.moveit_joint_goal_override

_POSE = [0.0, -1.2, 1.2, -1.0, -1.4, 0.0]
_APPROACH = "rskills/rskill-moveit-joints"


def test_approach_preferred_and_fatal_when_both_wired() -> None:
    action = resolve_starting_pose_action(
        approach_skill_id=_APPROACH,
        reset_to_pose_service="/openral/ur5e/reset_to_pose",
        starting_pose=_POSE,
    )
    assert action.mode == "approach"
    assert action.fatal_on_failure is True
    assert action.pose == _POSE


def test_reset_used_and_best_effort_when_only_reset_wired() -> None:
    action = resolve_starting_pose_action(
        approach_skill_id="",
        reset_to_pose_service="/openral/ur5e/reset_to_pose",
        starting_pose=_POSE,
    )
    assert action.mode == "reset"
    assert action.fatal_on_failure is False


def test_none_when_nothing_wired() -> None:
    action = resolve_starting_pose_action(
        approach_skill_id="",
        reset_to_pose_service="",
        starting_pose=_POSE,
    )
    assert action.mode == "none"


def test_none_when_no_starting_pose_even_when_wired() -> None:
    for empty in (None, []):
        action = resolve_starting_pose_action(
            approach_skill_id=_APPROACH,
            reset_to_pose_service="/openral/ur5e/reset_to_pose",
            starting_pose=empty,
        )
        assert action.mode == "none"
        assert action.pose == []


# ── MoveIt goal shaping (ADR-0054 joint block) ───────────────────────────────

_GOAL_JSON = (
    '{"joint": {"group_name": "panda_arm", '
    '"joint_names": ["panda_joint1", "panda_joint2"], "positions": [0.0, -0.785]}}'
)


def test_joint_names_extracted_in_manifest_order() -> None:
    assert joint_names_from_goal_json(_GOAL_JSON) == ["panda_joint1", "panda_joint2"]


def test_joint_names_raises_on_malformed_goal_json() -> None:
    import pytest

    with pytest.raises(ValueError, match="joint_names"):
        joint_names_from_goal_json('{"request": {}}')


def test_override_retargets_joint_positions_only() -> None:
    import json

    # The override replaces joint.positions; the manifest's joint_names order is
    # authoritative and preserved by the deep-merge.
    override = json.loads(moveit_joint_goal_override(["panda_joint1", "panda_joint2"], [0.3, -0.4]))
    assert override == {"joint": {"positions": [0.3, -0.4]}}


def test_override_raises_on_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError, match="length"):
        moveit_joint_goal_override(["panda_joint1"], [0.3, -0.4])
