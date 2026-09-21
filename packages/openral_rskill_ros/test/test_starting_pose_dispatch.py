"""Pure MoveIt starting-pose goal shaping."""

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
joint_names_from_goal_json = _starting_pose.joint_names_from_goal_json
moveit_joint_goal_override = _starting_pose.moveit_joint_goal_override

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
