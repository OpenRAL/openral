"""ADR-0054 — the joint-space MoveGroup builder (`goal_builder: "joint"`).

Pins the pure lowering of a ``joint`` goal block into a MoveGroup
``joint_constraints`` entry — the clean, LLM-facing replacement for the
hand-written constraints JSON the pre-ADR-0054 `openral-moveit-plan-arm`
(now `rskill-moveit-joints`) used to ship.
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill.joint_goal_rskill import joint_constraints_from_block


def test_joint_block_lowers_to_one_constraint_per_joint() -> None:
    entry = joint_constraints_from_block(
        {
            "group_name": "panda_arm",
            "joint_names": ["panda_joint1", "panda_joint2"],
            "positions": [0.0, -0.785],
            "position_tolerance_rad": 0.001,
        }
    )
    jcs = entry["joint_constraints"]
    assert [(c["joint_name"], c["position"]) for c in jcs] == [
        ("panda_joint1", 0.0),
        ("panda_joint2", -0.785),
    ]
    assert jcs[0]["tolerance_above"] == pytest.approx(0.001)
    assert jcs[0]["tolerance_below"] == pytest.approx(0.001)
    assert jcs[0]["weight"] == pytest.approx(1.0)


def test_joint_block_rejects_length_mismatch() -> None:
    with pytest.raises(ROSConfigError, match="length"):
        joint_constraints_from_block({"joint_names": ["panda_joint1"], "positions": [0.0, -0.785]})


def test_joint_block_requires_names_and_positions() -> None:
    with pytest.raises(ROSConfigError, match="joint_names"):
        joint_constraints_from_block({"positions": [0.0]})
