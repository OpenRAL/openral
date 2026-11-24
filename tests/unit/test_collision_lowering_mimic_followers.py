# SPDX-License-Identifier: Apache-2.0
"""A mimic follower finger is a modelled link: the lowering gives it kernel geometry.

The sim attachment producers name each gripper's follower finger as a touch link, and
the safety kernel rejects a whole attachment set naming a link it does not know. So
the links ``lower_robot`` may fit geometry onto (``modelled_links``) include every
``mimic_joints`` child, exactly like the joint and fixed-attachment links: dropping
them would delete checked geometry (less conservative) and stall every grasp.

Real manifests only; nothing is fetched (the committed model is the kernel's input).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_safety.urdf_lowering import modelled_links

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("robot", ["aloha_agilex", "droid_franka_robotiq", "so101_follower"])
def test_mimic_followers_are_modelled_links(robot: str) -> None:
    description = RobotDescription.from_yaml(str(_ROOT / "robots" / robot / "robot.yaml"))
    followers = {m.child_link for m in description.mimic_joints}
    assert followers, f"{robot} declares no mimic follower"
    assert followers <= modelled_links(description)


def test_aloha_agilex_followers_carry_committed_kernel_geometry() -> None:
    description = RobotDescription.from_yaml(str(_ROOT / "robots" / "aloha_agilex" / "robot.yaml"))
    followers = {m.child_link for m in description.mimic_joints}
    assert followers == {"fl_link8", "fr_link8"}
    assert followers <= {g.link_name for g in description.collision_geometry}
