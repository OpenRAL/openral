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


def test_a_mirror_finger_is_coupled_through_the_twins_actuator_map() -> None:
    """``twin_coupled_joints`` resolves the leader's actuator as the twin does (#754 review).

    With the gripper entry's own ``actuator_index`` unset, the twin drives the leader through
    ``sim.actuator_index``; the mirror coupling must still be found.
    """
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("gym_aloha", reason="the ALOHA twin ships with gym-aloha")
    from openral_core.assets import resolve_asset
    from openral_safety.urdf_lowering import twin_coupled_joints

    robot = RobotDescription.from_yaml(str(_ROOT / "robots" / "aloha_bimanual" / "robot.yaml"))
    assert robot.sim is not None and robot.assets.mjcf is not None
    model = mujoco.MjModel.from_xml_path(str(resolve_asset(robot.assets.mjcf, "mjcf")))
    grippers = [g.model_copy(update={"actuator_index": None}) for g in robot.sim.grippers]
    implicit = robot.model_copy(update={"sim": robot.sim.model_copy(update={"grippers": grippers})})
    leader, mirror = (
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in ("vx300s_left/left_finger", "vx300s_left/right_finger")
    )
    for desc in (robot, implicit):
        assert (mirror, 0.0, -1.0) in twin_coupled_joints(desc, model)[leader]
