# SPDX-License-Identifier: Apache-2.0
"""A mimic follower finger is a modelled link: the lowering gives it kernel geometry.

The sim attachment producers name each gripper's follower finger as a touch link, and
the safety kernel rejects a whole attachment set naming a link it does not know. So
``lower_robot`` must keep ``mimic_joints`` children in the model, exactly like the
joint and fixed-attachment links: dropping them would delete checked geometry (less
conservative) and stall every grasp.

Real manifest + its pinned URDF (aloha_agilex: the ``*_link8`` mirror fingers);
skips when the lowering dependencies or the RoboTwin files are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("trimesh")
pytest.importorskip("yourdfpy")

from openral_core import RobotDescription
from openral_core.assets import AssetFetchError
from openral_safety.urdf_lowering import lower_robot

_ROOT = Path(__file__).resolve().parents[2]


def test_mimic_followers_keep_their_kernel_geometry() -> None:
    manifest = _ROOT / "robots" / "aloha_agilex" / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    followers = {m.child_link for m in robot.mimic_joints}
    assert followers == {"fl_link8", "fr_link8"}
    try:
        lowered = lower_robot(robot, manifest_dir=manifest.parent, geometry_only=True)
    except AssetFetchError as exc:
        pytest.skip(f"RoboTwin files not fetchable on this host: {exc}")
    lowered_links = {g.link_name for g in lowered.collision_geometry}
    assert followers <= lowered_links, followers - lowered_links
    # ...and the committed model the kernel loads carries them too.
    assert followers <= {g.link_name for g in robot.collision_geometry}
