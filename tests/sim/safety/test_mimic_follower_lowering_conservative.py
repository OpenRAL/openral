# SPDX-License-Identifier: Apache-2.0
"""Keeping mimic followers in the lowering is at least as conservative as dropping them.

``modelled_links`` now includes every ``mimic_joints`` child, so ``lower_robot`` keeps
the follower fingers' geometry instead of filtering it out. This proves, on a real
lowering of a manifest with six mimic followers (``droid_franka_robotiq``, SRDF ACM so
the pairs are deterministic), that the change only ever adds checked geometry:

* every geometry the lowering emitted without followers is emitted unchanged;
* every added geometry sits on a mimic follower;
* no pair between two links the old lowering modelled becomes newly allowed.

Real URDF/SRDF, real fitter; skips only when the lowering extras are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yourdfpy", reason="URDF lowering extras not installed")
pytest.importorskip("trimesh", reason="URDF lowering extras not installed")

from openral_core import RobotDescription
from openral_safety.urdf_lowering import lower_robot

pytestmark = pytest.mark.sim

_MANIFEST = Path(__file__).resolve().parents[3] / "robots" / "droid_franka_robotiq" / "robot.yaml"


def test_mimic_followers_only_add_checked_geometry() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    followers = {m.child_link for m in robot.mimic_joints}
    assert len(followers) == 6
    with_followers = lower_robot(robot, manifest_dir=_MANIFEST.parent)
    without = lower_robot(
        robot.model_copy(update={"mimic_joints": []}), manifest_dir=_MANIFEST.parent
    )

    old = without.collision_geometry
    new = with_followers.collision_geometry
    assert all(g in new for g in old), "a geometry the old lowering emitted changed or vanished"
    added = [g for g in new if g not in old]
    assert added, "the followers gained no geometry"
    assert {g.link_name for g in added} <= followers

    old_links = {g.link_name for g in old}
    old_pairs = {frozenset(p) for p in without.allowed_collision_pairs}
    newly_allowed = {
        frozenset(p)
        for p in with_followers.allowed_collision_pairs
        if set(p) <= old_links and frozenset(p) not in old_pairs
    }
    assert not newly_allowed, f"pairs between already-modelled links newly allowed: {newly_allowed}"
