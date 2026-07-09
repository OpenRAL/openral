"""SRDF disable_collisions → ACM, against the real moveit_resources panda.srdf."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_safety.urdf_lowering import parse_srdf_disabled_pairs

_PANDA_SRDF = Path("/opt/ros/jazzy/share/moveit_resources_panda_moveit_config/config/panda.srdf")

pytestmark = pytest.mark.skipif(
    not _PANDA_SRDF.is_file(), reason="moveit_resources panda.srdf not installed"
)


def test_parses_panda_disable_collisions() -> None:
    pairs = parse_srdf_disabled_pairs(str(_PANDA_SRDF))
    assert frozenset({"panda_link1", "panda_link2"}) in pairs  # Adjacent
    assert frozenset({"panda_link1", "panda_link4"}) in pairs  # Never (the pair that regressed)
    assert frozenset({"panda_hand", "panda_leftfinger"}) in pairs
    assert len(pairs) == 34


def test_scoping_to_arm_links_drops_hand_finger_rows() -> None:
    pairs = parse_srdf_disabled_pairs(str(_PANDA_SRDF))
    arm = {
        p for p in pairs if all(link.startswith("panda_link") and link[10:].isdigit() for link in p)
    }
    assert frozenset({"panda_link5", "panda_link6"}) in arm
    assert frozenset({"panda_hand", "panda_link7"}) not in arm
