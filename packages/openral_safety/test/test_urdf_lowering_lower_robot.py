"""lower_robot: SRDF precedence, sampling fallback, scoping, geometry/acm flags.

The top-level entry that ties geometry + ACM together (ADR-0030). The panda
oracle: lowering panda_mobile against the Franka SRDF must reproduce the SRDF arm
disables exactly — including the link1↔link4 "Never" pair whose absence
false-E-stopped a live pi05 episode. Real manifests + real SRDF, no mocks (§1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_safety.urdf_lowering import lower_robot

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

_PANDA_SRDF = Path("/opt/ros/jazzy/share/moveit_resources_panda_moveit_config/config/panda.srdf")

# The Franka SRDF arm-link (1-7) disables PLUS the link5↔link7 capsule-junction
# extra (the short link6 makes panda_mobile's link5/link7 capsules always overlap;
# a mesh-based SRDF omits it, but the capsule kernel needs it or it E-stops every
# step). This 16-pair set matches the hand-aligned phase4b ACM (247cfb5) — now
# derived automatically from the geometry rather than hand-listed.
_EXPECTED_PANDA_ARM_ACM = {
    frozenset(p)
    for p in (
        ("panda_link1", "panda_link2"),
        ("panda_link2", "panda_link3"),
        ("panda_link3", "panda_link4"),
        ("panda_link4", "panda_link5"),
        ("panda_link5", "panda_link6"),
        ("panda_link6", "panda_link7"),
        ("panda_link1", "panda_link3"),
        ("panda_link1", "panda_link4"),
        ("panda_link2", "panda_link4"),
        ("panda_link2", "panda_link6"),
        ("panda_link3", "panda_link5"),
        ("panda_link3", "panda_link6"),
        ("panda_link3", "panda_link7"),
        ("panda_link4", "panda_link6"),
        ("panda_link4", "panda_link7"),
        ("panda_link5", "panda_link7"),  # capsule-junction extra (always-colliding)
    )
}


def _arm(pairs: list[tuple[str, str]]) -> set[frozenset[str]]:
    return {
        frozenset(p)
        for p in pairs
        if all(ln.startswith("panda_link") and ln[10:].isdigit() for ln in p)
    }


@pytest.mark.skipif(not _PANDA_SRDF.is_file(), reason="panda.srdf not installed")
def test_lower_panda_mobile_acm_matches_srdf_arm_set() -> None:
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    result = lower_robot(robot, srdf_path=str(_PANDA_SRDF), acm_only=True)
    assert result.acm_source == "srdf"
    assert _arm(result.allowed_collision_pairs) == _EXPECTED_PANDA_ARM_ACM
    # The pair that regressed (false-E-stopped pi05) must be present.
    assert ("panda_link1", "panda_link4") in result.allowed_collision_pairs


@pytest.mark.skipif(not _PANDA_SRDF.is_file(), reason="panda.srdf not installed")
def test_acm_pairs_are_sorted_and_scoped_to_geometry_links() -> None:
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    result = lower_robot(robot, srdf_path=str(_PANDA_SRDF), acm_only=True)
    pairs = result.allowed_collision_pairs
    # Deterministic, sorted output; every link is a panda_mobile geometry link
    # (link0 / hand / finger SRDF rows are scoped out).
    assert pairs == sorted(pairs)
    geom_links = {g.link_name for g in robot.collision_geometry}
    for a, b in pairs:
        assert a in geom_links and b in geom_links


def test_lower_robot_falls_back_to_sampling_without_srdf() -> None:
    """With srdf_path cleared → sampling fallback against the robot's own geometry.

    The sweep runs against panda_mobile's hand-tuned manifest capsules. Conservative
    rule: adjacency is disabled and the always-colliding link5↔link7 junction is
    disabled, but a far never-colliding pair (link1↔link4) stays CHECKED — a sweep
    can't prove it never collides, so without an SRDF it is not auto-disabled.
    """
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml").model_copy(
        update={"srdf_path": None}
    )
    result = lower_robot(robot, acm_only=True)
    assert result.acm_source == "sampling"
    pairs = set(result.allowed_collision_pairs)
    assert ("panda_link5", "panda_link6") in pairs  # adjacent
    assert ("panda_link5", "panda_link7") in pairs  # always-colliding junction
    assert ("panda_link1", "panda_link4") not in pairs  # never-collide → stays CHECKED
    # Determinism (the --check linchpin): identical across runs.
    again = lower_robot(robot, acm_only=True)
    assert result.allowed_collision_pairs == again.allowed_collision_pairs


def test_geometry_only_emits_capsules_no_acm() -> None:
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    result = lower_robot(robot, geometry_only=True)
    assert result.allowed_collision_pairs == []
    assert result.collision_geometry, "geometry_only must still emit collision_geometry"
    assert all(g.shape.radius_m > 0.0 for g in result.collision_geometry)


def test_lower_robot_requires_urdf_path() -> None:
    """A manifest with neither urdf_path NOR a sim MJCF raises, never guesses."""
    # openarm has no urdf_path but DOES have an MJCF (lowers via that path), so clear
    # the sim block to hit the no-source error.
    robot = RobotDescription.from_yaml("robots/openarm/robot.yaml").model_copy(update={"sim": None})
    assert robot.urdf_path is None
    with pytest.raises(ValueError, match="urdf_path"):
        lower_robot(robot, acm_only=True)
