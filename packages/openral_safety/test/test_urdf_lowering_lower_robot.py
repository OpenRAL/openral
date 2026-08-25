"""lower_robot: SRDF precedence, sampling fallback, scoping, geometry/acm flags.

The top-level entry that ties geometry + ACM together (part of the geometric
safety collision-checking system). The panda
oracle: lowering panda_mobile against the Franka SRDF must reproduce the SRDF arm
disables exactly — including the link1↔link4 "Never" pair whose absence
false-E-stopped a live pi05 episode. Real manifests + real SRDF, no mocks (§1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import AssetRefs
from openral_safety.urdf_lowering import lower_robot

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

_PANDA_SRDF = Path("/opt/ros/jazzy/share/moveit_resources_panda_moveit_config/config/panda.srdf")
_PANDA_MOBILE_DIR = Path("robots/panda_mobile")

# Exactly the Franka SRDF's arm-link (1-7) disables — no more.
#
# This set used to carry a 16th pair, link5↔link7, described as a "capsule-junction
# extra" that the sweep had certified as always-colliding. It was never
# always-colliding: measured on the URDF's own collision meshes over
# (panda_joint6, panda_joint7) — the only joints that move the pair — 914 of 14641
# poses interpenetrate by up to 48.3 mm, and 13.5% of that space separates the
# boxes outright. MoveIt agrees and emits no row for it. The old verdict came from
# a sweep that modelled each box as its inscribed sphere and drew 2000 random
# points from the arm's full 7-D joint box (issue #155).
#
# `lower_robot` now proves always-colliding instead of sampling it, so it no
# longer invents this pair. panda_mobile still *ships* the exemption — as an
# explicit `reason="User"` row in its own SRDF, carrying the measurement and the
# residual risk — which is why the with-manifest-SRDF path below still sees 16.
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
    """With srdf_path cleared → the no-SRDF fallback over the robot's own geometry.

    With neither mesh ground truth nor a human, only two justifications are
    available, and the result must contain nothing else:

    * **adjacent** — link5↔link6 is joint-connected, so it goes in;
    * **always-colliding** — nothing on this arm qualifies, because nothing on it
      is *provably* colliding at every reachable pose.

    Both of the pairs this asserts are absent are absent for a reason worth
    keeping distinct. link1↔link4 is (per MoveIt's mesh sweep) a never-collide
    pair, and a geometric sweep cannot prove a negative — so it stays CHECKED.
    link5↔link7 is the opposite: it genuinely collides in part of its range
    (issue #155), so it is not always-colliding either, and exempting it is a
    human decision recorded in the SRDF — never something this path may invent.
    """
    base = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    robot = base.model_copy(update={"assets": base.assets.model_copy(update={"srdf": None})})
    result = lower_robot(robot, acm_only=True, manifest_dir=_PANDA_MOBILE_DIR)
    assert result.acm_source == "sampling"
    pairs = set(result.allowed_collision_pairs)
    assert ("panda_link5", "panda_link6") in pairs  # adjacent
    assert ("panda_link1", "panda_link4") not in pairs  # never-collide → stays CHECKED
    assert ("panda_link5", "panda_link7") not in pairs  # sometimes-collides → stays CHECKED
    # Nothing but adjacency survives on this arm; spell it out so a future change
    # that starts exempting pairs here has to say so.
    adjacent = {frozenset({f"panda_link{i}", f"panda_link{i + 1}"}) for i in range(1, 7)}
    assert _arm(result.allowed_collision_pairs) == adjacent
    # Determinism (the --check linchpin): identical across runs, and no RNG at all.
    again = lower_robot(robot, acm_only=True)
    assert result.allowed_collision_pairs == again.allowed_collision_pairs


def test_geometry_only_emits_capsules_no_acm() -> None:
    robot = RobotDescription.from_yaml("robots/panda_mobile/robot.yaml")
    result = lower_robot(robot, geometry_only=True)
    assert result.allowed_collision_pairs == []
    assert result.collision_geometry, "geometry_only must still emit collision_geometry"
    assert all(g.shape.radius_m > 0.0 for g in result.collision_geometry)


def test_lower_robot_requires_urdf_path() -> None:
    """A manifest with neither a URDF NOR an MJCF asset raises, never guesses."""
    # openarm has no urdf but DOES have an MJCF (lowers via that path); clear BOTH
    # asset refs to hit the no-source error.
    robot = RobotDescription.from_yaml("robots/openarm/robot.yaml").model_copy(
        update={"assets": AssetRefs()}
    )
    assert robot.assets.urdf is None
    assert not robot.assets.mjcf
    with pytest.raises(ROSConfigError, match="urdf"):
        lower_robot(robot, acm_only=True)


def test_lower_robot_refuses_empty_geometry_from_unresolvable_meshes(tmp_path) -> None:
    """A URDF whose collision meshes don't resolve raises — never an empty model.

    Regression for the vendored-URDF portability bug: ur5e/ur10e/rizon4 shipped
    mesh paths absolutized into the vendoring machine's cache, so on any other
    host every mesh silently dropped and the lowering emitted an empty ACM (and
    zero geometry — a kernel that checks nothing). The lowering must fail loudly
    instead (§1.4). Real manifest + real (broken) URDF file, no mocks (§1.11).
    """
    import warnings

    import yaml

    urdf = tmp_path / "broken.urdf"
    urdf.write_text(
        '<robot name="broken"><link name="base_link"><collision><geometry>'
        '<mesh filename="/nonexistent/machine/specific/base.stl"/>'
        "</geometry></collision></link></robot>"
    )
    data = yaml.safe_load(Path("robots/ur5e/robot.yaml").read_text())
    data["assets"]["urdf"]["ref"] = f"file:{urdf}"
    data["assets"].pop("srdf", None)
    robot = RobotDescription.model_validate(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the per-mesh "not found" warning is expected
        with pytest.raises(ROSConfigError, match="zero collision geometry"):
            lower_robot(robot)
