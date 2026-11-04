"""select_lowering: a partially resolving URDF routes to the MJCF twin, nothing else moves.

OpenAMRobot 2.0's URDF mixes geometry that loads (the column, carriage and
plate boxes) with links that only half load: ``base_link`` carries a box
``<collision>`` next to ``package://openamrobot_description`` visual meshes no
resolver serves. ``lower_link_geometry`` refuses such a link (fitting the half
that loaded would under-cover it), and it must keep refusing. But the routing
question ``select_lowering`` asks is only "is this URDF usable for lowering?",
and a URDF that the lowering would refuse is not: the robot has an MJCF twin
built from the same URDF, so it lowers from that, as the OpenArm does.

The pinned route table proves the change moves no other robot: every in-tree
manifest selects the route it selected before (measured 2026-11-04 on the
pre-change code; openamr_v2 raised there). A new robot needs a row, which is
the point: a route is what the C++ kernel's collision model comes from.
Real manifests and URDFs only (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_safety.urdf_lowering import (
    _urdf_has_collision_geometry,
    lower_link_geometry,
    select_lowering,
)

pytest.importorskip("yourdfpy")
pytest.importorskip("trimesh")

_ROBOTS = Path("robots")
_OPENAMR_URDF = _ROBOTS / "openamr_v2" / "openamr_v2.urdf"

# robot -> route; None = select_lowering raises ROSConfigError (no lowerable asset).
_ROUTES: dict[str, str | None] = {
    "aloha_agilex": None,
    "aloha_bimanual": "mjcf",
    "anvil_openarm_v2": "mjcf",
    "droid_franka_robotiq": "srdf",
    "franka_panda": "srdf",
    "g1": "sampling",
    "galaxea_a1": None,
    "google_robot": None,
    "gr1": "sampling",
    "h1": "sampling",
    "openamr_v2": "mjcf",  # raised on a partially resolving base_link before
    "openarm": "mjcf",
    "panda_mobile": "srdf",
    "panda_mobile_vslam": "srdf",
    "pusht_2d": None,
    "r1pro": None,
    "rizon4": "srdf",
    "sawyer": "mjcf",
    "so100_follower": "sampling",
    "so101_follower": "sampling",
    "ur10e": "srdf",
    "ur5e": "srdf",
    "widowx": "mjcf",
}


def test_route_table_covers_every_manifest() -> None:
    assert {p.parent.name for p in _ROBOTS.glob("*/robot.yaml")} == set(_ROUTES)


@pytest.mark.parametrize("robot_id", sorted(_ROUTES))
def test_every_robot_keeps_its_route(robot_id: str) -> None:
    manifest = _ROBOTS / robot_id / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    expected = _ROUTES[robot_id]
    if expected is None:
        with pytest.raises(ROSConfigError, match="no lowerable asset"):
            select_lowering(robot, manifest_dir=manifest.parent)
        return
    assert select_lowering(robot, manifest_dir=manifest.parent) == expected


def test_partially_resolving_urdf_is_not_usable_for_routing() -> None:
    assert _urdf_has_collision_geometry(str(_OPENAMR_URDF)) is False


def test_partially_resolving_link_still_refuses_to_lower() -> None:
    """The routing answer never weakens the lowering's own fail-closed raise."""
    with pytest.raises(ROSConfigError, match="did not load while the rest of the link did"):
        lower_link_geometry(str(_OPENAMR_URDF))
