# SPDX-License-Identifier: Apache-2.0
"""The Galaxea A1's kernel finger geometry covers each finger's whole 30 mm stroke.

The A1's fingers are vendor ``fixed`` joints that slide 0-30 mm with no joint state, so
the kernel poses them closed forever. ``FixedAttachment.sweep_xyz`` makes the lowering
fit each finger over its travel instead; this proves the committed model does:

* every vertex of the finger's own URDF mesh, closed AND fully open, lies inside the
  committed exact hull (the hull is convex, so it holds the whole straight stroke);
* an obstacle point only the fully open finger occupies is inside it, and was outside
  the closed finger;
* the MJCF lowering refuses a declared sweep rather than silently dropping it.

Real manifest, real vendor URDF (pinned fetch); skips only when that is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("yourdfpy", reason="URDF lowering extras not installed")
from openral_core import LinkCollisionGeometry, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_core.kinematics import transform_from_xyz_rpy
from openral_safety.urdf_lowering import _link_clouds, lower_robot_from_mjcf
from scipy.spatial import ConvexHull

pytestmark = pytest.mark.sim

_MANIFEST = Path(__file__).resolve().parents[3] / "robots" / "galaxea_a1" / "robot.yaml"
_TOL_M = 1e-6


def _committed_hull(robot: RobotDescription, link: str) -> ConvexHull:
    """The finger's exact hull, as the kernel holds it, in the finger's own frame."""
    (geom,) = [g for g in robot.collision_geometry if g.link_name == link]
    assert geom.tight_geometry is not None, f"{link} lost its exact hull"
    place = np.asarray(transform_from_xyz_rpy(geom.origin_xyz_rpy[:3], geom.origin_xyz_rpy[3:]))
    verts = np.asarray(geom.tight_geometry.hull_vertices_m)
    return ConvexHull(verts @ place[:3, :3].T + place[:3, 3])


def _outside_m(hull: ConvexHull, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each point's largest signed distance past the hull's facets (<= 0 inside)."""
    distance: NDArray[np.float64] = np.max(
        points @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1
    )
    return distance


@pytest.fixture(scope="module")
def robot_and_meshes() -> tuple[RobotDescription, dict[str, NDArray[np.float64]]]:
    from openral_hal._galaxea_a1_assets import ensure_a1_urdf

    try:
        urdf = ensure_a1_urdf()
    except ROSConfigError as exc:
        pytest.skip(f"Galaxea A1 vendor files not fetchable on this host: {exc}")
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    meshes = {
        link: np.asarray(cloud.vertices)
        for link, cloud in _link_clouds(str(urdf), {}, None)
        if link in ("gripper1", "gripper2") and not isinstance(cloud, LinkCollisionGeometry)
    }
    return robot, meshes


@pytest.mark.parametrize("link", ["gripper1", "gripper2"])
def test_finger_hull_holds_the_closed_and_fully_open_finger(
    robot_and_meshes: tuple[RobotDescription, dict[str, NDArray[np.float64]]], link: str
) -> None:
    robot, meshes = robot_and_meshes
    (sweep,) = [a.sweep_xyz for a in robot.fixed_attachments if a.child_link == link]
    assert sweep == (0.0, 0.0, 0.03)
    hull = _committed_hull(robot, link)
    closed = meshes[link]
    fully_open = closed + np.asarray(sweep)
    assert _outside_m(hull, closed).max() <= _TOL_M
    assert _outside_m(hull, fully_open).max() <= _TOL_M

    # An obstacle the open finger reaches and the closed one never did.
    closed_hull = ConvexHull(closed)
    tip = fully_open[np.argmax(fully_open @ np.asarray(sweep))]
    assert _outside_m(closed_hull, tip[None])[0] > 0.02
    assert _outside_m(hull, tip[None])[0] <= _TOL_M


def test_mjcf_lowering_refuses_a_declared_sweep() -> None:
    robot = RobotDescription.from_yaml(str(_MANIFEST))
    with pytest.raises(ROSConfigError, match="sweep_xyz"):
        lower_robot_from_mjcf(robot, manifest_dir=_MANIFEST.parent)
