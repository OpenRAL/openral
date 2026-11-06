"""Every link's URDF meshes sit inside the link's committed collision primitives.

``test_collision_geometry_enclosure.py`` checks the MJCF twin's meshes. A robot
lowered from its MJCF can also declare a URDF whose meshes differ for the same
link, and that URDF is what robot_state_publisher, Foxglove and the Isaac import
draw. OpenArm's link6 is the case that found this: ``enactic/openarm_description``
is 10.6 mm (visual) to 13.0 mm (collision) larger at one corner than
``enactic/openarm_mujoco``, and a hull fitted to the twin alone missed it. The
lowering now fits each link to the union of both meshes
(``urdf_lowering._urdf_link_meshes``); this test holds the committed geometry to
that.

Real manifests and real upstream meshes (pinned clones); skipped when the URDF's
``package://`` meshes cannot be fetched on this host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("yourdfpy")

from openral_core import LinkCollisionGeometry, RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_safety.urdf_lowering import _urdf_link_meshes
from scipy.spatial.transform import Rotation

from tests.unit._collision_kernel_verdict import tight_escape_per_point_m

REPO = Path(__file__).resolve().parents[2]
# A refined box's DOP and hull are exact; the 4-dp rendering of the manifest moves
# them by well under this.
TOLERANCE_M = 1e-4


def _box_frame(geom: LinkCollisionGeometry, pts: npt.NDArray[np.float64]) -> Any:
    x, y, z, r, p, yaw = geom.origin_xyz_rpy
    rot = Rotation.from_euler("xyz", [r, p, yaw]).as_matrix()
    return (pts - np.array([x, y, z])) @ rot


def _escape(geom: LinkCollisionGeometry, pts: npt.NDArray[np.float64]) -> Any:
    """Per point, how far outside this primitive (its refinement when it has one)."""
    local = _box_frame(geom, pts)
    half = np.asarray(geom.shape.half_extents_m)  # type: ignore[union-attr]  # reason: OpenArm links are boxes
    box = (np.abs(local) - half).max(axis=1)
    if geom.tight_geometry is None:
        return box
    return np.maximum(box, tight_escape_per_point_m(geom, local))


@pytest.mark.parametrize("robot_id", ["openarm", "openamrobot_v2"])
def test_every_urdf_mesh_vertex_is_inside_its_links_primitives(robot_id: str) -> None:
    manifest = REPO / "robots" / robot_id / "robot.yaml"
    robot = RobotDescription.from_yaml(str(manifest))
    by_link: dict[str, list[LinkCollisionGeometry]] = {}
    for g in robot.collision_geometry:
        by_link.setdefault(g.link_name, []).append(g)
    arm_links = {ln: 0 for ln in by_link if ln.startswith("openarm_")}
    try:
        meshes = _urdf_link_meshes(robot, arm_links, manifest.parent)
    except ROSConfigError as exc:
        from openral_core.assets import AssetRefError

        if not isinstance(exc.__cause__, AssetRefError):
            raise  # a joint-origin mismatch or unloadable geometry is a regression
        pytest.skip(f"{robot_id}: URDF meshes unavailable (asset fetch): {exc}")
    assert meshes, f"{robot_id}: no arm link matched a URDF link"
    worst: dict[str, float] = {}
    for link, mesh in meshes.items():
        pts = np.asarray(mesh.vertices, dtype=float)
        escape = np.min([_escape(g, pts) for g in by_link[link]], axis=0)
        worst[link] = float(escape.max())
    outside = {ln: round(d * 1000, 2) for ln, d in worst.items() if d > TOLERANCE_M}
    assert not outside, f"{robot_id}: URDF mesh outside its primitives (mm): {outside}"
