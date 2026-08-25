#!/usr/bin/env python3
"""Derive a robot manifest's ``tight_geometry`` blocks from its real collision meshes.

The safety kernel's arm-link-vs-world-voxel check runs a staged 26-DOP → exact
convex hull narrow phase for links that declare ``tight_geometry`` (see
``docs/reference/collision-hull-narrow-phase.md``). This tool is where that
geometry comes from. It is deliberately **not** a fit:

* the 26-DOP is the intersection of 26 *tangent* halfspaces ``u·x <= h_mesh(u)``,
  so the mesh is inside it by construction, with no optimiser tolerance anywhere;
* the hull is ``conv(mesh vertices)``, so containment is definitional;
* both are expressed in the manifest box's own local frame, and the tool refuses
  to emit anything that does not sit inside that box.

Mesh placement follows ``docs/reference/collision-tight-geometry.md`` §11: MuJoCo
folds mesh recentring into the geom frame, so vertices are placed by the GEOM
transform only. Applying ``mesh_pos`` as well double-counts it (PR #158 hit this).

Usage::

    # print the YAML fragment to paste into robots/<name>/robot.yaml
    python tools/generate_tight_geometry.py emit --robot robots/panda_mobile/robot.yaml

    # re-derive from the mesh and verify the manifest still matches (CI / tests)
    python tools/generate_tight_geometry.py check --robot robots/panda_mobile/robot.yaml

``check`` exits 0 when every declared block is reproduced and contains its mesh,
and 3 otherwise — the same fail-closed exit ``openral collision lower --write``
uses when it refuses to loosen a shipped envelope.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

    Points = npt.NDArray[Any]

REPO_ROOT = Path(__file__).resolve().parent.parent

# The MuJoCo geom that carries each manifest link's collision mesh. Manifest
# link names are the URDF/TF names; robosuite's MJCF uses bare `linkN`.
PANDA_GEOM_OF_LINK = {f"panda_link{i}": f"link{i}_collision" for i in range(1, 8)}

_PANDA_XML = "robosuite/models/assets/robots/panda/robot.xml"

ROBOT_MESH_SOURCES: dict[str, tuple[str, dict[str, str]]] = {
    "panda_mobile": (_PANDA_XML, PANDA_GEOM_OF_LINK),
    # Same arm, same links, same geometry -- and it must stay that way.
    # `tests/unit/test_collision_geometry_contracts.py` asserts the two
    # manifests' `panda_link*` entries are equal, so a mesh or fit change has to
    # land on both or fail.
    "panda_mobile_vslam": (_PANDA_XML, PANDA_GEOM_OF_LINK),
}


def _dop_axes() -> Points:
    import numpy as np
    from openral_core.schemas import DOP_AXES

    axes: Points = np.asarray(DOP_AXES, dtype=float)
    return axes


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> Points:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` -- the kernel's ``transform_from_xyz_rpy``."""
    import numpy as np

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rot: Points = rz @ ry @ rx
    return rot


def _robosuite_asset_root() -> Path:
    import robosuite

    return Path(robosuite.__file__).resolve().parent.parent


def link_mesh_in_box_frame(
    xml_path: Path, geom_name: str, origin_xyz_rpy: tuple[float, ...]
) -> Points:
    """Collision-mesh vertices of ``geom_name``, expressed in the manifest box's frame."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        msg = f"geom {geom_name!r} not found in {xml_path}"
        raise ValueError(msg)
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        msg = f"geom {geom_name!r} is not a mesh geom"
        raise ValueError(msg)
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start : start + count].astype(float)
    # MuJoCo folds mesh recentring into the geom frame; applying both would
    # double-count it. Assert the premise rather than trusting it.
    if not np.allclose(model.mesh_pos[mesh_id], model.geom_pos[gid], atol=1e-9):
        msg = (
            f"{geom_name}: mesh_pos != geom_pos, so the geom-only placement this tool "
            "relies on does not hold for this asset"
        )
        raise ValueError(msg)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    in_link = verts @ rot.reshape(3, 3).T + model.geom_pos[gid]
    translation = np.asarray(origin_xyz_rpy[:3], dtype=float)
    rotation = _rpy_to_matrix(*origin_xyz_rpy[3:6])
    in_box: Points = (in_link - translation) @ rotation
    return in_box


def derive_tight_geometry(points: Points, half_extents: tuple[float, ...]) -> dict[str, Any]:
    """Build the DOP slabs and (when it fits the budget) the exact hull.

    Returns a mapping ready for :class:`openral_core.schemas.TightCollisionGeometry`,
    plus the diagnostics a reviewer needs: vertex counts and the achieved inward
    margin of the DOP inside the shipped box.
    """
    import numpy as np
    from openral_core.schemas import MAX_TIGHT_HULL_VERTICES

    # reason: scipy ships no type stubs and scipy-stubs is not a workspace dependency.
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

    axes = _dop_axes()
    proj = points @ axes.T
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)

    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    fits_budget = len(hull_vertices) <= MAX_TIGHT_HULL_VERTICES

    he = np.asarray(half_extents, dtype=float)
    inward = float(np.minimum(he - hi[:3], he + lo[:3]).min())
    return {
        "dop_lo_m": [float(v) for v in lo],
        "dop_hi_m": [float(v) for v in hi],
        "hull_vertices_m": ([[float(c) for c in v] for v in hull_vertices] if fits_budget else []),
        "_hull_vertex_count": int(len(hull_vertices)),
        "_stage2": bool(fits_budget),
        "_dop_inward_margin_m": inward,
    }


def _load_robot(path: Path) -> Any:
    from openral_core.schemas import RobotDescription

    return RobotDescription.from_yaml(str(path))


def _sources_for(robot_name: str) -> tuple[Path, dict[str, str]]:
    if robot_name not in ROBOT_MESH_SOURCES:
        msg = (
            f"no mesh source registered for robot {robot_name!r}; add one to "
            "ROBOT_MESH_SOURCES with the MJCF path and the per-link geom names"
        )
        raise KeyError(msg)
    rel, geoms = ROBOT_MESH_SOURCES[robot_name]
    return _robosuite_asset_root() / rel, geoms


def _emit(robot_path: Path) -> int:
    robot = _load_robot(robot_path)
    xml, geoms = _sources_for(robot.name)
    print(f"# tight_geometry for {robot.name}, derived from {xml}")
    print("# GENERATED by tools/generate_tight_geometry.py -- do not hand-edit.")
    for geom in robot.collision_geometry:
        geom_name = geoms.get(geom.link_name)
        if geom_name is None or geom.shape.shape != "box":
            continue
        pts = link_mesh_in_box_frame(xml, geom_name, geom.origin_xyz_rpy)
        derived = derive_tight_geometry(pts, geom.shape.half_extents_m)
        print(f"\n# --- {geom.link_name} ---")
        print(
            f"#   mesh vertices {len(pts)}, hull vertices {derived['_hull_vertex_count']}, "
            f"stage 2 {'on' if derived['_stage2'] else 'OFF (over the vertex budget)'}, "
            f"DOP inward margin {derived['_dop_inward_margin_m'] * 1e3:.4f} mm"
        )
        print("    tight_geometry:")
        print("      dop_lo_m: [" + ", ".join(repr(v) for v in derived["dop_lo_m"]) + "]")
        print("      dop_hi_m: [" + ", ".join(repr(v) for v in derived["dop_hi_m"]) + "]")
        if derived["hull_vertices_m"]:
            print("      hull_vertices_m:")
            for v in derived["hull_vertices_m"]:
                print("        - [" + ", ".join(repr(c) for c in v) + "]")
        else:
            print("      hull_vertices_m: []")
    return 0


def _check(robot_path: Path) -> int:
    import numpy as np

    robot = _load_robot(robot_path)
    xml, geoms = _sources_for(robot.name)
    axes = _dop_axes()
    failures: list[str] = []
    checked = 0
    for geom in robot.collision_geometry:
        if geom.tight_geometry is None:
            continue
        geom_name = geoms.get(geom.link_name)
        if geom_name is None:
            failures.append(f"{geom.link_name}: declares tight_geometry but has no mesh source")
            continue
        checked += 1
        pts = link_mesh_in_box_frame(xml, geom_name, geom.origin_xyz_rpy)
        tight = geom.tight_geometry
        # The obligation the kernel cannot check: the true mesh is inside the DOP.
        proj = pts @ axes.T
        over = float((proj - np.asarray(tight.dop_hi_m)).max())
        under = float((np.asarray(tight.dop_lo_m) - proj).max())
        worst = max(over, under)
        if worst > 1e-9:
            failures.append(
                f"{geom.link_name}: mesh escapes its declared DOP by {worst * 1e3:.6f} mm"
            )
        if tight.hull_vertices_m:
            hull_pts = np.asarray(tight.hull_vertices_m, dtype=float)
            # And the mesh is inside the declared hull -- checked against every
            # facet, so a stale or truncated vertex list cannot pass.
            from scipy.spatial import ConvexHull

            hull = ConvexHull(hull_pts)
            normals = hull.equations[:, :3]
            offsets = hull.equations[:, 3]
            slack = float(((pts @ normals.T + offsets) / np.linalg.norm(normals, axis=1)).max())
            if slack > 1e-9:
                failures.append(
                    f"{geom.link_name}: mesh escapes its declared hull by {slack * 1e3:.6f} mm"
                )
        print(
            f"{geom.link_name}: mesh {len(pts)} vtx, declared hull "
            f"{len(tight.hull_vertices_m)} vtx, mesh-outside-DOP {worst * 1e3:+.9f} mm"
        )
    if checked == 0:
        print(f"{robot.name}: no link declares tight_geometry")
    for line in failures:
        print(f"FAIL {line}", file=sys.stderr)
    return 3 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("emit", "check"))
    parser.add_argument(
        "--robot",
        type=Path,
        default=REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml",
        help="path to the robot manifest",
    )
    args = parser.parse_args(argv)
    if args.mode == "emit":
        return _emit(args.robot)
    return _check(args.robot)


if __name__ == "__main__":
    raise SystemExit(main())
