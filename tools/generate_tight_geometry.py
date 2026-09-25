#!/usr/bin/env python3
"""Derive a robot manifest's ``tight_geometry`` blocks from its real collision meshes.

The safety kernel's arm-link-vs-world-voxel check runs a staged 26-DOP → exact
convex hull narrow phase for links that declare ``tight_geometry`` (see
``docs/reference/collision-hull-narrow-phase.md``); this tool derives that
geometry, not fits it: the 26-DOP is the intersection of 26 tangent halfspaces
``u·x <= h_mesh(u)`` (mesh inside by construction, no optimiser tolerance) and
the hull is ``conv(mesh vertices)`` (containment definitional); both are in the
manifest box's own frame, and the tool refuses to emit anything outside that box.

The derivation itself (DOP, hull, budget refinement, overhang) lives in
``openral_safety.tight_geometry``, which the collision lowering also uses; this
tool keeps the panda_mobile mesh sources and the ``check`` gate.

Mesh placement follows ``docs/reference/collision-tight-geometry.md`` §11:
MuJoCo folds mesh recentring into the geom frame, so vertices are placed by the
GEOM transform only — applying ``mesh_pos`` too double-counts it (PR #158).

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
# `openral_safety` is a ROS package, not a workspace member: the pytest root
# conftest puts it on the path for the tests; a standalone run needs the same.
if str(REPO_ROOT / "packages" / "openral_safety") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "openral_safety"))

# Re-exported under their historical names: the tests and tools/stop_ee_speed.py
# reach them through this module.
from openral_safety.tight_geometry import (  # noqa: E402
    HULL_OVERHANG_SAFETY_MARGIN as _HULL_OVERHANG_SAFETY_MARGIN,
)
from openral_safety.tight_geometry import (  # noqa: E402
    _dop_axes,
    derive_tight_geometry,
    hull_overhang_m,
    refine_dop_to_budget,
)
from openral_safety.tight_geometry import round_up_m as _round_up_m  # noqa: E402

__all__ = [
    "PANDA_GEOM_OF_LINK",
    "ROBOT_MESH_SOURCES",
    "derive_tight_geometry",
    "hull_overhang_m",
    "link_mesh_faces",
    "link_mesh_in_box_frame",
    "main",
    "refine_dop_to_budget",
]

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


def link_mesh_faces(xml_path: Path, geom_name: str) -> Points:
    """Triangle indices for ``geom_name``'s mesh, local to its own vertex block.

    Indexes the same vertex order ``link_mesh_in_box_frame`` returns, so
    ``faces`` from this function and ``points`` from that one describe one
    consistent triangle mesh.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_faceadr[mesh_id]
    count = model.mesh_facenum[mesh_id]
    faces: Points = model.mesh_face[start : start + count].copy()
    return faces


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
        overhang = None
        if derived["hull_vertices_m"]:
            import numpy as np

            faces = link_mesh_faces(xml, geom_name)
            hull_pts = np.asarray(derived["hull_vertices_m"], dtype=float)
            sampled = hull_overhang_m(hull_pts, pts, faces)
            overhang = _round_up_m(sampled * _HULL_OVERHANG_SAFETY_MARGIN)
        print(f"\n# --- {geom.link_name} ---")
        print(
            f"#   mesh vertices {len(pts)}, hull vertices {derived['_hull_vertex_count']}, "
            f"stage 2 {'on' if derived['_stage2'] else 'OFF (over the vertex budget)'}, "
            f"DOP inward margin {derived['_dop_inward_margin_m'] * 1e3:.4f} mm"
            + (f", hull overhang {overhang * 1e3:.4f} mm" if overhang is not None else "")
        )
        print("    tight_geometry:")
        print("      dop_lo_m: [" + ", ".join(repr(v) for v in derived["dop_lo_m"]) + "]")
        print("      dop_hi_m: [" + ", ".join(repr(v) for v in derived["dop_hi_m"]) + "]")
        if derived["hull_vertices_m"]:
            print("      hull_vertices_m:")
            for v in derived["hull_vertices_m"]:
                print("        - [" + ", ".join(repr(c) for c in v) + "]")
            print(f"      hull_overhang_m: {overhang!r}")
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
            # reason: scipy ships no type stubs and scipy-stubs is not a workspace dependency.
            from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

            hull = ConvexHull(hull_pts)
            normals = hull.equations[:, :3]
            offsets = hull.equations[:, 3]
            slack = float(((pts @ normals.T + offsets) / np.linalg.norm(normals, axis=1)).max())
            if slack > 1e-9:
                failures.append(
                    f"{geom.link_name}: mesh escapes its declared hull by {slack * 1e3:.6f} mm"
                )
            # The other direction: re-sample the hull's overhang past the real
            # mesh, at a FINER grid than generation used, and refuse a shipped
            # number a denser sample would exceed. A generator checking its own
            # output at the same resolution would just reproduce it -- the
            # margin in `_emit` is what makes this independent re-sample a
            # real check rather than a coin flip against grid placement.
            faces = link_mesh_faces(xml, geom_name)
            fresh = hull_overhang_m(hull_pts, pts, faces, samples_per_edge=32)
            declared = tight.hull_overhang_m
            if declared is None:
                failures.append(f"{geom.link_name}: ships a hull but no hull_overhang_m")
            elif fresh > declared + 1e-6:
                failures.append(
                    f"{geom.link_name}: hull_overhang_m={declared * 1e3:.4f} mm understates the "
                    f"resampled overhang of {fresh * 1e3:.4f} mm"
                )
        elif tight.hull_overhang_m is not None:
            failures.append(
                f"{geom.link_name}: hull_overhang_m is set but hull_vertices_m is empty"
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
