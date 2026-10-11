"""Declared static geometry → base-frame surface points for the octomap bridge.

A cell declares its fixtures (walls, shelf frame, table, tote shell) as a
``StaticGeometry`` YAML (``DeployRuntime.static_geometry_file``). The deploy launch
samples every declared surface with ``surface_points`` at a quarter of a voxel
apart and hands the points to the octomap bridge (``static_points_file``), which marks
every cell holding one occupied in each published grid — so a fixture the head camera
has not seen still exists for the safety kernel
(``docs/reference/manipulation-workspace-contact.md`` §3.1).

Spacing ``s`` along every face and inside every triangle puts each point of a declared
surface within ``s`` of a sample, so a cell the surface crosses by more than ``s`` holds a
sample and is marked. A surface may dip into a cell by less than ``s`` without marking
it: the declared geometry is represented exactly to within ``s``. The launch samples at a
quarter of the grid resolution (3.75 mm at 15 mm cells), which bounds that gap. The
geometry only ever ADDS occupancy (the fail-closed direction).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from openral_core import StaticBox, StaticGeometry, StaticMesh
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz

__all__ = ["load_static_geometry", "surface_points", "write_points_file"]


def load_static_geometry(path: str | Path) -> tuple[StaticGeometry, Path]:
    """Load and validate a ``StaticGeometry`` YAML; returns it and its directory.

    Raises:
        ROSConfigError: the file is missing, not YAML, or does not validate.
    """
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text())
        return StaticGeometry.model_validate(data), p.parent
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ROSConfigError(f"static geometry {p}: {exc}") from exc


def _box_points(box: StaticBox, spacing: float) -> NDArray[np.float64]:
    """Points on the six faces of ``box`` (its own axes), at most ``spacing`` apart."""
    h = np.asarray(box.half_extents, dtype=np.float64)
    axes = [np.linspace(-v, v, max(2, math.ceil(2.0 * v / spacing) + 1)) for v in h]
    faces = []
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        u, v = np.meshgrid(axes[i], axes[j], indexing="ij")
        for sign in (-1.0, 1.0):
            pts = np.zeros((u.size, 3))
            pts[:, i], pts[:, j], pts[:, k] = u.ravel(), v.ravel(), sign * h[k]
            faces.append(pts)
    local = np.unique(np.concatenate(faces), axis=0)
    world = homogeneous_from_quat_xyz(box.xyz, box.quat_xyzw)
    out: NDArray[np.float64] = local @ world[:3, :3].T + world[:3, 3]
    return out


def _triangle_points(tri: NDArray[np.float64], spacing: float) -> NDArray[np.float64]:
    """Barycentric grid on one triangle, at most ``spacing`` between neighbours."""
    edge = max(float(np.linalg.norm(tri[a] - tri[b])) for a, b in ((0, 1), (1, 2), (2, 0)))
    n = max(1, math.ceil(edge / spacing))
    ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
    w = np.asarray([(i / n, j / n, 1.0 - (i + j) / n) for i, j in ij])
    out: NDArray[np.float64] = w @ tri
    return out


def surface_points(
    geometry: StaticGeometry, base_dir: Path, spacing_m: float
) -> NDArray[np.float64]:
    """Every declared surface sampled at most ``spacing_m`` apart, ``(N, 3)`` base frame.

    Example:
        >>> g = StaticGeometry.model_validate(
        ...     {
        ...         "static_geometry": [
        ...             {"kind": "box", "xyz": [1.0, 0.0, 0.0], "half_extents": [0.05, 0.1, 0.1]}
        ...         ]
        ...     }
        ... )
        >>> pts = surface_points(g, Path("."), 0.05)
        >>> bool(np.allclose(pts.min(axis=0), [0.95, -0.1, -0.1]))
        True
        >>> # Every point of the +x face is within the spacing of a sample.
        >>> face = np.array(
        ...     [[1.05, y, z] for y in np.linspace(-0.1, 0.1, 9) for z in np.linspace(-0.1, 0.1, 9)]
        ... )
        >>> d = np.min(np.linalg.norm(face[:, None] - pts[None], axis=2), axis=1)
        >>> bool(d.max() <= 0.05)
        True

    Raises:
        ROSConfigError: ``spacing_m`` is not positive, or a mesh file is missing or
            malformed.
    """
    if not spacing_m > 0.0:
        raise ROSConfigError(f"static geometry spacing must be > 0, got {spacing_m}.")
    chunks: list[NDArray[np.float64]] = []
    for entry in geometry.static_geometry:
        if isinstance(entry, StaticBox):
            chunks.append(_box_points(entry, spacing_m))
            continue
        assert isinstance(entry, StaticMesh)
        path = base_dir / entry.file
        try:
            with np.load(path) as data:
                verts = np.asarray(data["vertices"], dtype=np.float64)
                tris = np.asarray(data["triangles"], dtype=np.int64)
        except (OSError, KeyError, ValueError) as exc:
            raise ROSConfigError(f"static mesh {path}: {exc}") from exc
        if verts.ndim != 2 or verts.shape[1] != 3 or tris.ndim != 2 or tris.shape[1] != 3:
            raise ROSConfigError(f"static mesh {path}: vertices/triangles must be (N, 3).")
        if tris.size and (tris.min() < 0 or tris.max() >= len(verts)):
            raise ROSConfigError(f"static mesh {path}: a triangle indexes past the vertices.")
        chunks.extend(_triangle_points(verts[t], spacing_m) for t in tris)
    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks)


def write_points_file(points: NDArray[np.float64], path: str | Path) -> None:
    """Write ``points`` as the bridge's ``static_points_file`` (one ``x y z`` per line)."""
    with Path(path).open("w") as f:
        f.write("# declared static geometry, base_frame surface points (m)\n")
        np.savetxt(f, points, fmt="%.5f")
