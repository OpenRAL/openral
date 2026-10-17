"""Size-aware map confirmation (issue #418 design, step 4).

``E`` is the set of grid cells the camera's surface points fall in, minus the robot's own
space. A track is confirmed when ``|E ∩ occupied| ≥ max(2, ⌈0.5·|E|⌉)`` — cover on the cells
the camera *expects* occupied, not on a padded footprint (at 15 mm cells the padded rule
reaches 0.5 only for squares ≥ 111 mm; an orange has ``|E|`` ≈ 5-9) — **and**
``openral_core.object_primitives.map_confirms`` holds one body: the occupied cells above the
support inside the box's cell closure are one 26-connected component. A second body
wholly inside a declared hand's own space is ``map_split_own`` (leaked hand points: not a
foreign object); any other is ``map_split``.

Pure numpy, no ROS, no clock.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from openral_core import PlaceRegion
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.object_primitives import map_confirms
from openral_core.perception_geometry import VoxelLattice

__all__ = [
    "CoverVerdict",
    "cover_verdict",
    "own_cover",
    "read_static_cells",
    "static_cell_keys",
    "surface_cells",
]

Cell = tuple[int, int, int]


@dataclass(frozen=True)
class CoverVerdict:
    """Result of ``cover_verdict``.

    Attributes:
        visible: ``|E|``.
        occupied: ``|E ∩ occupied|``.
        kind: ``""`` (confirmed), ``"map_disagrees"``, ``"map_split"`` or ``"map_split_own"``.
        why: Human-readable reason for the trace.
    """

    visible: int
    occupied: int
    kind: str
    why: str = ""


def _inside(points: NDArray[np.float64], box: PlaceRegion) -> NDArray[np.bool_]:
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    local = (points - t[:3, 3]) @ t[:3, :3]
    return np.asarray(np.all(np.abs(local) <= np.asarray(box.half_extents), axis=1))


def surface_cells(
    grid: VoxelLattice,
    surface: NDArray[np.float64],
    *,
    exclude: Sequence[PlaceRegion] = (),
) -> frozenset[Cell]:
    """``E``: lattice cells holding ``surface`` points, minus cells centred in ``exclude``.

    Example:
        >>> import numpy as np
        >>> occ = np.zeros(8, np.uint8)
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (2, 2, 2), occ)
        >>> sorted(surface_cells(g, np.array([[0.05, 0.05, 0.05], [0.06, 0.04, 0.02]])))
        [(0, 0, 0)]
    """
    local = (surface - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
    ijk = np.unique(np.floor(local).astype(np.int64), axis=0)
    if exclude and len(ijk):
        centres = (ijk + 0.5) * grid.resolution @ grid.rotation().T + np.asarray(grid.origin)
        keep = ~np.any([_inside(centres, b) for b in exclude], axis=0)
        ijk = ijk[keep]
    return frozenset((int(i), int(j), int(k)) for i, j, k in ijk.tolist())


def read_static_cells(path: str) -> NDArray[np.float64]:
    r"""The declared static cells' centres from the launch's static-cells file.

    One ``x y z`` per line in the base frame (metres), ``#`` comments allowed; an empty file
    is no static geometry.

    Example:
        >>> import tempfile, os
        >>> with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        ...     _ = f.write("# cells\n0.1 0.2 0.3\n0.4 0.5 0.6\n")
        >>> read_static_cells(f.name).shape
        (2, 3)
        >>> os.unlink(f.name)
    """
    pts = np.loadtxt(path, comments="#", ndmin=2, dtype=np.float64)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3) if pts.size else np.zeros((0, 3))


def static_cell_keys(
    grid: VoxelLattice, centres: NDArray[np.float64], *, pad_cells: int = 0
) -> frozenset[Cell]:
    """The lattice cells of declared static geometry, grown by ``pad_cells``.

    ``centres`` are the declared cells' centres in the grid frame (the launch's static-cells
    file, one centre per cell the octomap bridge marks); each is mapped onto ``grid``'s lattice
    and every cell within ``pad_cells`` (Chebyshev) of one is kept, clipped to the lattice. The
    camera sees a wall about a cell off its declared box (#418 restock, 17.5 mm at 15 mm cells),
    so the kernel's K7 pads the same way.

    Example:
        >>> import numpy as np
        >>> g = VoxelLattice(
        ...     "b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (4, 4, 4), np.zeros(64, np.uint8)
        ... )
        >>> len(static_cell_keys(g, np.array([[0.15, 0.15, 0.15]]), pad_cells=1))
        27
        >>> sorted(static_cell_keys(g, np.array([[0.05, 0.05, 0.05]]), pad_cells=0))
        [(0, 0, 0)]
    """
    if not len(centres):
        return frozenset()
    base = np.floor(
        (np.asarray(centres) - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
    )
    out: set[Cell] = set()
    sx, sy, sz = grid.size
    steps = range(-pad_cells, pad_cells + 1)
    for i, j, k in np.unique(base.astype(np.int64), axis=0).tolist():
        for di in steps:
            for dj in steps:
                for dk in steps:
                    a, b, c = i + di, j + dj, k + dk
                    if 0 <= a < sx and 0 <= b < sy and 0 <= c < sz:
                        out.add((a, b, c))
    return frozenset(out)


def own_cover(grid: VoxelLattice, cells: frozenset[Cell], others: Sequence[PlaceRegion]) -> int:
    """``|E ∩ occupied|`` over the cells centred in none of ``others`` (other tracks' boxes).

    A new track's evidence must be its own: #418 PickOrange r6po_dbg4's ghost trk:5 was a
    mask over two tracked oranges, every occupied cell of it inside their boxes.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.ones(8, np.uint8)
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (2, 2, 2), occ)
        >>> left = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.05, 0.1, 0.1),
        ...     pose=Pose6D(xyz=(0.05, 0.1, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> own_cover(g, frozenset({(0, 0, 0), (1, 0, 0)}), [left])
        1
    """
    if not cells:
        return 0
    ijk = np.asarray(sorted(cells), dtype=np.float64)
    centres = (ijk + 0.5) * grid.resolution @ grid.rotation().T + np.asarray(grid.origin)
    if others:
        keep = ~np.any([_inside(centres, b) for b in others], axis=0)
        cells = frozenset(c for c, k in zip(sorted(cells), keep.tolist(), strict=True) if k)
    return _occupied_count(grid, cells)


def _occupied_count(grid: VoxelLattice, cells: frozenset[Cell]) -> int:
    sx, sy, sz = grid.size
    n = 0
    for i, j, k in cells:
        if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz and grid.occupancy[i + sx * (j + sy * k)]:
            n += 1
    return n


def cover_verdict(
    grid: VoxelLattice,
    cells: frozenset[Cell],
    box: PlaceRegion,
    *,
    support_z: float,
    own: Sequence[PlaceRegion] = (),
    others: Sequence[PlaceRegion] = (),
    static: frozenset[Cell] = frozenset(),
) -> CoverVerdict:
    """Confirm a track's box on one grid: size-aware cover, then one body.

    Args:
        grid: The newest voxel grid; ``box`` must be in its frame.
        cells: ``E`` from ``surface_cells`` at the track's last fit.
        box: The track's containing box.
        support_z: The support the box stands on.
        own: The declared hands' own spaces, grid frame.
        others: The other confirmed tracks' boxes, grid frame: a second body wholly inside
            one of them is that object, not a foreign body (design step 4: the closure
            cells minus other tracks).
        static: Lattice cells of the declared static geometry (``static_cell_keys``): a body
            wholly inside them is the declared wall, not a foreign body, for the ONE-BODY
            test only. The cover gate counts every occupied cell.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(27, np.uint8)
        >>> occ[[13, 22]] = 1  # the column at i=j=1, k=1..2
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (3, 3, 3), occ)
        >>> b = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.05, 0.05, 0.1),
        ...     pose=Pose6D(xyz=(0.15, 0.15, 0.2), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> cover_verdict(g, frozenset({(1, 1, 1), (1, 1, 2)}), b, support_z=0.1).kind
        ''
        >>> cover_verdict(g, frozenset({(0, 0, 1), (0, 1, 1), (1, 1, 2)}), b, support_z=0.1).kind
        'map_disagrees'
    """
    hit = _occupied_count(grid, cells)
    need = max(2, math.ceil(0.5 * len(cells)))
    if hit < need:
        return CoverVerdict(len(cells), hit, "map_disagrees", f"{hit}/{len(cells)} expected cells")
    pad = 0.5 * grid.resolution
    kind, why = map_confirms(
        grid, box, support_z=support_z, min_cover=0.0, pad_m=pad, own=others, static=static
    )
    if kind == "map_split" and own:
        own_kind, _ = map_confirms(
            grid,
            box,
            support_z=support_z,
            min_cover=0.0,
            pad_m=pad,
            own=[*others, *own],
            static=static,
        )
        if not own_kind:
            return CoverVerdict(len(cells), hit, "map_split_own", why)
    return CoverVerdict(len(cells), hit, kind, why)
