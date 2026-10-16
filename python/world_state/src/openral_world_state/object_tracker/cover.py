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

__all__ = ["CoverVerdict", "cover_verdict", "surface_cells"]

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
) -> CoverVerdict:
    """Confirm a track's box on one grid: size-aware cover, then one body.

    Args:
        grid: The newest voxel grid; ``box`` must be in its frame.
        cells: ``E`` from ``surface_cells`` at the track's last fit.
        box: The track's containing box.
        support_z: The support the box stands on.
        own: The declared hands' own spaces, grid frame.

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
    kind, why = map_confirms(grid, box, support_z=support_z, min_cover=0.0, pad_m=pad)
    if kind == "map_split" and own:
        own_kind, _ = map_confirms(
            grid, box, support_z=support_z, min_cover=0.0, pad_m=pad, own=own
        )
        if not own_kind:
            return CoverVerdict(len(cells), hit, "map_split_own", why)
    return CoverVerdict(len(cells), hit, kind, why)
