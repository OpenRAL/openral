"""Size-aware map confirmation (issue #418 design, step 4).

``E`` is the set of grid cells the camera's surface points fall in, minus the robot's own
space. A track is confirmed when ``|E ∩ occupied| ≥ max(2, ⌈0.5·|E|⌉)`` — cover on the cells
the camera *expects* occupied, not on a padded footprint (at 15 mm cells the padded rule
reaches 0.5 only for squares ≥ 111 mm; an orange has ``|E|`` ≈ 5-9) — **and**
``openral_core.object_primitives.map_confirms`` holds one body: the occupied cells above the
support inside the box's cell closure are one 26-connected component. A second body
wholly inside a declared hand's own space is ``map_split_own`` (leaked hand points: not a
foreign object); any other of at least ``MIN_SPLIT_CELLS`` cells is ``map_split``.

Pure numpy, no ROS, no clock.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from openral_core import IntrinsicsPinhole, PlaceRegion
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.object_primitives import map_confirms
from openral_core.perception_geometry import VoxelLattice, occupied_centers_in_box

__all__ = [
    "MIN_SPLIT_CELLS",
    "CoverVerdict",
    "cells_behind_robot",
    "cover_verdict",
    "hand_touches",
    "own_cover",
    "read_static_cells",
    "static_cell_keys",
    "surface_cells",
    "uncovered_cells",
]

Cell = tuple[int, int, int]

#: The fewest cells a second body inside a track's box needs to split it (user decision
#: 2026-10-21, Safety-WG review owed). 1-2 cells is a fragment of the object itself: a
#: curved banana's tip or a piece the hand's occlusion cuts off (Isaac R28: 1 cell, R31: 2,
#: both 100% inside the truth box) went SPLIT mid-grasp and dropped the target's exemption.
#: At 15 mm cells, 3 is ~3 cm: a smaller foreign item inside a target's box rides its
#: exemption, still under the workspace's speed and lead bounds. *Calibration point.*
MIN_SPLIT_CELLS = 3


@dataclass(frozen=True)
class CoverVerdict:
    """Result of ``cover_verdict``.

    Attributes:
        visible: ``|E|``.
        occupied: ``|E ∩ occupied|``.
        kind: ``""`` (confirmed), ``"map_disagrees"``, ``"map_split"``, ``"map_split_own"`` or
            ``"hidden"`` (fewer than two expected cells the camera can see: no evidence).
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


def cells_behind_robot(
    grid: VoxelLattice,
    cells: frozenset[Cell],
    depth_m: NDArray[np.float64],
    robot_mask: NDArray[np.uint8],
    k: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    tol_m: float = 0.01,
) -> frozenset[Cell]:
    """The cells of ``cells`` the camera cannot see now because the robot is in front.

    A cell is hidden when its centre projects onto a robot pixel (``robot_mask`` >= 128, the
    robot and its uncertain band) whose measured depth is nearer than the centre by more than
    ``tol_m``. A cell out of view, behind a non-robot pixel, or in front of the robot is not
    hidden. Issue #418 (Isaac baseline: a successful grasp keeps the fingers on the banana for
    255-466 steps), the visibility-aware cover: an expected cell the hand hides is no evidence
    the object is gone, so ``cover_verdict`` leaves it out of the count.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> occ = np.zeros(27, np.uint8)
        >>> g = VoxelLattice("b", (-0.15, -0.15, 0.85), (0, 0, 0, 1.0), 0.1, (3, 3, 3), occ)
        >>> k = IntrinsicsPinhole(fx=10.0, fy=10.0, cx=2.0, cy=2.0, width=4, height=4)
        >>> depth = np.full((4, 4), 0.5)  # the robot, 0.5 m in front of the camera
        >>> robot = np.zeros((4, 4), np.uint8)
        >>> robot[2, 2] = 255
        >>> sorted(
        ...     cells_behind_robot(g, frozenset({(1, 1, 1), (0, 0, 1)}), depth, robot, k, np.eye(4))
        ... )
        [(1, 1, 1)]
    """
    keys = sorted(cells)
    if not keys:
        return frozenset()
    centres = (np.asarray(keys, dtype=np.float64) + 0.5) * grid.resolution @ grid.rotation().T
    centres += np.asarray(grid.origin)
    t_cam = np.linalg.inv(t_base_from_cam)
    cam = centres @ t_cam[:3, :3].T + t_cam[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(k.fx * cam[:, 0] / z + k.cx)
        v = np.floor(k.fy * cam[:, 1] / z + k.cy)
    ok = (z > 0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < k.width) & (v >= 0)
    ok &= v < k.height
    ui, vi = np.where(ok, u, 0).astype(np.int64), np.where(ok, v, 0).astype(np.int64)
    d = depth_m[vi, ui]
    behind = ok & (robot_mask[vi, ui] >= 128) & np.isfinite(d) & (d > 0) & (d < z - tol_m)  # noqa: PLR2004  # reason: the robot / band pixel floor
    return frozenset(c for c, h in zip(keys, behind, strict=True) if h)


def hand_touches(box: PlaceRegion, hands: Sequence[PlaceRegion], margin_m: float) -> bool:
    """Whether any hand box reaches ``box`` grown by ``margin_m``: the hand is at the object.

    Tested on 27 samples per hand box (corners, edge and face centres, centre), which any
    overlap deeper than about half the hand box's smallest extent hits; fingers are
    centimetres across, so a grown-by-a-cell box catches contact.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> def b(x, h):
        ...     return PlaceRegion(
        ...         frame_id="b",
        ...         half_extents=(h, h, h),
        ...         pose=Pose6D(xyz=(x, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     )
        >>> (
        ...     hand_touches(b(0.0, 0.05), [b(0.07, 0.01)], 0.015),
        ...     hand_touches(b(0.0, 0.05), [b(0.2, 0.01)], 0.015),
        ... )
        (True, False)
    """
    grown = box.model_copy(update={"half_extents": tuple(h + margin_m for h in box.half_extents)})
    unit = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
    for hand in hands:
        t = homogeneous_from_quat_xyz(hand.pose.xyz, hand.pose.quat_xyzw)
        pts = (unit * np.asarray(hand.half_extents)) @ t[:3, :3].T + t[:3, 3]
        if _inside(pts, grown).any():
            return True
    return False


def uncovered_cells(grid: VoxelLattice, cells: frozenset[Cell], box: PlaceRegion) -> int:
    """How many of ``cells`` the map still holds occupied, centred outside ``box`` + a cell.

    A refit's box that leaves out the occupied surface cells of the track's last fit does not
    contain the object the map still shows there (Isaac R37/R39: a partial mask's clear
    refit shrank an unmoved banana's box to half and 5-6 cm off-centre, and the finger met
    the uncovered half). ``box`` must be in the grid's frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(27, np.uint8)
        >>> occ[[12, 13, 14]] = 1  # three cells along x at j=1, k=1
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (3, 3, 3), occ)
        >>> cells = frozenset({(0, 1, 1), (1, 1, 1), (2, 1, 1)})
        >>> def box(x, hx):
        ...     return PlaceRegion(
        ...         frame_id="b",
        ...         half_extents=(hx, 0.05, 0.05),
        ...         pose=Pose6D(xyz=(x, 0.15, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     )
        >>> uncovered_cells(g, cells, box(0.15, 0.15)), uncovered_cells(g, cells, box(0.05, 0.05))
        (0, 1)
    """
    sx, sy, sz = grid.size
    keys = [
        (i, j, k)
        for i, j, k in cells
        if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz and grid.occupancy[i + sx * (j + sy * k)]
    ]
    if not keys:
        return 0
    centres = (np.asarray(keys, dtype=np.float64) + 0.5) * grid.resolution @ grid.rotation().T
    centres += np.asarray(grid.origin)
    grown = box.model_copy(
        update={"half_extents": tuple(h + grid.resolution for h in box.half_extents)}
    )
    return int(np.count_nonzero(~_inside(centres, grown)))


def box_cells(
    grid: VoxelLattice, box: PlaceRegion, *, support_z: float, exclude: Sequence[PlaceRegion] = ()
) -> frozenset[Cell]:
    """The occupied cells centred in ``box`` a cell above the support, in none of ``exclude``.

    What the map holds of an object inside its box now, wherever in the box it lies: the
    tracker re-anchors a touched object's expected cells on them when a push inside the box
    emptied the old ones (Isaac R68: a banana pushed 66 mm inside its box held 12 of 26).
    The support's own layer is left out as ``map_confirms`` leaves it (Isaac R79: 212 of 235
    cells were the table top's layer, its centres a rounding error above ``support_z +
    resolution / 2``). ``box`` and ``exclude`` must be in the grid's frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(27, np.uint8)
        >>> occ[[4, 13, 22]] = 1  # the column at i=j=1, k=0..2
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0, 0, 0, 1.0), 0.1, (3, 3, 3), occ)
        >>> b = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.15, 0.15, 0.15),
        ...     pose=Pose6D(xyz=(0.15, 0.15, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> sorted(box_cells(g, b, support_z=0.0))  # the support layer's cell is left out
        [(1, 1, 1), (1, 1, 2)]
    """
    centres = occupied_centers_in_box(grid, box)
    centres = centres[centres[:, 2] > support_z + grid.resolution]
    if exclude and len(centres):
        centres = centres[~np.any([_inside(centres, b) for b in exclude], axis=0)]
    local = (centres - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
    return frozenset(tuple(c) for c in np.floor(local).astype(np.int64).tolist())


def cover_verdict(
    grid: VoxelLattice,
    cells: frozenset[Cell],
    box: PlaceRegion,
    *,
    support_z: float,
    own: Sequence[PlaceRegion] = (),
    others: Sequence[PlaceRegion] = (),
    static: frozenset[Cell] = frozenset(),
    hidden: frozenset[Cell] = frozenset(),
) -> CoverVerdict:
    """Confirm a track's box on one grid: size-aware cover, then one body.

    ``hidden`` (``cells_behind_robot``) leaves the expected cells the robot hides out of the
    cover count, numerator and denominator alike; under two visible expected cells there is no
    evidence either way (``"hidden"``).

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
        hidden: Expected cells the robot hides now (``cells_behind_robot``), left out of the
            cover count.

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
    if hidden:
        cells = cells - hidden
        if len(cells) < 2:  # noqa: PLR2004  # reason: the cover rule's own floor
            return CoverVerdict(len(cells), 0, "hidden", "the robot hides the expected cells")
    hit = _occupied_count(grid, cells)
    need = max(2, math.ceil(0.5 * len(cells)))
    if hit < need:
        return CoverVerdict(len(cells), hit, "map_disagrees", f"{hit}/{len(cells)} expected cells")
    pad = 0.5 * grid.resolution
    kind, why = map_confirms(
        grid,
        box,
        support_z=support_z,
        min_cover=0.0,
        pad_m=pad,
        own=others,
        static=static,
        min_split_cells=MIN_SPLIT_CELLS,
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
            min_split_cells=MIN_SPLIT_CELLS,
        )
        if not own_kind:
            return CoverVerdict(len(cells), hit, "map_split_own", why)
    return CoverVerdict(len(cells), hit, kind, why)
