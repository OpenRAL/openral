"""Object primitives from the voxel map — every component above a measured support, tracked.

The map-first half of ``docs/reference/object-primitives-design.md`` (issue #349): the
grasp region is needed *before* the fingers reach the target, and a VLA never pauses
for the head camera to get a clean, unoccluded fit. The octomap already holds every
object as a 26-connected component of occupied cells standing on a measured support
(the same component ``_grasp_target.map_completed_region`` grows a camera fit to), and
it remembers views from before the hand arrived. So the component *is* the primitive:
a gravity-aligned (yaw-only) box over its cell centres, its lower face one voxel above
the support (HZ-0115-6: never into the support layer), its yaw the footprint's PCA.

Pure numpy, no ROS, no clock. ``primitives_from_voxels`` extracts every primitive in a
search column; ``PrimitiveTracker`` keeps identities across grids and retracts what the
map stops holding; ``nearest_primitive`` picks the one a hand approaches, refusing a
tie. The grasp-target leg (``_grasp_target_leg``, ``primitives=True``) runs the chosen
box through its existing gates (map cover, whole-component, tracking, occlusion hold,
caps, cell closure) and publishes it as the declaration's region — the kernel contract
is unchanged and the camera is not consulted.

Every threshold is a *calibration point* (CLAUDE.md §1.2). What the map cannot do is
documented, not hidden: two bodies within one voxel of each other are one component
(HZ-0115-30), and a target standing on a same-footprint body is one component with it —
the camera's ``not_on_support`` gate is the only one that separates them, so the
design note keeps a camera confirmation as the production posture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from openral_core import PlaceRegion, Pose6D
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw

from openral_hal._grasp_target import (
    _NEIGHBOURS_26,
    VoxelLattice,
    _cell_centres,
    _check_frame,
    _components,
    _ijk,
    _in_region,
    occupied_centers_in_box,
    track_region,
)

__all__ = [
    "ObjectPrimitive",
    "PrimitiveFit",
    "PrimitiveTracker",
    "nearest_primitive",
    "point_box_gap_m",
    "primitives_from_voxels",
    "raised_anchor_xy",
]


@dataclass(frozen=True)
class PrimitiveFit:
    """One component of the map fitted as a box.

    Attributes:
        region: The tight gravity-aligned box over the component's cell centres, its
            lower face one voxel above the support.
        cell_count: Cells in the component.
    """

    region: PlaceRegion
    cell_count: int


@dataclass(frozen=True)
class ObjectPrimitive:
    """A tracked primitive: a ``PrimitiveFit`` under a stable identity.

    Attributes:
        primitive_id: Stable across grids while the box tracks (``PrimitiveTracker``).
        region: The newest fit's box.
        cell_count: The newest fit's cell count.
        first_seen_ns: Grid ``source_stamp`` the identity was minted at.
        last_seen_ns: Grid ``source_stamp`` of the newest fit.
    """

    primitive_id: int
    region: PlaceRegion
    cell_count: int
    first_seen_ns: int
    last_seen_ns: int


def _footprint_yaw(xy: NDArray[np.float64]) -> float:
    """Yaw of the footprint's major axis: 2-D PCA over the distinct cell columns."""
    if len(xy) < 2:  # noqa: PLR2004 — one column has no axis
        return 0.0
    _, _, vt = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    return math.atan2(float(vt[0, 1]), float(vt[0, 0]))


def primitives_from_voxels(
    grid: VoxelLattice,
    column: PlaceRegion,
    *,
    support_z: float,
    min_cells: int,
    max_half_extent_m: float,
    max_volume_m3: float,
    stamp_ns: int = 0,
    evidence_ref: str = "map_primitive",
) -> tuple[list[PrimitiveFit], list[str]]:
    """Every object standing on the support inside ``column``, each as a tight yaw box.

    The occupied cells of ``column`` more than one voxel above ``support_z`` (the layer
    touching the support is the support's own, as ``target_seed_from_voxels`` drops it)
    are labelled into 26-connected components. A component is a primitive when it has
    at least ``min_cells`` cells, its lowest cell sits within two voxels of the support
    (else it stands on something else — a shelf above, a riser), none of its cells
    touches the column's edge (the leg cannot vouch for what continues outside it, the
    same rule as ``map_completed_region``), and its box fits the caps. The box is the
    smallest gravity-aligned box in the footprint's PCA yaw holding every cell centre
    (each horizontal half-extent at least half a cell), its lower face pinned at
    ``support_z + resolution`` — never lower (HZ-0115-6) — and its top the highest
    cell's top face. ``cell_closed_region`` then grows it to hold every cell whole, as
    for a camera fit.

    Two bodies within one voxel of each other are one component and so one primitive
    (HZ-0115-30); the caps refuse a merged pair that grew past one graspable object,
    and the rest is the documented residual.

    Args:
        grid: The published lattice; must be z-up (yaw-only).
        column: The search column, in ``grid.frame_id``.
        support_z: The measured support top (``support_top_from_voxels``).
        min_cells: Fewest cells a component may have. *Calibration point.*
        max_half_extent_m: Per-axis cap on the tight box (``GraspDeclaration``'s).
        max_volume_m3: Volume cap on the tight box.
        stamp_ns: Stamp put on every region (the grid's ``source_stamp``).
        evidence_ref: Provenance put on every region.

    Returns:
        ``(fits, skipped)`` — the primitives, largest first, and one line per component
        refused (why, with its size), for the trace.

    Raises:
        ROSConfigError: On a frame mismatch or a tilted lattice.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(8 * 8 * 4, dtype=np.uint8)
        >>> occ[:64] = 1  # k=0: a table top
        >>> for k in (1, 2):  # a 2x2 post at i=j=2..3 and a 1x3 bar at i=6, j=1..3
        ...     for i, j in [(2, 2), (2, 3), (3, 2), (3, 3), (6, 1), (6, 2), (6, 3)]:
        ...         occ[i + 8 * (j + 8 * k)] = 1
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (8, 8, 4), occ)
        >>> col = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.45, 0.45, 0.25),
        ...     pose=Pose6D(xyz=(0.4, 0.4, 0.2), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> fits, skipped = primitives_from_voxels(
        ...     g, col, support_z=0.1, min_cells=2, max_half_extent_m=0.2, max_volume_m3=0.03
        ... )
        >>> [(f.cell_count, [round(v, 3) for v in f.region.pose.xyz]) for f in fits]
        [(4, [0.3, 0.3, 0.25]), (3, [0.65, 0.25, 0.25])]
        >>> skipped
        []
    """
    _check_frame(grid, column)
    centers = occupied_centers_in_box(grid, column)
    above = centers[centers[:, 2] > support_z + grid.resolution]
    fits: list[PrimitiveFit] = []
    skipped: list[str] = []
    if len(above) == 0:
        return fits, skipped
    ijk = _ijk(grid, above)
    res = grid.resolution
    for comp in sorted(_components(ijk), key=len, reverse=True):
        n = len(comp)
        if n < min_cells:
            skipped.append(f"{n} cells < min_cells {min_cells}")
            continue
        cells = ijk[comp]
        pts = above[comp]
        bottom = float(pts[:, 2].min()) - 0.5 * res
        if bottom - support_z > 2.0 * res + 1e-9:
            skipped.append(f"{n} cells not on the support ({bottom - support_z:.3f} m above it)")
            continue
        around = (cells[:, None, :] + np.asarray(_NEIGHBOURS_26, dtype=np.int64)[None]).reshape(
            -1, 3
        )
        if not _in_region(_cell_centres(grid, around), column).all():
            skipped.append(f"{n} cells touch the column's edge")
            continue
        footprint = np.unique(cells[:, :2], axis=0)
        yaw = _footprint_yaw(
            _cell_centres(grid, np.column_stack([footprint, footprint[:, :1]]))[:, :2]
        )
        c, s = math.cos(yaw), math.sin(yaw)
        rot2 = np.array(((c, -s), (s, c)))
        xy_mean = pts[:, :2].mean(axis=0)
        uv = (pts[:, :2] - xy_mean) @ rot2
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        half_xy = np.maximum((hi - lo) / 2.0, 0.5 * res)
        floor = support_z + res
        top = float(pts[:, 2].max()) + 0.5 * res
        if top <= floor:
            skipped.append(f"{n} cells with no height above the support")
            continue
        half = (float(half_xy[0]), float(half_xy[1]), (top - floor) / 2.0)
        if max(half) > max_half_extent_m or 8.0 * half[0] * half[1] * half[2] > max_volume_m3:
            skipped.append(
                f"{n} cells over the caps: half_extents {tuple(round(v, 3) for v in half)}"
            )
            continue
        centre_xy = xy_mean + rot2 @ ((lo + hi) / 2.0)
        region = PlaceRegion(
            frame_id=grid.frame_id,
            pose=Pose6D(
                xyz=(float(centre_xy[0]), float(centre_xy[1]), (top + floor) / 2.0),
                quat_xyzw=yaw_to_quat_xyzw(yaw),
                frame_id=grid.frame_id,
            ),
            half_extents=half,
            evidence_ref=evidence_ref,
            stamp_ns=stamp_ns,
        )
        fits.append(PrimitiveFit(region, n))
    return fits, skipped


def raised_anchor_xy(
    grid: VoxelLattice, column: PlaceRegion, near_xy: tuple[float, float]
) -> tuple[float, float] | None:
    """Where to anchor the support scan: the nearest top-surface cell standing above the floor.

    ``support_top_from_voxels`` anchors on the occupied column nearest a point; over a
    surface that fills the search column every point is occupied, so a hand beside the
    target (a lateral approach) anchors on the bare support and finds no layer under it.
    Here the anchor is the top-surface cell (no occupied cell directly above it) nearest
    ``near_xy`` among those more than one cell above the column's lowest top surface — the
    floor the hand works over — so the scan starts on whatever stands on it.

    Returns:
        The anchor's ``(x, y)`` in ``grid.frame_id``, or ``None`` when nothing in the
        column stands above its lowest surface.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(6 * 6 * 4, dtype=np.uint8)
        >>> occ[:36] = 1  # k=0: a table
        >>> occ[[36 + 4 + 6 * 4, 72 + 4 + 6 * 4]] = 1  # a 2-cell post at i=j=4
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (6, 6, 4), occ)
        >>> col = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.3, 0.3, 0.2),
        ...     pose=Pose6D(xyz=(0.3, 0.3, 0.2), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> raised_anchor_xy(g, col, (0.1, 0.1))  # the hand over bare table
        (0.45, 0.45)
    """
    centers = occupied_centers_in_box(grid, column)
    if len(centers) == 0:
        return None
    cells = _ijk(grid, centers).tolist()
    occupied = {tuple(c) for c in cells}
    top = np.array([(i, j, k + 1) not in occupied for i, j, k in cells], dtype=bool)
    tops = centers[top]
    raised = tops[tops[:, 2] > float(tops[:, 2].min()) + grid.resolution]
    if len(raised) == 0:
        return None
    d2 = np.sum((raised[:, :2] - np.asarray(near_xy, dtype=np.float64)) ** 2, axis=1)
    x, y = raised[int(np.argmin(d2)), :2]
    return float(x), float(y)


def point_box_gap_m(point: Sequence[float], region: PlaceRegion) -> float:
    """Distance from ``point`` to the oriented box ``region`` (0 inside).

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> r = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.1, 0.1, 0.1),
        ...     pose=Pose6D(xyz=(0.0, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> round(point_box_gap_m((0.0, 0.0, 0.3), r), 3), point_box_gap_m((0.05, 0.0, 0.0), r)
        (0.2, 0.0)
    """
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    local = t[:3, :3].T @ (np.asarray(point, dtype=np.float64) - t[:3, 3])
    outside = np.maximum(np.abs(local) - np.asarray(region.half_extents), 0.0)
    return float(np.linalg.norm(outside))


def nearest_primitive(
    regions: Sequence[PlaceRegion],
    hand_points: Sequence[tuple[float, float, float]],
    *,
    ambiguity_m: float,
) -> tuple[int | None, str]:
    """Which primitive a hand is at: the nearest to any hand point, unless tied.

    With no hand points (a named search box) only a lone primitive is unambiguous.
    Two primitives whose gaps to the hand differ by at most ``ambiguity_m`` are a tie
    (HZ-0115-2: guessing between two candidates is the mis-declaration), refused until
    the hand comes nearer one of them.

    Args:
        regions: The tracked primitives' boxes.
        hand_points: The approaching hand's TCP points, in the regions' frame.
        ambiguity_m: Gap difference at or below which two candidates tie (one voxel).

    Returns:
        ``(index, reason)`` — ``reason`` is empty iff ``index`` is set.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> def box(x):
        ...     return PlaceRegion(
        ...         frame_id="b",
        ...         half_extents=(0.03, 0.03, 0.05),
        ...         pose=Pose6D(xyz=(x, 0.0, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     )
        >>> nearest_primitive([box(0.3), box(0.5)], [(0.32, 0.0, 0.2)], ambiguity_m=0.02)[0]
        0
        >>> nearest_primitive([box(0.3), box(0.5)], [(0.4, 0.0, 0.2)], ambiguity_m=0.02)[1]
        "ambiguous: two primitives within 0.020 m of the hand's nearest (gaps 0.122, 0.122)"
        >>> nearest_primitive([box(0.3)], [], ambiguity_m=0.02)[0]
        0
    """
    if not regions:
        return None, "no primitive on the support inside the search column"
    if not hand_points:
        if len(regions) == 1:
            return 0, ""
        return None, (
            f"ambiguous: {len(regions)} primitives inside the search box, no hand to pick one"
        )
    gaps = [min(point_box_gap_m(p, r) for p in hand_points) for r in regions]
    order = sorted(range(len(gaps)), key=gaps.__getitem__)
    if len(order) > 1 and gaps[order[1]] - gaps[order[0]] <= ambiguity_m:
        return None, (
            f"ambiguous: two primitives within {ambiguity_m:.3f} m of the hand's nearest "
            f"(gaps {gaps[order[0]]:.3f}, {gaps[order[1]]:.3f})"
        )
    return order[0], ""


class PrimitiveTracker:
    """Identities for primitives across grids; retracts what the map stops holding.

    Association is geometric, label-free: a fit that barely moved (``track_region``
    within ``tol_m``, one voxel) refreshes its primitive; one whose centre lies in a
    primitive's box, or holds that box's centre, is the same object nudged or re-fitted
    (the same identity, the box replaced); anything else is a new primitive. A primitive
    inside the scanned column that no fit matched accrues a miss and is dropped after
    ``max_misses`` grids — the map cleared its cells (the object was taken away), or it
    merged into a neighbour; one outside the column is kept (not looked at).

    Pure and single-threaded by contract: the grasp-target leg updates it from its
    measurement tick only.

    Args:
        max_misses: Consecutive scanned grids without a match before a primitive is
            dropped. *Calibration point.*

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> def fit(x):
        ...     return PrimitiveFit(
        ...         PlaceRegion(
        ...             frame_id="b",
        ...             half_extents=(0.03, 0.03, 0.05),
        ...             pose=Pose6D(xyz=(x, 0.0, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...         ),
        ...         9,
        ...     )
        >>> col = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.5, 0.5, 0.5),
        ...     pose=Pose6D(xyz=(0.5, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> t = PrimitiveTracker(max_misses=2)
        >>> def ids(fits, stamp):
        ...     return [
        ...         p.primitive_id for p in t.update(fits, stamp_ns=stamp, tol_m=0.02, scanned=col)
        ...     ]
        >>> ids([fit(0.3), fit(0.6)], 1), [p.primitive_id for p in t.primitives]
        ([0, 1], [0, 1])
        >>> ids([fit(0.31)], 2), [p.primitive_id for p in t.primitives]  # one miss: kept
        ([0], [0, 1])
        >>> ids([fit(0.31)], 3), [p.primitive_id for p in t.primitives]  # two misses: dropped
        ([0], [0])
    """

    def __init__(self, *, max_misses: int = 3) -> None:
        """Start empty."""
        if max_misses < 1:
            raise ValueError(f"max_misses must be >= 1; got {max_misses}")
        self._max_misses = max_misses
        self._tracks: dict[int, tuple[ObjectPrimitive, int]] = {}  # id -> (primitive, misses)
        self._next_id = 0

    @property
    def primitives(self) -> list[ObjectPrimitive]:
        """Every tracked primitive, by identity."""
        return [p for p, _ in self._tracks.values()]

    def _match(self, region: PlaceRegion, tol_m: float, taken: set[int]) -> int | None:
        for pid, (prim, _) in self._tracks.items():
            if pid in taken:
                continue
            if track_region(prim.region, region, max_centroid_shift_m=tol_m, extents_tol_m=tol_m):
                return pid
        centre = np.asarray([region.pose.xyz], dtype=np.float64)
        for pid, (prim, _) in self._tracks.items():
            if pid in taken:
                continue
            held = np.asarray([prim.region.pose.xyz], dtype=np.float64)
            if bool(_in_region(centre, prim.region)[0]) or bool(_in_region(held, region)[0]):
                return pid
        return None

    def update(
        self, fits: Sequence[PrimitiveFit], *, stamp_ns: int, tol_m: float, scanned: PlaceRegion
    ) -> list[ObjectPrimitive]:
        """Fold one grid's fits in; return the primitives inside ``scanned`` (largest first).

        Args:
            fits: ``primitives_from_voxels`` of this grid's scan of ``scanned``.
            stamp_ns: The grid's ``source_stamp``.
            tol_m: The tracking tolerance (one voxel).
            scanned: The column the fits were extracted from: primitives whose centre lies
                in it and matched nothing are misses.
        """
        taken: set[int] = set()
        out: list[ObjectPrimitive] = []
        for fit in sorted(fits, key=lambda f: f.cell_count, reverse=True):
            pid = self._match(fit.region, tol_m, taken)
            if pid is None:
                pid = self._next_id
                self._next_id += 1
                first = stamp_ns
            else:
                first = self._tracks[pid][0].first_seen_ns
            taken.add(pid)
            prim = ObjectPrimitive(pid, fit.region, fit.cell_count, first, stamp_ns)
            self._tracks[pid] = (prim, 0)
            out.append(prim)
        for pid in list(self._tracks):
            if pid in taken:
                continue
            prim, misses = self._tracks[pid]
            centre = np.asarray([prim.region.pose.xyz], dtype=np.float64)
            if not bool(_in_region(centre, scanned)[0]):
                continue
            misses += 1
            if misses >= self._max_misses:
                del self._tracks[pid]
            else:
                self._tracks[pid] = (prim, misses)
        return out
