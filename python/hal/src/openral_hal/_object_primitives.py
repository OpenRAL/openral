"""Object instances before contact — camera-originated, map-confirmed, tracked.

``docs/reference/object-primitives-design.md`` (issue #349): the grasp region is needed
*before* the fingers reach the target, and a VLA never pauses for the head camera to get
a clean, unoccluded fit while a hand is armed. So the grasp-target leg measures every
object instance near a hand **before** any hand arms, from the head camera: one
``SegmentInView`` mask per instance (``instance_prompt`` says where to prompt), lifted
through the registered depth by the existing fit (``target_region_from_masks``), and
**confirmed** by the voxel map (``map_confirms``: the map holds the box, and the box holds
one body). The camera gives identity and separates objects the map merges (cartons packed
in a bin are one 26-connected blob in the map, HZ-0115-30); the map, which integrates
views from before the hand arrived, confirms each box and — while the declared hand
occludes the camera in the final approach — holds the confirmed box. The map never
originates a box.

Pure numpy, no ROS, no clock. ``PrimitiveTracker`` keeps identities across fits and drops
what goes stale or what the map contradicts; ``nearest_primitive`` picks the instance a
hand approaches, refusing a tie. The leg (``_grasp_target_leg``, ``premeasure=True``) runs
the chosen box through its existing gates before ``accept``; the kernel contract is
unchanged (no region, no exemption).

Every threshold is a *calibration point* (CLAUDE.md §1.2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import numpy as np
from openral_core import PlaceRegion
from openral_core.geometry import homogeneous_from_quat_xyz

from openral_hal._grasp_target import (
    VoxelLattice,
    _check_frame,
    _components,
    _ijk,
    _in_region,
    cell_closed_region,
    occupied_centers_in_box,
    region_covers_occupied,
    track_region,
)

__all__ = [
    "ObjectPrimitive",
    "PrimitiveFit",
    "PrimitiveTracker",
    "instance_prompt",
    "map_confirms",
    "nearest_primitive",
    "point_box_gap_m",
]


@dataclass(frozen=True)
class PrimitiveFit:
    """One camera instance fit, map-confirmed.

    Attributes:
        region: The instance's gravity-aligned box (``target_region_from_masks``), its
            lower face one voxel above the support, stamped no later than the depth
            frame and the grid it was confirmed on.
        cell_count: Occupied map cells inside the box (``region_covers_occupied``).
        support_z: The measured support top the box was fitted on.
    """

    region: PlaceRegion
    cell_count: int
    support_z: float


@dataclass(frozen=True)
class ObjectPrimitive:
    """A tracked primitive: a ``PrimitiveFit`` under a stable identity.

    Attributes:
        primitive_id: Stable across fits while the box tracks (``PrimitiveTracker``).
        region: The newest fit's box.
        cell_count: The newest fit's cell count.
        support_z: The newest fit's support top.
        first_seen_ns: Stamp of the fit the identity was minted at.
        last_seen_ns: Stamp of the newest camera fit (``region.stamp_ns``): the instance's
            ``confirmed_ns``, what its staleness is measured from.
    """

    primitive_id: int
    region: PlaceRegion
    cell_count: int
    support_z: float
    first_seen_ns: int
    last_seen_ns: int


def map_confirms(
    grid: VoxelLattice,
    region: PlaceRegion,
    *,
    support_z: float,
    min_cover: float,
    pad_m: float = 0.0,
) -> tuple[str, str]:
    """Whether the voxel map confirms a camera instance box: ``("", "")``, or ``(kind, why)``.

    The camera originates the box (one mask, one object); the map only confirms it. Two
    gates, both the producer's existing rules: the map must hold the box
    (``region_covers_occupied``, at least ``min_cover`` of its footprint's cells — else
    ``map_disagrees``: the object is gone, or the camera fitted something the map never
    saw), and the occupied cells above the support inside the box's cell closure (what the
    kernel would exempt) must be **one** 26-connected body — else ``map_split``: the box
    reaches a second body the map holds apart from it. A neighbour *touching* the instance
    is one body with it in the map (HZ-0115-30), so a packed pair passes, each instance
    with the sliver of its neighbour inside its own padded box — never the whole merged
    blob, which is what the map alone would have exempted. A neighbour a cell or two
    apart is the same sliver: a second body none of whose cells reaches inside the box
    shrunk by ``pad_m`` (the fit's own padding, sides and top) lies only in the padding
    band and does not split it.

    Args:
        grid: The published lattice; ``region`` must be in its frame.
        region: The camera instance's box.
        support_z: The support top the instance was fitted on.
        min_cover: ``region_covers_occupied``'s fraction. *Calibration point.*
        pad_m: The padding the fit added to the box's sides and top (``fit_pad_m``); a
            second body confined to that band is a sliver, not a split.

    Raises:
        ROSConfigError: On a frame mismatch or a tilted region.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(8 * 4 * 3, dtype=np.uint8)
        >>> for i in (1, 2, 5, 6):  # two 2x2 posts, one cell high, a 2-cell gap apart
        ...     for j in (1, 2):
        ...         occ[i + 8 * (j + 4 * 1)] = 1
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (8, 4, 3), occ)
        >>> def box(x, hx, hz=0.05):
        ...     return PlaceRegion(
        ...         frame_id="b",
        ...         half_extents=(hx, 0.1, hz),
        ...         pose=Pose6D(xyz=(x, 0.2, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     )
        >>> map_confirms(g, box(0.2, 0.1), support_z=0.0, min_cover=0.5)
        ('', '')
        >>> map_confirms(g, box(0.4, 0.3), support_z=0.0, min_cover=0.5)[0]
        'map_split'
        >>> map_confirms(g, box(0.4, 0.05), support_z=0.0, min_cover=0.5)[0]
        'map_disagrees'
        >>> wide = box(0.35, 0.25, hz=0.1)  # reaches 5 cm into the second post
        >>> map_confirms(g, wide, support_z=0.0, min_cover=0.5)[0]
        'map_split'
        >>> map_confirms(g, wide, support_z=0.0, min_cover=0.5, pad_m=0.06)  # only in the pad
        ('', '')
    """
    count, covered = region_covers_occupied(grid, region, min_fraction=min_cover)
    if not covered:
        return "map_disagrees", f"only {count} occupied cells inside the instance box"
    closed, _ = cell_closed_region(region, grid, max_half_extent_m=math.inf)
    centers = occupied_centers_in_box(grid, closed)
    above = centers[centers[:, 2] > support_z + grid.resolution]
    comps = sorted(_components(_ijk(grid, above)), key=len, reverse=True)
    hx, hy, hz = region.half_extents
    x, y, z = region.pose.xyz
    inner = region.model_copy(
        update={
            "half_extents": (max(hx - pad_m, 0.0), max(hy - pad_m, 0.0), max(hz - pad_m / 2, 0.0)),
            "pose": region.pose.model_copy(update={"xyz": (x, y, z - pad_m / 2)}),
        }
    )
    deep = [c for c in comps[1:] if _in_region(above[c], inner).any()]
    if deep:
        parts = [len(c) for c in comps]
        return "map_split", f"{len(parts)} separate bodies inside the instance box (cells {parts})"
    return "", ""


def instance_prompt(
    grid: VoxelLattice,
    column: PlaceRegion,
    covered: Sequence[PlaceRegion],
    *,
    near_xy: tuple[float, float],
    skip: Collection[tuple[int, int, int]] = (),
) -> tuple[tuple[float, float, float], tuple[int, int, int]] | None:
    """Where to prompt the segmenter next: a raised top surface no instance holds yet.

    ``SegmentInView`` takes one positive point, never a whole frame (there is no
    prompt-free mode), so the map says *where* to look and the camera says *what* is
    there. A top-surface cell is an occupied cell of ``column`` with nothing directly
    above it; it is *raised* when it stands more than one cell above the column's lowest
    top surface (the support the hand works over). The prompt is the raised top cell
    nearest ``near_xy`` that lies in no ``covered`` box and is not in ``skip`` (recently
    prompted), preferring an *interior* cell (all four side neighbours raised tops too):
    a packed pair's shared top is one surface in the map, and a prompt one cell inside
    its near edge lands on one body, where its centroid would land on the seam. Once that
    body's instance covers its cells, the next prompt lands on the neighbour — so two
    masks separate what the map merges.

    Returns:
        ``(point, cell)`` — the prompt (the cell's top-face centre, in
        ``grid.frame_id``) and its lattice cell (the ``skip`` key) — or ``None``.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(6 * 6 * 3, dtype=np.uint8)
        >>> occ[:36] = 1  # k=0: a table
        >>> occ[[36 + 4 + 6 * 4, 72 + 4 + 6 * 4]] = 1  # a 2-cell post at i=j=4
        >>> g = VoxelLattice("b", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (6, 6, 3), occ)
        >>> col = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.3, 0.3, 0.15),
        ...     pose=Pose6D(xyz=(0.3, 0.3, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> instance_prompt(g, col, [], near_xy=(0.1, 0.1))
        ((0.45, 0.45, 0.3), (4, 4, 2))
        >>> instance_prompt(g, col, [], near_xy=(0.1, 0.1), skip={(4, 4, 2)}) is None
        True
    """
    _check_frame(grid, column)
    centers = occupied_centers_in_box(grid, column)
    if len(centers) == 0:
        return None
    ijk = _ijk(grid, centers)
    occupied = {(i, j, k) for i, j, k in ijk.tolist()}
    top = np.array([(i, j, k + 1) not in occupied for i, j, k in ijk.tolist()], dtype=bool)
    tops, top_ijk = centers[top], ijk[top]
    raised = tops[:, 2] > float(tops[:, 2].min()) + grid.resolution
    tops, top_ijk = tops[raised], top_ijk[raised]
    free = np.array([(i, j, k) not in skip for i, j, k in top_ijk.tolist()], dtype=bool)
    for region in covered:
        if len(tops) and region.frame_id == grid.frame_id:
            free &= ~_in_region(tops, region)
    if not free.any():
        return None
    columns = {(i, j) for i, j, _ in top_ijk.tolist()}
    edge = np.array(
        [
            not all((i + di, j + dj) in columns for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            for i, j, _ in top_ijk.tolist()
        ],
        dtype=bool,
    )
    d2 = np.sum((tops[:, :2] - np.asarray(near_xy, dtype=np.float64)) ** 2, axis=1)
    candidates = np.flatnonzero(free)
    pick = int(candidates[np.lexsort((d2[candidates], edge[candidates]))[0]])
    x, y, z = (float(v) for v in tops[pick])
    i, j, k = (int(v) for v in top_ijk[pick])
    return (x, y, z + 0.5 * grid.resolution), (i, j, k)


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
    """Identities for instances across fits; drops what goes stale or is contradicted.

    Association is geometric, label-free (``SegmentInView`` names no class): a fit that
    barely moved (``track_region`` within ``tol_m``, one voxel) refreshes its instance;
    one whose centre lies in an instance's box, or holds that box's centre, is the same
    object nudged or re-fitted (the same identity, the box replaced); anything else is a
    new instance. An instance whose centre lies in ``scanned`` that no fit matched accrues
    a miss and is dropped after ``max_misses`` updates — the camera looked there and saw
    something else; one outside it is kept (not looked at). ``prune`` drops by predicate
    (the leg's staleness and map-contradiction rules); past ``max_tracks`` the instance
    seen longest ago is dropped (bounded work).

    Single-threaded by contract: the grasp-target leg touches it under the bridge lock.

    Args:
        max_misses: Consecutive scans without a match before an instance is dropped.
            *Calibration point.*
        max_tracks: Most instances held at once. *Calibration point.*

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
        ...         0.0,
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

    def __init__(self, *, max_misses: int = 3, max_tracks: int = 16) -> None:
        """Start empty."""
        if max_misses < 1 or max_tracks < 1:
            raise ValueError(
                f"max_misses and max_tracks must be >= 1; got {max_misses}, {max_tracks}"
            )
        self._max_misses = max_misses
        self._max_tracks = max_tracks
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
        """Fold fits in; return their instances (largest first).

        Args:
            fits: Map-confirmed camera fits of one capture.
            stamp_ns: The capture's stamp (the fits' ``region.stamp_ns``).
            tol_m: The tracking tolerance (one voxel).
            scanned: Where the capture looked: instances whose centre lies in it and
                matched nothing are misses.
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
            prim = ObjectPrimitive(pid, fit.region, fit.cell_count, fit.support_z, first, stamp_ns)
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
        while len(self._tracks) > self._max_tracks:
            del self._tracks[min(self._tracks, key=lambda p: self._tracks[p][0].last_seen_ns)]
        return out

    def prune(self, keep: Callable[[ObjectPrimitive], bool]) -> list[ObjectPrimitive]:
        """Drop every instance ``keep`` refuses; return what was dropped."""
        dropped = [p for p, _ in self._tracks.values() if not keep(p)]
        for prim in dropped:
            del self._tracks[prim.primitive_id]
        return dropped
