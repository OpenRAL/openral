"""Pre-grasp target geometry — seed from the voxel map, region from a SAM mask, map cross-check.

Pure numpy core of the grasp-target producer specified in
``docs/reference/real-pick-place-design.md`` §2.2. No ROS, no clock, no model:
the node that wires TF, the segmenter and ``OccupancyVoxels`` into these
functions is a follow-up (it needs ``GraspDeclaration`` on the wire first).

Flow, one call per step so every step is replayable from its inputs alone:

1. ``occupied_centers_in_box`` — occupied cells of the published voxel lattice
   whose centre lies in the scene's search box (the box only *seeds*
   perception; it never becomes the exemption region).
2. ``support_top_from_voxels`` — the support plane is **measured**: scanning a
   column under the search box top-down, the first layer whose top-surface cells
   ring the footprint of the target standing above it, its top face. The search
   box's own bottom (a lifted detection bbox) can sit below the real table top,
   so it is never taken as the support (HZ-0115-6). Both steps anchor the target
   on one cell — the top of the occupied column nearest the search box centre.
   ``target_seed_from_voxels`` clusters the cells above that plane and takes the
   anchored cluster's top-centre. Refuses on too few cells or on another
   comparably-sized cluster (HZ-0115-2, wrong object: guessing between two
   candidates is exactly the mis-declaration the hazard row names).
3. ``project_point`` — that seed into the camera image as SAM 2.1's positive
   point prompt.
4. ``target_region_from_mask`` — eroded mask → masked depth → base-frame cloud
   → gravity-aligned oriented box whose lower face sits **one voxel above the
   support plane, never below it** (HZ-0115-6: a region reaching into the
   support surface would exempt the very cells that stop the fingers from
   being driven into the table).
5. ``region_covers_occupied`` — the fitted region must contain occupied cells
   the kernel itself sees, or it describes nothing the map agrees with.
6. ``track_region`` — the 2-5 Hz re-prompt gate: a re-fit is only accepted as
   the same target if it barely moved. ``region_within`` tells a re-fit that
   shrank inside the held region (the approaching hand occluding part of the
   target) from one that reaches outside it (the target moved).
7. ``cell_closed_region`` — only what goes to the kernel: the held fit grown so
   every lattice cell it touches has its centre inside (the kernel's exemption
   test), horizontally and up, never down. The gates above keep comparing the
   tight fits. ``margin_grown_region`` first bloats it by the configured grasp-target
   margin: on every face (downward too) for the kernel's pre-handover region, on the
   sides and top only for the attached payload (HZ-0115-32).
8. ``occupied_touching_outside`` — before a fit is accepted, the region the kernel
   would get must hold the target's whole map component: no occupied cell above the
   support touches it from outside (a fit of the part the hand left in view would
   leave the far edge's cells unexempt).
9. ``map_completed_region`` — a fit that fails 8 is completed from the map instead: the
   region grows, in the fit's yaw and never down, to the centres of the target's whole
   26-connected map component (the map remembers views from before the hand arrived),
   refused when that component touches the search column's edge, exceeds the caps or
   leaves another body's cell inside the result.

Every threshold here is a **calibration point** (CLAUDE.md §1.2): the caps are
the design note's Safety-WG placeholders (half-extent ≤ 0.20 m, volume ≤
0.03 m³), the rest are chosen conservatively, not measured.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from openral_core import PlaceRegion, Pose6D
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw

from openral_hal._vision_attachment_evidence import backproject_masked_depth

if TYPE_CHECKING:  # pragma: no cover — typing only
    from collections.abc import Iterable, Sequence

    from openral_core import IntrinsicsPinhole

__all__ = [
    "TargetRefusal",
    "TargetRegionFit",
    "TargetSeed",
    "VoxelLattice",
    "cell_closed_region",
    "map_completed_region",
    "margin_grown_region",
    "mask_without_removed_points",
    "occupied_centers_in_box",
    "occupied_touching_outside",
    "project_point",
    "region_covers_occupied",
    "region_within",
    "support_top_from_voxels",
    "target_region_from_mask",
    "target_region_from_masks",
    "target_seed_from_voxels",
    "track_region",
]

# Unit-quaternion tolerance, the same one bucket2_markers.occupied_voxel_centers applies:
# an unset OccupancyVoxels.orientation is all zeros, never identity.
_UNIT_QUAT_TOL = 1e-6

# 26-connectivity offsets: octomap marks a single view's surface shell, which on any
# slanted face is only diagonally connected — 6-connectivity would shatter one object.
_NEIGHBOURS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


class TargetRefusal(StrEnum):
    """Why a step declined to produce a seed or a region (logged, never swallowed)."""

    TOO_FEW_CELLS = "too_few_cells"
    AMBIGUOUS = "ambiguous"
    TOO_FEW_POINTS = "too_few_points"
    NO_HEIGHT_ABOVE_SUPPORT = "no_height_above_support"
    NOT_ON_SUPPORT = "not_on_support"
    HALF_EXTENT_CAP = "half_extent_cap"
    VOLUME_CAP = "volume_cap"
    STACKED = "stacked"


# Most severe first: how ``target_region_from_masks`` reports a candidate set it refused
# whole. Every refusal but ``TOO_FEW_POINTS`` contradicts a held region (``not_on_support``
# only when it is not the closing hand's occlusion, ``_grasp_target_leg._refused_fit``).
_REFUSAL_SEVERITY = (
    TargetRefusal.STACKED,
    TargetRefusal.HALF_EXTENT_CAP,
    TargetRefusal.VOLUME_CAP,
    TargetRefusal.NO_HEIGHT_ABOVE_SUPPORT,
    TargetRefusal.NOT_ON_SUPPORT,
    TargetRefusal.TOO_FEW_POINTS,
)


#: Camera-mount error budget of a target fit (``target_region_from_mask``). *Calibration point.*
EXTRINSIC_ERROR_M = 0.01


def fit_pad_m(resolution: float, extrinsic_error_m: float = EXTRINSIC_ERROR_M) -> float:
    """The padding a target fit adds to its sides and top, metres.

    A cell whose centre is outside a box can still overlap it by half a cell diagonal,
    plus the camera mount's error.

    Example:
        >>> round(fit_pad_m(0.02), 4)
        0.0273
    """
    return math.sqrt(3.0) * resolution / 2.0 + extrinsic_error_m


@dataclass(frozen=True)
class VoxelLattice:
    """The fields of one ``openral_msgs/OccupancyVoxels`` the target producer reads.

    Same convention as ``openral_foxglove_bringup.bucket2_markers.occupied_voxel_centers``
    (not importable from the HAL layer, so re-derived here and cross-checked in
    ``tests/unit/test_grasp_target.py``): ``idx = x + size_x * (y + size_y * z)``;
    ``origin`` + ``orientation_xyzw`` is the pose of cell ``(0, 0, 0)``'s minimum
    corner in ``frame_id``; the lattice axes are the octomap's, so a cell centre is
    ``origin + R @ ((i, j, k) + 0.5) * resolution``.

    Attributes:
        frame_id: Frame the lattice pose is expressed in (the robot base frame).
        origin: Minimum corner of cell ``(0, 0, 0)``, metres.
        orientation_xyzw: Lattice orientation; must be a unit quaternion.
        resolution: Cell edge, metres.
        size: ``(size_x, size_y, size_z)``.
        occupancy: Flat ``size_x * size_y * size_z`` array; non-zero is occupied.

    Raises:
        ROSConfigError: On a length mismatch, a non-positive resolution or a non-unit
            orientation.
    """

    frame_id: str
    origin: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    resolution: float
    size: tuple[int, int, int]
    occupancy: NDArray[np.uint8]

    def __post_init__(self) -> None:
        expected = int(np.prod(self.size))
        if self.occupancy.size != expected:
            raise ROSConfigError(
                f"VoxelLattice: occupancy length {self.occupancy.size} != prod(size)={expected}."
            )
        if not self.resolution > 0.0:
            raise ROSConfigError(f"VoxelLattice: resolution {self.resolution!r} must be > 0.")
        norm2 = sum(q * q for q in self.orientation_xyzw)
        if not math.isfinite(norm2) or abs(norm2 - 1.0) > _UNIT_QUAT_TOL:
            raise ROSConfigError(
                f"VoxelLattice: orientation {self.orientation_xyzw!r} is not a unit quaternion."
            )

    def rotation(self) -> NDArray[np.float64]:
        """``(3, 3)`` rotation from lattice axes to ``frame_id``."""
        return homogeneous_from_quat_xyz((0.0, 0.0, 0.0), self.orientation_xyzw)[:3, :3]

    def occupied_centers(self) -> NDArray[np.float64]:
        """``(N, 3)`` centres of every occupied cell in ``frame_id`` (read-only, cached).

        The whole-grid scan runs once per lattice — the producer legs warm it where the
        grid arrives, outside the vision bridge's lock — and every later call is free.
        """
        cached: NDArray[np.float64] | None = self.__dict__.get("_occupied_centers")
        if cached is None:
            sx, sy, _ = self.size
            idx = np.flatnonzero(self.occupancy)
            ijk = np.stack((idx % sx, (idx // sx) % sy, idx // (sx * sy)), axis=1)
            cached = _cell_centres(self, ijk)
            cached.setflags(write=False)
            # Frozen: the lattice never changes, so neither do its centres.
            self.__dict__["_occupied_centers"] = cached
        return cached


@dataclass(frozen=True)
class TargetSeed:
    """Result of ``target_seed_from_voxels``.

    Attributes:
        point: Top-centre of the selected cluster in the lattice frame, or ``None``.
        refusal: Why no seed was produced; ``None`` when ``point`` is set.
        cluster_sizes: Cell counts of every cluster found, the anchored target's
            first, the rest descending (for the trace).
        bottom_z: Bottom face of the selected cluster's lowest cell, or ``None``.
    """

    point: tuple[float, float, float] | None
    refusal: TargetRefusal | None
    cluster_sizes: tuple[int, ...]
    bottom_z: float | None = None


@dataclass(frozen=True)
class TargetRegionFit:
    """Result of ``target_region_from_mask``.

    Attributes:
        region: The measured box in the base frame, or ``None`` when refused.
        refusal: Why no region was produced; ``None`` when ``region`` is set.
        point_count: Points that survived erosion and depth validity.
        depth_valid_fraction: Fraction of eroded-mask pixels with usable depth.
        half_extents: The fitted (padded) half-extents, also reported on refusal.
        box: The fitted box on a ``NOT_ON_SUPPORT`` refusal (the visible part, its lower
            face pinned one voxel above the support like a region's), else ``None``: the
            leg tells the closing hand occluding the target's lower part (that box inside
            the held region) from a contradiction (``_refused_fit``).
        z_span: Height of the (trimmed) cloud, top minus lowest point; 0 when too few.
        support_z: The support the region stands on: the measured one, or, when none
            is measured under the target (``unseen``), one voxel under the target's own
            lowest kept point — so ``region``'s lower face is always ``support_z +
            resolution``. ``None`` when nothing was fitted.
        support_unseen: ``True`` when ``support_z`` is the target's own lower face, not
            a measured surface.
    """

    region: PlaceRegion | None
    refusal: TargetRefusal | None
    point_count: int
    depth_valid_fraction: float
    half_extents: tuple[float, float, float]
    box: PlaceRegion | None = None
    z_span: float = 0.0
    support_z: float | None = None
    support_unseen: bool = False


def _in_region(points: NDArray[np.float64], region: PlaceRegion) -> NDArray[np.bool_]:
    """Exact point-in-OBB test, the kernel's own predicate (centre inside, faces inclusive)."""
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    local = (points - t[:3, 3]) @ t[:3, :3]
    return np.asarray(np.all(np.abs(local) <= np.asarray(region.half_extents), axis=1))


def _check_frame(grid: VoxelLattice, region: PlaceRegion) -> None:
    if region.frame_id != grid.frame_id:
        raise ROSConfigError(
            f"region frame {region.frame_id!r} != lattice frame {grid.frame_id!r}; "
            "a box measured in one frame and applied in another names the wrong volume."
        )


def occupied_centers_in_box(grid: VoxelLattice, box: PlaceRegion) -> NDArray[np.float64]:
    """Occupied cell centres of ``grid`` that lie inside the oriented ``box``.

    Args:
        grid: The published lattice.
        box: Search box (or any oriented box) in ``grid.frame_id``.

    Returns:
        ``(N, 3)`` centres in ``grid.frame_id``.

    Raises:
        ROSConfigError: If ``box.frame_id`` is not the lattice frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(8, dtype=np.uint8)
        >>> occ[7] = 1  # cell (1, 1, 1)
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (2, 2, 2), occ)
        >>> box = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.1, 0.1, 0.1),
        ...     pose=Pose6D(xyz=(0.15, 0.15, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> occupied_centers_in_box(g, box).round(3).tolist()
        [[0.15, 0.15, 0.15]]
    """
    _check_frame(grid, box)
    centers = grid.occupied_centers()
    return centers[_in_region(centers, box)]


def _layer_index(grid: VoxelLattice, z: NDArray[np.float64]) -> NDArray[np.int64]:
    """Lattice z-layer of each cell centre; the lattice must be z-up (yaw-only)."""
    if abs(grid.rotation()[2, 2] - 1.0) > _UNIT_QUAT_TOL:
        raise ROSConfigError(
            "VoxelLattice: a support layer needs a z-up (yaw-only) lattice; "
            f"orientation {grid.orientation_xyzw!r} tilts it."
        )
    return np.asarray(np.round((z - grid.origin[2]) / grid.resolution - 0.5), dtype=np.int64)


def support_top_from_voxels(
    grid: VoxelLattice,
    centers: NDArray[np.float64],
    *,
    near_xy: tuple[float, float],
    min_cells: int,
    probe_margin_m: float = 0.05,
    surface_centers: NDArray[np.float64] | None = None,
) -> float | None:
    """Measure the surface the target stands on: the top face of that layer.

    Scanned **top-down** over the lattice z-layers of ``centers`` (a column under
    the search box). The target is anchored once, on the highest cell of the
    occupied column horizontally nearest ``near_xy``; for each candidate layer
    ``k`` below that anchor, the target is the connected component of the cells
    *above* ``k`` that holds the anchor — the same object at every layer, so a
    taller neighbour in the column can never turn the target's own top face into
    its "support" (HZ-0115-2: the seed would land on the neighbour). ``k`` is the
    support when it holds at least ``min_cells`` **top-surface** cells (no
    occupied cell directly above) in the ring around that component's footprint —
    farther than one cell from it (the target's own side faces) and at most
    ``probe_margin_m`` from it. A surface extends laterally beyond what stands on
    it; the target's own layers do not, so a dense target top is never taken for
    the support, and the probe scales with the target's footprint instead of a
    fixed square. The ring skips the band right at the footprint only by one
    cell, so a head camera's occlusion shadow behind the target merely thins it.

    Ceiling: a target whose lower part flares more than one cell past its upper
    footprint (a pyramid, a handle) can stop the scan inside the target; the
    region then stands higher than the support — less exemption, never more.
    And a search box centred on bare support (between two objects, or the table
    seen through a hollow target) anchors on the support itself: no layer below
    it qualifies and there is no support — a refusal, never a guess.

    Args:
        grid: The lattice ``centers`` came from; must be z-up (yaw-only).
        centers: ``(N, 3)`` occupied centres in ``grid.frame_id``.
        near_xy: Where the target is expected (the search box centre), in
            ``grid.frame_id``.
        min_cells: Fewest ring cells a layer needs to be the support.
            *Calibration point.*
        probe_margin_m: Outer reach of the ring from the target footprint, metres;
            at least two cells, since the ring starts one cell out. *Calibration point.*
        surface_centers: Occupied centres the ring is counted on (default
            ``centers``). The ring lies outside the target's footprint and so
            mostly outside a search column padded tightly around it: pass the
            column grown by ``probe_margin_m`` plus a cell, or the whole map.
            The target and its anchor always come from ``centers``.

    Returns:
        The support's top face z in ``grid.frame_id``, or ``None`` when no layer
        qualifies — there is no measured support to stand a region on.

    Raises:
        ROSConfigError: On a tilted lattice (its cells form no horizontal layers), or a
            ``probe_margin_m`` under two cells (a ring of zero width).

    Example:
        >>> import numpy as np
        >>> occ = np.zeros(5 * 5 * 3, dtype=np.uint8)
        >>> occ[25:50] = 1  # the whole k=1 layer: a table top
        >>> occ[50 + 12] = 1  # one cell on it, at i=j=2
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (5, 5, 3), occ)
        >>> c, xy = g.occupied_centers(), (0.25, 0.25)
        >>> round(support_top_from_voxels(g, c, near_xy=xy, min_cells=4, probe_margin_m=0.2), 6)
        0.2
        >>> support_top_from_voxels(g, c, near_xy=xy, min_cells=17, probe_margin_m=0.2) is None
        True
    """
    if len(centers) == 0:
        return None
    _layer_index(grid, centers[:, 2])  # z-up check
    if probe_margin_m < 2.0 * grid.resolution - 1e-9:
        raise ROSConfigError(
            f"support probe_margin_m={probe_margin_m!r} is under two cells of the "
            f"{grid.resolution!r} m lattice: the ring between one cell and it would be empty."
        )
    ijk = _ijk(grid, centers)
    occupied = _cell_set(grid, centers)
    around = occupied if surface_centers is None else _cell_set(grid, surface_centers)
    i0, j0, k0 = (int(v) for v in ijk[_anchor(centers, ijk, near_xy)])
    start = (i0, j0, k0)
    reach = math.ceil(probe_margin_m / grid.resolution - 1e-9)
    for k in sorted({c[2] for c in occupied | around if c[2] < start[2]}, reverse=True):
        footprint = {(i, j) for i, j, _ in _component_from(occupied, [start], min_k=k + 1)}
        inner = _dilate(footprint, 1)
        ring = _dilate(footprint, reach) - inner
        surface = sum(
            1 for i, j, kk in around if kk == k and (i, j) in ring and (i, j, k + 1) not in around
        )
        if surface >= min_cells:
            return float(grid.origin[2] + (k + 1) * grid.resolution)
    return None


def _ijk(grid: VoxelLattice, centers: NDArray[np.float64]) -> NDArray[np.int64]:
    """``(N, 3)`` integer lattice cells holding ``centers``."""
    local = (centers - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
    return np.asarray(np.floor(local), dtype=np.int64)


def _cell_centres(grid: VoxelLattice, ijk: NDArray[np.int64]) -> NDArray[np.float64]:
    """``(N, 3)`` centres of integer lattice cells in ``grid.frame_id`` (inverse of ``_ijk``)."""
    local = (ijk.astype(np.float64) + 0.5) * grid.resolution
    return np.asarray(local @ grid.rotation().T + np.asarray(grid.origin), np.float64)


def _cell_set(grid: VoxelLattice, centers: NDArray[np.float64]) -> set[tuple[int, int, int]]:
    """Integer lattice cells of ``centers``."""
    return {(int(i), int(j), int(k)) for i, j, k in _ijk(grid, centers).tolist()}


def _anchor(
    centers: NDArray[np.float64], ijk: NDArray[np.int64], near_xy: tuple[float, float]
) -> int:
    """Row of the target's anchor: the highest cell of the column nearest ``near_xy``.

    Cells of one lattice column share their centre's xy exactly on a z-up lattice,
    so the distance ties within a column and the highest ``k`` breaks them.
    """
    dist2 = np.sum((centers[:, :2] - np.asarray(near_xy)) ** 2, axis=1)
    return int(np.lexsort((-ijk[:, 2], dist2))[0])


def _dilate(cells: set[tuple[int, int]], r: int) -> set[tuple[int, int]]:
    """Chebyshev dilation of a 2-D cell set by ``r`` cells."""
    steps = range(-r, r + 1)
    return {(i + di, j + dj) for i, j in cells for di in steps for dj in steps}


def _component_from(
    occupied: set[tuple[int, int, int]],
    starts: Iterable[tuple[int, int, int]],
    *,
    min_k: int | None = None,
) -> list[tuple[int, int, int]]:
    """The 26-connected component of ``starts`` among the occupied cells with ``k >= min_k``."""
    seen = set(starts)
    queue = deque(seen)
    while queue:
        ci, cj, ck = queue.popleft()
        for di, dj, dk in _NEIGHBOURS_26:
            n = (ci + di, cj + dj, ck + dk)
            if (min_k is None or n[2] >= min_k) and n in occupied and n not in seen:
                seen.add(n)
                queue.append(n)
    return list(seen)


def _components(ijk: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    """26-connected components of integer cells, as index arrays into ``ijk``."""
    lookup = {tuple(c): n for n, c in enumerate(ijk.tolist())}
    seen = np.zeros(len(ijk), dtype=bool)
    out: list[NDArray[np.int64]] = []
    for start in range(len(ijk)):
        if seen[start]:
            continue
        seen[start] = True
        members = [start]
        queue = deque([start])
        while queue:
            ci, cj, ck = ijk[queue.popleft()]
            for di, dj, dk in _NEIGHBOURS_26:
                n = lookup.get((ci + di, cj + dj, ck + dk))
                if n is not None and not seen[n]:
                    seen[n] = True
                    members.append(n)
                    queue.append(n)
        out.append(np.asarray(members, dtype=np.int64))
    return out


def target_seed_from_voxels(
    grid: VoxelLattice,
    centers: NDArray[np.float64],
    *,
    near_xy: tuple[float, float],
    support_z: float,
    min_cells: int,
    ambiguity_ratio: float = 0.5,
) -> TargetSeed:
    """Pick the one object in the search box and return its top-centre.

    Cells whose centre is within one voxel of the support plane are the support
    itself and are dropped; the rest are labelled into 26-connected components on
    the lattice. The target is the component holding the anchor — the highest cell
    of the occupied column nearest ``near_xy``, the same anchor
    ``support_top_from_voxels`` measures under — never merely the largest: a
    taller neighbour is not the object the search box centres on. When any other
    component holds at least ``ambiguity_ratio`` of the target's cells there is no
    single object to name and the seed is refused (HZ-0115-2).

    Args:
        grid: The lattice ``centers`` came from (its pose turns centres back into
            integer cells for the connectivity labelling).
        centers: ``(N, 3)`` occupied centres in ``grid.frame_id``, e.g. from
            ``occupied_centers_in_box``.
        near_xy: Where the target is expected (the search box centre).
        support_z: Height of the support plane in ``grid.frame_id`` (z up).
        min_cells: Fewest cells the selected cluster may have. *Calibration point.*
        ambiguity_ratio: Other cluster / target size at or above which the choice
            is ambiguous. *Calibration point.*

    Returns:
        A ``TargetSeed``; ``point`` is ``(x_mean, y_mean, z_top)`` where ``z_top`` is
        the top face of the highest cell, ``bottom_z`` the bottom face of the lowest.

    Example:
        >>> import numpy as np
        >>> occ = np.zeros(27, dtype=np.uint8)
        >>> occ[[4, 13, 22]] = 1  # the column at i=1, j=1, k=0..2
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (3, 3, 3), occ)
        >>> c = g.occupied_centers()
        >>> s = target_seed_from_voxels(g, c, near_xy=(0.15, 0.15), support_z=0.0, min_cells=2)
        >>> [round(v, 3) for v in s.point], s.cluster_sizes
        ([0.15, 0.15, 0.3], (2,))
    """
    above = centers[centers[:, 2] > support_z + grid.resolution]
    local = (above - np.asarray(grid.origin)) @ grid.rotation()
    ijk = np.floor(local / grid.resolution).astype(np.int64)
    if len(above) == 0:
        return TargetSeed(None, TargetRefusal.TOO_FEW_CELLS, ())
    anchor = _anchor(above, ijk, near_xy)
    comps = sorted(_components(ijk), key=lambda c: (anchor not in c, -len(c)))
    sizes = tuple(len(c) for c in comps)
    if sizes[0] < min_cells:
        return TargetSeed(None, TargetRefusal.TOO_FEW_CELLS, sizes)
    if any(size >= ambiguity_ratio * sizes[0] for size in sizes[1:]):
        return TargetSeed(None, TargetRefusal.AMBIGUOUS, sizes)
    cluster = above[comps[0]]
    x, y = cluster[:, :2].mean(axis=0)
    z_top = float(cluster[:, 2].max()) + 0.5 * grid.resolution
    bottom = float(cluster[:, 2].min()) - 0.5 * grid.resolution
    return TargetSeed((float(x), float(y), z_top), None, sizes, bottom)


def project_point(
    point_base: tuple[float, float, float],
    t_cam_from_base: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
) -> tuple[float, float] | None:
    """Project a base-frame point into the image, for SAM's positive point prompt.

    Args:
        point_base: Point in the base frame.
        t_cam_from_base: ``(4, 4)`` transform into the camera **optical** frame
            (REP-103: +x right, +y down, +z forward).
        intrinsics: Pinhole intrinsics of the image the prompt is for.

    Returns:
        Continuous ``(u, v)`` in the same pixel-centre convention as
        ``backproject_masked_depth`` (column ``c`` spans ``[c, c + 1)``), or ``None``
        when the point is behind the camera or outside the image.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=4, height=2, fx=2.0, fy=2.0, cx=2.0, cy=1.0)
        >>> project_point((0.125, 0.125, 0.5), np.eye(4), k)
        (2.5, 1.5)
        >>> project_point((0.0, 0.0, -1.0), np.eye(4), k) is None
        True
    """
    p = t_cam_from_base @ np.array((*point_base, 1.0))
    if p[2] <= 0.0:
        return None
    u = float(intrinsics.fx * p[0] / p[2] + intrinsics.cx)
    v = float(intrinsics.fy * p[1] / p[2] + intrinsics.cy)
    if not (0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height):
        return None
    return u, v


def _erode(mask: NDArray[np.bool_], px: int) -> NDArray[np.bool_]:
    """4-neighbour binary erosion, ``px`` passes; image-border pixels erode away."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(px):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        e[0, :] = e[-1, :] = False
        e[:, 0] = e[:, -1] = False
        m = e
    return m


def mask_without_removed_points(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    kept_points_base: NDArray[np.floating[Any]],
    *,
    cell_m: float = 0.005,
) -> NDArray[np.bool_]:
    """``mask`` without the pixels whose depth point the robot self-filter removed.

    ``kept_points_base`` is the robot self-filter's output for the SAME capture
    (``openral_octomap_bridge``'s ``robot_self_filter``: the robot's and a held
    payload's returns removed against the collision model at the measured joint
    state), in the base frame. A masked pixel is kept only if a kept point lies in its
    own or a neighbouring ``cell_m`` cell — the robot is never re-modelled here, so
    the fingers closing in on a target leave its fit exactly as they leave the map.
    Pixels with no depth are left as they are (they contribute nothing). Removing
    pixels only ever shrinks the fit.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=4, height=4, fx=4.0, fy=4.0, cx=2.0, cy=2.0)
        >>> m = np.ones((4, 4), dtype=bool)
        >>> mask_without_removed_points(m, np.ones((4, 4)), k, np.eye(4), np.zeros((0, 3))).any()
        np.False_
    """
    rows, cols = np.nonzero(mask & np.isfinite(depth_m) & (depth_m > 0.0))
    out = mask.copy()
    if len(rows) == 0:
        return out
    z = depth_m[rows, cols]
    pts_cam = np.stack(
        [
            (cols + 0.5 - intrinsics.cx) / intrinsics.fx * z,
            (rows + 0.5 - intrinsics.cy) / intrinsics.fy * z,
            z,
        ],
        axis=1,
    )
    pts = pts_cam @ np.asarray(t_base_from_cam)[:3, :3].T + np.asarray(t_base_from_cam)[:3, 3]

    def code(keys: NDArray[np.int64]) -> NDArray[np.int64]:
        # ponytail: 21 bits per axis (±5 km at 5 mm), plenty for one camera's view.
        k = keys + (1 << 20)
        return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]

    kept: NDArray[np.float64] = np.asarray(kept_points_base, dtype=np.float64).reshape(-1, 3)
    kept = kept[np.isfinite(kept).all(axis=1)]
    kept_codes = np.unique(code(np.floor(kept / cell_m).astype(np.int64)))
    own = np.floor(pts / cell_m).astype(np.int64)
    found = np.zeros(len(pts), dtype=bool)
    for offset in [*_NEIGHBOURS_26, (0, 0, 0)]:
        found |= np.isin(code(own + np.asarray(offset, dtype=np.int64)), kept_codes)
    out[rows[~found], cols[~found]] = False
    return out


def target_region_from_mask(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    *,
    support_z: float | None,
    resolution: float,
    frame_id: str,
    evidence_ref: str,
    stamp_ns: int = 0,
    erode_px: int = 2,
    extrinsic_error_m: float = EXTRINSIC_ERROR_M,
    trim_percentile: float = 1.0,
    min_points: int = 200,
    min_depth_m: float = 0.1,
    max_depth_m: float = 3.0,
    max_half_extent_m: float = 0.20,
    max_volume_m3: float = 0.03,
) -> TargetRegionFit:
    """Measure the target's oriented box in the base frame from one mask + depth frame.

    The box is **gravity-aligned** (yaw-only): its horizontal axes are the 2-D PCA
    axes of the cloud's footprint, its lower face is ``support_z + resolution`` and
    its upper face the cloud's top. A full 3-D OBB of a top-down view tilts with
    the visible surface, and its bottom face then cannot be extruded to the support
    plane without either growing past it or stopping short of it.

    **The lower face never goes below ``support_z + resolution`` (HZ-0115-6).** The
    cells holding the support surface have centres up to half a voxel above the
    plane; one full voxel keeps every one of them outside the region, so the kernel
    still stops the fingers at the table under the target. Padding is applied to the
    four sides and the top only.

    **The masked cloud must reach down to the support** — its lowest point (after
    the trim) within two voxels of ``support_z``, else ``NOT_ON_SUPPORT``. The
    voxel seed check sees a stack as one cluster: a target on a same-footprint
    box, or on a riser hidden under it, clusters with what it stands on, whose
    bottom is on the support, while the region's lower face is pinned to the
    support and would exempt the lower object's cells (HZ-0115-6). The mask names
    the target alone; a view that sees only its top face (straight down, or its
    lower part occluded) is refused too — less exemption, never more.

    **An unseen support is not a refusal.** With no support measured under the target
    (``support_z=None``: a bin floor the head camera cannot see), or a measured layer
    above the target's own lowest kept point by more than a voxel (a neighbour's top
    face, a bin rim: not what the target stands on), the lower face is the target's own
    lowest kept point — the box never reaches down into space nothing measured, and the
    stacked-body check needs a measured support under the target, so it does not apply.

    Padding is ``√3·resolution/2 + extrinsic_error_m`` — a cell whose centre is
    outside a box can still overlap it by half a cell diagonal, plus the camera
    mount's error. The cloud is trimmed to the ``[trim, 100 - trim]`` percentile
    range per axis first (flying pixels at the mask edge), which only shrinks the
    box, i.e. toward the conservative side.

    Args:
        mask: ``(H, W)`` SAM mask on the depth raster (resample with
            ``depth_cloud.resample_mask_nearest`` first).
        depth_m: ``(H, W)`` metric depth.
        intrinsics: Intrinsics at ``(H, W)`` (``depth_cloud.intrinsics_from_camera_info``).
        t_base_from_cam: ``(4, 4)`` optical-frame → base-frame transform.
        support_z: Support-plane height in the base frame, or ``None`` when no support
            was measured under the target.
        resolution: The voxel lattice's cell edge.
        frame_id: Base frame name stamped on the region.
        evidence_ref: Free-text provenance (camera, mask id, trace).
        stamp_ns: Producer stamp.
        erode_px: Mask erosion passes, removing the mixed-depth boundary.
        extrinsic_error_m: Camera-mount error budget. *Calibration point.*
        trim_percentile: Per-axis outlier trim. *Calibration point.*
        min_points: Fewest cloud points to fit at all. *Calibration point.*
        min_depth_m: Depth readings at or below this are invalid.
        max_depth_m: Depth readings at or above this are invalid.
        max_half_extent_m: Per-axis cap (Safety-WG placeholder).
        max_volume_m3: Volume cap (Safety-WG placeholder).

    Returns:
        A ``TargetRegionFit``; ``region`` is ``None`` on refusal.

    Raises:
        ROSConfigError: Via ``backproject_masked_depth`` on a shape/intrinsics mismatch.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=64, height=64, fx=64.0, fy=64.0, cx=32.0, cy=32.0)
        >>> m = np.zeros((64, 64), dtype=bool)
        >>> m[24:40, 24:40] = True
        >>> t = np.diag([1.0, -1.0, -1.0, 1.0])  # camera 1 m up looking straight down
        >>> t[2, 3] = 1.0
        >>> fit = target_region_from_mask(  # a 3 cm tile: straight down sees only its top
        ...     m,
        ...     np.full((64, 64), 0.9),
        ...     k,
        ...     t,
        ...     support_z=0.07,
        ...     resolution=0.02,
        ...     frame_id="base",
        ...     evidence_ref="doc",
        ...     min_points=10,
        ... )
        >>> fit.refusal is None, round(fit.region.pose.xyz[2] - fit.region.half_extents[2], 3)
        (True, 0.09)
    """
    eroded = _erode(mask, erode_px)
    pts_cam, valid = backproject_masked_depth(
        eroded, depth_m, intrinsics, min_depth_m=min_depth_m, max_depth_m=max_depth_m
    )
    n = len(pts_cam)
    if n < min_points:
        return TargetRegionFit(None, TargetRefusal.TOO_FEW_POINTS, n, valid, (0.0, 0.0, 0.0))
    pts = pts_cam @ t_base_from_cam[:3, :3].T + t_base_from_cam[:3, 3]

    xy = pts[:, :2]
    xy_mean = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - xy_mean, full_matrices=False)
    yaw = math.atan2(float(vt[0, 1]), float(vt[0, 0]))
    c, s = math.cos(yaw), math.sin(yaw)
    rot2 = np.array(((c, -s), (s, c)))
    uv = (xy - xy_mean) @ rot2
    lo_uv, hi_uv = np.percentile(uv, (trim_percentile, 100.0 - trim_percentile), axis=0)
    low, top = (
        float(v) for v in np.percentile(pts[:, 2], (trim_percentile, 100.0 - trim_percentile))
    )

    pad = fit_pad_m(resolution, extrinsic_error_m)
    unseen = support_z is None or low < support_z - resolution
    if unseen:
        support_z = low - resolution
    assert support_z is not None
    bottom = support_z + resolution
    z_hi = top + pad
    half_xy = (hi_uv - lo_uv) / 2.0 + pad
    half = (float(half_xy[0]), float(half_xy[1]), (z_hi - bottom) / 2.0)
    span = top - low
    if top <= bottom:
        return TargetRegionFit(
            None, TargetRefusal.NO_HEIGHT_ABOVE_SUPPORT, n, valid, half, z_span=span
        )
    if max(half) > max_half_extent_m:
        return TargetRegionFit(None, TargetRefusal.HALF_EXTENT_CAP, n, valid, half, z_span=span)
    if 8.0 * half[0] * half[1] * half[2] > max_volume_m3:
        return TargetRegionFit(None, TargetRefusal.VOLUME_CAP, n, valid, half, z_span=span)

    centre_xy = xy_mean + rot2 @ ((lo_uv + hi_uv) / 2.0)
    centre = (float(centre_xy[0]), float(centre_xy[1]), (z_hi + bottom) / 2.0)
    region = PlaceRegion(
        frame_id=frame_id,
        pose=Pose6D(xyz=centre, quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame_id),
        half_extents=half,
        evidence_ref=evidence_ref,
        stamp_ns=stamp_ns,
    )
    if not unseen and low > support_z + 2.0 * resolution + 1e-9:
        return TargetRegionFit(
            None, TargetRefusal.NOT_ON_SUPPORT, n, valid, half, box=region, z_span=span
        )
    return TargetRegionFit(
        region, None, n, valid, half, z_span=span, support_z=support_z, support_unseen=unseen
    )


def target_region_from_masks(
    masks: Sequence[NDArray[np.bool_]],
    depth_m: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    *,
    resolution: float,
    nested_fraction: float = 0.9,
    **fit_kwargs: Any,  # noqa: ANN401  # reason: forwarded verbatim to target_region_from_mask
) -> TargetRegionFit:
    """``target_region_from_mask`` over a segmenter's candidate masks, HZ-0115-6 kept whole.

    SAM returns nested candidates (subpart, part, whole). Each is fitted; the first one
    that fits, in the segmenter's order, is the answer, except:

    - **A body refused ``not_on_support`` vetoes every larger candidate holding it.** A
      candidate refused ``NOT_ON_SUPPORT`` whose cloud spans at least two voxels of
      height is a body standing on something; a larger candidate holding at least
      ``nested_fraction`` of its pixels and reaching the support adds what it stands on
      (an item on a riser of its footprint: the item mask is refused, the item + riser
      mask fits), and a region pinned to the support would exempt that lower body. Such a
      candidate is refused ``STACKED``. A top face alone (a can's lid: under two voxels of
      height) has no body to stand on anything and vetoes nothing, so the whole can still
      fits. ponytail: a subpart with its own height (a can's upper half) also vetoes the
      whole object — less exemption, never more; the Safety-WG owns relaxing it.
    - **A body standing above the measured support stands on its own lowest point.**
      When no candidate fits, the first body refused ``NOT_ON_SUPPORT`` (in the
      segmenter's order) that holds no other such body is re-fitted with the support
      unseen (``support_z=None``): its region reaches down to its own lowest kept point
      and never to the support, so whatever it stands on — a bin's floor, a riser —
      stays outside the exemption, and the veto above still refuses every larger
      candidate holding it.
    - **A candidate set refused whole reports its most severe refusal**
      (``_REFUSAL_SEVERITY``), never the last one tried: a contradiction is not read as a
      lost view because a smaller candidate came last.

    Args:
        masks: The candidates on the depth raster, in the segmenter's order.
        depth_m: ``(H, W)`` metric depth.
        intrinsics: Intrinsics at ``(H, W)``.
        t_base_from_cam: ``(4, 4)`` optical-frame → base-frame transform.
        resolution: The voxel lattice's cell edge.
        nested_fraction: Fraction of a refused body's pixels a larger candidate must hold to
            stand on it. *Calibration point.*
        **fit_kwargs: The rest of ``target_region_from_mask``'s keyword arguments.

    Returns:
        The chosen ``TargetRegionFit``; ``TOO_FEW_POINTS`` when ``masks`` is empty.
    """
    fits = [
        target_region_from_mask(
            mask, depth_m, intrinsics, t_base_from_cam, resolution=resolution, **fit_kwargs
        )
        for mask in masks
    ]
    bodies = [
        mask
        for mask, fit in zip(masks, fits, strict=True)
        if fit.refusal is TargetRefusal.NOT_ON_SUPPORT and fit.z_span >= 2.0 * resolution
    ]
    checked: list[TargetRegionFit] = []
    for mask, fit in zip(masks, fits, strict=True):
        size = np.count_nonzero(mask)
        stacked = any(
            np.count_nonzero(body) < size
            and np.count_nonzero(body & mask) >= nested_fraction * np.count_nonzero(body)
            for body in bodies
        )
        if fit.region is not None and not stacked:
            return fit
        checked.append(
            replace(fit, region=None, refusal=TargetRefusal.STACKED)
            if fit.region is not None
            else fit
        )
    if not checked:
        return TargetRegionFit(None, TargetRefusal.TOO_FEW_POINTS, 0, 0.0, (0.0, 0.0, 0.0))
    for mask in bodies:
        holds_body = any(
            other is not mask
            and np.count_nonzero(other) < np.count_nonzero(mask)
            and np.count_nonzero(other & mask) >= nested_fraction * np.count_nonzero(other)
            for other in bodies
        )
        if holds_body:
            continue
        own = target_region_from_mask(
            mask,
            depth_m,
            intrinsics,
            t_base_from_cam,
            resolution=resolution,
            **{**fit_kwargs, "support_z": None},
        )
        if own.region is not None:
            return own
    return min(checked, key=lambda f: _REFUSAL_SEVERITY.index(cast("TargetRefusal", f.refusal)))


def region_covers_occupied(
    grid: VoxelLattice, region: PlaceRegion, *, min_fraction: float = 0.5
) -> tuple[int, bool]:
    """Whether the map agrees there is something in ``region``.

    A target seen from above shows the kernel at least its top surface: one layer
    of occupied cells over the region's footprint. ``ok`` requires the occupied
    cells with centres inside the region to number at least ``min_fraction`` of
    that footprint (``4·hx·hy / resolution²``). *Calibration point.*

    Args:
        grid: The published lattice.
        region: The fitted region, in ``grid.frame_id``.
        min_fraction: Fraction of the footprint's cell count required.

    Returns:
        ``(count_inside, ok)``.

    Raises:
        ROSConfigError: If ``region.frame_id`` is not the lattice frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> g = VoxelLattice(
        ...     "base",
        ...     (0.0, 0.0, 0.0),
        ...     (0.0, 0.0, 0.0, 1.0),
        ...     0.1,
        ...     (2, 2, 1),
        ...     np.ones(4, dtype=np.uint8),
        ... )
        >>> r = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.1, 0.1, 0.05),
        ...     pose=Pose6D(xyz=(0.1, 0.1, 0.05), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> region_covers_occupied(g, r)
        (4, True)
    """
    _check_frame(grid, region)
    count = int(_in_region(grid.occupied_centers(), region).sum())
    hx, hy, _ = region.half_extents
    footprint_cells = 4.0 * hx * hy / grid.resolution**2
    return count, count >= min_fraction * footprint_cells


def track_region(
    previous: PlaceRegion,
    current: PlaceRegion,
    *,
    max_centroid_shift_m: float,
    extents_tol_m: float,
) -> bool:
    """Accept a re-fit as the same target: same frame, barely moved, same size.

    Horizontal half-extents are compared sorted, because the footprint's PCA may
    swap its two axes between frames on a near-square object.

    Args:
        previous: The last validated region.
        current: The fresh fit.
        max_centroid_shift_m: Largest accepted centre displacement (the design's
            one voxel).
        extents_tol_m: Largest accepted change of any half-extent.

    Returns:
        ``True`` when ``current`` may replace ``previous``.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> def box(x):
        ...     return PlaceRegion(
        ...         frame_id="base",
        ...         half_extents=(0.05, 0.04, 0.06),
        ...         pose=Pose6D(xyz=(x, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     )
        >>> track_region(box(0.3), box(0.31), max_centroid_shift_m=0.02, extents_tol_m=0.01)
        True
        >>> track_region(box(0.3), box(0.4), max_centroid_shift_m=0.02, extents_tol_m=0.01)
        False
    """
    if previous.frame_id != current.frame_id:
        return False
    shift = float(np.linalg.norm(np.subtract(current.pose.xyz, previous.pose.xyz)))
    a = (*sorted(previous.half_extents[:2]), previous.half_extents[2])
    b = (*sorted(current.half_extents[:2]), current.half_extents[2])
    size_ok = all(abs(p - q) <= extents_tol_m for p, q in zip(a, b, strict=True))
    return shift <= max_centroid_shift_m + 1e-9 and size_ok


def region_within(inner: PlaceRegion, outer: PlaceRegion, *, tol_m: float) -> bool:
    """Whether every corner of ``inner`` lies inside ``outer`` grown by ``tol_m``.

    A re-fit made while the gripper occludes part of the target shrinks and
    shifts inside the region held for it; one that reaches outside it describes
    volume the held region never vouched for.

    Args:
        inner: The fresh fit.
        outer: The held region.
        tol_m: Per-face growth of ``outer``, metres (one voxel in the leg).

    Returns:
        ``True`` when ``inner`` is contained; ``False`` across frames.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> def box(x, h):
        ...     return PlaceRegion(
        ...         frame_id="base",
        ...         half_extents=(h, h, h),
        ...         pose=Pose6D(xyz=(x, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     )
        >>> region_within(box(0.32, 0.02), box(0.3, 0.05), tol_m=0.0)
        True
        >>> region_within(box(0.36, 0.02), box(0.3, 0.05), tol_m=0.02)
        False
    """
    if inner.frame_id != outer.frame_id:
        return False
    t = homogeneous_from_quat_xyz(inner.pose.xyz, inner.pose.quat_xyzw)
    signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = (signs * np.asarray(inner.half_extents)) @ t[:3, :3].T + t[:3, 3]
    grown = outer.model_copy(update={"half_extents": tuple(h + tol_m for h in outer.half_extents)})
    return bool(_in_region(corners, grown).all())


def cell_closed_region(
    region: PlaceRegion, grid: VoxelLattice, *, max_half_extent_m: float
) -> tuple[PlaceRegion, bool]:
    """``region`` grown so every lattice cell it intersects has its centre inside — never down.

    The kernel exempts a world voxel only when the cell's *centre* lies in the grasp
    region (``grasp_target_exempts`` in ``cpp/openral_safety_kernel/src/collision.cpp``).
    A fit tight to the target's measured surface leaves the target's own boundary cells
    — the surface inside them, their centres up to half a cell outside the box — never
    exempt, and a finger hull swept over the target's top edge stops on them. A cell
    meets the box iff its centre lies in the box ⊕ that cell, which the box grown per
    local axis ``i`` by ``r/2 · Σ_j |(Rᵀ_box R_grid)_ij|`` contains: ``r/2 · (|cos θ| +
    |sin θ|) ≤ r/√2`` horizontally for a yaw ``θ`` against a z-up lattice, ``r/2``
    vertically. Grown horizontally on both sides and **up only**: the bottom stays where
    the fit put it, so the support layer under it stays non-exempt (ADR-0115,
    HZ-0115-6). Voxel-consistent: inside one cell the map cannot tell another body from
    the target, so exempting the cells the target touches exempts nothing the map could
    separate from it.

    Args:
        region: A gravity-aligned (yaw-only) fit in ``grid.frame_id``.
        grid: The lattice the region is vouched against.
        max_half_extent_m: Ceiling on each grown half-extent (the declaration's cap);
            a clamped vertical half-extent keeps the bottom fixed.

    Returns:
        ``(closed, clamped)`` — ``clamped`` when any half-extent hit the ceiling.

    Raises:
        ROSConfigError: On a frame mismatch or a region that is not yaw-only.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> origin, identity = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        >>> g = VoxelLattice("base", origin, identity, 0.02, (1, 1, 1), np.zeros(1, np.uint8))
        >>> r = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.05, 0.04, 0.03),
        ...     pose=Pose6D(xyz=(0.3, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> closed, clamped = cell_closed_region(r, g, max_half_extent_m=0.2)
        >>> [round(v, 3) for v in closed.half_extents], round(closed.pose.xyz[2], 3), clamped
        ([0.06, 0.05, 0.035], 0.105, False)
    """
    _check_frame(grid, region)
    rot = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)[:3, :3]
    if abs(rot[2, 2] - 1.0) > _UNIT_QUAT_TOL:
        raise ROSConfigError("cell_closed_region: the region must be gravity-aligned (yaw only).")
    grow = 0.5 * grid.resolution * np.abs(rot.T @ grid.rotation()).sum(axis=1)
    hx, hy, hz = region.half_extents
    wanted = (hx + float(grow[0]), hy + float(grow[1]), hz + float(grow[2]) / 2.0)
    half = tuple(min(h, max_half_extent_m) for h in wanted)
    x, y, z = region.pose.xyz
    closed = region.model_copy(
        update={
            "half_extents": half,
            # Bottom fixed: the top alone rises by what the vertical half-extent gained.
            "pose": region.pose.model_copy(update={"xyz": (x, y, z - hz + half[2])}),
        }
    )
    return closed, half != wanted


def lowered_to_support(region: PlaceRegion, support_z: float) -> PlaceRegion:
    """``region`` with its lower face moved down onto ``support_z``, the rest unchanged.

    The grasp target's held region stands one voxel above the support it was measured
    on (``target_region_from_mask``), so the target's own bottom layer lies outside it.
    The vision-attached payload is lowered onto the support top: once lifted, that layer
    rides under the payload's lower face, and outside the payload it is an obstacle the
    octomap bridge never clears (Isaac i56/i57: a stop on the carried can's own bottom
    layer the moment the support witness retired). The support's cells stay outside the
    box (their tops at ``support_z``). Only a lower face above ``support_z`` moves; the
    region's local z is up for the yaw-only fits the leg makes.

    Args:
        region: The held region.
        support_z: The measured support top, in ``region.frame_id``.

    Returns:
        The lowered region (same frame, yaw, stamp and provenance), or ``region`` when its
        lower face is not above ``support_z``.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> r = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.05, 0.04, 0.03),
        ...     pose=Pose6D(xyz=(0.3, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> g = lowered_to_support(r, 0.055)
        >>> round(g.half_extents[2], 4), round(g.pose.xyz[2] - g.half_extents[2], 4)
        (0.0375, 0.055)
        >>> lowered_to_support(r, 0.08) is r
        True
    """
    hx, hy, hz = region.half_extents
    x, y, z = region.pose.xyz
    drop = (z - hz) - support_z
    if not (math.isfinite(drop) and drop > 0.0):
        return region
    return region.model_copy(
        update={
            "half_extents": (hx, hy, hz + drop / 2.0),
            "pose": region.pose.model_copy(update={"xyz": (x, y, z - drop / 2.0)}),
        }
    )


def margin_grown_region(region: PlaceRegion, margin_m: float, *, down: bool) -> PlaceRegion:
    """``region`` bloated by ``margin_m`` on its four sides and top, and its bottom when ``down``.

    The grasp-target margin (design note §2.1 "Grasp-target margin", HZ-0115-32): before
    the handover the kernel exempts, for the declared finger links, the held region grown
    on every face — ``down=True``, so cells up to ``margin_m`` below the region's lower
    face (the support under the target among them) are exempt too; the attached payload
    is grown on the sides and the top only (``down=False``), never into the support.
    Growth is along the region's own axes (its local z is up for the yaw-only fits the
    leg makes); ``margin_m = 0`` returns ``region`` unchanged.

    Args:
        region: The held region.
        margin_m: Per-face growth, metres, ``>= 0``.
        down: Whether the lower face moves down by ``margin_m`` too.

    Returns:
        The grown region (same frame, yaw, stamp and provenance).

    Raises:
        ROSConfigError: On a negative or non-finite margin.

    Example:
        >>> from openral_core import PlaceRegion, Pose6D
        >>> r = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.05, 0.04, 0.03),
        ...     pose=Pose6D(xyz=(0.3, 0.0, 0.1), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> g = margin_grown_region(r, 0.02, down=False)
        >>> [round(v, 3) for v in g.half_extents], round(g.pose.xyz[2] - g.half_extents[2], 3)
        ([0.07, 0.06, 0.04], 0.07)
        >>> g = margin_grown_region(r, 0.02, down=True)
        >>> [round(v, 3) for v in g.half_extents], round(g.pose.xyz[2] - g.half_extents[2], 3)
        ([0.07, 0.06, 0.05], 0.05)
    """
    if not (math.isfinite(margin_m) and margin_m >= 0.0):
        raise ROSConfigError(f"margin_grown_region: margin {margin_m!r} must be finite and >= 0.")
    if margin_m == 0.0:
        return region
    hx, hy, hz = region.half_extents
    # Down too: both z faces move, the centre stays. Up only: the top moves, so the
    # centre rises by half the margin along the region's own z.
    rise = 0.0 if down else margin_m / 2.0
    rot = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)[:3, :3]
    centre = np.asarray(region.pose.xyz, dtype=np.float64) + rot[:, 2] * rise
    return region.model_copy(
        update={
            "half_extents": (hx + margin_m, hy + margin_m, hz + margin_m - rise),
            "pose": region.pose.model_copy(update={"xyz": tuple(float(v) for v in centre)}),
        }
    )


def occupied_touching_outside(
    grid: VoxelLattice, region: PlaceRegion, *, support_z: float
) -> NDArray[np.float64]:
    """Occupied cells above the support, outside ``region``, touching a cell inside it.

    The target as the map sees it is the 26-connected component of the occupied cells
    above the support (centres more than one voxel above ``support_z``, the layer
    ``target_seed_from_voxels`` drops too) holding the cells whose centres lie in
    ``region``. That component lies wholly in ``region`` exactly when no cell of it
    outside ``region`` is 26-adjacent to one inside — any path out of the region
    crosses that ring — so this looks one cell around the region, never the whole map,
    and needs no bound on how far the component reaches. Empty means the region holds
    every cell of the target the map has; anything else is a fit that saw part of it
    (the far side occluded by the hand) or something touching it the map cannot
    separate from it.

    Args:
        grid: The published lattice.
        region: The region the kernel gets (``cell_closed_region`` of the fit), in
            ``grid.frame_id``.
        support_z: The measured support top the fit stands on.

    Returns:
        ``(N, 3)`` centres of those outside cells in ``grid.frame_id``.

    Raises:
        ROSConfigError: If ``region.frame_id`` is not the lattice frame.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(4 * 1 * 3, dtype=np.uint8)
        >>> occ[[4, 5, 6]] = 1  # a 3-cell bar at k=1, i=0..2
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (4, 1, 3), occ)
        >>> def box(hx):
        ...     return PlaceRegion(
        ...         frame_id="base",
        ...         half_extents=(hx, 0.05, 0.05),
        ...         pose=Pose6D(xyz=(hx, 0.05, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     )
        >>> occupied_touching_outside(g, box(0.1), support_z=0.0).round(3).tolist()
        [[0.25, 0.05, 0.15]]
        >>> len(occupied_touching_outside(g, box(0.15), support_z=0.0))
        0
    """
    _check_frame(grid, region)
    centers = grid.occupied_centers()
    above = centers[centers[:, 2] > support_z + grid.resolution]
    # Only cells within two of the region can be inside it or touch one inside it.
    near = region.model_copy(
        update={"half_extents": tuple(h + 2.0 * grid.resolution for h in region.half_extents)}
    )
    above = above[_in_region(above, near)]
    inside = _in_region(above, region)
    cells = [tuple(c) for c in _ijk(grid, above).tolist()]
    touched = {
        (i + di, j + dj, k + dk)
        for (i, j, k), is_in in zip(cells, inside.tolist(), strict=True)
        if is_in
        for di, dj, dk in _NEIGHBOURS_26
    }
    outside = np.array(
        [not is_in and c in touched for c, is_in in zip(cells, inside.tolist(), strict=True)],
        dtype=bool,
    )
    return above[outside] if len(above) else above


def map_completed_region(
    grid: VoxelLattice,
    fit: PlaceRegion,
    column: PlaceRegion,
    *,
    support_z: float,
    max_half_extent_m: float,
    max_volume_m3: float,
) -> tuple[PlaceRegion | None, int, str]:
    """``fit`` grown to the target's whole map component, or why it cannot be.

    The camera confirms which object the target is (``fit``, its visible part); the map
    holds the rest, seen from earlier views (the far side the hovering hand hides from
    the head camera, Isaac i45). The component is the 26-connected set of occupied cells
    more than one voxel above ``support_z`` (``occupied_touching_outside``'s filter),
    seeded from those whose centres lie in ``cell_closed_region(fit)`` and flooded inside
    ``column``. The result is the smallest box in the fit's yaw holding ``fit`` and every
    component cell's centre, its bottom the fit's own (never down into the support): its
    ``cell_closed_region`` — what the kernel gets — then holds every component cell whole.

    Refused (``None``, the reason) when a component cell has a 26-neighbour whose centre
    lies outside ``column`` (the component touches the search edge: the leg cannot vouch
    the rest is the target), when the box exceeds ``max_half_extent_m`` or
    ``max_volume_m3``, or when its cell closure holds an occupied cell above the support
    that is not the component's (another body inside the box's corners). A body within
    one voxel of the target is 26-connected to it and merges into the component: the map
    cannot separate the two (a Safety-WG residual, HZ-0115-30).

    Args:
        grid: The published lattice.
        fit: The camera's gravity-aligned (yaw-only) fit, in ``grid.frame_id``.
        column: The search column the leg vouches for (``search_column``).
        support_z: The measured support top the fit stands on.
        max_half_extent_m: Per-axis cap on the completed box.
        max_volume_m3: Volume cap on the completed box.

    Returns:
        ``(completed, added, reason)`` — ``added`` the component cells beyond the fit's
        cell closure (taken from the map alone); ``reason`` empty iff ``completed`` is set.

    Raises:
        ROSConfigError: On a frame mismatch or a tilted ``fit``.

    Example:
        >>> import numpy as np
        >>> from openral_core import PlaceRegion, Pose6D
        >>> occ = np.zeros(6 * 3 * 3, dtype=np.uint8)
        >>> occ[[18 + 6 + i for i in range(1, 5)]] = 1  # a 4-cell bar at j=1, k=1, i=1..4
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (6, 3, 3), occ)
        >>> def box(x, hx, hy, hz):
        ...     return PlaceRegion(
        ...         frame_id="base",
        ...         half_extents=(hx, hy, hz),
        ...         pose=Pose6D(xyz=(x, 0.15, 0.15), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     )
        >>> fit = box(0.19, 0.09, 0.05, 0.05)  # x 0.1-0.28: the camera saw two of the 4 cells
        >>> done, added, _ = map_completed_region(
        ...     g,
        ...     fit,
        ...     box(0.3, 0.3, 0.15, 0.15),
        ...     support_z=0.0,
        ...     max_half_extent_m=0.2,
        ...     max_volume_m3=0.03,
        ... )
        >>> [round(v, 3) for v in done.half_extents], round(done.pose.xyz[0], 3), added
        ([0.175, 0.05, 0.05], 0.275, 2)
        >>> map_completed_region(  # a column x 0.1-0.5: the bar's end cells touch its edge
        ...     g,
        ...     fit,
        ...     box(0.3, 0.2, 0.15, 0.15),
        ...     support_z=0.0,
        ...     max_half_extent_m=0.2,
        ...     max_volume_m3=0.03,
        ... )[2]
        "the target's map component (4 cells) touches the search column's edge"
    """
    _check_frame(grid, fit)
    _check_frame(grid, column)
    seed_box, _ = cell_closed_region(fit, grid, max_half_extent_m=max_half_extent_m)

    def grown(box: PlaceRegion) -> PlaceRegion:
        return box.model_copy(
            update={"half_extents": tuple(h + 2.0 * grid.resolution for h in box.half_extents)}
        )

    centers = grid.occupied_centers()
    everywhere = centers[centers[:, 2] > support_z + grid.resolution]
    # Every cell the flood can reach lies within two of the seed box or the column.
    above = everywhere[
        _in_region(everywhere, grown(seed_box)) | _in_region(everywhere, grown(column))
    ]
    in_seed = _in_region(above, seed_box)
    cells = [(int(i), int(j), int(k)) for i, j, k in _ijk(grid, above).tolist()]
    seeds = {c for c, ok in zip(cells, in_seed.tolist(), strict=True) if ok}
    allowed = in_seed | _in_region(above, column)
    component = _component_from(
        {c for c, ok in zip(cells, allowed.tolist(), strict=True) if ok}, seeds
    )
    comp = np.asarray(component, dtype=np.int64).reshape(-1, 3)
    added = len(component) - len(seeds)
    around = (comp[:, None, :] + np.asarray(_NEIGHBOURS_26, dtype=np.int64)[None]).reshape(-1, 3)
    if not _in_region(_cell_centres(grid, around), column).all():
        return (
            None,
            added,
            f"the target's map component ({len(component)} cells) touches the search column's edge",
        )
    t = homogeneous_from_quat_xyz(fit.pose.xyz, fit.pose.quat_xyzw)
    local = (_cell_centres(grid, comp) - t[:3, 3]) @ t[:3, :3]
    half = np.asarray(fit.half_extents, dtype=np.float64)
    lo = np.minimum(-half, local.min(axis=0, initial=np.inf))
    hi = np.maximum(half, local.max(axis=0, initial=-np.inf))
    lo[2] = -half[2]  # the bottom stays the fit's: never down into the support
    centre = t[:3, 3] + t[:3, :3] @ ((lo + hi) / 2.0)
    completed = fit.model_copy(
        update={
            "half_extents": tuple(float(v) for v in (hi - lo) / 2.0),
            "pose": fit.pose.model_copy(update={"xyz": tuple(float(v) for v in centre)}),
        }
    )
    if max(completed.half_extents) > max_half_extent_m or completed.volume_m3() > max_volume_m3:
        return (
            None,
            added,
            f"the target's map component ({len(component)} cells) completes to half_extents "
            f"{tuple(round(v, 3) for v in completed.half_extents)}, over the "
            f"{max_half_extent_m} m / {max_volume_m3} m^3 caps",
        )
    closed, _ = cell_closed_region(completed, grid, max_half_extent_m=max_half_extent_m)
    members = set(component)
    # Over the closure's own extent, every occupied cell above the support: in a yaw other
    # than the column's its corners reach past the cells pre-filtered above.
    foreign = [
        c for c in _cell_set(grid, everywhere[_in_region(everywhere, closed)]) if c not in members
    ]
    if foreign:
        return (
            None,
            added,
            f"{len(foreign)} occupied cell(s) of another body inside the completed region",
        )
    return completed, added, ""
