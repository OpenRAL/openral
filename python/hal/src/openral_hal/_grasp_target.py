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
   so it is never taken as the support (HZ-01xx-6).
   ``target_seed_from_voxels`` clusters the cells above that plane,
   take the largest cluster's top-centre. Refuses on too few cells or on two
   comparably-sized clusters (HZ-01xx-2, wrong object: guessing between two
   candidates is exactly the mis-declaration the hazard row names).
3. ``project_point`` — that seed into the camera image as SAM 2.1's positive
   point prompt.
4. ``target_region_from_mask`` — eroded mask → masked depth → base-frame cloud
   → gravity-aligned oriented box whose lower face sits **one voxel above the
   support plane, never below it** (HZ-01xx-6: a region reaching into the
   support surface would exempt the very cells that stop the fingers from
   being driven into the table).
5. ``region_covers_occupied`` — the fitted region must contain occupied cells
   the kernel itself sees, or it describes nothing the map agrees with.
6. ``track_region`` — the 2-5 Hz re-prompt gate: a re-fit is only accepted as
   the same target if it barely moved. ``region_within`` tells a re-fit that
   shrank inside the held region (the approaching hand occluding part of the
   target) from one that reaches outside it (the target moved).

Every threshold here is a **calibration point** (CLAUDE.md §1.2): the caps are
the design note's Safety-WG placeholders (half-extent ≤ 0.20 m, volume ≤
0.03 m³), the rest are chosen conservatively, not measured.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from openral_core import PlaceRegion, Pose6D
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw

from openral_hal._vision_attachment_evidence import backproject_masked_depth

if TYPE_CHECKING:  # pragma: no cover — typing only
    from openral_core import IntrinsicsPinhole

__all__ = [
    "TargetRefusal",
    "TargetRegionFit",
    "TargetSeed",
    "VoxelLattice",
    "occupied_centers_in_box",
    "project_point",
    "region_covers_occupied",
    "region_within",
    "support_top_from_voxels",
    "target_region_from_mask",
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
    HALF_EXTENT_CAP = "half_extent_cap"
    VOLUME_CAP = "volume_cap"


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
        """``(N, 3)`` centres of every occupied cell in ``frame_id``."""
        sx, sy, _ = self.size
        idx = np.flatnonzero(self.occupancy)
        ijk = np.stack((idx % sx, (idx // sx) % sy, idx // (sx * sy)), axis=1)
        local = (ijk.astype(np.float64) + 0.5) * self.resolution
        return np.asarray(local @ self.rotation().T + np.asarray(self.origin), dtype=np.float64)


@dataclass(frozen=True)
class TargetSeed:
    """Result of ``target_seed_from_voxels``.

    Attributes:
        point: Top-centre of the selected cluster in the lattice frame, or ``None``.
        refusal: Why no seed was produced; ``None`` when ``point`` is set.
        cluster_sizes: Cell counts of every cluster found, descending (for the trace).
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
    """

    region: PlaceRegion | None
    refusal: TargetRefusal | None
    point_count: int
    depth_valid_fraction: float
    half_extents: tuple[float, float, float]


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
) -> float | None:
    """Measure the surface the target stands on: the top face of that layer.

    Scanned **top-down** over the lattice z-layers of ``centers`` (a column under
    the search box). For each candidate layer ``k``, the target is the connected
    component of the cells *above* ``k`` that holds the cell horizontally nearest
    ``near_xy``; ``k`` is the support when it holds at least ``min_cells``
    **top-surface** cells (no occupied cell directly above) in the ring around
    that component's footprint — farther than one cell from it (the target's own
    side faces) and at most ``probe_margin_m`` from it. A surface extends
    laterally beyond what stands on it; the target's own layers do not, so a
    dense target top is never taken for the support, and the probe scales with
    the target's footprint instead of a fixed square. The ring skips the band
    right at the footprint only by one cell, so a head camera's occlusion shadow
    behind the target merely thins it.

    Ceiling: a target whose lower part flares more than one cell past its upper
    footprint (a pyramid, a handle) can stop the scan inside the target; the
    region then stands higher than the support — less exemption, never more.

    Args:
        grid: The lattice ``centers`` came from; must be z-up (yaw-only).
        centers: ``(N, 3)`` occupied centres in ``grid.frame_id``.
        near_xy: Where the target is expected (the search box centre), in
            ``grid.frame_id``.
        min_cells: Fewest ring cells a layer needs to be the support.
            *Calibration point.*
        probe_margin_m: Outer reach of the ring from the target footprint, metres.
            *Calibration point.*

    Returns:
        The support's top face z in ``grid.frame_id``, or ``None`` when no layer
        qualifies — there is no measured support to stand a region on.

    Raises:
        ROSConfigError: On a tilted lattice (its cells form no horizontal layers).

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
    local = (centers - np.asarray(grid.origin)) @ grid.rotation()
    ijk = np.floor(local / grid.resolution).astype(np.int64)
    occupied = {(int(i), int(j), int(k)) for i, j, k in ijk.tolist()}
    dist2 = np.sum((centers[:, :2] - np.asarray(near_xy)) ** 2, axis=1)
    reach = max(2, math.ceil(probe_margin_m / grid.resolution - 1e-9))
    for k in sorted({c[2] for c in occupied}, reverse=True):
        above = ijk[:, 2] > k
        if not above.any():
            continue
        start = tuple(int(v) for v in ijk[np.flatnonzero(above)[np.argmin(dist2[above])]])
        footprint = {(i, j) for i, j, _ in _component_from(occupied, start, min_k=k + 1)}
        inner = _dilate(footprint, 1)
        ring = _dilate(footprint, reach) - inner
        surface = sum(
            1
            for i, j, kk in occupied
            if kk == k and (i, j) in ring and (i, j, k + 1) not in occupied
        )
        if surface >= min_cells:
            return float(grid.origin[2] + (k + 1) * grid.resolution)
    return None


def _dilate(cells: set[tuple[int, int]], r: int) -> set[tuple[int, int]]:
    """Chebyshev dilation of a 2-D cell set by ``r`` cells."""
    steps = range(-r, r + 1)
    return {(i + di, j + dj) for i, j in cells for di in steps for dj in steps}


def _component_from(
    occupied: set[tuple[int, int, int]], start: tuple[int, ...], *, min_k: int
) -> list[tuple[int, int, int]]:
    """The 26-connected component of ``start`` among the occupied cells with ``k >= min_k``."""
    first = (start[0], start[1], start[2])
    seen = {first}
    queue = deque([first])
    while queue:
        ci, cj, ck = queue.popleft()
        for di, dj, dk in _NEIGHBOURS_26:
            n = (ci + di, cj + dj, ck + dk)
            if n[2] >= min_k and n in occupied and n not in seen:
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
    support_z: float,
    min_cells: int,
    ambiguity_ratio: float = 0.5,
) -> TargetSeed:
    """Pick the one object in the search box and return its top-centre.

    Cells whose centre is within one voxel of the support plane are the support
    itself and are dropped; the rest are labelled into 26-connected components on
    the lattice. The largest component is the target unless the runner-up holds at
    least ``ambiguity_ratio`` of its cells — then there is no single object to name
    and the seed is refused (HZ-01xx-2).

    Args:
        grid: The lattice ``centers`` came from (its pose turns centres back into
            integer cells for the connectivity labelling).
        centers: ``(N, 3)`` occupied centres in ``grid.frame_id``, e.g. from
            ``occupied_centers_in_box``.
        support_z: Height of the support plane in ``grid.frame_id`` (z up).
        min_cells: Fewest cells the selected cluster may have. *Calibration point.*
        ambiguity_ratio: Runner-up / largest size at or above which the choice is
            ambiguous. *Calibration point.*

    Returns:
        A ``TargetSeed``; ``point`` is ``(x_mean, y_mean, z_top)`` where ``z_top`` is
        the top face of the highest cell, ``bottom_z`` the bottom face of the lowest.

    Example:
        >>> import numpy as np
        >>> occ = np.zeros(27, dtype=np.uint8)
        >>> occ[[4, 13, 22]] = 1  # the column at i=1, j=1, k=0..2
        >>> g = VoxelLattice("base", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.1, (3, 3, 3), occ)
        >>> s = target_seed_from_voxels(g, g.occupied_centers(), support_z=0.0, min_cells=2)
        >>> [round(v, 3) for v in s.point], s.cluster_sizes
        ([0.15, 0.15, 0.3], (2,))
    """
    above = centers[centers[:, 2] > support_z + grid.resolution]
    local = (above - np.asarray(grid.origin)) @ grid.rotation()
    ijk = np.floor(local / grid.resolution).astype(np.int64)
    comps = sorted(_components(ijk), key=len, reverse=True)
    sizes = tuple(len(c) for c in comps)
    if not comps or sizes[0] < min_cells:
        return TargetSeed(None, TargetRefusal.TOO_FEW_CELLS, sizes)
    if len(sizes) > 1 and sizes[1] >= ambiguity_ratio * sizes[0]:
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


def target_region_from_mask(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    *,
    support_z: float,
    resolution: float,
    frame_id: str,
    evidence_ref: str,
    stamp_ns: int = 0,
    erode_px: int = 2,
    extrinsic_error_m: float = 0.01,
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

    **The lower face never goes below ``support_z + resolution`` (HZ-01xx-6).** The
    cells holding the support surface have centres up to half a voxel above the
    plane; one full voxel keeps every one of them outside the region, so the kernel
    still stops the fingers at the table under the target. Padding is applied to the
    four sides and the top only.

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
        support_z: Support-plane height in the base frame.
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
        >>> fit = target_region_from_mask(
        ...     m,
        ...     np.full((64, 64), 0.9),
        ...     k,
        ...     t,
        ...     support_z=0.0,
        ...     resolution=0.02,
        ...     frame_id="base",
        ...     evidence_ref="doc",
        ...     min_points=10,
        ... )
        >>> fit.refusal is None, round(fit.region.pose.xyz[2] - fit.region.half_extents[2], 3)
        (True, 0.02)
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
    top = float(np.percentile(pts[:, 2], 100.0 - trim_percentile))

    pad = math.sqrt(3.0) * resolution / 2.0 + extrinsic_error_m
    bottom = support_z + resolution
    z_hi = top + pad
    half_xy = (hi_uv - lo_uv) / 2.0 + pad
    half = (float(half_xy[0]), float(half_xy[1]), (z_hi - bottom) / 2.0)
    if top <= bottom:
        return TargetRegionFit(None, TargetRefusal.NO_HEIGHT_ABOVE_SUPPORT, n, valid, half)
    if max(half) > max_half_extent_m:
        return TargetRegionFit(None, TargetRefusal.HALF_EXTENT_CAP, n, valid, half)
    if 8.0 * half[0] * half[1] * half[2] > max_volume_m3:
        return TargetRegionFit(None, TargetRefusal.VOLUME_CAP, n, valid, half)

    centre_xy = xy_mean + rot2 @ ((lo_uv + hi_uv) / 2.0)
    centre = (float(centre_xy[0]), float(centre_xy[1]), (z_hi + bottom) / 2.0)
    region = PlaceRegion(
        frame_id=frame_id,
        pose=Pose6D(xyz=centre, quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame_id),
        half_extents=half,
        evidence_ref=evidence_ref,
        stamp_ns=stamp_ns,
    )
    return TargetRegionFit(region, None, n, valid, half)


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
