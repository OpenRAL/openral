"""Containing fit — the #266 rule: a set that contains the solid, never a hull of samples.

Issue #418, design "The tracker algorithm" step 3. One camera mask is lifted to

    C = cone(dilate(S)) ∩ {z ≥ support_z} ∩ {z_opt ≥ z_front - ε}  union  prism(P, support_z)

* ``S`` — the mask minus robot pixels, plus the robot-adjacent uncertain band, plus an
  *occluder band* (``w_occ_px``) where a nearer surface (a neighbour, the robot) cuts the
  mask boundary; ``dilate`` adds ``k_px`` of mask and extrinsic error.
* ``P`` — the mask's own depth points; ``prism`` extrudes them down to the support.
* The cone is bounded because every camera in the three scenes looks down: each ray meets
  the support plane.

``C`` contains the solid under four assumptions, each logged per fit (``assumptions``):
A1 the silhouette is complete up to ``k_px`` and the bands (a mask touching the image
border is refused: the cone misses what the frame cut off); A2 the object rests on the
measured support (a surface point under it is refused: truncating there would cut the
solid); A3 depth is right within ``eps_m``; A4 no hidden part lies outside the prism and
more than ``w_occ_px`` behind an occluder (weakest in packed totes; logged as the
occluder pixel count) -- and no part hidden behind the robot is nearer the camera than the
object's own measured front (the front cut ignores robot pixels; logged as ``a5``).

The polytope is represented exactly by a few vertices: in optical-depth parametrisation the
front cut is the plane ``z_opt = z_front`` for every ray and the support is a plane, so
``cone(Q) ∩ slab`` for a convex image polygon ``Q`` is the hull of ``Q``'s corner rays at the
two planes (``z_front`` is lowered to the nearest support hit so the two cuts never cross
inside the cone). ``Q`` is the silhouette's 16-direction outer polygon, ``P``'s footprint the
same: 4 x 16 = 64 vertices, the kernel's per-hull cap, with no hull solver and no inward step.
The set is cut at a *ceiling*, the highest point any silhouette ray's first hit allows (no ray
meets the solid nearer than its first hit, and the rays look down): an oblique view's global
front cut otherwise boxes a low object far above itself.

Pure numpy, no ROS, no clock. Every default is a *calibration point* (CLAUDE.md §1.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionPrimitive,
    BoxShape,
    GraspDeclaration,
    IntrinsicsPinhole,
    PlaceRegion,
    Pose6D,
)
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw
from openral_core.perception_geometry import backproject_masked_depth
from openral_core.schemas import DOP_AXES, TightCollisionGeometry
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ROBOT_PIXEL",
    "UNCERTAIN_PIXEL",
    "ContainingFit",
    "FitParams",
    "containing_fit",
    "project_box",
    "visible_fraction",
]

#: ``robot_filter`` mask values (design "Robot filtering"): 0 free, 128 uncertain, 255 robot.
ROBOT_PIXEL = 255
UNCERTAIN_PIXEL = 128

_DOP_DIRS = 16  # 4 x 16 = 64 vertices = TrackedObject.MAX_HULL_VERTICES
_YAW_STEP_DEG = 1.0


class FitParams(BaseModel):
    """Containing-fit knobs. Every one is a *calibration point*.

    Attributes:
        k_px: Mask error plus extrinsic error, pixels.
        w_occ_px: Occluder band width where a nearer surface cuts the mask, pixels.
        eps_m: Depth error (A3) and the support tolerance (A2), metres.
        occluder_tol_m: How much nearer a neighbour pixel must be to count as an occluder.
        min_points: Fewest valid depth points a mask needs.
        min_depth_valid: Fewest fraction of mask pixels with usable depth (A3).
        min_depth_m: Depth readings at or below this are invalid.
        max_depth_m: Depth readings at or above this are invalid.

    Example:
        >>> FitParams().k_px
        2
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_px: int = Field(default=2, ge=1)
    w_occ_px: int = Field(default=4, ge=0)
    eps_m: float = Field(default=0.005, ge=0.0)
    occluder_tol_m: float = Field(default=0.01, gt=0.0)
    min_points: int = Field(default=20, ge=4)
    min_depth_valid: float = Field(default=0.5, ge=0.0, le=1.0)
    min_depth_m: float = Field(default=0.05, ge=0.0)
    max_depth_m: float = Field(default=5.0, gt=0.0)


@dataclass(frozen=True)
class ContainingFit:
    """Result of ``containing_fit``.

    Attributes:
        box: Gravity-aligned box containing ``C`` (+½ cell on the sides and top; the bottom
            is the support), or ``None`` when refused.
        tight: One box primitive with the 64-vertex hull of ``C`` as its stage-2 refinement,
            posed in ``box.frame_id``; ``None`` when refused.
        surface: ``(N, 3)`` mask depth points ``P`` in the base frame (the cover's ``E``).
        support_z: The support the fit was truncated at.
        assumptions: A1-A4 evidence for the trace (``a1_border_clear``, ``a2_gap_m``,
            ``a3_depth_valid``, ``a4_occluder_px``, ``a4_robot_beyond_band_px``: robot pixels
            past the occluder band where the robot touches the mask (``provisional`` when any),
            ``a5_robot_band_px``: silhouette pixels
            on or beside the robot, whose depth never sets the front cut).
        refusal: Empty when fitted, else why not.
        provisional: The robot touches the mask and reaches past the occluder band, so the
            fit may under-contain what it hides (A4 not held): good for exemptions and
            identity clearing only (``TrackedObject.provisional``), never a payload.
    """

    box: PlaceRegion | None
    tight: AttachedCollisionPrimitive | None
    surface: NDArray[np.float64]
    support_z: float
    assumptions: dict[str, float | bool] = field(default_factory=dict)
    refusal: str = ""
    provisional: bool = False


def _dilate(mask: NDArray[np.bool_], px: int) -> NDArray[np.bool_]:
    """8-neighbour binary dilation, ``px`` passes."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(px):
        p = np.pad(m, 1)
        m = np.zeros_like(m)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                m |= p[dr : dr + m.shape[0], dc : dc + m.shape[1]]
    return m


def _neighbour_max(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per pixel, the largest of its 8 neighbours (``-inf`` padded)."""
    p = np.pad(values, 1, constant_values=-np.inf)
    h, w = values.shape
    out = np.full(values.shape, -np.inf)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if (dr, dc) != (1, 1):
                out = np.maximum(out, p[dr : dr + h, dc : dc + w])
    return out


def _rect(box: tuple[float, float, float, float], shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """The pixels of ``box`` (``x0, y0, x1, y1``), clipped to ``shape``."""
    out = np.zeros(shape[:2], dtype=bool)
    c0, r0 = max(math.floor(box[0]), 0), max(math.floor(box[1]), 0)
    out[r0 : max(math.ceil(box[3]), 0), c0 : max(math.ceil(box[2]), 0)] = True
    return out


def _outer_polygon(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(16, 2)`` vertices of the 16-direction polygon that contains ``points``.

    Each side is a supporting line of the point set, so the polygon contains its hull and
    every side touches it; consecutive sides meet at the vertices.
    """
    th = np.arange(_DOP_DIRS) * (2.0 * math.pi / _DOP_DIRS)
    c, s = np.cos(th), np.sin(th)
    h = (points @ np.stack((c, s))).max(axis=0)
    c1, s1, h1 = np.roll(c, -1), np.roll(s, -1), np.roll(h, -1)
    det = math.sin(2.0 * math.pi / _DOP_DIRS)
    return np.stack(((h * s1 - h1 * s) / det, (c * h1 - c1 * h) / det), axis=1)


def _refused(
    why: str, surface: NDArray[np.float64], support_z: float, notes: dict[str, float | bool]
) -> ContainingFit:
    return ContainingFit(None, None, surface, support_z, notes, why)


def containing_fit(  # noqa: PLR0915  # reason: one named refusal per assumption
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    robot_mask: NDArray[np.uint8] | None,
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    *,
    support_z: float,
    resolution: float,
    frame_id: str,
    stamp_ns: int,
    params: FitParams | None = None,
    detector_box: tuple[float, float, float, float] | None = None,
) -> ContainingFit:
    """Fit the set ``C`` that contains the object one mask shows (module docstring).

    Args:
        mask: ``(H, W)`` segmenter mask of one object.
        depth_m: ``(H, W)`` metric depth registered to ``mask`` (optical z).
        robot_mask: ``(H, W)`` ``robot_filter`` mask at the same stamp (0/128/255), or
            ``None`` for a frame with no robot in view.
        intrinsics: Pinhole intrinsics of the frame.
        t_base_from_cam: ``(4, 4)`` optical frame in the grid frame at the depth stamp.
        support_z: Measured support top under the object, grid frame.
        resolution: Grid cell edge (the box pads half of it).
        frame_id: The grid frame.
        stamp_ns: Depth stamp, carried on the box.
        params: Calibration (``FitParams()`` when ``None``).
        detector_box: The open-vocabulary detector's box (pixels, ``x0, y0, x1, y1``) the
            mask was prompted with, if any: its rectangle joins the silhouette (A1), so a
            mask that lost a rim the detector saw is still contained. Never a track's own
            projected box (that would only ever grow itself).

    Returns:
        A ``ContainingFit``; ``refusal`` names the failed gate.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole
        >>> k = IntrinsicsPinhole(width=64, height=48, fx=60.0, fy=60.0, cx=32.0, cy=24.0)
        >>> t = np.eye(4)
        >>> t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # looking straight down
        >>> t[2, 3] = 0.5
        >>> depth = np.full((48, 64), 0.5)
        >>> depth[20:28, 28:36] = 0.45  # a 5 cm block on the table
        >>> m = depth < 0.49
        >>> f = containing_fit(
        ...     m, depth, None, k, t, support_z=0.0, resolution=0.01, frame_id="b", stamp_ns=1
        ... )
        >>> f.refusal, f.box.pose.xyz[2] > 0.02
        ('', True)
    """
    p = params or FitParams()
    robot = np.zeros(mask.shape, dtype=bool) if robot_mask is None else robot_mask == ROBOT_PIXEL
    uncertain = (
        np.zeros(mask.shape, dtype=bool) if robot_mask is None else robot_mask == UNCERTAIN_PIXEL
    )
    s0 = np.asarray(mask, dtype=bool) & ~robot
    # ``P``, the object's measured points: the mask's pixels the filter is sure are not
    # the robot. Its uncertain band straddles the arm's edge and may hold arm returns; it
    # widens the cone below, never the prism or the front cut.
    sure = s0 & ~uncertain
    raw, valid_fraction = backproject_masked_depth(
        sure if sure.any() else s0,
        depth_m,
        intrinsics,
        min_depth_m=p.min_depth_m,
        max_depth_m=p.max_depth_m,
    )
    surface = raw @ t_base_from_cam[:3, :3].T + t_base_from_cam[:3, 3]
    notes: dict[str, float | bool] = {"a3_depth_valid": valid_fraction}
    if len(surface) < p.min_points:
        return _refused("too_few_points", surface, support_z, notes)
    border = bool(s0[0, :].any() or s0[-1, :].any() or s0[:, 0].any() or s0[:, -1].any())
    notes["a1_border_clear"] = not border
    notes["a2_gap_m"] = float(surface[:, 2].min() - support_z)
    if border:
        return _refused("a1_border_truncated", surface, support_z, notes)
    if notes["a2_gap_m"] < -p.eps_m:
        return _refused("a2_below_support", surface, support_z, notes)
    if valid_fraction < p.min_depth_valid:
        return _refused("a3_depth_invalid", surface, support_z, notes)

    d = np.asarray(depth_m, dtype=np.float64)
    usable = np.isfinite(d) & (d > p.min_depth_m) & (d < p.max_depth_m)
    ring = _dilate(s0, 1) & ~s0
    mask_depth = _neighbour_max(np.where(s0 & usable, d, -np.inf))
    nearer = ring & (robot | (usable & (d < mask_depth - p.occluder_tol_m)))
    notes["a4_occluder_px"] = float(nearer.sum())
    # The bands re-add a hidden part up to ``w_occ_px`` behind an occluder, so a robot
    # region that touches the mask (across the filter's uncertain band) and reaches past
    # that band may hide more of the object than the fit can contain — the arm over a
    # plate's near rim cut 2.7 cm off it. Such a fit is PROVISIONAL (#418 decision
    # 2026-10-17): an under-contained box may only shrink what is exempted or cleared, so
    # it serves K2 and identity clearing, never a payload, until a fit with the robot back
    # inside the band clears the flag.
    reach = p.k_px + 1 + p.w_occ_px
    touch = _dilate(s0, p.k_px + 1) & robot
    beyond = _dilate(s0, reach + 1) & ~_dilate(s0, reach) & robot
    notes["a4_robot_beyond_band_px"] = float(beyond.sum()) if touch.any() else 0.0
    provisional = notes["a4_robot_beyond_band_px"] > 0
    silhouette = s0 | (uncertain & _dilate(s0, p.k_px + 1)) | _dilate(nearer, p.w_occ_px)
    measured = sure
    if detector_box is not None:
        rect = _rect(detector_box, silhouette.shape)
        silhouette |= rect
        measured = sure | (rect & ~robot & ~uncertain)  # the lost rim's depth sets the front
    sil = _dilate(silhouette, p.k_px)

    rows, cols = np.nonzero(sil)
    corners = np.concatenate(
        [np.stack((cols + du, rows + dv), axis=1) for du in (0, 1) for dv in (0, 1)]
    ).astype(np.float64)
    q = _outer_polygon(corners)
    u = (q[:, 0] - intrinsics.cx) / intrinsics.fx
    v = (q[:, 1] - intrinsics.cy) / intrinsics.fy
    rays = np.stack((u, v, np.ones(len(q))), axis=1) @ t_base_from_cam[:3, :3].T
    origin = t_base_from_cam[:3, 3]
    if not (np.all(rays[:, 2] < 0.0) and origin[2] > support_z):
        return _refused("ray_misses_support", surface, support_z, notes)
    z_support = (support_z - origin[2]) / rays[:, 2]
    # The front cut is the object's own measured front: the mask's pixels the robot filter
    # is sure are not the robot. Never a robot or robot-adjacent (uncertain) pixel's depth:
    # the arm hovers well above what it works over and cannot hold the object, and the
    # filter's uncertain band straddles the arm's edge, so either would lift the cone to
    # the arm (a head camera over a tote: +27 cm). The bands keep widening the cone
    # laterally; a hidden part nearer than that front behind the robot is A4 (logged, a5).
    front = measured & usable
    if not front.any():
        front = sil & usable & ~robot
    notes["a5_robot_band_px"] = float((sil & (robot | uncertain)).sum())
    z_front = min(float(d[front].min()) - p.eps_m, float(z_support.min()))
    footprint = _outer_polygon(surface[:, :2])
    top = float(surface[:, 2].max())
    # Every ray of the silhouette meets the solid no nearer than its first hit (an opaque
    # solid nearer would be that hit) and no nearer than the front cut; the rays look down,
    # so the solid stands no higher than the highest of those points. The global front cut
    # alone lifts an oblique view's far-rim rays well above the object (Spark banana R325,
    # 2026-10-23: a 5.6 cm bowl boxed 15 cm tall, its margin keeping the carried banana's
    # cells as the bowl's). A pixel without usable depth is bounded by the front cut only.
    sv, su = np.nonzero(sil)
    pu = (su + 0.5 - intrinsics.cx) / intrinsics.fx
    pv = (sv + 0.5 - intrinsics.cy) / intrinsics.fy
    pix = np.stack((pu, pv, np.ones(len(su))), axis=1)
    ray_z = pix @ t_base_from_cam[2, :3]
    hit = np.where(usable[sv, su], np.maximum(d[sv, su] - p.eps_m, z_front), z_front)
    ceiling = max(float((origin[2] + hit * ray_z).max()), top)
    notes["ceiling_z"] = ceiling
    verts = _clipped_vertices(origin, rays, z_front, z_support, footprint, support_z, top, ceiling)
    fit = _boxed(verts, surface, support_z, resolution, frame_id, stamp_ns, notes)
    return replace(fit, provisional=provisional) if provisional and fit.box is not None else fit


def _clipped_vertices(
    origin: NDArray[np.float64],
    rays: NDArray[np.float64],
    z_front: float,
    z_support: NDArray[np.float64],
    footprint: NDArray[np.float64],
    support_z: float,
    top: float,
    ceiling: float,
) -> NDArray[np.float64]:
    """64 vertices whose hull holds the cone cut by ``z <= ceiling`` and the prism.

    The layout ``multi_view`` reads: the corner rays at their nearest point, the same rays
    at the support, the prism's floor, its lid. Cut, a ray's nearest point is the deeper of
    the front cut and the ceiling; a face whose two rays start on different cuts holds the
    point where the cuts cross, so the lid is raised to the ceiling and widened over those
    points (a taller, wider prism only grows the set).
    """
    rz = rays[:, 2]
    front = origin + z_front * rays
    ground = origin + z_support[:, None] * rays
    floor = np.column_stack((footprint, np.full(len(footprint), support_z)))
    lid = np.column_stack((footprint, np.full(len(footprint), top)))
    if float(front[:, 2].max()) <= ceiling:
        return np.concatenate((front, ground, floor, lid))
    t_cap = (ceiling - origin[2]) / rz  # optical depth where each ray meets the ceiling
    want = (ceiling - origin[2]) / z_front  # the ray z slope that meets both cuts at once
    cross = []
    for i in range(len(rays)):
        j = (i + 1) % len(rays)
        if (t_cap[i] > z_front) == (t_cap[j] > z_front) or rz[i] == rz[j]:
            continue
        a = (want - rz[j]) / (rz[i] - rz[j])
        if 0.0 < a < 1.0:
            cross.append(origin[:2] + z_front * (a * rays[i, :2] + (1.0 - a) * rays[j, :2]))
    if cross:
        rim = _outer_polygon(np.concatenate((footprint, np.asarray(cross))))
        lid = np.column_stack((rim, np.full(len(rim), ceiling)))
    start = origin + np.maximum(z_front, t_cap)[:, None] * rays
    return np.concatenate((start, ground, floor, lid))


def _boxed(
    verts: NDArray[np.float64],
    surface: NDArray[np.float64],
    support_z: float,
    resolution: float,
    frame_id: str,
    stamp_ns: int,
    notes: dict[str, float | bool],
) -> ContainingFit:
    """Gravity-aligned box (min-area yaw) + 64-vertex tight hull of ``verts``."""
    yaws = np.deg2rad(np.arange(0.0, 90.0, _YAW_STEP_DEG))
    c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    xs = verts[:, 0] * c + verts[:, 1] * s
    ys = -verts[:, 0] * s + verts[:, 1] * c
    best = int(np.argmin(np.ptp(xs, axis=1) * np.ptp(ys, axis=1)))
    yaw = float(yaws[best])
    pad = 0.5 * resolution
    lx, hx, ly, hy = xs[best].min(), xs[best].max(), ys[best].min(), ys[best].max()
    cx_r, cy_r = 0.5 * (lx + hx), 0.5 * (ly + hy)
    zlo, zhi = support_z, float(verts[:, 2].max()) + pad
    half = (0.5 * (hx - lx) + pad, 0.5 * (hy - ly) + pad, 0.5 * (zhi - zlo))
    centre = (
        float(cx_r * math.cos(yaw) - cy_r * math.sin(yaw)),
        float(cx_r * math.sin(yaw) + cy_r * math.cos(yaw)),
        0.5 * (zlo + zhi),
    )
    cap = GraspDeclaration.MAX_HALF_EXTENT_M
    if max(half) > cap:
        return _refused("half_extent_cap", surface, support_z, notes)
    if 8.0 * half[0] * half[1] * half[2] > GraspDeclaration.MAX_VOLUME_M3:
        return _refused("volume_cap", surface, support_z, notes)
    pose = Pose6D(xyz=centre, quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame_id)
    box = PlaceRegion(
        frame_id=frame_id,
        pose=pose,
        half_extents=(float(half[0]), float(half[1]), float(half[2])),
        evidence_ref="object_tracker:silhouette_cone",
        stamp_ns=stamp_ns,
    )
    t = homogeneous_from_quat_xyz(centre, pose.quat_xyzw)
    local = (verts - t[:3, 3]) @ t[:3, :3]
    proj = local @ np.asarray(DOP_AXES, dtype=np.float64).T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    lo[:3] = np.maximum(lo[:3], -np.asarray(half))  # float slack only; the box holds them
    hi[:3] = np.minimum(hi[:3], np.asarray(half))
    tight = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=box.half_extents),
        pose_in_object=pose,
        tight_geometry=TightCollisionGeometry(
            dop_lo_m=tuple(float(v) for v in lo),
            dop_hi_m=tuple(float(v) for v in hi),
            hull_vertices_m=tuple((float(a), float(b), float(c_)) for a, b, c_ in local),
        ),
    )
    return ContainingFit(box, tight, surface, support_z, notes)


def project_box(
    box: PlaceRegion, intrinsics: IntrinsicsPinhole, t_base_from_cam: NDArray[np.float64]
) -> tuple[tuple[int, int, int, int], float] | None:
    """``((c0, r0, c1, r1), near_z)``: ``box``'s clipped image rectangle and nearest optical z.

    ``None`` when a corner is behind the camera or the rectangle misses the image.
    """
    tb = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    signs = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    corners = (signs * np.asarray(box.half_extents)) @ tb[:3, :3].T + tb[:3, 3]
    t_cam = np.linalg.inv(t_base_from_cam)
    cam = corners @ t_cam[:3, :3].T + t_cam[:3, 3]
    if np.any(cam[:, 2] <= 0.0):
        return None
    u = intrinsics.fx * cam[:, 0] / cam[:, 2] + intrinsics.cx
    v = intrinsics.fy * cam[:, 1] / cam[:, 2] + intrinsics.cy
    c0, c1 = max(int(u.min()), 0), min(math.ceil(u.max()), intrinsics.width)
    r0, r1 = max(int(v.min()), 0), min(math.ceil(v.max()), intrinsics.height)
    if c0 >= c1 or r0 >= r1:
        return None
    return (c0, r0, c1, r1), float(cam[:, 2].min())


def visible_fraction(
    box: PlaceRegion,
    depth_m: NDArray[np.float64],
    robot_mask: NDArray[np.uint8] | None,
    intrinsics: IntrinsicsPinhole,
    t_base_from_cam: NDArray[np.float64],
    *,
    tol_m: float = 0.01,
) -> float | None:
    """Z-buffered fraction of ``box``'s image footprint the camera sees, or ``None`` out of view.

    A pixel of the box's projected rectangle is hidden when the robot covers it (robot or
    uncertain), it has no depth, or the measured surface there is nearer than the box's
    nearest corner (an occluder in front: a neighbour, the arm).

    Ceiling: an occluder standing between the box's nearest corner and its surface counts
    as visible; the per-pixel box depth would catch it. Fine for the ``θ_vis`` gate.

    Example:
        >>> import numpy as np
        >>> from openral_core import IntrinsicsPinhole, PlaceRegion, Pose6D
        >>> k = IntrinsicsPinhole(width=64, height=48, fx=60.0, fy=60.0, cx=32.0, cy=24.0)
        >>> t = np.eye(4)
        >>> t[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        >>> t[2, 3] = 0.5
        >>> b = PlaceRegion(
        ...     frame_id="b",
        ...     half_extents=(0.03, 0.03, 0.03),
        ...     pose=Pose6D(xyz=(0.0, 0.0, 0.03), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ... )
        >>> visible_fraction(b, np.full((48, 64), 0.47), None, k, t)
        1.0
        >>> visible_fraction(b, np.full((48, 64), 0.2), None, k, t)
        0.0
    """
    projected = project_box(box, intrinsics, t_base_from_cam)
    if projected is None:
        return None
    (c0, r0, c1, r1), near_z = projected
    d = np.asarray(depth_m, dtype=np.float64)[r0:r1, c0:c1]
    seen = np.isfinite(d) & (d > 0.0) & (d >= near_z - tol_m)
    if robot_mask is not None:
        seen &= robot_mask[r0:r1, c0:c1] < UNCERTAIN_PIXEL
    return float(seen.mean())
