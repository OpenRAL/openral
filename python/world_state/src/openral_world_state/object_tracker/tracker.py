"""``ObjectTracker`` — ground · fit · confirm · track · schedule, one depth frame at a time.

Issue #418, design "The tracker algorithm" steps 0-7 and Addendum G. Pure: the lifecycle
node feeds each depth frame (with its robot mask, intrinsics and ``T_base←cam`` at the depth
stamp), the newest grid with its ``source_stamp``, and the answers to the requests it issued
on that frame; it gets back at most one new request, the escalations and the whole
``TrackedObjectSet``. No ROS, no clock read: lifetimes run on the stamps fed in, the GPU rate
on the ``wall_s`` fed in.

A frame missing its tf or its robot mask is used **only for ageing** (step 0): no fit, no
visibility change, no request.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from statistics import median
from typing import NamedTuple

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionPrimitive,
    FitKind,
    GraspDeclaration,
    IntrinsicsPinhole,
    ObjectQuery,
    ObjectQueryObservation,
    ObjectQueryStatus,
    ObjectRole,
    PlaceRegion,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
)
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_core.object_primitives import instance_prompt
from openral_core.perception_geometry import (
    VoxelLattice,
    backproject_masked_depth,
    project_point,
    support_top_from_voxels,
)
from openral_core.tracking_thresholds import load_tracking_thresholds
from pydantic import BaseModel, ConfigDict, Field

from openral_world_state.object_lift import aabb_iou_3d
from openral_world_state.object_tracker.catalog import CatalogDeclarations
from openral_world_state.object_tracker.cover import (
    Cell,
    CoverVerdict,
    _inside,
    box_cells,
    cells_behind_robot,
    cover_need,
    cover_verdict,
    hand_touches,
    own_cover,
    static_cell_keys,
    surface_cells,
    uncovered_cells,
)
from openral_world_state.object_tracker.fit import (
    ROBOT_PIXEL,
    ContainingFit,
    FitParams,
    containing_fit,
    project_box,
    visible_fraction,
)
from openral_world_state.object_tracker.grounding import QueryBook, QueryEscalation, StandingQuery
from openral_world_state.object_tracker.hidden_side import grow_hidden_side
from openral_world_state.object_tracker.multi_view import intersect_views
from openral_world_state.object_tracker.scheduler import (
    PRIORITY_BOUND,
    PRIORITY_DISAGREE,
    PRIORITY_QUERY,
    PRIORITY_REFRESH,
    Scheduler,
    TrackerRequest,
)
from openral_world_state.object_tracker.surface_model import ObjectSurfaceModels
from openral_world_state.object_tracker.tracks import TrackStore, age_tracks, associate

__all__ = [
    "CameraFrame",
    "ObjectTracker",
    "TrackerOutput",
    "TrackerParams",
    "TrackerResult",
    "duplicate_of",
]

log = structlog.get_logger(__name__)

_T = load_tracking_thresholds().tracker

# A refit "moved" its track when this many measured surface points lie more than one cell
# outside the stored box (max of the two): the box provably no longer contains the object.
_MOVED_MIN_POINTS = _T.moved_min_points
_MOVED_MIN_FRACTION = _T.moved_min_fraction
#: Fraction of a refused refit's measured surface that must lie in the track's box (grown by a
#: cell) for the refit to re-date it: the object was measured inside the box it is kept with.
#: *Calibration point.*
_REDATE_INSIDE_FRACTION = _T.redate_inside_fraction
#: The same for a refit the growth gate refused: its mask spilled onto a touching neighbour,
#: so part of its surface is that neighbour's. *Calibration point* (restock 2026-10-31: 16-32 %
#: of a spilled mask's points lay outside the kept box).
_REDATE_SPILL_FRACTION = _T.redate_spill_fraction
#: Refused clear growth fits that must agree before one is admitted (``_admit_growth``).
#: *Calibration point.*
_GROWTH_VOTES = _T.growth_votes
#: How close (extent, height, centre) the agreeing fits must be. *Calibration point.*
_GROWTH_AGREE_M = _T.growth_agree_m
#: Source seconds the agreeing fits must fall in. *Calibration point*: the full fit of a
#: partly-masked carton recurs every 2-3 s between accepted partial fits (Spark replay
#: bkA_on_rs_11 2026-11-03: three over about 5 s, none admitted at a 5 s window).
_GROWTH_WINDOW_NS = round(_T.growth_window_s * 1_000_000_000)


#: Source time a CONTEXT track's refresh, or a CONTEXT query's locate, counts as served
#: later than it was: the target and destination go first. *Calibration point.*
_CONTEXT_DEFER_NS = round(_T.context_defer_s * 1e9)
#: Margin a released match's box is grown by as its query's seed: down onto the support it
#: fell to, and around where it rolled. *Calibration point.*
PRIOR_MARGIN_M = _T.prior_margin_m
#: A new fit whose box overlaps a live track's box by at least this 3D IoU is that track seen
#: under another label, not a new object: never minted. *Calibration point* (restock
#: 2026-10-31: a generic "carton" query's fit over the target at IoU 0.8-0.9; two inflated
#: boxes of neighbouring 6.5 cm cartons 7 cm apart sit near 0.25).
#: Visible-fraction margin an OCCLUDED track needs to return to CONFIRMED.
_VIS_HYSTERESIS = _T.vis_hysteresis
_DUPLICATE_MINT_IOU = _T.duplicate_mint_iou


def _aabb(box: PlaceRegion) -> tuple[float, float, float, float, float, float]:
    """The axis-aligned extent of ``box``'s eight corners in its frame."""
    t = homogeneous_from_quat_xyz(box.pose.xyz, box.pose.quat_xyzw)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = (signs * np.asarray(box.half_extents)) @ t[:3, :3].T + t[:3, 3]
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return (*(float(v) for v in lo), *(float(v) for v in hi))  # type: ignore[return-value]  # reason: six floats by construction


def _aabb_overlap(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
) -> bool:
    """Whether two ``_aabb`` extents share volume."""
    return all(min(a[k + 3], b[k + 3]) > max(a[k], b[k]) for k in range(3))


def duplicate_of(box: PlaceRegion, tracks: Sequence[TrackedObject]) -> tuple[str, float] | None:
    """The live track whose box ``box`` overlaps at ``_DUPLICATE_MINT_IOU`` or more, and the IoU.

    Association is same-label only (a banana let go into the bowl is no refit of the bowl), so
    a generic query ("carton") fitted over a named one's object minted a second track on it; it
    never confirmed (``cover_already_tracked``) but lingered, and a late bind revived it beside
    the target (restock bench_fix_08, bound ambiguous). A banana in a bowl overlaps the bowl's
    box at a small IoU, the same object at a large one. LOST and RELEASED tracks hold no place.

    Example:
        >>> from openral_core import Pose6D
        >>> def b(x: float) -> PlaceRegion:
        ...     p = Pose6D(xyz=(x, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="base")
        ...     return PlaceRegion(frame_id="base", pose=p, half_extents=(0.06, 0.06, 0.06))
        >>> duplicate_of(b(0.0), []) is None
        True
    """
    new = _aabb(box)
    best: tuple[str, float] | None = None
    for o in tracks:
        if o.status in (TrackStatus.LOST, TrackStatus.RELEASED):
            continue
        iou = aabb_iou_3d(new, _aabb(o.box))
        if iou >= _DUPLICATE_MINT_IOU and (best is None or iou > best[1]):
            best = (o.object_id, iou)
    return best


def _grown(box: PlaceRegion, margin_m: float) -> PlaceRegion:
    """``box`` with every half-extent grown by ``margin_m``."""
    return box.model_copy(update={"half_extents": tuple(h + margin_m for h in box.half_extents)})


# -- vision + voxel fusion (exp/vision-voxel-fusion; Safety-WG review owed) ------------------
# Identity comes from the instance mask propagator's SAM2 video masks, geometry from the voxel
# map. The numbers live in tracking_thresholds.yaml (``vision``), each with its episode.
_V = load_tracking_thresholds().vision
_VISION_MIN_CELLS = _V.min_cells
_VISION_NEED_FRACTION = _V.need_fraction
_VISION_MAX_EXTENT = _V.max_extent_factor
_RECENT_S = _T.recent_s
_PLAUSIBLE_EXTENT = _T.plausible_extent_factor


def _hidden_cells(
    grid: VoxelLattice, cells: frozenset[Cell], frame: CameraFrame | None
) -> frozenset[Cell]:
    """The expected ``cells`` the robot hides from ``frame`` (none without its robot mask)."""
    if frame is None or frame.robot_mask is None or frame.t_base_from_cam is None:
        return frozenset()
    return cells_behind_robot(
        grid, cells, frame.depth_m, frame.robot_mask, frame.intrinsics, frame.t_base_from_cam
    )


def covers_destination(box: PlaceRegion, dest: PlaceRegion) -> bool:
    """Whether a TARGET box is the destination's own body, not something lying on or in it.

    They overlap by half the larger footprint (comparable sizes: a tiny partial "plate" fitted
    on an orange is not what the orange covers, yf2 ep3) and it stands 80 % of the destination's
    height. The plate minted as an "orange" and the bowl as a "banana" (yf1 ep4, yfb1 ep0/ep1)
    did both; an orange on the plate (8 % of its footprint) or a banana in the bowl (60 % of the
    bowl's height) does not.

    Example:
        >>> from openral_core import Pose6D
        >>> def b(hx, hz, z=0.0):
        ...     p = Pose6D(xyz=(0.0, 0.0, z), quat_xyzw=(0, 0, 0, 1), frame_id="b")
        ...     return PlaceRegion(frame_id="b", pose=p, half_extents=(hx, hx, hz))
        >>> plate = b(0.105, 0.025)
        >>> covers_destination(b(0.108, 0.031), plate), covers_destination(b(0.03, 0.025), plate)
        (True, False)
        >>> bowl = b(0.085, 0.035)
        >>> covers_destination(b(0.08, 0.02, -0.01), bowl)  # a banana lying in it
        False
        >>> yawed = b(0.093, 0.06, 0.01).model_copy(
        ...     update={
        ...         "pose": Pose6D(
        ...             xyz=(-0.004, -0.011, 0.007), quat_xyzw=(0, 0, 0.552, 0.834), frame_id="b"
        ...         )
        ...     }
        ... )
        >>> covers_destination(yawed, b(0.085, 0.053))  # the n2_dr ep0 bowl-sized "banana"
        True
    """
    a, d = _aabb(box), _aabb(dest)
    ix = max(0.0, min(a[3], d[3]) - max(a[0], d[0]))
    iy = max(0.0, min(a[4], d[4]) - max(a[1], d[1]))
    # The boxes' own footprints, not their axis-aligned extents: a box yawed 67 deg has an extent
    # 1.7x its area, and a bowl-sized "banana" over the bowl read 0.48 < 0.5 (n2_dr ep0, the
    # negative test). The intersection stays the extents' (never smaller: fail-closed).
    foot = 4.0 * max(
        box.half_extents[0] * box.half_extents[1], dest.half_extents[0] * dest.half_extents[1]
    )
    return foot > 0 and ix * iy >= 0.5 * foot and (a[5] - a[2]) >= 0.8 * (d[5] - d[2])


def _box_between(
    like: PlaceRegion, lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> PlaceRegion:
    """An axis-aligned box from ``lo`` to ``hi`` in ``like``'s frame."""
    return like.model_copy(
        update={
            "pose": like.pose.model_copy(
                update={
                    "xyz": tuple(float(v) for v in 0.5 * (lo + hi)),
                    "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
                }
            ),
            "half_extents": tuple(float(v) for v in 0.5 * (hi - lo)),
        }
    )


def _grow_to_size(
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    size: float | None,
    hands: Sequence[PlaceRegion],
) -> None:
    """Widen each horizontal axis of ``lo``/``hi`` (in place) to at least ``size``.

    Vision labels only what the camera sees, so a box of labelled cells shrinks to the visible
    part while the hand hides the rest: a 4.6 cm box on a 6.5 cm orange, and the gripper stopped
    on its hidden side 2.3 cm outside (yw3 ep7; the banana's hidden tip, ywb2 ep3). The missing
    width goes toward the nearest hand box, where the occluded part is; with no hand, to both
    sides. ``size`` is the object's measured clear size (its smaller extent).

    Example:
        >>> lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.04, 0.06, 0.05])
        >>> _grow_to_size(lo, hi, 0.06, [])
        >>> [round(float(v), 3) for v in (lo[0], hi[0], lo[1], hi[1])]
        [-0.01, 0.05, 0.0, 0.06]
    """
    if size is None or size <= 0:
        return
    centre = 0.5 * (lo + hi)
    hand = None
    if hands:
        hc = [np.asarray(h.pose.xyz, dtype=np.float64) for h in hands]
        hand = min(hc, key=lambda c: float(np.hypot(*(c[:2] - centre[:2]))))
    for k in (0, 1):
        short = size - (hi[k] - lo[k])
        if short <= 0:
            continue
        if hand is None:
            lo[k] -= 0.5 * short
            hi[k] += 0.5 * short
        elif hand[k] >= centre[k]:
            hi[k] += short
        else:
            lo[k] -= short


def _placed_box(box: PlaceRegion, dest: PlaceRegion) -> PlaceRegion | None:
    """Where a target let go over ``dest`` comes to rest, or ``None`` if it was not over it.

    Its box at release, twice as wide (it rolls) and reaching down to ``dest``'s bottom (it
    falls). Spark orange k9: the plate split each time an orange landed on it; accepting any
    second body in the plate's box instead also accepted a foreign object nobody placed.
    ponytail: one roll radius; an object rolling further splits the destination (fail-closed).

    Example:
        >>> from openral_core import Pose6D
        >>> def b(x, z, h):
        ...     p = Pose6D(xyz=(x, 0.0, z), quat_xyzw=(0, 0, 0, 1), frame_id="b")
        ...     return PlaceRegion(frame_id="b", pose=p, half_extents=(h, h, h))
        >>> plate = b(0.0, 0.0, 0.1)
        >>> p = _placed_box(b(0.02, 0.2, 0.03), plate)
        >>> [round(v, 3) for v in (*p.pose.xyz, *p.half_extents)]
        [0.02, 0.0, 0.065, 0.06, 0.06, 0.165]
        >>> _placed_box(b(0.3, 0.2, 0.03), plate) is None
        True
    """
    a, d = _aabb(box), _aabb(dest)
    cx, cy = 0.5 * (a[0] + a[3]), 0.5 * (a[1] + a[4])
    if not (d[0] <= cx <= d[3] and d[1] <= cy <= d[4]):
        return None
    hx, hy = 0.5 * (a[3] - a[0]), 0.5 * (a[4] - a[1])
    lo = np.array([a[0] - hx, a[1] - hy, d[2]])
    hi = np.array([a[3] + hx, a[4] + hy, max(a[5], d[5])])
    return _box_between(dest, lo, hi)


def _vision_need(ref: int) -> int:
    """Labelled cells a vision box needs: ``_VISION_NEED_FRACTION`` of ``ref``, at least 4.

    Example:
        >>> _vision_need(59), _vision_need(0)
        (18, 4)
    """
    return max(_VISION_MIN_CELLS, math.ceil(_VISION_NEED_FRACTION * ref))


def _enclosing(a: PlaceRegion, b: PlaceRegion) -> PlaceRegion:
    """The smallest box in ``a``'s orientation that holds both ``a`` and ``b`` (same frame).

    Example:
        >>> from openral_core import Pose6D
        >>> def box(x, hx):
        ...     p = Pose6D(xyz=(x, 0.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="b")
        ...     return PlaceRegion(frame_id="b", pose=p, half_extents=(hx, 0.1, 0.1))
        >>> u = _enclosing(box(0.0, 0.1), box(0.3, 0.1))
        >>> [round(v, 3) for v in u.pose.xyz], [round(h, 3) for h in u.half_extents]
        ([0.15, 0.0, 0.0], [0.25, 0.1, 0.1])
    """
    ta = homogeneous_from_quat_xyz(a.pose.xyz, a.pose.quat_xyzw)
    tb = homogeneous_from_quat_xyz(b.pose.xyz, b.pose.quat_xyzw)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = np.vstack(
        [
            ta[:3, 3] + (signs * np.asarray(a.half_extents)) @ ta[:3, :3].T,
            tb[:3, 3] + (signs * np.asarray(b.half_extents)) @ tb[:3, :3].T,
        ]
    )
    local = (corners - ta[:3, 3]) @ ta[:3, :3]
    lo, hi = local.min(axis=0), local.max(axis=0)
    centre = ta[:3, 3] + ta[:3, :3] @ ((lo + hi) / 2)
    return a.model_copy(
        update={
            "pose": a.pose.model_copy(update={"xyz": tuple(float(v) for v in centre)}),
            "half_extents": tuple(float(h) for h in (hi - lo) / 2),
        }
    )


_NS = 1_000_000_000


def provisional_oversize(extent_m: float, prior_m: float, factor: float) -> bool:
    """Is a provisional refit's longer horizontal ``extent_m`` more than ``factor`` x the prior?

    Example:
        >>> provisional_oversize(0.334, 0.09, 2.0), provisional_oversize(0.12, 0.09, 2.0)
        (True, False)
    """
    return extent_m > factor * prior_m


class TrackerParams(BaseModel):
    """Tracker calibration. Every value is a *calibration point* (CLAUDE.md §1.2).

    Attributes:
        theta_vis: Visible fraction below which an object is OCCLUDED (and never fitted).
        locate_min_score: The detection floor the tracker asks its locator for. A box is
            only a prompt (SAM, the containing fit and map confirmation filter it), so it is
            lower than a detector's own (OmDet 0.30: restock cartons scored 0.15-0.29).
        continuous_locate_s: Source seconds between locates of a RESOLVED query (detection
            runs for the whole goal; ``QueryBook.due``).
        theta_vis_provisional: The same floor for a fit through the robot (PROVISIONAL:
            exemption-shrinking and payload-union geometry only), so a target the arm half
            hides at goal start still grounds (#418 PickOrange orange_3: 0.40 visible).
        provisional_oversize_factor: A PROVISIONAL refit of a known track whose longer horizontal
            extent exceeds this many times the largest extent a CLEAR same-label track (or the
            track's own clear history) has shown is refused and the previous box kept (#418
            r16po_on1: a 5.8 cm orange refit to 33 cm through the arm's occluder band).
            *Calibration point.*
        refit_growth_factor: Any refit (moved or not, clear or provisional) whose longer
            horizontal extent or height exceeds this many times the track's FIRST clear fit is
            refused and the box kept, unless it is a re-orientation (every sorted extent and
            the volume within this factor of the first fit's: a tipped carried carton). A mask
            that spills onto a neighbour or the support always has points outside the box, so
            it read as "moved" and replaced the box unchecked; each larger box then prompted a
            larger mask (restock 2026-10-30: a 6 x 6 cm carton's box went 14 -> 24 -> 39 cm
            with the arms out of view). *Calibration point*: 1.5 keeps a
            box turned 45 degrees in the hand (sqrt 2). A refused clear refit seen at the same
            larger size and the reference's height ``_GROWTH_VOTES`` times running, reaching no
            other track and holding one body, is admitted as the new reference once per track
            (``_admit_growth``): a first fit from a partial mask no longer holds the box small
            for good (#479 review, restock seeds 111/112).
        static_pad_cells: Cells a declared static collider is padded by for the one-body test
            (the camera sees a wall about a cell off its declared box; K7 pads the same).
            *Calibration point.*
        low_score_confirm_grids: Consecutive confirming grids (distinct stamps) a track minted
            from a prompt under ``LOW_PROMPT_SCORE`` needs before CONFIRMED (one otherwise;
            #418 r6po_dbg4 ghost trk:5, score 0.171, confirmed on its first grid).
            *Calibration point.*
        max_age_s: Source seconds without a confirming grid before a track is ineligible.
        lost_s: Source seconds without evidence before an ineligible track is LOST.
        keep_lost_s: Source seconds a LOST track is kept as a re-association prior.
        disagree_grids: Fresh grids refuting a track before it is LOST.
        segment_hz: Segmenter / detector requests per wall second.
        segment_budget_wall_s: Wall seconds before an unanswered request is abandoned.
        search_start_s: Unresolved source seconds before active search is requested.
        search_max_s: Unresolved source seconds before a query is EXHAUSTED.
        support_min_cells: ``support_top_from_voxels``'s ring cells.
        support_probe_m: ``support_top_from_voxels``'s ring reach.
        fit: Containing-fit knobs.

    Example:
        >>> TrackerParams().theta_vis
        0.5
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    theta_vis: float = Field(default=_T.theta_vis, gt=0.0, le=1.0)
    theta_vis_provisional: float = Field(default=_T.theta_vis_provisional, gt=0.0, le=1.0)
    continuous_locate_s: float = Field(default=_T.continuous_locate_s, gt=0.0)
    locate_min_score: float = Field(default=_T.locate_min_score, ge=0.0, le=1.0)
    low_score_confirm_grids: int = Field(default=_T.low_score_confirm_grids, ge=1)
    static_pad_cells: int = Field(default=_T.static_pad_cells, ge=0)
    provisional_oversize_factor: float = Field(default=_T.provisional_oversize_factor, ge=1.0)
    refit_growth_factor: float = Field(default=_T.refit_growth_factor, ge=1.0)
    max_age_s: float = Field(default=_T.max_age_s, gt=0.0)
    lost_s: float = Field(default=_T.lost_s, gt=0.0)
    keep_lost_s: float = Field(default=_T.keep_lost_s, gt=0.0)
    disagree_grids: int = Field(default=_T.disagree_grids, ge=1)
    segment_hz: float = Field(default=_T.segment_hz, gt=0.0)
    segment_budget_wall_s: float = Field(default=_T.segment_budget_wall_s, gt=0.0)
    search_start_s: float = Field(default=_T.search_start_s, ge=0.0)
    search_max_s: float = Field(default=_T.search_max_s, gt=0.0)
    support_min_cells: int = Field(default=_T.support_min_cells, ge=1)
    support_probe_m: float = Field(default=_T.support_probe_m, gt=0.0)
    fit: FitParams = FitParams()


@dataclass(frozen=True)
class CameraFrame:
    """One depth frame at its source stamp ``t_d``.

    Attributes:
        camera: Manifest sensor name.
        stamp_ns: Depth stamp (source clock).
        depth_m: ``(H, W)`` metric optical-z depth.
        intrinsics: Pinhole intrinsics of the frame.
        t_base_from_cam: Optical frame in the grid frame at ``stamp_ns``; ``None`` when tf
            had no pose at the stamp.
        robot_mask: ``robot_filter`` mask at ``stamp_ns`` (0/128/255); ``None`` when missing.
    """

    camera: str
    stamp_ns: int
    depth_m: NDArray[np.float64]
    intrinsics: IntrinsicsPinhole
    t_base_from_cam: NDArray[np.float64] | None
    robot_mask: NDArray[np.uint8] | None


class _Next(Enum):
    """``_apply_mask``'s verdict: try the next (smaller) candidate mask."""

    NEXT = auto()


class _Candidate(NamedTuple):
    """A candidate mask's fit, support height, surface centroid and map cells."""

    fit: ContainingFit
    support_z: float
    centroid: tuple[float, ...]
    cells: frozenset[Cell]


#: Source seconds between INFO reports of one ``(key, reason)`` refusal.
REFUSAL_LOG_PERIOD_S = _T.refusal_log_period_s
#: Source seconds within which fits of one track from different cameras are combined
#: (``intersect_views``): the object must not have moved between them. *Calibration point.*
VIEW_WINDOW_S = _T.view_window_s
#: A locate's crop: the projected search region grown by this fraction of its size on
#: every side (a seed or workspace box need not be tight in the image). *Calibration point.*
LOCATE_ROI_PAD = _T.locate_roi_pad
#: A detector box more than this fraction ROBOT pixels is re-prompted on what it hides.
ROBOT_BOX_FRACTION = _T.robot_box_fraction
#: Negative point prompts placed on the robot in such a box.
ROBOT_BOX_NEGATIVES = _T.robot_box_negatives
#: A prompt box scored under this grounds a track that is logged when it confirms.
LOW_PROMPT_SCORE = _T.low_prompt_score
#: How far under the voxel-measured support top the camera's own support returns may lie and
#: still read as ON the layer boundary (``ObjectTracker._support_seen``). Sim depth is exact;
#: a real camera's noise is larger. *Calibration point.*
_SUPPORT_SEEN_TOL_M = _T.support_seen_tol_m
#: Tracks whose boxes stand for a measured object: a body inside one is that object.
_SETTLED = frozenset({TrackStatus.CONFIRMED, TrackStatus.OCCLUDED})
# The roles a hand touches by design: the object it picks and the one it places into
# (2026-10-21 decision; successful baseline runs touch the bowl for 79-131 steps, and runs
# that treated it as an obstacle stopped on it). A CONTEXT neighbour the hand meets still
# fails closed.
_CONTACT_ROLES = frozenset({ObjectRole.TARGET, ObjectRole.DESTINATION})
# A contact re-anchor's ceiling, as a multiple of what the track filled in the map at its
# last plain pass: more occupied cells in the box than this is something besides the object
# (a neighbour, a hand leak). *Calibration point*: 33-52 cells were held in the box while
# the hand hid the banana (3-7 in view).
_REANCHOR_MAX_RATIO = _T.reanchor_max_ratio


@dataclass(frozen=True)
class TrackerResult:
    """The answer to an issued ``TrackerRequest`` (by ``request_id``).

    Attributes:
        request_id: The request it answers.
        mask: ``SegmentInView``'s mask (``None``: no object at the prompt).
        boxes_xyxy: ``LocateInView``'s detections, pixels.
        box_scores: Their detector scores (parallel; empty = unknown).
        failure_reason: Bounded service cause when the detection did not complete;
            empty means a completed query, including a valid no-match result.
        alternatives: Smaller qualifying candidates, largest first, each tried in turn
            when the one before it cannot be fitted or splits on the map (a packed
            carton's "part" candidate is several cartons).
    """

    request_id: int
    mask: NDArray[np.bool_] | None = None
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    alternatives: tuple[NDArray[np.bool_], ...] = ()
    box_scores: tuple[float, ...] = ()
    failure_reason: str = ""


@dataclass(frozen=True)
class TrackerOutput:
    """What one ``feed`` produces.

    Attributes:
        requests: At most one request to issue on the fed frame.
        tracked: The whole track table.
        escalations: New active-search escalations.
        bound_fits: Fits that associated with a BOUND track, for the in-hand update.
    """

    requests: tuple[TrackerRequest, ...]
    tracked: TrackedObjectSet
    escalations: tuple[QueryEscalation, ...]
    bound_fits: tuple[tuple[str, ContainingFit], ...]


class ObjectTracker:
    """The Layer-2 object tracker core (module docstring).

    Attributes:
        store: The track table; binding / release use its ``get`` / ``all`` / ``replace``.
        queries: The standing queries.

    Example:
        >>> from openral_core import ObjectRole
        >>> tr = ObjectTracker("base")
        >>> q = ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET)
        >>> tr.open_query("g", "q0", q, "obj:orange@0", stamp_ns=0)
        >>> tr.queries.get("g", "q0").status.value
        'unresolved'
    """

    def __init__(self, frame_id: str, params: TrackerParams | None = None) -> None:
        """Start with no tracks and no queries, in the grid frame ``frame_id``."""
        self.params = params or TrackerParams()
        # Explicitly enabled by deployment config; never consumed as safety geometry.
        self.surface_models: ObjectSurfaceModels | None = None
        self.catalog_declarations: CatalogDeclarations | None = None
        p = self.params
        self.store = TrackStore(frame_id)
        self.queries = QueryBook(
            search_start_s=p.search_start_s,
            search_max_s=p.search_max_s,
            continuous_s=p.continuous_locate_s,
        )
        self._sched = Scheduler(segment_hz=p.segment_hz, budget_wall_s=p.segment_budget_wall_s)
        #: Every refusal so far by ``(key, reason)``: ``<label>@<camera>`` for a fit,
        #: ``camera:<name>`` for a frame (the node publishes it; see ``_refuse``).
        self.refusal_counts: Counter[tuple[str, str]] = Counter()
        self._refusal_logged_ns: dict[tuple[str, str], int] = {}
        # Per track, per camera: the newest clear fit and its camera pose (multi-view).
        self._views: dict[str, dict[str, tuple[ContainingFit, NDArray[np.float64], int]]] = {}
        #: Where queried objects are expected (the grasp declaration's workspace box, set
        #: by the session); a locate is cropped to it when its query has no seed.
        self.search_region: PlaceRegion | None = None
        self._research_logged: dict[tuple[str, str], int] = {}
        # Tracks minted from a prompt box scored under LOW_PROMPT_SCORE, until they confirm.
        self._low_score: dict[str, float] = {}
        # Per low-score track: its run of consecutive confirming grids (last stamp, count).
        self._low_score_run: dict[str, tuple[int, int]] = {}
        # Per track: the largest smaller-horizontal extent of its clear (non-provisional) fits.
        self._clear_size: dict[str, float] = {}
        # Per track: the largest longer-horizontal extent of its clear fits (the oversize guard).
        self._clear_extent: dict[str, float] = {}
        # Per label, per track: its largest longer-horizontal clear extent. Kept when the track
        # retires (pro#51: a lone dropped object's own track is retired at release, and its
        # regrasp mint had no clear size left to be held to).
        self._label_clear_extent: dict[str, dict[str, float]] = {}
        # Per track: the longer horizontal extent of its first clear fit, else of its first fit,
        # and whether that fit was clear (the growth gate); its height and its full extents.
        # (extent, clear, height, full extents x/y/z)
        # Growth admission (#479 review): each track's recent refused clear growth fits
        # (stamp, (extent, height), centre xy), and whether one was admitted (once per track).
        self._growth_votes: dict[str, list[tuple[int, tuple[float, float], tuple[float, float]]]]
        self._growth_votes = {}
        self._growth_admitted: dict[str, bool] = {}
        self._first_extent: dict[str, tuple[float, bool, float, tuple[float, ...]]] = {}
        # Declared static geometry (cell centres, grid frame) and its keys on the newest lattice.
        self._static_centres: NDArray[np.float64] = np.zeros((0, 3))
        self._static_cache: tuple[tuple[object, ...], frozenset[Cell]] | None = None
        self._cells: dict[str, frozenset[Cell]] = {}
        self._centroid: dict[str, tuple[float, float, float]] = {}
        self._disagree: dict[str, int] = {}
        # Per track: the grid stamp its last `map_disagrees` counted (one count per grid).
        self._disagree_ns: dict[str, int] = {}
        # What each track fills in the map (``box_cells``) at its last plain pass: a contact
        # re-anchor's reference while the hand hides or pushes the object.
        self._map_cells: dict[str, int] = {}
        # Per track, the most cells a plain pass ever showed: a vision box's reference (a
        # hidden pass shows 2-7 of an orange's ~20).
        self._map_cells_max: dict[str, int] = {}
        # Per released TARGET: the track as let go, its reference cells and the release stamp,
        # or ``None`` once re-found or expired (``_vision_refind``).
        self._let_go: dict[str, tuple[TrackedObject, int, int] | None] = {}
        # Per TARGET track, where it rests on a destination (``_note_placed``): a body there
        # is what was placed, not a foreign object splitting the destination.
        self._placed: dict[str, PlaceRegion] = {}
        # Per BOUND track: (epoch, the box a partial in-hand fit is held to; `_measure_bound`).
        self._bound_anchor: dict[str, tuple[int, PlaceRegion]] = {}
        self._served: dict[str, int] = {}
        #: Per BOUND track, where its payload rides now (set by the session each frame): the
        #: region its in-hand refresh is prompted at.
        self.bound_prompts: dict[str, PlaceRegion] = {}
        self.held_ids: frozenset[str] = frozenset()  # candidates of live holds: never aged
        self._pending: list[TrackerRequest] = []
        self._prompted: dict[tuple[str, str], set[tuple[int, int, int]]] = {}
        self._now_ns = 0
        self._grid_ns = -1

    def open_query(
        self,
        goal_id: str,
        query_id: str,
        query: ObjectQuery,
        declaration_key: str,
        *,
        stamp_ns: int,
        invocation_id: str = "",
    ) -> None:
        """Start a standing query (G.1) at goal dispatch, source time ``stamp_ns``."""
        self.queries.open(
            goal_id,
            query_id,
            query,
            declaration_key,
            stamp_ns=stamp_ns,
            invocation_id=invocation_id,
        )

    def close_goal(self, goal_id: str) -> None:
        """End every query of ``goal_id``; tracks persist across goals."""
        self.queries.close_goal(goal_id)
        self._pending = [r for r in self._pending if not r.query_key or r.query_key[0] != goal_id]

    def feed(  # noqa: PLR0912, PLR0915  # reason: flat pass over steps 0-7 with refusal propagation
        self,
        frame: CameraFrame,
        *,
        grid: VoxelLattice | None,
        grid_stamp_ns: int,
        wall_s: float,
        results: Sequence[TrackerResult] = (),
        hand_spaces: Sequence[PlaceRegion] = (),
    ) -> TrackerOutput:
        """Run steps 0-7 on one frame.

        Args:
            frame: The depth frame (``t_base_from_cam`` / ``robot_mask`` ``None`` = ageing only).
            grid: The newest voxel grid, in the tracker's frame, or ``None``.
            grid_stamp_ns: Its ``source_stamp``.
            wall_s: Wall clock, for the segmenter rate and budget only.
            results: Answers to requests issued on this same frame.
            hand_spaces: The declared hands' own spaces (grid frame): a split inside them is
                the hand's own leaked points, and their cells are never cover evidence.
        """
        self._now_ns = max(self._now_ns, frame.stamp_ns, grid_stamp_ns if grid else 0)
        usable = frame.t_base_from_cam is not None and frame.robot_mask is not None
        unavailable_reason: str | None = None
        if not usable:  # ageing only
            unavailable_reason = (
                "no_camera_pose" if frame.t_base_from_cam is None else "no_robot_mask"
            )
            self._refuse(f"camera:{frame.camera}", unavailable_reason, stamp_ns=frame.stamp_ns)
        bound_fits: list[tuple[str, ContainingFit]] = []
        for res in results:
            req = self._sched.complete(res.request_id)
            if req is None:
                continue
            if (req.camera, req.stamp_ns) != (frame.camera, frame.stamp_ns):
                log.warning("object_tracker.result_frame_mismatch", request_id=res.request_id)
                self.record_query_refusal(req, "result_frame_mismatch", frame)
            elif res.failure_reason.strip():
                self._refuse(
                    f"{req.label}@{req.camera}",
                    res.failure_reason.strip()[:128],
                    req=req,
                    frame=frame,
                )
            elif usable and req.service == "locate_in_view":
                if not res.boxes_xyxy:
                    self._refuse(
                        f"{req.label}@{req.camera}",
                        "locate_found_nothing",
                        req=req,
                        frame=frame,
                    )
                scores = res.box_scores or (None,) * len(res.boxes_xyxy)
                self._pending += [
                    self._robot_box_reprompt(
                        replace(
                            req, service="segment_in_view", box_xyxy=b, served_ns=-1,
                            request_id=0, prompt_score=sc,
                        ),
                        frame,
                        grid,
                    )
                    for b, sc in zip(res.boxes_xyxy, scores, strict=True)
                ]  # fmt: skip
            elif usable and (res.mask is None or grid is None):
                why = "no_mask" if res.mask is None else "no_grid"
                self._refuse(f"{req.label}@{req.camera}", why, req=req, frame=frame)
            elif usable and res.mask is not None and grid is not None:
                masks = (res.mask, *res.alternatives)
                for n, mask in enumerate(masks, start=1):
                    bound = self._apply_mask(
                        req, mask, frame, grid, grid_stamp_ns, hand_spaces, last=n == len(masks)
                    )
                    if bound is not _Next.NEXT:
                        break
                if isinstance(bound, tuple):
                    bound_fits.append(bound)
        if grid is not None and grid_stamp_ns > self._grid_ns:
            self._grid_ns = grid_stamp_ns
            for t in self.store.all():
                self._confirm(t, grid, grid_stamp_ns, hand_spaces, frame if usable else None)
        if usable:
            self._visibility(frame)
        age_tracks(
            self.store,
            self._now_ns,
            max_age_s=self.params.max_age_s,
            lost_s=self.params.lost_s,
            keep_lost_s=self.params.keep_lost_s,
            held=self.held_ids,
        )
        tracks = self.store.all()
        blocked = frozenset(
            t.object_id
            for t in tracks
            if self.catalog_declarations is not None
            and self.catalog_declarations.blocked(t.object_id, t.epoch)
        )
        escalations = self.queries.update(tracks, now_ns=self._now_ns, blocked_ids=blocked)
        if unavailable_reason is not None:
            pending_keys = {
                query.key
                for query in self.queries.due(now_ns=self._now_ns)
                if query.status in (ObjectQueryStatus.UNRESOLVED, ObjectQueryStatus.RESOLVING)
            }
            pending_keys.update((e.goal_id, e.query_id) for e in escalations)
            for query in self.queries.queries:
                if query.key not in pending_keys:
                    continue
                self.queries.record_observation(
                    *query.key,
                    ObjectQueryObservation(
                        camera=frame.camera,
                        stamp_ns=frame.stamp_ns,
                        reason=unavailable_reason,
                    ),
                )
        live = {t.object_id for t in tracks}
        # Track ids are never reused: drop per-track state for every row the store lost
        # (aged out, displaced by a mint, retired by the session's release).
        for per_track in (
            self._views, self._cells, self._centroid, self._disagree, self._disagree_ns,
            self._map_cells, self._served, self._low_score,
            self._low_score_run, self._first_extent, self._growth_votes, self._growth_admitted,
            self._bound_anchor,
        ):  # fmt: skip
            for oid in [k for k in per_track if k not in live]:
                del per_track[oid]
        # A released target's clear size outlives its track while its re-find window is open:
        # ``_vision_refind`` grows the hidden side to it (review of #521: the track retired ~3 s
        # after release, the window is ``_RECENT_S``, and a re-find boxed the visible half only).
        refindable = live | {k for k, v in self._let_go.items() if v is not None}
        for per_track in (self._clear_size, self._clear_extent):
            for oid in [k for k in per_track if k not in refindable]:
                del per_track[oid]
        if self.surface_models is not None:
            self.surface_models.retain(live)
        if self.catalog_declarations is not None:
            self.catalog_declarations.retain(live)
        issued = self._sched.next(
            self._candidates(frame, grid, hand_spaces) if usable else [], wall_s=wall_s
        )
        if issued is not None:
            self._mark_served(issued)
        return TrackerOutput(
            requests=() if issued is None else (issued,),
            tracked=self.store.snapshot(self._now_ns),
            escalations=tuple(escalations),
            bound_fits=tuple(bound_fits),
        )

    # -- step 1/3/6: a mask becomes a fit, the fit a track ---------------------------------

    def _note_low_score(self, object_id: str, score: float | None) -> None:
        """Remember a track minted from a prompt box under ``LOW_PROMPT_SCORE``."""
        if score is not None and score < LOW_PROMPT_SCORE:
            self._low_score[object_id] = score

    def _robot_box_reprompt(
        self, req: TrackerRequest, frame: CameraFrame, grid: VoxelLattice | None
    ) -> TrackerRequest:
        """``req`` re-prompted on the object a mostly-robot detector box hides, else ``req``.

        #418 PickOrange: the open-vocabulary locator boxed the arm over orange_3 as "orange";
        SAM, prompted with that box, segmented the arm. When more than
        ``ROBOT_BOX_FRACTION`` of the box is ROBOT, the prompt becomes a positive point on
        the free (never band, never robot) pixels inside the box that stand above the
        support -- the connected region of them lying most beside the robot (what it half
        hides; ``_most_hidden``), at its pixel nearest its centroid -- plus
        ``ROBOT_BOX_NEGATIVES`` negative points on the robot, the box kept as the bound
        the mask must lie in. The fit then goes through the robot: PROVISIONAL.
        """
        if req.box_xyxy is None or frame.robot_mask is None or frame.t_base_from_cam is None:
            return req
        h, w = frame.robot_mask.shape
        x0, y0 = max(int(req.box_xyxy[0]), 0), max(int(req.box_xyxy[1]), 0)
        x1, y1 = min(math.ceil(req.box_xyxy[2]), w), min(math.ceil(req.box_xyxy[3]), h)
        sub = frame.robot_mask[y0:y1, x0:x1]
        if sub.size == 0 or float((sub == ROBOT_PIXEL).mean()) <= ROBOT_BOX_FRACTION:
            return req
        depth = frame.depth_m[y0:y1, x0:x1]
        free = (sub == 0) & np.isfinite(depth) & (depth > 0)
        k = frame.intrinsics
        vv, uu = np.nonzero(free)
        z = depth[vv, uu]
        cam = np.stack([(uu + x0 - k.cx) * z / k.fx, (vv + y0 - k.cy) * z / k.fy, z], axis=1)
        t = frame.t_base_from_cam
        base_z = cam @ t[2, :3] + t[2, 3]
        cell = grid.resolution if grid is not None else 0.015
        support = float(np.percentile(base_z, 10)) if len(base_z) else 0.0
        above = np.zeros(sub.shape, dtype=bool)
        above[vv[base_z > support + cell], uu[base_z > support + cell]] = True
        region = _most_hidden(above, sub == ROBOT_PIXEL)
        if region is None:
            self._refuse(
                f"{req.label}@{req.camera}",
                "robot_box_nothing_beside",
                req=req,
                frame=frame,
            )
            return req
        ry, rx = np.nonzero(region)
        cy, cx = ry.mean(), rx.mean()
        i = int(np.argmin((ry - cy) ** 2 + (rx - cx) ** 2))
        point = (float(rx[i] + x0), float(ry[i] + y0))
        robot_v, robot_u = np.nonzero((sub == ROBOT_PIXEL) & np.isfinite(depth) & (depth > 0))
        order = np.argsort((robot_u - rx[i]) ** 2 + (robot_v - ry[i]) ** 2)
        negatives = tuple(
            (float(robot_u[j] + x0), float(robot_v[j] + y0))
            for j in order[:: max(len(order) // ROBOT_BOX_NEGATIVES, 1)][:ROBOT_BOX_NEGATIVES]
        )
        log.info(
            "object_tracker.robot_box_reprompt",
            label=req.label,
            camera=req.camera,
            robot_fraction=round(float((sub == ROBOT_PIXEL).mean()), 2),
            region_px=int(region.sum()),
            point=point,
        )
        return replace(
            req, box_xyxy=None, point_uv=point, within_xyxy=req.box_xyxy, negative_uv=negatives
        )

    def _locate_roi(
        self, region: PlaceRegion | None, frame: CameraFrame
    ) -> tuple[float, float, float, float] | None:
        """``region``'s image rectangle grown by ``LOCATE_ROI_PAD``, normalised.

        ``None`` when there is no region, it is in another frame, or it misses the image.
        """
        if region is None or region.frame_id != self.store.frame_id:
            return None
        assert frame.t_base_from_cam is not None  # reason: only usable frames issue
        rect = project_box(region, frame.intrinsics, frame.t_base_from_cam)
        if rect is None:
            return None
        (c0, r0, c1, r1), _ = rect
        pw, ph = LOCATE_ROI_PAD * (c1 - c0), LOCATE_ROI_PAD * (r1 - r0)
        w, h = frame.intrinsics.width, frame.intrinsics.height
        return (max(c0 - pw, 0) / w, max(r0 - ph, 0) / h, min(c1 + pw, w) / w, min(r1 + ph, h) / h)

    def _combine_views(
        self, oid: str, camera: str, fit: ContainingFit, frame: CameraFrame, resolution: float
    ) -> tuple[ContainingFit, FitKind]:
        """``fit`` intersected with the other cameras' recent clear fits of ``oid``.

        Each camera's newest clear (non-provisional) fit is kept; fits from other cameras
        no older than ``VIEW_WINDOW_S`` join ``intersect_views`` with this one as the
        primary. Its own fallbacks (a refused or too-small second silhouette, a measured
        point outside the result) return this fit unchanged.
        """
        assert frame.t_base_from_cam is not None  # reason: only usable frames fit
        views = self._views.setdefault(oid, {})
        if not fit.provisional:
            views[camera] = (fit, frame.t_base_from_cam, frame.stamp_ns)
        others = [
            (f, t)
            for cam, (f, t, stamp) in views.items()
            if cam != camera and abs(frame.stamp_ns - stamp) <= VIEW_WINDOW_S * 1e9
        ]
        if fit.provisional or not others or fit.box is None:
            return fit, FitKind.SILHOUETTE_CONE
        out = intersect_views(
            [(fit, frame.t_base_from_cam), *others],
            resolution=resolution,
            frame_id=self.store.frame_id,
            stamp_ns=frame.stamp_ns,
        )
        return out.fit, out.fit_kind

    def record_query_refusal(self, req: TrackerRequest, reason: str, frame: CameraFrame) -> None:
        """Record one refused observation against the request's standing query, if any.

        Refit/mismatch producers use this same entry point so evidence stays query-scoped
        and bounded instead of emitting one failure event per camera frame.
        """
        if req.query_key is None:
            return
        self.queries.record_observation(
            *req.query_key,
            ObjectQueryObservation(camera=req.camera, stamp_ns=frame.stamp_ns, reason=reason[:128]),
        )

    def _refuse(
        self,
        key: str,
        why: str,
        *,
        req: TrackerRequest | None = None,
        frame: CameraFrame | None = None,
        **numbers: object,
    ) -> None:
        """Count a refusal and report it at INFO, rate-limited per ``(key, why)``.

        ``object_tracker.refused`` at most once per ``REFUSAL_LOG_PERIOD_S`` source
        seconds per ``(key, why)``, with the running count.

        Never DEBUG-only: #418 restock grounded nothing for whole trials and no reason was
        visible at INFO.
        """
        if req is not None and frame is not None:
            self.record_query_refusal(req, why, frame)
        k = (key, why)
        self.refusal_counts[k] += 1
        last = self._refusal_logged_ns.get(k)
        if last is None or self._now_ns - last >= REFUSAL_LOG_PERIOD_S * 1e9:
            self._refusal_logged_ns[k] = self._now_ns
            log.info(
                "object_tracker.refused", key=key, why=why, count=self.refusal_counts[k], **numbers
            )

    def _support(self, grid: VoxelLattice, surface: NDArray[np.float64]) -> float | None:
        """The lowest the support under ``surface`` can be, or ``None`` when none is measured.

        ``support_top_from_voxels`` over the column around the mask's points, its ring
        counted only on cells that start below the lowest seen point (A2: nothing above it
        can be what the object stands on, so a neighbour's flank never becomes the support). The
        measured top face is quantised up to one cell above the real surface; the cone is
        truncated one cell lower, and the clearing floor stays at the measured top.
        """
        p = self.params
        near = surface[:, :2].mean(axis=0)
        top_xy = surface[int(np.argmax(surface[:, 2])), :2]
        spread = float(np.ptp(surface[:, :2], axis=0).max()) / 2
        reach = spread + p.support_probe_m + grid.resolution
        centers = grid.occupied_centers()
        column = centers[
            np.all(np.abs(centers[:, :2] - near) <= reach, axis=1)
            & (centers[:, 2] <= surface[:, 2].max() + grid.resolution)
        ]
        lowest = float(surface[:, 2].min())
        starts = centers[:, 2] - 0.5 * grid.resolution

        def measure(below: float) -> float | None:
            return support_top_from_voxels(
                grid,
                column,
                # Anchored on the object's highest seen point, not its centroid: a hollow body
                # (a bowl) has bare support at its centroid, and an anchor there finds none.
                near_xy=(float(top_xy[0]), float(top_xy[1])),
                min_cells=p.support_min_cells,
                probe_margin_m=p.support_probe_m,
                surface_centers=centers[starts < below],
            )

        # A cell that starts below the lowest seen point: the support's top layer is often the
        # one the object's own bottom sits in (a bowl on a table at a cell's mid-height), and
        # the cone is cut a cell under the measured top anyway.
        top = measure(lowest)
        if top is None:
            # A support whose surface lies on a layer boundary (#418 r6po_dbg7: the counter at
            # z 0.0302 over the 0.030 boundary) sits in the layer that starts AT the boundary,
            # and a seen point a hair under it (0.0300) drops that whole layer above. Only when
            # nothing is found: the layer starting within one cell above the lowest seen point.
            top = measure(lowest + grid.resolution)
            if top is not None:
                log.info("object_tracker.support_on_layer_boundary", lowest_seen_z=round(lowest, 4),
                         support_top_z=round(top, 4))  # fmt: skip
        return None if top is None else top - grid.resolution

    def _support_seen(
        self,
        frame: CameraFrame,
        mask: NDArray[np.bool_],
        surface: NDArray[np.float64],
        support_z: float,
        grid: VoxelLattice,
    ) -> float:
        """``support_z`` raised one cell when the camera sees the support above its measured top.

        The voxel probe measures the support's top face as a layer boundary: ``support_z`` +
        one cell. A support surface lying a hair above a boundary is mapped in BOTH layers
        (#418 r6po_dbg3-6: the counter at z 0.0302 over the 0.030 boundary filled the
        0.015-0.030 and 0.030-0.045 layers), and the probe takes the lower one: the support
        a cell low, so K7's support band missed the counter cells the hand reached. Raised
        one cell (never lowered) when both hold: the camera's own returns around the object
        (not the mask, not the robot), within a cell of that top, are one flat surface (10-90 %
        spread within a quarter cell: not a partial mask's own body around it) whose median
        lies at or above the top (within ``_SUPPORT_SEEN_TOL_M``: ON the boundary, not inside
        the measured layer); and the map holds the layer above as well, under those returns
        (at least ``support_min_cells`` of it occupied).
        """
        res = grid.resolution
        if frame.t_base_from_cam is None:
            return support_z
        p = self.params
        top = support_z + res
        k = frame.intrinsics
        rows, cols = np.nonzero(~mask & (frame.depth_m > 0) & np.isfinite(frame.depth_m))
        if frame.robot_mask is not None:
            keep = frame.robot_mask[rows, cols] == 0
            rows, cols = rows[keep], cols[keep]
        z = frame.depth_m[rows, cols]
        cam = np.stack(((cols + 0.5 - k.cx) / k.fx * z, (rows + 0.5 - k.cy) / k.fy * z, z), axis=1)
        t = frame.t_base_from_cam
        pts = cam @ t[:3, :3].T + t[:3, 3]
        centre = surface[:, :2].mean(axis=0)
        spread = float(np.ptp(surface[:, :2], axis=0).max()) / 2
        d = np.linalg.norm(pts[:, :2] - centre, axis=1)
        ring = (
            (d > spread + res)
            & (d <= spread + p.support_probe_m)
            & (np.abs(pts[:, 2] - top) <= res)
        )
        if int(ring.sum()) < p.support_min_cells:
            return support_z
        lo, seen, hi = (float(v) for v in np.percentile(pts[ring, 2], (10, 50, 90)))
        if hi - lo > _T.support_flatness_res_fraction * res or seen < top - _SUPPORT_SEEN_TOL_M:
            return support_z  # not one flat surface (a partial mask's own body), or inside
        # The layer above the measured top, under those returns: occupied in the map too?
        up = pts[ring].copy()
        up[:, 2] = top + 0.5 * res
        local = (up - np.asarray(grid.origin)) @ grid.rotation() / res
        ijk = np.unique(np.floor(local).astype(np.int64), axis=0)
        sx, sy, sz = grid.size
        inside = np.all((ijk >= 0) & (ijk < (sx, sy, sz)), axis=1)
        ijk = ijk[inside]
        held = int(np.count_nonzero(grid.occupancy[ijk[:, 0] + sx * (ijk[:, 1] + sy * ijk[:, 2])]))
        if held < p.support_min_cells:
            return support_z
        log.info("object_tracker.support_raised_to_seen_surface", seen_z=round(seen, 4),
                 voxel_top_z=round(top, 4), points=int(ring.sum()))  # fmt: skip
        return support_z + res

    def _fit(
        self, req: TrackerRequest, mask: NDArray[np.bool_], frame: CameraFrame, grid: VoxelLattice
    ) -> tuple[ContainingFit, float] | None:
        """Measure the support under the mask, fit, and gate on visibility; ``None`` = refused."""
        assert frame.t_base_from_cam is not None
        t_cam = frame.t_base_from_cam
        fp = self.params.fit
        raw, _ = backproject_masked_depth(
            mask & (frame.robot_mask == 0) if frame.robot_mask is not None else mask,
            frame.depth_m,
            frame.intrinsics,
            min_depth_m=fp.min_depth_m,
            max_depth_m=fp.max_depth_m,
        )
        key = f"{req.label}@{req.camera}"
        if len(raw) < fp.min_points:
            self._refuse(
                key,
                "too_few_points",
                req=req,
                frame=frame,
                points=len(raw),
                min_points=fp.min_points,
            )
            return None
        surface = raw @ t_cam[:3, :3].T + t_cam[:3, 3]
        spread = float(np.ptp(surface[:, :2], axis=0).max())
        if spread > 4.0 * GraspDeclaration.MAX_HALF_EXTENT_M:
            # No fit this wide passes `containing_fit`'s half_extent_cap at any yaw: its box
            # holds these points (the same `sure` set), and a box's axis span is at most
            # 2*sqrt(2) times its widest half-extent. Refused before the support scan
            # (masks of the whole table, 0.6-1.1 m, each cost ~2 s of the scan and stalled the
            # tracker until the workspace region went stale).
            self._refuse(key, "half_extent_cap", req=req, frame=frame, points=len(raw),
                         spread_m=round(spread, 4))  # fmt: skip
            return None
        support_z = self._support(grid, surface)
        if support_z is None:
            # What decided it (refits refused for 30+ s on a pushed banana).
            self._refuse(
                key,
                "no_support",
                req=req,
                frame=frame,
                points=len(raw),
                lowest_z=round(float(surface[:, 2].min()), 4),
                highest_z=round(float(surface[:, 2].max()), 4),
                spread_m=round(float(np.ptp(surface[:, :2], axis=0).max()), 4),
            )
            return None
        support_z = self._support_seen(frame, mask, surface, support_z, grid)
        fit = containing_fit(
            mask,
            frame.depth_m,
            frame.robot_mask,
            frame.intrinsics,
            t_cam,
            support_z=support_z,
            resolution=grid.resolution,
            frame_id=self.store.frame_id,
            stamp_ns=frame.stamp_ns,
            params=fp,
            # A detector's box (a query prompt), never a track's own projected box.
            detector_box=req.box_xyxy if req.object_id is None else None,
            part_box=req.box_xyxy or req.within_xyxy,
        )
        log.debug("object_tracker.fit", label=req.label, refusal=fit.refusal, **fit.assumptions)
        if fit.box is None or fit.tight is None:
            self._refuse(key, str(fit.refusal), req=req, frame=frame, points=len(raw))
            return None
        vis = visible_fraction(fit.box, frame.depth_m, frame.robot_mask, frame.intrinsics, t_cam)
        floor = self.params.theta_vis_provisional if fit.provisional else self.params.theta_vis
        if vis is not None and vis < floor:
            self._refuse(
                key,
                "occluded",
                req=req,
                frame=frame,
                visible=round(vis, 3),
                theta_vis=floor,
            )
            return None
        return fit, support_z

    def _apply_mask(
        self,
        req: TrackerRequest,
        mask: NDArray[np.bool_],
        frame: CameraFrame,
        grid: VoxelLattice,
        grid_stamp_ns: int,
        hands: Sequence[PlaceRegion],
        *,
        last: bool = True,
    ) -> tuple[str, ContainingFit] | _Next | None:
        """Fit ``mask`` into a track; ``_Next.NEXT`` when a smaller candidate should be tried.

        Not the ``last`` candidate: a refused fit, or one whose box holds two bodies on the
        map (``map_split``), hands over to the next one instead of minting a SPLIT track.
        Returns ``(object_id, fit)`` for a measured BOUND track, else ``None``.
        """
        gated = self._gate_candidate(req, mask, frame, grid, hands, last=last)
        if not isinstance(gated, _Candidate):
            return gated
        fit, support_z, centroid, cells = gated
        # Only a track of the mask's own label takes its fit: a banana let go into the bowl
        # is inside the bowl's box, and by position alone its every fit refit the bowl
        # (never re-found, the gripper stopped in its cells 25 s on).
        same = [t for t in self.store.all() if t.label == req.label]
        oid, kind = associate(same, centroid, voxel_m=grid.resolution)
        if kind == "tie":
            return None
        if fit.provisional and fit.box is not None:
            fit = self._grow_hidden(fit, oid, req.label, frame, grid, support_z)
        floor = support_z + grid.resolution
        if oid is None:
            # The refit size rule holds for a mint too: a fit through the robot far past every
            # clear same-label track swallowed something else, and as a new track nothing ever
            # shrank it (pro#51 restock seed 115: a dropped 8 cm carton minted PROVISIONAL at
            # 22 x 27 x 31 cm and its payload union stopped the arm at the bind). No track leaves
            # the cells plain obstacles; a later clear fit mints it.
            track = (
                None
                if self._duplicate_mint(req, fit, frame)
                or self._provisional_oversize(None, req.label, fit)
                else self._mint_track(
                    req, fit, cells, frame, grid, support_z=support_z, floor=floor
                )
            )
            if track is None:
                return None
        else:
            old = self.store.get(oid)
            assert old is not None
            if old.status is TrackStatus.BOUND:
                return self._measure_bound(old, fit, frame, grid)
            moved = self._moved(old, fit, grid)
            if self._refuse_refit(old, fit, cells, centroid, support_z, grid, hands, frame, moved):
                return None
            track, fit = self._refit_track(
                old,
                kind,
                req.camera,
                fit,
                cells,
                frame,
                grid,
                hands,
                moved,
                support_z=support_z,
                floor=floor,
            )
        self._retire_context_duplicates(track, fit)
        self._commit_fit(track, fit, cells, centroid, grid, grid_stamp_ns, hands, frame)
        return None

    def _retire_context_duplicates(self, track: TrackedObject, fit: ContainingFit) -> None:
        """A TARGET / DESTINATION fit retires (LOST) a CONTEXT track that boxes its object.

        The named query is the better evidence of what the object is. Restock bench_dup_05
        (2026-10-31): the target went LOST, the generic "carton" query minted over it and
        confirmed, and the target's refits then never confirmed (``cover_already_tracked``):
        the goal timed out ungrounded. A context track is only avoided; its cells stay on the
        map either way.
        """
        if track.role is ObjectRole.CONTEXT or fit.box is None:
            return
        others = [
            o
            for o in self.store.all()
            if o.role is ObjectRole.CONTEXT
            and o.object_id != track.object_id
            and o.status is not TrackStatus.BOUND  # a held payload is never retired
        ]
        while (dup := duplicate_of(fit.box, others)) is not None:
            old = self.store.get(dup[0])
            assert old is not None
            self.store.replace(old.model_copy(update={"status": TrackStatus.LOST}))
            others = [o for o in others if o.object_id != dup[0]]
            log.info("object_tracker.context_duplicate_retired", object_id=dup[0],
                     by=track.object_id, iou=round(dup[1], 2))  # fmt: skip

    def _gate_candidate(
        self,
        req: TrackerRequest,
        mask: NDArray[np.bool_],
        frame: CameraFrame,
        grid: VoxelLattice,
        hands: Sequence[PlaceRegion],
        *,
        last: bool,
    ) -> _Candidate | _Next | None:
        """The mask's fit and map cells, or why it is refused (``_Next.NEXT``: try the next)."""
        fitted = self._fit(req, mask, frame, grid)
        if fitted is None:
            return None if last else _Next.NEXT
        fit, support_z = fitted
        surface = fit.surface
        centroid = tuple(float(v) for v in surface.mean(axis=0))
        cells = surface_cells(grid, surface, exclude=hands)
        if not last and fit.box is not None:
            split = cover_verdict(
                grid, cells, fit.box, support_z=support_z, own=hands,
                others=self._other_boxes(centroid), static=self._static_keys(grid),
            )  # fmt: skip
            if split.kind == "map_split":
                self._refuse(
                    f"{req.label}@{req.camera}",
                    "candidate_split",
                    req=req,
                    frame=frame,
                    detail=split.why,
                )
                return _Next.NEXT
        return _Candidate(fit, support_z, centroid, cells)

    def _duplicate_mint(self, req: TrackerRequest, fit: ContainingFit, frame: CameraFrame) -> bool:
        """Refuse (``True``) a CONTEXT query's mint whose box is a live track's box.

        CONTEXT only: a context track is there to be avoided, which the live track already is;
        a TARGET / DESTINATION query refused because another query boxed its object first
        would never resolve, so those still mint.
        """
        q = self.queries.get(*req.query_key) if req.query_key else None
        if q is None or q.query.role is not ObjectRole.CONTEXT or fit.box is None:
            return False
        dup = duplicate_of(fit.box, self.store.all())
        if dup is None:
            return False
        self._refuse(f"{req.label}@{req.camera}", "duplicate_of_track", req=req, frame=frame,
                     track=dup[0], iou=round(dup[1], 2))  # fmt: skip
        return True

    def _mint_track(
        self,
        req: TrackerRequest,
        fit: ContainingFit,
        cells: frozenset[Cell],
        frame: CameraFrame,
        grid: VoxelLattice,
        *,
        support_z: float,
        floor: float,
    ) -> TrackedObject | None:
        """A new track for an unassociated fit of a standing query; ``None`` without one."""
        q = self.queries.get(*req.query_key) if req.query_key else None
        if q is None:
            log.info("object_tracker.refit_unassociated", label=req.label)
            return None
        if (
            q.query.role is ObjectRole.TARGET
            and fit.box is not None
            and self._implausible_box(q.query.label, fit.box, agg=max)
        ):
            # A clear fit of two touching oranges minted one 19-32 cm "orange" whose box then
            # blocked the real ones (yw1 ep8); provisional fits already had this gate.
            self._refuse(f"{req.label}@{req.camera}", "oversize_mint", req=req, frame=frame,
                         extent_m=round(2.0 * max(fit.box.half_extents[:2]), 3))  # fmt: skip
            return None
        if fit.box is not None and (
            hit := self._role_conflict(
                q.query.role, q.query.label, fit.box, frame.stamp_ns, mint=True
            )
        ):
            self._refuse(f"{req.label}@{req.camera}", "role_conflict", req=req, frame=frame,
                         conflict=hit)  # fmt: skip
            return None
        track = self.store.mint(
            label=q.query.label,
            role=q.query.role,
            declaration_key=q.declaration_key,
            box=fit.box,
            tight=(fit.tight,),
            fit_kind=FitKind.SILHOUETTE_CONE,
            support_z=support_z,
            support_ref="map",
            floor_z=floor,
            visible_cells=len(cells),
            occupied_cells=0,
            last_fit_stamp_ns=frame.stamp_ns,
            last_confirmed_stamp_ns=0,
            provisional=fit.provisional,
        )
        self._combine_views(track.object_id, req.camera, fit, frame, grid.resolution)
        self._note_low_score(track.object_id, req.prompt_score)
        return track

    def _measure_bound(
        self, old: TrackedObject, fit: ContainingFit, frame: CameraFrame, grid: VoxelLattice
    ) -> tuple[str, ContainingFit]:
        """A BOUND track's refit: observed, and its box follows the object in the hand."""
        oid = old.object_id
        self._observe_catalog(old, fit, frame)
        self._observe_surface(old, fit, frame)
        if fit.box is not None:
            # The unmoved-refit rule holds in the hand too: a mask of the part beside the
            # fingers shrank the box and left the rest of the object outside it (Spark pro#51
            # banana, 2026-11-05: 6 of 27 stops, a 12 x 6 cm bound box on an 18 cm banana, the
            # finger met its far end 0.4 s after the bind). The map cannot tell: the bridge
            # clears the bound box's cells at the bind and the camera re-adds them later. While
            # the fit still lies in the box (grown a cell, every point: a lift leaves it upward)
            # the object has not moved and the box holds both; past that it follows the object.
            # Safety-WG review owed.
            # Tested against the box the object was last seen whole in (the anchor), never the
            # kept box: enclosing each partial fit into the last kept box ratcheted it along a
            # slow lift (under a cell per fit) without bound.
            measured = fit.box
            epoch, anchor = self._bound_anchor.get(oid, (old.epoch, old.box))
            if epoch != old.epoch:
                anchor = old.box
            away = ~_inside(fit.surface, _grown(anchor, grid.resolution))
            outside = int(np.count_nonzero(away))
            if outside < max(_MOVED_MIN_POINTS, _MOVED_MIN_FRACTION * len(fit.surface)):
                fit = replace(fit, box=_enclosing(anchor, measured))
                log.info("object_tracker.bound_partial_kept", object_id=oid, outside=outside,
                         points=len(fit.surface))  # fmt: skip
            else:
                anchor = measured  # it moved: the anchor follows it
            self._bound_anchor[oid] = (old.epoch, anchor)
        if fit.box is not None:
            # The box follows the object as the camera sees it in the hand, never the
            # hand's kinematics (2026-10-22 decision; the bind-time box stayed on the table
            # while the banana was carried 23 cm, the kernel read the binding off the hand and
            # its own cells stopped the payload). A drop
            # moves the box with the object, off the hand. The payload's primitives and
            # the provisional flag stay the bind's. Safety-WG review owed.
            current = self.store.get(oid)
            assert current is not None
            self.store.replace(
                current.model_copy(
                    update={
                        "box": fit.box,
                        "tight": (fit.tight,) if fit.tight is not None else (),
                        "last_fit_stamp_ns": max(current.last_fit_stamp_ns, frame.stamp_ns),
                    }
                )
            )
            log.info("object_tracker.bound_box_measured", object_id=oid,
                     provisional=fit.provisional)  # fmt: skip
        return oid, fit

    def _moved(self, old: TrackedObject, fit: ContainingFit, grid: VoxelLattice) -> bool:
        """Whether the measured surface left ``old``'s box: the object moved."""
        # The measured surface leaves the stored box: the object moved (pushed by the
        # hand), so the old box no longer contains it. The
        # new fit replaces it even when occluded or through the robot, kept PROVISIONAL
        # unless clear (2026-10-21 decision: never a payload, still K2 / K7 geometry).
        # Through the robot, a point above the clear box is no move: a resting object
        # pushed along its support does not rise above its own clear ceiling, and a mask
        # through the hand takes in what it holds over the object (the bowl's every outside point
        # lay above it, 115 of 115, and its clear box was replaced, provisional). A clear fit still
        # counts every point.
        surface = fit.surface
        grown = _grown(old.box, grid.resolution)
        top = grown.pose.xyz[2] + grown.half_extents[2]
        away = ~_inside(surface, grown)
        if fit.provisional:
            away &= surface[:, 2] <= top
        outside = int(np.count_nonzero(away)) if fit.box is not None else 0
        moved = outside >= max(_MOVED_MIN_POINTS, _MOVED_MIN_FRACTION * len(surface))
        if moved:
            log.info(
                "object_tracker.moved_refit",
                object_id=old.object_id,
                status=old.status.value,
                outside=outside,
                points=len(surface),
                provisional=fit.provisional,
            )
        return moved

    def _refuse_refit(  # noqa: PLR0911  # reason: one return per named gate
        self,
        old: TrackedObject,
        fit: ContainingFit,
        cells: frozenset[Cell],
        centroid: tuple[float, ...],
        support_z: float,
        grid: VoxelLattice,
        hands: Sequence[PlaceRegion],
        frame: CameraFrame,
        moved: bool,
    ) -> bool:
        """Keep an unbound track's geometry against this refit; whether it was refused.

        Not a pure check: a refused refit still refreshes the track's evidence (and a
        provisional one re-dates it in the store). Each gate logs.
        """
        oid = old.object_id
        if (
            old.role is ObjectRole.CONTEXT
            and fit.box is not None
            and (
                dup := duplicate_of(
                    fit.box,
                    [o for o in self.store.all() if o.role is not ObjectRole.CONTEXT],
                )
            )
        ):
            # A context refit never revives a track on a named one's object (the refit side of
            # ``_duplicate_mint``): once a target's fit retired the carton track, each "carton"
            # refit brought it back TENTATIVE and the target's next fit retired it again.
            log.info("object_tracker.context_refit_duplicate", object_id=oid, track=dup[0],
                     iou=round(dup[1], 2))  # fmt: skip
            return True
        if fit.box is not None and (
            hit := self._role_conflict(old.role, old.label, fit.box, frame.stamp_ns, than=oid)
        ):
            log.info("object_tracker.refit_role_conflict", object_id=oid, conflict=hit)
            return True
        grown = self._overgrown(oid, fit, moved)
        if grown and not self._admit_growth(
            old, fit, cells, centroid, support_z, grid, hands, frame
        ):
            if self._redate_if_inside(old, fit, grid, frame, _REDATE_SPILL_FRACTION):
                log.info("object_tracker.refit_growth_redated", object_id=oid)
            return True
        if old.status is TrackStatus.OCCLUDED and not moved:
            self._refresh_evidence(oid, cells)
            redated = self._redate_if_inside(old, fit, grid, frame)
            log.info("object_tracker.occluded_refit_ignored", object_id=oid, redated=redated)
            return True
        if fit.provisional and not old.provisional and not moved:
            # A fit through the robot never replaces geometry measured clear of it,
            # unless that geometry no longer contains the object. It does re-date it:
            # the object was measured inside the box now, so the box is a fresh fit for
            # the grasp binder (2026-10-22 decision; the hand hides the target on the approach,
            # every refit was ignored, the clear fit was 2.6-4.7 s old at the grasp and 10 of 10
            # binds fell back to the pinch payload, whose uncovered banana the kernel then stopped
            # on). Safety-WG review owed.
            self._refresh_evidence(oid, cells)
            self.store.replace(
                old.model_copy(
                    update={"last_fit_stamp_ns": max(old.last_fit_stamp_ns, frame.stamp_ns)}
                )
            )
            log.info("object_tracker.provisional_refit_ignored", object_id=oid, redated=True)
            return True
        if not moved and self._provisional_oversize(oid, old.label, fit):
            # A moved fit replaces the box it left (2026-10-21 decision); an unmoved one several
            # times the object's clear size swallowed something else.
            self._refresh_evidence(oid, cells)
            return True
        prior = self._cells.get(oid)
        if not moved and fit.box is not None and prior:
            # An unmoved object's refit must still hold the surface the last fit measured
            # where the map still shows it (2026-10-21 decision; a partial mask's clear refit
            # halved the banana's box and the finger met the rest). A
            # true tightening passes: the surface it drops is no longer occupied.
            left = uncovered_cells(grid, prior, fit.box)
            if left >= max(_MOVED_MIN_POINTS, _MOVED_MIN_FRACTION * len(prior)):
                redated = self._redate_if_inside(old, fit, grid, frame)
                log.info("object_tracker.partial_refit_ignored", object_id=oid,
                         uncovered=left, cells=len(prior), redated=redated,
                         half_extents=[round(h, 3) for h in old.box.half_extents],
                         fit_half=[round(h, 3) for h in fit.box.half_extents])  # fmt: skip
                return True
        if fit.box is not None and old.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED):
            verdict = cover_verdict(
                grid,
                cells,
                fit.box,
                support_z=support_z,
                own=hands,
                others=self._other_boxes(centroid),
                static=self._static_keys(grid),
            )
            if verdict.kind == "map_split":
                # A mask that took in a neighbour too never replaces a confirmed box: the
                # track would go SPLIT and its next fit take a new id (a swap).
                log.info("object_tracker.split_refit_ignored", object_id=oid, why=verdict.why)
                return True
        if grown and fit.box is not None:
            # Admitted and past every later gate: the fit is committed and replaces the
            # reference (``_commit_fit``). Taken earlier, a later gate's refusal left the track
            # with no reference and its one admission spent (review of #479).
            self._first_extent.pop(oid, None)
            self._growth_votes.pop(oid, None)
            self._growth_admitted[oid] = True
            log.info("object_tracker.refit_growth_admitted", object_id=oid,
                     extent_m=round(2.0 * max(fit.box.half_extents[:2]), 3),
                     height_m=round(2.0 * fit.box.half_extents[2], 3))  # fmt: skip
        return False

    def _admit_growth(
        self,
        old: TrackedObject,
        fit: ContainingFit,
        cells: frozenset[Cell],
        centroid: tuple[float, ...],
        support_z: float,
        grid: VoxelLattice,
        hands: Sequence[PlaceRegion],
        frame: CameraFrame,
    ) -> bool:
        """Whether a growth-refused clear refit may become the track's new size reference.

        A first fit from a partial mask holds the reference small for good: restock seeds 111 and
        112 (2026-11-01/02), an 8.2 cm carton first fitted at 6 cm refused every 10.6 cm full
        fit for the whole episode, 3-4 cm under-covered. Admitted, once per track, when all hold:

        * the last ``_GROWTH_VOTES`` refused clear growth fits agree within ``_GROWTH_AGREE_M``
          in longer horizontal extent, height and centre, within ``_GROWTH_WINDOW_NS``;
        * the height stays within a cell of the reference's: a camera above sees the top, so a
          partial view loses a side, never the height; a spill takes in something taller (a
          neighbour, the hand, a shelf) or a fragment (Spark replays 2026-11-03: of 4 admissions
          without this check, 3 were spills 3.2-6.5 cm over the carton, 2 of them 5-6 cm taller);
        * its box reaches no live track the kept box (grown a cell) does not already touch: a
          spill onto a tracked neighbour (a duplicate track of this object overlaps the kept box
          too, so it never blocks);
        * ``cover_verdict`` is not ``map_split``: one body on the map.

        Once per track: a spill that feeds its own prompt (a larger box prompts a larger mask)
        cannot ratchet the box (restock 2026-10-30: 14 -> 24 -> 39 cm). The admitted fit
        replaces the reference once ``_refuse_refit``'s later gates pass it too. *Ceiling*: an
        object whose first clear fit missed its top stays short; growth onto an untracked object
        of the same height that the map sees as one body is admitted once. Safety-WG review owed.
        """
        oid = old.object_id
        ref = self._first_extent.get(oid)
        if fit.box is None or fit.provisional or ref is None or self._growth_admitted.get(oid):
            return False
        ext = (2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1]),
               2.0 * fit.box.half_extents[2])  # fmt: skip
        xy = (fit.box.pose.xyz[0], fit.box.pose.xyz[1])
        recent = [
            v for v in self._growth_votes.get(oid, []) if frame.stamp_ns - v[0] <= _GROWTH_WINDOW_NS
        ]
        votes = [*recent, (frame.stamp_ns, ext, xy)][-_GROWTH_VOTES:]
        self._growth_votes[oid] = votes
        if len(votes) < _GROWTH_VOTES or not all(
            abs(e[0] - ext[0]) <= _GROWTH_AGREE_M
            and abs(e[1] - ext[1]) <= _GROWTH_AGREE_M
            and math.hypot(c[0] - xy[0], c[1] - xy[1]) <= _GROWTH_AGREE_M
            for _, e, c in votes
        ):
            return False
        if abs(ext[1] - ref[2]) > grid.resolution:
            log.info("object_tracker.refit_growth_blocked", object_id=oid, by="height")
            return False
        kept = _aabb(_grown(old.box, grid.resolution))
        new = _aabb(fit.box)
        for o in self.store.all():
            if o.object_id == oid or o.status in (TrackStatus.LOST, TrackStatus.RELEASED):
                continue
            other = _aabb(o.box)
            if _aabb_overlap(new, other) and not _aabb_overlap(kept, other):
                log.info("object_tracker.refit_growth_blocked", object_id=oid, by=o.object_id)
                return False
        verdict = cover_verdict(
            grid, cells, fit.box, support_z=support_z, own=hands,
            others=self._other_boxes(centroid), static=self._static_keys(grid),
        )  # fmt: skip
        if verdict.kind == "map_split":
            log.info("object_tracker.refit_growth_blocked", object_id=oid, by="map_split")
            return False
        return True

    def _overgrown(self, oid: str, fit: ContainingFit, moved: bool) -> bool:
        """Refuse (``True``) a refit past ``refit_growth_factor`` x the track's first clear fit.

        Unlike ``_provisional_oversize`` this holds for a moved or a clear fit too, and the
        reference never grows with the fits it admits: the first clear fit, not the largest.
        A track with no clear fit yet is held to its first provisional fit, but a clear fit is
        never refused against one (a fit through the robot may undersize the object).
        The refused fit never changes the box; ``_refuse_refit`` re-dates it only when most of
        its surface lies inside the kept box. Safety-WG review owed: the kept box is the first
        clear containing fit.
        """
        ref = self._first_extent.get(oid)
        if ref is None or fit.box is None or (not ref[1] and not fit.provisional):
            return False
        first, _, first_h, first_full = ref
        factor = self.params.refit_growth_factor
        extent = 2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1])
        height = 2.0 * fit.box.half_extents[2]
        limit = factor * first
        # Height too (restock scene 00 2026-11-01: an 8 cm carton's refits took in a taller
        # neighbour and its box grew to 14.6 cm tall under a horizontal-only gate).
        limit_h = factor * first_h
        if extent <= limit and height <= limit_h:
            return False
        # A re-orientation is no growth: a carried 6 x 6 x 11 cm carton tipped 90 degrees is
        # 11 cm across, and refusing it kept re-dating the upright box while the protruding
        # part went unmodelled. Admitted when every sorted extent stays within the factor of
        # the first fit's and its volume within the factor too; a spill onto a neighbour or
        # the support adds volume (a 6 x 6 x 8 carton spilled to 12 x 6 x 8 is 2x).
        full = sorted(2.0 * h for h in fit.box.half_extents)
        ref_full = sorted(first_full)
        if all(f <= factor * r for f, r in zip(full, ref_full, strict=True)) and (
            float(np.prod(full)) <= factor * float(np.prod(ref_full))
        ):
            log.info("object_tracker.refit_reoriented", object_id=oid, moved=moved,
                     extents_m=[round(f, 3) for f in full])  # fmt: skip
            return False
        log.info("object_tracker.refit_growth_refused", object_id=oid, moved=moved,
                 extent_m=round(extent, 3), first_m=round(first, 3),
                 limit_m=round(limit, 3), height_m=round(height, 3),
                 limit_h_m=round(limit_h, 3))  # fmt: skip
        return True

    def _redate_if_inside(
        self,
        old: TrackedObject,
        fit: ContainingFit,
        grid: VoxelLattice,
        frame: CameraFrame,
        fraction: float = _REDATE_INSIDE_FRACTION,
    ) -> bool:
        """Re-date ``old``'s fit when a refused refit measured the object inside its box.

        The kept box still contains what the camera measured now, so it is fresh geometry for
        the grasp binder, as a provisional refit already re-dates it (2026-10-22 decision).
        Without it a partial or occluded refit left the box to age: restock 2026-10-30, the
        target's refits under the approaching hand were all refused, its fit was 9.4 s old at
        the grasp and the bind fell back to the pinch payload. A growth-refused refit re-dates
        at ``_REDATE_SPILL_FRACTION``: restock 2026-10-31, a mask spilling onto a touching
        carton was refused every frame and the clear target box aged out the same way.
        Safety-WG review owed.
        """
        surface = fit.surface
        if not len(surface):
            return False
        inside = int(np.count_nonzero(_inside(surface, _grown(old.box, grid.resolution))))
        if inside < fraction * len(surface):
            return False
        self.store.replace(
            old.model_copy(update={"last_fit_stamp_ns": max(old.last_fit_stamp_ns, frame.stamp_ns)})
        )
        return True

    def _refit_track(
        self,
        old: TrackedObject,
        kind: str,
        camera: str,
        fit: ContainingFit,
        cells: frozenset[Cell],
        frame: CameraFrame,
        grid: VoxelLattice,
        hands: Sequence[PlaceRegion],
        moved: bool,
        *,
        support_z: float,
        floor: float,
    ) -> tuple[TrackedObject, ContainingFit]:
        """``old`` refitted (views combined, a touched box kept), stored; and the fit used."""
        oid = old.object_id
        refit = old.status in (TrackStatus.SPLIT, TrackStatus.LOST)  # back to TENTATIVE
        fit, fit_kind = self._combine_views(oid, camera, fit, frame, grid.resolution)
        box = fit.box
        tight: tuple[AttachedCollisionPrimitive | None, ...] = (fit.tight,)
        if (
            fit.box is not None
            and not moved  # a moved object left its old box: the new fit replaces it
            and old.role in _CONTACT_ROLES
            and old.status in _SETTLED
            and hand_touches(old.box, hands, grid.resolution)
        ):
            # The hand at the object hides or pushes part of it, so a refit sees only
            # what is left in view (a clear refit 1.3 s before the stop shrank the banana's box
            # from 26 to 19 cm, the finger met the part it dropped). While
            # touched, a refit never shrinks the box (2026-10-21 decision), nor grows it:
            # unmoved, its surface lies in the old box grown by a cell, so the old box and
            # hulls stand and the refit only re-stamps it (a union per refit ratcheted the banana's
            # box to 32 x 39 cm, every prompt on it masked the table and the bind went stale).
            # Safety-WG review owed.
            box, tight = old.box, old.tight
            log.info("object_tracker.contact_refit_kept", object_id=oid)
        provisional = fit.provisional
        if moved and fit.provisional and not old.provisional and fit.box is not None:
            # A fit through the robot sees only what the robot leaves in view, so it may say
            # where the pushed object went but never how big it is: the clear box keeps its
            # size, re-centred in x/y on the new fit, and the track turns PROVISIONAL as before
            # (2026-10-21 decision: a pushed object's box is never a payload). 2026-10-31: the
            # through-robot box had replaced the clear one, a 2-4x different size. Its hulls
            # were posed for the old place, so they go. Safety-WG review owed.
            # A shift within one grid cell is no push the map could tell from noise (restock
            # 2026-10-31 mix_03: 160 points through the hand "moved" the target 1 cm, it turned
            # provisional and the grasp bound the pinch fallback): the clear box stands as is.
            x, y, _ = fit.box.pose.xyz
            shift = math.dist(old.box.pose.xyz[:2], (x, y))
            if shift <= grid.resolution:
                box, tight, provisional = old.box, old.tight, old.provisional
                log.info("object_tracker.covered_move_within_cell", object_id=oid,
                         shift_m=round(shift, 3))  # fmt: skip
            else:
                box = old.box.model_copy(
                    update={"pose": old.box.pose.model_copy(
                        update={"xyz": (x, y, old.box.pose.xyz[2])})}
                )  # fmt: skip
                tight = ()
                log.info("object_tracker.covered_move_size_kept", object_id=oid,
                         shift_m=round(shift, 3))  # fmt: skip
        track = old.model_copy(
            update={
                "fit_kind": fit_kind,
                "box": box,
                "tight": tight,
                "support_z": support_z,
                "floor_z": max(old.floor_z, floor),
                "visible_cells": len(cells),
                "last_fit_stamp_ns": frame.stamp_ns,
                "status": TrackStatus.TENTATIVE if refit else old.status,
                "identity_inferred": old.identity_inferred or kind == "inferred",
                "provisional": provisional,
            }
        )
        self.store.replace(track)
        return track, fit

    def _commit_fit(
        self,
        track: TrackedObject,
        fit: ContainingFit,
        cells: frozenset[Cell],
        centroid: tuple[float, ...],
        grid: VoxelLattice,
        grid_stamp_ns: int,
        hands: Sequence[PlaceRegion],
        frame: CameraFrame,
    ) -> None:
        """Record an accepted fit's evidence for ``track`` and re-confirm it on the map."""
        ref = self._first_extent.get(track.object_id)
        if fit.box is not None and (ref is None or (not ref[1] and not fit.provisional)):
            self._first_extent[track.object_id] = (
                2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1]),
                not fit.provisional,
                2.0 * fit.box.half_extents[2],
                tuple(2.0 * h for h in fit.box.half_extents),
            )
        if not fit.provisional and fit.box is not None:
            size = 2.0 * min(fit.box.half_extents[0], fit.box.half_extents[1])
            self._clear_size[track.object_id] = max(
                self._clear_size.get(track.object_id, 0.0), size
            )
            self._clear_extent[track.object_id] = max(
                self._clear_extent.get(track.object_id, 0.0),
                2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1]),
            )
            self._label_clear_extent.setdefault(track.label, {})[track.object_id] = (
                self._clear_extent[track.object_id]
            )
        self._cells[track.object_id] = cells
        self._centroid[track.object_id] = (centroid[0], centroid[1], centroid[2])
        self._disagree.pop(track.object_id, None)
        self._disagree_ns.pop(track.object_id, None)
        self._confirm(track, grid, grid_stamp_ns, hands, frame)
        self._observe_catalog(track, fit, frame)
        self._observe_surface(track, fit, frame)

    def _observe_catalog(
        self, track: TrackedObject, fit: ContainingFit, frame: CameraFrame
    ) -> None:
        catalog = self.catalog_declarations
        if catalog is None or fit.provisional or fit.box is None or len(fit.surface) < 2:  # noqa: PLR2004 # reason: a span needs two points
            return
        current = self.store.get(track.object_id)
        if current is None:
            return
        count = sum(
            t.label.strip().casefold() == current.label.strip().casefold()
            and t.status is not TrackStatus.LOST
            for t in self.store.all()
        )
        catalog.associate(
            current.object_id, current.epoch, current.label, observed_label_count=count
        )
        # Axis span is a measured lower bound on diameter; a fitted box is not.
        span = float(np.max(np.ptp(fit.surface, axis=0)))
        reason = catalog.observe_span(
            current.object_id,
            current.epoch,
            span,
            camera=frame.camera,
            stamp_ns=frame.stamp_ns,
            now_ns=self._now_ns,
        )
        if catalog.blocked(current.object_id, current.epoch):
            for query in self.queries.queries:
                if query.matches(current):
                    self.queries.record_observation(
                        *query.key,
                        ObjectQueryObservation(
                            camera=frame.camera,
                            stamp_ns=frame.stamp_ns,
                            reason=catalog.diagnostics()[current.object_id][:128],
                        ),
                    )
        log.info("object_tracker.catalog_observation", object_id=current.object_id, reason=reason)

    def _observe_surface(
        self, track: TrackedObject, fit: ContainingFit, frame: CameraFrame
    ) -> None:
        if self.surface_models is None or fit.provisional or fit.box is None:
            return
        current = self.store.get(track.object_id)
        if current is None or current.status not in (TrackStatus.CONFIRMED, TrackStatus.BOUND):
            return
        pose = homogeneous_from_quat_xyz(fit.box.pose.xyz, fit.box.pose.quat_xyzw)
        reason = self.surface_models.observe(
            current.object_id, current.epoch, fit.surface, pose, frame.stamp_ns
        )
        calibration = self.surface_models.config.free_space_calibration
        if (
            reason in ("accepted", "model_capacity")
            and calibration is not None
            and frame.camera in calibration.sensor_ids
            and frame.t_base_from_cam is not None
        ):
            self.surface_models.clear_observed_free(
                current.object_id,
                current.epoch,
                frame.stamp_ns,
                frame.depth_m,
                frame.robot_mask,
                frame.intrinsics,
                frame.t_base_from_cam,
                now_ns=self._now_ns,
                max_age_ns=int(calibration.max_age_s * _NS),
                depth_error_m=calibration.depth_error_m,
                pose_error_m=calibration.pose_error_m,
                pixel_error=calibration.pixel_error,
                min_depth_m=self.params.fit.min_depth_m,
                max_depth_m=self.params.fit.max_depth_m,
            )
        log.info("object_tracker.surface_model", object_id=track.object_id, reason=reason,
                 stamp_ns=frame.stamp_ns)  # fmt: skip

    # -- steps 4-5: map confirmation and status ---------------------------------------------

    def _confirm(  # noqa: PLR0912  # reason: each map verdict names its own outcome
        self,
        t: TrackedObject,
        grid: VoxelLattice,
        stamp_ns: int,
        hands: Sequence[PlaceRegion],
        frame: CameraFrame | None = None,
    ) -> None:
        if t.status is TrackStatus.RELEASED:
            self._note_release(t, stamp_ns)
            return
        cells = self._cells.get(t.object_id)
        if cells is None or t.status in (TrackStatus.BOUND, TrackStatus.LOST):
            return
        others = [
            o.box for o in self.store.all() if o.object_id != t.object_id and o.status in _SETTLED
        ]
        if t.role is ObjectRole.DESTINATION:
            others += self._resting_placed()  # what rests in it is no foreign body
        # Visibility-aware cover (2026-10-21 decision; Isaac baseline: a successful grasp keeps
        # the fingers on the banana for 255-466 steps): an expected cell the robot hides is no
        # evidence the object is gone. While the hand is at the target or destination, a second
        # body inside its box is the object fragmenting under contact (a thin banana's own
        # 6-cell piece, 5 cm from the fingers; the bowl pushed 99 mm), so it
        # does not block a confirmation the visible cover still has to earn. Safety-WG review
        # owed.
        hidden = _hidden_cells(grid, cells, frame)
        contact = (
            t.role in _CONTACT_ROLES
            and t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
            and hand_touches(t.box, hands, grid.resolution)
        )
        v = cover_verdict(
            grid,
            cells,
            t.box,
            support_z=t.support_z,
            own=hands,
            others=others,
            static=self._static_keys(grid),
            hidden=hidden,
        )
        exclude = [*others, *hands]
        if not v.kind:
            self._map_cells[t.object_id] = len(
                box_cells(grid, t.box, support_z=t.support_z, exclude=exclude)
            )
            self._map_cells_max[t.object_id] = max(  # pro#51: hidden passes show 2-7 of ~20
                self._map_cells_max.get(t.object_id, 0), self._map_cells[t.object_id]
            )
        ref = self._map_cells.get(t.object_id, v.visible if v.kind == "map_disagrees" else 0)
        if contact and v.kind in ("map_disagrees", "hidden") and ref:
            v = self._contact_reanchor(t, v, grid, ref, hands, others, exclude)
        if contact and v.kind in ("map_split", "map_split_own"):
            v = replace(v, kind="", why=f"split under contact: {v.why}")
        status = t.status
        update: dict[str, object] = {}
        if v.kind == "hidden" or self._contact_hold(t, v.kind, v.why, hands, grid.resolution):
            return  # no evidence either way: held, never re-stamped
        if not v.kind:
            self._disagree.pop(t.object_id, None)
            self._disagree_ns.pop(t.object_id, None)
            if status is TrackStatus.SPLIT:
                return  # ineligible until a camera re-fit
            update = {
                "occupied_cells": v.occupied,
                "visible_cells": v.visible,  # the kernel's cover bar: what the camera can see
                "last_confirmed_stamp_ns": stamp_ns,
            }
            if status is TrackStatus.TENTATIVE:
                # A destination's box is a containing box, cavity included: a target lying in
                # it is the goal state, not a ghost over another object's cells (the
                # dropped banana never re-confirmed against the bowl's box and
                # the gripper stopped in its cells). Only ownership; the split verdict above
                # still sees every box.
                owners = [
                    o.box
                    for o in self.store.all()
                    if o.object_id != t.object_id
                    and o.status in _SETTLED
                    and not (t.role is ObjectRole.TARGET and o.role is ObjectRole.DESTINATION)
                ]
                if not self._may_confirm(t, grid, cells, owners, v.occupied, stamp_ns):
                    return
                update["status"] = TrackStatus.CONFIRMED
            if t.role is ObjectRole.TARGET:
                self._note_placed(t)
        elif v.kind == "map_split":
            update = {"status": TrackStatus.SPLIT}
        elif v.kind == "map_disagrees":
            # `_commit_fit` and the per-grid pass can both confirm on one stamp: count it once.
            n = self._disagree.get(t.object_id, 0)
            if self._disagree_ns.get(t.object_id) != stamp_ns:
                n += 1
                self._disagree_ns[t.object_id] = stamp_ns
            self._disagree[t.object_id] = n
            if n >= self.params.disagree_grids:
                update = {"status": TrackStatus.LOST}
        if v.kind:
            self._low_score_run.pop(t.object_id, None)  # a refuting grid ends the run
            log.debug("object_tracker.cover", object_id=t.object_id, kind=v.kind, why=v.why)
        if update:
            self.store.replace(t.model_copy(update=update))

    def _contact_reanchor(
        self,
        t: TrackedObject,
        v: CoverVerdict,
        grid: VoxelLattice,
        ref: int,
        hands: Sequence[PlaceRegion],
        others: Sequence[PlaceRegion],
        exclude: Sequence[PlaceRegion],
    ) -> CoverVerdict:
        """``v``, or the verdict on the map's cells in the box when the hand moved the object.

        The hand pushes the object inside its box (a banana pushed 66 mm held 12 of its 26
        expected cells) or hides it (3-7 cells in view while the map held 33-52 in the box), and
        the hold ages it out mid-grasp. The object is where the map holds it in the box:
        re-anchored on those cells when they number half to ``_REANCHOR_MAX_RATIO`` times
        ``ref`` (what it filled at its last plain pass; 2026-10-21 decisions; the kernel's own
        in-box cover test; a push out of the box still fails, more is the box holding something
        else). Safety-WG review and hazard row owed.
        """
        body = box_cells(grid, t.box, support_z=t.support_z, exclude=exclude)
        if cover_need(ref) <= len(body) <= _REANCHOR_MAX_RATIO * ref:
            moved = cover_verdict(
                grid,
                body,
                t.box,
                support_z=t.support_z,
                own=hands,
                others=others,
                static=self._static_keys(grid),
            )
            if moved.kind != "map_disagrees":
                log.info("object_tracker.contact_reanchor", object_id=t.object_id,
                         was=v.why, cells=len(body), ref=ref)  # fmt: skip
                self._cells[t.object_id] = body
                return moved
        return v

    def _note_placed(self, t: TrackedObject) -> None:
        """Record where TARGET ``t`` rests on a settled destination (``_placed_box``), if it does.

        At its release over one, and at every confirmation of a target lying on one: an orange
        that dropped or rolled onto the plate, re-found there, is as placed as one let go over it
        (A/B ab1b_po seed 8: with release-only records the plate and the orange on it each split
        on the other, and the plate was re-minted five times). A foreign object never gets here
        without first being confirmed under a target's identity.
        """
        for d in self.store.all():
            if d.role is ObjectRole.DESTINATION and d.status in _SETTLED:
                placed = _placed_box(t.box, d.box)
                if placed is not None:
                    if t.object_id not in self._placed:
                        log.info("object_tracker.placed", object_id=t.object_id,
                                 destination=d.object_id)  # fmt: skip
                    self._placed[t.object_id] = placed
                    return

    def _resting_placed(self) -> list[PlaceRegion]:
        """The ``_placed`` boxes whose target still rests in them; the others are forgotten.

        A target picked up again (BOUND) or confirmed outside its placed box left it: the
        region no longer explains a body there (a foreign object set in it must split the
        destination). A retired target's record stays (it was never seen leaving) until a held
        payload is over it: the object was picked up again under a new id (review of #521: the
        record outlived that re-grasp and excused a foreign body set there for the session).
        """
        held = [o.box.pose.xyz for o in self.store.all() if o.status is TrackStatus.BOUND]
        for oid, placed in list(self._placed.items()):
            t = self.store.get(oid)
            if t is None:
                a = _aabb(placed)
                if any(a[0] <= x <= a[3] and a[1] <= y <= a[4] for x, y, _ in held):
                    log.info("object_tracker.unplaced", object_id=oid, status="retired_regrasped")
                    del self._placed[oid]
                continue
            if t.status is TrackStatus.RELEASED:
                continue
            a, (x, y) = _aabb(placed), t.box.pose.xyz[:2]
            if t.status is TrackStatus.BOUND or not (a[0] <= x <= a[3] and a[1] <= y <= a[4]):
                log.info("object_tracker.unplaced", object_id=oid, status=t.status.value)
                del self._placed[oid]
        return list(self._placed.values())

    def _note_release(self, t: TrackedObject, stamp_ns: int) -> None:
        """Record a let-go track; it is retired with its release window.

        No fit, no confirmation (the session's ``_expire_released`` retires it). A TARGET keeps
        its identity for ``_RECENT_S``: its video mask can re-find it where it fell
        (``_vision_refind``); once per release.
        """
        if t.role is ObjectRole.TARGET and t.object_id not in self._let_go:
            ref = max(
                self._map_cells.get(t.object_id, 0),
                self._map_cells_max.get(t.object_id, 0),
                t.visible_cells,  # a grasp before any plain map pass has only the fit's
            )
            self._let_go[t.object_id] = (t, ref, stamp_ns) if ref > 0 else None
            self._note_placed(t)
        self._cells.pop(t.object_id, None)
        self._map_cells.pop(t.object_id, None)

    # -- role and size plausibility (one place for the mint, refit and vision gates) ---------

    @staticmethod
    def _older(o: TrackedObject, than: str | None) -> bool:
        """Whether track ``o`` was minted before ``than`` (``None``: a mint, every track is).

        Two comparable boxes, one TARGET and one DESTINATION, cannot say which is the impostor;
        the first identity, confirmed on a clear view, wins (yfb2 ep1/ep3: a bowl-sized "banana"
        and the real bowl each blocked the other).
        """
        if than is None:
            return True
        try:
            return int(o.object_id.split(":")[1]) < int(than.split(":")[1])
        except (IndexError, ValueError):
            return True

    def _inside_target(self, box: PlaceRegion | None, than: str | None = None) -> str | None:
        """The live, older TARGET a DESTINATION box would really be, or ``None``.

        Either the target's own size (covers it) or a fit much smaller than the target lying
        mostly on it: a partial mask on it (yf2 ep3: a 4 x 2 cm "plate" on an orange). A
        destination is never inside a target.
        """
        if box is None:
            return None
        a = _aabb(box)
        foot = (a[3] - a[0]) * (a[4] - a[1])
        for o in self.store.all():
            if o.role is not ObjectRole.TARGET or o.status not in _SETTLED:
                continue
            if not self._older(o, than):
                continue
            if covers_destination(box, o.box):
                return o.object_id
            d = _aabb(o.box)
            ix = max(0.0, min(a[3], d[3]) - max(a[0], d[0]))
            iy = max(0.0, min(a[4], d[4]) - max(a[1], d[1]))
            target_foot = (d[3] - d[0]) * (d[4] - d[1])
            if foot > 0 and ix * iy >= 0.6 * foot and foot <= 0.5 * target_foot:
                return o.object_id
        return None

    def _smaller_copy(self, label: str, box: PlaceRegion | None) -> str | None:
        """A live DESTINATION of ``label`` that a new destination fit is a much smaller copy of.

        A scene's destination is already tracked; a partial mask of something else under its
        label (yf3 ep4: a 5 x 21 cm "plate" strip beside the real plate) minted a second one.
        Under half the live one's footprint is refused; a destination re-minted after its track
        was lost (a pushed bowl) is not.
        """
        if box is None:
            return None
        a = _aabb(box)
        foot = (a[3] - a[0]) * (a[4] - a[1])
        for o in self.store.all():
            if (
                o.role is ObjectRole.DESTINATION
                and o.label == label
                and o.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
            ):
                d = _aabb(o.box)
                if foot < 0.5 * (d[3] - d[0]) * (d[4] - d[1]):
                    return o.object_id
        return None

    def _covered_destination(
        self, box: PlaceRegion | None, stamp_ns: int, than: str | None = None
    ) -> str | None:
        """The older DESTINATION (live, or lost within ``_RECENT_S``) a TARGET box would be."""
        if box is None:
            return None
        for o in self.store.all():
            if o.role is not ObjectRole.DESTINATION or o.status is TrackStatus.RELEASED:
                continue
            if not self._older(o, than):
                continue
            if o.status is TrackStatus.LOST and (
                stamp_ns - o.last_confirmed_stamp_ns > int(_RECENT_S * _NS)
            ):
                continue
            if covers_destination(box, o.box):
                return o.object_id
        return None

    def _role_conflict(
        self,
        role: ObjectRole,
        label: str,
        box: PlaceRegion | None,
        stamp_ns: int,
        *,
        than: str | None = None,
        mint: bool = False,
    ) -> str | None:
        """The track a ``role`` box would be an impostor of, or ``None``.

        The one role rule behind every mint, refit and vision box: a TARGET never takes a
        destination's box, a DESTINATION never takes a target's, and a new destination is never
        a much smaller copy of a live one (``mint`` only). Only tracks older than ``than`` count:
        the first identity wins.
        """
        if role is ObjectRole.TARGET:
            return self._covered_destination(box, stamp_ns, than)
        if role is ObjectRole.DESTINATION:
            hit = self._inside_target(box, than)
            return hit or (self._smaller_copy(label, box) if mint else None)
        return None

    def _implausible_box(
        self,
        label: str,
        box: PlaceRegion,
        exclude: str | None = None,
        agg: Callable[[list[float]], float] = median,
    ) -> bool:
        """Whether ``box`` is wider than ``_PLAUSIBLE_EXTENT`` x its label's PEERS' clear extent.

        Peers: every other track of the label (``exclude`` is the box's own track, whose own
        history would vouch for itself: a 32 cm "orange" judged plausible by its own 32 cm
        fit). ``agg`` is ``median`` to judge a track that blocks others (one bad fit cannot
        loosen it) and ``max`` to gate a mint (the largest instance seen admits a larger one of
        the same label). ponytail: a label's first, unseen-size instance is never gated.
        """
        extents = [
            self._clear_extent.get(o.object_id, 2.0 * max(o.box.half_extents[:2]))
            for o in self.store.all()
            if o.label == label
            and o.object_id != exclude
            and o.status is TrackStatus.CONFIRMED
            and not o.provisional
        ] or [v for k, v in self._label_clear_extent.get(label, {}).items() if k != exclude]
        extent = 2.0 * max(box.half_extents[0], box.half_extents[1])
        return bool(extents) and extent > _PLAUSIBLE_EXTENT * agg(extents)

    # -- vision + voxel fusion ---------------------------------------------------------------

    def apply_instance_cells(
        self,
        cells_by_id: dict[str, frozenset[Cell]],
        grid: VoxelLattice,
        stamp_ns: int,
        hands: Sequence[PlaceRegion],
    ) -> int:
        """Fit each track's box to the voxels its video mask labels; return how many moved.

        EXPERIMENT (exp/vision-voxel-fusion; Safety-WG review owed). ``cells_by_id`` comes from
        the instance mask propagator: per track, the occupied cells its SAM2 video mask (robot
        pixels removed) lifted into through depth, each kept while the cell stays occupied.
        Vision says which object a cell is; the map says where. A confirmed, occluded or lost
        TARGET / DESTINATION gets the box ``_vision_fit`` accepts and is re-confirmed; one whose
        cells have not moved is only re-dated. A released target is re-minted
        (``_vision_refind``).
        """
        res = grid.resolution
        moved = 0
        for oid, cells in cells_by_id.items():
            t = self.store.get(oid)
            if t is None or t.status is TrackStatus.RELEASED:
                moved += self._vision_refind(oid, cells, grid, stamp_ns, hands)
                continue
            if t.role not in _CONTACT_ROLES or t.status not in (
                TrackStatus.CONFIRMED,
                TrackStatus.OCCLUDED,
                TrackStatus.LOST,
            ):
                continue
            if stamp_ns < max(t.last_confirmed_stamp_ns, t.last_fit_stamp_ns):
                continue  # the mask lags: a newer fit or confirmation already placed it
            ref = max(self._map_cells.get(oid, 0), self._map_cells_max.get(oid, 0), t.visible_cells)
            fit = self._vision_fit(oid, t, cells, ref, grid, stamp_ns, hands, than=oid)
            if isinstance(fit, str):
                if not fit.startswith("few_cells"):
                    log.info("object_tracker.vision_box_no", object_id=oid, why=fit)
                continue
            box, live, v = fit
            # Unmoved while (nearly) every labelled cell lies in the box grown a cell (the rule
            # `_measure_bound` applies to a fit) AND the map still fills the box (its cover
            # rule): a partly hidden object's visible cells sit off its centre, and a centroid
            # test replaced its box with the visible half's; a moved one's visible half can
            # still lie in the old box, but the spot it left is seen empty.
            ijk = np.asarray(sorted(live), dtype=np.float64)
            centres = np.asarray(grid.origin) + ((ijk + 0.5) * res) @ grid.rotation().T
            outside = int(np.count_nonzero(~_inside(centres, _grown(t.box, res))))
            unmoved = outside < max(_MOVED_MIN_POINTS, _MOVED_MIN_FRACTION * len(live)) and len(
                box_cells(grid, t.box, support_z=t.support_z)
            ) >= cover_need(ref)
            if t.status is TrackStatus.CONFIRMED and unmoved:
                self._cells[oid] = live  # unmoved: keep the fit's box, re-date it on vision
                # Camera-measured inside its box: a fresh fit for the grasp binder too, as
                # ``_redate_if_inside`` re-dates a refused refit (``_fresh_ns`` reads the fit).
                self.store.replace(
                    t.model_copy(
                        update={"last_fit_stamp_ns": stamp_ns, "last_confirmed_stamp_ns": stamp_ns}
                    )
                )
                continue
            shift = float(np.hypot(*(np.asarray(box.pose.xyz[:2]) - t.box.pose.xyz[:2])))
            if shift > res or t.status is not TrackStatus.CONFIRMED:
                log.info(
                    "object_tracker.vision_box", object_id=oid, cells=len(live), ref=ref,
                    status=t.status.value, shift_m=round(shift, 3),
                    was_xyz=[round(c, 3) for c in t.box.pose.xyz],
                    xyz=[round(c, 3) for c in box.pose.xyz],
                )  # fmt: skip
                moved += 1
            self._cells[oid] = live
            self._disagree.pop(oid, None)
            self._disagree_ns.pop(oid, None)
            bottom = box.pose.xyz[2] - box.half_extents[2]
            self.store.replace(
                t.model_copy(
                    update={
                        "box": box,
                        # Hulls were posed for the old box: they go with it.
                        "tight": t.tight if box == t.box else (),
                        "status": TrackStatus.CONFIRMED,
                        "last_fit_stamp_ns": stamp_ns,
                        "last_confirmed_stamp_ns": stamp_ns,
                        "support_z": min(t.support_z, float(bottom)),
                        "occupied_cells": v.occupied or len(live),
                        "visible_cells": v.visible or len(live),
                    }
                )
            )
        return moved

    def _vision_refind(
        self,
        oid: str,
        cells: frozenset[Cell],
        grid: VoxelLattice,
        stamp_ns: int,
        hands: Sequence[PlaceRegion],
    ) -> int:
        """Mint a released target again on the cells its video mask still follows; 1 if minted.

        The mask keeps an object's identity through a slip or a drop, where the map alone
        cannot (yw1 / yv1 ep0: an orange slipped from the jaw and stayed within 3 cm, so its
        cells were not new occupancy; the track retired and the jaw's next touch stopped).
        Within ``_RECENT_S`` of the release, once (``_let_go``), through ``_vision_fit``.
        """
        entry = self._let_go.get(oid)
        if entry is None:
            return 0
        track, ref, since = entry
        if stamp_ns < since:
            return 0  # the mask lags the release: the object may still be in the jaw
        if stamp_ns - since > int(_RECENT_S * _NS):
            self._let_go[oid] = None
            return 0
        fit = self._vision_fit(oid, track, cells, ref, grid, stamp_ns, hands, refind=True)
        if isinstance(fit, str):
            log.info("object_tracker.vision_refind_no", object_id=oid, why=fit)
            return 0
        box, live, v = fit
        fields = {
            k: getattr(track, k)
            for k in type(track).model_fields
            if k not in ("object_id", "epoch", "status")
        }
        fields.update(
            box=box,
            tight=(),  # the released track's hulls were posed for its old box
            # The release support and floor stand (floor_z >= support_z): a mask that labels
            # only the upper part of the object must not lift the support above the floor.
            support_z=min(track.support_z, float(box.pose.xyz[2] - box.half_extents[2])),
            last_confirmed_stamp_ns=stamp_ns,
            last_fit_stamp_ns=stamp_ns,
            visible_cells=v.visible or len(live),
            occupied_cells=v.occupied or len(live),
        )
        new = self.store.mint(**fields)
        self.store.replace(new.model_copy(update={"status": TrackStatus.CONFIRMED}))
        self._cells[new.object_id] = live
        self._map_cells[new.object_id] = len(live)
        self._let_go[oid] = None
        self._placed.pop(oid, None)  # the old id is gone; the new one rests where it is now
        self._note_placed(new)
        log.info(
            "object_tracker.vision_refound", object_id=new.object_id, was=oid, cells=len(live),
            ref=ref, was_xyz=[round(c, 3) for c in track.box.pose.xyz],
            xyz=[round(c, 3) for c in box.pose.xyz],
        )  # fmt: skip
        return 1

    def _vision_fit(
        self,
        oid: str,
        track: TrackedObject,
        cells: frozenset[Cell],
        ref: int,
        grid: VoxelLattice,
        stamp_ns: int,
        hands: Sequence[PlaceRegion],
        *,
        than: str | None = None,
        refind: bool = False,
    ) -> tuple[PlaceRegion, frozenset[Cell], CoverVerdict] | str:
        """The box ``track``'s labelled ``cells`` give it, or why not (a short reason).

        In order: the cells still occupied and above the support layer (mask edges bleed onto
        it); at least ``_vision_need(ref)``; their box, stood on the support when they sit near
        it and widened to the object's clear size toward the hand hiding the rest; no wider than
        ``_VISION_MAX_EXTENT`` x its clear extent; mostly in no other plausible track of the same
        role; no role conflict (judged on the cells' own extent: a banana's cells in the bowl
        stand 60 % of its height, the box grown to the table stood the bowl's full height,
        ywb1 ep5); and the cover verdict. Returns the box, the cells, the verdict and the cells'
        x/y centroid.
        """
        sx, sy, sz = grid.size
        occ, res, z0 = grid.occupancy, grid.resolution, grid.origin[2]
        occupied = [
            c
            for c in cells
            if 0 <= c[0] < sx
            and 0 <= c[1] < sy
            and 0 <= c[2] < sz
            and occ[c[0] + sx * (c[1] + sy * c[2])]
        ]
        support = track.support_z
        if refind and occupied:
            # A released track keeps its in-hand box and support (7.5 cm up: A/B ab1b_po seeds
            # 2/3, every re-find of a slipped orange kept 0 of its 7-33 labelled cells). It
            # rests on whatever is under its lowest labelled layer now.
            support = z0 + min(c[2] for c in occupied) * res
        live = frozenset(c for c in occupied if z0 + (c[2] + 0.5) * res > support + 0.5 * res)
        if len(live) < _vision_need(ref):
            return f"few_cells {len(live)}<{_vision_need(ref)} labelled={len(cells)} ref={ref}"
        ijk = np.asarray(sorted(live), dtype=np.float64)
        centres = np.asarray(grid.origin) + ((ijk + 0.5) * res) @ grid.rotation().T
        lo = centres.min(axis=0) - 0.5 * res
        hi = centres.max(axis=0) + 0.5 * res
        cells_box = _box_between(track.box, lo.copy(), hi.copy())
        if lo[2] - support < 2.0 * max(track.box.half_extents[2], res):
            lo[2] = min(lo[2], support)  # masks see the top: it stands on its support
        _grow_to_size(lo, hi, self._clear_size.get(oid), hands)
        prior = self._extent_prior(oid, track.label)
        limit = (
            _VISION_MAX_EXTENT * prior[0]
            if prior
            else 3.0 * max(track.box.half_extents[0], track.box.half_extents[1])
        )
        extent = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        if extent > limit + res:
            return f"oversize {extent:.3f}>{limit:.3f}"
        same = [
            o.box
            for o in self.store.all()
            if o.object_id != oid
            and o.status in _SETTLED
            and o.role is track.role
            and not self._implausible_box(o.label, o.box, o.object_id)
        ]
        if same:
            inside = np.any([_inside(centres, b) for b in same], axis=0)
            if int(np.count_nonzero(inside)) * 2 > len(live):
                return "other_track"
        if self._role_conflict(track.role, track.label, cells_box, stamp_ns, than=than):
            return "role_conflict"
        box = _box_between(track.box, lo, hi)
        others = [o.box for o in self.store.all() if o.object_id != oid and o.status in _SETTLED]
        v = cover_verdict(
            grid,
            live,
            box,
            support_z=float(lo[2]) if refind else track.support_z,
            own=hands,
            others=others,
            static=self._static_keys(grid),
        )
        # The refit rule (``_refuse_refit``): a box the map sees as two bodies never confirms
        # (it took in a neighbour); a second body in the hand's own space or a box the camera
        # cannot see stays as before.
        if v.kind in ("map_disagrees", "map_split"):
            return f"verdict {v.kind}: {v.why}"
        return box, live, v

    def _refresh_evidence(self, object_id: str, cells: frozenset[Cell]) -> None:
        """An ignored refit still measured where the object's surface is now: confirm on it.

        The refit leaves the box alone (it stays the conservative geometry), but its surface,
        inside that box, replaces the cells the cover test expects (2026-10-21 decision;
        a banana pushed 9 cm inside its 26 cm first box left the first fit's cells half
        empty, 40 of the 42 needed, and its refits through the hand were ignored, so it was
        never re-confirmed and dropped out mid-push).
        """
        if len(cells) >= 2:  # noqa: PLR2004  # reason: the cover rule's own floor
            self._cells[object_id] = cells

    @staticmethod
    def _contact_hold(
        t: TrackedObject, kind: str, why: str, hands: Sequence[PlaceRegion], margin_m: float
    ) -> bool:
        """Hold a refuted TARGET / DESTINATION track while a hand is at it, never re-stamped.

        A hand at the target fragments its map (fingers bisect it, a push leaves its old
        cells), which ``map_split`` / ``map_disagrees`` read as a foreign body
        or a vanished object. Held, not refuted (2026-10-21 decision): the kernel stops
        exempting the box once its confirmation is older than ``grasp_region_max_age_s``
        (2 s in deploy sim), exactly as a split inside the hand's own space
        (``map_split_own``) ages out. A camera refit re-stamps it. Safety-WG row owed.
        """
        if (
            kind in ("map_split", "map_disagrees")
            and t.role in _CONTACT_ROLES
            and t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED)
            and hand_touches(t.box, hands, margin_m)
        ):
            log.info("object_tracker.contact_hold", object_id=t.object_id, kind=kind, why=why)
            return True
        return False

    def set_static_geometry(self, centres: NDArray[np.float64]) -> None:
        """Declare static geometry for the ONE-BODY test only.

        ``centres`` are cell centres in the grid frame, one per cell the octomap bridge
        marks. A body wholly inside them, padded by ``static_pad_cells``, is the declared wall,
        not a foreign object (#418 restock: the target's overshooting fit reached the source
        tote's wall and split). The fit and the cover gate still use every measured point.
        """
        self._static_centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
        self._static_cache = None

    def _static_keys(self, grid: VoxelLattice) -> frozenset[Cell]:
        """The declared static cells on ``grid``'s lattice (cached per lattice)."""
        if not len(self._static_centres):
            return frozenset()
        key = (grid.origin, grid.resolution, grid.size, grid.orientation_xyzw)
        if self._static_cache is None or self._static_cache[0] != key:
            keys = static_cell_keys(
                grid, self._static_centres, pad_cells=self.params.static_pad_cells
            )
            self._static_cache = (key, keys)
        return self._static_cache[1]

    def _provisional_oversize(self, object_id: str | None, label: str, fit: ContainingFit) -> bool:
        """Refuse (``True``) a PROVISIONAL fit several times the object's known clear size.

        A fit through the robot that is far past what a clear fit of this object or label ever
        showed swallowed something else: a refit keeps the previous box (an exemption only
        shrinks), a mint (``object_id`` ``None``) mints nothing.
        """
        if not fit.provisional or fit.box is None:
            return False
        prior = self._extent_prior(object_id, label)
        extent = 2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1])
        factor = self.params.provisional_oversize_factor
        if prior is None or not provisional_oversize(extent, prior[0], factor):
            return False
        log.info("object_tracker.provisional_oversize", object_id=object_id,
                 extent_m=round(extent, 3), prior_m=round(prior[0], 3),
                 limit_m=round(factor * prior[0], 3), source=prior[1])  # fmt: skip
        return True

    def oversize_provisional(self, track: TrackedObject) -> bool:
        """A PROVISIONAL track whose box is past the oversize factor of its clear prior.

        The size rule ``_provisional_oversize`` holds a fit to, applied to the track as it
        stands (moved refits are exempt from it and grow the box), for the bind to drop it.
        """
        if not track.provisional:
            return False
        prior = self._extent_prior(track.object_id, track.label)
        extent = 2.0 * max(track.box.half_extents[0], track.box.half_extents[1])
        factor = self.params.provisional_oversize_factor
        return prior is not None and provisional_oversize(extent, prior[0], factor)

    def _extent_prior(self, object_id: str | None, label: str) -> tuple[float, str] | None:
        """The longest horizontal extent a CLEAR fit of this object or label has shown, or ``None``.

        Only clear (non-provisional) evidence: the track's own clear history, else the median
        over the clear CONFIRMED same-label tracks, else the median over every (since retired)
        track of the label. The median, not the largest: one bad clear fit (two touching oranges
        fitted as one 19-32 cm "orange", yw1 ep8) would otherwise loosen every same-label size
        gate for the rest of the session. Never a per-label constant; ``None`` blocks nothing.
        """
        if object_id is not None and object_id in self._clear_extent:
            return self._clear_extent[object_id], "own_clear_fit"
        extents = [
            # A same-label track's largest clear fit, not its current (possibly smaller) box.
            self._clear_extent.get(
                o.object_id, 2.0 * max(o.box.half_extents[0], o.box.half_extents[1])
            )
            for o in self.store.all()
            if o.label == label
            and o.object_id != object_id
            and o.status is TrackStatus.CONFIRMED
            and not o.provisional
        ]
        if extents:
            return median(extents), "same_label_tracks"
        if hist := self._label_clear_extent.get(label):
            return median(hist.values()), "label_history"
        return None

    def _measured_size(self, object_id: str | None, label: str) -> tuple[float, str] | None:
        """How big a provisional track's object is, and where that came from, or ``None``.

        In order: the track's own largest clear fit (its smaller horizontal extent); else the
        largest such extent among CONFIRMED clear tracks of the same label (#418 PickOrange:
        orange_1 / orange_2 size orange_3 under the jaw); else unknown (no growth). Never a
        per-label constant.
        """
        if object_id is not None and object_id in self._clear_size:
            return self._clear_size[object_id], "own_clear_fit"
        sizes = [
            2.0 * min(o.box.half_extents[0], o.box.half_extents[1])
            for o in self.store.all()
            if o.label == label
            and o.object_id != object_id
            and o.status is TrackStatus.CONFIRMED
            and not o.provisional
        ]
        return (max(sizes), "same_label_tracks") if sizes else None

    def _grow_hidden(  # reason: the fit step's own state, passed through
        self,
        fit: ContainingFit,
        object_id: str | None,
        label: str,
        frame: CameraFrame,
        grid: VoxelLattice,
        support_z: float,
    ) -> ContainingFit:
        """``fit`` with its box grown into the robot-hidden side to the measured size.

        PROVISIONAL fits only (K2 exemption geometry; never a payload, K3 refuses it).
        Issue #418 OT-6 / OT-14 amendment (Safety-WG review pending).
        """
        measured = self._measured_size(object_id, label)
        if measured is None or fit.box is None or frame.t_base_from_cam is None:
            return fit
        others = [
            o.box
            for o in self.store.all()
            if o.object_id != object_id and o.status not in (TrackStatus.LOST, TrackStatus.RELEASED)
        ]
        box, grown = grow_hidden_side(
            fit.box,
            measured[0],
            depth_m=frame.depth_m,
            robot_mask=frame.robot_mask,
            intrinsics=frame.intrinsics,
            t_base_from_cam=frame.t_base_from_cam,
            grid=grid,
            support_z=support_z,
            others=others,
        )
        if not grown:
            return fit
        log.info("object_tracker.hidden_side_grown", object_id=object_id, label=label,
                 size_m=round(measured[0], 4), source=measured[1], grown=grown)  # fmt: skip
        return replace(fit, box=box)

    def _may_confirm(  # reason: the confirm step's own state, passed through
        self,
        t: TrackedObject,
        grid: VoxelLattice,
        cells: frozenset[Cell],
        others: Sequence[PlaceRegion],
        occupied: int,
        stamp_ns: int,
    ) -> bool:
        """May TENTATIVE ``t``, whose cover just held on the grid at ``stamp_ns``, confirm?

        Its evidence must be its own (OT-13: #418 r6po_dbg4's ghost trk:5 was one mask over
        two tracked oranges), and a low-score prompt's track needs
        ``low_score_confirm_grids`` consecutive confirming grids.
        """
        if own_cover(grid, cells, others) < cover_need(occupied):
            self._refuse(f"{t.label}@{t.object_id}", "cover_already_tracked", occupied=occupied)
            self._low_score_run.pop(t.object_id, None)
            return False
        score = self._low_score.get(t.object_id)
        if score is None:
            return True
        last, n = self._low_score_run.get(t.object_id, (-1, 0))
        n = n if stamp_ns == last else n + 1
        self._low_score_run[t.object_id] = (stamp_ns, n)
        if n < self.params.low_score_confirm_grids:
            return False
        self._low_score.pop(t.object_id)
        self._low_score_run.pop(t.object_id)
        log.info("object_tracker.low_score_prompt_confirmed", object_id=t.object_id,
                 label=t.label, score=score, grids=n)  # fmt: skip
        return True

    def _other_boxes(self, centroid: tuple[float, ...]) -> list[PlaceRegion]:
        """Settled tracks' boxes a new fit at ``centroid`` is not a refit of."""
        c = np.asarray(centroid[:3], dtype=np.float64)
        out = []
        for o in self.store.all():
            if o.status not in _SETTLED:
                continue
            t = homogeneous_from_quat_xyz(o.box.pose.xyz, o.box.pose.quat_xyzw)
            if not np.all(np.abs((c - t[:3, 3]) @ t[:3, :3]) <= np.asarray(o.box.half_extents)):
                out.append(o.box)
        return out

    def _visibility(self, frame: CameraFrame) -> None:
        assert frame.t_base_from_cam is not None
        for t in self.store.all():
            if t.status not in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED):
                continue
            vis = visible_fraction(
                t.box, frame.depth_m, frame.robot_mask, frame.intrinsics, frame.t_base_from_cam
            )
            if vis is None:
                continue
            # Hysteresis: back to CONFIRMED only past ``theta_vis + _VIS_HYSTERESIS``. A hand
            # hovering over the target held it near the threshold and flipped it every frame
            # (restock scene 08 2026-11-01: OCCLUDED -> CONFIRMED in the second of the grasp, so
            # the map's confirmation stopped counting and the bind went untracked, fit 1.5 s old).
            occluded = t.status is TrackStatus.OCCLUDED
            hidden = vis < self.params.theta_vis + (_VIS_HYSTERESIS if occluded else 0.0)
            if hidden != occluded:
                new = TrackStatus.OCCLUDED if hidden else TrackStatus.CONFIRMED
                self.store.replace(t.model_copy(update={"status": new}))

    # -- step 7: candidate prompts -----------------------------------------------------------

    def _candidates(  # noqa: PLR0912, PLR0915  # reason: one flat pass, pending/queries/tracks
        self,
        frame: CameraFrame,
        grid: VoxelLattice | None,
        hands: Sequence[PlaceRegion] = (),
    ) -> list[TrackerRequest]:
        assert frame.t_base_from_cam is not None
        t_cam = frame.t_base_from_cam
        cam_from_base = np.asarray(np.linalg.inv(t_cam), dtype=np.float64)
        cam, stamp = frame.camera, frame.stamp_ns
        out: list[TrackerRequest] = []
        max_age = int(self.params.max_age_s * _NS)
        self._pending = [r for r in self._pending if self._now_ns - r.stamp_ns <= max_age]
        for r in self._pending:
            q = self.queries.get(*r.query_key) if r.query_key else None
            if r.camera == frame.camera and q is not None:
                prio = PRIORITY_QUERY if self._searching(q) else PRIORITY_REFRESH
                out.append(replace(r, stamp_ns=stamp, priority=prio))
        tracks = self.store.all()
        for q in self.queries.due(now_ns=self._now_ns):
            prio = PRIORITY_QUERY if self._searching(q) else PRIORITY_REFRESH
            q_served = q.last_served_ns + (
                _CONTEXT_DEFER_NS if q.query.role is ObjectRole.CONTEXT else 0
            )
            # A released match's box, grown down onto its support (the seed's raised tops
            # stand above the column's lowest): the object is first looked for where it was
            # let go (16 stops on a released banana the locator, boxing the table, never
            # re-found).
            seed = _grown(q.prior, PRIOR_MARGIN_M) if q.prior is not None else q.query.seed
            base = TrackerRequest(
                "locate_in_view", cam, stamp, prio, q_served, q.query.label, q.key,
                roi_norm_xyxy=self._locate_roi(seed or self.search_region, frame),
                min_score=self.params.locate_min_score,
                prompt=", ".join((q.query.label, *q.query.synonyms)),
            )  # fmt: skip
            # A CONTEXT query's seed only crops the locator: a point prompt on a raised top
            # in a cluttered tote masked its walls and slivers as "carton";
            # the detector's box prompt is the class evidence.
            if (
                seed is not None
                and grid is not None
                and seed.frame_id == grid.frame_id
                and q.query.role is not ObjectRole.CONTEXT
            ):
                skip = self._prompted.setdefault(q.key, set())
                # A CONTEXT track never hides its object from a named query: it only marks
                # something to avoid, and the named query's fit retires it
                # (``_retire_context_duplicates``). Restock, arms out of view (pro#51,
                # q5090 2026-11-10): the "carton" query boxed the target first, every
                # "stock object" prompt skipped its cells and masked the tote wall behind it,
                # and the grasp found the carton's box 4.6-5.7 s stale.
                hit = instance_prompt(
                    grid,
                    seed,
                    [
                        t.box
                        for t in tracks
                        if t.status is not TrackStatus.LOST and t.role is not ObjectRole.CONTEXT
                    ],
                    near_xy=(seed.pose.xyz[0], seed.pose.xyz[1]),
                    skip=skip,
                )
                uv = project_point(hit[0], cam_from_base, frame.intrinsics) if hit else None
                if hit is not None and uv is not None:
                    # The seed's image rectangle rides along as a bound on the mask pick,
                    # not as SAM's box prompt (a seed is wider than the instance, and SAM
                    # segments what fills a box): SAM's "whole" of a point on one packed
                    # carton is the tote's contents, which the bound refuses.
                    rect = project_box(seed, frame.intrinsics, t_cam)
                    within = (
                        None
                        if rect is None
                        else (
                            float(rect[0][0]),
                            float(rect[0][1]),
                            float(rect[0][2]),
                            float(rect[0][3]),
                        )
                    )
                    seg = replace(base, service="segment_in_view", point_uv=uv)
                    out.append(replace(seg, within_xyxy=within, seed_cell=hit[1]))
                    continue
                # Every seed cell tried: the detector's turn. The cells start over once a
                # locate is actually served (``_mark_served``), not merely offered here; a
                # released match's prior is spent.
                if q.prior is not None:
                    q.prior = None
                    log.info("object_tracker.query_prior_spent", key=q.key)
            out.append(base)
        for t in tracks:
            served = max(t.last_fit_stamp_ns, self._served.get(t.object_id, -1))
            # A BOUND track is prompted where its payload rides now (the session's
            # `bound_prompts`, the hand's kinematics), not at its last measured box: prompted at
            # the grasp site the segmenter masked the table (~100 in-hand refits refused
            # half_extent_cap). The prompt only; the fit is the camera's.
            where = self.bound_prompts.get(t.object_id, t.box)
            if t.status is TrackStatus.RELEASED:
                continue  # never followed: the target query finds the object afresh
            if self._disagree.get(t.object_id) and t.status is not TrackStatus.OCCLUDED:
                prio, xyz = PRIORITY_DISAGREE, self._centroid.get(t.object_id, t.box.pose.xyz)
            elif t.status is TrackStatus.BOUND:
                prio, xyz = PRIORITY_BOUND, where.pose.xyz
            elif t.status is TrackStatus.CONFIRMED:
                prio, xyz = PRIORITY_REFRESH, self._centroid.get(t.object_id, t.box.pose.xyz)
                # A hand at the target: its refresh goes first among refreshes, so the grasp
                # binder has a fit under max_age_s (sharing the refresh slots with the bowl, the
                # banana's fit was 2.1-2.5 s old at the grasp). ponytail: ordering only, the query
                # alternation stays.
                at_hand = (
                    grid is not None
                    and t.role in _CONTACT_ROLES
                    and hand_touches(t.box, hands, grid.resolution)
                )
                served = -1 if at_hand else served
                if t.role is ObjectRole.CONTEXT:
                    # A neighbour's refresh yields to the target's and destination's (three
                    # tote cartons shared the slot round-robin and the
                    # target's fit was 1.5 s old at the grasp; the bind went untracked).
                    served += _CONTEXT_DEFER_NS
            else:
                continue  # OCCLUDED is the map's job; TENTATIVE/SPLIT/LOST wait for queries
            uv = project_point(xyz, cam_from_base, frame.intrinsics)
            box = project_box(where, frame.intrinsics, t_cam)
            if uv is None and box is None:
                continue
            xyxy = (
                None
                if box is None
                else (float(box[0][0]), float(box[0][1]), float(box[0][2]), float(box[0][3]))
            )
            out.append(
                TrackerRequest(
                    "segment_in_view", cam, stamp, prio, served, t.label,
                    object_id=t.object_id, point_uv=uv, box_xyxy=xyxy,
                )
            )  # fmt: skip
        return out

    def _searching(self, q: StandingQuery) -> bool:
        """Whether ``q`` still competes at query priority.

        Unresolved, holding an unspent release prior, or a ``count: "all"`` query inside its
        first ``search_start_s`` of source time: one instance resolves it, but the others are
        still to be grounded, and at refresh priority an unresolved neighbour query (a plate)
        would starve it.
        """
        if q.status is not ObjectQueryStatus.RESOLVED:
            return True
        if q.prior is not None:
            # A released match not found afresh yet: another match (a stale provisional
            # duplicate) keeps the query RESOLVED, and at refresh priority the hand's own
            # refresh (the bowl it just placed over, served first) starved it (pro#51 replay,
            # 2026-11-05: the dropped banana was never re-fitted; the regrasp stopped on it).
            return True
        window = int(self.params.search_start_s * _NS)
        return q.query.count == "all" and self._now_ns - q.since_ns < window

    def _mark_served(self, req: TrackerRequest) -> None:
        if req.object_id is not None:
            self._served[req.object_id] = self._now_ns
        if req.query_key is not None and req.seed_cell is not None:
            # The seed's next prompt is another raised top; the detector gets its turn once
            # every one was tried, and the cells start over after it.
            self._prompted.setdefault(req.query_key, set()).add(req.seed_cell)
        elif req.query_key is not None and req.service == "locate_in_view":
            self._prompted.pop(req.query_key, None)
        if req.query_key is not None and (q := self.queries.get(*req.query_key)) is not None:
            q.last_served_ns = self._now_ns
            # A detector box's prompt, as a box or (re-prompted beside the robot) as the
            # bound of a point: served once, then dropped from the pending prompts.
            served_box = req.box_xyxy or req.within_xyxy
            if req.service == "segment_in_view" and served_box is not None:
                self._pending = [
                    r for r in self._pending
                    if (r.query_key, r.box_xyxy or r.within_xyxy) != (req.query_key, served_box)
                ]  # fmt: skip
            if (
                req.service == "locate_in_view"
                and q.status is ObjectQueryStatus.RESOLVED
                and q.query.count == "all"
                and self._now_ns - self._research_logged.get(q.key, -(10**18)) >= 5 * _NS
            ):
                # A resolved count-all query keeps detecting (more instances may appear or
                # come out from behind the arm); say so at INFO, at most every 5 s.
                self._research_logged[q.key] = self._now_ns
                log.info("object_tracker.count_all_research", label=q.query.label,
                         camera=req.camera, goal_id=q.goal_id)  # fmt: skip


#: Pixels within which a free region counts as beside the robot (past the uncertain band).
_BESIDE_PX = _T.beside_px
#: Fewest pixels a region beside the robot needs to be prompted on.
_BESIDE_MIN_PX = _T.beside_min_px


def _most_hidden(region: NDArray[np.bool_], robot: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
    """The 4-connected component of ``region`` most of whose pixels lie beside ``robot``.

    Beside = within ``_BESIDE_PX`` (Chebyshev) of a robot pixel; components under
    ``_BESIDE_MIN_PX`` pixels or with none beside are ignored; ``None`` when nothing is left.
    What the robot half hides is mostly beside it; a neighbour merely touching its band is not.
    """
    h, w = region.shape
    k = _BESIDE_PX
    near = np.zeros_like(robot)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            near[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] |= robot[
                max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
            ]
    label = np.zeros(region.shape, dtype=np.int32)
    best, best_score, n = 0, 0.0, 0
    for sy, sx in zip(*np.nonzero(region), strict=True):
        if label[sy, sx]:
            continue
        n += 1
        stack, size, beside = [(int(sy), int(sx))], 0, 0
        label[sy, sx] = n
        while stack:
            y, x = stack.pop()
            size += 1
            beside += int(near[y, x])
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and region[ny, nx] and not label[ny, nx]:
                    label[ny, nx] = n
                    stack.append((ny, nx))
        score = beside / size
        if size >= _BESIDE_MIN_PX and score > best_score:
            best, best_score = n, score
    return label == best if best else None
