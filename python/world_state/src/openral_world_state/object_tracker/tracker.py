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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
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
from pydantic import BaseModel, ConfigDict, Field

from openral_world_state.object_tracker.catalog import CatalogDeclarations
from openral_world_state.object_tracker.cover import (
    Cell,
    _inside,
    box_cells,
    cells_behind_robot,
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

__all__ = ["CameraFrame", "ObjectTracker", "TrackerOutput", "TrackerParams", "TrackerResult"]

log = structlog.get_logger(__name__)

# A refit "moved" its track when this many measured surface points lie more than one cell
# outside the stored box (max of the two): the box provably no longer contains the object.
_MOVED_MIN_POINTS = 3
_MOVED_MIN_FRACTION = 0.05


#: Source time a CONTEXT track's refresh, or a CONTEXT query's locate, counts as served
#: later than it was: the target and destination go first. *Calibration point.*
_CONTEXT_DEFER_NS = 2_000_000_000
#: Margin a released match's box is grown by as its query's seed: down onto the support it
#: fell to, and around where it rolled. *Calibration point.*
PRIOR_MARGIN_M = 0.05


def _grown(box: PlaceRegion, margin_m: float) -> PlaceRegion:
    """``box`` with every half-extent grown by ``margin_m``."""
    return box.model_copy(update={"half_extents": tuple(h + margin_m for h in box.half_extents)})


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

    theta_vis: float = Field(default=0.5, gt=0.0, le=1.0)
    theta_vis_provisional: float = Field(default=0.25, gt=0.0, le=1.0)
    continuous_locate_s: float = Field(default=2.0, gt=0.0)
    locate_min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    low_score_confirm_grids: int = Field(default=3, ge=1)
    static_pad_cells: int = Field(default=1, ge=0)
    provisional_oversize_factor: float = Field(default=2.0, ge=1.0)
    max_age_s: float = Field(default=2.0, gt=0.0)
    lost_s: float = Field(default=6.0, gt=0.0)
    keep_lost_s: float = Field(default=30.0, gt=0.0)
    disagree_grids: int = Field(default=3, ge=1)
    segment_hz: float = Field(default=2.0, gt=0.0)
    segment_budget_wall_s: float = Field(default=5.0, gt=0.0)
    search_start_s: float = Field(default=5.0, ge=0.0)
    search_max_s: float = Field(default=60.0, gt=0.0)
    support_min_cells: int = Field(default=4, ge=1)
    support_probe_m: float = Field(default=0.06, gt=0.0)
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
REFUSAL_LOG_PERIOD_S = 5.0
#: Source seconds within which fits of one track from different cameras are combined
#: (``intersect_views``): the object must not have moved between them. *Calibration point.*
VIEW_WINDOW_S = 0.5
#: A locate's crop: the projected search region grown by this fraction of its size on
#: every side (a seed or workspace box need not be tight in the image). *Calibration point.*
LOCATE_ROI_PAD = 0.25
#: A detector box more than this fraction ROBOT pixels is re-prompted on what it hides.
ROBOT_BOX_FRACTION = 0.5
#: Negative point prompts placed on the robot in such a box.
ROBOT_BOX_NEGATIVES = 3
#: A prompt box scored under this grounds a track that is logged when it confirms.
LOW_PROMPT_SCORE = 0.3
#: How far under the voxel-measured support top the camera's own support returns may lie and
#: still read as ON the layer boundary (``ObjectTracker._support_seen``). Sim depth is exact;
#: a real camera's noise is larger. *Calibration point.*
_SUPPORT_SEEN_TOL_M = 0.001
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
_REANCHOR_MAX_RATIO = 2.0


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
            self._map_cells, self._served, self._clear_size, self._clear_extent, self._low_score,
            self._low_score_run,
        ):  # fmt: skip
            for oid in [k for k in per_track if k not in live]:
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
        if hi - lo > 0.25 * res or seen < top - _SUPPORT_SEEN_TOL_M:
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
            track = self._mint_track(req, fit, cells, frame, grid, support_z=support_z, floor=floor)
            if track is None:
                return None
        else:
            old = self.store.get(oid)
            assert old is not None
            if old.status is TrackStatus.BOUND:
                return self._measure_bound(old, fit, frame)
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
        self._commit_fit(track, fit, cells, centroid, grid, grid_stamp_ns, hands, frame)
        return None

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
        self, old: TrackedObject, fit: ContainingFit, frame: CameraFrame
    ) -> tuple[str, ContainingFit]:
        """A BOUND track's refit: observed, and its box follows the object in the hand."""
        oid = old.object_id
        self._observe_catalog(old, fit, frame)
        self._observe_surface(old, fit, frame)
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

    def _refuse_refit(
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
        if old.status is TrackStatus.OCCLUDED and not moved:
            self._refresh_evidence(oid, cells)
            log.info("object_tracker.occluded_refit_ignored", object_id=oid)
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
                log.info("object_tracker.partial_refit_ignored", object_id=oid,
                         uncovered=left, cells=len(prior),
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
        return False

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
                "provisional": fit.provisional,
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
        if not fit.provisional and fit.box is not None:
            size = 2.0 * min(fit.box.half_extents[0], fit.box.half_extents[1])
            self._clear_size[track.object_id] = max(
                self._clear_size.get(track.object_id, 0.0), size
            )
            self._clear_extent[track.object_id] = max(
                self._clear_extent.get(track.object_id, 0.0),
                2.0 * max(fit.box.half_extents[0], fit.box.half_extents[1]),
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
            # Let go, not followed: no fit, no confirmation, retired with its release window
            # (the session's `_expire_released`; a released box left where the object let go
            # explained none of where it fell or rolled).
            self._cells.pop(t.object_id, None)
            self._map_cells.pop(t.object_id, None)
            return
        cells = self._cells.get(t.object_id)
        if cells is None or t.status in (TrackStatus.BOUND, TrackStatus.LOST):
            return
        others = [
            o.box for o in self.store.all() if o.object_id != t.object_id and o.status in _SETTLED
        ]
        # Visibility-aware cover (2026-10-21 decision; Isaac baseline: a successful grasp keeps
        # the fingers on the banana for 255-466 steps): an expected cell the robot hides is no
        # evidence the object is gone. While the hand is at the target or destination, a second
        # body inside its box is the object fragmenting under contact (a thin banana's own
        # 6-cell piece, 5 cm from the fingers; the bowl pushed 99 mm), so it
        # does not block a confirmation the visible cover still has to earn. Safety-WG review
        # owed.
        hidden = (
            cells_behind_robot(
                grid,
                cells,
                frame.depth_m,
                frame.robot_mask,
                frame.intrinsics,
                frame.t_base_from_cam,
            )
            if frame is not None
            and frame.robot_mask is not None
            and frame.t_base_from_cam is not None
            else frozenset()
        )
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
        ref = self._map_cells.get(t.object_id, v.visible if v.kind == "map_disagrees" else 0)
        if contact and v.kind in ("map_disagrees", "hidden") and ref:
            # The hand pushes the object inside its box (a banana pushed 66 mm held 12 of its
            # 26 expected cells) or hides it (3-7 cells in view while the map held 33-52 in the
            # box), and the hold ages it out mid-grasp. The
            # object is where the map holds it in the box: re-anchored on those cells when
            # they number half to `_REANCHOR_MAX_RATIO` times what it filled at its last plain
            # pass (2026-10-21 decisions; the kernel's own in-box cover test; a push out of
            # the box still fails, more is the box holding something else). Safety-WG review
            # and hazard row owed.
            body = box_cells(grid, t.box, support_z=t.support_z, exclude=exclude)
            if max(2, math.ceil(0.5 * ref)) <= len(body) <= _REANCHOR_MAX_RATIO * ref:
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
                    v = moved
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

    @property
    def static_centres(self) -> NDArray[np.float64]:
        """The declared static cell centres (grid frame), as ``set_static_geometry`` set them."""
        return self._static_centres

    def set_static_geometry(self, centres: NDArray[np.float64]) -> None:
        """Declare static geometry for the ONE-BODY test and the untracked payload's floor cut.

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

    def _provisional_oversize(self, object_id: str, label: str, fit: ContainingFit) -> bool:
        """Refuse (``True``) a PROVISIONAL refit several times the object's known clear size.

        A refit through the robot that is far past what a clear fit of this object or label ever
        showed swallowed something else: the previous box stays (an exemption only shrinks).
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

    def _extent_prior(self, object_id: str | None, label: str) -> tuple[float, str] | None:
        """The longest horizontal extent a CLEAR fit of this object or label has shown, or ``None``.

        Only clear (non-provisional) evidence: the track's own clear history, else the clear
        CONFIRMED same-label tracks. Never a per-label constant; ``None`` blocks nothing.
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
        return (max(extents), "same_label_tracks") if extents else None

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
        if own_cover(grid, cells, others) < max(2, math.ceil(0.5 * occupied)):
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
            hidden = vis < self.params.theta_vis
            if hidden != (t.status is TrackStatus.OCCLUDED):
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
                hit = instance_prompt(
                    grid,
                    seed,
                    [t.box for t in tracks if t.status is not TrackStatus.LOST],
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

        Unresolved, or a ``count: "all"`` query inside its first ``search_start_s`` of
        source time: one instance resolves it, but the others are still to be grounded,
        and at refresh priority an unresolved neighbour query (a plate) would starve it.
        """
        if q.status is not ObjectQueryStatus.RESOLVED:
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
_BESIDE_PX = 8
#: Fewest pixels a region beside the robot needs to be prompted on.
_BESIDE_MIN_PX = 10


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
