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

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    FitKind,
    IntrinsicsPinhole,
    ObjectQuery,
    ObjectQueryStatus,
    PlaceRegion,
    TrackedObject,
    TrackedObjectSet,
    TrackStatus,
)
from openral_core.object_primitives import instance_prompt
from openral_core.perception_geometry import (
    VoxelLattice,
    backproject_masked_depth,
    project_point,
    support_top_from_voxels,
)
from pydantic import BaseModel, ConfigDict, Field

from openral_world_state.object_tracker.cover import Cell, cover_verdict, surface_cells
from openral_world_state.object_tracker.fit import (
    ROBOT_PIXEL,
    ContainingFit,
    FitParams,
    containing_fit,
    project_box,
    visible_fraction,
)
from openral_world_state.object_tracker.grounding import QueryBook, QueryEscalation
from openral_world_state.object_tracker.scheduler import (
    PRIORITY_BOUND,
    PRIORITY_DISAGREE,
    PRIORITY_QUERY,
    PRIORITY_REFRESH,
    PRIORITY_RELEASED,
    Scheduler,
    TrackerRequest,
)
from openral_world_state.object_tracker.tracks import TrackStore, age_tracks, associate

__all__ = ["CameraFrame", "ObjectTracker", "TrackerOutput", "TrackerParams", "TrackerResult"]

log = structlog.get_logger(__name__)

_NS = 1_000_000_000


class TrackerParams(BaseModel):
    """Tracker calibration. Every value is a *calibration point* (CLAUDE.md §1.2).

    Attributes:
        theta_vis: Visible fraction below which an object is OCCLUDED (and never fitted).
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


#: ``_apply_mask``: try the next (smaller) candidate.
_NEXT = object()


@dataclass(frozen=True)
class TrackerResult:
    """The answer to an issued ``TrackerRequest`` (by ``request_id``).

    Attributes:
        request_id: The request it answers.
        mask: ``SegmentInView``'s mask (``None``: no object at the prompt).
        boxes_xyxy: ``LocateInView``'s detections, pixels.
        alternatives: Smaller qualifying candidates, largest first, each tried in turn
            when the one before it cannot be fitted or splits on the map (a packed
            carton's "part" candidate is several cartons).
    """

    request_id: int
    mask: NDArray[np.bool_] | None = None
    boxes_xyxy: tuple[tuple[float, float, float, float], ...] = ()
    alternatives: tuple[NDArray[np.bool_], ...] = ()


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
        p = self.params
        self.store = TrackStore(frame_id)
        self.queries = QueryBook(search_start_s=p.search_start_s, search_max_s=p.search_max_s)
        self._sched = Scheduler(segment_hz=p.segment_hz, budget_wall_s=p.segment_budget_wall_s)
        self._cells: dict[str, frozenset[Cell]] = {}
        self._centroid: dict[str, tuple[float, float, float]] = {}
        self._disagree: dict[str, int] = {}
        self._served: dict[str, int] = {}
        self._released: set[str] = set()
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
    ) -> None:
        """Start a standing query (G.1) at goal dispatch, source time ``stamp_ns``."""
        self.queries.open(goal_id, query_id, query, declaration_key, stamp_ns=stamp_ns)

    def close_goal(self, goal_id: str) -> None:
        """End every query of ``goal_id``; tracks persist across goals."""
        self.queries.close_goal(goal_id)
        self._pending = [r for r in self._pending if not r.query_key or r.query_key[0] != goal_id]

    def feed(  # noqa: PLR0912  # reason: one flat pass over steps 0-7
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
        if not usable:
            log.debug("object_tracker.ageing_only", camera=frame.camera, stamp_ns=frame.stamp_ns)
        bound_fits: list[tuple[str, ContainingFit]] = []
        for res in results:
            req = self._sched.complete(res.request_id)
            if req is None:
                continue
            if (req.camera, req.stamp_ns) != (frame.camera, frame.stamp_ns):
                log.warning("object_tracker.result_frame_mismatch", request_id=res.request_id)
            elif usable and req.service == "locate_in_view":
                self._pending += [
                    replace(req, service="segment_in_view", box_xyxy=b, served_ns=-1, request_id=0)
                    for b in res.boxes_xyxy
                ]
            elif usable and res.mask is not None and grid is not None:
                masks = (res.mask, *res.alternatives)
                for n, mask in enumerate(masks, start=1):
                    bound = self._apply_mask(
                        req, mask, frame, grid, grid_stamp_ns, hand_spaces, last=n == len(masks)
                    )
                    if bound is not _NEXT:
                        break
                if isinstance(bound, tuple):
                    bound_fits.append(bound)
        if grid is not None and grid_stamp_ns > self._grid_ns:
            self._grid_ns = grid_stamp_ns
            for t in self.store.all():
                self._confirm(t, grid, grid_stamp_ns, hand_spaces)
        if usable:
            self._visibility(frame)
        age_tracks(
            self.store,
            self._now_ns,
            max_age_s=self.params.max_age_s,
            lost_s=self.params.lost_s,
            keep_lost_s=self.params.keep_lost_s,
        )
        escalations = self.queries.update(self.store.all(), now_ns=self._now_ns)
        issued = self._sched.next(self._candidates(frame, grid) if usable else [], wall_s=wall_s)
        if issued is not None:
            self._mark_served(issued)
        return TrackerOutput(
            requests=() if issued is None else (issued,),
            tracked=self.store.snapshot(self._now_ns),
            escalations=tuple(escalations),
            bound_fits=tuple(bound_fits),
        )

    # -- step 1/3/6: a mask becomes a fit, the fit a track ---------------------------------

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
        spread = float(np.ptp(surface[:, :2], axis=0).max()) / 2
        reach = spread + p.support_probe_m + grid.resolution
        centers = grid.occupied_centers()
        column = centers[
            np.all(np.abs(centers[:, :2] - near) <= reach, axis=1)
            & (centers[:, 2] <= surface[:, 2].max() + grid.resolution)
        ]
        top = support_top_from_voxels(
            grid,
            column,
            near_xy=(float(near[0]), float(near[1])),
            min_cells=p.support_min_cells,
            probe_margin_m=p.support_probe_m,
            # A cell that starts below the lowest seen point: the support's top layer is
            # often the one the object's own bottom sits in (a bowl on a table at a cell's
            # mid-height), and the cone is cut a cell under the measured top anyway.
            surface_centers=centers[centers[:, 2] - 0.5 * grid.resolution < surface[:, 2].min()],
        )
        return None if top is None else top - grid.resolution

    def _fit(
        self, req: TrackerRequest, mask: NDArray[np.bool_], frame: CameraFrame, grid: VoxelLattice
    ) -> tuple[ContainingFit, float] | None:
        """Measure the support under the mask, fit, and gate on visibility; ``None`` = refused."""
        assert frame.t_base_from_cam is not None
        t_cam = frame.t_base_from_cam
        fp = self.params.fit
        raw, _ = backproject_masked_depth(
            mask & (frame.robot_mask != ROBOT_PIXEL) if frame.robot_mask is not None else mask,
            frame.depth_m,
            frame.intrinsics,
            min_depth_m=fp.min_depth_m,
            max_depth_m=fp.max_depth_m,
        )
        if len(raw) < fp.min_points:
            log.debug("object_tracker.fit_refused", why="too_few_points", label=req.label)
            return None
        surface = raw @ t_cam[:3, :3].T + t_cam[:3, 3]
        support_z = self._support(grid, surface)
        if support_z is None:
            log.info("object_tracker.fit_refused", why="no_support", label=req.label)
            return None
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
        )
        log.debug("object_tracker.fit", label=req.label, refusal=fit.refusal, **fit.assumptions)
        if fit.box is None or fit.tight is None:
            log.info("object_tracker.fit_refused", why=fit.refusal, label=req.label)
            return None
        vis = visible_fraction(fit.box, frame.depth_m, frame.robot_mask, frame.intrinsics, t_cam)
        if vis is not None and vis < self.params.theta_vis:
            log.info("object_tracker.fit_deferred", why="occluded", visible=vis, label=req.label)
            return None
        return fit, support_z

    def _apply_mask(  # noqa: PLR0911  # reason: each gate names its own outcome
        self,
        req: TrackerRequest,
        mask: NDArray[np.bool_],
        frame: CameraFrame,
        grid: VoxelLattice,
        grid_stamp_ns: int,
        hands: Sequence[PlaceRegion],
        *,
        last: bool = True,
    ) -> tuple[str, ContainingFit] | None | object:
        """Fit ``mask`` into a track; ``_NEXT`` when a smaller candidate should be tried.

        Not the ``last`` candidate: a refused fit, or one whose box holds two bodies on the
        map (``map_split``), hands over to the next one instead of minting a SPLIT track.
        """
        fitted = self._fit(req, mask, frame, grid)
        if fitted is None:
            return None if last else _NEXT
        fit, support_z = fitted
        surface = fit.surface
        centroid = tuple(float(v) for v in surface.mean(axis=0))
        cells = surface_cells(grid, surface, exclude=hands)
        if not last and fit.box is not None:
            split = cover_verdict(grid, cells, fit.box, support_z=support_z, own=hands)
            if split.kind == "map_split":
                log.info("object_tracker.candidate_split", label=req.label, why=split.why)
                return _NEXT
        oid, kind = associate(self.store.all(), centroid, voxel_m=grid.resolution)
        if kind == "tie":
            return None
        floor = support_z + grid.resolution
        if oid is None:
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
            )
        else:
            old = self.store.get(oid)
            assert old is not None
            if old.status is TrackStatus.OCCLUDED:
                log.info("object_tracker.occluded_refit_ignored", object_id=oid)
                return None
            if old.status is TrackStatus.BOUND:
                return oid, fit
            refit = old.status in (TrackStatus.SPLIT, TrackStatus.LOST)  # back to TENTATIVE
            track = old.model_copy(
                update={
                    "box": fit.box,
                    "tight": (fit.tight,),
                    "support_z": support_z,
                    "floor_z": max(old.floor_z, floor),
                    "visible_cells": len(cells),
                    "last_fit_stamp_ns": frame.stamp_ns,
                    "status": TrackStatus.TENTATIVE if refit else old.status,
                    "identity_inferred": old.identity_inferred or kind == "inferred",
                }
            )
            self.store.replace(track)
        self._cells[track.object_id] = cells
        self._centroid[track.object_id] = (centroid[0], centroid[1], centroid[2])
        self._disagree.pop(track.object_id, None)
        self._confirm(track, grid, grid_stamp_ns, hands)
        return None

    # -- steps 4-5: map confirmation and status ---------------------------------------------

    def _confirm(
        self, t: TrackedObject, grid: VoxelLattice, stamp_ns: int, hands: Sequence[PlaceRegion]
    ) -> None:
        if t.status is TrackStatus.RELEASED and t.object_id not in self._released:
            self._released.add(t.object_id)  # needs a fresh fit in the fall column
            self._cells.pop(t.object_id, None)
        cells = self._cells.get(t.object_id)
        if cells is None or t.status in (TrackStatus.BOUND, TrackStatus.LOST):
            return
        v = cover_verdict(grid, cells, t.box, support_z=t.support_z, own=hands)
        status = t.status
        update: dict[str, object] = {}
        if not v.kind:
            self._disagree.pop(t.object_id, None)
            if status is TrackStatus.SPLIT:
                return  # ineligible until a camera re-fit
            update = {"occupied_cells": v.occupied, "last_confirmed_stamp_ns": stamp_ns}
            if status is TrackStatus.TENTATIVE:
                update["status"] = TrackStatus.CONFIRMED
            elif status is TrackStatus.RELEASED:
                update.update(status=TrackStatus.CONFIRMED, epoch=t.epoch + 1)
                self._released.discard(t.object_id)
        elif v.kind == "map_split" and status is not TrackStatus.RELEASED:
            update = {"status": TrackStatus.SPLIT}
        elif v.kind == "map_disagrees" and status is not TrackStatus.RELEASED:
            n = self._disagree.get(t.object_id, 0) + 1
            self._disagree[t.object_id] = n
            if n >= self.params.disagree_grids:
                update = {"status": TrackStatus.LOST}
        if v.kind:
            log.debug("object_tracker.cover", object_id=t.object_id, kind=v.kind, why=v.why)
        if update:
            self.store.replace(t.model_copy(update=update))

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

    def _candidates(self, frame: CameraFrame, grid: VoxelLattice | None) -> list[TrackerRequest]:
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
                resolved = q.status is ObjectQueryStatus.RESOLVED
                prio = PRIORITY_REFRESH if resolved else PRIORITY_QUERY
                out.append(replace(r, stamp_ns=stamp, priority=prio))
        tracks = self.store.all()
        for q in self.queries.due():
            prio = PRIORITY_REFRESH if q.status is ObjectQueryStatus.RESOLVED else PRIORITY_QUERY
            base = TrackerRequest(
                "locate_in_view", cam, stamp, prio, q.last_served_ns, q.query.label, q.key
            )
            seed = q.query.seed
            if seed is not None and grid is not None and seed.frame_id == grid.frame_id:
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
                    within = None if rect is None else tuple(float(v) for v in rect[0])
                    seg = replace(base, service="segment_in_view", point_uv=uv)
                    out.append(replace(seg, within_xyxy=within))
                    continue
                skip.clear()  # every seed cell tried: start over, and ask the detector
            out.append(base)
        for t in tracks:
            served = max(t.last_fit_stamp_ns, self._served.get(t.object_id, -1))
            if t.status is TrackStatus.RELEASED:
                x, y, _ = t.box.pose.xyz  # the fall column, down to the support
                prio, xyz = PRIORITY_RELEASED, (x, y, t.support_z + 0.5 * t.box.half_extents[2])
            elif self._disagree.get(t.object_id) and t.status is not TrackStatus.OCCLUDED:
                prio, xyz = PRIORITY_DISAGREE, self._centroid.get(t.object_id, t.box.pose.xyz)
            elif t.status is TrackStatus.BOUND:
                prio, xyz = PRIORITY_BOUND, t.box.pose.xyz
            elif t.status is TrackStatus.CONFIRMED:
                prio, xyz = PRIORITY_REFRESH, self._centroid.get(t.object_id, t.box.pose.xyz)
            else:
                continue  # OCCLUDED is the map's job; TENTATIVE/SPLIT/LOST wait for queries
            uv = project_point(xyz, cam_from_base, frame.intrinsics)
            box = project_box(t.box, frame.intrinsics, t_cam)
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

    def _mark_served(self, req: TrackerRequest) -> None:
        if req.object_id is not None:
            self._served[req.object_id] = self._now_ns
        if req.query_key is not None and (q := self.queries.get(*req.query_key)) is not None:
            q.last_served_ns = self._now_ns
            if req.service == "segment_in_view" and req.box_xyxy is not None:
                self._pending = [
                    r for r in self._pending
                    if (r.query_key, r.box_xyxy) != (req.query_key, req.box_xyxy)
                ]  # fmt: skip
