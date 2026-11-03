"""Grasp binding and in-hand update of tracked objects (issue #418).

At ATTACH the hand's pinch volume ``J_h`` (``pinch.HandModel``, at the measured jaw by
joint-state FK, grown by one cell) is intersected with the tracks:

* free tracks of any role but ``DESTINATION`` whose measured geometry was fit within
  ``max_track_fit_age_s`` of the attach stamp and holds the pinch centre (``J_h``'s
  centre) within one cell — the tight k-DOP when the track has one, else its box. A
  ``RELEASED`` track is never a
  candidate: let go, it is not followed, and a regrasp binds the object's fresh track (or
  the fallback). A DESTINATION is where the payload goes, never the payload: a jaw
  closing inside a bowl's box bound the bowl (a 23 x 28 cm payload on the hand, a
  self-collision stop). A padded
  box that merely *meets* the swept pinch volume is not enough: a tracked neighbour beside
  an ungrounded held object would be bound in its place (p2po_on2);
* a grounded candidate (CONFIRMED / OCCLUDED) drops any TENTATIVE / SPLIT one: an
  unconfirmed fit over the same cells (a mask over two oranges) is no second object;
* exactly one: ``TRACKED_OBJECT`` binding of that id (``pose_in_link = T_link^-1 T_obj``,
  unioned with the same shape re-centred on the pinch axis);
* two or more: the union of the candidates and the fallback box, ``PINCH_FALLBACK``, no
  identity (nothing is cleared by identity);
* only PROVISIONAL tracks (fitted through the robot, maybe short): the union of those
  boxes and the fallback box, ``PINCH_FALLBACK``, no identity, a re-ground asked for;
* none: ``PINCH_FALLBACK``, the fallback box (``HandModel.fallback_in_link``: the jaw
  opening seen before closure across the pads, the full stroke along the other two axes);
  logged ``untracked_grasp`` and a re-ground is asked for.

Every fallback payload is cut at the palm plane.

The bound geometry then only grows (the kernel's K4 unions revisions): a REGRASP unions
the shape re-centred on the new pinch axis. At most ``MAX_PRIMITIVES``; past it the update
is refused and logged. The map never grows a payload: cells that moved with the hand were
mostly not the object (2026-11-02 restock audit: 46 of 62 grown pieces off the object, 6 cm
median and up to 20 cm outside it, behind it on the head camera's view ray), and the
fallback box already covered every banana and orange at the grasp.

Python proposes geometry and ids; the safety kernel disposes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
    ObjectRole,
    PlaceRegion,
    Pose6D,
    TrackedObject,
    TrackStatus,
)
from openral_core.exceptions import ROSConfigError
from openral_core.geometry import homogeneous_from_quat_xyz, rotation_to_quat_wxyz
from openral_core.kinematics import Box, bounding_half_extents
from openral_core.schemas import DOP_AXES, MAX_TIGHT_HULL_VERTICES, TightCollisionGeometry
from openral_core.tracking_thresholds import load_tracking_thresholds

from openral_world_state.object_tracker.pinch import HandModel, PinchVolume

__all__ = ["BindOutcome", "Binding", "GraspBinder", "TrackStoreLike"]

#: ``AttachedCollisionPrimitive.geometry_source``: the evidence behind a payload primitive.
GeometrySource = Literal["fallback", "measured", "grown"]

log = structlog.get_logger(__name__)

_B = load_tracking_thresholds().binding
#: Kernel cap on one object's primitives (K4); mirrors ``AttachedCollisionObject``.
MAX_PRIMITIVES = load_tracking_thresholds().caps.bound_primitives
#: Confidence of a tracked binding: the same as ``region_attachment``'s measured region.
TRACKED_CONFIDENCE = _B.tracked_confidence
#: Confidence of a pinch fallback: the same as the HAL's jaw-span fallback.
FALLBACK_CONFIDENCE = _B.fallback_confidence
_EPS_M = 1e-6
_NS = 1_000_000_000
_FREE = {TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.SPLIT}
#: Tracks a hand may bind with confidence: the map confirmed them.
_GROUNDED = {TrackStatus.CONFIRMED, TrackStatus.OCCLUDED}
#: Tracks whose fit the map has not confirmed (or has refuted as two bodies).
_UNCONFIRMED = {TrackStatus.TENTATIVE, TrackStatus.SPLIT}
_UNIT = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
#: Tracked objects whose box centre lies this close (m, horizontally) to the pinch centre give
#: an untracked grasp the support it happened over. *Calibration point.*
_SUPPORT_NEAR_M = _B.support_near_m


class TrackStoreLike(Protocol):
    """The track set the tracker core owns (C1's ``TrackStore``)."""

    def get(self, object_id: str) -> TrackedObject | None:
        """The track named ``object_id``, if any."""

    def all(self) -> list[TrackedObject]:
        """Every track."""

    def replace(self, obj: TrackedObject) -> None:
        """Store ``obj`` under its id (bumps the set revision)."""


class BindOutcome(StrEnum):
    """How an ATTACH resolved."""

    TRACKED = "tracked"
    AMBIGUOUS = "ambiguous"
    PROVISIONAL = "provisional"
    UNTRACKED = "untracked"


@dataclass(frozen=True)
class Binding:
    """One hand's held payload as the tracker publishes it.

    Attributes:
        outcome: Tracked, ambiguous or untracked.
        attachment: The ``AttachedCollisionObject`` for ``/openral/attachment_state``.
        candidates: Track ids that met ``J_h``.
        reground: The tracker should re-ground after lift (an untracked grasp).
    """

    outcome: BindOutcome
    attachment: AttachedCollisionObject
    candidates: tuple[str, ...]
    reground: bool


def _box_holds(box: Box, point: NDArray[np.float64], tol: float) -> bool:
    """Whether ``box`` grown by ``tol`` holds ``point``."""
    c, r, h = box
    return bool(np.all(np.abs((point - c) @ r) <= h + tol))


def _holds(track: TrackedObject, point: NDArray[np.float64], tol: float) -> bool:
    """Whether ``track``'s measured geometry holds ``point`` within ``tol``.

    The tight k-DOP (``TightCollisionGeometry.dop_lo_m``/``dop_hi_m`` over ``DOP_AXES`` in
    the primitive's frame; a containing superset of the hull, tighter than the padded box)
    of any primitive when the track has one, else its box.
    """
    tight = [p for p in track.tight if p.tight_geometry is not None]
    if not tight:
        return _box_holds(_box(track.box), point, tol)
    axes = np.asarray(DOP_AXES, dtype=np.float64)
    for prim in tight:
        tg = prim.tight_geometry
        assert tg is not None  # reason: filtered above
        t = _t(prim.pose_in_object)
        proj = axes @ ((point - t[:3, 3]) @ t[:3, :3])
        if np.all(proj >= np.asarray(tg.dop_lo_m) - tol) and np.all(
            proj <= np.asarray(tg.dop_hi_m) + tol
        ):
            return True
    return False


def _pinch_holds(
    track: TrackedObject, pinch: Box, tol: float, *, centre_only: bool = False
) -> bool:
    """Whether ``track``'s geometry holds the pinch centre or the fingertips' level below it.

    The fingertips' level is the centre lowered to the pinch volume's bottom (skipped with
    ``centre_only``). A thin object grasped from above sits below the pad centre: the Robotiq's
    pinch centre rode 1.8-4.6 cm above a 3.7 cm banana's tight box, and every such grasp bound
    an untracked pinch payload. The lowered point stays under the pads, so a neighbour beside
    the hand still never meets it.
    """
    centre, rot, half = pinch
    low = np.array(centre, dtype=np.float64)
    low[2] -= float(np.abs(rot[2, :]) @ half)
    return _holds(track, centre, tol) or (not centre_only and _holds(track, low, tol))


def _excess(track: TrackedObject, point: NDArray[np.float64]) -> list[float]:
    """How far ``point`` lies past ``track``'s first tight DOP (else box), per DOP axis (m)."""
    tight = [p for p in track.tight if p.tight_geometry is not None]
    if not tight:
        c, r, h = _box(track.box)
        return [round(float(v), 3) for v in np.maximum(np.abs((point - c) @ r) - h, 0.0)]
    tg = tight[0].tight_geometry
    assert tg is not None  # reason: filtered above
    t = _t(tight[0].pose_in_object)
    proj = np.asarray(DOP_AXES, dtype=np.float64) @ ((point - t[:3, 3]) @ t[:3, :3])
    over = np.maximum(proj - np.asarray(tg.dop_hi_m), np.asarray(tg.dop_lo_m) - proj)
    return [round(float(v), 3) for v in np.maximum(over, 0.0)]


def _t(pose: Pose6D) -> NDArray[np.float64]:
    return homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)


def _pose(t: NDArray[np.float64], frame_id: str) -> Pose6D:
    w, x, y, z = rotation_to_quat_wxyz(t[:3, :3])
    return Pose6D(
        xyz=(float(t[0, 3]), float(t[1, 3]), float(t[2, 3])),
        quat_xyzw=(x, y, z, w),
        frame_id=frame_id,
    )


def _frame(centre: NDArray[np.float64], rot: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, centre
    return t


def _box(region: PlaceRegion) -> Box:
    t = _t(region.pose)
    return (t[:3, 3], t[:3, :3], np.asarray(region.half_extents, dtype=np.float64))


def _prim(
    t_obj_prim: NDArray[np.float64],
    half: Sequence[float] | NDArray[np.float64],
    object_id: str,
    source: GeometrySource = "fallback",
) -> AttachedCollisionPrimitive:
    """One box primitive; ``source`` tells the kernel whether it is evidence of the object."""
    return AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=tuple(float(h) for h in half)),
        pose_in_object=_pose(t_obj_prim, object_id),
        geometry_source=source,
    )


def _moved(p: AttachedCollisionPrimitive, t: NDArray[np.float64]) -> AttachedCollisionPrimitive:
    """``p`` with ``t`` (object frame) applied to its pose."""
    return p.model_copy(
        update={"pose_in_object": _pose(t @ _t(p.pose_in_object), p.pose_in_object.frame_id)}
    )


def _clip_at_plane(
    verts: NDArray[np.float64], along: NDArray[np.float64], plane: float
) -> NDArray[np.float64]:
    """The vertices of ``conv(verts)`` cut by ``along >= plane``, as a point set.

    Exact for a convex set: the polytope's vertices on the kept side are the original vertices
    there plus the points where a segment between a kept and a dropped vertex crosses the plane
    (every such pair is considered, so no edge of the hull is missed).
    """
    keep = along >= plane
    out = [verts[keep]]
    if keep.any() and not keep.all():
        a, b = verts[keep][:, None, :], verts[~keep][None, :, :]
        da, db = (along[keep] - plane)[:, None], (plane - along[~keep])[None, :]
        out.append((a + (da / (da + db))[..., None] * (b - a)).reshape(-1, 3))
    return np.asarray(np.unique(np.round(np.concatenate(out), 10), axis=0), dtype=np.float64)


def _palm_clipped(
    prims: Sequence[AttachedCollisionPrimitive],
    t_ao: NDArray[np.float64],
    pinch: PinchVolume,
    object_id: str,
) -> list[AttachedCollisionPrimitive]:
    """``prims`` (object frame, object at ``t_ao`` in the attach link) cut at the palm plane.

    Nothing the hand holds lies behind its palm (the hand is there), so a fitted shape
    reaching past it (a containing box of a 17 cm banana re-centred on the pinch volume reached
    the Franka's wrist: DROID r2dr_on1, a payload-vs-panda_link7 false stop) is cut by the plane:
    a primitive wholly behind it is dropped, one wholly in front is kept as it is (hull and all),
    and one the plane crosses becomes a box in the pinch axes (the broad-phase bound and every
    other check's geometry) around the EXACT cut of its solid, refined by that cut's 26-DOP and
    vertices as the kernel's payload-vs-voxel ``tight_geometry`` (OT-16 amendment, 2026-10-18:
    same containment, tighter representation). The solid is the primitive's hull when it has one,
    else its box. The new box lies inside the old pinch-axes box (the bounding box of the whole
    primitive with the back face on the plane) and holds solid ∩ {along >= palm}; a hull of more
    than ``MAX_TIGHT_HULL_VERTICES`` points keeps the DOP only (a containing superset).
    """
    out: list[AttachedCollisionPrimitive] = []
    a = pinch.axes[:, 1]
    dop = np.asarray(DOP_AXES, dtype=np.float64)
    for p in prims:
        t = t_ao @ _t(p.pose_in_object)
        half = np.asarray(bounding_half_extents(p.shape), dtype=np.float64)
        corners = (_UNIT * half) @ t[:3, :3].T + t[:3, 3]
        along = corners @ a
        if along.min() >= pinch.palm_m - _EPS_M:
            out.append(p)
            continue
        if along.max() <= pinch.palm_m:
            continue
        hull = (
            np.asarray(p.tight_geometry.hull_vertices_m, dtype=np.float64)
            if p.tight_geometry is not None and p.tight_geometry.hull_vertices_m
            else None
        )
        solid_local = hull if hull is not None else _UNIT * half
        solid = solid_local @ t[:3, :3].T + t[:3, 3]
        cut = _clip_at_plane(solid, solid @ a, pinch.palm_m)
        if not len(cut):
            continue
        local = cut @ pinch.axes
        lo, hi = local.min(axis=0), local.max(axis=0)
        lo[1] = pinch.palm_m
        centre = (lo + hi) / 2.0
        new_half = np.maximum((hi - lo) / 2.0, 1e-6)
        t_link = _frame(pinch.axes @ centre, pinch.axes)
        in_box = (cut - pinch.axes @ centre) @ pinch.axes
        proj = in_box @ dop.T
        dlo, dhi = proj.min(axis=0), proj.max(axis=0)
        dlo[:3], dhi[:3] = -new_half, new_half  # the box's own slabs: the cut lies inside them
        keep_hull = len(in_box) <= MAX_TIGHT_HULL_VERTICES
        out.append(
            AttachedCollisionPrimitive(
                shape=BoxShape(half_extents_m=tuple(float(h) for h in new_half)),
                pose_in_object=_pose(np.linalg.inv(t_ao) @ t_link, object_id),
                tight_geometry=TightCollisionGeometry(
                    dop_lo_m=tuple(float(v) for v in dlo),
                    dop_hi_m=tuple(float(v) for v in dhi),
                    hull_vertices_m=tuple((float(x), float(y), float(z)) for x, y, z in in_box)
                    if keep_hull
                    else (),
                ),
                geometry_source=p.geometry_source,
            )
        )
    return out or list(prims)


def _support_clipped(
    prims: Sequence[AttachedCollisionPrimitive],
    t_fo: NDArray[np.float64],
    support_z: float,
    object_id: str,
) -> list[AttachedCollisionPrimitive]:
    """``prims`` (object frame, object at ``t_fo`` in the grid frame) cut at ``support_z``.

    Nothing the hand holds reaches below the surface it rested on at the grasp (the surface
    is there), so a box reaching under it -- the pinch fallback's full stroke centred on a
    carton in a tote reached a cell under the tote floor -- is cut along its most vertical axis
    where its cut face's HIGHEST point meets the surface: everything cut lies at or below the
    surface, whatever the hand's tilt, and the box only ever shrinks (a re-boxing in gravity's axes
    grew one sideways into the robot's own forearm). A box wholly above is kept as it is; one wholly
    below is dropped.
    """
    out: list[AttachedCollisionPrimitive] = []
    for p in prims:
        t = t_fo @ _t(p.pose_in_object)
        half = np.asarray(bounding_half_extents(p.shape), dtype=np.float64)
        reach = float(np.abs(t[2, :3]) @ half)
        if float(t[2, 3]) - reach >= support_z:
            out.append(p)
            continue
        if float(t[2, 3]) + reach <= support_z:
            continue
        k = int(np.argmax(np.abs(t[2, :3])))
        dz = float(t[2, k])
        rise = reach - abs(dz) * float(half[k])  # the other axes' reach up from the face
        cut = (support_z - rise - float(t[2, 3])) / dz
        lo, hi = (max(-half[k], cut), half[k]) if dz > 0 else (-half[k], min(half[k], cut))
        if hi <= lo:
            out.append(p)  # the cut face cannot clear the surface: kept whole
            continue
        shift = np.eye(4)
        shift[k, 3] = 0.5 * (lo + hi)
        half = half.copy()
        half[k] = 0.5 * (hi - lo)
        out.append(_prim(_t(p.pose_in_object) @ shift, half, object_id, p.geometry_source))
    return out


class GraspBinder:
    """One hand's binding: ATTACH and REGRASP.

    Args:
        hand: The gripper (``HandModel``).
        frame_id: The grid frame every track box is in; FK runs from it, so it must be a
            link of the manifest (its ``base_frame``).
        cell_m: Grid resolution; ``J_h`` is grown by one cell.
        max_track_fit_age_s: Maximum source-time age of free-track fitted geometry.

    Raises:
        ROSConfigError: ``frame_id`` is not the manifest's ``base_frame``.
    """

    def __init__(
        self, hand: HandModel, *, frame_id: str, cell_m: float, max_track_fit_age_s: float
    ) -> None:
        """Bind nothing yet; use the tracker's existing fit-age eligibility bound."""
        if frame_id != hand.description.base_frame:
            raise ROSConfigError(
                f"tracks in {frame_id!r} but FK roots at {hand.description.base_frame!r}."
            )
        if not math.isfinite(max_track_fit_age_s) or max_track_fit_age_s <= 0.0:
            raise ROSConfigError("max_track_fit_age_s must be finite and positive.")
        self.hand, self.frame_id, self.cell_m = hand, frame_id, cell_m
        self._max_track_fit_age_ns = int(max_track_fit_age_s * _NS)
        self.held: Binding | None = None
        self._shape: list[AttachedCollisionPrimitive] = []  # object frame, revision 0
        # The jaw the current closing stroke started from (its most open reading), and the
        # most closed one since: a reopening past the release gap starts a new stroke.
        self._opening: float | None = None
        self._closed_since: float | None = None

    def observe_jaw(self, jaw: float) -> None:
        """Fold in a measured jaw; the opening of the last closing stroke bounds the fallback.

        The held part passed between the pads as they closed from that opening, so it is at
        most that wide. Not the most open jaw of the whole hold: a regrasp that closes from
        part way would otherwise take a full-stroke width from an earlier opening (Spark
        bkA_on_rs_10, 2026-11-02: 21 cm across a 5 cm carton, against the torso). A
        reopening counts as a new stroke only past half the manifest's ``stall_gap`` (the
        jaw's release hysteresis), so reading jitter never narrows the bound.
        """
        if not math.isfinite(jaw):
            return
        cal = self.hand.joint.closure_calibration
        tol = cal.stall_gap / 2.0 if cal is not None else 0.0
        d = abs(jaw - self.hand.closed)
        if self._opening is None or self._closed_since is None:
            self._opening, self._closed_since = jaw, d
            return
        peak = abs(self._opening - self.hand.closed)
        if d > peak:
            self._opening, self._closed_since = jaw, d  # still opening
        elif self._closed_since < peak - tol and d > self._closed_since + tol:
            self._opening, self._closed_since = jaw, d  # closed, then reopened: a new stroke
        else:
            self._closed_since = min(self._closed_since, d)

    # -- geometry -------------------------------------------------------------------
    def pinch_in_frame(self, positions: Mapping[str, float]) -> Box:
        """``J_h`` (grown by one cell) in the grid frame at ``positions``."""
        j = self.hand.pinch_in_link(positions[self.hand.joint.name]).grown(self.cell_m)
        t = self.hand.t_frame_from_attach(self.frame_id, positions)
        return (t[:3, :3] @ j.centre + t[:3, 3], t[:3, :3] @ j.axes, j.half)

    def object_pose(self, positions: Mapping[str, float]) -> Pose6D:
        """The held object's frame in the grid frame: ``FK(q) . pose_in_link``."""
        if self.held is None:
            raise ROSConfigError("object_pose: nothing is held.")
        t = self.hand.t_frame_from_attach(self.frame_id, positions)
        return _pose(t @ _t(self.held.attachment.pose_in_link), self.frame_id)

    # -- ATTACH ---------------------------------------------------------------------
    def on_attach(
        self,
        store: TrackStoreLike,
        positions: Mapping[str, float],
        stamp_ns: int,
        *,
        excluded_track_ids: frozenset[str] = frozenset(),
    ) -> Binding:
        """Bind fresh free geometry meeting ``J_h``."""
        jh = self.pinch_in_frame(positions)
        centre = jh[0]
        cands = []
        stale: list[tuple[str, int, str]] = []
        for track in store.all():
            if track.status in _FREE and _pinch_holds(track, jh, self.cell_m):
                if track.role is ObjectRole.DESTINATION:
                    log.info("object_tracker.destination_not_bound", object_id=track.object_id,
                             hand=self.hand.joint.name)  # fmt: skip
                    continue
                fit_age_ns = stamp_ns - _fresh_ns(track)
                if track.object_id in excluded_track_ids:
                    stale.append((track.object_id, fit_age_ns, "late_hold_stale_fit"))
                    continue
                if fit_age_ns < 0:
                    stale.append((track.object_id, fit_age_ns, "future_fit"))
                    continue
                if fit_age_ns > self._max_track_fit_age_ns:
                    stale.append((track.object_id, fit_age_ns, "stale_fit"))
                    continue
                cands.append(track)
        if stale:
            log.info(
                "object_tracker.stale_attach_candidate",
                hand=self.hand.joint.name,
                max_track_fit_age_ns=self._max_track_fit_age_ns,
                candidates=[
                    {"object_id": object_id, "fit_age_ns": age_ns, "reason": reason}
                    for object_id, age_ns, reason in stale
                ],
            )
        jaw = positions[self.hand.joint.name]
        if any(t.status in _GROUNDED for t in cands):
            # A grounded object beats an unconfirmed fit of the same cells (#418 r6po_dbg9: a
            # never-confirmed fit over two oranges made orange_2's grasp AMBIGUOUS).
            cands = [t for t in cands if t.status not in _UNCONFIRMED]
        provisional = [t for t in cands if t.provisional]
        cands = [t for t in cands if not t.provisional]
        ids = tuple(t.object_id for t in cands)
        if len(cands) == 1:
            binding = self._bind(store, cands[0], positions, stamp_ns)
        elif not cands and provisional:
            # Never identity or payload geometry alone (TrackedObject.provisional), but
            # measured: unioned with the fallback box it only grows the payload.
            ids = tuple(t.object_id for t in provisional)
            binding = self._fallback(provisional, positions, stamp_ns, provisional=True)
            log.warning("object_tracker.provisional_grasp", hand=self.hand.joint.name,
                        candidates=ids, jaw=jaw)  # fmt: skip
        else:
            # With no candidate, the support the tracked objects around the pinch stand on
            # (a tote's floor, a table): the grasp happened over it (an
            # untracked pinch box reached under the tote while lifting).
            near = [
                t.support_z
                for t in store.all()
                if t.status in _GROUNDED
                and float(np.linalg.norm(np.asarray(t.box.pose.xyz[:2]) - centre[:2]))
                <= _SUPPORT_NEAR_M
            ]
            binding = self._fallback(
                cands, positions, stamp_ns, support_z=min(near) if near and not cands else None
            )
            event = "object_tracker.ambiguous_grasp" if cands else "object_tracker.untracked_grasp"
            # Where the pinch missed each free track: its excess past the tight DOP (or box)
            # per axis, in the primitive frame (diagnosis: a pinch above a thin object's box).
            missed = {t.object_id: _excess(t, centre) for t in store.all() if t.status in _FREE}
            log.warning(event, hand=self.hand.joint.name, candidates=ids, jaw=jaw,
                        pinch=[round(float(v), 3) for v in centre], missed=missed)  # fmt: skip
        self._opening = None
        self._closed_since = None
        self.held = binding
        log.info(
            "object_tracker.bound",
            hand=self.hand.joint.name,
            outcome=binding.outcome.value,
            object_id=binding.attachment.object_id,
            candidates=ids,
            jaw=jaw,
        )
        return binding

    def _bind(
        self,
        store: TrackStoreLike,
        track: TrackedObject,
        positions: Mapping[str, float],
        stamp_ns: int,
    ) -> Binding:
        oid = track.object_id
        t_fa = self.hand.t_frame_from_attach(self.frame_id, positions)
        t_fo = _t(track.box.pose)
        t_ao = np.linalg.inv(t_fa) @ t_fo
        src: GeometrySource = "measured"
        shape = (
            [
                _moved(
                    h.model_copy(
                        update={
                            "pose_in_object": h.pose_in_object.model_copy(update={"frame_id": oid}),
                            "geometry_source": src,
                        }
                    ),
                    np.asarray(np.linalg.inv(t_fo), dtype=np.float64),
                )
                for h in track.tight
            ]
            if track.tight
            else [_prim(np.eye(4), track.box.half_extents, oid, src)]
        )
        pinch = self.hand.pinch_in_link(positions[self.hand.joint.name])
        c = pinch.axes[:, 0]
        # As fitted, unioned with the same shape slid onto the pinch axis.
        shift = -float((t_ao[:3, 3] - pinch.centre) @ c) * c
        slide = _frame(t_ao[:3, :3].T @ shift, np.eye(3))
        prims = shape + (
            [_moved(p, slide) for p in shape] if np.linalg.norm(shift) > _EPS_M else []
        )
        prims = _palm_clipped(prims, t_ao, pinch, oid)
        epoch = track.epoch
        pose_in_link = _pose(t_ao, self.hand.attach_link)
        store.replace(
            track.model_copy(
                update={
                    "status": TrackStatus.BOUND,
                    "epoch": epoch,
                    "bound_hand": self.hand.joint.name,
                    "attach_link": self.hand.attach_link,
                    "pose_in_link": pose_in_link,
                }
            )
        )
        self._shape = shape
        attachment = AttachedCollisionObject(
            object_id=oid,
            attach_link=self.hand.attach_link,
            touch_links=list(self.hand.touch_links),
            primitives=prims,
            pose_in_link=pose_in_link,
            confidence=TRACKED_CONFIDENCE,
            evidence_kind=AttachmentEvidenceKind.TRACKED_OBJECT,
            evidence_ref=f"{track.declaration_key}|epoch={epoch}",
            stamp_ns=stamp_ns,
            epoch=epoch,
        )
        return Binding(BindOutcome.TRACKED, attachment, (oid,), reground=False)

    def _fallback(
        self,
        cands: Sequence[TrackedObject],
        positions: Mapping[str, float],
        stamp_ns: int,
        *,
        provisional: bool = False,
        support_z: float | None = None,
    ) -> Binding:
        oid = f"pinch:{self.hand.joint.name}@{stamp_ns}"
        fb = self.hand.fallback_in_link(positions[self.hand.joint.name], self._opening)
        t_ao = _frame(fb.centre, fb.axes)
        prims = [_prim(np.eye(4), fb.half, oid)]
        t_of = np.linalg.inv(self.hand.t_frame_from_attach(self.frame_id, positions) @ t_ao)
        boxes = [_box(t.box) for t in cands]
        if len(boxes) > MAX_PRIMITIVES - 1:  # one box around them all, in the pinch axes
            pts = (
                np.concatenate([(_UNIT * h) @ r.T + c for c, r, h in boxes]) @ t_of[:3, :3].T
                + t_of[:3, 3]
            )
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            prims.append(_prim(_frame((lo + hi) / 2.0, np.eye(3)), (hi - lo) / 2.0, oid))
        else:
            # Evidence of the held object only when exactly ONE provisional candidate holds
            # the pinch centre; several (or ambiguous) boxes may belong to neighbours.
            src: GeometrySource = "measured" if provisional and len(cands) == 1 else "fallback"
            prims += [_prim(t_of @ _frame(c, r), h, oid, src) for c, r, h in boxes]
        prims = prims[:1] + _palm_clipped(prims[1:], t_ao, fb, oid) if len(prims) > 1 else prims
        floor = min(t.support_z for t in cands) if cands else support_z
        if floor is not None:
            # Cut at the lowest support the candidates (else the tracked neighbours) stand on.
            t_fo = self.hand.t_frame_from_attach(self.frame_id, positions) @ t_ao
            prims = _support_clipped(prims, t_fo, floor, oid) or prims
        self._shape = prims[:1]
        attachment = AttachedCollisionObject(
            object_id=oid,
            attach_link=self.hand.attach_link,
            touch_links=list(self.hand.touch_links),
            primitives=prims,
            pose_in_link=_pose(t_ao, self.hand.attach_link),
            confidence=FALLBACK_CONFIDENCE,
            evidence_kind=AttachmentEvidenceKind.PINCH_FALLBACK,
            evidence_ref=(
                ("provisional:" if provisional else "ambiguous:")
                + ",".join(t.object_id for t in cands)
            )
            if cands
            else "untracked_grasp",
            stamp_ns=stamp_ns,
        )
        outcome = (
            BindOutcome.PROVISIONAL
            if provisional
            else (BindOutcome.AMBIGUOUS if cands else BindOutcome.UNTRACKED)
        )
        return Binding(
            outcome,
            attachment,
            tuple(t.object_id for t in cands),
            reground=provisional or not cands,
        )

    # -- REGRASP --------------------------------------------------------------------
    def _union(self, added: Sequence[AttachedCollisionPrimitive], stamp_ns: int) -> Binding | None:
        assert self.held is not None  # reason: callers check
        att = self.held.attachment
        if len(att.primitives) + len(added) > MAX_PRIMITIVES:
            log.warning(
                "object_tracker.bound_geometry_cap",
                object_id=att.object_id,
                why="regrasp",
                held=len(att.primitives),
                added=len(added),
            )
            return None
        grown = att.model_copy(
            update={
                "primitives": [*att.primitives, *added],
                "stamp_ns": stamp_ns,
                "geometry_revision": att.geometry_revision + 1,
            }
        )
        self.held = Binding(
            self.held.outcome,
            AttachedCollisionObject.model_validate(grown.model_dump()),
            self.held.candidates,
            self.held.reground,
        )
        log.info(
            "object_tracker.bound_geometry_grew",
            object_id=att.object_id,
            why="regrasp",
            revision=grown.geometry_revision,
            primitives=len(grown.primitives),
        )
        return self.held

    def on_regrasp(self, positions: Mapping[str, float], stamp_ns: int) -> Binding | None:
        """REGRASP: union the shape re-centred on the new pinch axis (never replaces)."""
        if self.held is None:
            return None
        pinch = self.hand.pinch_in_link(positions[self.hand.joint.name])
        t_ao = _t(self.held.attachment.pose_in_link)
        c = pinch.axes[:, 0]
        shift = -float((t_ao[:3, 3] - pinch.centre) @ c) * c
        if np.linalg.norm(shift) <= _EPS_M:
            return None  # already on the pinch axis: nothing new to union
        slide = _frame(t_ao[:3, :3].T @ shift, np.eye(3))
        return self._union([_moved(p, slide) for p in self._shape], stamp_ns)

    def on_release(self) -> None:
        """Forget the held payload (the track itself goes through ``release.release_track``)."""
        self.held, self._shape = None, []


def _fresh_ns(track: TrackedObject) -> int:
    """When ``track``'s geometry was last current, for the bind's ``max_track_fit_age_s``.

    The camera fit's stamp; for an OCCLUDED TARGET track also the map's last agreement with
    its cells. Under the hand the camera cannot refit the object, but the map still sees its
    uncovered cells where the box says (a hidden or contact-held verdict never re-stamps
    ``last_confirmed_stamp_ns``). A visible track needs a camera fit: the map alone never
    re-dates it. Restock 2026-10-31 (arms-out start): a clear 2x box, map-confirmed 0.17 s
    before the grasp, was 4 s past its last fit and bound the pinch fallback. Only the
    goal's target: an occluded neighbour kept bindable the same way made the next grasp
    AMBIGUOUS (its 2x box met the pinch), where it stays stale as before. Safety-WG review
    owed.
    """
    if track.status is TrackStatus.OCCLUDED and track.role is ObjectRole.TARGET:
        return max(track.last_fit_stamp_ns, track.last_confirmed_stamp_ns)
    return track.last_fit_stamp_ns
