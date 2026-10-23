"""Grasp binding and in-hand update of tracked objects (issue #418).

At ATTACH the hand's pinch volume ``J_h`` (``pinch.HandModel``, at the measured jaw by
joint-state FK, grown by one cell) is intersected with the tracks:

* free tracks of any role but ``DESTINATION`` whose measured geometry was fit within
  ``max_track_fit_age_s`` of the attach stamp and holds the pinch centre (``J_h``'s
  centre) within one cell — the tight k-DOP when the track has one, else its box. A
  ``RELEASED`` track is never a
  candidate: let go, it is not followed, and a regrasp binds the object's fresh track (or
  the fallback). A DESTINATION is where the payload goes, never the payload: a jaw
  closing inside a bowl's box bound the bowl (2026-10-22, Spark R193 / R205: a 23 x 28 cm
  payload on the hand, a self-collision stop). A padded
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
the shape re-centred on the new pinch axis, and unexplained occupied cells 26-connected
to the payload are fitted by a rotation about the pinch axis and the arc is unioned. At
most ``MAX_PRIMITIVES``; past it the update is refused and logged.

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
from openral_core.perception_geometry import VoxelLattice
from openral_core.schemas import DOP_AXES

from openral_world_state.object_tracker.pinch import HandModel, PinchVolume

__all__ = ["BindOutcome", "Binding", "GraspBinder", "TrackStoreLike"]

#: ``AttachedCollisionPrimitive.geometry_source``: the evidence behind a payload primitive.
GeometrySource = Literal["fallback", "measured", "grown"]

log = structlog.get_logger(__name__)

#: Kernel cap on one object's primitives (K4); mirrors ``AttachedCollisionObject``.
MAX_PRIMITIVES = 16
#: Confidence of a tracked binding: the same as ``region_attachment``'s measured region.
TRACKED_CONFIDENCE = 0.7
#: Confidence of a pinch fallback: the same as the HAL's jaw-span fallback.
FALLBACK_CONFIDENCE = 0.2
#: Rotation search step about the pinch axis.
_THETA_STEP = math.radians(2.0)
_EPS_M = 1e-6
_EPS_RAD = 1e-9
_NS = 1_000_000_000
_FREE = {TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.SPLIT}
#: Tracks a hand may bind with confidence: the map confirmed them.
_GROUNDED = {TrackStatus.CONFIRMED, TrackStatus.OCCLUDED}
#: Tracks whose fit the map has not confirmed (or has refuted as two bodies).
_UNCONFIRMED = {TrackStatus.TENTATIVE, TrackStatus.SPLIT}
_UNIT = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_N26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if i or j or k]
#: How far from the pinch centre an untracked payload may grow (m). *Calibration point.*
#: ponytail: one fixed bound on the search; a manifest field if a gripper carries longer.
_GROW_RADIUS_M = 0.3
#: Tracked objects whose box centre lies this close (m, horizontally) to the pinch centre give
#: an untracked grasp the support it happened over. *Calibration point.*
_SUPPORT_NEAR_M = 0.15
#: A cell moves with the hand only once the hand moved this many cells between sightings. The
#: threshold separates "its first place is free now" (it left) from "the same cell again" (a
#: body longer than the step still fills its own first place), so an UNTRACKED / PROVISIONAL
#: payload, which has no measured box to start from, waits for two cells.
_RIGID_MIN_MOVE_CELLS = 2.0
#: ...a TRACKED payload starts from a measured, cleared box and only absorbs loose cells
#: connected to it that nothing else explains, so half a cell is enough: growth starts on the
#: first frames of hand motion, not after the object has already moved 3 cm (#418 r6po_dbg8:
#: the held orange drifted 0.9 cm in the hand within 80 ms of the bind).
_TRACKED_MIN_MOVE_CELLS = 0.5


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
    pinch centre rode 1.8-4.6 cm above a 3.7 cm banana's tight box (Spark R347 / R352 / R355,
    2026-10-23), and every such grasp bound an untracked pinch payload. The lowered point
    stays under the pads, so a neighbour beside the hand still never meets it.
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


def _palm_clipped(
    prims: Sequence[AttachedCollisionPrimitive],
    t_ao: NDArray[np.float64],
    pinch: PinchVolume,
    object_id: str,
) -> list[AttachedCollisionPrimitive]:
    """``prims`` (object frame, object at ``t_ao`` in the attach link) cut at the palm plane.

    Nothing the hand holds lies behind its palm (the hand is there), so a fitted shape
    reaching past it — a containing box of a 17 cm banana re-centred on the pinch volume
    reached the Franka's wrist (DROID r2dr_on1, a payload-vs-panda_link7 false stop) — is
    replaced by its box in the pinch axes with the back face on the palm plane. That box
    holds every part of the primitive in front of the palm; a primitive wholly behind it
    is dropped; one wholly in front is kept as it is (hull and all).
    """
    out: list[AttachedCollisionPrimitive] = []
    a = pinch.axes[:, 1]
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
        local = corners @ pinch.axes
        lo, hi = local.min(axis=0), local.max(axis=0)
        lo[1] = pinch.palm_m
        t_link = _frame(pinch.axes @ ((lo + hi) / 2.0), pinch.axes)
        out.append(
            _prim(np.linalg.inv(t_ao) @ t_link, (hi - lo) / 2.0, object_id, p.geometry_source)
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
    carton in a tote reached a cell under the tote floor (Spark restock R284/R296/R309,
    2026-10-23) -- is cut along its most vertical axis where its cut face's HIGHEST point
    meets the surface: everything cut lies at or below the surface, whatever the hand's
    tilt, and the box only ever shrinks (a re-boxing in gravity's axes grew one sideways into
    the robot's own forearm, R315). A box wholly above is kept as it is; one wholly below is
    dropped.
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


def _rotation_about(
    axis: NDArray[np.float64], point: NDArray[np.float64], theta: float
) -> NDArray[np.float64]:
    """``(4, 4)`` rotation by ``theta`` about the line ``point + s * axis`` (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    r = np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    return _frame(point - r @ point, r)


def _outside_m(points: NDArray[np.float64], boxes: Sequence[Box]) -> NDArray[np.float64]:
    """Per point, the distance outside the nearest of ``boxes`` (0 inside one)."""
    best = np.full(len(points), np.inf)
    for c, r, h in boxes:
        local = np.abs((points - c) @ r)
        best = np.minimum(best, np.linalg.norm(np.maximum(local - h, 0.0), axis=1))
    return best


class GraspBinder:
    """One hand's binding: ATTACH, REGRASP and map-protrusion growth.

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
        self._theta = 0.0
        self._opening: float | None = None  # the most open jaw since the last ATTACH
        # Untracked growth: attach-frame cell key -> its grid-frame centre when first seen.
        self._sighted: dict[tuple[int, int, int], NDArray[np.float64]] = {}
        # The world at the hold's first growth call: occupied lattice cells that were neither
        # the payload's connected body nor the hand. None until that call.
        self._static: set[tuple[int, ...]] | None = None

    def observe_jaw(self, jaw: float) -> None:
        """Fold in a measured jaw; the most open one bounds the next fallback's width."""
        if math.isfinite(jaw) and (
            self._opening is None
            or abs(jaw - self.hand.closed) > abs(self._opening - self.hand.closed)
        ):
            self._opening = jaw

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

    def _held_boxes(self, positions: Mapping[str, float]) -> list[Box]:
        assert self.held is not None  # reason: callers check
        t = _t(self.object_pose(positions))
        out: list[Box] = []
        for p in self.held.attachment.primitives:
            tp = t @ _t(p.pose_in_object)
            out.append((tp[:3, 3], tp[:3, :3], bounding_half_extents(p.shape)))
        return out

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
                fit_age_ns = stamp_ns - track.last_fit_stamp_ns
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
            # (a tote's floor, a table): the grasp happened over it (Spark restock R305,
            # 2026-10-23: an untracked pinch box reached under the tote while lifting).
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
        self._sighted, self._static = {}, None
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
        self._shape, self._theta = shape, 0.0
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
        self._shape, self._theta = prims[:1], 0.0
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

    # -- growth ---------------------------------------------------------------------
    def _union(
        self, added: Sequence[AttachedCollisionPrimitive], stamp_ns: int, why: str
    ) -> Binding | None:
        assert self.held is not None  # reason: callers check
        att = self.held.attachment
        if len(att.primitives) + len(added) > MAX_PRIMITIVES:
            log.warning(
                "object_tracker.bound_geometry_cap",
                object_id=att.object_id,
                why=why,
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
            why=why,
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
        return self._union([_moved(p, slide) for p in self._shape], stamp_ns, "regrasp")

    def grow_by_protrusion(
        self,
        store: TrackStoreLike,
        grid: VoxelLattice,
        positions: Mapping[str, float],
        stamp_ns: int,
    ) -> Binding | None:
        """Union the arc of a rotation about the pinch axis that explains protruding cells.

        Protruding cells are occupied cells near the payload (within its reach of the pinch
        axis) that nothing explains — not the bound geometry, not another track, not the
        hand (never evidence) — 26-connected to the bound geometry. The rotation ``theta``
        about the pinch axis whose rotated shape leaves every protruding cell centre less
        than one cell outside is fitted; the arc from the previous ``theta`` is unioned.
        ``None`` when nothing protrudes, the fit fails, or the cap would be passed.
        """
        if self.held is None or self.held.outcome is not BindOutcome.TRACKED:
            return None
        if grid.frame_id != self.frame_id:
            raise ROSConfigError(f"grid frame {grid.frame_id!r} != track frame {self.frame_id!r}.")
        cell = grid.resolution
        t_fa = self.hand.t_frame_from_attach(self.frame_id, positions)
        pinch = self.hand.pinch_in_link(positions[self.hand.joint.name])
        p, c = t_fa[:3, :3] @ pinch.centre + t_fa[:3, 3], t_fa[:3, :3] @ pinch.axes[:, 0]
        held = self._held_boxes(positions)
        corners = np.concatenate([(_UNIT * h) @ r.T + o for o, r, h in held])
        rel = corners - p
        reach = float(np.max(np.linalg.norm(rel - np.outer(rel @ c, c), axis=1)))
        ax_lo, ax_hi = float((rel @ c).min()), float((rel @ c).max())
        occ = grid.occupied_centers()
        rel = occ - p
        along = rel @ c
        radial = np.linalg.norm(rel - np.outer(along, c), axis=1)
        near = occ[(radial <= reach + cell) & (along >= ax_lo - cell) & (along <= ax_hi + cell)]
        if not len(near):
            return None
        others = [
            _box(t.box)
            for t in store.all()
            if t.object_id != self.held.attachment.object_id and t.status is not TrackStatus.BOUND
        ]
        explained = _outside_m(near, held) <= cell / 2.0
        for boxes in (others, self.hand.hand_boxes(self.frame_id, positions)):
            if boxes:
                explained |= _outside_m(near, boxes) <= cell / 2.0
        loose = near[~explained]
        prot = (
            self._connected(grid, loose, _outside_m(loose, held) <= cell) if len(loose) else loose
        )
        if not len(prot):
            return None
        # Fit theta: rotate the revision-0 shape about the pinch axis (grid frame).
        t_fo = _t(self.object_pose(positions))
        shape = [(t_fo @ _t(s.pose_in_object), bounding_half_extents(s.shape)) for s in self._shape]

        def residual(theta: float) -> float:
            rot = _rotation_about(c, p, theta)
            return float(
                _outside_m(prot, [((rot @ t)[:3, 3], (rot @ t)[:3, :3], h) for t, h in shape]).max()
            )

        thetas = np.arange(-math.pi, math.pi, _THETA_STEP)
        scores = np.array([residual(float(th)) for th in thetas])
        best = float(scores.min())
        if best >= cell:
            log.info(
                "object_tracker.protrusion_unexplained",
                object_id=self.held.attachment.object_id,
                cells=len(prot),
                residual_m=best,
            )
            return None
        ok = thetas[scores <= best + _EPS_M]
        theta = float(ok[np.argmin(np.abs(ok - self._theta))])
        if abs(theta - self._theta) < _EPS_RAD:
            return None
        n = max(1, math.ceil(abs(theta - self._theta) * max(reach, cell) / cell))
        t_of = np.asarray(np.linalg.inv(t_fo), dtype=np.float64)
        p_o, c_o = t_of[:3, :3] @ p + t_of[:3, 3], t_of[:3, :3] @ c
        added = [
            _moved(
                s.model_copy(update={"geometry_source": "grown"}),
                _rotation_about(c_o, p_o, self._theta + (theta - self._theta) * k / n),
            )
            for k in range(1, n + 1)
            for s in self._shape
        ]
        grown = self._union(added, stamp_ns, "protrusion")
        if grown is not None:
            self._theta = theta
        return grown

    def grow_untracked(
        self,
        store: TrackStoreLike,
        grid: VoxelLattice,
        positions: Mapping[str, float],
        stamp_ns: int,
    ) -> Binding | None:
        """Grow an UNTRACKED / PROVISIONAL / TRACKED payload by the cells that move with the hand.

        A fallback box cannot know how long an unknown object is (a banana outruns any
        stroke). Occupied cells within ``_GROW_RADIUS_M`` of the pinch centre that neither
        the hand nor another track explains, 26-connected to the payload and in front of
        the palm, are keyed in the attach link. A key seen again after the hand moved at
        least ``_RIGID_MIN_MOVE_CELLS`` cells, whose first grid-frame place is free on this
        grid, moved with the hand: it is held. The static world fails one of the two (a
        table under a hand sliding along it keeps its key but not its freed place). The new
        held cells' box (object frame, + half a cell) is unioned (K4: only grows).
        ``None`` when nothing new moved with the hand, or past the primitive cap.

        A TRACKED payload grows the same way (it only ever grows: K4), from half a cell of
        hand travel (``_TRACKED_MIN_MOVE_CELLS``) instead of two: the clear fit it starts from
        is measured, and the cells it takes are loose, 26-connected to it, in front of the
        palm and not another track's, the hand's, or a place the world still holds. Cells at or
        under the held tracks' ``floor_z`` (the support top + one cell) are the support and are
        never taken: the payload sweep clears them, which would make a flat support look held.
        A payload with no track of its own takes the highest ``floor_z`` in the scene.

        A cell whose place was occupied at the hold's first growth call, by anything but the
        payload's connected body or the hand, is the static world: it never grows the payload.
        Lowering a payload onto a plate, a plate rim cell (or an object resting on it) can take
        a key whose first place has since emptied (#418 R368: the orange swelled onto the plate
        and the orange already on it, and stopped on the rim). ponytail: a static thing the map
        did not hold at that call is still unguarded; the floor and other tracks cover the rest.
        """
        held_b = self.held
        if held_b is None or held_b.outcome not in (
            BindOutcome.UNTRACKED,
            BindOutcome.PROVISIONAL,
            BindOutcome.TRACKED,
        ):
            return None
        tracked = held_b.outcome is BindOutcome.TRACKED
        min_move = _TRACKED_MIN_MOVE_CELLS if tracked else _RIGID_MIN_MOVE_CELLS
        if grid.frame_id != self.frame_id:
            raise ROSConfigError(f"grid frame {grid.frame_id!r} != track frame {self.frame_id!r}.")
        cell = grid.resolution
        t_fa = self.hand.t_frame_from_attach(self.frame_id, positions)
        pinch = self.hand.pinch_in_link(positions[self.hand.joint.name])
        p = t_fa[:3, :3] @ pinch.centre + t_fa[:3, 3]
        occ = grid.occupied_centers()
        near = occ[np.linalg.norm(occ - p, axis=1) <= _GROW_RADIUS_M]
        if not len(near):
            return None
        held = self._held_boxes(positions)
        mine = set(held_b.candidates)
        others = [
            _box(t.box)
            for t in store.all()
            if t.object_id not in mine and t.status is not TrackStatus.BOUND
        ]

        def within(boxes: Sequence[Box]) -> NDArray[np.bool_]:
            return _outside_m(near, boxes) <= cell / 2.0 if boxes else np.zeros(len(near), bool)

        hand_cells = within(self.hand.hand_boxes(self.frame_id, positions))
        explained = hand_cells | within(others)
        # The support is never the payload: a cell at or under the clearing floor of a track
        # this payload is (the support top + one cell) is the table or shelf it stood on. The
        # payload sweep clears those cells from the octree while the payload overlaps them, so
        # their first place reads "free" and a flat support looks like it moved with the hand
        # (#418 r10dr_on2: the banana swelled onto the table, z <= 0, and stopped on it).
        floor = max(
            (t.floor_z for t in (store.get(i) for i in mine) if t is not None), default=-math.inf
        )
        if floor == -math.inf:
            # A payload with no track of its own (a pinch fallback) takes the scene's tracked
            # supports' floor (Isaac R84: it grew onto the table the sweep had cleared under the
            # hand and stopped on a cell 2 cm under the top). ponytail: the highest floor in the
            # scene; a lower shelf's payload then grows less, so it fails closed at the kernel.
            floor = max((t.floor_z for t in store.all()), default=-math.inf)
        loose = near[~explained & (near[:, 2] > floor)]
        if not len(loose):
            return None
        linked = self._connected(grid, loose, _outside_m(loose, held) <= cell)
        lattice = grid.rotation()
        origin = np.asarray(grid.origin)

        def ijk(x: NDArray[np.float64]) -> NDArray[np.int64]:
            return np.asarray(np.floor((x - origin) @ lattice / cell), dtype=np.int64)

        occupied = {tuple(v) for v in ijk(occ).tolist()}
        if self._static is None:
            body = {tuple(v) for v in ijk(np.vstack([linked, near[hand_cells]])).tolist()}
            self._static = occupied - body
        t_af = np.linalg.inv(t_fa)
        local = linked @ t_af[:3, :3].T + t_af[:3, 3]
        front = local @ pinch.axes[:, 1] >= pinch.palm_m  # nothing held lies behind the palm
        linked, local = linked[front], local[front]
        rigid = []
        for key, world in zip(
            np.floor(local / cell + 0.5).astype(int).tolist(), linked, strict=True
        ):
            first = self._sighted.setdefault(tuple(key), world)
            if (
                np.linalg.norm(world - first) >= min_move * cell
                and tuple(ijk(first[None])[0].tolist()) not in occupied
                and tuple(ijk(world[None])[0].tolist()) not in self._static
            ):
                rigid.append(world)
        new: NDArray[np.float64] = np.asarray(rigid, dtype=np.float64).reshape(-1, 3)
        new = new[_outside_m(new, held) > cell / 2.0] if len(new) else new
        if not len(new):
            return None
        t_of = np.linalg.inv(_t(self.object_pose(positions)))
        pts = new @ t_of[:3, :3].T + t_of[:3, 3]
        lo, hi = pts.min(axis=0) - cell / 2.0, pts.max(axis=0) + cell / 2.0
        prim = _prim(
            _frame((lo + hi) / 2.0, np.eye(3)),
            (hi - lo) / 2.0,
            held_b.attachment.object_id,
            "grown",
        )
        return self._union([prim], stamp_ns, "moves_with_hand")

    @staticmethod
    def _connected(
        grid: VoxelLattice, cells: NDArray[np.float64], seed: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """The ``cells`` 26-connected (on the lattice) to the ``seed`` ones."""
        ijk = np.floor(
            (cells - np.asarray(grid.origin)) @ grid.rotation() / grid.resolution
        ).astype(int)
        index = {tuple(v): i for i, v in enumerate(ijk.tolist())}
        todo = [i for i in np.flatnonzero(seed).tolist()]
        seen = set(todo)
        while todo:
            i, j, k = ijk[todo.pop()]
            for di, dj, dk in _N26:
                n = index.get((i + di, j + dj, k + dk))
                if n is not None and n not in seen:
                    seen.add(n)
                    todo.append(n)
        return cells[sorted(seen)]

    def on_release(self) -> None:
        """Forget the held payload (the track itself goes through ``release.release_track``)."""
        self.held, self._shape, self._theta = None, [], 0.0
        self._sighted, self._static = {}, None
