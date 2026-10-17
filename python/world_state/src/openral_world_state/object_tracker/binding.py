"""Grasp binding and in-hand update of tracked objects (issue #418).

At ATTACH the hand's pinch volume ``J_h`` (``pinch.HandModel``, at the measured jaw by
joint-state FK, grown by one cell) is intersected with the tracks:

* free tracks of **any** role whose measured geometry holds the pinch centre (``J_h``'s
  centre) within one cell — the tight k-DOP when the track has one, else its box — and
  ``RELEASED`` tracks whose fall column holds it (a regrasp before re-grounding). A padded
  box that merely *meets* the swept pinch volume is not enough: a tracked neighbour beside
  an ungrounded held object would be bound in its place (p2po_on2);
* exactly one: ``TRACKED_OBJECT`` binding of that id (``pose_in_link = T_link^-1 T_obj``,
  unioned with the same shape re-centred on the pinch axis; a ``RELEASED`` one is bound
  re-centred on the pinch volume under ``epoch + 1``);
* two or more: the union of the candidates and the pinch box, ``PINCH_FALLBACK``, no
  identity (nothing is cleared by identity);
* none: ``PINCH_FALLBACK``, the pad gap x pad width box past the fingertips, never toward
  the palm; logged ``untracked_grasp`` and a re-ground is asked for.

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
from typing import Protocol

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import (
    AttachedCollisionObject,
    AttachedCollisionPrimitive,
    AttachmentEvidenceKind,
    BoxShape,
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

from openral_world_state.object_tracker.pinch import HandModel
from openral_world_state.object_tracker.release import fall_column

__all__ = ["BindOutcome", "Binding", "GraspBinder", "TrackStoreLike"]

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
_FREE = {TrackStatus.TENTATIVE, TrackStatus.CONFIRMED, TrackStatus.OCCLUDED, TrackStatus.SPLIT}
_UNIT = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_N26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if i or j or k]


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
    t_obj_prim: NDArray[np.float64], half: Sequence[float] | NDArray[np.float64], object_id: str
) -> AttachedCollisionPrimitive:
    return AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=tuple(float(h) for h in half)),
        pose_in_object=_pose(t_obj_prim, object_id),
    )


def _moved(p: AttachedCollisionPrimitive, t: NDArray[np.float64]) -> AttachedCollisionPrimitive:
    """``p`` with ``t`` (object frame) applied to its pose."""
    return p.model_copy(
        update={"pose_in_object": _pose(t @ _t(p.pose_in_object), p.pose_in_object.frame_id)}
    )


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

    Raises:
        ROSConfigError: ``frame_id`` is not the manifest's ``base_frame``.
    """

    def __init__(self, hand: HandModel, *, frame_id: str, cell_m: float) -> None:
        """Bind nothing yet."""
        if frame_id != hand.description.base_frame:
            raise ROSConfigError(
                f"tracks in {frame_id!r} but FK roots at {hand.description.base_frame!r}."
            )
        self.hand, self.frame_id, self.cell_m = hand, frame_id, cell_m
        self.held: Binding | None = None
        self._shape: list[AttachedCollisionPrimitive] = []  # object frame, revision 0
        self._theta = 0.0

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
        self, store: TrackStoreLike, positions: Mapping[str, float], stamp_ns: int
    ) -> Binding:
        """Bind whatever meets ``J_h`` at the measured configuration."""
        jh = self.pinch_in_frame(positions)
        centre = jh[0]
        cands = [
            t
            for t in store.all()
            if (t.status in _FREE and _holds(t, centre, self.cell_m))
            or (
                t.status is TrackStatus.RELEASED
                and _box_holds(_box(fall_column(t, self.cell_m)), centre, self.cell_m)
            )
        ]
        jaw = positions[self.hand.joint.name]
        provisional = tuple(t.object_id for t in cands if t.provisional)
        if provisional:  # never payload geometry (TrackedObject.provisional)
            log.warning("object_tracker.provisional_grasp", hand=self.hand.joint.name,
                        candidates=provisional, jaw=jaw)  # fmt: skip
            cands = [t for t in cands if not t.provisional]
        ids = tuple(t.object_id for t in cands)
        if len(cands) == 1:
            binding = self._bind(store, cands[0], positions, stamp_ns)
        else:
            binding = self._fallback(cands, positions, stamp_ns)
            event = "object_tracker.ambiguous_grasp" if cands else "object_tracker.untracked_grasp"
            log.warning(event, hand=self.hand.joint.name, candidates=ids, jaw=jaw)
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
        shape = (
            [
                _moved(
                    h.model_copy(
                        update={
                            "pose_in_object": h.pose_in_object.model_copy(update={"frame_id": oid})
                        }
                    ),
                    np.asarray(np.linalg.inv(t_fo), dtype=np.float64),
                )
                for h in track.tight
            ]
            if track.tight
            else [_prim(np.eye(4), track.box.half_extents, oid)]
        )
        pinch = self.hand.pinch_in_link(positions[self.hand.joint.name])
        c = pinch.axes[:, 0]
        released = track.status is TrackStatus.RELEASED
        if released:  # its pose is stale: bind it re-centred on the pinch volume
            t_ao[:3, 3] = pinch.centre
            prims = list(shape)
        else:  # as fitted, unioned with the same shape slid onto the pinch axis
            shift = -float((t_ao[:3, 3] - pinch.centre) @ c) * c
            slide = _frame(t_ao[:3, :3].T @ shift, np.eye(3))
            prims = shape + (
                [_moved(p, slide) for p in shape] if np.linalg.norm(shift) > _EPS_M else []
            )
        epoch = track.epoch + 1 if released else track.epoch
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
        self, cands: Sequence[TrackedObject], positions: Mapping[str, float], stamp_ns: int
    ) -> Binding:
        oid = f"pinch:{self.hand.joint.name}@{stamp_ns}"
        fb = self.hand.fallback_in_link(positions[self.hand.joint.name])
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
            prims += [_prim(t_of @ _frame(c, r), h, oid) for c, r, h in boxes]
        self._shape, self._theta = prims[:1], 0.0
        attachment = AttachedCollisionObject(
            object_id=oid,
            attach_link=self.hand.attach_link,
            touch_links=list(self.hand.touch_links),
            primitives=prims,
            pose_in_link=_pose(t_ao, self.hand.attach_link),
            confidence=FALLBACK_CONFIDENCE,
            evidence_kind=AttachmentEvidenceKind.PINCH_FALLBACK,
            evidence_ref=("ambiguous:" + ",".join(t.object_id for t in cands))
            if cands
            else "untracked_grasp",
            stamp_ns=stamp_ns,
        )
        outcome = BindOutcome.AMBIGUOUS if cands else BindOutcome.UNTRACKED
        return Binding(outcome, attachment, tuple(t.object_id for t in cands), reground=not cands)

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
            _moved(s, _rotation_about(c_o, p_o, self._theta + (theta - self._theta) * k / n))
            for k in range(1, n + 1)
            for s in self._shape
        ]
        grown = self._union(added, stamp_ns, "protrusion")
        if grown is not None:
            self._theta = theta
        return grown

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
