"""Release of a bound tracked object and its return to tracking (issue #418, "Release").

* ``JawReleaseMonitor``: the jaw-open trigger (the kernel's K5 rule, mirrored for the
  tracker): the measured jaw opened past ``hold + release_gap`` for ``release_dwell``.
  In t9 the record stayed attached with the jaw at 0.988, waiting for a DETACH.
* ``release_track``: the track goes ``RELEASED`` at the release pose.
* ``fall_column``: where a released object can have landed; its horizontal gate grows
  with the release height (g20 / t5 mid-air drops).
* ``reassociate_released``: a fit confirmed in exactly one fall column returns that
  track ``CONFIRMED`` under the same id, ``epoch + 1``, on its new support.

All stamps are source-clock nanoseconds (depth / grid stamps), never the node clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import FitKind, PlaceRegion, Pose6D, TrackedObject, TrackStatus
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw

if TYPE_CHECKING:
    from openral_world_state.object_tracker.binding import TrackStoreLike
    from openral_world_state.object_tracker.pinch import HandModel

__all__ = ["JawReleaseMonitor", "fall_column", "reassociate_released", "release_track"]

log = structlog.get_logger(__name__)

_UNIT = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


@dataclass
class JawReleaseMonitor:
    """Jaw-open release: the jaw further from closed than ``hold + release_gap``, held.

    Attributes:
        hold: Jaw position the trigger held at (the stall), joint units.
        closed: The joint's ``closed_position``.
        release_gap: How much further open than ``hold`` reads as open.
        release_dwell_ns: How long it must stay open (source clock).

    Example:
        >>> m = JawReleaseMonitor(hold=0.45, closed=0.0, release_gap=0.075, release_dwell_ns=10)
        >>> m.observe(0.99, 0), m.observe(0.99, 5), m.observe(0.99, 10)
        (False, False, True)
    """

    hold: float
    closed: float
    release_gap: float
    release_dwell_ns: int
    _since_ns: int | None = None
    _last_ns: int | None = None

    @classmethod
    def for_hand(cls, hand: HandModel, *, hold: float, release_dwell_s: float) -> JawReleaseMonitor:
        """``release_gap`` is half the manifest's ``stall_gap`` (its release hysteresis)."""
        cal = hand.joint.closure_calibration
        assert cal is not None  # reason: HandModel refuses an uncalibrated jaw
        return cls(hold, hand.closed, cal.stall_gap / 2.0, int(release_dwell_s * 1e9))

    def observe(self, jaw: float, stamp_ns: int) -> bool:
        """Fold one measured jaw sample in; ``True`` once open for the whole dwell."""
        if self._last_ns is not None and stamp_ns < self._last_ns:
            self._since_ns = None  # the source clock went back (a sim reset): start over
        self._last_ns = stamp_ns
        if abs(jaw - self.closed) - abs(self.hold - self.closed) <= self.release_gap:
            self._since_ns = None
            return False
        if self._since_ns is None:
            self._since_ns = stamp_ns
        return stamp_ns - self._since_ns >= self.release_dwell_ns


def _gravity_box(frame_id: str, t: NDArray[np.float64], half: NDArray[np.float64]) -> PlaceRegion:
    """The yaw-only box containing the oriented box ``(t, half)`` (a carried body tilts)."""
    corners = (_UNIT * half) @ t[:3, :3].T + t[:3, 3]
    yaw = math.atan2(float(t[1, 0]), float(t[0, 0]))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = corners @ rot
    lo, hi = local.min(axis=0), local.max(axis=0)
    centre = rot @ ((lo + hi) / 2.0)
    return PlaceRegion(
        frame_id=frame_id,
        pose=Pose6D(xyz=tuple(centre.tolist()), quat_xyzw=yaw_to_quat_xyzw(yaw), frame_id=frame_id),
        half_extents=tuple(((hi - lo) / 2.0).tolist()),
        evidence_ref="release_pose",
    )


def release_track(
    store: TrackStoreLike,
    object_id: str,
    *,
    object_pose: Pose6D,
    support_z: float,
    cell_m: float,
    stamp_ns: int,
) -> TrackedObject | None:
    """Mark a bound track ``RELEASED`` at its release pose and return it.

    Args:
        store: The track set.
        object_id: The bound track.
        object_pose: The object frame (the box's) in the grid frame at release:
            ``FK(q) . pose_in_link`` (``GraspBinder.object_pose``).
        support_z: The support top measured under the release pose: the fall column's
            floor.
        cell_m: Grid resolution.
        stamp_ns: Release stamp.

    Returns:
        The released track, ``None`` when ``object_id`` is not bound, or the gravity-aligned
        box at the release pose breaks the caps (then the track is ``LOST``, logged).
    """
    track = store.get(object_id)
    if track is None or track.status is not TrackStatus.BOUND:
        return None
    t = homogeneous_from_quat_xyz(object_pose.xyz, object_pose.quat_xyzw)
    box = _gravity_box(track.box.frame_id, t, np.asarray(track.box.half_extents))
    base = {"bound_hand": None, "attach_link": None, "pose_in_link": None, "tight": ()}
    try:
        released = TrackedObject.model_validate(
            {
                **track.model_dump(),
                **base,
                "status": TrackStatus.RELEASED,
                "box": box,
                "support_z": support_z,
                "support_ref": "map",
                "floor_z": support_z + cell_m,
                "last_fit_stamp_ns": max(track.last_fit_stamp_ns, stamp_ns),
            }
        )
    except ValueError as exc:  # the tilted body's yaw box breaks the caps
        log.warning("object_tracker.release_unrepresentable", object_id=object_id, error=str(exc))
        released = track.model_copy(update={**base, "status": TrackStatus.LOST})
    store.replace(released)
    log.info("object_tracker.released", object_id=object_id, status=released.status.value)
    return released if released.status is TrackStatus.RELEASED else None


def fall_column(track: TrackedObject, cell_m: float) -> PlaceRegion:
    """Where a ``RELEASED`` track can have landed: the release box down to its support.

    The horizontal gate is one cell plus the release height (a 45 deg cone).

    ponytail: no bounce/roll model; widen the gate if t5-like drops land outside it.

    Example:
        >>> box = PlaceRegion(
        ...     frame_id="base",
        ...     pose=Pose6D(xyz=(0.2, 0.0, 0.13), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ...     half_extents=(0.03, 0.03, 0.03),
        ... )
        >>> t = TrackedObject(
        ...     object_id="trk:1",
        ...     epoch=0,
        ...     label="orange",
        ...     role="target",
        ...     declaration_key="k",
        ...     status="released",
        ...     box=box,
        ...     fit_kind="silhouette_cone",
        ...     support_z=0.0,
        ...     support_ref="map",
        ...     floor_z=0.01,
        ...     visible_cells=4,
        ...     occupied_cells=4,
        ...     last_fit_stamp_ns=1,
        ...     last_confirmed_stamp_ns=1,
        ... )
        >>> [round(v, 3) for v in fall_column(t, 0.01).half_extents]
        [0.14, 0.14, 0.09]
    """
    b = track.box
    top, low = b.pose.xyz[2] + b.half_extents[2], b.pose.xyz[2] - b.half_extents[2]
    bottom = min(low, track.support_z)
    gate = cell_m + max(0.0, low - track.support_z)
    cap = PlaceRegion.MAX_HALF_EXTENT_M
    return b.model_copy(
        update={
            "pose": b.pose.model_copy(
                update={"xyz": (b.pose.xyz[0], b.pose.xyz[1], (top + bottom) / 2.0)}
            ),
            "half_extents": (
                min(b.half_extents[0] + gate, cap),
                min(b.half_extents[1] + gate, cap),
                min((top - bottom) / 2.0 + cell_m, cap),
            ),
            "evidence_ref": f"fall_column:{track.object_id}",
        }
    )


def reassociate_released(
    store: TrackStoreLike,
    fit: PlaceRegion,
    *,
    fit_kind: FitKind,
    support_z: float,
    support_ref: str,
    visible_cells: int,
    occupied_cells: int,
    cell_m: float,
    stamp_ns: int,
) -> TrackedObject | None:
    """Return the one ``RELEASED`` track whose fall column holds a confirmed fit's centre.

    The caller has already map-confirmed ``fit`` (size-aware cover, one body). Exactly one
    column must hold it: two is ambiguous and binds nothing (logged), none is a new object.
    The track returns ``CONFIRMED`` under its id, ``epoch + 1`` (the kernel retired the
    old epoch at release), ``identity_inferred``, on the new support.
    """
    centre = np.asarray(fit.pose.xyz)
    hits = []
    for track in store.all():
        if track.status is not TrackStatus.RELEASED:
            continue
        col = fall_column(track, cell_m)
        t = homogeneous_from_quat_xyz(col.pose.xyz, col.pose.quat_xyzw)
        if np.all(np.abs((centre - t[:3, 3]) @ t[:3, :3]) <= np.asarray(col.half_extents)):
            hits.append(track)
    if len(hits) != 1:
        if hits:
            log.info("object_tracker.reassociate_ambiguous", ids=[t.object_id for t in hits])
        return None
    track = hits[0]
    confirmed = track.model_copy(
        update={
            "status": TrackStatus.CONFIRMED,
            "epoch": track.epoch + 1,
            "box": fit,
            "fit_kind": fit_kind,
            "support_z": support_z,
            "support_ref": support_ref,
            "floor_z": support_z + cell_m,
            "visible_cells": visible_cells,
            "occupied_cells": occupied_cells,
            "last_fit_stamp_ns": stamp_ns,
            "last_confirmed_stamp_ns": stamp_ns,
            "identity_inferred": True,
        }
    )
    confirmed = TrackedObject.model_validate(confirmed.model_dump())
    store.replace(confirmed)
    log.info("object_tracker.reconfirmed", object_id=track.object_id, epoch=confirmed.epoch)
    return confirmed
