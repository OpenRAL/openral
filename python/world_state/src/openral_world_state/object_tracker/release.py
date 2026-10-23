"""Release of a bound tracked object (issue #418, "Release").

* ``JawReleaseMonitor``: the jaw-open trigger (the kernel's K5 rule, mirrored for the
  tracker): the measured jaw opened past ``hold + release_gap`` for ``release_dwell``.
  In t9 the record stayed attached with the jaw at 0.988, waiting for a DETACH.
* ``release_track``: the track goes ``RELEASED`` at the release pose, for the release
  window only (the kernel freezes the payload there); then the session retires it. A
  released object is never followed: it may roll or fall, and the target query finds it
  afresh under a new id (2026-10-22, Spark R177-R185).

All stamps are source-clock nanoseconds (depth / grid stamps), never the node clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import PlaceRegion, Pose6D, TrackedObject, TrackStatus
from openral_core.geometry import homogeneous_from_quat_xyz, yaw_to_quat_xyzw

if TYPE_CHECKING:
    from openral_world_state.object_tracker.binding import TrackStoreLike
    from openral_world_state.object_tracker.pinch import HandModel

__all__ = ["JawReleaseMonitor", "release_track"]

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
