"""The track table, association by continuity and ageing in source time (issue #418, steps 5-6).

``TrackStore`` holds the ``TrackedObject`` rows and the set's ``revision``. Its three-method
contract — ``get`` / ``all`` / ``replace`` — is what the binding and release modules call;
``mint`` and ``drop`` are the tracker's own.

Free objects move only when contacted, so association is continuity (``associate``):

* a fit refreshes the live track whose box holds its centroid, or whose centre is within one
  voxel of it;
* two such tracks whose distances differ by at most one voxel are a **tie**, and the fit
  updates nothing (``nearest_primitive``'s rule, HZ-0115-2);
* a fit near exactly one LOST / RELEASED track inside its gate (one diameter; for a release,
  the fall column, growing with the release height) inherits that id with
  ``identity_inferred``;
* with ≤ 32 tracks no assignment solver is needed; a swap between identical instances is
  geometry-benign (OT-4).

Every lifetime runs on the **source** clock (depth stamps, grid ``source_stamp``), never on
the node clock: a 0.1x real-time sim does not stretch them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from openral_core import TrackedObject, TrackedObjectSet, TrackStatus
from openral_core.exceptions import ROSConfigError
from openral_core.object_primitives import point_box_gap_m

__all__ = ["LIVE", "TrackStore", "age_tracks", "associate"]

log = structlog.get_logger(__name__)

#: Statuses a fit can refresh by continuity.
LIVE = frozenset(
    {
        TrackStatus.TENTATIVE,
        TrackStatus.CONFIRMED,
        TrackStatus.OCCLUDED,
        TrackStatus.SPLIT,
        TrackStatus.BOUND,
    }
)
_NS = 1_000_000_000


class TrackStore:
    """The tracker's ``TrackedObject`` rows; every change bumps ``revision``.

    Contract for the binding / release modules: ``get(object_id)``, ``all()`` and
    ``replace(obj)``. ``replace`` swaps a row for an updated copy (``model_copy(update=...)``)
    under the same ``object_id``; it never mints, and refuses an epoch that goes backwards.

    Example:
        >>> from openral_core import FitKind, ObjectRole, PlaceRegion, Pose6D
        >>> s = TrackStore("base")
        >>> box = PlaceRegion(
        ...     frame_id="base",
        ...     half_extents=(0.03, 0.03, 0.03),
        ...     pose=Pose6D(xyz=(0.1, 0.0, 0.03), quat_xyzw=(0, 0, 0, 1), frame_id="base"),
        ... )
        >>> t = s.mint(
        ...     label="orange",
        ...     role=ObjectRole.TARGET,
        ...     declaration_key="obj:orange@0",
        ...     box=box,
        ...     fit_kind=FitKind.SILHOUETTE_CONE,
        ...     support_z=0.0,
        ...     support_ref="map",
        ...     floor_z=0.01,
        ...     visible_cells=5,
        ...     occupied_cells=0,
        ...     last_fit_stamp_ns=7,
        ...     last_confirmed_stamp_ns=0,
        ... )
        >>> t.object_id, t.status.value, s.revision
        ('trk:1', 'tentative', 1)
        >>> s.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
        >>> s.get("trk:1").status.value, s.revision
        ('confirmed', 2)
    """

    def __init__(self, frame_id: str) -> None:
        """Start empty, in the grid frame ``frame_id``."""
        self.frame_id = frame_id
        self._rows: dict[str, TrackedObject] = {}
        self._revision = 0
        self._next = 1

    @property
    def revision(self) -> int:
        """Strictly increasing: bumped by every ``mint`` / ``replace`` / ``drop``."""
        return self._revision

    def get(self, object_id: str) -> TrackedObject | None:
        """The row for ``object_id``, or ``None``."""
        return self._rows.get(object_id)

    def all(self) -> list[TrackedObject]:
        """Every row, in mint order."""
        return list(self._rows.values())

    def replace(self, obj: TrackedObject) -> None:
        """Swap the row ``obj.object_id`` for ``obj``; bumps ``revision``.

        Raises:
            ROSConfigError: On an unknown id, a frame mismatch or an epoch going backwards.
        """
        old = self._rows.get(obj.object_id)
        if old is None:
            raise ROSConfigError(f"TrackStore.replace: unknown track {obj.object_id!r}")
        if obj.box.frame_id != self.frame_id:
            raise ROSConfigError(
                f"TrackStore.replace: {obj.object_id} box in {obj.box.frame_id!r}, "
                f"not the store's {self.frame_id!r}"
            )
        if obj.epoch < old.epoch:
            raise ROSConfigError(
                f"TrackStore.replace: {obj.object_id} epoch {obj.epoch} < {old.epoch}"
            )
        if obj.status is not old.status:
            log.info(
                "object_tracker.status",
                object_id=obj.object_id,
                epoch=obj.epoch,
                from_status=old.status.value,
                to_status=obj.status.value,
            )
        self._rows[obj.object_id] = obj
        self._revision += 1

    def mint(self, **fields: Any) -> TrackedObject:  # noqa: ANN401  # reason: TrackedObject fields
        """Mint a TENTATIVE ``trk:<n>`` (never reused) from ``TrackedObject`` fields.

        Past ``TrackedObjectSet.MAX_OBJECTS`` the LOST row seen longest ago is dropped
        first; with none to drop the mint is refused.

        Raises:
            ROSConfigError: When the table is full of non-LOST tracks.
        """
        if len(self._rows) >= TrackedObjectSet.MAX_OBJECTS:
            lost = [r for r in self._rows.values() if r.status is TrackStatus.LOST]
            if not lost:
                raise ROSConfigError("TrackStore.mint: 32 live tracks; refusing a 33rd")
            self.drop(min(lost, key=lambda r: r.last_fit_stamp_ns).object_id)
        obj = TrackedObject(
            object_id=f"trk:{self._next}", epoch=0, status=TrackStatus.TENTATIVE, **fields
        )
        self._next += 1
        self._rows[obj.object_id] = obj
        self._revision += 1
        log.info("object_tracker.mint", object_id=obj.object_id, label=obj.label)
        return obj

    def drop(self, object_id: str) -> None:
        """Forget a row (its id is never reused)."""
        if self._rows.pop(object_id, None) is not None:
            self._revision += 1

    def snapshot(self, stamp_ns: int) -> TrackedObjectSet:
        """The whole table as one ``TrackedObjectSet``."""
        return TrackedObjectSet(
            frame_id=self.frame_id, stamp_ns=stamp_ns, revision=self._revision, objects=self.all()
        )


def _gap(track: TrackedObject, centroid: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(centroid - np.asarray(track.box.pose.xyz)))


def _holds(track: TrackedObject, centroid: NDArray[np.float64]) -> bool:
    return point_box_gap_m(centroid.tolist(), track.box) == 0.0


def associate(
    tracks: Sequence[TrackedObject], centroid: Sequence[float], *, voxel_m: float
) -> tuple[str | None, str]:
    """Which track a fit centred at ``centroid`` belongs to.

    Returns:
        ``(object_id, kind)`` with ``kind`` one of ``"continuity"``, ``"inferred"`` (a
        LOST / RELEASED track inside its gate), ``"new"`` (``object_id`` is ``None``) or
        ``"tie"`` (``None``: update nothing).

    Example:
        >>> from openral_core import FitKind, ObjectRole, PlaceRegion, Pose6D
        >>> def trk(n, x):
        ...     box = PlaceRegion(
        ...         frame_id="b",
        ...         half_extents=(0.03, 0.03, 0.03),
        ...         pose=Pose6D(xyz=(x, 0.0, 0.03), quat_xyzw=(0, 0, 0, 1), frame_id="b"),
        ...     )
        ...     return TrackedObject(
        ...         object_id=f"trk:{n}",
        ...         epoch=0,
        ...         label="orange",
        ...         role=ObjectRole.TARGET,
        ...         declaration_key="k",
        ...         status=TrackStatus.CONFIRMED,
        ...         box=box,
        ...         fit_kind=FitKind.SILHOUETTE_CONE,
        ...         support_z=0.0,
        ...         support_ref="map",
        ...         floor_z=0.0,
        ...         visible_cells=5,
        ...         occupied_cells=5,
        ...         last_fit_stamp_ns=0,
        ...         last_confirmed_stamp_ns=0,
        ...     )
        >>> ts = [trk(1, 0.0), trk(2, 0.05)]
        >>> associate(ts, (0.004, 0.0, 0.03), voxel_m=0.015)
        ('trk:1', 'continuity')
        >>> associate(ts, (0.025, 0.0, 0.03), voxel_m=0.015)
        (None, 'tie')
        >>> associate(ts, (0.3, 0.0, 0.03), voxel_m=0.015)
        (None, 'new')
    """
    c = np.asarray(centroid, dtype=np.float64)
    near = [t for t in tracks if t.status in LIVE and (_holds(t, c) or _gap(t, c) <= voxel_m)]
    if near:
        near.sort(key=lambda t: _gap(t, c))
        if len(near) > 1 and _gap(near[1], c) - _gap(near[0], c) <= voxel_m:
            log.info("object_tracker.tie", candidates=[t.object_id for t in near[:2]])
            return None, "tie"
        return near[0].object_id, "continuity"
    gated = []
    for t in tracks:
        diameter = 2.0 * max(t.box.half_extents[:2])
        if t.status is TrackStatus.LOST and _gap(t, c) <= diameter:
            gated.append(t)
        elif t.status is TrackStatus.RELEASED:
            drop = max(t.box.pose.xyz[2] - t.support_z, 0.0)  # the fall column
            horiz = float(np.linalg.norm(c[:2] - np.asarray(t.box.pose.xyz[:2])))
            if horiz <= max(diameter, drop) and c[2] <= t.box.pose.xyz[2] + voxel_m:
                gated.append(t)
    if len(gated) == 1:
        return gated[0].object_id, "inferred"
    if gated:
        log.info("object_tracker.tie", candidates=[t.object_id for t in gated])
        return None, "tie"
    return None, "new"


def age_tracks(
    store: TrackStore, now_ns: int, *, max_age_s: float, lost_s: float, keep_lost_s: float
) -> None:
    """Age every free track on the source clock ``now_ns``.

    CONFIRMED / OCCLUDED with no confirming grid for ``max_age_s`` become TENTATIVE
    (ineligible); TENTATIVE / SPLIT with neither a fit nor a confirmation for ``lost_s``
    become LOST; LOST rows are forgotten after ``keep_lost_s``. BOUND and RELEASED belong to
    binding / release and are not aged here.
    """
    for t in store.all():
        seen = max(t.last_fit_stamp_ns, t.last_confirmed_stamp_ns)
        if t.status in (TrackStatus.CONFIRMED, TrackStatus.OCCLUDED):
            if now_ns - t.last_confirmed_stamp_ns > max_age_s * _NS:
                store.replace(t.model_copy(update={"status": TrackStatus.TENTATIVE}))
        elif t.status in (TrackStatus.TENTATIVE, TrackStatus.SPLIT):
            if now_ns - seen > lost_s * _NS:
                store.replace(t.model_copy(update={"status": TrackStatus.LOST}))
        elif t.status is TrackStatus.LOST and now_ns - seen > keep_lost_s * _NS:
            store.drop(t.object_id)
