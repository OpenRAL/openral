"""Object-centric tracker core (issue #418) — ROS-free; the lifecycle node wraps it.

Grounds prompt-named objects as standing queries, fits each mask to a set that
*contains* the object (``fit``), confirms it on the voxel map with size-aware cover
(``cover``), tracks identity by continuity in source time (``tracks``), and schedules one
segmenter request at a time (``scheduler``). ``ObjectTracker.feed`` is the whole API: one
depth frame in, at most one request plus the ``TrackedObjectSet`` out.

**Interface for binding and release** (``binding.py`` / ``release.py`` in this package):
they mutate tracks only through ``ObjectTracker.store``, a ``TrackStore`` with exactly

* ``get(object_id) -> TrackedObject | None``;
* ``all() -> list[TrackedObject]`` (mint order);
* ``replace(obj: TrackedObject) -> None`` — swaps the row ``obj.object_id`` for ``obj``
  (build it with ``model_copy(update=...)``) and bumps the set's ``revision``. It never
  mints; it raises ``ROSConfigError`` on an unknown id, a box outside the store's frame or
  an ``epoch`` going backwards.

The tracker's side of that contract: it never fits, ages or confirms a BOUND track (a fit
that associates with one is handed back in ``TrackerOutput.bound_fits`` for the in-hand
update); a track set to RELEASED is searched for in its fall column, needs a fresh camera
fit there, and returns CONFIRMED with ``epoch + 1`` once the map confirms it.
"""

from openral_world_state.object_tracker.cover import (
    CoverVerdict,
    cover_verdict,
    own_cover,
    surface_cells,
)
from openral_world_state.object_tracker.fit import (
    ROBOT_PIXEL,
    UNCERTAIN_PIXEL,
    ContainingFit,
    FitParams,
    containing_fit,
    project_box,
    visible_fraction,
)
from openral_world_state.object_tracker.grounding import QueryBook, QueryEscalation, StandingQuery
from openral_world_state.object_tracker.multi_view import MultiViewFit, intersect_views
from openral_world_state.object_tracker.scheduler import Scheduler, TrackerRequest
from openral_world_state.object_tracker.session import GraspEventKind, TrackerSession, TrackingMode
from openral_world_state.object_tracker.tracker import (
    CameraFrame,
    ObjectTracker,
    TrackerOutput,
    TrackerParams,
    TrackerResult,
)
from openral_world_state.object_tracker.tracks import TrackStore, age_tracks, associate

__all__ = [
    "ROBOT_PIXEL",
    "UNCERTAIN_PIXEL",
    "CameraFrame",
    "ContainingFit",
    "CoverVerdict",
    "FitParams",
    "GraspEventKind",
    "MultiViewFit",
    "ObjectTracker",
    "QueryBook",
    "QueryEscalation",
    "Scheduler",
    "StandingQuery",
    "TrackStore",
    "TrackerOutput",
    "TrackerParams",
    "TrackerRequest",
    "TrackerResult",
    "TrackerSession",
    "TrackingMode",
    "age_tracks",
    "associate",
    "containing_fit",
    "cover_verdict",
    "intersect_views",
    "own_cover",
    "project_box",
    "surface_cells",
    "visible_fraction",
]
