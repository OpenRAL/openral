"""The tracker combines its depth cameras' fits of one object (issue #418 phase 4 wiring).

``intersect_views`` (multi_view.py) was only reachable from tests: the tracker kept the
latest single-camera fit. A refit of a track from camera B within ``VIEW_WINDOW_S`` of a
fit from camera A is now intersected with A's cone, so the track box shrinks to what both
views allow and still holds the solid. Camera A is the PickOrange top camera from its
committed scene; camera B is the same camera turned 90 deg about the object.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_core import FitKind, ObjectQuery, ObjectRole
from openral_world_state.object_tracker import ObjectTracker

from tests.unit._tracker_scenes import RIGS, inside_hull, sphere, step, surface_samples
from tests.unit.test_object_tracker_multi_view import _hull, _turned, _volume

pytest.importorskip("scipy", reason="the hull containment check uses scipy ConvexHull")


def _tracker(frame: str) -> ObjectTracker:
    tr = ObjectTracker(frame)
    tr.open_query("g", "q0", ObjectQuery(label="orange", count="one", role=ObjectRole.TARGET),
                  "obj:orange@0", stamp_ns=0)  # fmt: skip
    return tr


def test_a_second_cameras_refit_intersects_the_first_cameras_cone() -> None:
    top = RIGS["so101_top"]()
    side = _turned(top, math.pi / 2)
    x, y = top.centre_xy
    ball = sphere((x, y, top.support_z + 0.028), 0.028)
    tr = _tracker(top.frame)
    step(tr, top, [ball], stamp_ns=1_000_000_000, wall_s=0.0)
    (first,) = tr.store.all()
    assert first.fit_kind is FitKind.SILHOUETTE_CONE
    step(tr, side, [ball], stamp_ns=1_200_000_000, wall_s=10.0)
    (both,) = tr.store.all()
    assert both.object_id == first.object_id
    assert both.fit_kind is FitKind.MULTI_VIEW_CONE
    assert _volume(both.box) < 0.95 * _volume(first.box)
    assert inside_hull(surface_samples(ball), _hull_of(both)).all()


def test_views_too_far_apart_in_time_are_not_combined() -> None:
    top = RIGS["so101_top"]()
    side = _turned(top, math.pi / 2)
    x, y = top.centre_xy
    ball = sphere((x, y, top.support_z + 0.028), 0.028)
    tr = _tracker(top.frame)
    step(tr, top, [ball], stamp_ns=1_000_000_000, wall_s=0.0)
    step(tr, side, [ball], stamp_ns=3_000_000_000, wall_s=10.0)
    (track,) = tr.store.all()
    assert track.fit_kind is FitKind.SILHOUETTE_CONE


def _hull_of(track: object) -> np.ndarray:  # type: ignore[type-arg]  # reason: test helper
    from openral_core.geometry import homogeneous_from_quat_xyz

    prim = track.tight[0]  # type: ignore[attr-defined]  # reason: a TrackedObject
    v = np.asarray(prim.tight_geometry.hull_vertices_m)
    t = homogeneous_from_quat_xyz(prim.pose_in_object.xyz, prim.pose_in_object.quat_xyzw)
    return np.asarray(v @ t[:3, :3].T + t[:3, 3])


del _hull  # the multi-view module test's helper reads a ContainingFit, not a track
