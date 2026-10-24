"""A provisional refit several times the object's known size is refused (#418, hazard OT-6/OT-14).

r16po_on1: the orange's provisional box (half 0.058 x 0.045) was replaced at sim 8.77 s by a refit
through the arm's occluder band, half 0.096 x 0.167 (33 cm for a 5.8 cm orange), and went on to be
the payload's measured part. The size prior comes only from CLEAR same-label evidence and never
blocks a clear fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from openral_core import FitKind, ObjectQuery, ObjectRole, PlaceRegion, TrackStatus
from openral_world_state.object_tracker import ObjectTracker, TrackerParams
from openral_world_state.object_tracker.fit import ContainingFit
from openral_world_state.object_tracker.tracker import CameraFrame, provisional_oversize

from tests.unit.test_object_tracker_hidden_side import (
    _FIX,
    _FRAME,
    _RES,
    _SUPPORT_Z,
    _box,
    _grid,
    _k,
)
from tests.unit.test_object_tracker_session import _BANANA, _S, _decl, _frame, _session

_FACTOR = TrackerParams().provisional_oversize_factor


def _tracker_with_clear_oranges() -> ObjectTracker:
    tr = ObjectTracker(_FRAME)
    tr.open_query("g", "q0", ObjectQuery(label="orange", count="all", role=ObjectRole.TARGET),
                  "deploy@0", stamp_ns=0)  # fmt: skip
    for name in ("box_trk2", "box_trk4"):  # the real r6 clear oranges
        t = tr.store.mint(
            label="orange", role=ObjectRole.TARGET, declaration_key="deploy@0", box=_box(name),
            tight=(), fit_kind=FitKind.SILHOUETTE_CONE, support_z=_SUPPORT_Z, support_ref="map",
            floor_z=_SUPPORT_Z + _RES, visible_cells=1, occupied_cells=1, last_fit_stamp_ns=1,
            last_confirmed_stamp_ns=1,
        )  # fmt: skip
        tr.store.replace(t.model_copy(update={"status": TrackStatus.CONFIRMED}))
    return tr


def test_the_prior_is_clear_same_label_extent_else_own_history_else_none() -> None:
    tr = ObjectTracker(_FRAME)
    assert tr._extent_prior(None, "orange") is None, "no clear evidence: nothing is blocked"
    tr = _tracker_with_clear_oranges()
    expected = max(2.0 * max(_box(n).half_extents[:2]) for n in ("box_trk2", "box_trk4"))
    assert tr._extent_prior(None, "orange") == (expected, "same_label_tracks")
    assert tr._extent_prior(None, "banana") is None, "a different label shares nothing"
    [first, _] = tr.store.all()
    tr._clear_extent[first.object_id] = 0.2  # its clear fits once showed more than its box now
    assert tr._extent_prior(None, "orange") == (0.2, "same_label_tracks")
    tr._clear_extent["trk:9"] = 0.07
    assert tr._extent_prior("trk:9", "orange") == (0.07, "own_clear_fit")


def test_the_r16po_on1_refit_is_oversize_and_its_previous_box_is_not() -> None:
    prior = _tracker_with_clear_oranges()._extent_prior(None, "orange")
    assert prior is not None
    assert provisional_oversize(2.0 * 0.167, prior[0], _FACTOR), "half 0.096 x 0.167 at 8.77 s"
    assert not provisional_oversize(2.0 * 0.058, prior[0], _FACTOR), "the box it replaced"
    assert not provisional_oversize(2.0 * 0.045, prior[0], _FACTOR)


@pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
def test_the_factor_is_the_only_threshold(factor: float) -> None:
    assert provisional_oversize(0.19, 0.09, factor) is (factor * 0.09 < 0.19)


def test_a_clear_refit_larger_than_the_prior_is_never_refused() -> None:
    """The rule lives behind ``fit.provisional``: a clear fit is accepted whatever the prior."""
    s = _session("droid_franka_robotiq")
    wall = 0.0
    s.on_grasp_declaration(_decl(100))
    for n in range(4):
        wall = _frame(s, [_BANANA], n * _S // 2, wall + 1)
    [track] = s.snapshot().objects
    assert track.status is TrackStatus.CONFIRMED and not track.provisional
    s.tracker._clear_extent[track.object_id] = 0.05  # a prior far below the banana's true size
    for n in range(4, 8):
        wall = _frame(s, [_BANANA], n * _S // 2, wall + 1)
    [after] = s.snapshot().objects
    assert 2.0 * max(after.box.half_extents[:2]) > _FACTOR * 0.05, "the clear box really is larger"
    assert after.status is TrackStatus.CONFIRMED and after.object_id == track.object_id


@pytest.mark.parametrize("moved", [False, True])
def test_an_unmoved_oversize_provisional_refit_is_refused_and_a_moved_one_is_not(
    moved: bool,
) -> None:
    """The gate in ``_refuse_refit``: provisional over provisional, the r16po_on1 numbers. A moved
    fit replaces the box it left (2026-10-21 decision), so the size rule never pins it."""
    tr = ObjectTracker(_FRAME)
    t = tr.store.mint(
        label="orange", role=ObjectRole.TARGET, declaration_key="deploy@0", box=_box("box_trk2"),
        tight=(), fit_kind=FitKind.SILHOUETTE_CONE, support_z=_SUPPORT_Z, support_ref="map",
        floor_z=_SUPPORT_Z + _RES, visible_cells=1, occupied_cells=1, last_fit_stamp_ns=1,
        last_confirmed_stamp_ns=1, provisional=True,
    )  # fmt: skip
    tr._clear_extent[t.object_id] = 2.0 * 0.058
    pose = _box("box_trk2").pose  # the refit sits on the real orange, 33 cm across
    box = PlaceRegion(frame_id=_FRAME, pose=pose, half_extents=(0.096, 0.167, 0.05))
    fit = ContainingFit(box=box, tight=None, surface=np.zeros((0, 3)), support_z=_SUPPORT_Z,
                        provisional=True)  # fmt: skip
    frame = CameraFrame(
        "top_depth", int(_FIX["stamp_ns"]), np.asarray(_FIX["depth_m"], dtype=np.float64), _k(),
        np.asarray(_FIX["t_base_from_cam"], dtype=np.float64),
        np.asarray(_FIX["robot_mask"], dtype=np.uint8),
    )  # fmt: skip
    refused = tr._refuse_refit(
        t, fit, frozenset(), pose.xyz, _SUPPORT_Z, _grid(), (), frame, moved
    )  # fmt: skip
    assert refused is not moved
