"""A refit never grows a track past its first clear fit; a refused refit inside the box re-dates it.

Restock 2026-10-30 (Spark, arms out of view): a 6 x 6 cm carton's box went 14 -> 24 -> 39 cm. A
mask that spills onto a neighbour or the support always has points outside the box, so it read
as "moved" and replaced the box with no size gate. The bloated box then refused the later,
smaller refits as partial, so its fit aged to 9.4 s and the grasp bound the pinch fallback.
Fixtures: the recorded PickOrange frame and its clear orange tracks.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray
from openral_core import FitKind, ObjectRole, PlaceRegion, TrackedObject, TrackStatus
from openral_world_state.object_tracker import ObjectTracker, TrackerParams
from openral_world_state.object_tracker.fit import ContainingFit
from openral_world_state.object_tracker.tracker import CameraFrame

from tests.unit.test_object_tracker_hidden_side import (
    _FIX,
    _FRAME,
    _RES,
    _SUPPORT_Z,
    _box,
    _grid,
    _k,
)

_FACTOR = TrackerParams().refit_growth_factor


def _frame(stamp_ns: int) -> CameraFrame:
    return CameraFrame(
        "top_depth", stamp_ns, np.asarray(_FIX["depth_m"], dtype=np.float64), _k(),
        np.asarray(_FIX["t_base_from_cam"], dtype=np.float64),
        np.asarray(_FIX["robot_mask"], dtype=np.uint8),
    )  # fmt: skip


def _tracker(status: TrackStatus = TrackStatus.CONFIRMED) -> tuple[ObjectTracker, TrackedObject]:
    tr = ObjectTracker(_FRAME)
    t = tr.store.mint(
        label="orange", role=ObjectRole.TARGET, declaration_key="deploy@0", box=_box("box_trk2"),
        tight=(), fit_kind=FitKind.SILHOUETTE_CONE, support_z=_SUPPORT_Z, support_ref="map",
        floor_z=_SUPPORT_Z + _RES, visible_cells=1, occupied_cells=1, last_fit_stamp_ns=1,
        last_confirmed_stamp_ns=1,
    )  # fmt: skip
    t = t.model_copy(update={"status": status})
    tr.store.replace(t)
    return tr, t


def _scaled(
    box: PlaceRegion, k: float, surface: NDArray[np.float64] | None = None
) -> ContainingFit:
    h = box.half_extents
    grown = box.model_copy(update={"half_extents": (k * h[0], k * h[1], h[2])})
    return ContainingFit(
        box=grown, tight=None, support_z=_SUPPORT_Z,
        surface=np.zeros((0, 3)) if surface is None else surface,
    )  # fmt: skip


def _first(tr: ObjectTracker, t: TrackedObject) -> float:
    tr._first_extent[t.object_id] = (2.0 * max(t.box.half_extents[:2]), True)
    return tr._first_extent[t.object_id][0]


@pytest.mark.parametrize("moved", [False, True])
@pytest.mark.parametrize("k", [1.2, _FACTOR, 1.6, 2.6])
def test_a_refit_past_the_factor_is_refused_moved_or_not(moved: bool, k: float) -> None:
    tr, t = _tracker()
    _first(tr, t)
    fit = _scaled(t.box, k)
    refused = tr._overgrown(t.object_id, fit, moved)
    assert refused is (k > _FACTOR)
    if k > _FACTOR:  # through _refuse_refit too, before any other gate
        assert tr._refuse_refit(t, fit, frozenset(), t.box.pose.xyz, _SUPPORT_Z, _grid(), (),
                                _frame(2), moved)  # fmt: skip


def test_without_a_clear_fit_there_is_no_reference_and_nothing_is_refused() -> None:
    tr, t = _tracker()
    assert not tr._overgrown(t.object_id, _scaled(t.box, 5.0), True)


def test_the_reference_is_the_first_clear_fit_so_admitted_growth_cannot_ratchet() -> None:
    """Two 1.4x moved refits in a row: the second is 1.96x the first fit and is refused."""
    tr, t = _tracker()
    first = _first(tr, t)
    step = _scaled(t.box, 1.4)
    assert not tr._overgrown(t.object_id, step, True)
    tr._commit_fit(t, step, frozenset(), t.box.pose.xyz, _grid(), 2, (), _frame(2))
    assert tr._first_extent[t.object_id] == (first, True), "an admitted refit never moves it"
    assert step.box is not None
    assert tr._overgrown(t.object_id, _scaled(step.box, 1.4), True)


def _prov(box: PlaceRegion, k: float) -> ContainingFit:
    return dataclasses.replace(_scaled(box, k), provisional=True)


def test_a_provisional_first_fit_holds_provisional_refits_until_a_clear_fit_replaces_it() -> None:
    """2026-10-31 replay down_s16: a track minted through the hand grew 26 -> 40 cm unchecked."""
    tr, t = _tracker()
    w = 2.0 * max(t.box.half_extents[:2])
    tr._commit_fit(t, _prov(t.box, 1.0), frozenset(), t.box.pose.xyz, _grid(), 2, (), _frame(2))
    assert tr._first_extent[t.object_id] == pytest.approx((w, False))
    assert tr._overgrown(t.object_id, _prov(t.box, 1.6), True), "provisional growth is gated"
    big = _scaled(t.box, 2.0)
    assert not tr._overgrown(t.object_id, big, True), "a clear fit is never refused against one"
    tr._commit_fit(t, big, frozenset(), t.box.pose.xyz, _grid(), 3, (), _frame(3))
    assert tr._first_extent[t.object_id] == pytest.approx((2.0 * w, True))
    tr._commit_fit(t, _scaled(t.box, 1.0), frozenset(), t.box.pose.xyz, _grid(), 4, (), _frame(4))
    assert tr._first_extent[t.object_id] == pytest.approx((2.0 * w, True)), "the first clear stays"


def _surface(box: PlaceRegion, offset_m: float) -> NDArray[np.float64]:
    """Eight points around ``box``'s centre, shifted along x by ``offset_m``."""
    c = np.asarray(box.pose.xyz, dtype=np.float64)
    h = 0.5 * np.asarray(box.half_extents, dtype=np.float64)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    out: NDArray[np.float64] = c + corners * h + np.array([offset_m, 0.0, 0.0])
    return out


def test_a_refused_refit_measured_inside_the_box_re_dates_it() -> None:
    tr, t = _tracker()
    fit = _scaled(t.box, 0.8, _surface(t.box, 0.0))
    assert tr._redate_if_inside(t, fit, _grid(), _frame(5_000_000_000))
    stored = tr.store.get(t.object_id)
    assert stored is not None and stored.last_fit_stamp_ns == 5_000_000_000
    assert stored.box == t.box, "the box is kept; only its fit is re-dated"


def test_a_refused_refit_measured_elsewhere_does_not_re_date_it() -> None:
    tr, t = _tracker()
    away = 3.0 * t.box.half_extents[0] + 2 * _RES
    fit = _scaled(t.box, 0.8, _surface(t.box, away))
    assert not tr._redate_if_inside(t, fit, _grid(), _frame(5_000_000_000))
    stored = tr.store.get(t.object_id)
    assert stored is not None and stored.last_fit_stamp_ns == 1


def test_an_occluded_refit_inside_the_box_re_dates_it() -> None:
    tr, t = _tracker(TrackStatus.OCCLUDED)
    _first(tr, t)
    fit = _scaled(t.box, 0.9, _surface(t.box, 0.0))
    assert tr._refuse_refit(t, fit, frozenset(), t.box.pose.xyz, _SUPPORT_Z, _grid(), (),
                            _frame(7_000_000_000), False)  # fmt: skip
    stored = tr.store.get(t.object_id)
    assert stored is not None and stored.last_fit_stamp_ns == 7_000_000_000
    assert stored.box == t.box


def test_a_moved_refit_through_the_robot_keeps_the_clear_size() -> None:
    """2026-10-31: a refit through the hand replaced the clear box with one 2-4x off in size.
    It may move the box (the object was pushed), never resize it."""
    tr, t = _tracker()
    _first(tr, t)
    shifted = t.box.model_copy(
        update={"pose": t.box.pose.model_copy(update={"xyz": (
            t.box.pose.xyz[0] + 0.03, t.box.pose.xyz[1] - 0.01, t.box.pose.xyz[2] + 0.02)})},
    )  # fmt: skip
    small = shifted.model_copy(update={"half_extents": tuple(0.6 * h for h in t.box.half_extents)})
    fit = ContainingFit(box=small, tight=None, surface=np.zeros((0, 3)), support_z=_SUPPORT_Z,
                        provisional=True)  # fmt: skip
    track, _ = tr._refit_track(t, "seen", "top_depth", fit, frozenset(), _frame(3), _grid(), (),
                               True, support_z=_SUPPORT_Z, floor=_SUPPORT_Z + _RES)  # fmt: skip
    assert track.box.half_extents == t.box.half_extents, "the size is the clear fit's"
    assert track.box.pose.xyz == pytest.approx(
        (shifted.pose.xyz[0], shifted.pose.xyz[1], t.box.pose.xyz[2])
    ), "re-centred in x/y only"
    assert track.provisional, "a pushed object's box stays no payload (2026-10-21 decision)"
    assert track.last_fit_stamp_ns == 3


def test_a_moved_clear_refit_still_replaces_the_box() -> None:
    tr, t = _tracker()
    _first(tr, t)
    fit = _scaled(t.box, 1.2)
    track, _ = tr._refit_track(t, "seen", "top_depth", fit, frozenset(), _frame(3), _grid(), (),
                               True, support_z=_SUPPORT_Z, floor=_SUPPORT_Z + _RES)  # fmt: skip
    assert fit.box is not None and track.box == fit.box


def test_a_growth_refused_refit_mostly_inside_the_box_re_dates_it() -> None:
    """2026-10-31 restock arms-out: a mask spilling onto a touching carton was refused every
    frame and the clear target box aged to 9.4 s at the grasp."""
    tr, t = _tracker()
    _first(tr, t)
    inside = _surface(t.box, 0.0)
    spill = _surface(t.box, 3.0 * t.box.half_extents[0] + 2 * _RES)[:3]  # 3 of 11 outside
    fit = _scaled(t.box, 2.0, np.vstack([inside, spill]))
    assert tr._refuse_refit(t, fit, frozenset(), t.box.pose.xyz, _SUPPORT_Z, _grid(), (),
                            _frame(9_000_000_000), True)  # fmt: skip
    stored = tr.store.get(t.object_id)
    assert stored is not None and stored.last_fit_stamp_ns == 9_000_000_000
    assert stored.box == t.box, "re-dated, never resized"


def test_a_growth_refused_refit_mostly_elsewhere_does_not_re_date_it() -> None:
    tr, t = _tracker()
    _first(tr, t)
    away = _surface(t.box, 3.0 * t.box.half_extents[0] + 2 * _RES)
    fit = _scaled(t.box, 2.0, np.vstack([_surface(t.box, 0.0)[:2], away]))
    assert tr._refuse_refit(t, fit, frozenset(), t.box.pose.xyz, _SUPPORT_Z, _grid(), (),
                            _frame(9_000_000_000), True)  # fmt: skip
    stored = tr.store.get(t.object_id)
    assert stored is not None and stored.last_fit_stamp_ns == 1


def test_a_covered_move_within_a_cell_keeps_the_clear_box_and_stays_bindable() -> None:
    """2026-10-31 restock mix_03: a 1 cm "move" through the hand turned the target provisional
    and the grasp bound the pinch fallback."""
    tr, t = _tracker()
    _first(tr, t)
    nudged = t.box.model_copy(
        update={"pose": t.box.pose.model_copy(update={"xyz": (
            t.box.pose.xyz[0] + 0.5 * _RES, t.box.pose.xyz[1], t.box.pose.xyz[2])})}
    )  # fmt: skip
    fit = ContainingFit(box=nudged, tight=None, surface=np.zeros((0, 3)), support_z=_SUPPORT_Z,
                        provisional=True)  # fmt: skip
    track, _ = tr._refit_track(t, "seen", "top_depth", fit, frozenset(), _frame(3), _grid(), (),
                               True, support_z=_SUPPORT_Z, floor=_SUPPORT_Z + _RES)  # fmt: skip
    assert track.box == t.box
    assert not track.provisional
    assert track.last_fit_stamp_ns == 3


def test_the_growth_reference_leaves_the_clear_history_to_every_clear_fit() -> None:
    """The oversize guard's clear history takes every clear fit and no provisional one."""
    tr, t = _tracker()
    tr._commit_fit(t, _prov(t.box, 3.0), frozenset(), t.box.pose.xyz, _grid(), 2, (), _frame(2))
    assert t.object_id not in tr._clear_extent, "a provisional fit is no clear history"
    w = 2.0 * max(t.box.half_extents[:2])
    for k, stamp in ((1.0, 3), (1.3, 4)):
        tr._commit_fit(t, _scaled(t.box, k), frozenset(), t.box.pose.xyz, _grid(), stamp, (),
                       _frame(stamp))  # fmt: skip
    assert tr._clear_extent[t.object_id] == pytest.approx(1.3 * w)
    assert tr._first_extent[t.object_id] == pytest.approx((w, True))
