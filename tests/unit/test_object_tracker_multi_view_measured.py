"""Multi-view under the default ``FitParams`` (``extent="measured"``): views combine by union.

A measured box holds what its camera saw, so ``intersect_views`` used to read each second
view's cone off its tight hull, which only a containing fit lays out as a cone: every
measured second view was refused ``cone_layout`` and the tracker never combined two cameras.
The fit now carries its silhouette cone, and measured views combine by the union of what they
saw, a second view's points kept only inside the primary's cone (spill out of it is dropped).
Rigs and solids as ``test_object_tracker_multi_view``: a scene's real depth camera, and the
same camera turned about the vertical through the object.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from openral_core import FitKind, ObjectQuery, ObjectRole, PlaceRegion
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_world_state.object_tracker import (
    ContainingFit,
    FitParams,
    ObjectTracker,
    containing_fit,
    intersect_views,
)

from tests.unit._tracker_scenes import RES, RIGS, Rig, box, render, step, surface_samples

pytest.importorskip("scipy", reason="the multi-view rigs import scipy-backed helpers")
_turned = pytest.importorskip("tests.unit.test_object_tracker_multi_view")._turned


def _fit(rig: Rig, solids: list[object], labels_in: tuple[int, ...] = (1,)) -> ContainingFit:
    depth, labels = render(solids, rig)  # type: ignore[arg-type]
    return containing_fit(
        np.isin(labels, labels_in),
        depth,
        np.zeros(depth.shape, np.uint8),
        rig.k,
        rig.t,
        support_z=rig.support_z,
        resolution=RES,
        frame_id=rig.frame,
        stamp_ns=1,
        params=FitParams(),  # the default: measured
    )


def _inside(points: np.ndarray, region: PlaceRegion | None, tol: float = 1e-6) -> np.ndarray:
    assert region is not None
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    local = (points - t[:3, 3]) @ t[:3, :3]
    return np.all(np.abs(local) <= np.asarray(region.half_extents) + tol, axis=1)


def _carton(rig: Rig, dx: float = 0.0) -> object:
    x, y = rig.centre_xy
    return box((x + dx, y, rig.support_z + 0.04), (0.041, 0.024, 0.04), 0.4)


@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_a_second_measured_view_adds_the_side_the_first_could_not_see(rig_name: str) -> None:
    top = RIGS[rig_name]()
    side = _turned(top, math.pi / 2)
    carton = _carton(top)
    first, second = _fit(top, [carton]), _fit(side, [carton])
    assert first.extent == "measured" and first.cone is not None
    out = intersect_views(
        [(first, top.t), (second, side.t)], resolution=RES, frame_id=top.frame, stamp_ns=1
    )
    assert out.refusals[1] == "contributed", out.refusals
    assert out.fit_kind is FitKind.MULTI_VIEW_CONE
    # never smaller than either view: every point either camera measured is boxed
    assert _inside(first.surface, out.fit.box).all()
    assert _inside(second.surface, out.fit.box).all()
    solid = surface_samples(carton)  # type: ignore[arg-type]
    assert _inside(solid, out.fit.box, 0.01).mean() >= _inside(solid, first.box, 0.01).mean()


@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_a_second_views_spill_onto_a_neighbour_is_not_boxed(rig_name: str) -> None:
    top = RIGS[rig_name]()
    side = _turned(top, math.pi / 2)
    carton, neighbour = _carton(top), _carton(top, dx=0.15)
    first = _fit(top, [carton, neighbour])
    spilled = _fit(side, [carton, neighbour], labels_in=(1, 2))  # mask took the neighbour too
    out = intersect_views(
        [(first, top.t), (spilled, side.t)], resolution=RES, frame_id=top.frame, stamp_ns=1
    )
    far = surface_samples(neighbour)  # type: ignore[arg-type]
    # the neighbour's far half is outside the primary's cone, so it never joins the box
    assert out.fit.box is not None
    assert not _inside(far, out.fit.box, 0.0).all()
    assert _inside(first.surface, out.fit.box).all()


def test_the_tracker_combines_two_cameras_with_default_params() -> None:
    top = RIGS["so101_top"]()
    side = _turned(top, math.pi / 2)
    carton = _carton(top)
    tr = ObjectTracker(top.frame)  # default TrackerParams / FitParams
    tr.open_query("g", "q0", ObjectQuery(label="carton", count="one", role=ObjectRole.TARGET),
                  "obj:carton@0", stamp_ns=0)  # fmt: skip
    step(tr, top, [carton], stamp_ns=1_000_000_000, wall_s=0.0)
    (first,) = tr.store.all()
    step(tr, side, [carton], stamp_ns=1_200_000_000, wall_s=10.0)
    (both,) = tr.store.all()
    assert both.object_id == first.object_id
    assert both.fit_kind is FitKind.MULTI_VIEW_CONE
