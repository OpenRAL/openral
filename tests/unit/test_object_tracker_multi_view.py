"""Multi-view containing fit (issue #418 phase 4): the intersection never cuts the solid.

No committed manifest or scene carries two posed depth cameras that see the same object
(the scenes' second cameras are wrist RGB on a moving link). So each rig is a scene's real
depth camera (``_tracker_scenes``: real intrinsics, mount height and tilt), and the second
view is the same camera turned about the vertical axis through the object by a drawn yaw —
a second real-geometry viewpoint of the same solid. Solids are ray-cast exactly into both.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from openral_core import FitKind, PlaceRegion, TrackedObject
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_world_state.object_tracker import ContainingFit, containing_fit, intersect_views

from tests.unit._tracker_scenes import RES, RIGS, Rig, inside_hull, render, surface_samples

pytest.importorskip("scipy", reason="the test's hull containment uses scipy ConvexHull")

# After the scipy skip: that module skips at import without scipy.
_solid = pytest.importorskip("tests.unit.test_object_tracker_fit")._solid

_RIGS = {name: make() for name, make in RIGS.items()}


def _turned(rig: Rig, yaw: float) -> Rig:
    """``rig``'s camera turned by ``yaw`` about the vertical through its object centre."""
    c, s = math.cos(yaw), math.sin(yaw)
    turn = np.eye(4)
    turn[:2, :2] = ((c, -s), (s, c))
    x, y = rig.centre_xy
    shift = np.eye(4)
    shift[:2, 3] = (x, y)
    t = shift @ turn @ np.linalg.inv(shift) @ rig.t
    return replace(rig, name=f"{rig.name}@{yaw:.2f}", t=t)


def _fit(rig: Rig, solid: object, mask_erode: int = 0) -> ContainingFit:
    depth, labels = render([solid], rig)  # type: ignore[list-item]
    mask = labels == 1
    for _ in range(mask_erode):  # a too-small segmenter mask
        mask = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
        mask = mask & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    return containing_fit(
        mask,
        depth,
        np.zeros(depth.shape, np.uint8),
        rig.k,
        rig.t,
        support_z=rig.support_z,
        resolution=RES,
        frame_id=rig.frame,
        stamp_ns=1,
    )


def _hull(fit: ContainingFit) -> NDArray[np.float64]:
    assert fit.tight is not None and fit.tight.tight_geometry is not None
    pose = fit.tight.pose_in_object
    t = homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)
    return np.asarray(np.asarray(fit.tight.tight_geometry.hull_vertices_m) @ t[:3, :3].T + t[:3, 3])


def _in_box(points: NDArray[np.float64], region: PlaceRegion | None) -> bool:
    assert region is not None
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    local = (points - t[:3, 3]) @ t[:3, :3]
    return bool(np.all(np.abs(local) <= np.asarray(region.half_extents) + 1e-9))


def _volume(region: PlaceRegion | None) -> float:
    assert region is not None
    return float(np.prod(region.half_extents))


@pytest.mark.parametrize("rig_name", sorted(_RIGS))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["sphere", "cylinder", "box", "banana"]),
    dx=st.floats(-0.04, 0.04),
    dy=st.floats(-0.04, 0.04),
    a=st.floats(0.0, 1.0),
    b=st.floats(0.0, 1.0),
    c=st.floats(0.0, 1.0),
    yaw=st.floats(0.6, 2.6),
)
def test_the_intersection_never_cuts_the_solid(
    rig_name: str, kind: str, dx: float, dy: float, a: float, b: float, c: float, yaw: float
) -> None:
    rig = _RIGS[rig_name]
    other = _turned(rig, yaw)
    solid = _solid(kind, rig, dx, dy, a, b, c)
    one, two = _fit(rig, solid), _fit(other, solid)
    assert one.refusal == "", one.refusal
    out = intersect_views(
        [(one, rig.t), (two, other.t)], resolution=RES, frame_id=rig.frame, stamp_ns=1
    )
    pts = surface_samples(solid)
    assert _in_box(pts, out.fit.box)
    hull = _hull(out.fit)
    assert len(hull) <= TrackedObject.MAX_HULL_VERTICES
    assert inside_hull(pts, hull).all(), out.refusals
    assert _volume(out.fit.box) <= _volume(one.box) + 1e-12
    if out.fit_kind is FitKind.MULTI_VIEW_CONE:
        assert out.fit.box is not None
        assert out.fit.box.evidence_ref == "object_tracker:multi_view_cone"


def test_a_second_view_cuts_the_first_views_ray_shadow() -> None:
    """Not vacuous: a side view removes the top camera's far-side shadow."""
    rig = _RIGS["so101_top"]
    other = _turned(rig, math.pi / 2)
    solid = _solid("sphere", rig, 0.0, 0.0, 0.5, 0.0, 0.0)
    one, two = _fit(rig, solid), _fit(other, solid)
    out = intersect_views(
        [(one, rig.t), (two, other.t)], resolution=RES, frame_id=rig.frame, stamp_ns=1
    )
    assert out.fit_kind is FitKind.MULTI_VIEW_CONE, out.refusals
    assert out.refusals == {0: "primary", 1: "contributed"}
    assert _volume(out.fit.box) < 0.95 * _volume(one.box)
    assert inside_hull(surface_samples(solid), _hull(out.fit)).all()


def test_a_too_small_or_truncated_second_silhouette_is_refused() -> None:
    rig = _RIGS["so101_top"]
    other = _turned(rig, math.pi / 2)
    solid = _solid("sphere", rig, 0.0, 0.0, 0.5, 0.0, 0.0)
    one = _fit(rig, solid)
    small = _fit(other, solid, mask_erode=4)  # the segmenter lost the rim in view two
    assert small.refusal == ""
    out = intersect_views(
        [(one, rig.t), (small, other.t)], resolution=RES, frame_id=rig.frame, stamp_ns=1
    )
    assert out.fit_kind is FitKind.SILHOUETTE_CONE
    assert out.refusals[1] == "silhouette_too_small"
    assert out.fit is one

    # Pushed half out of view two's frame: its own fit refuses (A1), so it never cuts.
    far = _turned(rig, math.pi / 2)
    edge = _solid("sphere", rig, 0.0, 0.0, 0.5, 0.0, 0.0)
    k = far.k
    far = replace(far, k=k.model_copy(update={"cx": k.cx + 0.5 * k.width}))
    cut = _fit(far, edge)
    out = intersect_views(
        [(one, rig.t), (cut, far.t)], resolution=RES, frame_id=rig.frame, stamp_ns=1
    )
    assert out.fit_kind is FitKind.SILHOUETTE_CONE
    assert out.refusals[1] in {"a1_border_truncated", "too_few_points"}
    assert out.fit is one
