"""The palm-plane cut keeps the primitive's orientation: same containment, tighter (#418, OT-16).

For a payload primitive the plane crosses, the cut is the EXACT convex solid solid ∩ {along >=
palm} (box or hull), represented as a box in the pinch axes (broad phase, every other check) with
the cut's 26-DOP and vertices as the kernel's payload-vs-voxel ``tight_geometry``. Hypothesis on
the real hands' pinch axes and palm planes:

* containment kept: every sampled point of the original solid on the hand side lies in the new
  box, in the new DOP and (when it ships vertices) in the new hull;
* never looser: the new box lies inside the old pinch-axes bounding box, and the new vertices
  lie inside the original solid and on the hand side.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from openral_core.kinematics import bounding_half_extents
from openral_core.schemas import DOP_AXES
from openral_world_state.object_tracker.binding import _UNIT, _frame, _palm_clipped, _prim, _t
from scipy.spatial import ConvexHull, QhullError

from tests.unit._object_tracker_fixtures import ROBOTS, arm_at_hold

EPS = 1e-7
_DOP = np.asarray(DOP_AXES, dtype=np.float64)
_STATS = {"crossing": 0, "checked_points": 0}


@cache
def _hand(robot: str) -> tuple[object, NDArray[np.float64]]:
    binder, q, _ = arm_at_hold(robot)
    return binder.hand.pinch_in_link(q[binder.hand.joint.name]), np.eye(4)


def _rot(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    x, y, z, w = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _old_box(
    prim: object, t_ao: NDArray[np.float64], pinch: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The pre-amendment result: the pinch-axes box of all corners, back face on the plane."""
    t = t_ao @ _t(prim.pose_in_object)  # type: ignore[attr-defined]
    half = np.asarray(bounding_half_extents(prim.shape), dtype=np.float64)  # type: ignore[attr-defined]
    local = ((_UNIT * half) @ t[:3, :3].T + t[:3, 3]) @ pinch.axes  # type: ignore[attr-defined]
    lo, hi = local.min(axis=0), local.max(axis=0)
    lo[1] = pinch.palm_m  # type: ignore[attr-defined]
    return lo, hi


_vec = st.floats(-0.12, 0.12, allow_nan=False)
_half = st.floats(0.004, 0.16, allow_nan=False)
_quat = st.tuples(*[st.floats(-1, 1, allow_nan=False)] * 4).filter(
    lambda q: sum(c * c for c in q) > 1e-3
)


@pytest.mark.parametrize("robot", list(ROBOTS))
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    centre=st.tuples(_vec, _vec, _vec),
    half=st.tuples(_half, _half, _half),
    quat=_quat,
    shift=st.floats(-0.1, 0.1, allow_nan=False),
    seed=st.integers(0, 2**31 - 1),
)
def test_the_exact_cut_keeps_the_hand_side_solid_and_is_never_looser(
    robot: str,
    centre: tuple[float, float, float],
    half: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    shift: float,
    seed: int,
) -> None:
    pinch0, _ = _hand(robot)
    pinch = dataclasses.replace(pinch0, palm_m=pinch0.palm_m + shift)  # type: ignore[type-var]
    t_ao = _frame(np.asarray(pinch.centre), np.eye(3))
    prim = _prim(_frame(np.asarray(centre), _rot(quat)), half, "o", "measured")
    out = _palm_clipped([prim], t_ao, pinch, "o")
    t = t_ao @ _t(prim.pose_in_object)
    corners = (_UNIT * np.asarray(half)) @ t[:3, :3].T + t[:3, 3]
    along = corners @ pinch.axes[:, 1]
    if along.min() >= pinch.palm_m - 1e-9 or along.max() <= pinch.palm_m:
        assert out == [prim], "wholly in front or wholly behind: unchanged"
        return
    [new] = out
    _STATS["crossing"] += 1
    assert new.geometry_source == "measured" and new.tight_geometry is not None
    tn = t_ao @ _t(new.pose_in_object)
    nhalf = np.asarray(bounding_half_extents(new.shape))
    # never looser: the new box inside the old pinch-axes bounding box
    lo, hi = _old_box(prim, t_ao, pinch)
    ncorn = ((_UNIT * nhalf) @ tn[:3, :3].T + tn[:3, 3]) @ pinch.axes
    assert np.all(ncorn >= lo - EPS) and np.all(ncorn <= hi + EPS)
    # containment kept: sampled hand-side points of the original solid are in the new geometry
    rng = np.random.default_rng(seed)
    pts_local = rng.uniform(-1.0, 1.0, size=(400, 3)) * np.asarray(half)
    pts = pts_local @ t[:3, :3].T + t[:3, 3]
    pts = pts[pts @ pinch.axes[:, 1] >= pinch.palm_m]
    if not len(pts):
        return
    _STATS["checked_points"] += len(pts)
    q_local = (pts - tn[:3, 3]) @ tn[:3, :3]
    assert np.all(np.abs(q_local) <= nhalf + EPS), "in the new box"
    proj = q_local @ _DOP.T
    tg = new.tight_geometry
    assert np.all(proj >= np.asarray(tg.dop_lo_m) - EPS) and np.all(
        proj <= np.asarray(tg.dop_hi_m) + EPS
    ), "in the new DOP"
    verts = np.asarray(tg.hull_vertices_m, dtype=np.float64)
    if len(verts) >= 4:
        try:
            eq = ConvexHull(verts).equations
        except QhullError:
            eq = None  # a grazing, flat cut: the DOP above is the refinement
        if eq is not None:
            assert np.all(q_local @ eq[:, :3].T + eq[:, 3] <= EPS), "in the new hull"
        # never looser: the new vertices lie in the original solid and on the hand side
        vw = verts @ tn[:3, :3].T + tn[:3, 3]
        v_orig = (vw - t[:3, 3]) @ t[:3, :3]
        assert np.all(np.abs(v_orig) <= np.asarray(half) + EPS)
        assert np.all(vw @ pinch.axes[:, 1] >= pinch.palm_m - EPS)


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_hull_primitive_is_cut_from_its_hull_not_its_box(robot: str) -> None:
    """A tracked fit's hull (a tetrahedron here) is cut as the hull: tighter than its box."""
    from openral_core import AttachedCollisionPrimitive, BoxShape
    from openral_core.schemas import TightCollisionGeometry

    pinch, _ = _hand(robot)
    t_ao = _frame(np.asarray(pinch.centre), np.eye(3))
    h = 0.05
    tet = [(-h, -h, -h), (h, h, -h), (h, -h, h), (-h, h, h)]
    prim = AttachedCollisionPrimitive(
        shape=BoxShape(half_extents_m=(h, h, h)),
        pose_in_object=_prim(np.eye(4), (h, h, h), "o").pose_in_object,
        tight_geometry=TightCollisionGeometry(
            dop_lo_m=tuple(float(v) for v in (np.asarray(tet) @ _DOP.T).min(axis=0)),
            dop_hi_m=tuple(float(v) for v in (np.asarray(tet) @ _DOP.T).max(axis=0)),
            hull_vertices_m=tuple(tet),
        ),
        geometry_source="measured",
    )
    shifted = dataclasses.replace(pinch, palm_m=float(pinch.centre @ pinch.axes[:, 1]))  # type: ignore[type-var]
    [new] = _palm_clipped([prim], t_ao, shifted, "o")
    box_cut = _palm_clipped([_prim(np.eye(4), (h, h, h), "o", "measured")], t_ao, shifted, "o")[0]
    vol = lambda p: float(np.prod(bounding_half_extents(p.shape)))  # noqa: E731
    assert vol(new) <= vol(box_cut) + 1e-12, "no larger than the cut of the plain box"


def test_the_property_was_not_vacuous() -> None:
    """Runs after the hypothesis tests above (file order): the plane really crossed the boxes."""
    assert _STATS["crossing"] > 100, _STATS
    assert _STATS["checked_points"] > 5000, _STATS
