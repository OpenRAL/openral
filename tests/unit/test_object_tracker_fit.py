"""Containing fit (issue #418 step 3): the fitted set contains the solid, on the real cameras.

Spheres, cylinders, yawed boxes and a curved banana-like solid are ray-cast into the three
scenes' committed depth cameras (``_tracker_scenes``); every surface point of the solid
must lie in the fitted box **and** in the 64-vertex tight hull — the property a hull of
sampled surface points breaks for curved objects (#266).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from openral_core import PlaceRegion, TrackedObject
from openral_core.geometry import homogeneous_from_quat_xyz
from openral_world_state.object_tracker import (
    ROBOT_PIXEL,
    UNCERTAIN_PIXEL,
    ContainingFit,
    FitParams,
    containing_fit,
)

from tests.unit._tracker_scenes import (
    RES,
    RIGS,
    Rig,
    Solid,
    banana,
    box,
    cylinder,
    inside_hull,
    render,
    sphere,
    surface_samples,
)

pytest.importorskip("scipy", reason="scipy.spatial.ConvexHull checks hull containment")

_RIGS = {name: make() for name, make in RIGS.items()}
_CONTAINING = FitParams(extent="containing")


def _solid(kind: str, rig: Rig, dx: float, dy: float, a: float, b: float, c: float) -> Solid:
    x, y, z0 = rig.centre_xy[0] + dx, rig.centre_xy[1] + dy, rig.support_z
    if kind == "sphere":
        r = 0.02 + 0.03 * a
        return sphere((x, y, z0 + r), r)
    if kind == "cylinder":
        return cylinder((x, y), 0.02 + 0.02 * a, z0, z0 + 0.03 + 0.09 * b)
    if kind == "box":
        h = (0.015 + 0.035 * a, 0.015 + 0.035 * b, 0.01 + 0.04 * c)
        return box((x, y, z0 + h[2]), h, math.pi * c)
    return banana((x, y), z0, 0.06 + 0.04 * a, 0.015 + 0.006 * b, 0.8 + 0.8 * c, math.pi * a)


def _fit(rig: Rig, solids: list[Solid], **kw: Any) -> ContainingFit:
    depth, labels = render(solids, rig)
    depth = kw.pop("depth", depth)
    return containing_fit(
        kw.pop("mask", labels == 1),
        depth,
        kw.pop("robot_mask", np.zeros(depth.shape, np.uint8)),
        rig.k,
        rig.t,
        support_z=kw.pop("support_z", rig.support_z),
        resolution=RES,
        frame_id=rig.frame,
        stamp_ns=1,
        params=kw.pop("params", _CONTAINING),  # the #266 set these tests pin
        **kw,
    )


def _hull_base(fit: ContainingFit) -> NDArray[np.float64]:
    assert fit.tight is not None and fit.tight.tight_geometry is not None
    pose = fit.tight.pose_in_object
    t = homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)
    hull = np.asarray(fit.tight.tight_geometry.hull_vertices_m).reshape(-1, 3)
    return np.asarray(hull @ t[:3, :3].T + t[:3, 3])


def _in_tight(points: NDArray[np.float64], fit: ContainingFit) -> NDArray[np.bool_]:
    """In the box, its DOP and (when it has one) its hull: the set the kernel checks."""
    from openral_core.schemas import DOP_AXES

    assert fit.tight is not None and fit.tight.tight_geometry is not None
    geo = fit.tight.tight_geometry
    pose = fit.tight.pose_in_object
    t = homogeneous_from_quat_xyz(pose.xyz, pose.quat_xyzw)
    proj = ((points - t[:3, 3]) @ t[:3, :3]) @ np.asarray(DOP_AXES, dtype=np.float64).T
    ok: NDArray[np.bool_] = _in_box(points, fit.box) & np.all(
        (proj >= np.asarray(geo.dop_lo_m) - 1e-9) & (proj <= np.asarray(geo.dop_hi_m) + 1e-9),
        axis=1,
    )
    return ok & inside_hull(points, _hull_base(fit)) if geo.hull_vertices_m else ok


def _in_box(points: NDArray[np.float64], region: PlaceRegion | None) -> NDArray[np.bool_]:
    assert region is not None
    t = homogeneous_from_quat_xyz(region.pose.xyz, region.pose.quat_xyzw)
    local = (points - t[:3, 3]) @ t[:3, :3]
    return np.asarray(np.all(np.abs(local) <= np.asarray(region.half_extents) + 1e-9, axis=1))


@pytest.mark.parametrize("rig_name", sorted(_RIGS))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["sphere", "cylinder", "box", "banana"]),
    dx=st.floats(-0.05, 0.05),
    dy=st.floats(-0.05, 0.05),
    a=st.floats(0.0, 1.0),
    b=st.floats(0.0, 1.0),
    c=st.floats(0.0, 1.0),
)
def test_fitted_set_contains_the_solid(
    rig_name: str, kind: str, dx: float, dy: float, a: float, b: float, c: float
) -> None:
    rig = _RIGS[rig_name]
    solid = _solid(kind, rig, dx, dy, a, b, c)
    fit = _fit(rig, [solid])
    assert fit.refusal == "", (rig_name, kind, fit.refusal, fit.assumptions)
    pts = surface_samples(solid)
    assert pts[:, 2].min() >= rig.support_z - 1e-9  # resting on the support (A2)
    assert _in_box(pts, fit.box).all()
    assert len(_hull_base(fit)) <= TrackedObject.MAX_HULL_VERTICES
    assert _in_tight(pts, fit).all()


def test_six_pole_hull_misses_a_sphere_the_cone_contains() -> None:
    """#266: the hull of a sphere's six sampled poles lies inside its surface; C does not."""
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    r = 0.024  # the ∅4.8 cm orange
    c = np.array((x, y, rig.support_z + r))
    poles = c + r * np.vstack((np.eye(3), -np.eye(3)))
    diag = c + r * np.ones(3) / math.sqrt(3)  # a surface point between three poles
    assert not inside_hull(diag[None], poles)[0]
    # The octahedron's face sits r/√3 from the centre: 10 mm inside the 24 mm surface.
    assert r - r / math.sqrt(3) > 0.01
    fit = _fit(rig, [sphere(tuple(c), r)])
    pts = surface_samples(sphere(tuple(c), r))
    assert _in_tight(np.vstack((pts, diag)), fit).all()


def test_robot_pixels_are_unknown_and_grow_the_fit() -> None:
    """A fingertip over the mask's edge: robot pixels leave the mask, the bands re-add them.

    The fingertip is ``w_occ_px`` wide with the filter's uncertain band around it (A4
    holds); a wider hidden part is the visibility gate's job (``θ_vis``), not the fit's.
    """
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    solid = sphere((x, y, rig.support_z + 0.024), 0.024)
    depth, labels = render([solid], rig)
    rows, cols = np.nonzero(labels == 1)
    robot = np.zeros(depth.shape, np.uint8)
    tip = cols >= cols.max() - 3
    robot[rows[tip], cols[tip]] = ROBOT_PIXEL
    band = (robot == 0) & (np.abs(np.arange(depth.shape[1]) - (cols.max() - 4)) <= 2)[None, :]
    robot[band & np.isin(np.arange(depth.shape[0]), rows)[:, None]] = UNCERTAIN_PIXEL
    fit = _fit(rig, [solid], robot_mask=robot)
    assert fit.refusal == "" and fit.assumptions["a4_occluder_px"] > 0
    assert _in_tight(surface_samples(solid), fit).all()


def test_an_arm_hovering_beside_the_mask_does_not_lift_the_front_cut() -> None:
    """A head camera over a tote sees a finger next to the carton, 10 cm nearer (no wider
    than the occluder band): the band widens the cone laterally, but the cut stays at the
    carton's own front."""
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    solid = box((x, y, rig.support_z + 0.03), (0.03, 0.02, 0.03), 0.0)
    depth, labels = render([solid], rig)
    rows, cols = np.nonzero(labels == 1)
    robot = np.zeros(depth.shape, np.uint8)
    arm = (slice(rows.min(), rows.max() + 1), slice(cols.max() + 1, cols.max() + 5))
    robot[arm] = ROBOT_PIXEL
    depth[arm] = float(depth[labels == 1].min()) - 0.10
    # The filter's uncertain band straddles the arm's edge: two of its columns are arm.
    edge = (slice(rows.min(), rows.max() + 1), slice(cols.max() - 1, cols.max() + 1))
    robot[edge] = UNCERTAIN_PIXEL
    depth[edge] = float(depth[labels == 1].min()) - 0.10
    fit = _fit(rig, [solid], robot_mask=robot, depth=depth)
    assert fit.refusal == "" and fit.assumptions["a5_robot_band_px"] > 0
    assert _in_tight(surface_samples(solid), fit).all()
    assert fit.box is not None
    top = fit.box.pose.xyz[2] + fit.box.half_extents[2]
    assert top < rig.support_z + 0.06 + 2 * RES  # not lifted toward the arm


def test_a_robot_wider_than_the_occluder_band_makes_the_fit_provisional() -> None:
    """The arm over a plate's near rim: it touches the mask and reaches past the band, so
    whatever it hides may exceed what the band re-adds. The fit is PROVISIONAL."""
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    plate = cylinder((x, y), 0.08, rig.support_z, rig.support_z + 0.02)
    depth, labels = render([plate], rig)
    rows, cols = np.nonzero(labels == 1)
    robot = np.zeros(depth.shape, np.uint8)
    arm = (slice(rows.min(), rows.max() + 1), slice(cols.max() - 10, cols.max() + 20))
    robot[arm] = ROBOT_PIXEL
    depth[arm] = float(depth[labels == 1].min()) - 0.15
    fit = _fit(rig, [plate], robot_mask=robot, depth=depth)
    assert fit.refusal == "" and fit.box is not None and fit.provisional
    assert fit.assumptions["a4_robot_beyond_band_px"] > 0
    clear = _fit(rig, [plate])
    assert clear.refusal == "" and not clear.provisional


def test_the_detector_box_restores_a_rim_the_mask_lost() -> None:
    """SAM's plate mask lost its near rim (8 px); the detector's box still holds it (A1)."""
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    plate = cylinder((x, y), 0.08, rig.support_z, rig.support_z + 0.02)
    _, labels = render([plate], rig)
    full = labels == 1
    rows, cols = np.nonzero(full)
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    lost = full & (np.arange(full.shape[0])[:, None] < rows.max() - 8)
    pts = surface_samples(plate)
    without = _fit(rig, [plate], mask=lost)
    assert without.refusal == "" and not _in_box(pts, without.box).all()  # the A1 failure
    fit = _fit(rig, [plate], mask=lost, detector_box=box)
    assert fit.refusal == "" and _in_box(pts, fit.box).all()
    assert _in_tight(pts, fit).all()


def test_refusals_name_the_broken_assumption() -> None:
    rig = _RIGS["so101_top"]
    x, y = rig.centre_xy
    # A2: the measured support above the object's bottom would truncate the solid.
    high = _fit(rig, [box((x, y, rig.support_z + 0.04), (0.03, 0.03, 0.04), 0.0)],
                support_z=rig.support_z + 0.03)  # fmt: skip
    assert high.refusal == "a2_below_support" and high.box is None
    # A1: a mask cut by the image border hides what the frame lost.
    depth, labels = render([box((x, y, rig.support_z + 0.02), (0.03, 0.03, 0.02), 0.0)], rig)
    mask = labels == 1
    mask[np.nonzero(mask)[0], 0] = True  # the same rows also touch column 0
    fit = containing_fit(
        mask, depth, None, rig.k, rig.t, support_z=rig.support_z, resolution=RES,
        frame_id=rig.frame, stamp_ns=1,
    )  # fmt: skip
    assert fit.refusal == "a1_border_truncated"


@pytest.mark.parametrize("rig_name", sorted(_RIGS))
def test_an_oblique_view_does_not_box_far_above_the_object(rig_name: str) -> None:
    """Spark banana R325 (2026-10-23): the global front cut lifted the far-rim rays, so a
    5.6 cm bowl boxed 15 cm tall and its margin kept the carried banana's cells as the bowl's.
    No ray meets the solid nearer than its first hit, so the box tops out at the highest
    first hit: the rim, one depth tolerance and half a cell above it."""
    rig = _RIGS[rig_name]
    x, y = rig.centre_xy
    top = rig.support_z + 0.056
    solid = cylinder((x, y), 0.08, rig.support_z, top)
    fit = _fit(rig, [solid])
    assert fit.refusal == "", fit.refusal
    assert fit.box is not None
    zhi = fit.box.pose.xyz[2] + fit.box.half_extents[2]
    assert zhi <= top + 0.02, (rig_name, zhi - top)
    assert _in_tight(surface_samples(solid), fit).all()


@pytest.mark.parametrize("rig_name", sorted(_RIGS))
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(["cylinder", "box"]),
    dx=st.floats(-0.05, 0.05),
    dy=st.floats(-0.05, 0.05),
    a=st.floats(0.0, 1.0),
    b=st.floats(0.0, 1.0),
    c=st.floats(0.0, 1.0),
)
# DROID exterior: the 2 cm-tall box's far corner lay 8.3 mm from its nearest sample.
@example(kind="box", dx=0.0, dy=0.0, a=0.9375, b=0.0, c=0.0)
def test_the_measured_box_contains_an_upright_prism(
    rig_name: str, kind: str, dx: float, dy: float, a: float, b: float, c: float
) -> None:
    """``extent="measured"`` (the default) boxes what the camera saw, extruded to the support.
    A solid whose top face spans its footprint (a carton, a can) is whole inside it from any
    of the rigs; a part hidden past what the camera saw (the far side of a sphere seen
    obliquely: up to 5 mm on the OpenArm head, 22 mm on DROID's exterior camera) is not."""
    rig = _RIGS[rig_name]
    solid = _solid(kind, rig, dx, dy, a, b, c)
    fit = _fit(rig, [solid], params=FitParams())
    assert fit.refusal == "", (rig_name, kind, fit.refusal)
    pts = surface_samples(solid)
    assert _in_box(pts, fit.box).all()
    assert _in_tight(pts, fit).all()


def test_a_carton_is_boxed_near_its_own_size_not_its_shadow() -> None:
    """Restock 2026-10-31: from the OpenArm head camera a 6 x 6 x 11 cm carton measured
    6.9 x 7.0 x 11 cm and was boxed 12.2 x 12.4 x 12.3 cm, the cone of everything it could
    hide; the kernel then stopped the carried box 5 cm short of the shelf board."""
    rig = _RIGS["openarm_head"]
    x, y = rig.centre_xy
    carton = box((x, y, rig.support_z + 0.055), (0.03, 0.03, 0.055), 0.0)
    measured = _fit(rig, [carton], params=FitParams())
    cone = _fit(rig, [carton])  # the #266 containing set
    assert measured.box is not None and cone.box is not None
    pad = 0.5 * RES  # half a cell on each side
    assert max(measured.box.half_extents[:2]) <= 0.03 + pad + 1e-6, measured.box.half_extents
    assert max(cone.box.half_extents[:2]) > 1.5 * 0.03, cone.box.half_extents


def test_a_mask_spilled_onto_a_neighbour_is_cut_to_what_the_detector_boxed() -> None:
    """Restock scene 13 2026-10-31: the mask took in a carton 1.6 cm away and the 6 x 6 cm
    target was boxed 18 x 9 cm. The depth-connected part the detector boxed is kept."""
    rig = _RIGS["openarm_head"]
    x, y = rig.centre_xy
    target = box((x, y, rig.support_z + 0.055), (0.03, 0.03, 0.055), 0.0)
    neighbour = box((x - 0.076, y, rig.support_z + 0.04), (0.03, 0.03, 0.04), 0.0)
    _, labels = render([target, neighbour], rig)
    rows, cols = np.nonzero(labels == 1)
    prompt = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    rows, cols = np.nonzero(labels > 0)
    both = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    spilled = labels > 0
    kw: dict[str, Any] = {"mask": spilled, "params": FitParams()}
    whole = _fit(rig, [target, neighbour], **kw)
    cut = _fit(rig, [target, neighbour], detector_box=prompt, **kw)
    kept = _fit(rig, [target, neighbour], detector_box=both, **kw)
    assert whole.box is not None and cut.box is not None and kept.box is not None
    assert max(whole.box.half_extents[:2]) > 0.06, whole.box.half_extents
    assert max(cut.box.half_extents[:2]) <= 0.03 + RES, cut.box.half_extents
    assert _in_box(surface_samples(target), cut.box).all()
    assert cut.assumptions["spill_dropped_points"] > 0
    assert kept.box.half_extents == whole.box.half_extents  # a box over both drops nothing
    # A refit prompts with the track's projected box: it picks the parts the same way.
    refit = _fit(rig, [target, neighbour], part_box=prompt, **kw)
    assert refit.box is not None and refit.box.half_extents == cut.box.half_extents


def test_a_target_an_occluder_cuts_in_two_keeps_both_parts() -> None:
    """A thin occluder across the target splits its mask; both parts lie in the detector's box,
    so neither is dropped."""
    rig = _RIGS["openarm_head"]
    x, y = rig.centre_xy
    target = box((x, y, rig.support_z + 0.055), (0.03, 0.03, 0.055), 0.0)
    _, labels = render([target], rig)
    whole = labels == 1
    rows, cols = np.nonzero(whole)
    prompt = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    mid = (cols.min() + cols.max()) // 2
    cut = whole.copy()
    cut[:, mid - 1 : mid + 2] = False
    fit = _fit(rig, [target], mask=cut, detector_box=prompt, params=FitParams())
    ref = _fit(rig, [target], mask=whole, params=FitParams())
    assert fit.box is not None and ref.box is not None
    assert "spill_dropped_points" not in fit.assumptions
    assert np.allclose(fit.box.half_extents, ref.box.half_extents, atol=RES / 2)


def test_a_mask_over_the_strip_the_carton_stands_on_is_boxed_to_the_carton() -> None:
    """Restock scene 09 2026-11-01: refit masks took in the white strip under the carton
    (every point outside the carton 1 mm above its bottom) and boxed it 14.5 x 12 cm."""
    rig = _RIGS["openarm_head"]
    x, y = rig.centre_xy
    strip_top = rig.support_z + 0.004
    strip = box((x, y, rig.support_z + 0.002), (0.15, 0.04, 0.002), 0.0)
    carton = box((x, y, strip_top + 0.055), (0.03, 0.03, 0.055), 0.0)
    _, labels = render([carton, strip], rig)
    rows, cols = np.nonzero(labels == 1)
    near = np.zeros(labels.shape, bool)
    near[rows.min() - 15 : rows.max() + 16, cols.min() - 15 : cols.max() + 16] = True
    spilled = (labels == 1) | ((labels == 2) & near)
    fit = _fit(rig, [carton, strip], mask=spilled, support_z=strip_top, params=FitParams())
    assert fit.box is not None
    assert max(fit.box.half_extents[:2]) <= 0.03 + RES, fit.box.half_extents
    assert _in_box(surface_samples(carton), fit.box).all()
    assert fit.assumptions["support_dropped_points"] > 0
